"""
Gather supplementary country-level data from the World Bank Open Data API.

This is the "second source" of data (a government/public data source, as
distinct from the TMDB API) used to add regional economic context -- e.g.
GDP per capita and population by country and year -- so that movie
popularity/revenue by production country/region can be compared against the
size and wealth of that country's audience. No API key is required.

Website:  https://data.worldbank.org/
API docs: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392
Core endpoint used: GET https://api.worldbank.org/v2/country/all/indicator/{INDICATOR}
Example GET:
  https://api.worldbank.org/v2/country/all/indicator/NY.GDP.PCAP.CD?format=json&per_page=20000&date=1990:2024
"""

import json
from pathlib import Path

import requests
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://api.worldbank.org/v2/country/all/indicator"
YEAR_RANGE = "1990:2024"

INDICATORS = {
    "NY.GDP.PCAP.CD": "gdp_per_capita_usd",
    "SP.POP.TOTL": "population",
}


def fetch_indicator(indicator_code, column_name):
    all_rows = []
    page = 1
    while True:
        resp = requests.get(
            f"{BASE_URL}/{indicator_code}",
            params={"format": "json", "per_page": 20000, "date": YEAR_RANGE, "page": page},
            timeout=30,
        )
        resp.raise_for_status()
        meta, rows = resp.json()
        if not rows:
            break
        all_rows.extend(rows)
        if page >= meta["pages"]:
            break
        page += 1

    records = [
        {
            "country_code": r["countryiso3code"],
            "country_name": r["country"]["value"],
            "year": int(r["date"]),
            column_name: r["value"],
        }
        for r in all_rows
        if r["countryiso3code"]
    ]
    return pd.DataFrame(records)


def fetch_country_reference():
    """ISO alpha-2 <-> alpha-3 country code lookup (needed to join TMDB's
    alpha-2 production-country codes onto World Bank's alpha-3 codes)."""
    resp = requests.get(
        f"https://api.worldbank.org/v2/country",
        params={"format": "json", "per_page": 500},
        timeout=30,
    )
    resp.raise_for_status()
    _, rows = resp.json()
    records = [
        {"iso2_code": r["iso2Code"], "country_code": r["id"], "country_name": r["name"]}
        for r in rows
        if r["region"]["value"] != "Aggregates"
    ]
    return pd.DataFrame(records)


def main():
    frames = [fetch_indicator(code, name) for code, name in INDICATORS.items()]
    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on=["country_code", "country_name", "year"], how="outer")

    out_path = RAW_DIR / "worldbank_country_indicators_raw.csv"
    merged.to_csv(out_path, index=False)
    print(f"Saved {len(merged)} rows to {out_path}")

    ref_df = fetch_country_reference()
    ref_path = RAW_DIR / "worldbank_country_reference.csv"
    ref_df.to_csv(ref_path, index=False)
    print(f"Saved {len(ref_df)} rows to {ref_path}")


if __name__ == "__main__":
    main()
