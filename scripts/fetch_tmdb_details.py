"""
Enrich the discover-endpoint results with per-movie details.

TMDB's /discover/movie endpoint (used by fetch_tmdb.py) does not return
budget, revenue, runtime, or production countries -- those only come from
the /movie/{id} details endpoint. This script reads data/raw/tmdb_movies_raw.csv
(produced by fetch_tmdb.py), calls the details endpoint for each movie id,
and writes an enriched CSV.

Core endpoint: GET https://api.themoviedb.org/3/movie/{movie_id}
Example GET:   https://api.themoviedb.org/3/movie/27205?api_key=API_KEY

Run fetch_tmdb.py first.
"""

import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"

MAX_WORKERS = 8
DETAIL_FIELDS = [
    "budget", "revenue", "runtime", "status", "tagline",
    "production_countries", "spoken_languages", "production_companies",
]


def fetch_details(movie_id, retries=4):
    resp = None
    for attempt in range(retries):
        try:
            resp = requests.get(
                f"{BASE_URL}/movie/{movie_id}",
                params={"api_key": API_KEY},
                timeout=30,
            )
            break
        except requests.exceptions.RequestException:
            if attempt == retries - 1:
                return {"id": movie_id}
            time.sleep(2 ** attempt)
    if resp is None or resp.status_code != 200:
        return {"id": movie_id}
    data = resp.json()
    row = {"id": movie_id}
    for field in DETAIL_FIELDS:
        value = data.get(field)
        if field == "production_countries":
            row["production_countries"] = "|".join(c["iso_3166_1"] for c in (value or []))
        elif field == "spoken_languages":
            row["spoken_languages"] = "|".join(l["iso_639_1"] for l in (value or []))
        elif field == "production_companies":
            row["production_companies"] = "|".join(c["name"] for c in (value or []))
        else:
            row[field] = value
    return row


def main():
    if not API_KEY:
        sys.exit("ERROR: TMDB_API_KEY is not set. See .env.example.")

    raw_path = RAW_DIR / "tmdb_movies_raw.csv"
    if not raw_path.exists():
        sys.exit(f"ERROR: {raw_path} not found. Run fetch_tmdb.py first.")

    df = pd.read_csv(raw_path)
    ids = df["id"].dropna().astype(int).unique().tolist()
    print(f"Fetching details for {len(ids)} movies with {MAX_WORKERS} workers...")

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fetch_details, mid): mid for mid in ids}
        for i, fut in enumerate(as_completed(futures), start=1):
            results.append(fut.result())
            if i % 200 == 0:
                print(f"  {i}/{len(ids)} done")

    details_df = pd.DataFrame(results)
    out_path = RAW_DIR / "tmdb_movie_details_raw.csv"
    details_df.to_csv(out_path, index=False)
    print(f"Saved details for {len(details_df)} movies to {out_path}")


if __name__ == "__main__":
    main()
