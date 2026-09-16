"""
Clean and merge the raw data pulled by fetch_tmdb.py, fetch_tmdb_details.py,
and fetch_worldbank.py into a single analysis-ready CSV.

Cleaning steps (see DataPrep_EDA tab on the website for the write-up):
  - drop duplicate movies, drop rows missing an essential field (release date)
  - coerce numeric columns and drop rows that fail to coerce
  - treat TMDB's budget/revenue == 0 as "unknown" (missing), not "free movie",
    and flag which rows have real budget/revenue data instead of dropping them
  - remove impossible values (negative votes, near-zero runtime, etc.)
  - derive release_year / release_month / decade from release_date
  - derive primary_genre, num_genres, primary_country, num_spoken_languages
  - derive profit and ROI where budget and revenue are both known
  - discretize popularity and rating into tiers
  - min-max normalize the core numeric columns for later modeling
  - join in World Bank GDP-per-capita / population for the primary
    production country and release year
"""

import ast
import json
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
CLEAN_DIR = ROOT / "data" / "clean"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

LANGUAGE_NAMES = {
    "en": "English", "fr": "French", "es": "Spanish", "hi": "Hindi", "ja": "Japanese",
    "ko": "Korean", "zh": "Mandarin", "cn": "Cantonese", "de": "German", "it": "Italian",
    "pt": "Portuguese", "ru": "Russian", "ta": "Tamil", "te": "Telugu", "ar": "Arabic",
    "tr": "Turkish", "pl": "Polish", "nl": "Dutch", "sv": "Swedish", "da": "Danish",
    "fi": "Finnish", "no": "Norwegian", "cs": "Czech", "el": "Greek", "he": "Hebrew",
    "id": "Indonesian", "th": "Thai", "vi": "Vietnamese", "fa": "Persian", "ur": "Urdu",
    "bn": "Bengali", "ml": "Malayalam", "kn": "Kannada", "mr": "Marathi", "gu": "Gujarati",
    "pa": "Punjabi", "ro": "Romanian", "hu": "Hungarian", "uk": "Ukrainian",
}


def _parse_listish(value):
    """Safely parse a stringified Python list back into a real list."""
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def load_discover():
    df = pd.read_csv(RAW_DIR / "tmdb_movies_raw.csv")
    df["genre_names"] = df["genre_names"].apply(_parse_listish)
    return df


def load_details():
    path = RAW_DIR / "tmdb_movie_details_raw.csv"
    if not path.exists():
        print(f"WARNING: {path} not found -- run fetch_tmdb_details.py for budget/revenue/runtime. "
              "Continuing without those columns.")
        return None
    return pd.read_csv(path)


def load_worldbank():
    ind_path = RAW_DIR / "worldbank_country_indicators_raw.csv"
    ref_path = RAW_DIR / "worldbank_country_reference.csv"
    if not ind_path.exists() or not ref_path.exists():
        print(f"WARNING: World Bank raw files not found -- run fetch_worldbank.py. "
              "Continuing without GDP/population columns.")
        return None
    indicators = pd.read_csv(ind_path)
    reference = pd.read_csv(ref_path)
    return indicators.merge(reference[["iso2_code", "country_code"]], on="country_code", how="left")


def clean(df, details, worldbank):
    before = len(df)
    df = df.drop_duplicates(subset="id").copy()

    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    for col in ["popularity", "vote_average", "vote_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["release_date", "title", "popularity", "vote_average", "vote_count"])
    df = df[df["vote_count"] >= 0]
    after_essential = len(df)

    df["release_year"] = df["release_date"].dt.year
    df["release_month"] = df["release_date"].dt.month
    df["decade"] = (df["release_year"] // 10) * 10

    df["num_genres"] = df["genre_names"].apply(len)
    df["primary_genre"] = df["genre_names"].apply(lambda g: g[0] if g else "Unknown")
    df["genres"] = df["genre_names"].apply(lambda g: "|".join(g))

    df["language_name"] = df["original_language"].map(LANGUAGE_NAMES).fillna(df["original_language"])
    df["overview_length"] = df["overview"].fillna("").str.split().apply(len)

    if details is not None:
        details = details.copy()
        for col in ["budget", "revenue", "runtime"]:
            details[col] = pd.to_numeric(details[col], errors="coerce")
        # TMDB uses 0 to mean "unknown", not "free" / "instant" -- treat as missing.
        details.loc[details["budget"] <= 0, "budget"] = np.nan
        details.loc[details["revenue"] <= 0, "revenue"] = np.nan
        details.loc[details["runtime"] < 5, "runtime"] = np.nan  # remove impossible runtimes

        details["primary_country"] = details["production_countries"].fillna("").apply(
            lambda s: s.split("|")[0] if s else "Unknown"
        )
        details["num_spoken_languages"] = details["spoken_languages"].fillna("").apply(
            lambda s: len([x for x in s.split("|") if x])
        )

        df = df.merge(
            details[["id", "budget", "revenue", "runtime", "status", "primary_country",
                      "production_countries", "num_spoken_languages", "production_companies"]],
            on="id", how="left",
        )
        df["has_budget_data"] = df["budget"].notna()
        df["has_revenue_data"] = df["revenue"].notna()
        df["profit"] = np.where(df["has_budget_data"] & df["has_revenue_data"],
                                 df["revenue"] - df["budget"], np.nan)
        df["roi"] = np.where(df["has_budget_data"] & df["has_revenue_data"] & (df["budget"] > 0),
                              df["profit"] / df["budget"], np.nan)
    else:
        df["primary_country"] = "Unknown"

    df["rating_tier"] = pd.cut(
        df["vote_average"], bins=[-0.1, 5, 6.5, 8, 10],
        labels=["Poor", "Mixed", "Good", "Excellent"],
    )
    df["popularity_tier"] = pd.qcut(
        df["popularity"], q=4, labels=["Low", "Medium", "High", "Very High"], duplicates="drop"
    )

    if worldbank is not None:
        df = df.merge(
            worldbank.rename(columns={"iso2_code": "primary_country", "year": "release_year"}),
            on=["primary_country", "release_year"], how="left",
        )

    def min_max(col):
        if col not in df.columns:
            return
        valid = df[col].notna()
        span = df.loc[valid, col].max() - df.loc[valid, col].min()
        df[f"{col}_norm"] = np.nan
        if span > 0:
            df.loc[valid, f"{col}_norm"] = (df.loc[valid, col] - df.loc[valid, col].min()) / span

    for col in ["popularity", "vote_average", "vote_count", "budget", "revenue", "runtime"]:
        min_max(col)

    print(f"Rows: raw={before}, after de-dup={len(df.drop_duplicates(subset='id'))}, "
          f"after essential cleaning={after_essential}, final={len(df)}")
    return df


def main():
    df = load_discover()
    details = load_details()
    worldbank = load_worldbank()

    cleaned = clean(df, details, worldbank)

    keep_cols = [c for c in [
        "id", "title", "original_title", "original_language", "language_name",
        "release_date", "release_year", "release_month", "decade",
        "genres", "primary_genre", "num_genres",
        "popularity", "vote_average", "vote_count", "rating_tier", "popularity_tier",
        "budget", "revenue", "has_budget_data", "has_revenue_data", "profit", "roi",
        "runtime", "primary_country", "production_countries", "num_spoken_languages",
        "production_companies", "overview_length",
        "gdp_per_capita_usd", "population",
        "popularity_norm", "vote_average_norm", "vote_count_norm",
        "budget_norm", "revenue_norm", "runtime_norm",
    ] if c in cleaned.columns]

    out_path = CLEAN_DIR / "tmdb_movies_clean.csv"
    cleaned[keep_cols].to_csv(out_path, index=False)
    print(f"Saved cleaned dataset ({len(cleaned)} rows, {len(keep_cols)} columns) to {out_path}")


if __name__ == "__main__":
    main()
