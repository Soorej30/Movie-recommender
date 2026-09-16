"""
Gather movie data from The Movie Database (TMDB) API.

TMDB website:      https://www.themoviedb.org/
API docs:           https://developer.themoviedb.org/reference/discover-movie
Core endpoint used: GET https://api.themoviedb.org/3/discover/movie
Example GET:        https://api.themoviedb.org/3/discover/movie?api_key=API_KEY&sort_by=popularity.desc&primary_release_date.gte=2000-01-01&primary_release_date.lte=2023-12-31&with_original_language=en&vote_count.gte=50&vote_average.gte=0&vote_average.lte=10&page=1

Requires a TMDB API key. Get one (free) at:
  1. Create an account at https://www.themoviedb.org/signup
  2. Go to Settings -> API -> request an API key ("Developer" use is fine for a
     student project) at https://www.themoviedb.org/settings/api
  3. Copy the "API Key (v3 auth)" value.
  4. Put it in a .env file at the project root:  TMDB_API_KEY=xxxxxxxx
     (copy .env.example to .env and fill it in)

Filters supported below (edit the CONFIG block):
  - year range (release date window)
  - genres
  - original language
  - region (release-date region used by TMDB)
  - vote average (rating) range
  - minimum vote count
  - sort/discovery order (popularity, vote_average, revenue, release_date)
"""

import os
import sys
import time
import json
from pathlib import Path

import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# CONFIG: edit these filters to control what gets pulled from TMDB.
# Leave a field as None / empty to not filter on it.
# ---------------------------------------------------------------------------
CONFIG = {
    "year_start": 1990,
    "year_end": 2024,
    "genres": [],                  # e.g. ["Action", "Drama"] -> resolved to TMDB genre ids
    "original_language": None,     # e.g. "en", "hi", "ko", "fr" (ISO 639-1); None = all languages
    "region": None,                # e.g. "US", "IN", "KR" (ISO 3166-1); None = no region filter
    "vote_average_gte": 0.0,
    "vote_average_lte": 10.0,
    "vote_count_gte": 50,          # drop movies with almost no votes (noisy ratings)
    "sort_by": "popularity.desc",  # popularity.desc | vote_average.desc | revenue.desc | release_date.desc
    "max_pages": 100,              # 20 results/page -> up to 2,000 rows (TMDB caps discover at 500 pages)
}

SESSION = requests.Session()


def _get(path, params, retries=4):
    params = {**params, "api_key": API_KEY}
    for attempt in range(retries):
        try:
            resp = SESSION.get(f"{BASE_URL}{path}", params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def fetch_genre_map():
    """Return {genre_name: genre_id} using the /genre/movie/list endpoint."""
    data = _get("/genre/movie/list", {"language": "en-US"})
    return {g["name"]: g["id"] for g in data["genres"]}


def build_discover_params(genre_ids):
    params = {
        "sort_by": CONFIG["sort_by"],
        "include_adult": "false",
        "include_video": "false",
        "vote_average.gte": CONFIG["vote_average_gte"],
        "vote_average.lte": CONFIG["vote_average_lte"],
        "vote_count.gte": CONFIG["vote_count_gte"],
    }
    if CONFIG["year_start"]:
        params["primary_release_date.gte"] = f"{CONFIG['year_start']}-01-01"
    if CONFIG["year_end"]:
        params["primary_release_date.lte"] = f"{CONFIG['year_end']}-12-31"
    if genre_ids:
        params["with_genres"] = ",".join(str(g) for g in genre_ids)
    if CONFIG["original_language"]:
        params["with_original_language"] = CONFIG["original_language"]
    if CONFIG["region"]:
        params["region"] = CONFIG["region"]
    return params


def fetch_movies():
    genre_map = fetch_genre_map()
    genre_ids = [genre_map[g] for g in CONFIG["genres"] if g in genre_map]
    params = build_discover_params(genre_ids)

    first_page = _get("/discover/movie", {**params, "page": 1})
    total_pages = min(first_page["total_pages"], CONFIG["max_pages"])
    print(f"TMDB reports {first_page['total_results']} matching movies "
          f"across {first_page['total_pages']} pages; fetching {total_pages} page(s).")

    all_results = list(first_page["results"])
    for page in range(2, total_pages + 1):
        data = _get("/discover/movie", {**params, "page": page})
        all_results.extend(data["results"])
        if page % 20 == 0:
            print(f"  fetched page {page}/{total_pages}")
        time.sleep(0.02)  # stay well under TMDB's rate limit

    id_to_genre = {v: k for k, v in genre_map.items()}
    for m in all_results:
        m["genre_names"] = [id_to_genre.get(gid, str(gid)) for gid in m.get("genre_ids", [])]

    return all_results, genre_map


def main():
    if not API_KEY:
        sys.exit(
            "ERROR: TMDB_API_KEY is not set.\n"
            "Copy .env.example to .env and put your TMDB v3 API key in it, then re-run."
        )

    movies, genre_map = fetch_movies()

    df = pd.json_normalize(movies)
    raw_json_path = RAW_DIR / "tmdb_movies_raw.json"
    raw_csv_path = RAW_DIR / "tmdb_movies_raw.csv"

    with open(raw_json_path, "w") as f:
        json.dump(movies, f, indent=2)
    df.to_csv(raw_csv_path, index=False)

    with open(RAW_DIR / "tmdb_genre_map.json", "w") as f:
        json.dump(genre_map, f, indent=2)

    print(f"Saved {len(df)} movies to:\n  {raw_csv_path}\n  {raw_json_path}")


if __name__ == "__main__":
    main()
