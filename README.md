# What Drives Movie Success? — CSCI 5612 Project (Module 1)

Topic: what combination of genre, budget, cast/crew, language, region, and
release timing is associated with a movie becoming popular, highly rated, and
commercially successful. Data comes from The Movie Database (TMDB) API, with
World Bank Open Data used as a secondary, government-sourced dataset for
country-level economic context.

## Repo layout

```
scripts/            Python data-gathering, cleaning, and EDA scripts
data/raw/           Raw pulls from TMDB and World Bank
data/clean/         Cleaned, analysis-ready dataset
website/            The static project website (all course-required tabs)
```

## One-time setup

1. Get a free TMDB API key:
   - Create an account: https://www.themoviedb.org/signup
   - Request an API key under Settings -> API: https://www.themoviedb.org/settings/api
   - Copy the "API Key (v3 auth)" value.
2. `cp .env.example .env` and paste the key in as `TMDB_API_KEY=...`.
3. Install dependencies: `pip install -r scripts/requirements.txt`

## Running the pipeline

Run in this order from the project root:

```bash
python scripts/fetch_tmdb.py          # pulls movies matching the filters in CONFIG (year, genre, language, region, rating, vote count)
python scripts/fetch_tmdb_details.py  # adds budget, revenue, runtime, production countries per movie
python scripts/fetch_worldbank.py     # pulls GDP-per-capita / population by country (no key needed)
python scripts/clean_data.py          # cleans + merges everything into data/clean/tmdb_movies_clean.csv
python scripts/eda_visualize.py       # generates all figures into website/assets/img/
```

Edit the `CONFIG` dictionary at the top of `scripts/fetch_tmdb.py` to change
which years, genres, languages, region, rating range, vote-count threshold,
or sort order are pulled from TMDB.

## Viewing the site locally

```bash
cd website && python -m http.server 8000
```
then open http://localhost:8000

## Publishing with GitHub Pages

1. Push this repo to GitHub.
2. In repo Settings -> Pages, set the source to the `main` branch, root folder.
3. The live site's home page will be at `https://<username>.github.io/<repo>/website/index.html`.
   (Relative links from `website/*.html` to `../data/...` and `../scripts/...`
   rely on Pages serving the whole repo, not just the `website/` folder --
   keep the source set to the repo root.)

## Notes on data size

`fetch_tmdb.py` defaults to pulling up to 2,000 movies (`max_pages: 100`,
20 results/page) sorted by popularity. That is enough for meaningful EDA and
modeling in later modules without producing an unwieldy repo. Increase
`max_pages` (TMDB caps discover queries at 500 pages / ~10,000 results) if a
larger sample is needed later.
