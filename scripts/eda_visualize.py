"""
Generate the EDA figures and raw/clean data-preview images used on the
DataPrep_EDA website tab. Run after clean_data.py.

All figures are saved as PNGs with titles and labeled axes into
website/assets/img/eda/. The two-sentence captions that belong under each
figure on the website are written directly in website/dataprep_eda.html,
not regenerated here, so that the wording can be edited independently of
the plotting code.
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
CLEAN_DIR = ROOT / "data" / "clean"
EDA_IMG_DIR = ROOT / "website" / "assets" / "img" / "eda"
PREP_IMG_DIR = ROOT / "website" / "assets" / "img" / "dataprep"
EDA_IMG_DIR.mkdir(parents=True, exist_ok=True)
PREP_IMG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="YlOrRd")
PALETTE = "YlOrRd"
HEATMAP_CMAP = "RdGy_r"
ACCENT_RED = "#e10600"
FIGSIZE = (8, 5)


def savefig(name):
    plt.tight_layout()
    plt.savefig(EDA_IMG_DIR / name, dpi=150)
    plt.close()
    print(f"  saved {name}")


def save_table_image(df, path, title, max_rows=8, max_cols=8):
    subset = df.iloc[:max_rows, :max_cols].copy()
    for col in subset.columns:
        subset[col] = subset[col].astype(str).str.slice(0, 22)
    fig, ax = plt.subplots(figsize=(min(2 + 1.6 * len(subset.columns), 16), 0.5 * (max_rows + 2)))
    ax.axis("off")
    ax.set_title(title, fontsize=12, weight="bold", pad=12)
    table = ax.table(cellText=subset.values, colLabels=subset.columns, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#e10600")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#fdf2e9" if row % 2 == 0 else "#ffffff")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {path.name}")


def main():
    df = pd.read_csv(CLEAN_DIR / "tmdb_movies_clean.csv", parse_dates=["release_date"])
    raw_df = pd.read_csv(RAW_DIR / "tmdb_movies_raw.csv")

    print("Saving raw/clean data preview images...")
    save_table_image(
        raw_df[["id", "title", "release_date", "original_language", "popularity",
                "vote_average", "vote_count"]],
        PREP_IMG_DIR / "raw_sample.png", "Raw TMDB API Response (sample)",
    )
    save_table_image(
        df[["title", "release_year", "primary_genre", "language_name",
            "popularity", "vote_average", "rating_tier", "popularity_tier"]],
        PREP_IMG_DIR / "clean_sample.png", "Cleaned Dataset (sample)",
    )

    print("Generating EDA figures...")

    # 1. Histogram of release years
    plt.figure(figsize=FIGSIZE)
    sns.histplot(df["release_year"], bins=30, color=ACCENT_RED)
    plt.title("Distribution of Movies by Release Year")
    plt.xlabel("Release Year")
    plt.ylabel("Number of Movies")
    savefig("hist_release_years.png")

    # 2. Bar chart: count by genre
    top_genres = df["primary_genre"].value_counts().head(12)
    plt.figure(figsize=FIGSIZE)
    sns.barplot(x=top_genres.values, y=top_genres.index, hue=top_genres.index, palette=PALETTE, legend=False)
    plt.title("Movie Count by Primary Genre (Top 12)")
    plt.xlabel("Number of Movies")
    plt.ylabel("Genre")
    savefig("bar_genre_counts.png")

    # 3. Box plot: popularity by top genres
    top_genre_names = top_genres.head(8).index
    plt.figure(figsize=FIGSIZE)
    sns.boxplot(data=df[df["primary_genre"].isin(top_genre_names)],
                x="primary_genre", y="popularity", hue="primary_genre", palette=PALETTE, legend=False)
    plt.title("Popularity Distribution by Genre")
    plt.xlabel("Genre")
    plt.ylabel("Popularity Score")
    plt.xticks(rotation=35, ha="right")
    savefig("box_popularity_by_genre.png")

    # 4. Scatter: budget vs revenue
    bd = df.dropna(subset=["budget", "revenue"])
    if len(bd) > 5:
        plt.figure(figsize=FIGSIZE)
        sns.scatterplot(data=bd, x="budget", y="revenue", hue="primary_genre",
                         legend=False, alpha=0.6, palette=PALETTE)
        plt.xscale("log")
        plt.yscale("log")
        plt.title("Production Budget vs. Box-Office Revenue (log scale)")
        plt.xlabel("Budget (USD, log scale)")
        plt.ylabel("Revenue (USD, log scale)")
        savefig("scatter_budget_revenue.png")

    # 5. Line: average rating by decade
    by_decade = df.groupby("decade")["vote_average"].mean().reset_index()
    plt.figure(figsize=FIGSIZE)
    sns.lineplot(data=by_decade, x="decade", y="vote_average", marker="o",
                 color=ACCENT_RED)
    plt.title("Average Audience Rating by Decade")
    plt.xlabel("Decade")
    plt.ylabel("Average Vote (0-10)")
    savefig("line_avg_rating_by_decade.png")

    # 6. Bar: top languages
    top_langs = df["language_name"].value_counts().head(10)
    plt.figure(figsize=FIGSIZE)
    sns.barplot(x=top_langs.values, y=top_langs.index, hue=top_langs.index, palette=PALETTE, legend=False)
    plt.title("Top 10 Original Languages by Movie Count")
    plt.xlabel("Number of Movies")
    plt.ylabel("Language")
    savefig("bar_top_languages.png")

    # 7. Correlation heatmap
    num_cols = [c for c in ["popularity", "vote_average", "vote_count", "budget",
                             "revenue", "runtime", "profit", "roi"] if c in df.columns]
    plt.figure(figsize=(7, 6))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap=HEATMAP_CMAP, center=0)
    plt.title("Correlation Between Numeric Movie Attributes")
    savefig("heatmap_correlation.png")

    # 8. Histogram: vote_average distribution
    plt.figure(figsize=FIGSIZE)
    sns.histplot(df["vote_average"], bins=25, kde=True, color=ACCENT_RED)
    plt.title("Distribution of Audience Vote Average")
    plt.xlabel("Vote Average (0-10)")
    plt.ylabel("Number of Movies")
    savefig("hist_vote_average.png")

    # 9. Bar: avg popularity by production country
    country_pop = (df[df["primary_country"] != "Unknown"]
                   .groupby("primary_country")["popularity"].mean()
                   .sort_values(ascending=False).head(10))
    if len(country_pop) > 0:
        plt.figure(figsize=FIGSIZE)
        sns.barplot(x=country_pop.values, y=country_pop.index, hue=country_pop.index, palette=PALETTE, legend=False)
        plt.title("Average Popularity by Production Country (Top 10)")
        plt.xlabel("Average Popularity Score")
        plt.ylabel("Production Country (ISO code)")
        savefig("bar_avg_popularity_by_country.png")

    # 10. Scatter: vote_count vs vote_average
    sample = df.sample(min(1500, len(df)), random_state=42)
    plt.figure(figsize=FIGSIZE)
    sns.scatterplot(data=sample, x="vote_count", y="vote_average",
                     hue="primary_genre", alpha=0.5, legend=False, palette=PALETTE)
    plt.xscale("log")
    plt.title("Vote Count vs. Vote Average")
    plt.xlabel("Vote Count (log scale)")
    plt.ylabel("Vote Average (0-10)")
    savefig("scatter_votecount_voteaverage.png")

    # 11. Line: movies released per year
    per_year = df.groupby("release_year").size().reset_index(name="count")
    plt.figure(figsize=FIGSIZE)
    sns.lineplot(data=per_year, x="release_year", y="count",
                 color=ACCENT_RED)
    plt.title("Number of Movies Released per Year")
    plt.xlabel("Release Year")
    plt.ylabel("Number of Movies")
    savefig("line_movies_per_year.png")

    # 12. Scatter: GDP per capita vs avg popularity by country
    if "gdp_per_capita_usd" in df.columns:
        gdp = (df.dropna(subset=["gdp_per_capita_usd"])
               .groupby("primary_country")
               .agg(avg_popularity=("popularity", "mean"),
                    gdp_per_capita_usd=("gdp_per_capita_usd", "mean"),
                    n=("id", "count")).reset_index())
        gdp = gdp[gdp["n"] >= 5]
        if len(gdp) > 3:
            plt.figure(figsize=FIGSIZE)
            sns.scatterplot(data=gdp, x="gdp_per_capita_usd", y="avg_popularity",
                             size="n", legend=False, color=ACCENT_RED)
            plt.title("Production-Country GDP per Capita vs. Average Movie Popularity")
            plt.xlabel("GDP per Capita (USD)")
            plt.ylabel("Average Popularity Score")
            savefig("scatter_gdp_popularity.png")

    print("Done.")


if __name__ == "__main__":
    main()
