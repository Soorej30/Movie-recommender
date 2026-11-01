"""
Shared data prep for the supervised tabs (Naive Bayes and Decision Trees).
Imported by naive_bayes.py and decision_tree.py, not run on its own.

Label: reception_class, built from TMDB vote_average
    Low     vote_average <= 6.5
    Medium  6.5 < vote_average <= 7.2
    High    vote_average > 7.2
The cut points sit near the tertiles, so the three classes are close to
balanced (unlike rating_tier, where "Poor" has only 13 movies).

Features: only things known BEFORE a movie is released (budget, runtime,
genres, language, release timing, production country). Vote count,
popularity and revenue are left out on purpose -- they are measured after
release and would leak the answer into the model.

Both models use the SAME stratified 75/25 train/test split (same movie ids
on each side), so their accuracies can be compared directly.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = ROOT / "data" / "clean"
PREP_DIR = ROOT / "data" / "prepared"
PREP_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
TEST_SIZE = 0.25
CLASSES = ["Low", "Medium", "High"]
RATING_BINS = [0, 6.5, 7.2, 10]

ACCENT_GOLD = "#ae8625"
ACCENT_RED = "#8a1c1c"
CM_CMAP = LinearSegmentedColormap.from_list("gold", ["#fbf6e9", ACCENT_GOLD, ACCENT_RED])
CLASS_COLORS = {"Low": "#241f38", "Medium": ACCENT_GOLD, "High": ACCENT_RED}


def load_labeled():
    """Cleaned movies with a known budget, plus the reception_class label."""
    df = pd.read_csv(CLEAN_DIR / "tmdb_movies_clean.csv")
    df = df.dropna(subset=["budget", "runtime"]).reset_index(drop=True)
    df["reception_class"] = pd.cut(df["vote_average"], RATING_BINS, labels=CLASSES).astype(str)
    df["budget_musd"] = df["budget"] / 1e6
    return df


def genre_dummies(df):
    g = df["genres"].fillna("").str.get_dummies("|")
    g.columns = ["genre_" + c.lower().replace(" ", "_") for c in g.columns]
    return g


def split_ids(df):
    """Stratified split on movie id, so both models see identical train/test movies."""
    train_ids, test_ids = train_test_split(
        df["id"], test_size=TEST_SIZE, stratify=df["reception_class"], random_state=SEED)
    assert set(train_ids).isdisjoint(test_ids)
    return set(train_ids), set(test_ids)


def save_table_image(df, path, title, max_rows=8, max_cols=10, float_fmt="{:.2f}"):
    subset = df.iloc[:max_rows, :max_cols].copy()
    for col in subset.columns:
        if pd.api.types.is_float_dtype(subset[col]):
            subset[col] = subset[col].map(float_fmt.format)
        else:
            subset[col] = subset[col].astype(str).str.slice(0, 22)
    fig, ax = plt.subplots(figsize=(min(2 + 1.45 * len(subset.columns), 18), 0.42 * (len(subset) + 2)))
    ax.axis("off")
    ax.set_title(title, fontsize=12, weight="bold", pad=10)
    table = ax.table(cellText=subset.values, colLabels=subset.columns, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#8a1c1c")
            cell.set_text_props(color="#f0e6c8", weight="bold")
        else:
            cell.set_facecolor("#f5ecd6" if row % 2 == 0 else "#ffffff")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {path.name}")


def save_split_image(y_train, y_test, path, title):
    """Small bar chart showing train/test sizes and class balance."""
    counts = pd.DataFrame({
        f"Training set (n={len(y_train)})": y_train.value_counts(),
        f"Testing set (n={len(y_test)})": y_test.value_counts(),
    }).reindex(CLASSES)
    ax = counts.T.plot(kind="barh", stacked=True, figsize=(8, 2.8),
                       color=[CLASS_COLORS[c] for c in CLASSES])
    for i, col in enumerate(counts.columns):
        left = 0
        for c in CLASSES:
            v = counts.loc[c, col]
            ax.text(left + v / 2, i, str(v), ha="center", va="center", color="white", fontsize=9)
            left += v
    ax.set_title(title)
    ax.set_xlabel("Number of movies")
    ax.legend(title="Reception class", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {path.name}")


def plot_confusion(cm, path, title):
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(cm, cmap=CM_CMAP)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > cm.max() * 0.55 else "#2b1d10")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES)
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title(title)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {path.name}")
