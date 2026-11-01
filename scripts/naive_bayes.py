"""
Naive Bayes for the Naive Bayes website tab. Run after clean_data.py.

Task: predict a movie's reception class (Low / Medium / High audience
rating) from attributes known before release.

Multinomial NB needs non-negative count features, so every attribute is
turned into indicator (0/1) columns:
  genres            one column per TMDB genre (a movie can have several)
  budget band       under $20M / $20-60M / $60-120M / over $120M
  runtime band      under 100 / 100-119 / 120-139 / 140+ minutes
  release era       before 1990 / 1990s / 2000s / 2010s / 2020s
  release season    winter / spring / summer / fall
  language          English / Japanese / other
  country           US / UK / other
  genre count       1-2 / 3 / 4+
The bands use fixed cut points (not quantiles of the whole dataset) so no
information from the test set leaks into how the training data is coded.

Multinomial NB (required) and Bernoulli NB (for comparison, since the
features are binary) are both trained on the same training set and scored
on the same held-out testing set.

Outputs:
  data/prepared/nb_labeled.csv, nb_train.csv, nb_test.csv
  website/assets/img/naive_bayes/*.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from supervised_prep import (ROOT, PREP_DIR, CLASSES, CLASS_COLORS, ACCENT_GOLD, ACCENT_RED,
                             load_labeled, genre_dummies, split_ids, save_table_image,
                             save_split_image, plot_confusion)

IMG_DIR = ROOT / "website" / "assets" / "img" / "naive_bayes"
IMG_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

SEASONS = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
           6: "summer", 7: "summer", 8: "summer", 9: "fall", 10: "fall", 11: "fall"}


def build_features(df):
    parts = [genre_dummies(df)]
    bands = pd.DataFrame({
        "budget": pd.cut(df["budget_musd"], [-1, 20, 60, 120, np.inf],
                         labels=["under_20m", "20_60m", "60_120m", "over_120m"]),
        "runtime": pd.cut(df["runtime"], [0, 99, 119, 139, np.inf],
                          labels=["under_100", "100_119", "120_139", "140_plus"]),
        "era": pd.cut(df["release_year"], [0, 1989, 1999, 2009, 2019, 3000],
                      labels=["before_1990", "1990s", "2000s", "2010s", "2020s"]),
        "season": df["release_month"].map(SEASONS),
        "lang": df["original_language"].map({"en": "english", "ja": "japanese"}).fillna("other"),
        "country": df["primary_country"].map({"US": "us", "GB": "uk"}).fillna("other"),
        "n_genres": pd.cut(df["num_genres"], [0, 2, 3, np.inf], labels=["1_2", "3", "4_plus"]),
    })
    parts.append(pd.get_dummies(bands.astype(str), prefix=list(bands.columns), prefix_sep="_"))
    return pd.concat(parts, axis=1).astype(int)


def concept_figure(model, X_test, y_test, titles):
    """How NB scores one real test movie: prior x likelihood -> posterior."""
    i = int(np.flatnonzero(titles.str.contains("Whiplash"))[0]) if titles.str.contains("Whiplash").any() else 0
    x = X_test.iloc[[i]]
    log_prior = model.class_log_prior_
    log_like = (x.values @ model.feature_log_prob_.T).ravel()
    post = model.predict_proba(x).ravel()
    order = [list(model.classes_).index(c) for c in CLASSES]
    present = [c for c in x.columns if x.iloc[0][c] == 1]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    cols = [CLASS_COLORS[c] for c in CLASSES]
    axes[0].bar(CLASSES, np.exp(log_prior[order]), color=cols)
    axes[0].set_title("1. Prior  P(class)\nshare of each class in training data")
    axes[1].bar(CLASSES, log_like[order], color=cols)
    axes[1].set_title("2. Log likelihood of this movie's features\nsum of log P(feature | class)")
    axes[2].bar(CLASSES, post[order], color=cols)
    for k, v in enumerate(post[order]):
        axes[2].text(k, v + 0.01, f"{v:.2f}", ha="center")
    axes[2].set_title("3. Posterior  P(class | features)\nprior x likelihood, normalized")
    axes[2].set_ylim(0, 1)
    fig.suptitle(f"How Multinomial NB scores one test movie: {titles.iloc[i]} "
                 f"(actual class: {y_test.iloc[i]})", fontsize=12, weight="bold")
    fig.text(0.5, -0.04, "Features present: " + ", ".join(present), ha="center", fontsize=9, wrap=True)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "concept_nb_scoring.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  saved concept_nb_scoring.png")


def top_features_figure(model, columns):
    """Features whose likelihood differs most between High and Low."""
    idx = {c: list(model.classes_).index(c) for c in CLASSES}
    diff = model.feature_log_prob_[idx["High"]] - model.feature_log_prob_[idx["Low"]]
    s = pd.Series(diff, index=columns).sort_values()
    show = pd.concat([s.head(10), s.tail(10)])
    plt.figure(figsize=(8, 7))
    plt.barh(show.index, show.values, color=[ACCENT_RED if v > 0 else "#241f38" for v in show.values])
    plt.axvline(0, c="black", lw=0.8)
    plt.title("Which features push a movie toward High or Low?\n"
              "log P(feature | High) minus log P(feature | Low), Multinomial NB")
    plt.xlabel("<- more typical of Low rated movies      more typical of High rated movies ->")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "nb_top_features.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  saved nb_top_features.png")
    return s


def main():
    df = load_labeled()
    X = build_features(df)
    y = df["reception_class"]
    print(f"  {len(df)} labeled movies, {X.shape[1]} indicator features")
    print(y.value_counts().reindex(CLASSES).to_string())

    labeled = pd.concat([df[["id", "title"]], X, y], axis=1)
    labeled.to_csv(PREP_DIR / "nb_labeled.csv", index=False)

    train_ids, test_ids = split_ids(df)
    tr, te = df["id"].isin(train_ids).values, df["id"].isin(test_ids).values
    labeled[tr].to_csv(PREP_DIR / "nb_train.csv", index=False)
    labeled[te].to_csv(PREP_DIR / "nb_test.csv", index=False)
    X_train, X_test, y_train, y_test = X[tr], X[te], y[tr], y[te]

    # data images
    before_cols = ["title", "genres", "budget_musd", "runtime", "release_year", "release_month",
                   "original_language", "primary_country", "vote_average", "reception_class"]
    save_table_image(df[before_cols], IMG_DIR / "data_before.png",
                     "Before: cleaned movies with the new label (reception_class)", float_fmt="{:.1f}")
    show_cols = ["genre_action", "genre_drama", "genre_animation", "budget_over_120m",
                 "runtime_120_139", "era_2010s", "season_summer", "lang_english", "reception_class"]
    save_table_image(labeled[show_cols], IMG_DIR / "data_after.png",
                     f"After: labeled 0/1 indicator data ({X.shape[1]} feature columns, sample shown)")
    save_table_image(labeled[tr][["id", "title"] + show_cols[:5] + ["reception_class"]],
                     IMG_DIR / "train_sample.png", f"Training set sample ({tr.sum()} movies)", max_rows=6)
    save_table_image(labeled[te][["id", "title"] + show_cols[:5] + ["reception_class"]],
                     IMG_DIR / "test_sample.png", f"Testing set sample ({te.sum()} movies)", max_rows=6)
    save_split_image(y_train, y_test, IMG_DIR / "train_test_split.png",
                     "Stratified 75 / 25 split: same class mix on both sides")

    # models
    baseline = y_test.value_counts(normalize=True).max()
    results = {}
    for name, model in [("Multinomial NB", MultinomialNB(alpha=1.0)),
                        ("Bernoulli NB", BernoulliNB(alpha=1.0))]:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        cm = confusion_matrix(y_test, pred, labels=CLASSES)
        results[name] = (model, acc, cm)
        print(f"\n  {name}: test accuracy = {acc:.3f}  (train {model.score(X_train, y_train):.3f})")
        print(classification_report(y_test, pred, labels=CLASSES, digits=3))
        print(cm)
        slug = name.split()[0].lower()
        plot_confusion(cm, IMG_DIR / f"cm_{slug}.png", f"{name}: confusion matrix\naccuracy = {acc:.1%}")
    print(f"  majority class baseline = {baseline:.3f}")

    plt.figure(figsize=(7, 4))
    names = ["Majority class\nbaseline"] + list(results)
    accs = [baseline] + [results[n][1] for n in results]
    bars = plt.bar(names, accs, color=["#9c8c6c", ACCENT_RED, ACCENT_GOLD])
    for b, a in zip(bars, accs):
        plt.text(b.get_x() + b.get_width() / 2, a + 0.01, f"{a:.1%}", ha="center")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy on the testing set")
    plt.title("Naive Bayes accuracy vs. always guessing the biggest class")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "nb_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  saved nb_accuracy.png")

    mnb = results["Multinomial NB"][0]
    s = top_features_figure(mnb, X.columns)
    print("\n  most High-leaning:", list(s.tail(8).index[::-1]))
    print("  most Low-leaning:", list(s.head(8).index))
    concept_figure(mnb, X_test, y_test, df.loc[te, "title"].reset_index(drop=True))
    print("Done.")


if __name__ == "__main__":
    main()
