"""
Linear and logistic regression for the Regression website tab.
Run after clean_data.py.

Two models, same pre-release features as the Decision Trees tab (budget in
$M, runtime, release year/month, genre count, spoken languages, English /
US flags, one 0/1 column per genre):

  Linear regression    numeric target  revenue_musd  (worldwide box office, $M)
  Logistic regression  two-class label is_high_rated (1 if vote_average > 7.2,
                       i.e. the "High" class from the other supervised tabs)

Both use the SAME stratified 75/25 split of movie ids as naive_bayes.py and
decision_tree.py. For linear regression, movies with no revenue (or a
placeholder revenue under $1M) are dropped from each side of the split.

Logistic regression is compared against Multinomial Naive Bayes trained on
the Naive Bayes tab's features with the same two-class label and split.

Outputs:
  data/prepared/reg_labeled.csv, reg_train.csv, reg_test.csv
  website/assets/img/regression/*.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, accuracy_score,
                             confusion_matrix, classification_report)

from supervised_prep import (ROOT, PREP_DIR, SEED, ACCENT_GOLD, ACCENT_RED, load_labeled, split_ids,
                             save_table_image, save_split_image, plot_confusion)
from decision_tree import build_features
import naive_bayes

IMG_DIR = ROOT / "website" / "assets" / "img" / "regression"
IMG_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

DARK = "#241f38"
BIN_LABELS = ["Not High", "High"]
BIN_COLORS = {"Not High": DARK, "High": ACCENT_RED}
NUMERIC = ["budget_musd", "runtime", "release_year", "release_month", "num_genres", "num_spoken_languages"]
MIN_REVENUE_MUSD = 1.0


def savefig(name):
    plt.savefig(IMG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {name}")


# ---------------------------------------------------------------- concept figures

def concept_figures():
    rng = np.random.default_rng(SEED)

    # 1. linear vs logistic on toy data
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    x = rng.uniform(5, 200, 40)
    y = 2.6 * x + rng.normal(0, 70, 40) + 20
    b1, b0 = np.polyfit(x, y, 1)
    xs = np.linspace(0, 205, 100)
    ax = axes[0]
    for xi, yi in zip(x, y):
        ax.plot([xi, xi], [yi, b0 + b1 * xi], color="grey", lw=0.8, alpha=0.7)
    ax.scatter(x, y, color=ACCENT_GOLD, edgecolor="white", s=40, zorder=3, label="movies (toy data)")
    ax.plot(xs, b0 + b1 * xs, color=ACCENT_RED, lw=2.5, label=f"fitted line: y = {b0:.0f} + {b1:.2f}x")
    ax.set_xlabel("budget ($M)")
    ax.set_ylabel("revenue ($M)")
    ax.set_title("Linear regression predicts a NUMBER\nline chosen to make the squared grey gaps (residuals) smallest")
    ax.legend(loc="upper left", fontsize=9)

    ax = axes[1]
    xr = rng.uniform(80, 180, 70)
    p_true = 1 / (1 + np.exp(-(xr - 130) / 9))
    yb = (rng.random(70) < p_true).astype(int)
    toy = LogisticRegression().fit(xr.reshape(-1, 1), yb)
    xs2 = np.linspace(80, 180, 200)
    ax.scatter(xr, yb + rng.normal(0, 0.02, 70), color=[BIN_COLORS[BIN_LABELS[v]] for v in yb],
               edgecolor="white", s=40, zorder=3)
    ax.plot(xs2, toy.predict_proba(xs2.reshape(-1, 1))[:, 1], color=ACCENT_GOLD, lw=2.5,
            label="P(High) from the sigmoid")
    ax.axhline(0.5, ls=":", color="grey")
    ax.text(82, 0.53, "0.5 cut-off", fontsize=9, color="grey")
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0 = Not High", "0.5", "1 = High"])
    ax.set_xlabel("runtime (minutes)")
    ax.set_title("Logistic regression predicts a PROBABILITY\nthen a CLASS: an S-curve squeezed between 0 and 1")
    ax.legend(loc="center right", fontsize=9)
    plt.tight_layout()
    savefig("concept_linear_vs_logistic.png")

    # 2. sigmoid: score -> probability -> class
    z = np.linspace(-7, 7, 300)
    s = 1 / (1 + np.exp(-z))
    plt.figure(figsize=(8, 4.8))
    plt.fill_between(z, 0, 1, where=z < 0, color=DARK, alpha=0.06)
    plt.fill_between(z, 0, 1, where=z >= 0, color=ACCENT_RED, alpha=0.06)
    plt.plot(z, s, color=ACCENT_RED, lw=3)
    plt.axhline(0.5, ls=":", c="grey")
    plt.axvline(0, ls=":", c="grey")
    for zi in [-3, 0, 2]:
        pi = 1 / (1 + np.exp(-zi))
        plt.scatter([zi], [pi], color=ACCENT_GOLD, s=70, zorder=4, edgecolor=DARK)
        plt.annotate(f"z = {zi}  ->  p = {pi:.2f}", (zi, pi), xytext=(zi + 0.4, pi - 0.12) if zi > -1 else (zi - 3.2, pi + 0.1),
                     fontsize=10)
    plt.text(-6.8, 0.9, "predict Not High\n(p < 0.5, z < 0)", fontsize=10, color=DARK)
    plt.text(3.2, 0.12, "predict High\n(p >= 0.5, z >= 0)", fontsize=10, color=ACCENT_RED)
    plt.xlabel("linear score  z = b0 + b1*x1 + b2*x2 + ...   (the log-odds)")
    plt.ylabel("sigmoid(z) = 1 / (1 + e^(-z))")
    plt.title("The sigmoid turns any score into a probability between 0 and 1")
    plt.tight_layout()
    savefig("concept_sigmoid.png")


# ---------------------------------------------------------------- data

def build_data():
    df = load_labeled()
    X = build_features(df)
    df["revenue_musd"] = (df["revenue"] / 1e6).round(2)
    df["is_high_rated"] = (df["vote_average"] > 7.2).astype(int)
    labeled = pd.concat([df[["id", "title"]], X, df[["vote_average", "revenue_musd", "is_high_rated"]]], axis=1)
    labeled.to_csv(PREP_DIR / "reg_labeled.csv", index=False)

    train_ids, test_ids = split_ids(df)
    tr, te = df["id"].isin(train_ids).values, df["id"].isin(test_ids).values
    assert not (tr & te).any()
    labeled[tr].to_csv(PREP_DIR / "reg_train.csv", index=False)
    labeled[te].to_csv(PREP_DIR / "reg_test.csv", index=False)

    show = ["title", "budget_musd", "runtime", "release_year", "num_genres", "is_english",
            "genre_drama", "genre_action", "revenue_musd", "vote_average", "is_high_rated"]
    save_table_image(labeled[show], IMG_DIR / "data_labeled.png",
                     f"Data for both models ({X.shape[1]} features, two targets, sample shown)", float_fmt="{:.1f}")
    small = ["id", "title", "budget_musd", "runtime", "genre_drama", "revenue_musd", "is_high_rated"]
    save_table_image(labeled[tr][small], IMG_DIR / "train_sample.png",
                     f"Training set sample ({tr.sum()} movies)", max_rows=6, float_fmt="{:.1f}")
    save_table_image(labeled[te][small], IMG_DIR / "test_sample.png",
                     f"Testing set sample ({te.sum()} movies)", max_rows=6, float_fmt="{:.1f}")
    ybin = labeled["is_high_rated"].map({0: "Not High", 1: "High"})
    save_split_image(ybin[tr], ybin[te], IMG_DIR / "train_test_split.png",
                     "Same 75 / 25 split as the other supervised tabs (logistic label shown)",
                     classes=BIN_LABELS, colors=BIN_COLORS, legend_title="is_high_rated")
    return df, X, tr, te


# ---------------------------------------------------------------- linear regression

def linear_regression(df, X, tr, te):
    print("\nLinear regression: revenue_musd")
    has_rev = (df["revenue_musd"] >= MIN_REVENUE_MUSD).values
    y = df["revenue_musd"]
    a, b = tr & has_rev, te & has_rev
    print(f"  movies with revenue >= ${MIN_REVENUE_MUSD:.0f}M: train {a.sum()}, test {b.sum()}")
    model = LinearRegression().fit(X[a], y[a])
    pred = model.predict(X[b])
    rmse = mean_squared_error(y[b], pred) ** 0.5
    mae = mean_absolute_error(y[b], pred)
    r2 = r2_score(y[b], pred)
    r2_train = model.score(X[a], y[a])
    base = np.full(b.sum(), y[a].mean())
    base_rmse = mean_squared_error(y[b], base) ** 0.5
    base_mae = mean_absolute_error(y[b], base)
    print(f"  test RMSE {rmse:.1f}  MAE {mae:.1f}  R2 {r2:.3f}  (train R2 {r2_train:.3f})")
    print(f"  baseline (predict training mean {y[a].mean():.1f}): RMSE {base_rmse:.1f}  MAE {base_mae:.1f}")
    print(f"  intercept {model.intercept_:.1f}")
    coefs = pd.Series(model.coef_, index=X.columns)
    print(coefs.sort_values().round(2).to_string())

    # predicted vs actual
    plt.figure(figsize=(7, 6.2))
    lim = max(y[b].max(), pred.max()) * 1.05
    plt.scatter(y[b], pred, s=22, alpha=0.65, color=ACCENT_GOLD, edgecolor=DARK, linewidth=0.3)
    plt.plot([min(0, pred.min()), lim], [min(0, pred.min()), lim], ls="--", color=ACCENT_RED, lw=2,
             label="perfect prediction")
    for t, yt, pt in zip(df.loc[b, "title"], y[b], pred):
        if yt > 1500 or abs(yt - pt) > 900:
            plt.annotate(t, (yt, pt), fontsize=8, xytext=(4, 4), textcoords="offset points")
    plt.xlabel("Actual revenue ($M)")
    plt.ylabel("Predicted revenue ($M)")
    plt.title(f"Linear regression on {b.sum()} test movies\nRMSE \\${rmse:.0f}M, MAE \\${mae:.0f}M, R² = {r2:.2f}")
    plt.legend(loc="upper left")
    plt.tight_layout()
    savefig("linreg_pred_vs_actual.png")

    # residuals
    resid = y[b] - pred
    plt.figure(figsize=(7.5, 4.2))
    plt.scatter(pred, resid, s=18, alpha=0.6, color=DARK)
    plt.axhline(0, color=ACCENT_RED, lw=2)
    plt.xlabel("Predicted revenue ($M)")
    plt.ylabel("Actual minus predicted ($M)")
    plt.title("Residuals fan out for big predictions:\nthe model is least sure about blockbusters")
    plt.tight_layout()
    savefig("linreg_residuals.png")

    # coefficients
    c = coefs.sort_values()
    plt.figure(figsize=(8, 0.33 * len(c) + 1.5))
    plt.barh(c.index, c.values, color=[ACCENT_RED if v > 0 else DARK for v in c.values])
    plt.axvline(0, color="grey", lw=1)
    plt.xlabel("Change in predicted revenue ($M) per 1-unit increase, others held fixed")
    plt.title("Linear regression coefficients\n(0/1 columns: effect of having that genre / flag)")
    plt.tight_layout()
    savefig("linreg_coefficients.png")
    return dict(rmse=rmse, mae=mae, r2=r2, r2_train=r2_train, base_rmse=base_rmse, coefs=coefs)


# ---------------------------------------------------------------- logistic regression

def logistic_regression(df, X, tr, te):
    print("\nLogistic regression: is_high_rated")
    y = df["is_high_rated"]
    binary = [c for c in X.columns if c not in NUMERIC]
    model = make_pipeline(
        ColumnTransformer([("scale", StandardScaler(), NUMERIC)], remainder="passthrough",
                          verbose_feature_names_out=False),
        LogisticRegression(max_iter=5000))
    model.fit(X[tr], y[tr])
    pred = model.predict(X[te])
    acc = accuracy_score(y[te], pred)
    baseline = 1 - y[tr].mean()
    cm = confusion_matrix(y[te], pred, labels=[0, 1])
    print(f"  test accuracy {acc:.3f}  (train {model.score(X[tr], y[tr]):.3f}), "
          f"baseline (always Not High) {1 - y[te].mean():.3f}")
    print(classification_report(y[te], pred, target_names=BIN_LABELS, digits=3))
    print(cm)
    plot_confusion(cm, IMG_DIR / "logreg_confusion.png",
                   f"Logistic regression: confusion matrix\naccuracy = {acc:.1%}", labels=BIN_LABELS)

    lr = model[-1]
    names = model[0].get_feature_names_out()
    coefs = pd.Series(lr.coef_[0], index=names)
    sds = X.loc[tr, NUMERIC].std()
    print(f"  intercept {lr.intercept_[0]:.3f}")
    print("  numeric SDs:", sds.round(2).to_dict())
    table = pd.DataFrame({"coef": coefs, "odds_ratio": np.exp(coefs)}).sort_values("coef")
    print(table.round(3).to_string())

    # odds-ratio chart
    plt.figure(figsize=(8, 0.33 * len(table) + 1.5))
    orr = table["odds_ratio"]
    plt.barh(table.index, (orr - 1).abs(), left=np.minimum(orr, 1),
             color=[ACCENT_RED if v > 1 else DARK for v in orr])
    for i, v in enumerate(orr):
        plt.text(v * (1.06 if v > 1 else 0.94), i, f"{v:.2f}", va="center",
                 ha="left" if v > 1 else "right", fontsize=8)
    plt.axvline(1, color="grey", lw=1.5, ls="--")
    plt.xscale("log")
    ticks = [0.2, 0.33, 0.5, 1, 2, 3, 5, 10]
    plt.xticks(ticks, [str(t) for t in ticks])
    plt.xlim(0.15, 14)
    plt.xlabel("Odds ratio for High rating (log scale; 1 = no effect)\n"
               "numeric: per 1 standard deviation  |  0/1 columns: having it vs not")
    plt.title("Logistic regression: what multiplies the odds of a High rating")
    plt.tight_layout()
    savefig("logreg_odds_ratios.png")

    # predicted probabilities by true class
    proba = model.predict_proba(X[te])[:, 1]
    plt.figure(figsize=(8, 4.2))
    bins = np.linspace(0, 1, 26)
    for v, lab in enumerate(BIN_LABELS):
        plt.hist(proba[y[te].values == v], bins=bins, alpha=0.6, color=BIN_COLORS[lab], label=f"actually {lab}")
    plt.axvline(0.5, color=ACCENT_GOLD, lw=2.5, label="0.5 cut-off")
    plt.xlabel("Predicted P(High) from the sigmoid")
    plt.ylabel("Test movies")
    plt.title("Probabilities the logistic model gave the test movies")
    plt.legend()
    plt.tight_layout()
    savefig("logreg_probabilities.png")

    # Naive Bayes with the same two-class label, same split
    nb_X = naive_bayes.build_features(df)
    nb = MultinomialNB(alpha=1.0).fit(nb_X[tr], y[tr])
    nb_pred = nb.predict(nb_X[te])
    nb_acc = accuracy_score(y[te], nb_pred)
    nb_cm = confusion_matrix(y[te], nb_pred, labels=[0, 1])
    print(f"\n  Multinomial NB (same binary label): test accuracy {nb_acc:.3f}")
    print(classification_report(y[te], nb_pred, target_names=BIN_LABELS, digits=3))
    print(nb_cm)
    plot_confusion(nb_cm, IMG_DIR / "nb_binary_confusion.png",
                   f"Multinomial NB, same two-class label\naccuracy = {nb_acc:.1%}", labels=BIN_LABELS)

    plt.figure(figsize=(7, 4))
    names_ = ["Baseline\n(always Not High)", "Multinomial\nNaive Bayes", "Logistic\nregression"]
    accs = [1 - y[te].mean(), nb_acc, acc]
    bars = plt.bar(names_, accs, color=["#9c8c6c", ACCENT_GOLD, ACCENT_RED])
    for bar, a in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width() / 2, a + 0.01, f"{a:.1%}", ha="center")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy on the testing set")
    plt.title(f"High vs Not High on the same {te.sum()} test movies")
    plt.tight_layout()
    savefig("logreg_vs_nb.png")
    return dict(acc=acc, nb_acc=nb_acc, cm=cm, nb_cm=nb_cm, table=table, sds=sds)


def main():
    print("Concept figures...")
    concept_figures()
    df, X, tr, te = build_data()
    print(f"  {len(df)} movies, {X.shape[1]} features, train {tr.sum()}, test {te.sum()}, "
          f"High share {df['is_high_rated'].mean():.3f}")
    linear_regression(df, X, tr, te)
    logistic_regression(df, X, tr, te)
    print("Done.")


if __name__ == "__main__":
    main()
