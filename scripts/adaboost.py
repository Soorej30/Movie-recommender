"""
AdaBoost for the AdaBoost website tab. Run after clean_data.py.

Same task and same train/test split as naive_bayes.py and decision_tree.py:
predict reception class (Low / Medium / High audience rating) from
attributes known before release.

sklearn's AdaBoostClassifier needs numeric input. Numeric columns (budget in
$M, runtime, release year/month, genre count, spoken languages) are used as
they are, genres are already one 0/1 column each, and three categorical
columns are one-hot encoded:
  original_language   en, ja, fr, ... -> one 0/1 column per frequent value
  primary_country     US, GB, FR, ... -> one 0/1 column per frequent value
  release_season      winter / spring / summer / fall
The encoder is fit on the TRAINING rows only; any value seen in fewer than
20 training movies is pooled into an "infrequent" column.

Base learner: a decision stump (DecisionTreeClassifier(max_depth=1)).
Tuning: 5-fold stratified cross-validation on the training set over
learning_rate x n_estimators (1..500 rounds, read off with staged_predict).
The best setting is refit on the whole training set and scored once on the
testing set, then compared with Tree 2 from the Decision Trees tab.

Outputs:
  data/prepared/ada_before.csv, ada_labeled.csv, ada_train.csv, ada_test.csv
  website/assets/img/adaboost/*.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from supervised_prep import (ROOT, PREP_DIR, SEED, CLASSES, CLASS_COLORS, ACCENT_GOLD, ACCENT_RED,
                             load_labeled, genre_dummies, split_ids, save_table_image,
                             save_split_image, plot_confusion)
import decision_tree

IMG_DIR = ROOT / "website" / "assets" / "img" / "adaboost"
IMG_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

DARK = "#241f38"
SEASONS = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
           6: "summer", 7: "summer", 8: "summer", 9: "fall", 10: "fall", 11: "fall"}
NUMERIC = ["budget_musd", "runtime", "release_year", "release_month", "num_genres", "num_spoken_languages"]
CATEGORICAL = ["original_language", "primary_country", "release_season"]
LEARNING_RATES = [0.05, 0.1, 0.25, 0.5, 1.0]
MAX_ROUNDS = 500
LR_COLORS = ["#c9b27a", ACCENT_GOLD, "#b5562b", ACCENT_RED, DARK]


def savefig(name):
    plt.savefig(IMG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {name}")


def stump():
    return DecisionTreeClassifier(max_depth=1, random_state=SEED)


# ---------------------------------------------------------------- concept figures

def concept_bagging_vs_boosting():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

    def box(ax, x, y, w, h, text, fc, tc="white", fs=10):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", fc=fc, ec=DARK, lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=fs)

    def arrow(ax, x1, y1, x2, y2, color=DARK):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color=color, lw=1.6))

    ax = axes[0]
    ax.set_title("Bagging (e.g. random forest): trees built in PARALLEL", fontsize=13, weight="bold")
    box(ax, 3, 8.4, 4, 1.1, "Training data", "#9c8c6c")
    for i, x in enumerate([0.4, 3.6, 6.8]):
        box(ax, x, 5.9, 2.8, 1.1, f"bootstrap sample {i + 1}\n(random rows, with replacement)", "#e9dab3", DARK, 8.5)
        arrow(ax, 5, 8.4, x + 1.4, 7.05)
        box(ax, x, 3.6, 2.8, 1.1, f"full tree {i + 1}", ACCENT_GOLD)
        arrow(ax, x + 1.4, 5.9, x + 1.4, 4.75)
        arrow(ax, x + 1.4, 3.6, 5, 2.15)
    box(ax, 2.5, 1.0, 5, 1.1, "equal-weight vote / average", DARK)
    ax.text(5, 0.3, "trees never see each other; averaging mainly lowers variance", ha="center", fontsize=9.5,
            style="italic")

    ax = axes[1]
    ax.set_title("Boosting (e.g. AdaBoost): stumps built in SEQUENCE", fontsize=13, weight="bold")
    ys = [8.3, 5.9, 3.5]
    for i, y in enumerate(ys):
        box(ax, 0.2, y, 3.0, 1.1, f"data, weights w{i + 1}" + ("\n(all equal)" if i == 0 else "\n(mistakes heavier)"),
            "#e9dab3", DARK, 9)
        box(ax, 4.0, y, 2.6, 1.1, f"stump {i + 1}\nalpha{i + 1}", ACCENT_GOLD)
        arrow(ax, 3.25, y + 0.55, 3.95, y + 0.55)
        arrow(ax, 6.65, y + 0.55, 7.35, 1.6, ACCENT_RED)
        if i < 2:
            arrow(ax, 5.3, y, 1.7, ys[i + 1] + 1.15, ACCENT_RED)
            ax.text(3.5, y - 0.55, "up-weight what\nstump got wrong", fontsize=8, color=ACCENT_RED, ha="center")
    box(ax, 7.4, 0.9, 2.5, 1.3, "weighted vote\nsum of alpha_m x vote_m", DARK, fs=9)
    ax.text(5, 0.3, "each new stump focuses on the rows the earlier ones missed; lowers bias", ha="center",
            fontsize=9.5, style="italic")
    plt.tight_layout()
    savefig("concept_bagging_vs_boosting.png")


def concept_rounds():
    """Hand-rolled binary AdaBoost on toy data, so every weight update is visible."""
    rng = np.random.default_rng(7)
    n = 60
    X = np.column_stack([rng.uniform(0, 10, n), rng.uniform(0, 10, n)])
    y = np.where((X[:, 0] > 6) | (X[:, 1] > 7), 1, -1)
    flip = rng.random(n) < 0.06
    y[flip] *= -1
    w = np.full(n, 1 / n)
    rounds, F = [], np.zeros(n)
    for _ in range(3):
        s = DecisionTreeClassifier(max_depth=1).fit(X, y, sample_weight=w)
        pred = s.predict(X)
        err = w[pred != y].sum()
        alpha = 0.5 * np.log((1 - err) / err)
        rounds.append((s, w.copy(), pred, err, alpha))
        F += alpha * pred
        w = w * np.exp(-alpha * y * pred)
        w /= w.sum()

    col = {1: ACCENT_RED, -1: DARK}
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.2))
    xx, yy = np.meshgrid(np.linspace(0, 10, 200), np.linspace(0, 10, 200))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    for k, (s, wk, pred, err, alpha) in enumerate(rounds):
        ax = axes[k]
        zz = s.predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, zz, levels=[-2, 0, 2], colors=[DARK, ACCENT_RED], alpha=0.10)
        wrong = pred != y
        ax.scatter(X[:, 0], X[:, 1], s=wk * n * 45, c=[col[v] for v in y], edgecolor="white", zorder=3)
        ax.scatter(X[wrong, 0], X[wrong, 1], s=wk[wrong] * n * 45 + 60, facecolor="none",
                   edgecolor=ACCENT_GOLD, lw=2, zorder=4)
        f, t = s.tree_.feature[0], s.tree_.threshold[0]
        if f == 0:
            ax.axvline(t, color=ACCENT_GOLD, lw=3)
        else:
            ax.axhline(t, color=ACCENT_GOLD, lw=3)
        ax.set_title(f"Round {k + 1}: stump '{'x1' if f == 0 else 'x2'} > {t:.1f}'\n"
                     f"weighted error = {err:.2f}  ->  alpha = {alpha:.2f}", fontsize=11)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_xticks([])
        ax.set_yticks([])
    ax = axes[3]
    G = np.zeros(len(grid))
    for s, *_rest, alpha in rounds:
        G += alpha * s.predict(grid)
    ax.contourf(xx, yy, np.sign(G).reshape(xx.shape), levels=[-2, 0, 2], colors=[DARK, ACCENT_RED], alpha=0.18)
    ax.scatter(X[:, 0], X[:, 1], s=40, c=[col[v] for v in y], edgecolor="white", zorder=3)
    acc = (np.sign(F) == y).mean()
    ax.set_title(f"Final weighted vote of all 3 stumps\nsign(a1*h1 + a2*h2 + a3*h3): {acc:.0%} correct", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.suptitle("AdaBoost round by round (toy data). Dot size = row weight; gold rings = rows this stump got wrong,"
                 " which get heavier next round", fontsize=12.5, y=1.02)
    plt.tight_layout()
    savefig("concept_rounds.png")


def concept_alpha():
    e = np.linspace(0.01, 0.99, 300)
    plt.figure(figsize=(8, 4.6))
    plt.plot(e, 0.5 * np.log((1 - e) / e), color=ACCENT_RED, lw=2.5, label="two classes: 0.5 ln((1-err)/err)")
    plt.plot(e, np.log((1 - e) / e) + np.log(2), color=ACCENT_GOLD, lw=2.5,
             label="three classes (SAMME): ln((1-err)/err) + ln(2)")
    plt.axhline(0, color="grey", lw=1)
    plt.axvline(0.5, ls=":", color=ACCENT_RED)
    plt.axvline(2 / 3, ls=":", color=ACCENT_GOLD)
    plt.text(0.51, 2.6, "coin flip\n(2 classes)", color=ACCENT_RED, fontsize=9)
    plt.text(0.68, 2.6, "random guess\n(3 classes)", color=ACCENT_GOLD, fontsize=9)
    plt.ylim(-2.5, 4)
    plt.xlabel("Weighted error of a stump")
    plt.ylabel("alpha (the stump's say in the vote)")
    plt.title("Accurate stumps get a big alpha; one no better than guessing gets alpha = 0")
    plt.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    savefig("concept_alpha.png")


# ---------------------------------------------------------------- data

def raw_frame(df):
    out = df[["id", "title"]].copy()
    for c in NUMERIC:
        out[c] = df[c]
    out["original_language"] = df["original_language"]
    out["primary_country"] = df["primary_country"].fillna("unknown")
    out["release_season"] = df["release_month"].map(SEASONS)
    out["genres"] = df["genres"]
    return out


def encode(raw, tr):
    enc = OneHotEncoder(min_frequency=20, handle_unknown="infrequent_if_exist", sparse_output=False)
    enc.fit(raw.loc[tr, CATEGORICAL])
    onehot = pd.DataFrame(enc.transform(raw[CATEGORICAL]).astype(int),
                          columns=enc.get_feature_names_out(CATEGORICAL), index=raw.index)
    onehot.columns = [c.replace("infrequent_sklearn", "other") for c in onehot.columns]
    X = pd.concat([raw[NUMERIC], onehot, genre_dummies(raw)], axis=1)
    return X, onehot, enc


# ---------------------------------------------------------------- tuning

def tune(X_train, y_train):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    Xv, yv = X_train.to_numpy(), y_train.to_numpy()
    val = {lr: np.zeros((5, MAX_ROUNDS)) for lr in LEARNING_RATES}
    trn = {lr: np.zeros((5, MAX_ROUNDS)) for lr in LEARNING_RATES}
    for f, (a, b) in enumerate(skf.split(X_train, y_train)):
        for lr in LEARNING_RATES:
            m = AdaBoostClassifier(estimator=stump(), n_estimators=MAX_ROUNDS, learning_rate=lr, random_state=SEED)
            m.fit(Xv[a], yv[a])
            v = list(m.staged_score(Xv[b], yv[b]))
            t = list(m.staged_score(Xv[a], yv[a]))
            # stops early only if a stump is perfect; pad with the last value
            val[lr][f] = v + [v[-1]] * (MAX_ROUNDS - len(v))
            trn[lr][f] = t + [t[-1]] * (MAX_ROUNDS - len(t))
        print(f"  fold {f + 1} done")
    mean_val = {lr: val[lr].mean(0) for lr in LEARNING_RATES}
    mean_trn = {lr: trn[lr].mean(0) for lr in LEARNING_RATES}
    best_lr, best_n, best_acc = None, None, -1
    for lr in LEARNING_RATES:
        n = int(np.argmax(mean_val[lr])) + 1
        if mean_val[lr][n - 1] > best_acc:
            best_lr, best_n, best_acc = lr, n, mean_val[lr][n - 1]
    for lr in LEARNING_RATES:
        n = int(np.argmax(mean_val[lr])) + 1
        print(f"  lr {lr}: best rounds {n}, cv acc {mean_val[lr][n - 1]:.3f}; "
              f"at 50/100/250/500: " + ", ".join(f"{mean_val[lr][k - 1]:.3f}" for k in [50, 100, 250, 500]))
    print(f"  BEST: learning_rate {best_lr}, n_estimators {best_n}, cv acc {best_acc:.3f}")
    return mean_val, mean_trn, best_lr, best_n, best_acc


def tuning_figures(mean_val, mean_trn, best_lr, best_n, best_acc, baseline):
    rounds = np.arange(1, MAX_ROUNDS + 1)
    plt.figure(figsize=(9.5, 5.2))
    for lr, c in zip(LEARNING_RATES, LR_COLORS):
        plt.plot(rounds, mean_val[lr], color=c, lw=2, label=f"learning_rate = {lr}")
    plt.scatter([best_n], [best_acc], s=120, color=ACCENT_RED, edgecolor="white", zorder=5)
    plt.annotate(f"best: lr {best_lr}, {best_n} rounds\n{best_acc:.1%} validation accuracy", (best_n, best_acc),
                 xytext=(best_n + 30 if best_n < 300 else best_n - 180, best_acc - 0.09), fontsize=10,
                 bbox=dict(boxstyle="round", fc="white", ec=DARK, alpha=0.9),
                 arrowprops=dict(arrowstyle="->", color=DARK))
    plt.axhline(baseline, ls=":", color="grey", label="majority class baseline")
    plt.xlabel("Number of boosting rounds (stumps)")
    plt.ylabel("Mean 5-fold validation accuracy")
    plt.title("Validation accuracy as rounds are added (5-fold CV inside the training set)")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    savefig("tuning_rounds_curve.png")

    ns = [10, 25, 50, 100, 200, 300, 400, 500]
    grid = pd.DataFrame({n: [mean_val[lr][n - 1] for lr in LEARNING_RATES] for n in ns}, index=LEARNING_RATES)
    plt.figure(figsize=(9, 3.8))
    sns.heatmap(grid, annot=True, fmt=".3f", cmap=sns.light_palette(ACCENT_RED, as_cmap=True),
                cbar_kws={"label": "validation accuracy"})
    plt.xlabel("n_estimators")
    plt.ylabel("learning_rate")
    plt.title("Tuning grid: mean 5-fold validation accuracy")
    plt.tight_layout()
    savefig("tuning_grid_heatmap.png")

    plt.figure(figsize=(8.5, 4.6))
    plt.plot(rounds, mean_trn[best_lr], color=ACCENT_GOLD, lw=2, label="training folds")
    plt.plot(rounds, mean_val[best_lr], color=ACCENT_RED, lw=2, label="validation fold")
    plt.axvline(best_n, ls="--", color=DARK, label=f"chosen: {best_n} rounds")
    plt.xlabel("Number of boosting rounds")
    plt.ylabel("Mean accuracy")
    plt.title(f"learning_rate = {best_lr}: training keeps rising, validation levels off")
    plt.legend()
    plt.tight_layout()
    savefig("tuning_train_vs_val.png")


# ---------------------------------------------------------------- main

def main():
    print("Concept figures...")
    concept_bagging_vs_boosting()
    concept_rounds()
    concept_alpha()

    df = load_labeled()
    y = df["reception_class"]
    train_ids, test_ids = split_ids(df)
    tr, te = df["id"].isin(train_ids).values, df["id"].isin(test_ids).values
    assert not (tr & te).any()

    raw = raw_frame(df)
    raw_lab = pd.concat([raw, y], axis=1)
    raw_lab.to_csv(PREP_DIR / "ada_before.csv", index=False)
    X, onehot, enc = encode(raw, tr)
    labeled = pd.concat([df[["id", "title"]], X, y], axis=1)
    labeled.to_csv(PREP_DIR / "ada_labeled.csv", index=False)
    labeled[tr].to_csv(PREP_DIR / "ada_train.csv", index=False)
    labeled[te].to_csv(PREP_DIR / "ada_test.csv", index=False)
    print(f"  {len(df)} movies, {X.shape[1]} features ({onehot.shape[1]} one-hot), train {tr.sum()}, test {te.sum()}")
    print("  one-hot columns:", list(onehot.columns))

    save_table_image(raw_lab[["title", "budget_musd", "runtime", "original_language", "primary_country",
                              "release_season", "genres", "reception_class"]],
                     IMG_DIR / "data_before.png", "Before: categorical text columns (sample)", float_fmt="{:.1f}")
    show = ["title", "original_language_en", "original_language_ja", "original_language_other",
            "primary_country_US", "primary_country_GB", "release_season_fall", "release_season_summer"]
    show = [c for c in show if c in labeled.columns]
    save_table_image(labeled[show], IMG_DIR / "data_onehot.png",
                     "One-hot encoding: one 0/1 column per category value (sample)", max_cols=12)
    show2 = ["title", "budget_musd", "runtime", "release_year", "original_language_en", "primary_country_US",
             "release_season_summer", "genre_drama", "genre_animation", "reception_class"]
    save_table_image(labeled[show2], IMG_DIR / "data_after.png",
                     f"After: all-numeric data for AdaBoost ({X.shape[1]} feature columns, sample shown)",
                     float_fmt="{:.1f}")
    small = ["id", "title", "budget_musd", "runtime", "original_language_en", "genre_drama", "reception_class"]
    save_table_image(labeled[tr][small], IMG_DIR / "train_sample.png",
                     f"Training set sample ({tr.sum()} movies)", max_rows=6, float_fmt="{:.1f}")
    save_table_image(labeled[te][small], IMG_DIR / "test_sample.png",
                     f"Testing set sample ({te.sum()} movies)", max_rows=6, float_fmt="{:.1f}")
    save_split_image(y[tr], y[te], IMG_DIR / "train_test_split.png",
                     "Stratified 75 / 25 split (same movies as the Decision Trees tab)")

    X_train, X_test, y_train, y_test = X[tr], X[te], y[tr], y[te]
    baseline = y_test.value_counts(normalize=True).max()

    print("\nTuning (5-fold CV on the training set)...")
    mean_val, mean_trn, best_lr, best_n, best_acc = tune(X_train, y_train)
    tuning_figures(mean_val, mean_trn, best_lr, best_n, best_acc, y_train.value_counts(normalize=True).max())

    # default settings for reference
    default = AdaBoostClassifier(estimator=stump(), random_state=SEED).fit(X_train, y_train)
    default_acc = default.score(X_test, y_test)
    print(f"\n  default AdaBoost (50 rounds, lr 1.0): test {default_acc:.3f}")

    model = AdaBoostClassifier(estimator=stump(), n_estimators=best_n, learning_rate=best_lr, random_state=SEED)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred, labels=CLASSES)
    print(f"  tuned AdaBoost: test accuracy {acc:.3f} (train {model.score(X_train, y_train):.3f}), "
          f"baseline {baseline:.3f}")
    print(classification_report(y_test, pred, labels=CLASSES, digits=3))
    print(cm)
    plot_confusion(cm, IMG_DIR / "cm_adaboost.png",
                   f"AdaBoost ({best_n} stumps, lr {best_lr}): confusion matrix\naccuracy = {acc:.1%}")

    # the first stumps and their alphas
    rows = []
    for i, (est, a) in enumerate(zip(model.estimators_[:12], model.estimator_weights_[:12])):
        f, t = est.tree_.feature[0], est.tree_.threshold[0]
        left = est.classes_[np.argmax(est.tree_.value[est.tree_.children_left[0]])]
        right = est.classes_[np.argmax(est.tree_.value[est.tree_.children_right[0]])]
        rows.append({"round": i + 1, "question": f"{X.columns[f]} <= {t:.1f}?",
                     "yes ->": left, "no ->": right, "alpha": a})
    stumps = pd.DataFrame(rows)
    print(stumps.to_string(index=False))
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    labels = [f"{r['round']:>2}. {r['question']}  yes: {r['yes ->']}, no: {r['no ->']}" for r in rows]
    ax.barh(labels[::-1], stumps["alpha"][::-1], color=ACCENT_GOLD, edgecolor=DARK)
    for i, v in enumerate(stumps["alpha"][::-1]):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("alpha (weight of the stump in the final vote)")
    ax.set_title(f"The first 12 of {len(model.estimators_)} stumps and their alphas")
    plt.tight_layout()
    savefig("first_stumps_alpha.png")

    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values()
    imp = imp[imp > 0.005]
    plt.figure(figsize=(7.5, 0.33 * len(imp) + 1.5))
    plt.barh(imp.index, imp.values, color=ACCENT_RED)
    plt.xlabel("Importance (alpha-weighted share of splits)")
    plt.title("Tuned AdaBoost: which features the stumps used")
    plt.tight_layout()
    savefig("feature_importance.png")
    print(imp.sort_values(ascending=False).round(3).head(15).to_string())

    # comparison with decision tree 2 (same movies, its own features)
    dt_X = decision_tree.build_features(df)
    t2 = DecisionTreeClassifier(criterion="entropy", max_depth=4, min_samples_leaf=15, random_state=SEED)
    t2.fit(dt_X[tr], y_train)
    t2_acc = t2.score(dt_X[te], y_test)
    single = stump().fit(X_train, y_train)
    single_acc = single.score(X_test, y_test)
    t2_cm = confusion_matrix(y_test, t2.predict(dt_X[te]), labels=CLASSES)
    print(f"\n  single stump: {single_acc:.3f}   Decision Tree 2: {t2_acc:.3f}   AdaBoost: {acc:.3f}")
    print("  per-class recall  tree2:", (t2_cm.diagonal() / t2_cm.sum(1)).round(3),
          " ada:", (cm.diagonal() / cm.sum(1)).round(3))

    plt.figure(figsize=(8, 4.2))
    names = ["Baseline", "One stump", "Decision Tree 2\n(entropy, depth 4)", "AdaBoost\n(default, 50 rounds)",
             f"AdaBoost tuned\n({best_n} rounds, lr {best_lr})"]
    accs = [baseline, single_acc, t2_acc, default_acc, acc]
    bars = plt.bar(names, accs, color=["#9c8c6c", "#c9b27a", ACCENT_GOLD, "#b5562b", ACCENT_RED])
    for bar, a in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width() / 2, a + 0.01, f"{a:.1%}", ha="center")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy on the testing set")
    plt.title(f"Same {te.sum()} test movies: AdaBoost vs the best decision tree")
    plt.xticks(fontsize=9)
    plt.tight_layout()
    savefig("ada_vs_tree.png")
    print("Done.")


if __name__ == "__main__":
    main()
