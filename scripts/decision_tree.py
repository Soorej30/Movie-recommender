"""
Decision trees for the Decision Trees website tab. Run after clean_data.py.

Same task and same train/test split as naive_bayes.py: predict reception
class (Low / Medium / High audience rating) from attributes known before
release. sklearn trees need numeric input, so features are numeric columns
(budget in $M, runtime, release year/month, genre count, spoken languages),
0/1 flags (English original language, US primary country) and one 0/1 column
per genre. Trees split on thresholds, so no scaling or binning is needed.

Three trees are compared:
  Tree 1  Gini,    max_depth = 3
  Tree 2  Entropy, max_depth = 4, min_samples_leaf = 15
  Tree 3  Gini,    max_depth = 3, with Tree 1's root feature removed
plus a depth sweep to show overfitting.

Outputs:
  data/prepared/dt_labeled.csv, dt_train.csv, dt_test.csv
  website/assets/img/decision_trees/*.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from supervised_prep import (ROOT, PREP_DIR, SEED, CLASSES, CLASS_COLORS, ACCENT_GOLD, ACCENT_RED,
                             load_labeled, genre_dummies, split_ids, save_table_image,
                             save_split_image, plot_confusion)

IMG_DIR = ROOT / "website" / "assets" / "img" / "decision_trees"
IMG_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")
TREE_COLORS = [CLASS_COLORS[c] for c in CLASSES]


def build_features(df):
    base = pd.DataFrame({
        "budget_musd": df["budget_musd"].round(2),
        "runtime": df["runtime"],
        "release_year": df["release_year"],
        "release_month": df["release_month"],
        "num_genres": df["num_genres"],
        "num_spoken_languages": df["num_spoken_languages"],
        "is_english": (df["original_language"] == "en").astype(int),
        "is_us": (df["primary_country"] == "US").astype(int),
    })
    return pd.concat([base, genre_dummies(df)], axis=1)


def recolor_tree(ax, model):
    """plot_tree colors by its own palette; repaint each box by majority class."""
    texts = [t for t in ax.texts if t.get_bbox_patch() is not None]
    for t, node_value in zip(texts, model.tree_.value):
        cls = model.classes_[int(np.argmax(node_value))]
        purity = node_value.max() / node_value.sum()
        t.get_bbox_patch().set_facecolor(CLASS_COLORS[cls])
        t.get_bbox_patch().set_alpha(0.25 + 0.6 * (purity - 1 / 3) / (2 / 3))


def draw_tree(model, features, path, title, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    plot_tree(model, feature_names=list(features), class_names=list(model.classes_),
              filled=True, rounded=True, impurity=True, proportion=False, fontsize=9, ax=ax)
    recolor_tree(ax, model)
    ax.set_title(title, fontsize=14, weight="bold")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {path.name}")


# ---------------------------------------------------------------- concept figures

def concept_figures():
    rng = np.random.default_rng(SEED)
    n = 120
    budget = rng.uniform(5, 200, n)
    runtime = rng.uniform(85, 170, n)
    hit = ((runtime > 130) | (budget > 140)).astype(int)
    flip = rng.random(n) < 0.08
    hit[flip] = 1 - hit[flip]
    Xt = pd.DataFrame({"budget_musd": budget, "runtime": runtime})
    yt = np.where(hit == 1, "High", "Low")
    toy = DecisionTreeClassifier(max_depth=2, random_state=SEED).fit(Xt, yt)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1.25, 1]})
    plot_tree(toy, feature_names=list(Xt.columns), class_names=list(toy.classes_), filled=True,
              rounded=True, fontsize=9, ax=axes[0], impurity=True)
    texts = [t for t in axes[0].texts if t.get_bbox_patch() is not None]
    for t, v in zip(texts, toy.tree_.value):
        t.get_bbox_patch().set_facecolor(CLASS_COLORS[toy.classes_[int(np.argmax(v))]])
        t.get_bbox_patch().set_alpha(0.45)
    axes[0].set_title("A small tree: root node, branches, leaves")

    xx, yy = np.meshgrid(np.linspace(0, 205, 300), np.linspace(80, 175, 300))
    grid = pd.DataFrame({"budget_musd": xx.ravel(), "runtime": yy.ravel()})
    zz = (toy.predict(grid) == "High").reshape(xx.shape)
    axes[1].contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5], colors=[CLASS_COLORS["Low"], CLASS_COLORS["High"]], alpha=0.18)
    for cls in ["Low", "High"]:
        m = yt == cls
        axes[1].scatter(budget[m], runtime[m], c=CLASS_COLORS[cls], s=22, label=cls, edgecolor="white")
    axes[1].set_xlabel("budget_musd")
    axes[1].set_ylabel("runtime")
    axes[1].legend(title="toy class")
    axes[1].set_title("The same tree as boxes in feature space\nevery split is a straight cut on one feature")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "concept_tree_and_regions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  saved concept_tree_and_regions.png")

    p = np.linspace(0.001, 0.999, 300)
    gini = 1 - (p ** 2 + (1 - p) ** 2)
    ent = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    plt.figure(figsize=(7, 4.6))
    plt.plot(p, ent, lw=2.5, color=ACCENT_RED, label="Entropy")
    plt.plot(p, gini, lw=2.5, color=ACCENT_GOLD, label="Gini impurity")
    plt.axvline(0.5, ls=":", c="grey")
    plt.text(0.5, 1.03, "50/50 mix: most impure", ha="center")
    plt.text(0.02, 0.06, "pure node", fontsize=9)
    plt.text(0.83, 0.06, "pure node", fontsize=9)
    plt.ylim(0, 1.12)
    plt.xlabel("Share of class A in a node (two classes)")
    plt.ylabel("Impurity")
    plt.title("Gini and Entropy: both are 0 for a pure node\nand peak when the classes are evenly mixed")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "concept_impurity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  saved concept_impurity.png")


def evaluate(name, model, X_train, y_train, X_test, y_test, slug):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred, labels=CLASSES)
    print(f"\n  {name}: test accuracy = {acc:.3f}  (train {model.score(X_train, y_train):.3f}), "
          f"depth {model.get_depth()}, {model.get_n_leaves()} leaves")
    print(classification_report(y_test, pred, labels=CLASSES, digits=3))
    print(cm)
    plot_confusion(cm, IMG_DIR / f"cm_{slug}.png", f"{name}: confusion matrix\naccuracy = {acc:.1%}")
    return acc


def main():
    print("Concept figures...")
    concept_figures()

    df = load_labeled()
    X = build_features(df)
    y = df["reception_class"]
    labeled = pd.concat([df[["id", "title"]], X, y], axis=1)
    labeled.to_csv(PREP_DIR / "dt_labeled.csv", index=False)

    train_ids, test_ids = split_ids(df)
    tr, te = df["id"].isin(train_ids).values, df["id"].isin(test_ids).values
    labeled[tr].to_csv(PREP_DIR / "dt_train.csv", index=False)
    labeled[te].to_csv(PREP_DIR / "dt_test.csv", index=False)
    X_train, X_test, y_train, y_test = X[tr], X[te], y[tr], y[te]
    print(f"  {len(df)} movies, {X.shape[1]} features, train {tr.sum()}, test {te.sum()}")

    show = ["title", "budget_musd", "runtime", "release_year", "release_month", "num_genres",
            "is_english", "genre_action", "genre_drama", "reception_class"]
    save_table_image(labeled[show], IMG_DIR / "data_labeled.png",
                     f"Labeled numeric data for the trees ({X.shape[1]} feature columns, sample shown)",
                     float_fmt="{:.1f}")
    small = ["id", "title", "budget_musd", "runtime", "genre_drama", "reception_class"]
    save_table_image(labeled[tr][small], IMG_DIR / "train_sample.png",
                     f"Training set sample ({tr.sum()} movies)", max_rows=6, float_fmt="{:.1f}")
    save_table_image(labeled[te][small], IMG_DIR / "test_sample.png",
                     f"Testing set sample ({te.sum()} movies)", max_rows=6, float_fmt="{:.1f}")
    save_split_image(y_train, y_test, IMG_DIR / "train_test_split.png",
                     "Stratified 75 / 25 split (same movies as the Naive Bayes tab)")

    baseline = y_test.value_counts(normalize=True).max()
    print(f"  majority class baseline = {baseline:.3f}")

    # Tree 1
    t1 = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=SEED)
    acc1 = evaluate("Tree 1 (Gini, depth 3)", t1, X_train, y_train, X_test, y_test, "tree1")
    draw_tree(t1, X.columns, IMG_DIR / "tree1_gini_depth3.png",
              f"Tree 1: Gini, max depth 3 (test accuracy {acc1:.1%})", (22, 9))
    root = X.columns[t1.tree_.feature[0]]
    print(f"  tree 1 root feature: {root} <= {t1.tree_.threshold[0]:.2f}")

    # Tree 2
    t2 = DecisionTreeClassifier(criterion="entropy", max_depth=4, min_samples_leaf=15, random_state=SEED)
    acc2 = evaluate("Tree 2 (Entropy, depth 4)", t2, X_train, y_train, X_test, y_test, "tree2")
    draw_tree(t2, X.columns, IMG_DIR / "tree2_entropy_depth4.png",
              f"Tree 2: Entropy, max depth 4, min 15 movies per leaf (test accuracy {acc2:.1%})", (34, 11))

    # Tree 3: drop tree 1's root feature to force a different tree
    cols3 = [c for c in X.columns if c != root]
    t3 = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=SEED)
    acc3 = evaluate(f"Tree 3 (Gini, depth 3, no {root})", t3, X_train[cols3], y_train,
                    X_test[cols3], y_test, "tree3")
    draw_tree(t3, cols3, IMG_DIR / "tree3_gini_no_root.png",
              f"Tree 3: Gini, max depth 3, without '{root}' (test accuracy {acc3:.1%})", (22, 9))
    print(f"  tree 3 root feature: {cols3[t3.tree_.feature[0]]} <= {t3.tree_.threshold[0]:.2f}")

    # feature importance for tree 2
    imp = pd.Series(t2.feature_importances_, index=X.columns).sort_values()
    imp = imp[imp > 0]
    plt.figure(figsize=(7.5, 0.38 * len(imp) + 1.5))
    plt.barh(imp.index, imp.values, color=ACCENT_GOLD)
    plt.title("Tree 2: feature importance\n(total impurity reduction from splits on each feature)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "tree2_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  saved tree2_importance.png")
    print(imp.sort_values(ascending=False).round(3).to_string())

    # depth sweep: shows overfitting
    depths = range(1, 16)
    tr_acc, te_acc = [], []
    for d in depths:
        m = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=SEED).fit(X_train, y_train)
        tr_acc.append(m.score(X_train, y_train))
        te_acc.append(m.score(X_test, y_test))
    full = DecisionTreeClassifier(random_state=SEED).fit(X_train, y_train)
    print(f"\n  unlimited depth tree: depth {full.get_depth()}, {full.get_n_leaves()} leaves, "
          f"train {full.score(X_train, y_train):.3f}, test {full.score(X_test, y_test):.3f}")
    print("  depth sweep test acc:", [round(a, 3) for a in te_acc])
    plt.figure(figsize=(8, 4.6))
    plt.plot(depths, tr_acc, marker="o", color=ACCENT_GOLD, lw=2, label="training accuracy")
    plt.plot(depths, te_acc, marker="o", color=ACCENT_RED, lw=2, label="testing accuracy")
    plt.axhline(baseline, ls=":", c="grey", label="majority class baseline")
    plt.xlabel("max_depth")
    plt.ylabel("Accuracy")
    plt.title("Deeper trees memorize the training set but stop improving on new movies")
    plt.xticks(list(depths))
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMG_DIR / "depth_sweep.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  saved depth_sweep.png")

    plt.figure(figsize=(7, 4))
    names = ["Baseline", "Tree 1\nGini d3", "Tree 2\nEntropy d4", f"Tree 3\nGini d3\nno {root}"]
    accs = [baseline, acc1, acc2, acc3]
    bars = plt.bar(names, accs, color=["#9c8c6c", ACCENT_RED, ACCENT_GOLD, "#241f38"])
    for b, a in zip(bars, accs):
        plt.text(b.get_x() + b.get_width() / 2, a + 0.01, f"{a:.1%}", ha="center")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy on the testing set")
    plt.title("Decision tree accuracy on held out movies")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "tree_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  saved tree_accuracy.png")
    print("Done.")


if __name__ == "__main__":
    main()
