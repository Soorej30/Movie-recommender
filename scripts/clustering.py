"""
K-means and hierarchical clustering for the Clustering website tab.
Run after clean_data.py.

Steps:
  1. Prepare unlabeled numeric data: keep six numeric "success" attributes,
     drop movies missing budget/revenue, log-transform the heavily skewed
     money/attention columns, and standardize everything to z-scores.
     Titles, genres and tiers are removed -- they are only re-attached
     afterwards to interpret the clusters.
  2. K-means (Euclidean) for k = 2..10: elbow (WCSS) and silhouette scores,
     then a side-by-side comparison of k = 3, 4 and 5.
  3. Hierarchical clustering (agglomerative, average linkage) using
     cosine distance (1 - cosine similarity), with dendrograms, cosine
     silhouette scores for cut levels k = 2..10, and a comparison with
     k-means via a crosstab and the adjusted Rand index.

Outputs:
  data/prepared/clustering_input.csv        unlabeled, standardized numeric data
  data/prepared/clustering_input_labeled.csv  the same rows + titles/clusters (for interpretation only)
  website/assets/img/clustering/*.png       all figures used on the tab
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, set_link_color_palette
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = ROOT / "data" / "clean"
PREP_DIR = ROOT / "data" / "prepared"
IMG_DIR = ROOT / "website" / "assets" / "img" / "clustering"
PREP_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

# same gold/maroon palette as eda_visualize.py, plus a categorical set for clusters
HEATMAP_CMAP = LinearSegmentedColormap.from_list("hp_diverging", ["#8a1c1c", "#f5ecd6", "#ae8625"])
CLUSTER_COLORS = ["#8a1c1c", "#c9a227", "#241f38", "#3d6b4f", "#4a6fa5", "#b5651d",
                  "#7a5c99", "#5f8f8b", "#a33f6b", "#6b5738"]
CLUSTER_CMAP = ListedColormap(CLUSTER_COLORS)
ACCENT_GOLD = "#ae8625"
ACCENT_RED = "#8a1c1c"
set_link_color_palette(CLUSTER_COLORS[:6])
sns.set_theme(style="whitegrid")
FIGSIZE = (8, 5)
SEED = 42

FEATURES = ["budget", "revenue", "popularity", "vote_count", "vote_average", "runtime"]
LOG_FEATURES = ["budget", "revenue", "popularity", "vote_count"]
COMPARE_KS = [3, 4, 5]
K_RANGE = range(2, 11)
DENDRO_SAMPLE = 60


def savefig(name):
    plt.tight_layout()
    plt.savefig(IMG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {name}")


def save_table_image(df, name, title, max_rows=8, float_fmt="{:.3f}"):
    subset = df.head(max_rows).copy()
    for col in subset.columns:
        if pd.api.types.is_float_dtype(subset[col]):
            subset[col] = subset[col].map(float_fmt.format)
        else:
            subset[col] = subset[col].astype(str).str.slice(0, 24)
    fig, ax = plt.subplots(figsize=(min(2 + 1.5 * len(subset.columns), 16), 0.5 * (max_rows + 2)))
    ax.axis("off")
    ax.set_title(title, fontsize=12, weight="bold", pad=12)
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
    savefig(name)


# ---------------------------------------------------------------- concept figures

def concept_figures():
    """Two illustrative figures for the Overview: partitional vs hierarchical
    clustering, and Euclidean vs cosine distance."""
    rng = np.random.default_rng(SEED)
    centers = np.array([[1, 1], [5, 1.5], [3, 5]])
    pts = np.vstack([c + rng.normal(scale=0.6, size=(12, 2)) for c in centers])
    km = KMeans(n_clusters=3, n_init=10, random_state=SEED).fit(pts)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    ax.scatter(pts[:, 0], pts[:, 1], c=km.labels_, cmap=CLUSTER_CMAP, vmin=0, vmax=9,
               s=55, edgecolor="white")
    ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], marker="X", s=260,
               c="black", label="centroid")
    ax.set_title("Partitional (k-means, k = 3)\nflat split into exactly k groups")
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    ax.legend(loc="lower right")

    ax = axes[1]
    dendrogram(linkage(pts, method="average"), ax=ax, no_labels=True,
               color_threshold=2.5, above_threshold_color="#6b5738")
    ax.axhline(2.5, ls="--", c=ACCENT_RED)
    ax.text(len(pts) * 10 * 0.98, 2.6, "cut here -> 3 clusters", ha="right", color=ACCENT_RED)
    ax.set_title("Hierarchical (agglomerative)\nnested merges shown as a dendrogram")
    ax.set_xlabel("individual points")
    ax.set_ylabel("merge distance")
    savefig("concept_partitional_vs_hierarchical.png")

    # Euclidean vs cosine
    a, b, c = np.array([1.0, 2.0]), np.array([3.0, 6.0]), np.array([2.4, 1.0])
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    for vec, name, col in [(a, "A", ACCENT_RED), (b, "B", ACCENT_RED), (c, "C", "#241f38")]:
        ax.annotate("", xy=vec, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2))
        ax.text(vec[0] + 0.1, vec[1] + 0.1, name, fontsize=13, weight="bold", color=col)
    ax.plot([a[0], c[0]], [a[1], c[1]], ls=":", c=ACCENT_GOLD, lw=2)
    ax.plot([a[0], b[0]], [a[1], b[1]], ls=":", c=ACCENT_GOLD, lw=2)

    def cos_d(u, v):
        return 1 - u @ v / (np.linalg.norm(u) * np.linalg.norm(v))

    ax.text(6.8, 0.1,
            f"Euclidean:  d(A,B) = {np.linalg.norm(a - b):.2f}   d(A,C) = {np.linalg.norm(a - c):.2f}\n"
            f"Cosine:      d(A,B) = {cos_d(a, b):.2f}   d(A,C) = {cos_d(a, c):.2f}",
            fontsize=10, va="bottom", ha="right", family="monospace",
            bbox=dict(facecolor="#f5ecd6", edgecolor=ACCENT_GOLD))
    ax.set_xlim(-0.3, 7)
    ax.set_ylim(-0.3, 7)
    ax.set_aspect("equal")
    ax.set_title("Euclidean vs. Cosine Distance\nA is closer to C by length, but points the same way as B")
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    savefig("concept_distance_metrics.png")


# ---------------------------------------------------------------- data prep

def prepare_data():
    df = pd.read_csv(CLEAN_DIR / "tmdb_movies_clean.csv")
    df = df.dropna(subset=FEATURES).reset_index(drop=True)

    before = df[["title", "primary_genre", "rating_tier"] + FEATURES].copy()
    save_table_image(before, "data_before.png",
                     "Before: cleaned movie records (labels + raw numeric values)", float_fmt="{:,.1f}")

    X = df[FEATURES].astype(float).copy()
    for col in LOG_FEATURES:
        X[col] = np.log1p(X[col])
    X = X.rename(columns={c: f"log_{c}" for c in LOG_FEATURES})
    X_scaled = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

    X_scaled.to_csv(PREP_DIR / "clustering_input.csv", index=False)
    save_table_image(X_scaled, "data_after.png",
                     "After: unlabeled, log-transformed, standardized numeric data")
    print(f"  prepared {len(X_scaled)} movies x {X_scaled.shape[1]} features")
    return df, X_scaled


# ---------------------------------------------------------------- k-means

def run_kmeans(X):
    wcss, sils = [], []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(X)
        wcss.append(km.inertia_)
        sils.append(silhouette_score(X, km.labels_))
    sil_df = pd.DataFrame({"k": list(K_RANGE), "wcss": wcss, "silhouette": sils})
    best_k = int(sil_df.loc[sil_df["silhouette"].idxmax(), "k"])
    print(sil_df.round(4).to_string(index=False))
    print(f"  best k by silhouette: {best_k}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].plot(sil_df["k"], sil_df["wcss"], marker="o", color=ACCENT_GOLD, lw=2)
    axes[0].set_title("Elbow Method: WCSS vs. k")
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("Within-cluster sum of squares")
    axes[1].plot(sil_df["k"], sil_df["silhouette"], marker="o", color=ACCENT_RED, lw=2)
    axes[1].axvline(best_k, ls="--", c="#241f38", label=f"best k = {best_k}")
    for k in COMPARE_KS:
        s = sil_df.loc[sil_df["k"] == k, "silhouette"].item()
        axes[1].annotate(f"{s:.3f}", (k, s), textcoords="offset points", xytext=(0, 8),
                         ha="center", fontsize=9)
    axes[1].set_title("Silhouette Method: average silhouette vs. k")
    axes[1].set_xlabel("Number of clusters (k)")
    axes[1].set_ylabel("Average silhouette score")
    axes[1].legend()
    for ax in axes:
        ax.set_xticks(list(K_RANGE))
    savefig("kmeans_elbow_silhouette.png")

    # compare k = 3, 4, 5 on a 2-D PCA projection (for display only)
    proj = PCA(n_components=2, random_state=SEED).fit(X.values)
    X2 = proj.transform(X.values)
    models = {k: KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(X) for k in COMPARE_KS}
    fig, axes = plt.subplots(1, len(COMPARE_KS), figsize=(16, 5), sharex=True, sharey=True)
    for ax, k in zip(axes, COMPARE_KS):
        km = models[k]
        cent2 = proj.transform(km.cluster_centers_)
        ax.scatter(X2[:, 0], X2[:, 1], c=km.labels_, cmap=CLUSTER_CMAP, vmin=0, vmax=9,
                   s=10, alpha=0.6)
        ax.scatter(cent2[:, 0], cent2[:, 1], marker="X", s=260, c="black", edgecolor="white")
        s = sil_df.loc[sil_df["k"] == k, "silhouette"].item()
        ax.set_title(f"k-means, k = {k}  (silhouette = {s:.3f})")
        ax.set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    savefig("kmeans_k_comparison.png")
    return sil_df, best_k, models


def describe_clusters(df, labels, name):
    """Cluster profiles in original units (medians) plus their size."""
    prof = df.assign(cluster=labels).groupby("cluster").agg(
        n=("title", "size"),
        budget_musd=("budget", lambda s: s.median() / 1e6),
        revenue_musd=("revenue", lambda s: s.median() / 1e6),
        popularity=("popularity", "median"),
        vote_count=("vote_count", "median"),
        vote_average=("vote_average", "median"),
        runtime=("runtime", "median"),
        top_genre=("primary_genre", lambda s: s.value_counts().index[0]),
    )
    print(f"\n  {name} cluster profiles (medians):")
    print(prof.round(2).to_string())
    return prof


def centroid_heatmap(X, km, df, k):
    cent = pd.DataFrame(km.cluster_centers_, columns=X.columns)
    sizes = np.bincount(km.labels_)
    cent.index = [f"Cluster {i} (n={sizes[i]})" for i in range(k)]
    plt.figure(figsize=(9, 0.9 * k + 2))
    sns.heatmap(cent, annot=True, fmt=".2f", cmap=HEATMAP_CMAP, center=0,
                cbar_kws={"label": "z-score (std. devs from average movie)"})
    plt.title(f"K-means Centroids (k = {k}) in Standardized Units")
    plt.xticks(rotation=25, ha="right")
    savefig(f"kmeans_centroids_k{k}.png")

    # scatter in interpretable original units
    plt.figure(figsize=FIGSIZE)
    plt.scatter(df["budget"], df["vote_average"], c=km.labels_, cmap=CLUSTER_CMAP,
                vmin=0, vmax=9, s=12, alpha=0.6)
    handles = [plt.Line2D([], [], marker="o", ls="", color=CLUSTER_COLORS[i],
                          label=f"Cluster {i}") for i in range(k)]
    plt.legend(handles=handles, loc="lower right")
    plt.xscale("log")
    plt.title(f"K-means Clusters (k = {k}): Budget vs. Rating")
    plt.xlabel("Budget (USD, log scale)")
    plt.ylabel("Vote Average (0-10)")
    savefig(f"kmeans_budget_rating_k{k}.png")


# ---------------------------------------------------------------- hierarchical

def run_hclust(X, df):
    dist = pdist(X.values, metric="cosine")
    Z = linkage(dist, method="average")

    from scipy.spatial.distance import squareform
    D = squareform(dist)
    rows = []
    for k in K_RANGE:
        lab = fcluster(Z, t=k, criterion="maxclust")
        n_found = len(np.unique(lab))
        sil = silhouette_score(D, lab, metric="precomputed") if n_found > 1 else np.nan
        sizes = sorted(np.bincount(lab)[1:], reverse=True)
        rows.append({"k": k, "silhouette_cosine": sil, "sizes": sizes})
    hc_df = pd.DataFrame(rows)
    best_k = int(hc_df.loc[hc_df["silhouette_cosine"].idxmax(), "k"])
    print("\n  hclust (cosine, average linkage):")
    print(hc_df.round(4).to_string(index=False))
    print(f"  best k by cosine silhouette: {best_k}")

    plt.figure(figsize=FIGSIZE)
    plt.plot(hc_df["k"], hc_df["silhouette_cosine"], marker="o", color=ACCENT_RED, lw=2)
    plt.axvline(best_k, ls="--", c="#241f38", label=f"best cut k = {best_k}")
    plt.title("Hierarchical Clustering (cosine): silhouette by number of clusters")
    plt.xlabel("Number of clusters after cutting the tree (k)")
    plt.ylabel("Average silhouette (cosine distance)")
    plt.xticks(list(K_RANGE))
    plt.legend()
    savefig("hclust_silhouette.png")

    # full dendrogram, truncated to the top merges so it stays readable
    cut_height = Z[-(best_k - 1), 2]
    plt.figure(figsize=(12, 5.5))
    dendrogram(Z, truncate_mode="lastp", p=40, show_leaf_counts=True, leaf_rotation=90,
               color_threshold=cut_height - 1e-9, above_threshold_color="#6b5738")
    plt.axhline(cut_height, ls="--", c=ACCENT_RED)
    plt.title(f"Dendrogram of all {len(X)} movies (cosine distance, average linkage; top 40 branches)\n"
              f"dashed line = cut giving k = {best_k} clusters; leaf labels show (# movies)")
    plt.xlabel("Cluster of movies (size)")
    plt.ylabel("Cosine distance at merge")
    savefig("hclust_dendrogram_full.png")

    # labeled dendrogram on a small, well-known sample
    sample_idx = df.nlargest(DENDRO_SAMPLE, "vote_count").index
    Xs = X.loc[sample_idx]
    Zs = linkage(pdist(Xs.values, metric="cosine"), method="average")
    labels = [f"{t[:28]} ({int(y)})" for t, y in
              zip(df.loc[sample_idx, "title"], df.loc[sample_idx, "release_year"])]
    plt.figure(figsize=(9, 14))
    dendrogram(Zs, labels=labels, orientation="left", leaf_font_size=8,
               color_threshold=0.2, above_threshold_color="#6b5738")
    plt.title(f"Dendrogram of the {DENDRO_SAMPLE} most-voted movies\n(cosine distance, average linkage)")
    plt.xlabel("Cosine distance at merge")
    savefig("hclust_dendrogram_sample.png")

    return Z, hc_df, best_k


def compare_methods(X, df, km_labels, k_km, Z, k_hc):
    hc_labels = fcluster(Z, t=k_hc, criterion="maxclust") - 1
    ari = adjusted_rand_score(km_labels, hc_labels)

    # also compare at the same k so the comparison is apples to apples
    hc_same_k = fcluster(Z, t=k_km, criterion="maxclust") - 1
    ari_same = adjusted_rand_score(km_labels, hc_same_k)
    print(f"\n  ARI k-means(k={k_km}) vs hclust(k={k_hc}): {ari:.3f}")
    print(f"  ARI k-means(k={k_km}) vs hclust(k={k_km}): {ari_same:.3f}")

    ct = pd.crosstab(pd.Series(km_labels, name=f"k-means cluster (k={k_km})"),
                     pd.Series(hc_same_k, name=f"hclust cluster (k={k_km})"))
    print(ct)
    plt.figure(figsize=(6.5, 5))
    sns.heatmap(ct, annot=True, fmt="d", cmap=LinearSegmentedColormap.from_list(
        "gold", ["#fbf6e9", ACCENT_GOLD, ACCENT_RED]))
    plt.title(f"K-means vs. Hierarchical (cosine) at k = {k_km}\nadjusted Rand index = {ari_same:.2f}")
    savefig(f"compare_kmeans_hclust_k{k_km}.png")

    proj = PCA(n_components=2, random_state=SEED).fit(X.values)
    X2 = proj.transform(X.values)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    axes[0].scatter(X2[:, 0], X2[:, 1], c=km_labels, cmap=CLUSTER_CMAP, vmin=0, vmax=9, s=10, alpha=0.6)
    axes[0].set_title(f"K-means (Euclidean), k = {k_km}")
    axes[1].scatter(X2[:, 0], X2[:, 1], c=hc_same_k, cmap=CLUSTER_CMAP, vmin=0, vmax=9, s=10, alpha=0.6)
    axes[1].set_title(f"Hierarchical (cosine, average linkage), k = {k_km}")
    for ax in axes:
        ax.set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    savefig(f"compare_pca_scatter_k{k_km}.png")
    return hc_labels, hc_same_k, ari, ari_same


def main():
    print("Concept figures...")
    concept_figures()
    print("Preparing data...")
    df, X = prepare_data()
    print("K-means...")
    sil_df, best_k, models = run_kmeans(X)
    for k in COMPARE_KS:
        describe_clusters(df, models[k].labels_, f"k-means k={k}")
    km_best = models[best_k] if best_k in models else \
        KMeans(n_clusters=best_k, n_init=20, random_state=SEED).fit(X)
    # the silhouette-best k plus k = 4, which gives a richer, more interpretable split
    detail_ks = sorted({best_k, 4})
    km_detail = {k: models.get(k) or KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(X)
                 for k in detail_ks}
    for k in detail_ks:
        centroid_heatmap(X, km_detail[k], df, k)
    print("Hierarchical clustering...")
    Z, hc_df, hc_k = run_hclust(X, df)
    for k in detail_ks:
        _, hc_same_k, _, _ = compare_methods(X, df, km_detail[k].labels_, k, Z, hc_k)
        describe_clusters(df, hc_same_k, f"hclust k={k}")
    hc_same_k = fcluster(Z, t=best_k, criterion="maxclust") - 1

    labeled = df[["id", "title", "release_year", "primary_genre", "rating_tier"]].join(X)
    labeled[f"kmeans_k{best_k}"] = km_best.labels_
    labeled[f"hclust_k{best_k}"] = hc_same_k
    labeled.to_csv(PREP_DIR / "clustering_input_labeled.csv", index=False)

    # example movies per k-means cluster, for the write-up
    for c in range(best_k):
        ex = labeled[labeled[f"kmeans_k{best_k}"] == c].merge(df[["id", "vote_count"]], on="id")
        print(f"  cluster {c} examples:", ", ".join(ex.nlargest(6, "vote_count")["title"]))
    print("Done.")


if __name__ == "__main__":
    main()
