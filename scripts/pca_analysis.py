"""
Principal Component Analysis for the PCA website tab. Run after clean_data.py.

Steps:
  1. Prepare data: PCA needs only numeric, continuous columns with no
     missing values, put on a common scale. Ten numeric movie attributes are
     kept, movies missing budget/revenue are dropped, the heavily skewed
     money/attention columns are log-transformed, and every column is
     standardized to mean 0 / std 1 (so PCA runs on the correlation matrix).
  2. Fit PCA with all components, report eigenvalues, explained variance and
     how many components are needed for 80% / 95% of the variance.
  3. Visualize: scree plot, loadings heatmap, biplot (PC1 vs PC2 with
     variable arrows), correlation circle, and a 3-D view of PC1-PC3.

Outputs:
  data/prepared/pca_input.csv          standardized numeric data fed to PCA
  data/prepared/pca_components.csv     the movies projected onto PC1-PC3
  website/assets/img/pca/*.png         all figures used on the tab
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = ROOT / "data" / "clean"
PREP_DIR = ROOT / "data" / "prepared"
IMG_DIR = ROOT / "website" / "assets" / "img" / "pca"
PREP_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

HEATMAP_CMAP = LinearSegmentedColormap.from_list("hp_diverging", ["#8a1c1c", "#f5ecd6", "#ae8625"])
TIER_COLORS = {"Poor": "#241f38", "Mixed": "#4a6fa5", "Good": "#c9a227", "Excellent": "#8a1c1c"}
ACCENT_GOLD = "#ae8625"
ACCENT_RED = "#8a1c1c"
INK = "#2b1d10"
sns.set_theme(style="whitegrid")
FIGSIZE = (8, 5)
SEED = 42

FEATURES = ["budget", "revenue", "popularity", "vote_count", "vote_average", "runtime",
            "release_year", "num_genres", "num_spoken_languages", "overview_length"]
LOG_FEATURES = ["budget", "revenue", "popularity", "vote_count"]


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
    fig, ax = plt.subplots(figsize=(min(2 + 1.4 * len(subset.columns), 18), 0.45 * (max_rows + 2)))
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
    """Overview illustrations: eigenvectors of a 2-D point cloud, and
    projecting correlated 3-D data down onto 2 principal components."""
    rng = np.random.default_rng(SEED)
    cov = np.array([[3.0, 1.8], [1.8, 1.6]])
    pts = rng.multivariate_normal([0, 0], cov, size=300)
    vals, vecs = np.linalg.eigh(np.cov(pts.T))
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.scatter(pts[:, 0], pts[:, 1], s=10, alpha=0.45, color="#6b5738")
    for i, (col, name) in enumerate([(ACCENT_RED, "PC1"), (ACCENT_GOLD, "PC2")]):
        v = vecs[:, i] * 2 * np.sqrt(vals[i])
        ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=3))
        ax.text(v[0] * 1.12, v[1] * 1.12, f"{name}\nλ = {vals[i]:.2f}", color=col,
                weight="bold", ha="center", fontsize=10)
    ax.set_aspect("equal")
    ax.set_title("Eigenvectors of the covariance matrix\ndirection = eigenvector, arrow length = 2·√(eigenvalue)")
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")

    proj = (pts - pts.mean(0)) @ vecs
    ax = axes[1]
    ax.scatter(proj[:, 0], np.zeros_like(proj[:, 0]) + rng.normal(scale=0.03, size=len(proj)),
               s=10, alpha=0.45, color=ACCENT_RED)
    ax.set_yticks([])
    ax.set_ylim(-1, 1)
    ax.set_title(f"After projecting onto PC1 only (2-D → 1-D)\n"
                 f"PC1 keeps {vals[0] / vals.sum():.0%} of the original variance")
    ax.set_xlabel("PC1 score")
    savefig("concept_eigenvectors.png")

    # 3-D -> 2-D projection
    latent = rng.normal(size=(400, 2)) * [2.5, 1.0]
    basis = np.array([[0.7, 0.5, 0.5], [-0.3, 0.8, -0.5]])
    X3 = latent @ basis + rng.normal(scale=0.25, size=(400, 3))
    p = PCA(n_components=3).fit(X3)
    Z = p.transform(X3)
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.scatter(X3[:, 0], X3[:, 1], X3[:, 2], s=8, c=Z[:, 0], cmap="hp_pc", alpha=0.7)
    ax.set_title("Original 3-D data\n(points lie close to a tilted plane)")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(Z[:, 0], Z[:, 1], s=8, c=Z[:, 0], cmap="hp_pc", alpha=0.7)
    ev = p.explained_variance_ratio_
    ax2.set_title(f"Projected onto PC1 and PC2 (3-D → 2-D)\n"
                  f"{ev[:2].sum():.1%} of variance kept, PC3 ({ev[2]:.1%}) dropped")
    ax2.set_xlabel(f"PC1 ({ev[0]:.1%})")
    ax2.set_ylabel(f"PC2 ({ev[1]:.1%})")
    ax2.set_aspect("equal")
    savefig("concept_projection_3d_to_2d.png")


# ---------------------------------------------------------------- data prep

def prepare_data():
    df = pd.read_csv(CLEAN_DIR / "tmdb_movies_clean.csv")
    df = df.dropna(subset=FEATURES).reset_index(drop=True)

    save_table_image(df[["title", "primary_genre"] + FEATURES], "data_before.png",
                     "Before: cleaned movie records (labels + raw numeric values)",
                     float_fmt="{:,.1f}")

    X = df[FEATURES].astype(float).copy()
    for col in LOG_FEATURES:
        X[col] = np.log1p(X[col])
    X = X.rename(columns={c: f"log_{c}" for c in LOG_FEATURES})
    X_scaled = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
    X_scaled.to_csv(PREP_DIR / "pca_input.csv", index=False)
    save_table_image(X_scaled, "data_after.png",
                     "After: numeric-only, log-transformed, standardized data fed to PCA",
                     float_fmt="{:.2f}")
    print(f"  prepared {len(X_scaled)} movies x {X_scaled.shape[1]} features")
    return df, X_scaled


# ---------------------------------------------------------------- PCA

def run_pca(df, X):
    pca = PCA(random_state=SEED).fit(X)
    scores = pca.transform(X)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    n80 = int(np.argmax(cum >= 0.80) + 1)
    n95 = int(np.argmax(cum >= 0.95) + 1)

    summary = pd.DataFrame({
        "PC": [f"PC{i + 1}" for i in range(len(evr))],
        "eigenvalue": pca.explained_variance_,
        "explained_var": evr,
        "cumulative": cum,
    })
    print(summary.round(4).to_string(index=False))
    print(f"  components for 80% variance: {n80};  for 95%: {n95}")

    loadings = pd.DataFrame(pca.components_.T, index=X.columns,
                            columns=[f"PC{i + 1}" for i in range(len(evr))])
    print("\n  loadings (eigenvectors), first 4 PCs:")
    print(loadings.iloc[:, :4].round(3).to_string())

    # scree + cumulative
    idx = np.arange(1, len(evr) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(idx, evr, color=ACCENT_GOLD, label="explained variance (per PC)")
    for i, v in zip(idx, evr):
        ax.text(i, v + 0.005, f"{v:.1%}", ha="center", fontsize=8)
    ax.plot(idx, cum, marker="o", color=ACCENT_RED, lw=2, label="cumulative")
    ax.axhline(0.80, ls=":", c=INK)
    ax.axhline(0.95, ls="--", c=INK)
    ax.text(len(evr) + 0.4, 0.80, "80%", va="center")
    ax.text(len(evr) + 0.4, 0.95, "95%", va="center")
    ax.set_xticks(idx)
    ax.set_xticklabels([f"PC{i}" for i in idx])
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Scree Plot: {n80} PCs keep 80% of variance, {n95} PCs keep 95%")
    ax.set_ylabel("Proportion of variance")
    ax.legend(loc="center right")
    savefig("pca_scree.png")

    # eigenvalue (Kaiser) plot
    plt.figure(figsize=FIGSIZE)
    plt.plot(idx, pca.explained_variance_, marker="o", color=ACCENT_RED, lw=2)
    plt.axhline(1.0, ls="--", c=ACCENT_GOLD, label="eigenvalue = 1 (Kaiser rule)")
    plt.xticks(idx, [f"PC{i}" for i in idx])
    plt.title("Eigenvalues of the Correlation Matrix")
    plt.ylabel("Eigenvalue (variance captured)")
    plt.legend()
    savefig("pca_eigenvalues.png")

    # loadings heatmap
    n_show = max(n80, 4)
    plt.figure(figsize=(1.2 * n_show + 3, 6))
    sns.heatmap(loadings.iloc[:, :n_show], annot=True, fmt=".2f", cmap=HEATMAP_CMAP, center=0,
                vmin=-1, vmax=1, cbar_kws={"label": "loading (eigenvector weight)"})
    plt.title("PCA Loadings: how each variable feeds each component")
    savefig("pca_loadings_heatmap.png")

    # biplot PC1 vs PC2, points colored by rating tier
    fig, ax = plt.subplots(figsize=(10, 8))
    for tier, col in TIER_COLORS.items():
        m = df["rating_tier"] == tier
        if m.any():
            ax.scatter(scores[m, 0], scores[m, 1], s=10, alpha=0.5, color=col,
                       label=f"{tier} rating ({m.sum()})")
    scale = np.abs(scores[:, :2]).max() * 0.8
    for var in loadings.index:
        x, y = loadings.loc[var, "PC1"] * scale, loadings.loc[var, "PC2"] * scale
        ax.annotate("", xy=(x, y), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6))
        ax.text(x * 1.1, y * 1.1, var, fontsize=10, weight="bold", color=INK, ha="center",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1))
    ax.axhline(0, c="grey", lw=0.8)
    ax.axvline(0, c="grey", lw=0.8)
    ax.set_xlabel(f"PC1 ({evr[0]:.1%} of variance)")
    ax.set_ylabel(f"PC2 ({evr[1]:.1%} of variance)")
    ax.set_title("PCA Biplot: movies (points) and original variables (arrows)\n"
                 "colors = rating tier, added after PCA for interpretation only")
    ax.legend(loc="lower left", markerscale=2)
    savefig("pca_biplot.png")

    # correlation circle: correlation between each variable and PC1/PC2
    corr = loadings.iloc[:, :2] * np.sqrt(pca.explained_variance_[:2])
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color=ACCENT_GOLD, lw=2))
    for var in corr.index:
        x, y = corr.loc[var, "PC1"], corr.loc[var, "PC2"]
        ax.annotate("", xy=(x, y), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=ACCENT_RED, lw=1.8))
        ax.text(x * 1.08, y * 1.08, var, fontsize=9, ha="center", color=INK)
    ax.axhline(0, c="grey", lw=0.8)
    ax.axvline(0, c="grey", lw=0.8)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.set_xlabel(f"Correlation with PC1 ({evr[0]:.1%})")
    ax.set_ylabel(f"Correlation with PC2 ({evr[1]:.1%})")
    ax.set_title("Correlation Circle (variables vs. PC1 and PC2)")
    savefig("pca_correlation_circle.png")

    # 3-D scatter PC1-PC3 colored by rating
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(projection="3d")
    for tier, col in TIER_COLORS.items():
        m = (df["rating_tier"] == tier).values
        if m.any():
            ax.scatter(scores[m, 0], scores[m, 1], scores[m, 2], s=7, alpha=0.5, color=col, label=tier)
    ax.set_xlabel(f"PC1 ({evr[0]:.1%})")
    ax.set_ylabel(f"PC2 ({evr[1]:.1%})")
    ax.set_zlabel(f"PC3 ({evr[2]:.1%})")
    ax.set_box_aspect(None, zoom=0.85)  # keep the z-axis label inside the figure
    ax.set_title(f"Movies in the first three principal components\n({cum[2]:.1%} of total variance)")
    ax.legend(title="Rating tier", markerscale=2)
    savefig("pca_3d.png")

    comp = df[["id", "title", "primary_genre", "rating_tier"]].copy()
    for i in range(3):
        comp[f"PC{i + 1}"] = scores[:, i]
    comp.to_csv(PREP_DIR / "pca_components.csv", index=False)

    print("\n  extremes along PC1 / PC2:")
    for pc in ["PC1", "PC2"]:
        print(f"   {pc} high:", ", ".join(comp.nlargest(5, pc)["title"]))
        print(f"   {pc} low :", ", ".join(comp.nsmallest(5, pc)["title"]))
    return summary, loadings


def main():
    matplotlib.colormaps.register(
        LinearSegmentedColormap.from_list("hp_pc", ["#241f38", "#8a1c1c", "#c9a227"]),
        name="hp_pc", force=True)
    print("Concept figures...")
    concept_figures()
    print("Preparing data...")
    df, X = prepare_data()
    print("PCA...")
    run_pca(df, X)
    print("Done.")


if __name__ == "__main__":
    main()
