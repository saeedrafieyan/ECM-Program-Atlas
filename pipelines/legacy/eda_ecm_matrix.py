from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler


try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


DATASET_NAME = "rna_tissue_consensus"

INPUT_MATRIX = Path(f"data/processed/{DATASET_NAME}/ecm_expression_log2_zscore.csv")
OUTPUT_DIR = Path(f"outputs/eda/{DATASET_NAME}")


def load_matrix(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Matrix file not found: {path}")

    matrix = pd.read_csv(path, index_col=0)

    # Ensure all values are numeric.
    matrix = matrix.apply(pd.to_numeric, errors="coerce")
    matrix = matrix.fillna(0.0)

    print(f"[INFO] Loaded matrix: {path}")
    print(f"[INFO] Shape: {matrix.shape[0]} samples x {matrix.shape[1]} genes")

    return matrix


def run_pca(matrix: pd.DataFrame, n_components: int = 10) -> Tuple[pd.DataFrame, PCA]:
    n_components = min(n_components, matrix.shape[0], matrix.shape[1])

    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(matrix.values)

    columns = [f"PC{i + 1}" for i in range(n_components)]
    pca_df = pd.DataFrame(coords, index=matrix.index, columns=columns)

    explained = pd.DataFrame({
        "PC": columns,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
    })

    explained.to_csv(OUTPUT_DIR / "pca_explained_variance.csv", index=False)
    pca_df.to_csv(OUTPUT_DIR / "pca_coordinates.csv")

    print("\n[PCA explained variance]")
    print(explained)

    return pca_df, pca


def save_pca_top_genes(matrix: pd.DataFrame, pca: PCA, n_top: int = 25) -> None:
    gene_names = np.array(matrix.columns)

    records = []

    for pc_idx in range(min(5, pca.components_.shape[0])):
        loadings = pca.components_[pc_idx]

        top_positive_idx = np.argsort(loadings)[-n_top:][::-1]
        top_negative_idx = np.argsort(loadings)[:n_top]

        for rank, idx in enumerate(top_positive_idx, start=1):
            records.append({
                "PC": f"PC{pc_idx + 1}",
                "direction": "positive",
                "rank": rank,
                "gene": gene_names[idx],
                "loading": loadings[idx],
            })

        for rank, idx in enumerate(top_negative_idx, start=1):
            records.append({
                "PC": f"PC{pc_idx + 1}",
                "direction": "negative",
                "rank": rank,
                "gene": gene_names[idx],
                "loading": loadings[idx],
            })

    top_genes = pd.DataFrame(records)
    top_genes.to_csv(OUTPUT_DIR / "pca_top_loading_genes.csv", index=False)


def plot_pca(pca_df: pd.DataFrame, pca: PCA) -> None:
    pc1_var = pca.explained_variance_ratio_[0] * 100
    pc2_var = pca.explained_variance_ratio_[1] * 100

    plt.figure(figsize=(10, 8))
    plt.scatter(pca_df["PC1"], pca_df["PC2"], s=60)

    for tissue, row in pca_df.iterrows():
        plt.text(row["PC1"], row["PC2"], tissue, fontsize=8)

    plt.xlabel(f"PC1 ({pc1_var:.2f}% variance)")
    plt.ylabel(f"PC2 ({pc2_var:.2f}% variance)")
    plt.title("PCA of ECM Gene Expression, HPA Tissue Consensus")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pca_pc1_pc2.png", dpi=300)
    plt.close()


def run_umap(matrix: pd.DataFrame) -> pd.DataFrame | None:
    if not HAS_UMAP:
        print("[WARNING] umap-learn is not installed. Skipping UMAP.")
        return None

    reducer = umap.UMAP(
        n_neighbors=8,
        min_dist=0.2,
        metric="euclidean",
        random_state=42,
    )

    coords = reducer.fit_transform(matrix.values)

    umap_df = pd.DataFrame(
        coords,
        index=matrix.index,
        columns=["UMAP1", "UMAP2"],
    )

    umap_df.to_csv(OUTPUT_DIR / "umap_coordinates.csv")

    plt.figure(figsize=(10, 8))
    plt.scatter(umap_df["UMAP1"], umap_df["UMAP2"], s=60)

    for tissue, row in umap_df.iterrows():
        plt.text(row["UMAP1"], row["UMAP2"], tissue, fontsize=8)

    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title("UMAP of ECM Gene Expression, HPA Tissue Consensus")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "umap.png", dpi=300)
    plt.close()

    return umap_df


def compute_similarity(matrix: pd.DataFrame) -> None:
    similarity = cosine_similarity(matrix.values)
    sim_df = pd.DataFrame(similarity, index=matrix.index, columns=matrix.index)

    sim_df.to_csv(OUTPUT_DIR / "tissue_cosine_similarity_matrix.csv")

    nearest_records = []

    for tissue in sim_df.index:
        neighbors = sim_df.loc[tissue].drop(index=tissue).sort_values(ascending=False)

        for rank, (neighbor, score) in enumerate(neighbors.head(10).items(), start=1):
            nearest_records.append({
                "tissue": tissue,
                "rank": rank,
                "nearest_tissue": neighbor,
                "cosine_similarity": score,
            })

    nearest_df = pd.DataFrame(nearest_records)
    nearest_df.to_csv(OUTPUT_DIR / "nearest_tissue_neighbors.csv", index=False)

    print("\n[Nearest-neighbor examples]")
    print(nearest_df.head(20))


def run_clustering(matrix: pd.DataFrame, n_clusters: int = 8) -> None:
    n_clusters = min(n_clusters, matrix.shape[0])

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="euclidean",
        linkage="ward",
    )

    cluster_labels = model.fit_predict(matrix.values)

    cluster_df = pd.DataFrame({
        "sample": matrix.index,
        "cluster": cluster_labels,
    }).sort_values(["cluster", "sample"])

    cluster_df.to_csv(OUTPUT_DIR / "agglomerative_clusters.csv", index=False)

    print("\n[Clusters]")
    for cluster_id, group in cluster_df.groupby("cluster"):
        samples = group["sample"].tolist()
        print(f"Cluster {cluster_id}: {samples}")


def plot_correlation_heatmap(matrix: pd.DataFrame) -> None:
    corr = np.corrcoef(matrix.values)
    corr_df = pd.DataFrame(corr, index=matrix.index, columns=matrix.index)
    corr_df.to_csv(OUTPUT_DIR / "tissue_correlation_matrix.csv")

    plt.figure(figsize=(12, 10))
    plt.imshow(corr_df.values, aspect="auto")
    plt.colorbar(label="Pearson correlation")
    plt.xticks(range(len(corr_df.columns)), corr_df.columns, rotation=90, fontsize=6)
    plt.yticks(range(len(corr_df.index)), corr_df.index, fontsize=6)
    plt.title("Tissue Correlation Based on ECM Gene Expression")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tissue_correlation_heatmap.png", dpi=300)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    matrix = load_matrix(INPUT_MATRIX)

    # The matrix is already z-scored, but this protects against any accidental scaling issue.
    scaled_values = StandardScaler().fit_transform(matrix.values)
    matrix_scaled = pd.DataFrame(
        scaled_values,
        index=matrix.index,
        columns=matrix.columns,
    )

    pca_df, pca = run_pca(matrix_scaled, n_components=10)
    save_pca_top_genes(matrix_scaled, pca, n_top=25)
    plot_pca(pca_df, pca)

    run_umap(matrix_scaled)
    compute_similarity(matrix_scaled)
    run_clustering(matrix_scaled, n_clusters=8)
    plot_correlation_heatmap(matrix_scaled)

    print("\n[DONE]")
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()