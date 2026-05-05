from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


RAW_HPA_FILE = Path("data/raw/hpa/rna_tissue_consensus.tsv.zip")
MATRISOME_FILE = Path("data/raw/matrisome/human_matrisome.xlsx")
OUTPUT_DIR = Path("outputs/specificity/rna_tissue_consensus")


TISSUE_SYSTEM_MAP: Dict[str, str] = {
    # Central nervous system and neural-related tissues
    "amygdala": "CNS",
    "basal ganglia": "CNS",
    "cerebellum": "CNS",
    "cerebral cortex": "CNS",
    "choroid plexus": "CNS",
    "hippocampal formation": "CNS",
    "hypothalamus": "CNS",
    "midbrain": "CNS",
    "retina": "CNS",
    "spinal cord": "CNS",

    # Immune / hematopoietic / lymphoid
    "appendix": "immune_lymphoid",
    "bone marrow": "immune_lymphoid",
    "lymph node": "immune_lymphoid",
    "spleen": "immune_lymphoid",
    "thymus": "immune_lymphoid",
    "tonsil": "immune_lymphoid",

    # Digestive / hepatobiliary / oral
    "colon": "digestive",
    "duodenum": "digestive",
    "esophagus": "digestive",
    "gallbladder": "digestive",
    "liver": "digestive",
    "pancreas": "digestive",
    "rectum": "digestive",
    "salivary gland": "digestive",
    "small intestine": "digestive",
    "stomach": "digestive",
    "tongue": "digestive",

    # Reproductive
    "breast": "reproductive",
    "cervix": "reproductive",
    "endometrium": "reproductive",
    "epididymis": "reproductive",
    "fallopian tube": "reproductive",
    "ovary": "reproductive",
    "placenta": "reproductive",
    "prostate": "reproductive",
    "seminal vesicle": "reproductive",
    "testis": "reproductive",
    "vagina": "reproductive",

    # Musculoskeletal / stromal / connective-rich
    "adipose tissue": "connective_muscle_vascular",
    "blood vessel": "connective_muscle_vascular",
    "heart muscle": "connective_muscle_vascular",
    "skeletal muscle": "connective_muscle_vascular",
    "smooth muscle": "connective_muscle_vascular",
    "skin": "connective_muscle_vascular",

    # Endocrine
    "adrenal gland": "endocrine",
    "parathyroid gland": "endocrine",
    "pituitary gland": "endocrine",
    "thyroid gland": "endocrine",

    # Urinary
    "kidney": "urinary",
    "urinary bladder": "urinary",

    # Respiratory
    "lung": "respiratory",
}


def normalize_gene_symbol(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .replace({"": np.nan, "NAN": np.nan, "NONE": np.nan})
    )


def load_matrisome_genes(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Matrisome file not found: {path}")

    matrisome_df = pd.read_excel(path, header=1)
    matrisome_df.columns = [str(c).strip() for c in matrisome_df.columns]

    if "Gene Symbol" not in matrisome_df.columns:
        raise ValueError(
            "Could not find 'Gene Symbol' column in Matrisome file. "
            f"Columns found: {matrisome_df.columns.tolist()}"
        )

    genes = (
        normalize_gene_symbol(matrisome_df["Gene Symbol"])
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    return genes


def load_hpa_consensus_matrix(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"HPA file not found: {path}")

    df = pd.read_csv(path, sep="\t", compression="zip")
    required = ["Gene", "Gene name", "Tissue", "nTPM"]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing HPA columns: {missing}")

    df = df.copy()
    df["_gene_symbol_upper"] = normalize_gene_symbol(df["Gene name"])
    df["nTPM"] = pd.to_numeric(df["nTPM"], errors="coerce")
    df = df.dropna(subset=["_gene_symbol_upper", "nTPM"])

    matrix = df.pivot_table(
        index="Tissue",
        columns="_gene_symbol_upper",
        values="nTPM",
        aggfunc="mean",
    )

    matrix = matrix.fillna(0.0)
    matrix = matrix.sort_index()
    matrix = matrix.reindex(sorted(matrix.columns), axis=1)

    # Remove genes with no variation across tissues.
    variable_genes = matrix.var(axis=0) > 0
    matrix = matrix.loc[:, variable_genes]

    # Log transform, then z-score each gene across tissues.
    matrix_log2 = np.log2(matrix + 1.0)

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(matrix_log2.values)

    matrix_scaled = pd.DataFrame(
        scaled_values,
        index=matrix_log2.index,
        columns=matrix_log2.columns,
    )

    return matrix_scaled


def build_tissue_metadata(tissues: List[str]) -> pd.DataFrame:
    records = []

    for tissue in tissues:
        system = TISSUE_SYSTEM_MAP.get(tissue, "other")
        records.append({"tissue": tissue, "tissue_system": system})

    metadata = pd.DataFrame(records)
    return metadata


def run_agglomerative_clustering(X: np.ndarray, n_clusters: int) -> np.ndarray:
    try:
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="euclidean",
            linkage="ward",
        )
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity="euclidean",
            linkage="ward",
        )

    return model.fit_predict(X)


def nearest_neighbor_same_group_score(
    X: np.ndarray,
    labels: np.ndarray,
    k: int = 5,
) -> float:
    similarity = cosine_similarity(X)

    # Exclude self-neighbor.
    np.fill_diagonal(similarity, -np.inf)

    same_group_scores = []

    for i in range(similarity.shape[0]):
        neighbor_idx = np.argsort(similarity[i])[::-1][:k]
        same_fraction = np.mean(labels[neighbor_idx] == labels[i])
        same_group_scores.append(same_fraction)

    return float(np.mean(same_group_scores))


def safe_silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return np.nan

    if len(unique_labels) >= X.shape[0]:
        return np.nan

    return float(silhouette_score(X, labels, metric="euclidean"))


def compute_metrics(
    matrix: pd.DataFrame,
    tissue_labels: np.ndarray,
    n_clusters: int,
) -> Dict[str, float]:
    X = matrix.values

    n_components = min(10, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)

    cluster_labels = run_agglomerative_clustering(X, n_clusters=n_clusters)

    metrics = {
        "n_features": matrix.shape[1],
        "pca_pc1_variance": float(pca.explained_variance_ratio_[0]),
        "pca_pc1_pc2_variance": float(np.sum(pca.explained_variance_ratio_[:2])),
        "pca_pc1_to_pc5_variance": float(np.sum(pca.explained_variance_ratio_[:5])),
        "silhouette_original_space": safe_silhouette_score(X, tissue_labels),
        "silhouette_pca10_space": safe_silhouette_score(X_pca, tissue_labels),
        "nearest_neighbor_same_system_at_3": nearest_neighbor_same_group_score(
            X, tissue_labels, k=3
        ),
        "nearest_neighbor_same_system_at_5": nearest_neighbor_same_group_score(
            X, tissue_labels, k=5
        ),
        "ari_agglomerative_vs_system": float(
            adjusted_rand_score(tissue_labels, cluster_labels)
        ),
        "nmi_agglomerative_vs_system": float(
            normalized_mutual_info_score(tissue_labels, cluster_labels)
        ),
    }

    return metrics


def empirical_p_value_higher_is_better(ecm_value: float, random_values: np.ndarray) -> float:
    random_values = random_values[~np.isnan(random_values)]

    if len(random_values) == 0 or np.isnan(ecm_value):
        return np.nan

    return float((np.sum(random_values >= ecm_value) + 1) / (len(random_values) + 1))


def plot_random_distribution(
    metric_name: str,
    ecm_value: float,
    random_values: np.ndarray,
    output_path: Path,
) -> None:
    random_values = random_values[~np.isnan(random_values)]

    plt.figure(figsize=(8, 5))
    plt.hist(random_values, bins=30, alpha=0.8)
    plt.axvline(ecm_value, linewidth=3, label="ECM genes")
    plt.xlabel(metric_name)
    plt.ylabel("Random gene-set frequency")
    plt.title(f"ECM vs random non-ECM genes: {metric_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_pca_plot(
    matrix: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    X = matrix.values

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)

    pca_df = pd.DataFrame(
        coords,
        index=matrix.index,
        columns=["PC1", "PC2"],
    )

    pc1 = pca.explained_variance_ratio_[0] * 100
    pc2 = pca.explained_variance_ratio_[1] * 100

    plt.figure(figsize=(10, 8))
    plt.scatter(pca_df["PC1"], pca_df["PC2"], s=60)

    for tissue, row in pca_df.iterrows():
        plt.text(row["PC1"], row["PC2"], tissue, fontsize=8)

    plt.xlabel(f"PC1 ({pc1:.2f}% variance)")
    plt.ylabel(f"PC2 ({pc2:.2f}% variance)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=300,
        help="Number of random non-ECM gene-set repeats.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.random_seed)

    print("[INFO] Loading Matrisome genes...")
    matrisome_genes = load_matrisome_genes(MATRISOME_FILE)

    print("[INFO] Loading HPA consensus expression matrix...")
    full_matrix = load_hpa_consensus_matrix(RAW_HPA_FILE)

    print(f"[INFO] Full matrix shape: {full_matrix.shape}")

    tissue_metadata = build_tissue_metadata(full_matrix.index.tolist())
    tissue_metadata.to_csv(OUTPUT_DIR / "tissue_system_metadata.csv", index=False)

    tissue_labels = tissue_metadata["tissue_system"].values
    n_systems = len(np.unique(tissue_labels))
    n_clusters = min(n_systems, full_matrix.shape[0])

    available_genes = set(full_matrix.columns)
    ecm_genes = sorted(set(matrisome_genes).intersection(available_genes))
    non_ecm_genes = sorted(available_genes.difference(ecm_genes))

    print(f"[INFO] Available full-transcriptome genes: {len(available_genes)}")
    print(f"[INFO] Matched ECM genes: {len(ecm_genes)}")
    print(f"[INFO] Available non-ECM genes: {len(non_ecm_genes)}")
    print(f"[INFO] Tissue systems: {sorted(np.unique(tissue_labels).tolist())}")

    if len(ecm_genes) == 0:
        raise ValueError("No ECM genes matched HPA gene names.")

    if len(non_ecm_genes) < len(ecm_genes):
        raise ValueError("Not enough non-ECM genes for random baseline sampling.")

    ecm_matrix = full_matrix.loc[:, ecm_genes]

    print("[INFO] Computing ECM metrics...")
    ecm_metrics = compute_metrics(
        matrix=ecm_matrix,
        tissue_labels=tissue_labels,
        n_clusters=n_clusters,
    )
    ecm_metrics["feature_set"] = "matrisome_ecm"

    print("[INFO] Computing full-transcriptome reference metrics...")
    full_metrics = compute_metrics(
        matrix=full_matrix,
        tissue_labels=tissue_labels,
        n_clusters=n_clusters,
    )
    full_metrics["feature_set"] = "full_transcriptome"

    random_records = []

    print(f"[INFO] Running {args.n_repeats} random non-ECM baselines...")

    for repeat_idx in range(args.n_repeats):
        sampled_genes = rng.choice(
            non_ecm_genes,
            size=len(ecm_genes),
            replace=False,
        )

        random_matrix = full_matrix.loc[:, sampled_genes]

        metrics = compute_metrics(
            matrix=random_matrix,
            tissue_labels=tissue_labels,
            n_clusters=n_clusters,
        )
        metrics["feature_set"] = "random_non_ecm"
        metrics["repeat"] = repeat_idx + 1

        random_records.append(metrics)

        if (repeat_idx + 1) % 25 == 0:
            print(f"  Completed {repeat_idx + 1}/{args.n_repeats}")

    random_df = pd.DataFrame(random_records)
    random_df.to_csv(OUTPUT_DIR / "random_baseline_metrics.csv", index=False)

    reference_df = pd.DataFrame([ecm_metrics, full_metrics])
    reference_df.to_csv(OUTPUT_DIR / "reference_metrics.csv", index=False)

    metric_names = [
        col for col in random_df.columns
        if col not in ["feature_set", "repeat"]
    ]

    comparison_records = []

    for metric in metric_names:
        ecm_value = float(ecm_metrics[metric])
        full_value = float(full_metrics[metric])
        random_values = random_df[metric].astype(float).values

        random_mean = float(np.nanmean(random_values))
        random_std = float(np.nanstd(random_values, ddof=1))

        if random_std > 0:
            ecm_z = float((ecm_value - random_mean) / random_std)
        else:
            ecm_z = np.nan

        p_value = empirical_p_value_higher_is_better(
            ecm_value=ecm_value,
            random_values=random_values,
        )

        comparison_records.append({
            "metric": metric,
            "ecm_value": ecm_value,
            "random_mean": random_mean,
            "random_std": random_std,
            "ecm_z_score_vs_random": ecm_z,
            "empirical_p_value_higher_is_better": p_value,
            "full_transcriptome_value": full_value,
        })

        plot_random_distribution(
            metric_name=metric,
            ecm_value=ecm_value,
            random_values=random_values,
            output_path=OUTPUT_DIR / f"random_distribution_{metric}.png",
        )

    comparison_df = pd.DataFrame(comparison_records)
    comparison_df.to_csv(OUTPUT_DIR / "metric_comparison_summary.csv", index=False)

    # Save PCA visualization for ECM and one random reference set.
    save_pca_plot(
        matrix=ecm_matrix,
        title="PCA using Matrisome ECM genes",
        output_path=OUTPUT_DIR / "pca_matrisome_ecm.png",
    )

    sampled_reference_genes = rng.choice(
        non_ecm_genes,
        size=len(ecm_genes),
        replace=False,
    )
    random_reference_matrix = full_matrix.loc[:, sampled_reference_genes]

    save_pca_plot(
        matrix=random_reference_matrix,
        title="PCA using one random non-ECM gene set",
        output_path=OUTPUT_DIR / "pca_random_non_ecm_reference.png",
    )

    print("\n[DONE]")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print("\nMain file to inspect:")
    print(OUTPUT_DIR / "metric_comparison_summary.csv")

    print("\nMetric comparison:")
    print(comparison_df)


if __name__ == "__main__":
    main()