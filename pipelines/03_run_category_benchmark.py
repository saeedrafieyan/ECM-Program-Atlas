from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

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
OUTPUT_DIR = Path("outputs/category_analysis/rna_tissue_consensus")


TISSUE_SYSTEM_MAP: Dict[str, str] = {
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

    "appendix": "immune_lymphoid",
    "bone marrow": "immune_lymphoid",
    "lymph node": "immune_lymphoid",
    "spleen": "immune_lymphoid",
    "thymus": "immune_lymphoid",
    "tonsil": "immune_lymphoid",

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

    "adipose tissue": "connective_muscle_vascular",
    "blood vessel": "connective_muscle_vascular",
    "heart muscle": "connective_muscle_vascular",
    "skeletal muscle": "connective_muscle_vascular",
    "smooth muscle": "connective_muscle_vascular",
    "skin": "connective_muscle_vascular",

    "adrenal gland": "endocrine",
    "parathyroid gland": "endocrine",
    "pituitary gland": "endocrine",
    "thyroid gland": "endocrine",

    "kidney": "urinary",
    "urinary bladder": "urinary",

    "lung": "respiratory",
}


def safe_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def normalize_gene_symbol(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .replace({"": np.nan, "NAN": np.nan, "NONE": np.nan})
    )


def load_matrisome_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Matrisome file not found: {path}")

    df = pd.read_excel(path, header=1)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all")

    required = ["Matrisome Division", "Matrisome Category", "Gene Symbol"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required Matrisome columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    df = df.copy()
    df["_gene_symbol_upper"] = normalize_gene_symbol(df["Gene Symbol"])
    df = df.dropna(subset=["_gene_symbol_upper"])
    df = df.drop_duplicates(subset=["_gene_symbol_upper"])

    return df


def load_hpa_consensus_matrix(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"HPA file not found: {path}")

    df = pd.read_csv(path, sep="\t", compression="zip")

    required = ["Gene", "Gene name", "Tissue", "nTPM"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing HPA columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

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

    variable_genes = matrix.var(axis=0) > 0
    matrix = matrix.loc[:, variable_genes]

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
        records.append({
            "tissue": tissue,
            "tissue_system": TISSUE_SYSTEM_MAP.get(tissue, "other"),
        })

    return pd.DataFrame(records)


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


def safe_silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return np.nan

    if len(unique_labels) >= X.shape[0]:
        return np.nan

    return float(silhouette_score(X, labels, metric="euclidean"))


def nearest_neighbor_same_group_score(
    X: np.ndarray,
    labels: np.ndarray,
    k: int = 5,
) -> float:
    similarity = cosine_similarity(X)
    np.fill_diagonal(similarity, -np.inf)

    scores = []

    for i in range(similarity.shape[0]):
        neighbor_idx = np.argsort(similarity[i])[::-1][:k]
        scores.append(np.mean(labels[neighbor_idx] == labels[i]))

    return float(np.mean(scores))


def compute_metrics(
    matrix: pd.DataFrame,
    tissue_labels: np.ndarray,
    n_clusters: int,
) -> Dict[str, float]:
    X = matrix.values

    n_components = min(10, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)

    cluster_labels = run_agglomerative_clustering(
        X=X,
        n_clusters=n_clusters,
    )

    metrics = {
        "n_features": matrix.shape[1],
        "pca_pc1_variance": float(pca.explained_variance_ratio_[0]),
        "pca_pc1_pc2_variance": float(np.sum(pca.explained_variance_ratio_[:2])),
        "pca_pc1_to_pc5_variance": float(
            np.sum(pca.explained_variance_ratio_[: min(5, n_components)])
        ),
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


def empirical_p_value_higher_is_better(
    observed_value: float,
    random_values: np.ndarray,
) -> float:
    random_values = random_values[~np.isnan(random_values)]

    if len(random_values) == 0 or np.isnan(observed_value):
        return np.nan

    return float((np.sum(random_values >= observed_value) + 1) / (len(random_values) + 1))


def build_feature_sets(
    matrisome_df: pd.DataFrame,
    available_genes: set[str],
    min_genes: int,
) -> Dict[str, List[str]]:
    feature_sets: Dict[str, List[str]] = {}

    all_matrisome = sorted(
        set(matrisome_df["_gene_symbol_upper"]).intersection(available_genes)
    )
    feature_sets["all_matrisome"] = all_matrisome

    for division, group in matrisome_df.groupby("Matrisome Division"):
        genes = sorted(set(group["_gene_symbol_upper"]).intersection(available_genes))
        if len(genes) >= min_genes:
            feature_sets[f"division__{safe_name(str(division))}"] = genes

    for category, group in matrisome_df.groupby("Matrisome Category"):
        genes = sorted(set(group["_gene_symbol_upper"]).intersection(available_genes))
        if len(genes) >= min_genes:
            feature_sets[f"category__{safe_name(str(category))}"] = genes

    return feature_sets


def plot_metric_bar(
    summary_df: pd.DataFrame,
    metric: str,
    output_path: Path,
) -> None:
    plot_df = summary_df.sort_values(metric, ascending=False)

    plt.figure(figsize=(12, 6))
    plt.bar(plot_df["feature_set"], plot_df[metric])
    plt.xticks(rotation=75, ha="right")
    plt.ylabel(metric)
    plt.title(f"Matrisome category comparison: {metric}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_random_distribution(
    feature_set: str,
    metric: str,
    observed_value: float,
    random_values: np.ndarray,
    output_path: Path,
) -> None:
    random_values = random_values[~np.isnan(random_values)]

    plt.figure(figsize=(8, 5))
    plt.hist(random_values, bins=30, alpha=0.8)
    plt.axvline(observed_value, linewidth=3, label=feature_set)
    plt.xlabel(metric)
    plt.ylabel("Random gene-set frequency")
    plt.title(f"{feature_set} vs random genes: {metric}")
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
        default=1000,
        help="Number of random non-ECM baselines per category.",
    )
    parser.add_argument(
        "--min-genes",
        type=int,
        default=10,
        help="Minimum genes required to evaluate a Matrisome category.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "pca_plots").mkdir(exist_ok=True)
    (OUTPUT_DIR / "random_distributions").mkdir(exist_ok=True)

    rng = np.random.default_rng(args.random_seed)

    print("[INFO] Loading Matrisome metadata...")
    matrisome_df = load_matrisome_metadata(MATRISOME_FILE)

    print("[INFO] Loading HPA consensus matrix...")
    full_matrix = load_hpa_consensus_matrix(RAW_HPA_FILE)

    available_genes = set(full_matrix.columns)
    matrisome_genes = set(matrisome_df["_gene_symbol_upper"]).intersection(available_genes)
    non_ecm_genes = sorted(available_genes.difference(matrisome_genes))

    tissue_metadata = build_tissue_metadata(full_matrix.index.tolist())
    tissue_metadata.to_csv(OUTPUT_DIR / "tissue_system_metadata.csv", index=False)

    tissue_labels = tissue_metadata["tissue_system"].values
    n_systems = len(np.unique(tissue_labels))
    n_clusters = min(n_systems, full_matrix.shape[0])

    print(f"[INFO] Full matrix shape: {full_matrix.shape}")
    print(f"[INFO] Available Matrisome genes: {len(matrisome_genes)}")
    print(f"[INFO] Available non-ECM genes: {len(non_ecm_genes)}")
    print(f"[INFO] Tissue systems: {sorted(np.unique(tissue_labels).tolist())}")

    feature_sets = build_feature_sets(
        matrisome_df=matrisome_df,
        available_genes=available_genes,
        min_genes=args.min_genes,
    )

    gene_count_df = pd.DataFrame([
        {"feature_set": name, "n_genes": len(genes)}
        for name, genes in feature_sets.items()
    ]).sort_values("n_genes", ascending=False)

    gene_count_df.to_csv(OUTPUT_DIR / "category_gene_counts.csv", index=False)

    print("\n[INFO] Feature sets:")
    print(gene_count_df)

    observed_records = []
    random_records = []
    comparison_records = []

    for feature_set_name, genes in feature_sets.items():
        print("\n" + "=" * 100)
        print(f"[INFO] Processing feature set: {feature_set_name}")
        print(f"[INFO] Number of genes: {len(genes)}")

        feature_matrix = full_matrix.loc[:, genes]

        observed_metrics = compute_metrics(
            matrix=feature_matrix,
            tissue_labels=tissue_labels,
            n_clusters=n_clusters,
        )

        observed_metrics["feature_set"] = feature_set_name
        observed_metrics["n_genes"] = len(genes)
        observed_records.append(observed_metrics)

        save_pca_plot(
            matrix=feature_matrix,
            title=f"PCA: {feature_set_name}",
            output_path=OUTPUT_DIR / "pca_plots" / f"pca_{feature_set_name}.png",
        )

        if len(non_ecm_genes) < len(genes):
            print("[WARNING] Not enough non-ECM genes for random baseline. Skipping.")
            continue

        feature_random_records = []

        for repeat_idx in range(args.n_repeats):
            sampled_genes = rng.choice(
                non_ecm_genes,
                size=len(genes),
                replace=False,
            )

            random_matrix = full_matrix.loc[:, sampled_genes]

            random_metrics = compute_metrics(
                matrix=random_matrix,
                tissue_labels=tissue_labels,
                n_clusters=n_clusters,
            )

            random_metrics["feature_set"] = feature_set_name
            random_metrics["repeat"] = repeat_idx + 1
            random_metrics["n_genes"] = len(genes)

            random_records.append(random_metrics)
            feature_random_records.append(random_metrics)

            if (repeat_idx + 1) % 100 == 0:
                print(f"  Random baselines completed: {repeat_idx + 1}/{args.n_repeats}")

        feature_random_df = pd.DataFrame(feature_random_records)

        metric_cols = [
            col for col in observed_metrics.keys()
            if col not in ["feature_set", "n_genes"]
        ]

        for metric in metric_cols:
            observed_value = float(observed_metrics[metric])
            random_values = feature_random_df[metric].astype(float).values

            random_mean = float(np.nanmean(random_values))
            random_std = float(np.nanstd(random_values, ddof=1))

            if random_std > 0:
                z_score = float((observed_value - random_mean) / random_std)
            else:
                z_score = np.nan

            p_value = empirical_p_value_higher_is_better(
                observed_value=observed_value,
                random_values=random_values,
            )

            comparison_records.append({
                "feature_set": feature_set_name,
                "n_genes": len(genes),
                "metric": metric,
                "observed_value": observed_value,
                "random_mean": random_mean,
                "random_std": random_std,
                "z_score_vs_random": z_score,
                "empirical_p_value_higher_is_better": p_value,
            })

            if metric in [
                "silhouette_pca10_space",
                "nearest_neighbor_same_system_at_3",
                "ari_agglomerative_vs_system",
                "nmi_agglomerative_vs_system",
            ]:
                plot_random_distribution(
                    feature_set=feature_set_name,
                    metric=metric,
                    observed_value=observed_value,
                    random_values=random_values,
                    output_path=(
                        OUTPUT_DIR
                        / "random_distributions"
                        / f"{feature_set_name}__{metric}.png"
                    ),
                )

    observed_df = pd.DataFrame(observed_records)
    random_df = pd.DataFrame(random_records)
    comparison_df = pd.DataFrame(comparison_records)

    observed_df.to_csv(OUTPUT_DIR / "observed_category_metrics.csv", index=False)
    random_df.to_csv(OUTPUT_DIR / "random_baseline_category_metrics.csv", index=False)
    comparison_df.to_csv(OUTPUT_DIR / "category_metric_comparison_summary.csv", index=False)

    key_metrics = [
        "pca_pc1_pc2_variance",
        "silhouette_pca10_space",
        "nearest_neighbor_same_system_at_3",
        "ari_agglomerative_vs_system",
        "nmi_agglomerative_vs_system",
    ]

    for metric in key_metrics:
        if metric in observed_df.columns:
            plot_metric_bar(
                summary_df=observed_df,
                metric=metric,
                output_path=OUTPUT_DIR / f"barplot_{metric}.png",
            )

    print("\n[DONE]")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print("\nMain files to inspect:")
    print(OUTPUT_DIR / "category_gene_counts.csv")
    print(OUTPUT_DIR / "observed_category_metrics.csv")
    print(OUTPUT_DIR / "category_metric_comparison_summary.csv")


if __name__ == "__main__":
    main()