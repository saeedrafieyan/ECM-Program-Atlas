from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Callable, Dict, List

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


RAW_DIR = Path("data/raw")
MATRISOME_FILE = RAW_DIR / "matrisome" / "human_matrisome.xlsx"
OUTPUT_DIR = Path("outputs/cross_dataset_reproducibility")


DATASET_CONFIGS = {
    "rna_tissue_consensus": {
        "path": RAW_DIR / "hpa" / "rna_tissue_consensus.tsv.zip",
        "sample_col": "Tissue",
        "expr_col": "nTPM",
        "label_mapper": "tissue_system",
    },
    "rna_tissue_hpa": {
        "path": RAW_DIR / "hpa" / "rna_tissue_hpa.tsv.zip",
        "sample_col": "Tissue",
        "expr_col": "nTPM",
        "label_mapper": "tissue_system",
    },
    "rna_tissue_gtex": {
        "path": RAW_DIR / "hpa" / "rna_tissue_gtex.tsv.zip",
        "sample_col": "Tissue",
        "expr_col": "nTPM",
        "label_mapper": "tissue_system",
    },
    "rna_tissue_detail_gtex": {
        "path": RAW_DIR / "hpa" / "rna_tissue_detail_gtex.tsv.zip",
        "sample_col": "Source tissue",
        "expr_col": "nTPM",
        "label_mapper": "gtex_detail_system",
    },
    "rna_brain_hpa": {
        "path": RAW_DIR / "hpa" / "rna_brain_hpa.tsv.zip",
        "sample_col": "Subregion",
        "expr_col": "nTPM",
        "label_mapper": "brain_region",
    },
    "rna_pfc_brain_hpa": {
        "path": RAW_DIR / "hpa" / "rna_pfc_brain_hpa.tsv.zip",
        "sample_col": "Subregion",
        "expr_col": "nTPM",
        "label_mapper": "brain_region",
    },
    "rna_single_cell_type": {
        "path": RAW_DIR / "hpa" / "rna_single_cell_type.tsv.zip",
        "sample_col": "Cell type",
        "expr_col": "nCPM",
        "label_mapper": "cell_lineage",
    },
}


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


def normalize_text(text: str) -> str:
    return str(text).strip().lower()


def normalize_gene_symbol(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .replace({"": np.nan, "NAN": np.nan, "NONE": np.nan})
    )


def safe_name(name: str) -> str:
    name = normalize_text(name)
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def map_tissue_system(sample: str) -> str:
    key = normalize_text(sample)
    return TISSUE_SYSTEM_MAP.get(key, "unknown")


def map_gtex_detail_system(sample: str) -> str:
    s = normalize_text(sample)

    rules = [
        ("CNS", ["brain", "cerebell", "cortex", "hippocampus", "amygdala"]),
        ("immune_lymphoid", ["blood", "spleen", "lymph", "bone marrow"]),
        ("digestive", ["colon", "esophagus", "stomach", "liver", "pancreas", "small intestine", "terminal ileum"]),
        ("reproductive", ["breast", "uterus", "vagina", "ovary", "testis", "prostate", "cervix"]),
        ("connective_muscle_vascular", ["adipose", "artery", "heart", "muscle", "skin", "nerve"]),
        ("endocrine", ["adrenal", "thyroid", "pituitary"]),
        ("urinary", ["kidney", "bladder"]),
        ("respiratory", ["lung"]),
    ]

    for label, keywords in rules:
        if any(keyword in s for keyword in keywords):
            return label

    return "unknown"


def map_brain_region(sample: str) -> str:
    s = normalize_text(sample)

    rules = [
        ("cerebral_cortex", ["cortex", "frontal", "temporal", "parietal", "occipital", "prefrontal"]),
        ("hippocampal_formation", ["hippocampus", "dentate", "subiculum", "ca1", "ca2", "ca3", "ca4"]),
        ("amygdala", ["amygdala"]),
        ("basal_ganglia", ["basal ganglia", "caudate", "putamen", "globus pallidus", "accumbens"]),
        ("thalamus", ["thalamus"]),
        ("hypothalamus", ["hypothalamus"]),
        ("midbrain", ["midbrain", "substantia", "tectum", "tegmentum"]),
        ("cerebellum", ["cerebell"]),
        ("brainstem", ["pons", "medulla"]),
        ("spinal_cord", ["spinal cord"]),
        ("choroid_plexus", ["choroid plexus"]),
        ("white_matter", ["white matter", "corpus callosum"]),
    ]

    for label, keywords in rules:
        if any(keyword in s for keyword in keywords):
            return label

    return "unknown"


def map_cell_lineage(sample: str) -> str:
    s = normalize_text(sample)

    rules = [
        ("stromal", ["fibroblast", "pericyte", "stromal", "mesenchymal", "chondrocyte", "osteoblast"]),
        ("endothelial", ["endothelial"]),
        ("epithelial", ["epithelial", "keratinocyte", "basal cell", "club cell", "alveolar", "goblet"]),
        ("immune", ["t-cell", "t cell", "b-cell", "b cell", "macrophage", "monocyte", "dendritic", "neutrophil", "nk-cell", "nk cell", "plasma cell", "mast cell"]),
        ("neural", ["neuron", "astrocyte", "oligodendrocyte", "microglia", "schwann"]),
        ("muscle", ["muscle", "myocyte", "smooth muscle", "cardiomyocyte"]),
        ("endocrine", ["endocrine", "beta cell", "alpha cell", "delta cell", "adrenal", "thyroid"]),
        ("adipocyte", ["adipocyte"]),
        ("germ_reproductive", ["spermatid", "spermatocyte", "sertoli", "leydig", "oocyte"]),
        ("hematopoietic", ["erythroid", "megakaryocyte", "hematopoietic"]),
    ]

    for label, keywords in rules:
        if any(keyword in s for keyword in keywords):
            return label

    return "unknown"


LABEL_MAPPERS: Dict[str, Callable[[str], str]] = {
    "tissue_system": map_tissue_system,
    "gtex_detail_system": map_gtex_detail_system,
    "brain_region": map_brain_region,
    "cell_lineage": map_cell_lineage,
}


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    p = p_values.astype(float).to_numpy()
    n = len(p)

    order = np.argsort(p)
    ranked = p[order]

    adjusted = np.empty(n, dtype=float)
    cumulative_min = 1.0

    for i in range(n - 1, -1, -1):
        rank = i + 1
        value = ranked[i] * n / rank
        cumulative_min = min(cumulative_min, value)
        adjusted[order[i]] = min(cumulative_min, 1.0)

    return pd.Series(adjusted, index=p_values.index)


def load_matrisome_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Matrisome file not found: {path}")

    df = pd.read_excel(path, header=1)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all")

    required = ["Matrisome Division", "Matrisome Category", "Gene Symbol"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing Matrisome columns: {missing}")

    df = df.copy()
    df["_gene_symbol_upper"] = normalize_gene_symbol(df["Gene Symbol"])
    df = df.dropna(subset=["_gene_symbol_upper"])
    df = df.drop_duplicates(subset=["_gene_symbol_upper"])

    return df


def load_expression_matrix(path: Path, sample_col: str, expr_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expression file not found: {path}")

    df = pd.read_csv(path, sep="\t", compression="zip")

    required = ["Gene name", sample_col, expr_col]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing columns in {path}: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    df = df.copy()
    df["_gene_symbol_upper"] = normalize_gene_symbol(df["Gene name"])
    df[expr_col] = pd.to_numeric(df[expr_col], errors="coerce")
    df = df.dropna(subset=["_gene_symbol_upper", sample_col, expr_col])

    matrix = df.pivot_table(
        index=sample_col,
        columns="_gene_symbol_upper",
        values=expr_col,
        aggfunc="mean",
    )

    matrix = matrix.fillna(0.0)
    matrix = matrix.sort_index()
    matrix = matrix.reindex(sorted(matrix.columns), axis=1)

    variable_genes = matrix.var(axis=0) > 0
    matrix = matrix.loc[:, variable_genes]

    matrix_log2 = np.log2(matrix + 1.0)

    scaled_values = StandardScaler().fit_transform(matrix_log2.values)

    matrix_scaled = pd.DataFrame(
        scaled_values,
        index=matrix_log2.index,
        columns=matrix_log2.columns,
    )

    return matrix_scaled


def build_labels(samples: List[str], mapper_name: str) -> pd.DataFrame:
    mapper = LABEL_MAPPERS[mapper_name]

    records = []
    for sample in samples:
        label = mapper(sample)
        records.append({"sample": sample, "label": label})

    return pd.DataFrame(records)


def filter_labeled_samples(
    matrix: pd.DataFrame,
    labels_df: pd.DataFrame,
    min_group_size: int,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    labels_df = labels_df.copy()
    labels_df = labels_df[labels_df["label"] != "unknown"]

    counts = labels_df["label"].value_counts()
    valid_labels = counts[counts >= min_group_size].index.tolist()

    labels_df = labels_df[labels_df["label"].isin(valid_labels)]

    valid_samples = labels_df["sample"].tolist()

    matrix_filtered = matrix.loc[valid_samples].copy()
    labels = labels_df["label"].values

    return matrix_filtered, labels, labels_df


def build_feature_sets(
    matrisome_df: pd.DataFrame,
    available_genes: set[str],
    min_genes: int,
) -> Dict[str, List[str]]:
    feature_sets: Dict[str, List[str]] = {}

    all_matrisome = sorted(
        set(matrisome_df["_gene_symbol_upper"]).intersection(available_genes)
    )

    if len(all_matrisome) >= min_genes:
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
    k: int,
) -> float:
    if X.shape[0] <= k:
        return np.nan

    similarity = cosine_similarity(X)
    np.fill_diagonal(similarity, -np.inf)

    scores = []

    for i in range(X.shape[0]):
        neighbor_idx = np.argsort(similarity[i])[::-1][:k]
        scores.append(np.mean(labels[neighbor_idx] == labels[i]))

    return float(np.mean(scores))


def compute_metrics(
    matrix: pd.DataFrame,
    labels: np.ndarray,
    n_clusters: int,
) -> Dict[str, float]:
    X = matrix.values

    n_components = min(10, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)

    cluster_labels = run_agglomerative_clustering(X, n_clusters=n_clusters)

    return {
        "n_samples": matrix.shape[0],
        "n_features": matrix.shape[1],
        "n_label_classes": len(np.unique(labels)),
        "pca_pc1_variance": float(pca.explained_variance_ratio_[0]),
        "pca_pc1_pc2_variance": float(np.sum(pca.explained_variance_ratio_[:2])),
        "pca_pc1_to_pc5_variance": float(np.sum(pca.explained_variance_ratio_[: min(5, n_components)])),
        "silhouette_original_space": safe_silhouette_score(X, labels),
        "silhouette_pca10_space": safe_silhouette_score(X_pca, labels),
        "nearest_neighbor_same_label_at_3": nearest_neighbor_same_group_score(X, labels, k=3),
        "nearest_neighbor_same_label_at_5": nearest_neighbor_same_group_score(X, labels, k=5),
        "ari_agglomerative_vs_label": float(adjusted_rand_score(labels, cluster_labels)),
        "nmi_agglomerative_vs_label": float(normalized_mutual_info_score(labels, cluster_labels)),
    }


def empirical_p_value_higher_is_better(observed_value: float, random_values: np.ndarray) -> float:
    random_values = random_values[~np.isnan(random_values)]

    if len(random_values) == 0 or np.isnan(observed_value):
        return np.nan

    return float((np.sum(random_values >= observed_value) + 1) / (len(random_values) + 1))


def plot_cross_dataset_heatmap(summary: pd.DataFrame, metric: str, output_path: Path) -> None:
    metric_df = summary[summary["metric"] == metric].copy()

    if metric_df.empty:
        return

    pivot = metric_df.pivot_table(
        index="feature_set",
        columns="dataset",
        values="z_score_vs_random",
        aggfunc="mean",
    )

    plt.figure(figsize=(12, 7))
    plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(label="z-score vs matched random genes")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title(f"Cross-dataset reproducibility: {metric}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-repeats", type=int, default=300)
    parser.add_argument("--min-genes", type=int, default=10)
    parser.add_argument("--min-group-size", type=int, default=2)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.random_seed)

    matrisome_df = load_matrisome_metadata(MATRISOME_FILE)

    observed_records = []
    random_records = []
    comparison_records = []
    dataset_summary_records = []

    for dataset_name, config in DATASET_CONFIGS.items():
        print("\n" + "#" * 100)
        print(f"[DATASET] {dataset_name}")

        dataset_output_dir = OUTPUT_DIR / dataset_name
        dataset_output_dir.mkdir(parents=True, exist_ok=True)

        path = config["path"]

        if not path.exists():
            print(f"[SKIP] Missing file: {path}")
            continue

        matrix = load_expression_matrix(
            path=path,
            sample_col=config["sample_col"],
            expr_col=config["expr_col"],
        )

        labels_df = build_labels(matrix.index.tolist(), config["label_mapper"])
        labels_df.to_csv(dataset_output_dir / "sample_labels_raw.csv", index=False)

        matrix_labeled, labels, labels_filtered_df = filter_labeled_samples(
            matrix=matrix,
            labels_df=labels_df,
            min_group_size=args.min_group_size,
        )

        labels_filtered_df.to_csv(dataset_output_dir / "sample_labels_used.csv", index=False)

        print(f"[INFO] Original matrix: {matrix.shape}")
        print(f"[INFO] Labeled matrix: {matrix_labeled.shape}")
        print(f"[INFO] Labels: {labels_filtered_df['label'].value_counts().to_dict()}")

        if matrix_labeled.shape[0] < 6 or len(np.unique(labels)) < 2:
            print("[SKIP] Not enough labeled samples or label classes.")
            dataset_summary_records.append({
                "dataset": dataset_name,
                "status": "skipped",
                "reason": "not enough labeled samples/classes",
                "n_samples_original": matrix.shape[0],
                "n_samples_labeled": matrix_labeled.shape[0],
                "n_label_classes": len(np.unique(labels)),
            })
            continue

        available_genes = set(matrix_labeled.columns)
        matrisome_genes = set(matrisome_df["_gene_symbol_upper"]).intersection(available_genes)
        non_ecm_genes = sorted(available_genes.difference(matrisome_genes))

        feature_sets = build_feature_sets(
            matrisome_df=matrisome_df,
            available_genes=available_genes,
            min_genes=args.min_genes,
        )

        gene_counts = pd.DataFrame([
            {
                "dataset": dataset_name,
                "feature_set": feature_set_name,
                "n_genes": len(genes),
            }
            for feature_set_name, genes in feature_sets.items()
        ])

        gene_counts.to_csv(dataset_output_dir / "feature_set_gene_counts.csv", index=False)

        n_clusters = len(np.unique(labels))

        dataset_summary_records.append({
            "dataset": dataset_name,
            "status": "processed",
            "reason": "",
            "n_samples_original": matrix.shape[0],
            "n_samples_labeled": matrix_labeled.shape[0],
            "n_label_classes": len(np.unique(labels)),
            "n_available_genes": len(available_genes),
            "n_matrisome_genes": len(matrisome_genes),
            "n_non_ecm_genes": len(non_ecm_genes),
        })

        for feature_set_name, genes in feature_sets.items():
            print(f"[INFO] {dataset_name} | {feature_set_name} | genes={len(genes)}")

            feature_matrix = matrix_labeled.loc[:, genes]

            observed_metrics = compute_metrics(
                matrix=feature_matrix,
                labels=labels,
                n_clusters=n_clusters,
            )

            observed_metrics.update({
                "dataset": dataset_name,
                "feature_set": feature_set_name,
            })

            observed_records.append(observed_metrics)

            feature_random_records = []

            if len(non_ecm_genes) < len(genes):
                print("[WARNING] Not enough non-ECM genes for matched random baseline.")
                continue

            for repeat_idx in range(args.n_repeats):
                sampled_genes = rng.choice(non_ecm_genes, size=len(genes), replace=False)
                random_matrix = matrix_labeled.loc[:, sampled_genes]

                random_metrics = compute_metrics(
                    matrix=random_matrix,
                    labels=labels,
                    n_clusters=n_clusters,
                )

                random_metrics.update({
                    "dataset": dataset_name,
                    "feature_set": feature_set_name,
                    "repeat": repeat_idx + 1,
                })

                random_records.append(random_metrics)
                feature_random_records.append(random_metrics)

            feature_random_df = pd.DataFrame(feature_random_records)

            metric_cols = [
                col for col in observed_metrics.keys()
                if col not in ["dataset", "feature_set"]
            ]

            for metric in metric_cols:
                observed_value = float(observed_metrics[metric])
                random_values = feature_random_df[metric].astype(float).values

                random_mean = float(np.nanmean(random_values))
                random_std = float(np.nanstd(random_values, ddof=1))

                z_score = (
                    float((observed_value - random_mean) / random_std)
                    if random_std > 0 else np.nan
                )

                p_value = empirical_p_value_higher_is_better(
                    observed_value=observed_value,
                    random_values=random_values,
                )

                comparison_records.append({
                    "dataset": dataset_name,
                    "feature_set": feature_set_name,
                    "metric": metric,
                    "observed_value": observed_value,
                    "random_mean": random_mean,
                    "random_std": random_std,
                    "z_score_vs_random": z_score,
                    "empirical_p_value_higher_is_better": p_value,
                    "n_genes": len(genes),
                })

    observed_df = pd.DataFrame(observed_records)
    random_df = pd.DataFrame(random_records)
    comparison_df = pd.DataFrame(comparison_records)
    dataset_summary_df = pd.DataFrame(dataset_summary_records)

    if not comparison_df.empty:
        comparison_df["fdr_bh_p_value"] = benjamini_hochberg(
            comparison_df["empirical_p_value_higher_is_better"]
        )

    observed_df.to_csv(OUTPUT_DIR / "cross_dataset_observed_metrics.csv", index=False)
    random_df.to_csv(OUTPUT_DIR / "cross_dataset_random_metrics.csv", index=False)
    comparison_df.to_csv(OUTPUT_DIR / "cross_dataset_metric_comparison.csv", index=False)
    dataset_summary_df.to_csv(OUTPUT_DIR / "cross_dataset_summary.csv", index=False)

    key_metrics = [
        "pca_pc1_pc2_variance",
        "silhouette_pca10_space",
        "nearest_neighbor_same_label_at_3",
        "ari_agglomerative_vs_label",
        "nmi_agglomerative_vs_label",
    ]

    key_summary = comparison_df[comparison_df["metric"].isin(key_metrics)].copy()
    key_summary.to_csv(OUTPUT_DIR / "cross_dataset_key_metric_summary.csv", index=False)

    for metric in key_metrics:
        plot_cross_dataset_heatmap(
            summary=comparison_df,
            metric=metric,
            output_path=OUTPUT_DIR / f"heatmap_{metric}.png",
        )

    print("\n[DONE]")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print("\nMain files to inspect:")
    print(OUTPUT_DIR / "cross_dataset_summary.csv")
    print(OUTPUT_DIR / "cross_dataset_key_metric_summary.csv")
    print(OUTPUT_DIR / "cross_dataset_metric_comparison.csv")


if __name__ == "__main__":
    main()