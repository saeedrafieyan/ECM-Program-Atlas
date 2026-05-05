from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
import plotly.express as px


BASE_DIR = Path("outputs/latent_baseline_embeddings")

REFERENCE_ANNOTATIONS = (
    BASE_DIR
    / "rna_tissue_consensus"
    / "curated_recurring_ecm_programs"
    / "combined_nmf_module_annotations_curated_programs.csv"
)

OUTPUT_DIR = BASE_DIR / "cross_dataset_nmf_program_reproducibility"

FEATURE_SETS = [
    "core_matrisome",
    "ecm_glycoproteins",
    "proteoglycans",
    "collagens",
]

DATASETS = [
    "rna_tissue_consensus",
    "rna_tissue_hpa",
    "rna_tissue_gtex",
    "rna_tissue_detail_gtex",
    "rna_brain_hpa",
    "rna_single_cell_type",
]


def split_gene_string(gene_string: str) -> List[str]:
    if pd.isna(gene_string):
        return []

    genes = [
        gene.strip().upper()
        for gene in str(gene_string).split(",")
        if gene.strip()
    ]

    return genes


def safe_name(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def load_reference_annotations(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Reference annotation file not found:\n{path}\n\n"
            "Run src/create_curated_ecm_program_summary.py first."
        )

    df = pd.read_csv(path)

    required = [
        "dataset",
        "feature_set",
        "component",
        "module_name",
        "ecm_program_curated",
        "confidence",
        "short_interpretation",
        "top_samples",
        "top_genes",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Reference annotation file is missing columns: {missing}"
        )

    df["reference_genes"] = df["top_genes"].apply(split_gene_string)

    return df


def load_nmf_top_genes(path: Path, top_n_genes: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"NMF top genes file not found: {path}")

    df = pd.read_csv(path)

    required = ["component", "rank", "gene", "weight"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"NMF top genes file is missing columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    df["gene"] = df["gene"].astype(str).str.upper()
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df = df.dropna(subset=["rank"])

    df = df[df["rank"] <= top_n_genes].copy()

    return df


def load_nmf_sample_coordinates(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None

    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)

    return df


def get_top_samples(
    sample_df: pd.DataFrame | None,
    component: str,
    top_n_samples: int,
) -> str:
    if sample_df is None:
        return ""

    if component not in sample_df.columns:
        return ""

    top_samples = (
        sample_df[component]
        .sort_values(ascending=False)
        .head(top_n_samples)
        .index.astype(str)
        .tolist()
    )

    return "; ".join(top_samples)


def build_component_gene_sets(nmf_top_genes_df: pd.DataFrame) -> Dict[str, Set[str]]:
    component_gene_sets: Dict[str, Set[str]] = {}

    for component, group in nmf_top_genes_df.groupby("component"):
        genes = set(group["gene"].astype(str).str.upper().tolist())
        component_gene_sets[str(component)] = genes

    return component_gene_sets


def jaccard_similarity(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0

    intersection = len(a.intersection(b))
    union = len(a.union(b))

    if union == 0:
        return 0.0

    return intersection / union


def overlap_coefficient(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0

    denominator = min(len(a), len(b))

    if denominator == 0:
        return 0.0

    return len(a.intersection(b)) / denominator


def compare_target_modules_to_reference(
    dataset: str,
    feature_set: str,
    target_component_genes: Dict[str, Set[str]],
    target_sample_df: pd.DataFrame | None,
    reference_df: pd.DataFrame,
    top_n_samples: int,
) -> pd.DataFrame:
    reference_subset = reference_df[
        reference_df["feature_set"].astype(str).eq(feature_set)
    ].copy()

    records = []

    for target_component, target_genes in target_component_genes.items():
        for ref_row in reference_subset.itertuples():
            reference_component = str(ref_row.component)
            reference_genes = set(ref_row.reference_genes)

            intersection = sorted(target_genes.intersection(reference_genes))

            records.append(
                {
                    "dataset": dataset,
                    "feature_set": feature_set,
                    "target_component": target_component,
                    "target_top_samples": get_top_samples(
                        sample_df=target_sample_df,
                        component=target_component,
                        top_n_samples=top_n_samples,
                    ),
                    "reference_component": reference_component,
                    "reference_module_name": ref_row.module_name,
                    "reference_ecm_program": ref_row.ecm_program_curated,
                    "reference_confidence": ref_row.confidence,
                    "n_target_genes": len(target_genes),
                    "n_reference_genes": len(reference_genes),
                    "n_overlapping_genes": len(intersection),
                    "overlapping_genes": ", ".join(intersection),
                    "jaccard": jaccard_similarity(target_genes, reference_genes),
                    "overlap_coefficient": overlap_coefficient(
                        target_genes,
                        reference_genes,
                    ),
                }
            )

    return pd.DataFrame(records)


def select_best_matches(
    all_matches: pd.DataFrame,
    min_overlap_genes: int,
    min_overlap_coefficient: float,
) -> pd.DataFrame:
    if all_matches.empty:
        return all_matches

    sort_cols = [
        "dataset",
        "feature_set",
        "target_component",
        "overlap_coefficient",
        "jaccard",
        "n_overlapping_genes",
    ]

    best = (
        all_matches
        .sort_values(sort_cols, ascending=[True, True, True, False, False, False])
        .groupby(["dataset", "feature_set", "target_component"], as_index=False)
        .head(1)
        .copy()
    )

    best["is_reproduced_match"] = (
        (best["n_overlapping_genes"] >= min_overlap_genes)
        & (best["overlap_coefficient"] >= min_overlap_coefficient)
    )

    best["assigned_ecm_program"] = best.apply(
        lambda row: row["reference_ecm_program"]
        if row["is_reproduced_match"]
        else "Unassigned/weak match",
        axis=1,
    )

    return best


def make_program_reproducibility_summary(best_matches: pd.DataFrame) -> pd.DataFrame:
    reproduced = best_matches[best_matches["is_reproduced_match"]].copy()

    if reproduced.empty:
        return pd.DataFrame()

    records = []

    for program, group in reproduced.groupby("assigned_ecm_program"):
        datasets = sorted(group["dataset"].unique().tolist())
        feature_sets = sorted(group["feature_set"].unique().tolist())

        records.append(
            {
                "ecm_program": program,
                "n_reproduced_modules": group.shape[0],
                "n_datasets": len(datasets),
                "datasets": "; ".join(datasets),
                "n_feature_sets": len(feature_sets),
                "feature_sets": "; ".join(feature_sets),
                "mean_overlap_coefficient": group["overlap_coefficient"].mean(),
                "mean_jaccard": group["jaccard"].mean(),
                "mean_n_overlapping_genes": group["n_overlapping_genes"].mean(),
            }
        )

    summary = pd.DataFrame(records)

    summary = summary.sort_values(
        ["n_datasets", "n_feature_sets", "n_reproduced_modules"],
        ascending=False,
    )

    return summary


def make_presence_matrix(best_matches: pd.DataFrame) -> pd.DataFrame:
    reproduced = best_matches[best_matches["is_reproduced_match"]].copy()

    if reproduced.empty:
        return pd.DataFrame()

    presence = (
        reproduced.groupby(["assigned_ecm_program", "dataset"])
        .size()
        .reset_index(name="n_modules")
    )

    matrix = presence.pivot_table(
        index="assigned_ecm_program",
        columns="dataset",
        values="n_modules",
        fill_value=0,
        aggfunc="sum",
    )

    return matrix


def make_feature_set_presence_matrix(best_matches: pd.DataFrame) -> pd.DataFrame:
    reproduced = best_matches[best_matches["is_reproduced_match"]].copy()

    if reproduced.empty:
        return pd.DataFrame()

    presence = (
        reproduced.groupby(["assigned_ecm_program", "feature_set"])
        .size()
        .reset_index(name="n_modules")
    )

    matrix = presence.pivot_table(
        index="assigned_ecm_program",
        columns="feature_set",
        values="n_modules",
        fill_value=0,
        aggfunc="sum",
    )

    return matrix


def plot_heatmap(matrix: pd.DataFrame, title: str, output_path: Path) -> None:
    if matrix.empty:
        return

    fig = px.imshow(
        matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        title=title,
        labels={
            "x": "Dataset / feature set",
            "y": "ECM program",
            "color": "Number of reproduced modules",
        },
    )

    fig.update_layout(
        width=1150,
        height=max(600, 40 * matrix.shape[0]),
        template="plotly_white",
    )

    fig.write_html(str(output_path), include_plotlyjs="cdn", full_html=True)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--top-n-genes",
        type=int,
        default=30,
        help="Number of top genes per NMF component used for matching.",
    )

    parser.add_argument(
        "--top-n-samples",
        type=int,
        default=10,
        help="Number of top samples stored for each target component.",
    )

    parser.add_argument(
        "--min-overlap-genes",
        type=int,
        default=4,
        help="Minimum number of overlapping genes required for reproduced match.",
    )

    parser.add_argument(
        "--min-overlap-coefficient",
        type=float,
        default=0.13,
        help="Minimum overlap coefficient required for reproduced match.",
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    reference_df = load_reference_annotations(REFERENCE_ANNOTATIONS)

    all_match_records = []

    for dataset in DATASETS:
        for feature_set in FEATURE_SETS:
            embedding_dir = BASE_DIR / dataset / feature_set

            nmf_top_genes_path = embedding_dir / "nmf_top_genes.csv"
            nmf_sample_coordinates_path = embedding_dir / "nmf_sample_coordinates.csv"

            if not nmf_top_genes_path.exists():
                print(f"[SKIP] Missing: {nmf_top_genes_path}")
                continue

            print(f"[PROCESSING] {dataset} | {feature_set}")

            target_top_genes_df = load_nmf_top_genes(
                path=nmf_top_genes_path,
                top_n_genes=args.top_n_genes,
            )

            target_component_genes = build_component_gene_sets(target_top_genes_df)

            target_sample_df = load_nmf_sample_coordinates(nmf_sample_coordinates_path)

            matches = compare_target_modules_to_reference(
                dataset=dataset,
                feature_set=feature_set,
                target_component_genes=target_component_genes,
                target_sample_df=target_sample_df,
                reference_df=reference_df,
                top_n_samples=args.top_n_samples,
            )

            all_match_records.append(matches)

    if not all_match_records:
        raise FileNotFoundError("No NMF top-gene files were found.")

    all_matches = pd.concat(all_match_records, ignore_index=True)

    best_matches = select_best_matches(
        all_matches=all_matches,
        min_overlap_genes=args.min_overlap_genes,
        min_overlap_coefficient=args.min_overlap_coefficient,
    )

    program_summary = make_program_reproducibility_summary(best_matches)
    dataset_presence_matrix = make_presence_matrix(best_matches)
    feature_presence_matrix = make_feature_set_presence_matrix(best_matches)

    all_matches.to_csv(OUTPUT_DIR / "all_module_reference_matches.csv", index=False)
    best_matches.to_csv(OUTPUT_DIR / "best_module_reference_matches.csv", index=False)
    program_summary.to_csv(
        OUTPUT_DIR / "cross_dataset_program_reproducibility_summary.csv",
        index=False,
    )
    dataset_presence_matrix.to_csv(
        OUTPUT_DIR / "cross_dataset_program_presence_matrix.csv"
    )
    feature_presence_matrix.to_csv(
        OUTPUT_DIR / "cross_feature_set_program_presence_matrix.csv"
    )

    plot_heatmap(
        matrix=dataset_presence_matrix,
        title=(
            "Cross-dataset reproducibility of curated ECM programs<br>"
            "<sup>Best NMF module matches based on top-gene overlap with tissue-consensus reference modules</sup>"
        ),
        output_path=OUTPUT_DIR / "cross_dataset_program_presence_heatmap.html",
    )

    plot_heatmap(
        matrix=feature_presence_matrix,
        title=(
            "Cross-feature-set reproducibility of curated ECM programs<br>"
            "<sup>Best NMF module matches based on top-gene overlap with tissue-consensus reference modules</sup>"
        ),
        output_path=OUTPUT_DIR / "cross_feature_set_program_presence_heatmap.html",
    )

    print("\n[SAVED]")
    print(OUTPUT_DIR / "all_module_reference_matches.csv")
    print(OUTPUT_DIR / "best_module_reference_matches.csv")
    print(OUTPUT_DIR / "cross_dataset_program_reproducibility_summary.csv")
    print(OUTPUT_DIR / "cross_dataset_program_presence_matrix.csv")
    print(OUTPUT_DIR / "cross_feature_set_program_presence_matrix.csv")
    print(OUTPUT_DIR / "cross_dataset_program_presence_heatmap.html")
    print(OUTPUT_DIR / "cross_feature_set_program_presence_heatmap.html")

    print("\n[PROGRAM REPRODUCIBILITY SUMMARY]")
    print(program_summary)


if __name__ == "__main__":
    main()