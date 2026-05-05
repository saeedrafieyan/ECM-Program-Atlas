from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


RAW_PATH = Path("data/raw/matrisomedb/matrisomedb_human_export.tsv")
OUTPUT_DIR = Path("data/processed/matrisomedb")


REQUIRED_COLUMNS = [
    "Gene",
    "UniProt",
    "Sample type",
    "Tissue",
    "Species",
    "Repository",
    "Reference",
    "Confidence Score",
    "Description",
    "Coverage Map File",
    "NSAF",
]


NORMAL_KEYWORDS = [
    "normal",
    "healthy",
    "control",
    "non-tumor",
    "non tumor",
    "noncancer",
    "non-cancer",
    "distant mucosa",
    "normal distant",
    "normal adjacent",
    "adjacent normal",
]


DISEASE_KEYWORDS = [
    "carcinoma",
    "cancer",
    "tumor",
    "tumour",
    "metastasis",
    "adenocarcinoma",
    "fibrotic",
    "fibrosis",
    "keloid",
    "stenosis",
    "atherosclerosis",
    "aneurysm",
    "lesion",
    "disease",
    "pathological",
    "symptomatic",
    "asymptomatic carotid",
    "hgs ovarian carcinoma",
]


def normalize_text(value: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value


def normalize_gene_symbol(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .replace({"": np.nan, "NAN": np.nan, "NONE": np.nan})
    )


def classify_sample_type(sample_type: str) -> str:
    """
    Heuristic classification.

    This is intentionally conservative:
    - disease keywords override normal keywords
    - normal adjacent/distant samples are marked normal_like, but should be interpreted carefully
    """
    text = normalize_text(sample_type)

    if any(keyword in text for keyword in DISEASE_KEYWORDS):
        return "disease_like"

    if any(keyword in text for keyword in NORMAL_KEYWORDS):
        return "normal_like"

    return "uncertain"


def load_export(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing MatrisomeDB export: {path}")

    if path.suffix.lower() == ".tsv":
        df = pd.read_csv(path, sep="\t")
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"MatrisomeDB export is missing columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    return df


def clean_export(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["gene_symbol"] = normalize_gene_symbol(df["Gene"])
    df["uniprot_id"] = df["UniProt"].astype(str).str.strip()
    df["sample_type"] = df["Sample type"].astype(str).str.strip()
    df["tissue"] = df["Tissue"].astype(str).str.strip()
    df["species"] = df["Species"].astype(str).str.strip()
    df["repository"] = df["Repository"].astype(str).str.strip()
    df["reference"] = df["Reference"].astype(str).str.strip()
    df["description"] = df["Description"].astype(str).str.strip()

    df["confidence_score"] = pd.to_numeric(df["Confidence Score"], errors="coerce")
    df["nsaf"] = pd.to_numeric(df["NSAF"], errors="coerce")

    df = df.dropna(subset=["gene_symbol", "tissue", "nsaf"])

    df["log1p_nsaf"] = np.log1p(df["nsaf"])
    df["condition_group"] = df["sample_type"].apply(classify_sample_type)

    # A sample unit is a tissue + sample type + repository combination.
    # This helps avoid collapsing different studies too early.
    df["sample_id"] = (
        df["tissue"].astype(str)
        + " | "
        + df["sample_type"].astype(str)
        + " | "
        + df["repository"].astype(str)
    )

    useful_cols = [
        "sample_id",
        "gene_symbol",
        "uniprot_id",
        "tissue",
        "sample_type",
        "condition_group",
        "species",
        "repository",
        "reference",
        "description",
        "confidence_score",
        "nsaf",
        "log1p_nsaf",
    ]

    return df[useful_cols].copy()


def summarize_sample_types(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df[
            [
                "sample_id",
                "tissue",
                "sample_type",
                "condition_group",
                "repository",
                "reference",
            ]
        ]
        .drop_duplicates()
        .sort_values(["condition_group", "tissue", "sample_type"])
        .reset_index(drop=True)
    )

    return summary


def summarize_tissues(df: pd.DataFrame) -> pd.DataFrame:
    tissue_summary = (
        df.groupby(["condition_group", "tissue"])
        .agg(
            n_rows=("gene_symbol", "size"),
            n_genes=("gene_symbol", "nunique"),
            n_uniprot=("uniprot_id", "nunique"),
            n_sample_types=("sample_type", "nunique"),
            n_repositories=("repository", "nunique"),
            mean_nsaf=("nsaf", "mean"),
            median_nsaf=("nsaf", "median"),
            max_nsaf=("nsaf", "max"),
        )
        .reset_index()
        .sort_values(["condition_group", "n_genes"], ascending=[True, False])
    )

    return tissue_summary


def build_tissue_gene_matrices(df: pd.DataFrame, output_dir: Path) -> Dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)

    mean_log_nsaf = df.pivot_table(
        index="tissue",
        columns="gene_symbol",
        values="log1p_nsaf",
        aggfunc="mean",
        fill_value=0.0,
    )

    max_log_nsaf = df.pivot_table(
        index="tissue",
        columns="gene_symbol",
        values="log1p_nsaf",
        aggfunc="max",
        fill_value=0.0,
    )

    detection_count = df.pivot_table(
        index="tissue",
        columns="gene_symbol",
        values="sample_id",
        aggfunc="nunique",
        fill_value=0,
    )

    binary_detection = (detection_count > 0).astype(int)

    mean_log_nsaf = mean_log_nsaf.sort_index().reindex(sorted(mean_log_nsaf.columns), axis=1)
    max_log_nsaf = max_log_nsaf.sort_index().reindex(sorted(max_log_nsaf.columns), axis=1)
    detection_count = detection_count.sort_index().reindex(sorted(detection_count.columns), axis=1)
    binary_detection = binary_detection.sort_index().reindex(sorted(binary_detection.columns), axis=1)

    mean_log_nsaf.to_csv(output_dir / "tissue_gene_mean_log_nsaf.csv")
    max_log_nsaf.to_csv(output_dir / "tissue_gene_max_log_nsaf.csv")
    detection_count.to_csv(output_dir / "tissue_gene_detection_count.csv")
    binary_detection.to_csv(output_dir / "tissue_gene_binary_detection.csv")

    matrix_summary = pd.DataFrame(
        [
            {
                "matrix": "tissue_gene_mean_log_nsaf",
                "n_tissues": mean_log_nsaf.shape[0],
                "n_genes": mean_log_nsaf.shape[1],
            },
            {
                "matrix": "tissue_gene_max_log_nsaf",
                "n_tissues": max_log_nsaf.shape[0],
                "n_genes": max_log_nsaf.shape[1],
            },
            {
                "matrix": "tissue_gene_detection_count",
                "n_tissues": detection_count.shape[0],
                "n_genes": detection_count.shape[1],
            },
            {
                "matrix": "tissue_gene_binary_detection",
                "n_tissues": binary_detection.shape[0],
                "n_genes": binary_detection.shape[1],
            },
        ]
    )

    matrix_summary.to_csv(output_dir / "matrix_summary.csv", index=False)

    return {
        "mean_log_nsaf": mean_log_nsaf,
        "max_log_nsaf": max_log_nsaf,
        "detection_count": detection_count,
        "binary_detection": binary_detection,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = load_export(RAW_PATH)
    clean_df = clean_export(raw_df)

    clean_df.to_csv(OUTPUT_DIR / "matrisomedb_cleaned_long_table.csv", index=False)

    sample_summary = summarize_sample_types(clean_df)
    sample_summary.to_csv(
        OUTPUT_DIR / "matrisomedb_sample_type_annotation.csv",
        index=False,
    )

    tissue_summary = summarize_tissues(clean_df)
    tissue_summary.to_csv(
        OUTPUT_DIR / "matrisomedb_tissue_summary.csv",
        index=False,
    )

    print("[INFO] Cleaned MatrisomeDB table")
    print(f"Rows: {clean_df.shape[0]}")
    print(f"Genes: {clean_df['gene_symbol'].nunique()}")
    print(f"Tissues: {clean_df['tissue'].nunique()}")
    print(f"Sample IDs: {clean_df['sample_id'].nunique()}")
    print("\nCondition groups:")
    print(clean_df["condition_group"].value_counts())

    # Build all-sample matrix
    build_tissue_gene_matrices(
        df=clean_df,
        output_dir=OUTPUT_DIR / "all_samples",
    )

    # Build condition-specific matrices
    for condition_group, group_df in clean_df.groupby("condition_group"):
        print(f"\n[INFO] Building matrices for: {condition_group}")
        print(f"Rows: {group_df.shape[0]}")
        print(f"Genes: {group_df['gene_symbol'].nunique()}")
        print(f"Tissues: {group_df['tissue'].nunique()}")

        build_tissue_gene_matrices(
            df=group_df,
            output_dir=OUTPUT_DIR / condition_group,
        )

    global_summary = pd.DataFrame(
        [
            {
                "condition_group": "all_samples",
                "n_rows": clean_df.shape[0],
                "n_genes": clean_df["gene_symbol"].nunique(),
                "n_tissues": clean_df["tissue"].nunique(),
                "n_sample_ids": clean_df["sample_id"].nunique(),
                "n_repositories": clean_df["repository"].nunique(),
                "n_references": clean_df["reference"].nunique(),
            }
        ]
        + [
            {
                "condition_group": condition,
                "n_rows": group.shape[0],
                "n_genes": group["gene_symbol"].nunique(),
                "n_tissues": group["tissue"].nunique(),
                "n_sample_ids": group["sample_id"].nunique(),
                "n_repositories": group["repository"].nunique(),
                "n_references": group["reference"].nunique(),
            }
            for condition, group in clean_df.groupby("condition_group")
        ]
    )

    global_summary.to_csv(OUTPUT_DIR / "matrisomedb_global_summary.csv", index=False)

    print("\n[SAVED]")
    print(OUTPUT_DIR / "matrisomedb_cleaned_long_table.csv")
    print(OUTPUT_DIR / "matrisomedb_sample_type_annotation.csv")
    print(OUTPUT_DIR / "matrisomedb_tissue_summary.csv")
    print(OUTPUT_DIR / "matrisomedb_global_summary.csv")
    print(OUTPUT_DIR / "all_samples")
    print(OUTPUT_DIR / "normal_like")
    print(OUTPUT_DIR / "disease_like")
    print(OUTPUT_DIR / "uncertain")


if __name__ == "__main__":
    main()