from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

MATRISOME_FILE = RAW_DIR / "matrisome" / "human_matrisome.xlsx"


DATASET_CONFIGS: Dict[str, Dict[str, str]] = {
    "rna_tissue_consensus": {
        "path": str(RAW_DIR / "hpa" / "rna_tissue_consensus.tsv.zip"),
        "sample_col": "Tissue",
        "expr_col": "nTPM",
    },
    "rna_tissue_hpa": {
        "path": str(RAW_DIR / "hpa" / "rna_tissue_hpa.tsv.zip"),
        "sample_col": "Tissue",
        "expr_col": "nTPM",
    },
    "rna_tissue_gtex": {
        "path": str(RAW_DIR / "hpa" / "rna_tissue_gtex.tsv.zip"),
        "sample_col": "Tissue",
        "expr_col": "nTPM",
    },
    "rna_tissue_detail_gtex": {
        "path": str(RAW_DIR / "hpa" / "rna_tissue_detail_gtex.tsv.zip"),
        "sample_col": "Source tissue",
        "expr_col": "nTPM",
    },
    "rna_brain_hpa": {
        "path": str(RAW_DIR / "hpa" / "rna_brain_hpa.tsv.zip"),
        "sample_col": "Subregion",
        "expr_col": "nTPM",
    },
    "rna_pfc_brain_hpa": {
        "path": str(RAW_DIR / "hpa" / "rna_pfc_brain_hpa.tsv.zip"),
        "sample_col": "Subregion",
        "expr_col": "nTPM",
    },
    "rna_single_cell_type": {
        "path": str(RAW_DIR / "hpa" / "rna_single_cell_type.tsv.zip"),
        "sample_col": "Cell type",
        "expr_col": "nCPM",
    },
}


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def load_matrisome(path: Path) -> pd.DataFrame:
    """
    The downloaded Matrisome Excel file has:
    - Row 0: title, e.g., 'Homo sapiens matrisome masterlist'
    - Row 1: real column names, e.g., 'Matrisome Division', 'Gene Symbol', etc.

    Therefore we use header=1.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Matrisome file not found: {path}\n"
            "Place the Homo sapiens matrisome Excel file at:\n"
            "data/raw/matrisome/human_matrisome.xlsx"
        )

    df = pd.read_excel(path, header=1)
    df = clean_column_names(df)

    # Remove fully empty rows.
    df = df.dropna(how="all")

    print("\n[Matrisome]")
    print(f"Shape after header correction: {df.shape}")
    print("Columns:")
    print(df.columns.tolist())
    print(df.head())

    return df


def find_matrisome_gene_symbol_column(df: pd.DataFrame) -> str:
    exact_candidates = [
        "Gene Symbol",
        "Gene symbol",
        "gene_symbol",
        "HGNC Symbol",
        "HGNC symbol",
        "Symbol",
        "symbol",
        "Gene",
        "gene",
    ]

    for col in exact_candidates:
        if col in df.columns:
            return col

    # Flexible fallback.
    lower_map = {col: col.lower() for col in df.columns}
    for col, lower_col in lower_map.items():
        if "symbol" in lower_col and "gene" in lower_col:
            return col

    raise ValueError(
        "Could not identify the gene symbol column in Matrisome file.\n"
        f"Available columns: {df.columns.tolist()}"
    )


def normalize_gene_symbol(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .replace({"NAN": np.nan, "NONE": np.nan, "": np.nan})
    )


def load_hpa_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"HPA file not found: {path}")

    df = pd.read_csv(path, sep="\t", compression="zip")
    df = clean_column_names(df)

    return df


def validate_hpa_columns(
    df: pd.DataFrame,
    sample_col: str,
    expr_col: str,
) -> None:
    required_cols = ["Gene", "Gene name", sample_col, expr_col]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )


def build_ecm_matrix(
    hpa_df: pd.DataFrame,
    matrisome_df: pd.DataFrame,
    sample_col: str,
    expr_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validate_hpa_columns(hpa_df, sample_col=sample_col, expr_col=expr_col)

    matrisome_gene_col = find_matrisome_gene_symbol_column(matrisome_df)

    matrisome_df = matrisome_df.copy()
    hpa_df = hpa_df.copy()

    matrisome_df["_gene_symbol_upper"] = normalize_gene_symbol(
        matrisome_df[matrisome_gene_col]
    )
    hpa_df["_gene_symbol_upper"] = normalize_gene_symbol(hpa_df["Gene name"])

    matrisome_genes = set(
        matrisome_df["_gene_symbol_upper"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    hpa_genes = set(
        hpa_df["_gene_symbol_upper"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    matched_genes = sorted(matrisome_genes.intersection(hpa_genes))
    missing_genes = sorted(matrisome_genes.difference(hpa_genes))

    ecm_df = hpa_df[hpa_df["_gene_symbol_upper"].isin(matched_genes)].copy()

    # Convert expression values safely.
    ecm_df[expr_col] = pd.to_numeric(ecm_df[expr_col], errors="coerce")
    ecm_df = ecm_df.dropna(subset=[expr_col])

    matrix = ecm_df.pivot_table(
        index=sample_col,
        columns="_gene_symbol_upper",
        values=expr_col,
        aggfunc="mean",
    )

    matrix = matrix.sort_index()
    matrix = matrix.reindex(sorted(matrix.columns), axis=1)
    matrix = matrix.fillna(0.0)

    matrix_log2 = np.log2(matrix + 1.0)

    matched_gene_metadata = (
        matrisome_df[matrisome_df["_gene_symbol_upper"].isin(matched_genes)]
        .drop_duplicates(subset=["_gene_symbol_upper"])
        .copy()
    )

    missing_gene_metadata = pd.DataFrame({"missing_matrisome_gene": missing_genes})

    return matrix, matrix_log2, matched_gene_metadata, missing_gene_metadata


def zscore_columns(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score each gene across samples.
    Constant genes are set to 0.
    """
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0, ddof=0)

    z = (matrix - means) / stds.replace(0, np.nan)
    z = z.fillna(0.0)

    return z


def save_dataset_outputs(
    dataset_name: str,
    matrix_raw: pd.DataFrame,
    matrix_log2: pd.DataFrame,
    matched_gene_metadata: pd.DataFrame,
    missing_gene_metadata: pd.DataFrame,
) -> None:
    output_dir = PROCESSED_DIR / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_zscore = zscore_columns(matrix_log2)

    matrix_raw.to_csv(output_dir / "ecm_expression_raw.csv")
    matrix_log2.to_csv(output_dir / "ecm_expression_log2.csv")
    matrix_zscore.to_csv(output_dir / "ecm_expression_log2_zscore.csv")

    matched_gene_metadata.to_csv(output_dir / "matched_matrisome_gene_metadata.csv", index=False)
    missing_gene_metadata.to_csv(output_dir / "missing_matrisome_genes.csv", index=False)

    summary = {
        "dataset": dataset_name,
        "n_samples": matrix_log2.shape[0],
        "n_matched_ecm_genes": matrix_log2.shape[1],
        "n_missing_matrisome_genes": missing_gene_metadata.shape[0],
        "raw_matrix_path": str(output_dir / "ecm_expression_raw.csv"),
        "log2_matrix_path": str(output_dir / "ecm_expression_log2.csv"),
        "zscore_matrix_path": str(output_dir / "ecm_expression_log2_zscore.csv"),
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    print("\n" + "=" * 80)
    print(f"[SAVED] {dataset_name}")
    print(f"Samples: {matrix_log2.shape[0]}")
    print(f"Matched ECM genes: {matrix_log2.shape[1]}")
    print(f"Missing Matrisome genes: {missing_gene_metadata.shape[0]}")
    print(f"Output directory: {output_dir}")


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    matrisome_df = load_matrisome(MATRISOME_FILE)

    global_summaries = []

    for dataset_name, config in DATASET_CONFIGS.items():
        print("\n" + "#" * 100)
        print(f"Processing dataset: {dataset_name}")

        path = Path(config["path"])
        sample_col = config["sample_col"]
        expr_col = config["expr_col"]

        if not path.exists():
            print(f"[SKIP] Missing file: {path}")
            continue

        hpa_df = load_hpa_dataset(path)

        print(f"HPA shape: {hpa_df.shape}")
        print(f"HPA columns: {hpa_df.columns.tolist()}")

        matrix_raw, matrix_log2, matched_gene_metadata, missing_gene_metadata = build_ecm_matrix(
            hpa_df=hpa_df,
            matrisome_df=matrisome_df,
            sample_col=sample_col,
            expr_col=expr_col,
        )

        save_dataset_outputs(
            dataset_name=dataset_name,
            matrix_raw=matrix_raw,
            matrix_log2=matrix_log2,
            matched_gene_metadata=matched_gene_metadata,
            missing_gene_metadata=missing_gene_metadata,
        )

        global_summaries.append(
            {
                "dataset": dataset_name,
                "n_samples": matrix_log2.shape[0],
                "n_matched_ecm_genes": matrix_log2.shape[1],
                "n_missing_matrisome_genes": missing_gene_metadata.shape[0],
            }
        )

    if global_summaries:
        summary_df = pd.DataFrame(global_summaries)
        summary_df.to_csv(PROCESSED_DIR / "all_dataset_summary.csv", index=False)

        print("\n" + "=" * 100)
        print("[GLOBAL SUMMARY]")
        print(summary_df)

    print("\nDone.")


if __name__ == "__main__":
    main()