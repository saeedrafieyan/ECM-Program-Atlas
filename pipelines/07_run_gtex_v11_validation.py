from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


RAW_DIR = Path("data/raw/gtex_v11_sample_level")
PROCESSED_DIR = Path("data/processed/gtex_v11_sample_level")

MATRISOME_FILE = Path("data/raw/matrisome/human_matrisome.xlsx")

CURATED_PROGRAM_FILE = (
    Path("outputs")
    / "latent_baseline_embeddings"
    / "rna_tissue_consensus"
    / "curated_recurring_ecm_programs"
    / "combined_nmf_module_annotations_curated_programs.csv"
)


PROGRAM_ORDER = [
    "Vascular/stromal/interstitial ECM",
    "Epithelial/mucosal basement membrane ECM",
    "CNS/neural ECM",
    "Retinal/sensory ECM",
    "Immune/lymphoid remodeling ECM",
    "Stromal remodeling ECM",
    "Renal/endothelial basement membrane ECM",
    "Hepatic/plasma-associated ECM",
    "Reproductive-specialized ECM",
]


def find_file(patterns: list[str]) -> Path:
    for pattern in patterns:
        matches = sorted(RAW_DIR.glob(pattern))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"Could not find file matching any of these patterns in {RAW_DIR}:\n"
        + "\n".join(patterns)
    )


def normalize_gene_symbol(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .replace({"": np.nan, "NAN": np.nan, "NONE": np.nan})
    )


def split_comma_list(value: str) -> List[str]:
    if pd.isna(value):
        return []

    return [
        item.strip().upper()
        for item in str(value).split(",")
        if item.strip()
    ]


def extract_subject_id(sample_id: str) -> str:
    """
    GTEx sample ID example:
        GTEX-1117F-0226-SM-5GZZ7

    Subject ID:
        GTEX-1117F
    """
    parts = str(sample_id).split("-")
    if len(parts) >= 2 and parts[0] == "GTEX":
        return "-".join(parts[:2])
    return ""


def load_matrisome_genes(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Matrisome file not found: {path}")

    df = pd.read_excel(path, header=1)
    df.columns = [str(col).strip() for col in df.columns]

    if "Gene Symbol" not in df.columns:
        raise ValueError(
            f"Could not find 'Gene Symbol' in Matrisome file. Columns: {df.columns.tolist()}"
        )

    genes = (
        normalize_gene_symbol(df["Gene Symbol"])
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    return sorted(genes)


def load_curated_program_gene_sets(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Curated ECM program file not found: {path}\n"
            "Run the curated program pipeline first."
        )

    df = pd.read_csv(path)

    required = [
        "feature_set",
        "component",
        "ecm_program_curated",
        "top_genes",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Curated program file is missing columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    records = []

    for program, group in df.groupby("ecm_program_curated"):
        genes: list[str] = []
        modules: list[str] = []

        for row in group.itertuples():
            genes.extend(split_comma_list(row.top_genes))
            modules.append(f"{row.feature_set}:{row.component}")

        unique_genes = sorted(set(genes))

        records.append(
            {
                "ecm_program": program,
                "n_reference_modules": len(modules),
                "reference_modules": "; ".join(modules),
                "n_program_genes": len(unique_genes),
                "program_genes": ", ".join(unique_genes),
            }
        )

    program_df = pd.DataFrame(records)

    program_df["ecm_program"] = pd.Categorical(
        program_df["ecm_program"],
        categories=PROGRAM_ORDER,
        ordered=True,
    )

    program_df = program_df.sort_values("ecm_program")

    return program_df


def load_gtex_expression_parquet(path: Path) -> pd.DataFrame:
    print(f"[INFO] Loading GTEx expression parquet: {path}")
    print(f"[INFO] File size: {path.stat().st_size / (1024 ** 3):.2f} GB")

    df = pd.read_parquet(path, engine="pyarrow")

    # In your inspection output, Name is the pandas index and Description is a column.
    if "Name" not in df.columns:
        df = df.copy()
        df.insert(0, "Name", df.index.astype(str))

    if "Description" not in df.columns:
        raise ValueError(
            f"Expression matrix must contain a 'Description' column. Columns preview: {df.columns[:20].tolist()}"
        )

    print(f"[INFO] Loaded expression matrix shape: {df.shape}")
    return df


def filter_to_matrisome_expression(
    expr_df: pd.DataFrame,
    matrisome_genes: list[str],
) -> pd.DataFrame:
    expr_df = expr_df.copy()
    expr_df["gene_symbol"] = normalize_gene_symbol(expr_df["Description"])

    matrisome_set = set(matrisome_genes)

    sample_cols = [
        col for col in expr_df.columns
        if str(col).startswith("GTEX-")
    ]

    print(f"[INFO] Number of GTEx sample columns detected: {len(sample_cols)}")

    filtered = expr_df[expr_df["gene_symbol"].isin(matrisome_set)].copy()

    print(f"[INFO] Matrisome-filtered rows before gene-symbol aggregation: {filtered.shape[0]}")
    print(f"[INFO] Unique matched Matrisome genes: {filtered['gene_symbol'].nunique()}")

    if filtered.empty:
        raise ValueError("No Matrisome genes matched the GTEx expression matrix.")

    # Keep only gene symbol + sample columns.
    filtered = filtered[["gene_symbol"] + sample_cols]

    # Aggregate duplicate gene symbols if they exist.
    expression_by_gene = filtered.groupby("gene_symbol", as_index=True)[sample_cols].mean()

    # Convert genes × samples to samples × genes.
    expression_by_sample = expression_by_gene.T
    expression_by_sample.index.name = "sample_id"

    # TPM values should be non-negative.
    expression_by_sample = expression_by_sample.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # log2(TPM + 1)
    expression_log2 = np.log2(expression_by_sample + 1.0)

    print(f"[INFO] Matrisome sample-level matrix shape: {expression_log2.shape}")

    return expression_log2


def load_sample_metadata(path: Path) -> pd.DataFrame:
    print(f"[INFO] Loading sample metadata: {path}")
    df = pd.read_csv(path, sep="\t")

    required = ["SAMPID", "SMTS", "SMTSD"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Sample attributes file is missing columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    meta = df.copy()
    meta["sample_id"] = meta["SAMPID"].astype(str)
    meta["subject_id"] = meta["sample_id"].apply(extract_subject_id)
    meta["tissue"] = meta["SMTS"].astype(str)
    meta["tissue_detail"] = meta["SMTSD"].astype(str)

    keep_cols = [
        "sample_id",
        "subject_id",
        "tissue",
        "tissue_detail",
        "SAMPID",
        "SMTS",
        "SMTSD",
    ]

    optional_cols = [
        "SMRIN",
        "SMTSISCH",
        "SMTSPAX",
        "SMNABTCH",
        "SMGEBTCH",
        "ANALYTE_TYPE",
        "SMGTC",
        "SMVQCFL",
    ]

    keep_cols.extend([col for col in optional_cols if col in meta.columns])

    meta = meta[keep_cols].drop_duplicates(subset=["sample_id"])

    print(f"[INFO] Sample metadata shape: {meta.shape}")
    print(f"[INFO] Unique tissues: {meta['tissue'].nunique()}")
    print(f"[INFO] Unique tissue details: {meta['tissue_detail'].nunique()}")

    return meta


def load_subject_metadata(path: Path) -> pd.DataFrame:
    print(f"[INFO] Loading subject phenotypes: {path}")
    df = pd.read_csv(path, sep="\t")

    required = ["SUBJID", "SEX", "AGE", "DTHHRDY"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Subject phenotype file is missing columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    meta = df.copy()
    meta["subject_id"] = meta["SUBJID"].astype(str)
    meta["sex"] = meta["SEX"]
    meta["age"] = meta["AGE"].astype(str)
    meta["death_hardy_score"] = meta["DTHHRDY"]

    keep_cols = [
        "subject_id",
        "sex",
        "age",
        "death_hardy_score",
        "SUBJID",
        "SEX",
        "AGE",
        "DTHHRDY",
    ]

    meta = meta[keep_cols].drop_duplicates(subset=["subject_id"])

    print(f"[INFO] Subject metadata shape: {meta.shape}")

    return meta


def build_sample_metadata(
    expression_log2: pd.DataFrame,
    sample_meta: pd.DataFrame,
    subject_meta: pd.DataFrame,
) -> pd.DataFrame:
    expression_samples = pd.DataFrame({"sample_id": expression_log2.index.astype(str)})

    metadata = expression_samples.merge(
        sample_meta,
        on="sample_id",
        how="left",
    )

    metadata["subject_id"] = metadata["subject_id"].fillna(
        metadata["sample_id"].apply(extract_subject_id)
    )

    metadata = metadata.merge(
        subject_meta,
        on="subject_id",
        how="left",
    )

    missing_tissue = metadata["tissue"].isna().sum()
    missing_subject = metadata["sex"].isna().sum()

    print(f"[INFO] Final sample metadata shape: {metadata.shape}")
    print(f"[INFO] Samples missing tissue metadata: {missing_tissue}")
    print(f"[INFO] Samples missing subject metadata: {missing_subject}")

    return metadata


def zscore_genes(expression_log2: pd.DataFrame) -> pd.DataFrame:
    means = expression_log2.mean(axis=0)
    stds = expression_log2.std(axis=0, ddof=0).replace(0, np.nan)

    z = expression_log2.sub(means, axis=1).div(stds, axis=1)
    z = z.fillna(0.0)

    return z


def compute_program_scores(
    expression_log2: pd.DataFrame,
    program_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    available_genes = set(expression_log2.columns)

    expression_z = zscore_genes(expression_log2)

    log_score_records = []
    z_score_records = []
    availability_records = []

    log_scores = pd.DataFrame(index=expression_log2.index)
    z_scores = pd.DataFrame(index=expression_log2.index)

    for row in program_df.itertuples():
        program = str(row.ecm_program)
        genes = split_comma_list(row.program_genes)

        available = sorted(set(genes).intersection(available_genes))
        missing = sorted(set(genes).difference(available_genes))

        availability_records.append(
            {
                "ecm_program": program,
                "n_program_genes": len(set(genes)),
                "n_available_genes": len(available),
                "n_missing_genes": len(missing),
                "availability_fraction": len(available) / len(set(genes)) if genes else np.nan,
                "available_genes": ", ".join(available),
                "missing_genes": ", ".join(missing),
            }
        )

        if not available:
            print(f"[WARNING] No available genes for program: {program}")
            log_scores[program] = np.nan
            z_scores[program] = np.nan
            continue

        log_scores[program] = expression_log2[available].mean(axis=1)
        z_scores[program] = expression_z[available].mean(axis=1)

    availability_df = pd.DataFrame(availability_records)

    log_scores.index.name = "sample_id"
    z_scores.index.name = "sample_id"

    return log_scores, z_scores, availability_df


def summarize_by_group(
    scores_with_meta: pd.DataFrame,
    score_columns: list[str],
    group_cols: list[str],
    score_type: str,
) -> pd.DataFrame:
    records = []

    grouped = scores_with_meta.groupby(group_cols, dropna=False)

    for group_values, group_df in grouped:
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        group_info = dict(zip(group_cols, group_values))

        n_samples = group_df["sample_id"].nunique()
        n_subjects = group_df["subject_id"].nunique()

        for program in score_columns:
            values = pd.to_numeric(group_df[program], errors="coerce").dropna()

            if values.empty:
                continue

            record = {
                **group_info,
                "score_type": score_type,
                "ecm_program": program,
                "n_samples": n_samples,
                "n_subjects": n_subjects,
                "mean_score": values.mean(),
                "median_score": values.median(),
                "std_score": values.std(ddof=0),
                "q25_score": values.quantile(0.25),
                "q75_score": values.quantile(0.75),
            }

            records.append(record)

    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-matrisome-csv",
        action="store_true",
        help="Also save the sample-level Matrisome expression matrix as CSV.gz. Parquet is always saved.",
    )
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    gene_tpm_file = find_file([
        "gtex_v11_gene_tpm.parquet",
        "*gene_tpm.parquet",
        "*RNASeQCv2.4.3_gene_tpm.parquet",
    ])

    sample_attr_file = find_file([
        "gtex_v11_sample_attributes.txt",
        "*SampleAttributesDS.txt",
    ])

    subject_pheno_file = find_file([
        "gtex_v11_subject_phenotypes.txt",
        "*SubjectPhenotypesDS.txt",
    ])

    print("[FILES]")
    print(f"Gene TPM: {gene_tpm_file}")
    print(f"Sample attributes: {sample_attr_file}")
    print(f"Subject phenotypes: {subject_pheno_file}")

    matrisome_genes = load_matrisome_genes(MATRISOME_FILE)
    program_df = load_curated_program_gene_sets(CURATED_PROGRAM_FILE)

    program_df.to_csv(
        PROCESSED_DIR / "gtex_v11_reference_ecm_program_gene_sets.csv",
        index=False,
    )

    expr_df = load_gtex_expression_parquet(gene_tpm_file)

    matrisome_expr_log2 = filter_to_matrisome_expression(
        expr_df=expr_df,
        matrisome_genes=matrisome_genes,
    )

    # Free memory from full expression dataframe.
    del expr_df

    matrisome_expr_log2.to_parquet(
        PROCESSED_DIR / "gtex_v11_matrisome_expression_log2.parquet",
        engine="pyarrow",
    )

    if args.save_matrisome_csv:
        matrisome_expr_log2.to_csv(
            PROCESSED_DIR / "gtex_v11_matrisome_expression_log2.csv.gz",
            compression="gzip",
        )

    sample_meta = load_sample_metadata(sample_attr_file)
    subject_meta = load_subject_metadata(subject_pheno_file)

    metadata = build_sample_metadata(
        expression_log2=matrisome_expr_log2,
        sample_meta=sample_meta,
        subject_meta=subject_meta,
    )

    metadata.to_csv(
        PROCESSED_DIR / "gtex_v11_sample_metadata.csv",
        index=False,
    )

    log_scores, z_scores, availability_df = compute_program_scores(
        expression_log2=matrisome_expr_log2,
        program_df=program_df,
    )

    availability_df.to_csv(
        PROCESSED_DIR / "gtex_v11_program_gene_availability.csv",
        index=False,
    )

    log_scores.to_csv(
        PROCESSED_DIR / "gtex_v11_program_scores_log2_mean.csv",
    )

    z_scores.to_csv(
        PROCESSED_DIR / "gtex_v11_program_scores_zscore_mean.csv",
    )

    log_scores_with_meta = metadata.merge(
        log_scores.reset_index(),
        on="sample_id",
        how="left",
    )

    z_scores_with_meta = metadata.merge(
        z_scores.reset_index(),
        on="sample_id",
        how="left",
    )

    log_scores_with_meta.to_csv(
        PROCESSED_DIR / "gtex_v11_program_scores_log2_mean_with_metadata.csv",
        index=False,
    )

    z_scores_with_meta.to_csv(
        PROCESSED_DIR / "gtex_v11_program_scores_zscore_mean_with_metadata.csv",
        index=False,
    )

    score_cols = list(log_scores.columns)

    tissue_log_summary = summarize_by_group(
        scores_with_meta=log_scores_with_meta,
        score_columns=score_cols,
        group_cols=["tissue"],
        score_type="log2_mean",
    )

    tissue_z_summary = summarize_by_group(
        scores_with_meta=z_scores_with_meta,
        score_columns=score_cols,
        group_cols=["tissue"],
        score_type="zscore_mean",
    )

    tissue_detail_log_summary = summarize_by_group(
        scores_with_meta=log_scores_with_meta,
        score_columns=score_cols,
        group_cols=["tissue", "tissue_detail"],
        score_type="log2_mean",
    )

    tissue_detail_z_summary = summarize_by_group(
        scores_with_meta=z_scores_with_meta,
        score_columns=score_cols,
        group_cols=["tissue", "tissue_detail"],
        score_type="zscore_mean",
    )

    tissue_summary = pd.concat(
        [tissue_log_summary, tissue_z_summary],
        ignore_index=True,
    )

    tissue_detail_summary = pd.concat(
        [tissue_detail_log_summary, tissue_detail_z_summary],
        ignore_index=True,
    )

    tissue_summary.to_csv(
        PROCESSED_DIR / "gtex_v11_tissue_program_summary.csv",
        index=False,
    )

    tissue_detail_summary.to_csv(
        PROCESSED_DIR / "gtex_v11_tissue_detail_program_summary.csv",
        index=False,
    )

    global_summary = pd.DataFrame(
        [
            {
                "n_expression_samples": matrisome_expr_log2.shape[0],
                "n_matched_matrisome_genes": matrisome_expr_log2.shape[1],
                "n_programs": len(score_cols),
                "n_samples_with_tissue_metadata": metadata["tissue"].notna().sum(),
                "n_samples_missing_tissue_metadata": metadata["tissue"].isna().sum(),
                "n_subjects": metadata["subject_id"].nunique(),
                "n_tissues": metadata["tissue"].nunique(),
                "n_tissue_details": metadata["tissue_detail"].nunique(),
            }
        ]
    )

    global_summary.to_csv(
        PROCESSED_DIR / "gtex_v11_global_summary.csv",
        index=False,
    )

    print("\n[GLOBAL SUMMARY]")
    print(global_summary)

    print("\n[SAVED]")
    print(PROCESSED_DIR / "gtex_v11_matrisome_expression_log2.parquet")
    print(PROCESSED_DIR / "gtex_v11_sample_metadata.csv")
    print(PROCESSED_DIR / "gtex_v11_program_gene_availability.csv")
    print(PROCESSED_DIR / "gtex_v11_program_scores_log2_mean.csv")
    print(PROCESSED_DIR / "gtex_v11_program_scores_zscore_mean.csv")
    print(PROCESSED_DIR / "gtex_v11_program_scores_log2_mean_with_metadata.csv")
    print(PROCESSED_DIR / "gtex_v11_program_scores_zscore_mean_with_metadata.csv")
    print(PROCESSED_DIR / "gtex_v11_tissue_program_summary.csv")
    print(PROCESSED_DIR / "gtex_v11_tissue_detail_program_summary.csv")
    print(PROCESSED_DIR / "gtex_v11_global_summary.csv")


if __name__ == "__main__":
    main()