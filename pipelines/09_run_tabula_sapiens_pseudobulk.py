from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse


RAW_DIR = Path("data/raw/tabula_sapiens")
PROCESSED_DIR = Path("data/processed/tabula_sapiens")

FULL_H5AD = RAW_DIR / "TabulaSapiens.h5ad"

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


def load_matrisome_genes(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Matrisome file: {path}")

    df = pd.read_excel(path, header=1)
    df.columns = [str(col).strip() for col in df.columns]

    if "Gene Symbol" not in df.columns:
        raise ValueError(
            f"Could not find Gene Symbol column in Matrisome file. "
            f"Columns: {df.columns.tolist()}"
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
            f"Missing curated ECM program file: {path}\n"
            "Run the curated ECM program pipeline first."
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
        genes: List[str] = []
        modules: List[str] = []

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


def get_gene_symbols(adata: ad.AnnData) -> pd.Series:
    if "gene_symbol" in adata.var.columns:
        symbols = adata.var["gene_symbol"]
    else:
        symbols = pd.Series(adata.var_names, index=adata.var_names)

    symbols = normalize_gene_symbol(symbols)
    return symbols


def choose_layer(adata: ad.AnnData, preferred_layer: str | None) -> str | None:
    available_layers = list(adata.layers.keys())

    if preferred_layer is not None:
        if preferred_layer in available_layers:
            return preferred_layer
        raise ValueError(
            f"Requested layer '{preferred_layer}' not found. "
            f"Available layers: {available_layers}"
        )

    for candidate in ["decontXcounts", "raw_counts"]:
        if candidate in available_layers:
            return candidate

    return None


def load_matrisome_subset(
    h5ad_path: Path,
    matrisome_genes: List[str],
    layer_name: str | None,
) -> Tuple[ad.AnnData, str | None]:
    if not h5ad_path.exists():
        raise FileNotFoundError(f"Missing Tabula Sapiens file: {h5ad_path}")

    print(f"[INFO] Opening {h5ad_path}")
    adata_backed = ad.read_h5ad(h5ad_path, backed="r")

    chosen_layer = choose_layer(adata_backed, preferred_layer=layer_name)

    print(f"[INFO] Shape: {adata_backed.n_obs} cells × {adata_backed.n_vars} genes")
    print(f"[INFO] Available layers: {list(adata_backed.layers.keys())}")
    print(f"[INFO] Chosen layer: {chosen_layer if chosen_layer else 'X'}")

    symbols = get_gene_symbols(adata_backed)
    matrisome_set = set(matrisome_genes)

    keep_mask = symbols.isin(matrisome_set).values
    n_keep = int(keep_mask.sum())

    print(f"[INFO] Matrisome genes matched in Tabula Sapiens: {n_keep}")

    if n_keep == 0:
        raise ValueError("No Matrisome genes matched Tabula Sapiens gene symbols.")

    # Load only Matrisome genes into memory.
    adata = adata_backed[:, keep_mask].to_memory()
    adata_backed.file.close()

    subset_symbols = get_gene_symbols(adata)
    adata.var["gene_symbol_clean"] = subset_symbols.values

    return adata, chosen_layer


def extract_expression_matrix(adata: ad.AnnData, layer_name: str | None):
    if layer_name is not None:
        X = adata.layers[layer_name]
    else:
        X = adata.X

    if sparse.issparse(X):
        return X.tocsr()

    return sparse.csr_matrix(X)


def build_obs_metadata(adata: ad.AnnData, method_filter: str | None) -> pd.DataFrame:
    obs = adata.obs.copy()

    required = [
        "organ_tissue",
        "method",
        "donor",
        "cell_ontology_class",
        "free_annotation",
        "compartment",
        "gender",
    ]

    missing = [col for col in required if col not in obs.columns]
    if missing:
        raise ValueError(
            f"Missing required obs columns: {missing}. "
            f"Available columns: {obs.columns.tolist()}"
        )

    obs = obs.reset_index().rename(columns={"index": "cell_id"})

    obs["cell_id"] = obs["cell_id"].astype(str)
    obs["organ"] = obs["organ_tissue"].astype(str)
    obs["method"] = obs["method"].astype(str)
    obs["donor"] = obs["donor"].astype(str)
    obs["cell_type"] = obs["cell_ontology_class"].astype(str)
    obs["free_annotation"] = obs["free_annotation"].astype(str)
    obs["compartment"] = obs["compartment"].astype(str)
    obs["gender"] = obs["gender"].astype(str)

    if "anatomical_information" in obs.columns:
        obs["anatomical_information"] = obs["anatomical_information"].astype(str)
    else:
        obs["anatomical_information"] = ""

    if method_filter is not None:
        before = obs.shape[0]
        obs = obs[obs["method"].eq(method_filter)].copy()
        after = obs.shape[0]
        print(f"[INFO] Method filter: {method_filter}. Cells before={before}, after={after}")

    return obs


def aggregate_duplicate_genes(
    matrix: sparse.csr_matrix,
    gene_symbols: List[str],
) -> Tuple[sparse.csr_matrix, List[str]]:
    """
    If multiple features map to the same gene symbol, sum them.
    """
    gene_symbols = [str(g).upper() for g in gene_symbols]

    unique_genes = sorted(set(gene_symbols))
    gene_to_col = {gene: i for i, gene in enumerate(unique_genes)}

    rows = []
    cols = []
    data = []

    for original_col, gene in enumerate(gene_symbols):
        rows.append(original_col)
        cols.append(gene_to_col[gene])
        data.append(1)

    mapping = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(len(gene_symbols), len(unique_genes)),
    )

    aggregated = matrix @ mapping

    return aggregated.tocsr(), unique_genes


def create_group_labels(
    obs: pd.DataFrame,
    min_cells_per_group: int,
) -> pd.DataFrame:
    group_cols = [
        "donor",
        "organ",
        "cell_type",
        "free_annotation",
        "compartment",
        "method",
        "gender",
        "anatomical_information",
    ]

    group_counts = (
        obs.groupby(group_cols, dropna=False)
        .size()
        .reset_index(name="n_cells")
    )

    group_counts["pseudobulk_id"] = [
        f"PB{i:07d}" for i in range(group_counts.shape[0])
    ]

    group_counts = group_counts[group_counts["n_cells"] >= min_cells_per_group].copy()

    obs_with_groups = obs.merge(
        group_counts[group_cols + ["pseudobulk_id", "n_cells"]],
        on=group_cols,
        how="left",
    )

    return obs_with_groups


def build_pseudobulk_counts(
    X: sparse.csr_matrix,
    obs_with_groups: pd.DataFrame,
    valid_cell_indices: np.ndarray,
    gene_names: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    obs_valid = obs_with_groups.iloc[valid_cell_indices].copy()
    obs_valid = obs_valid.dropna(subset=["pseudobulk_id"]).copy()

    if obs_valid.empty:
        raise ValueError("No valid pseudobulk groups after filtering.")

    # Map cells to pseudobulk groups.
    unique_groups = obs_valid["pseudobulk_id"].drop_duplicates().tolist()
    group_to_idx = {group: i for i, group in enumerate(unique_groups)}

    cell_positions = obs_valid.index.to_numpy()
    group_positions = obs_valid["pseudobulk_id"].map(group_to_idx).to_numpy()

    data = np.ones(len(cell_positions), dtype=np.float32)

    cell_to_group = sparse.csr_matrix(
        (data, (cell_positions, group_positions)),
        shape=(obs_with_groups.shape[0], len(unique_groups)),
    )

    # group × genes = group-cell transpose × cell-gene
    pseudobulk = cell_to_group.T @ X
    pseudobulk = pseudobulk.tocsr()

    pb_counts = pd.DataFrame.sparse.from_spmatrix(
        pseudobulk,
        index=unique_groups,
        columns=gene_names,
    )

    metadata_cols = [
        "pseudobulk_id",
        "donor",
        "organ",
        "cell_type",
        "free_annotation",
        "compartment",
        "method",
        "gender",
        "anatomical_information",
        "n_cells",
    ]

    metadata = (
        obs_valid[metadata_cols]
        .drop_duplicates(subset=["pseudobulk_id"])
        .set_index("pseudobulk_id")
        .loc[unique_groups]
        .reset_index()
    )

    return pb_counts, metadata


def normalize_counts_to_log_cpm(counts: pd.DataFrame) -> pd.DataFrame:
    dense = counts.sparse.to_dense().astype(float)

    totals = dense.sum(axis=1).replace(0, np.nan)
    cpm = dense.div(totals, axis=0) * 1e6
    log_cpm = np.log1p(cpm)
    log_cpm = log_cpm.fillna(0.0)

    return log_cpm


def zscore_genes(matrix: pd.DataFrame) -> pd.DataFrame:
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0, ddof=0).replace(0, np.nan)

    z = matrix.sub(means, axis=1).div(stds, axis=1)
    z = z.fillna(0.0)

    return z


def compute_program_scores(
    log_cpm: pd.DataFrame,
    program_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    available_genes = set(log_cpm.columns)
    z_matrix = zscore_genes(log_cpm)

    log_scores = pd.DataFrame(index=log_cpm.index)
    z_scores = pd.DataFrame(index=log_cpm.index)
    availability_records = []

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
            log_scores[program] = np.nan
            z_scores[program] = np.nan
            continue

        log_scores[program] = log_cpm[available].mean(axis=1)
        z_scores[program] = z_matrix[available].mean(axis=1)

    log_scores.index.name = "pseudobulk_id"
    z_scores.index.name = "pseudobulk_id"

    availability_df = pd.DataFrame(availability_records)

    return log_scores, z_scores, availability_df


def summarize_scores(
    scores_with_meta: pd.DataFrame,
    score_columns: List[str],
    group_cols: List[str],
    score_type: str,
) -> pd.DataFrame:
    records = []

    for group_values, group in scores_with_meta.groupby(group_cols, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        group_info = dict(zip(group_cols, group_values))

        for program in score_columns:
            values = pd.to_numeric(group[program], errors="coerce").dropna()

            if values.empty:
                continue

            records.append(
                {
                    **group_info,
                    "score_type": score_type,
                    "ecm_program": program,
                    "n_pseudobulk": group["pseudobulk_id"].nunique(),
                    "n_donors": group["donor"].nunique(),
                    "n_cells_total": group["n_cells"].sum(),
                    "mean_score": values.mean(),
                    "median_score": values.median(),
                    "std_score": values.std(ddof=0),
                    "q25_score": values.quantile(0.25),
                    "q75_score": values.quantile(0.75),
                }
            )

    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--layer",
        type=str,
        default=None,
        help="Expression layer to use. Default: decontXcounts if available, else raw_counts, else X.",
    )

    parser.add_argument(
        "--method-filter",
        type=str,
        default=None,
        help="Optional method filter, e.g., 10X. Leave empty to include all methods.",
    )

    parser.add_argument(
        "--min-cells-per-group",
        type=int,
        default=20,
        help="Minimum cells required per donor × organ × cell type pseudobulk group.",
    )

    parser.add_argument(
        "--save-counts",
        action="store_true",
        help="Save pseudobulk count matrix. This may create a large file.",
    )

    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    matrisome_genes = load_matrisome_genes(MATRISOME_FILE)
    program_df = load_curated_program_gene_sets(CURATED_PROGRAM_FILE)

    program_df.to_csv(
        PROCESSED_DIR / "tabula_sapiens_reference_ecm_program_gene_sets.csv",
        index=False,
    )

    adata, chosen_layer = load_matrisome_subset(
        h5ad_path=FULL_H5AD,
        matrisome_genes=matrisome_genes,
        layer_name=args.layer,
    )

    obs = build_obs_metadata(
        adata=adata,
        method_filter=args.method_filter,
    )

    X = extract_expression_matrix(adata, chosen_layer)

    if args.method_filter is not None:
        # Need subset expression matrix to filtered cells.
        keep_indices = obs.index.to_numpy()
        X = X[keep_indices, :]

        # Reset obs index to match X row positions.
        obs = obs.reset_index(drop=True)

    gene_symbols = adata.var["gene_symbol_clean"].astype(str).str.upper().tolist()

    X, unique_gene_symbols = aggregate_duplicate_genes(
        matrix=X,
        gene_symbols=gene_symbols,
    )

    print(f"[INFO] Expression matrix after gene aggregation: {X.shape}")

    obs_with_groups = create_group_labels(
        obs=obs,
        min_cells_per_group=args.min_cells_per_group,
    )

    valid_cell_indices = np.arange(obs_with_groups.shape[0])

    pb_counts, pb_metadata = build_pseudobulk_counts(
        X=X,
        obs_with_groups=obs_with_groups,
        valid_cell_indices=valid_cell_indices,
        gene_names=unique_gene_symbols,
    )

    print(f"[INFO] Pseudobulk counts shape: {pb_counts.shape}")
    print(f"[INFO] Pseudobulk metadata shape: {pb_metadata.shape}")

    if args.save_counts:
        pb_counts.to_csv(
            PROCESSED_DIR / "tabula_sapiens_pseudobulk_counts.csv.gz",
            compression="gzip",
        )

    pb_metadata.to_csv(
        PROCESSED_DIR / "tabula_sapiens_pseudobulk_metadata.csv",
        index=False,
    )

    log_cpm = normalize_counts_to_log_cpm(pb_counts)

    log_cpm.to_csv(
        PROCESSED_DIR / "tabula_sapiens_pseudobulk_log_cpm.csv.gz",
        compression="gzip",
    )

    log_scores, z_scores, availability_df = compute_program_scores(
        log_cpm=log_cpm,
        program_df=program_df,
    )

    availability_df.to_csv(
        PROCESSED_DIR / "tabula_sapiens_program_gene_availability.csv",
        index=False,
    )

    log_scores.to_csv(
        PROCESSED_DIR / "tabula_sapiens_ecm_program_scores_log_cpm.csv",
    )

    z_scores.to_csv(
        PROCESSED_DIR / "tabula_sapiens_ecm_program_scores_zscore.csv",
    )

    log_scores_with_meta = pb_metadata.merge(
        log_scores.reset_index(),
        on="pseudobulk_id",
        how="left",
    )

    z_scores_with_meta = pb_metadata.merge(
        z_scores.reset_index(),
        on="pseudobulk_id",
        how="left",
    )

    log_scores_with_meta.to_csv(
        PROCESSED_DIR / "tabula_sapiens_ecm_program_scores_log_cpm_with_metadata.csv",
        index=False,
    )

    z_scores_with_meta.to_csv(
        PROCESSED_DIR / "tabula_sapiens_ecm_program_scores_zscore_with_metadata.csv",
        index=False,
    )

    score_cols = [program for program in PROGRAM_ORDER if program in z_scores.columns]

    organ_celltype_log = summarize_scores(
        scores_with_meta=log_scores_with_meta,
        score_columns=score_cols,
        group_cols=["organ", "cell_type", "compartment", "method"],
        score_type="log_cpm_mean",
    )

    organ_celltype_z = summarize_scores(
        scores_with_meta=z_scores_with_meta,
        score_columns=score_cols,
        group_cols=["organ", "cell_type", "compartment", "method"],
        score_type="zscore_mean",
    )

    celltype_z = summarize_scores(
        scores_with_meta=z_scores_with_meta,
        score_columns=score_cols,
        group_cols=["cell_type", "compartment", "method"],
        score_type="zscore_mean",
    )

    compartment_z = summarize_scores(
        scores_with_meta=z_scores_with_meta,
        score_columns=score_cols,
        group_cols=["compartment", "method"],
        score_type="zscore_mean",
    )

    organ_z = summarize_scores(
        scores_with_meta=z_scores_with_meta,
        score_columns=score_cols,
        group_cols=["organ", "method"],
        score_type="zscore_mean",
    )

    organ_celltype_summary = pd.concat(
        [organ_celltype_log, organ_celltype_z],
        ignore_index=True,
    )

    organ_celltype_summary.to_csv(
        PROCESSED_DIR / "tabula_sapiens_organ_celltype_program_summary.csv",
        index=False,
    )

    celltype_z.to_csv(
        PROCESSED_DIR / "tabula_sapiens_celltype_program_summary.csv",
        index=False,
    )

    compartment_z.to_csv(
        PROCESSED_DIR / "tabula_sapiens_compartment_program_summary.csv",
        index=False,
    )

    organ_z.to_csv(
        PROCESSED_DIR / "tabula_sapiens_organ_program_summary.csv",
        index=False,
    )

    global_summary = pd.DataFrame(
        [
            {
                "source_file": str(FULL_H5AD),
                "chosen_layer": chosen_layer if chosen_layer else "X",
                "method_filter": args.method_filter if args.method_filter else "all",
                "min_cells_per_group": args.min_cells_per_group,
                "n_cells_original": int(adata.n_obs),
                "n_cells_used": int(obs.shape[0]),
                "n_matched_matrisome_genes": int(len(unique_gene_symbols)),
                "n_pseudobulk_groups": int(pb_metadata.shape[0]),
                "n_donors": int(pb_metadata["donor"].nunique()),
                "n_organs": int(pb_metadata["organ"].nunique()),
                "n_cell_types": int(pb_metadata["cell_type"].nunique()),
                "n_compartments": int(pb_metadata["compartment"].nunique()),
                "n_methods": int(pb_metadata["method"].nunique()),
                "n_ecm_programs": int(len(score_cols)),
            }
        ]
    )

    global_summary.to_csv(
        PROCESSED_DIR / "tabula_sapiens_global_summary.csv",
        index=False,
    )

    print("\n[GLOBAL SUMMARY]")
    print(global_summary)

    print("\n[SAVED]")
    print(PROCESSED_DIR / "tabula_sapiens_pseudobulk_metadata.csv")
    print(PROCESSED_DIR / "tabula_sapiens_pseudobulk_log_cpm.csv.gz")
    print(PROCESSED_DIR / "tabula_sapiens_program_gene_availability.csv")
    print(PROCESSED_DIR / "tabula_sapiens_ecm_program_scores_zscore_with_metadata.csv")
    print(PROCESSED_DIR / "tabula_sapiens_organ_celltype_program_summary.csv")
    print(PROCESSED_DIR / "tabula_sapiens_celltype_program_summary.csv")
    print(PROCESSED_DIR / "tabula_sapiens_compartment_program_summary.csv")
    print(PROCESSED_DIR / "tabula_sapiens_organ_program_summary.csv")
    print(PROCESSED_DIR / "tabula_sapiens_global_summary.csv")


if __name__ == "__main__":
    main()