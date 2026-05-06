from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Sequence

import h5py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import sparse
from scipy.stats import rankdata

from ecm_program_atlas.scoring import (
    ProgramGeneSet,
    load_programs_from_curated_table,
)


DEFAULT_PROGRAM_TABLE = Path(
    "results/tables/frozen/combined_nmf_module_annotations_curated_programs.csv"
)

DEFAULT_OUTPUT_DIR = Path("results/revision_spatial_validation")

DEFAULT_DATASETS = {
    "breast_cancer_cytassist_ffpe": Path("data/raw/spatial_visium/breast_cancer_cytassist_ffpe"),
    "human_lymph_node": Path("data/raw/spatial_visium/human_lymph_node"),
}


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


BASEMENT_MEMBRANE_GENES = [
    "COL4A1", "COL4A2", "COL4A3", "COL4A4", "COL4A5", "COL4A6",
    "LAMA1", "LAMA2", "LAMA3", "LAMA4", "LAMA5",
    "LAMB1", "LAMB2", "LAMB3", "LAMB4",
    "LAMC1", "LAMC2", "LAMC3",
    "NID1", "NID2", "HSPG2", "AGRN", "COL18A1",
    "COL7A1", "COL17A1",
]

INTERSTITIAL_ECM_GENES = [
    "COL1A1", "COL1A2", "COL3A1", "COL5A1", "COL5A2",
    "COL6A1", "COL6A2", "COL6A3", "COL12A1", "COL14A1",
    "COL15A1", "DCN", "LUM", "BGN", "FMOD", "FBLN1", "FBLN2",
    "FBN1", "ELN", "FN1", "TNC", "THBS1", "THBS2", "SPARC",
]

PERIVASCULAR_ECM_GENES = [
    "COL4A1", "COL4A2", "COL18A1", "HSPG2", "NID1", "NID2",
    "LAMA4", "LAMA5", "LAMB1", "LAMB2", "AGRN", "VWF", "EMCN",
    "MCAM", "PDGFRB", "RGS5", "ACTA2", "TAGLN", "CSPG4",
]


KEY_SCORE_COLUMNS = [
    "Vascular/stromal/interstitial ECM",
    "Epithelial/mucosal basement membrane ECM",
    "Stromal remodeling ECM",
    "Renal/endothelial basement membrane ECM",
    "Immune/lymphoid remodeling ECM",
    "Reproductive-specialized ECM",
    "Reference basement membrane ECM",
    "Reference interstitial ECM",
    "Reference perivascular ECM",
]


def ensure_dirs(output_dir: Path) -> tuple[Path, Path, Path, Path]:
    table_dir = output_dir / "tables"
    html_dir = output_dir / "figures" / "html"
    png_dir = output_dir / "figures" / "png"
    report_dir = output_dir / "reports"

    for folder in [table_dir, html_dir, png_dir, report_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    return table_dir, html_dir, png_dir, report_dir


def save_figure(
    fig: go.Figure,
    name: str,
    html_dir: Path,
    png_dir: Path,
    width: int = 850,
    height: int = 850,
) -> None:
    fig.update_layout(width=width, height=height)

    html_path = html_dir / f"{name}.html"
    png_path = png_dir / f"{name}.png"

    fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)

    try:
        fig.write_image(str(png_path), scale=3)
        print(f"[SAVED] {png_path}")
    except Exception as exc:
        print(f"[WARNING] PNG export failed for {name}: {exc}")

    print(f"[SAVED] {html_path}")


def normalize_gene(gene: str) -> str:
    return str(gene).strip().upper()


def decode_array(values) -> list[str]:
    return [
        x.decode() if isinstance(x, bytes) else str(x)
        for x in values
    ]


def load_curated_programs(path: Path) -> list[ProgramGeneSet]:
    if not path.exists():
        raise FileNotFoundError(f"Missing curated program table:\n{path}")

    programs = load_programs_from_curated_table(
        str(path),
        program_col="ecm_program_curated",
        genes_col="top_genes",
    )

    lookup = {program.name: program for program in programs}
    ordered = [lookup[name] for name in PROGRAM_ORDER if name in lookup]

    if len(ordered) != len(PROGRAM_ORDER):
        missing = sorted(set(PROGRAM_ORDER).difference(lookup))
        raise ValueError(f"Missing curated ECM programs: {missing}")

    references = [
        ProgramGeneSet(
            "Reference basement membrane ECM",
            tuple(sorted(set(map(normalize_gene, BASEMENT_MEMBRANE_GENES)))),
        ),
        ProgramGeneSet(
            "Reference interstitial ECM",
            tuple(sorted(set(map(normalize_gene, INTERSTITIAL_ECM_GENES)))),
        ),
        ProgramGeneSet(
            "Reference perivascular ECM",
            tuple(sorted(set(map(normalize_gene, PERIVASCULAR_ECM_GENES)))),
        ),
    ]

    return ordered + references


def read_10x_h5_sparse(path: Path) -> tuple[list[str], list[str], sparse.csr_matrix, pd.DataFrame]:
    """
    Read 10x filtered_feature_bc_matrix.h5 efficiently.

    Returns
    -------
    barcodes:
        Spot barcodes.
    genes:
        Unique aggregated gene symbols.
    counts:
        Sparse spot × gene count matrix.
    feature_metadata:
        Full feature metadata before aggregation.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing matrix H5:\n{path}")

    with h5py.File(path, "r") as h5:
        if "matrix" not in h5:
            raise ValueError(f"H5 file does not contain /matrix group: {path}")

        m = h5["matrix"]

        data = m["data"][:]
        indices = m["indices"][:]
        indptr = m["indptr"][:]
        shape = tuple(m["shape"][:])

        features = m["features"]

        feature_ids = decode_array(features["id"][:])
        feature_names = decode_array(features["name"][:])
        feature_types = decode_array(features["feature_type"][:])

        genomes = (
            decode_array(features["genome"][:])
            if "genome" in features
            else [""] * len(feature_names)
        )

        barcodes = decode_array(m["barcodes"][:])

    feature_df = pd.DataFrame(
        {
            "feature_id": feature_ids,
            "feature_name": feature_names,
            "gene_symbol": [normalize_gene(x) for x in feature_names],
            "feature_type": feature_types,
            "genome": genomes,
        }
    )

    # 10x H5 matrix shape is features × barcodes.
    feature_by_barcode = sparse.csc_matrix(
        (data, indices, indptr),
        shape=shape,
    )

    gene_expression_mask = feature_df["feature_type"].eq("Gene Expression").values
    gene_symbols_raw = feature_df.loc[gene_expression_mask, "gene_symbol"].tolist()

    # Convert to barcodes × gene-expression-features.
    spot_by_feature = feature_by_barcode.T.tocsr()[:, gene_expression_mask]

    # Aggregate duplicate gene symbols by sparse matrix multiplication.
    unique_genes = sorted(set(gene_symbols_raw))
    gene_to_idx = {gene: idx for idx, gene in enumerate(unique_genes)}

    rows = np.arange(len(gene_symbols_raw), dtype=np.int64)
    cols = np.array([gene_to_idx[gene] for gene in gene_symbols_raw], dtype=np.int64)
    vals = np.ones(len(gene_symbols_raw), dtype=np.float32)

    feature_to_gene = sparse.csr_matrix(
        (vals, (rows, cols)),
        shape=(len(gene_symbols_raw), len(unique_genes)),
    )

    spot_by_gene = (spot_by_feature @ feature_to_gene).tocsr()

    return barcodes, unique_genes, spot_by_gene, feature_df


def read_positions(spatial_dir: Path) -> pd.DataFrame:
    tissue_positions = spatial_dir / "tissue_positions.csv"
    tissue_positions_list = spatial_dir / "tissue_positions_list.csv"

    if tissue_positions.exists():
        df = pd.read_csv(tissue_positions)
    elif tissue_positions_list.exists():
        df = pd.read_csv(
            tissue_positions_list,
            header=None,
            names=[
                "barcode",
                "in_tissue",
                "array_row",
                "array_col",
                "pxl_row_in_fullres",
                "pxl_col_in_fullres",
            ],
        )
    else:
        raise FileNotFoundError(
            f"No tissue_positions.csv or tissue_positions_list.csv found in {spatial_dir}"
        )

    rename_map = {
        "barcode": "barcode",
        "in_tissue": "in_tissue",
        "array_row": "array_row",
        "array_col": "array_col",
        "pxl_row_in_fullres": "pxl_row_in_fullres",
        "pxl_col_in_fullres": "pxl_col_in_fullres",
        "pxl_row_in_fullres ": "pxl_row_in_fullres",
        "pxl_col_in_fullres ": "pxl_col_in_fullres",
    }

    df = df.rename(columns=rename_map)

    required = [
        "barcode",
        "in_tissue",
        "array_row",
        "array_col",
        "pxl_row_in_fullres",
        "pxl_col_in_fullres",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Positions file missing columns {missing}. Available columns: {df.columns.tolist()}"
        )

    df = df[required].copy()
    df["barcode"] = df["barcode"].astype(str)
    df["in_tissue"] = pd.to_numeric(df["in_tissue"], errors="coerce").fillna(0).astype(int)

    return df


def read_scalefactors(spatial_dir: Path) -> dict:
    path = spatial_dir / "scalefactors_json.json"

    if not path.exists():
        raise FileNotFoundError(f"Missing scalefactors_json.json in {spatial_dir}")

    return json.loads(path.read_text())


def build_program_weight_matrix(
    genes: Sequence[str],
    programs: Sequence[ProgramGeneSet],
) -> tuple[sparse.csr_matrix, pd.DataFrame]:
    gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}

    rows = []
    cols = []
    vals = []

    availability_records = []

    for program_idx, program in enumerate(programs):
        program_genes = sorted(set(map(normalize_gene, program.genes)))
        available = [gene for gene in program_genes if gene in gene_to_idx]
        missing = sorted(set(program_genes).difference(gene_to_idx))

        availability_records.append(
            {
                "ecm_program": program.name,
                "n_program_genes": len(program_genes),
                "n_available_genes": len(available),
                "n_missing_genes": len(missing),
                "availability_fraction": len(available) / len(program_genes) if program_genes else np.nan,
                "available_genes": ", ".join(available),
                "missing_genes": ", ".join(missing),
            }
        )

        if not available:
            continue

        weight = 1.0 / len(available)

        for gene in available:
            rows.append(gene_to_idx[gene])
            cols.append(program_idx)
            vals.append(weight)

    W = sparse.csr_matrix(
        (vals, (rows, cols)),
        shape=(len(genes), len(programs)),
        dtype=np.float32,
    )

    availability = pd.DataFrame(availability_records)

    return W, availability


def normalize_selected_genes_to_log_cpm(
    counts: sparse.csr_matrix,
    selected_gene_indices: np.ndarray,
) -> np.ndarray:
    """
    Compute log1p(CPM) for selected genes only, using total counts across all genes.
    """
    totals = np.asarray(counts.sum(axis=1)).ravel().astype(np.float64)
    totals[totals == 0] = np.nan

    selected = counts[:, selected_gene_indices].astype(np.float64).tocsr()

    scale = 1e6 / totals
    scale = np.nan_to_num(scale, nan=0.0, posinf=0.0, neginf=0.0)

    selected = selected.multiply(scale[:, None])
    selected.data = np.log1p(selected.data)

    return selected.toarray()


def zscore_dense_columns(matrix: np.ndarray) -> np.ndarray:
    means = np.nanmean(matrix, axis=0)
    stds = np.nanstd(matrix, axis=0)

    stds[stds == 0] = np.nan
    z = (matrix - means) / stds
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    return z


def score_programs_sparse(
    counts: sparse.csr_matrix,
    genes: Sequence[str],
    programs: Sequence[ProgramGeneSet],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Vectorized scoring:
    1. Build gene × program weight matrix.
    2. Keep only genes used by any program.
    3. Normalize selected genes to log1p CPM.
    4. Z-score genes across spots.
    5. Matrix multiply selected_z @ program_weights.
    """
    genes = list(genes)

    program_weight_full, availability = build_program_weight_matrix(genes, programs)

    used_gene_indices = np.unique(program_weight_full.nonzero()[0])

    if len(used_gene_indices) == 0:
        raise ValueError("None of the program genes were found in the spatial dataset.")

    selected_genes = [genes[idx] for idx in used_gene_indices]

    # Reduce program weight matrix to selected genes.
    program_weight_selected = program_weight_full[used_gene_indices, :]

    selected_log_cpm = normalize_selected_genes_to_log_cpm(
        counts=counts,
        selected_gene_indices=used_gene_indices,
    )

    selected_z = zscore_dense_columns(selected_log_cpm)

    scores = selected_z @ program_weight_selected.toarray()

    score_df = pd.DataFrame(
        scores,
        columns=[program.name for program in programs],
    )

    return score_df, availability


def merge_scores_positions(
    barcodes: Sequence[str],
    scores: pd.DataFrame,
    positions: pd.DataFrame,
    dataset: str,
) -> pd.DataFrame:
    score_df = scores.copy()
    score_df.insert(0, "barcode", list(barcodes))
    score_df["barcode"] = score_df["barcode"].astype(str)

    merged = positions.merge(score_df, on="barcode", how="inner")
    merged = merged[merged["in_tissue"].eq(1)].copy()
    merged.insert(0, "dataset", dataset)

    return merged


def robust_color_range(values: pd.Series) -> tuple[float, float]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy()

    if len(x) == 0:
        return -1.0, 1.0

    lo = float(np.percentile(x, 2))
    hi = float(np.percentile(x, 98))

    if lo == hi:
        lo = float(np.min(x))
        hi = float(np.max(x))

    if lo == hi:
        lo -= 1.0
        hi += 1.0

    return lo, hi


def plot_spatial_score(
    spatial_scores: pd.DataFrame,
    score_col: str,
    dataset: str,
    html_dir: Path,
    png_dir: Path,
) -> None:
    plot_df = spatial_scores.copy()
    plot_df[score_col] = pd.to_numeric(plot_df[score_col], errors="coerce")

    cmin, cmax = robust_color_range(plot_df[score_col])

    fig = go.Figure(
        data=go.Scattergl(
            x=plot_df["pxl_col_in_fullres"],
            y=-plot_df["pxl_row_in_fullres"],
            mode="markers",
            marker=dict(
                color=plot_df[score_col],
                colorscale="RdBu",
                cmin=cmin,
                cmax=cmax,
                colorbar=dict(title="z-score"),
                size=5,
                opacity=0.9,
            ),
            customdata=plot_df[["barcode", "array_row", "array_col"]],
            hovertemplate=(
                "Barcode: %{customdata[0]}<br>"
                "Array row: %{customdata[1]}<br>"
                "Array col: %{customdata[2]}<br>"
                f"{score_col}: " + "%{marker.color:.3f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=f"{dataset}: {score_col}",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        template="plotly_white",
        margin=dict(l=20, r=20, t=70, b=20),
    )

    safe_col = (
        score_col.lower()
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
    )

    save_figure(
        fig,
        name=f"spatial_map_{dataset}_{safe_col}",
        html_dir=html_dir,
        png_dir=png_dir,
        width=850,
        height=850,
    )


def compute_correlations_numpy(spatial_scores: pd.DataFrame, dataset: str) -> pd.DataFrame:
    score_cols = [
        col for col in spatial_scores.columns
        if col in PROGRAM_ORDER or col.startswith("Reference ")
    ]

    if len(score_cols) < 2:
        return pd.DataFrame()

    X = spatial_scores[score_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()

    # Pearson
    pearson = np.corrcoef(X, rowvar=False)

    # Spearman via ranks per column.
    rank_X = np.apply_along_axis(rankdata, 0, X)
    spearman = np.corrcoef(rank_X, rowvar=False)

    records = []

    for i, col_a in enumerate(score_cols):
        for j in range(i + 1, len(score_cols)):
            col_b = score_cols[j]

            records.append(
                {
                    "dataset": dataset,
                    "feature_a": col_a,
                    "feature_b": col_b,
                    "n_spots": X.shape[0],
                    "spearman_r": float(spearman[i, j]),
                    "pearson_r": float(pearson[i, j]),
                }
            )

    return pd.DataFrame(records)


def plot_correlation_heatmap(corr_df: pd.DataFrame, dataset: str, html_dir: Path, png_dir: Path) -> None:
    if corr_df.empty:
        return

    features = sorted(set(corr_df["feature_a"]).union(set(corr_df["feature_b"])))
    matrix = pd.DataFrame(np.eye(len(features)), index=features, columns=features)

    for row in corr_df.itertuples():
        matrix.loc[row.feature_a, row.feature_b] = row.spearman_r
        matrix.loc[row.feature_b, row.feature_a] = row.spearman_r

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns.tolist(),
            y=matrix.index.tolist(),
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="Spearman r"),
            hovertemplate=(
                "Feature A: %{y}<br>"
                "Feature B: %{x}<br>"
                "Spearman r: %{z:.3f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=f"{dataset}: spot-level ECM program correlations",
        template="plotly_white",
        margin=dict(l=320, r=60, t=90, b=300),
        xaxis=dict(tickangle=45),
    )

    save_figure(
        fig,
        name=f"spatial_score_correlation_heatmap_{dataset}",
        html_dir=html_dir,
        png_dir=png_dir,
        width=1200,
        height=1100,
    )


def select_key_scores(dataset: str) -> list[str]:
    keys = [
        "Vascular/stromal/interstitial ECM",
        "Epithelial/mucosal basement membrane ECM",
        "Stromal remodeling ECM",
        "Renal/endothelial basement membrane ECM",
        "Immune/lymphoid remodeling ECM",
        "Reference basement membrane ECM",
        "Reference interstitial ECM",
        "Reference perivascular ECM",
    ]

    if dataset == "human_lymph_node":
        keys.append("Immune/lymphoid remodeling ECM")

    if dataset == "breast_cancer_cytassist_ffpe":
        keys.append("Reproductive-specialized ECM")

    seen = set()
    return [x for x in keys if not (x in seen or seen.add(x))]


def export_plots_parallel(
    spatial_scores: pd.DataFrame,
    dataset: str,
    html_dir: Path,
    png_dir: Path,
    n_jobs: int,
) -> None:
    key_scores = select_key_scores(dataset)
    key_scores = [score for score in key_scores if score in spatial_scores.columns]

    if n_jobs <= 1:
        for score_col in key_scores:
            plot_spatial_score(spatial_scores, score_col, dataset, html_dir, png_dir)
        return

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = {
            executor.submit(plot_spatial_score, spatial_scores, score_col, dataset, html_dir, png_dir): score_col
            for score_col in key_scores
        }

        for future in as_completed(futures):
            score_col = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"[WARNING] Plot failed for {dataset} | {score_col}: {exc}")


def process_dataset(
    dataset: str,
    folder: Path,
    programs: Sequence[ProgramGeneSet],
    table_dir: Path,
    html_dir: Path,
    png_dir: Path,
    n_jobs: int,
) -> dict:
    print("=" * 100)
    print(f"[DATASET] {dataset}")
    print(f"Folder: {folder}")

    matrix_path = folder / "filtered_feature_bc_matrix.h5"
    spatial_dir = folder / "spatial"

    barcodes, genes, counts, feature_df = read_10x_h5_sparse(matrix_path)
    positions = read_positions(spatial_dir)
    scalefactors = json.loads((spatial_dir / "scalefactors_json.json").read_text())

    # Align to positions. Keep matrix order based on barcodes.
    barcode_to_idx = {barcode: idx for idx, barcode in enumerate(barcodes)}
    common_positions = positions[positions["barcode"].isin(barcode_to_idx)].copy()
    common_indices = common_positions["barcode"].map(barcode_to_idx).to_numpy()

    counts = counts[common_indices, :]
    barcodes_aligned = common_positions["barcode"].tolist()

    score_df, availability = score_programs_sparse(
        counts=counts,
        genes=genes,
        programs=programs,
    )
    score_df.index = barcodes_aligned

    spatial_scores = merge_scores_positions(
        barcodes=barcodes_aligned,
        scores=score_df,
        positions=common_positions,
        dataset=dataset,
    )

    corr = compute_correlations_numpy(spatial_scores, dataset=dataset)

    feature_df.to_csv(table_dir / f"{dataset}_feature_metadata.csv", index=False)
    availability.to_csv(table_dir / f"{dataset}_program_gene_availability.csv", index=False)
    spatial_scores.to_csv(table_dir / f"{dataset}_spatial_program_scores.csv", index=False)
    corr.to_csv(table_dir / f"{dataset}_spatial_program_correlations.csv", index=False)

    with (table_dir / f"{dataset}_scalefactors.json").open("w", encoding="utf-8") as file:
        json.dump(scalefactors, file, indent=2)

    export_plots_parallel(
        spatial_scores=spatial_scores,
        dataset=dataset,
        html_dir=html_dir,
        png_dir=png_dir,
        n_jobs=n_jobs,
    )

    plot_correlation_heatmap(
        corr_df=corr,
        dataset=dataset,
        html_dir=html_dir,
        png_dir=png_dir,
    )

    return {
        "dataset": dataset,
        "folder": str(folder),
        "n_features_total": int(feature_df.shape[0]),
        "n_gene_expression_features": int(feature_df["feature_type"].eq("Gene Expression").sum()),
        "n_spots_matrix": int(counts.shape[0]),
        "n_genes_matrix": int(counts.shape[1]),
        "n_spots_positions": int(positions.shape[0]),
        "n_spots_in_tissue": int(spatial_scores.shape[0]),
        "n_programs_scored": int(len(programs)),
        "mean_program_gene_availability": float(availability["availability_fraction"].mean()),
        "min_program_gene_availability": float(availability["availability_fraction"].min()),
    }


def write_report(dataset_summaries: list[dict], report_dir: Path) -> None:
    report_path = report_dir / "r7_spatial_validation_summary.md"

    lines = []
    lines.append("# R7 Focused Spatial Validation Summary\n")
    lines.append("## Purpose\n")
    lines.append(
        "This analysis projects the nine curated ECM programs onto two public 10x Visium datasets: "
        "a breast cancer CytAssist FFPE dataset and a healthy human lymph node dataset."
    )

    lines.append("\n## Datasets\n")
    for item in dataset_summaries:
        lines.append(f"### {item['dataset']}\n")
        lines.append(f"- Spots in tissue scored: {item['n_spots_in_tissue']}\n")
        lines.append(f"- Genes in matrix: {item['n_genes_matrix']}\n")
        lines.append(f"- ECM/reference programs scored: {item['n_programs_scored']}\n")
        lines.append(f"- Mean program gene availability: {item['mean_program_gene_availability']:.3f}\n")
        lines.append(f"- Minimum program gene availability: {item['min_program_gene_availability']:.3f}\n")

    lines.append("\n## Interpretation\n")
    lines.append(
        "The spatial maps should be interpreted as transcriptomic localization of ECM program activity, "
        "not as direct ECM protein deposition or matrix architecture."
    )

    lines.append("\n## Limitations\n")
    lines.append("- Visium spots contain multiple cells.\n")
    lines.append("- Transcriptomic ECM scores do not directly measure deposited ECM protein.\n")
    lines.append("- Spot-level ECM program scores should be interpreted alongside histology and ECM imaging where available.\n")
    lines.append("- This is a focused validation, not a full spatial ECM atlas.\n")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAVED] {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--program-table", type=Path, default=DEFAULT_PROGRAM_TABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--breast-dir", type=Path, default=DEFAULT_DATASETS["breast_cancer_cytassist_ffpe"])
    parser.add_argument("--lymph-dir", type=Path, default=DEFAULT_DATASETS["human_lymph_node"])
    parser.add_argument("--n-jobs", type=int, default=1)

    args = parser.parse_args()

    table_dir, html_dir, png_dir, report_dir = ensure_dirs(args.output_dir)

    programs = load_curated_programs(args.program_table)

    datasets = {
        "breast_cancer_cytassist_ffpe": args.breast_dir,
        "human_lymph_node": args.lymph_dir,
    }

    summaries = []
    for dataset, folder in datasets.items():
        summaries.append(
            process_dataset(
                dataset=dataset,
                folder=folder,
                programs=programs,
                table_dir=table_dir,
                html_dir=html_dir,
                png_dir=png_dir,
                n_jobs=args.n_jobs,
            )
        )

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(table_dir / "r7_spatial_dataset_summary.csv", index=False)

    write_report(summaries, report_dir=report_dir)

    metadata = {
        "program_table": str(args.program_table),
        "datasets": {key: str(value) for key, value in datasets.items()},
        "output_dir": str(args.output_dir),
        "n_jobs": args.n_jobs,
    }

    with (args.output_dir / "r7_spatial_validation_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\n[DONE]")
    print(f"Output folder: {args.output_dir}")
    print(f"Tables: {table_dir}")
    print(f"Figures HTML: {html_dir}")
    print(f"Figures PNG: {png_dir}")
    print(f"Report: {report_dir / 'r7_spatial_validation_summary.md'}")


if __name__ == "__main__":
    main()