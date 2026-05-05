from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import pearsonr, spearmanr


PROJECT_ROOT = Path(".")

CURATED_PROGRAM_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "latent_baseline_embeddings"
    / "rna_tissue_consensus"
    / "curated_recurring_ecm_programs"
    / "combined_nmf_module_annotations_curated_programs.csv"
)

GTEX_MATRIX = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "gtex_v11_sample_level"
    / "gtex_v11_matrisome_expression_log2.parquet"
)

GTEX_METADATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "gtex_v11_sample_level"
    / "gtex_v11_sample_metadata.csv"
)

GTEX_MEAN_SCORE_WITH_METADATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "gtex_v11_sample_level"
    / "gtex_v11_program_scores_zscore_mean_with_metadata.csv"
)

TABULA_MATRIX = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tabula_sapiens"
    / "tabula_sapiens_pseudobulk_log_cpm.csv.gz"
)

TABULA_METADATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tabula_sapiens"
    / "tabula_sapiens_pseudobulk_metadata.csv"
)

TABULA_MEAN_SCORE_WITH_METADATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tabula_sapiens"
    / "tabula_sapiens_ecm_program_scores_zscore_with_metadata.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "rank_based_ecm_program_scoring"
TABLE_DIR = OUTPUT_DIR / "tables"
HTML_DIR = OUTPUT_DIR / "figures" / "html"
PNG_DIR = OUTPUT_DIR / "figures" / "png"
REPORT_DIR = OUTPUT_DIR / "reports"


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


def ensure_dirs() -> None:
    for folder in [TABLE_DIR, HTML_DIR, PNG_DIR, REPORT_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def save_figure(fig: go.Figure, name: str, width: int = 1350, height: int = 850) -> None:
    html_path = HTML_DIR / f"{name}.html"
    png_path = PNG_DIR / f"{name}.png"

    fig.update_layout(width=width, height=height)
    fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)

    try:
        fig.write_image(str(png_path), scale=3)
        print(f"[SAVED] {png_path}")
    except Exception as exc:
        print(f"[WARNING] Could not export PNG for {name}: {exc}")

    print(f"[SAVED] {html_path}")


def normalize_gene(gene: str) -> str:
    return str(gene).strip().upper()


def split_comma_list(value: str) -> List[str]:
    if pd.isna(value):
        return []

    return [
        normalize_gene(item)
        for item in str(value).split(",")
        if item.strip()
    ]


def zscore_columns(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy().astype(float)

    means = x.mean(axis=0)
    stds = x.std(axis=0, ddof=0).replace(0, np.nan)

    z = x.sub(means, axis=1).div(stds, axis=1)
    z = z.fillna(0.0)

    return z


def load_curated_program_gene_sets(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing curated program file: {path}")

    df = pd.read_csv(path)

    required = [
        "feature_set",
        "component",
        "ecm_program_curated",
        "top_genes",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Curated program file is missing columns: {missing}")

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


def load_matrix(path: Path, dataset_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {dataset_name} matrix: {path}")

    print(f"[INFO] Loading {dataset_name} matrix: {path}")

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, index_col=0)

    df.index = df.index.astype(str)
    df.columns = [normalize_gene(col) for col in df.columns]

    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    print(f"[INFO] {dataset_name} matrix shape: {df.shape}")

    return df


def load_metadata(path: Path, id_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata file: {path}")

    meta = pd.read_csv(path)
    meta[id_col] = meta[id_col].astype(str)

    return meta


def compute_rank_based_scores(
    matrix: pd.DataFrame,
    program_df: pd.DataFrame,
    dataset_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Rank all available Matrisome genes within each sample.

    Higher expression gets higher percentile rank.

    rank_percentile_score:
        mean percentile rank of program genes.

    top10_fraction_score:
        fraction of program genes among top 10% expressed Matrisome genes.

    top20_fraction_score:
        fraction of program genes among top 20% expressed Matrisome genes.
    """
    available_genes = set(matrix.columns)
    n_genes_total = matrix.shape[1]

    print(f"[INFO] Computing within-sample ranks for {dataset_name}")
    ranks = matrix.rank(axis=1, method="average", ascending=True)

    if n_genes_total > 1:
        percentile = (ranks - 1.0) / (n_genes_total - 1.0)
    else:
        percentile = ranks * 0.0

    top10_threshold = matrix.quantile(0.90, axis=1)
    top20_threshold = matrix.quantile(0.80, axis=1)

    rank_score_df = pd.DataFrame(index=matrix.index)
    top10_df = pd.DataFrame(index=matrix.index)
    top20_df = pd.DataFrame(index=matrix.index)

    availability_records = []

    for row in program_df.itertuples():
        program = str(row.ecm_program)
        program_genes = set(split_comma_list(row.program_genes))

        available = sorted(program_genes.intersection(available_genes))
        missing = sorted(program_genes.difference(available_genes))

        availability_records.append(
            {
                "dataset": dataset_name,
                "ecm_program": program,
                "n_program_genes": len(program_genes),
                "n_available_genes": len(available),
                "n_missing_genes": len(missing),
                "availability_fraction": len(available) / len(program_genes)
                if program_genes
                else np.nan,
                "available_genes": ", ".join(available),
                "missing_genes": ", ".join(missing),
            }
        )

        if not available:
            rank_score_df[program] = np.nan
            top10_df[program] = np.nan
            top20_df[program] = np.nan
            continue

        rank_score_df[program] = percentile[available].mean(axis=1)

        # For each sample, determine whether program genes are above top10/top20 threshold.
        top10_df[program] = matrix[available].ge(top10_threshold, axis=0).mean(axis=1)
        top20_df[program] = matrix[available].ge(top20_threshold, axis=0).mean(axis=1)

    rank_score_df.index.name = "sample_id"
    top10_df.index.name = "sample_id"
    top20_df.index.name = "sample_id"

    availability_df = pd.DataFrame(availability_records)

    return rank_score_df, top10_df, top20_df, availability_df


def merge_scores_with_metadata(
    rank_scores: pd.DataFrame,
    top10_scores: pd.DataFrame,
    top20_scores: pd.DataFrame,
    metadata: pd.DataFrame,
    id_col: str,
) -> pd.DataFrame:
    rank = rank_scores.reset_index().rename(columns={"sample_id": id_col})
    top10 = top10_scores.reset_index().rename(columns={"sample_id": id_col})
    top20 = top20_scores.reset_index().rename(columns={"sample_id": id_col})

    rank = rank.rename(
        columns={program: f"{program}__rank_percentile" for program in PROGRAM_ORDER if program in rank.columns}
    )
    top10 = top10.rename(
        columns={program: f"{program}__top10_fraction" for program in PROGRAM_ORDER if program in top10.columns}
    )
    top20 = top20.rename(
        columns={program: f"{program}__top20_fraction" for program in PROGRAM_ORDER if program in top20.columns}
    )

    merged = metadata.merge(rank, on=id_col, how="right")
    merged = merged.merge(top10, on=id_col, how="left")
    merged = merged.merge(top20, on=id_col, how="left")

    return merged


def summarize_by_group(
    scores_with_meta: pd.DataFrame,
    group_cols: List[str],
    id_col: str,
    score_suffix: str,
    dataset_name: str,
) -> pd.DataFrame:
    score_cols = [
        f"{program}__{score_suffix}"
        for program in PROGRAM_ORDER
        if f"{program}__{score_suffix}" in scores_with_meta.columns
    ]

    records = []

    for group_values, group in scores_with_meta.groupby(group_cols, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        group_info = dict(zip(group_cols, group_values))

        n_units = group[id_col].nunique()

        n_donors = group["donor"].nunique() if "donor" in group.columns else np.nan
        if "subject_id" in group.columns:
            n_donors = group["subject_id"].nunique()

        n_cells = group["n_cells"].sum() if "n_cells" in group.columns else np.nan

        for col in score_cols:
            program = col.replace(f"__{score_suffix}", "")

            values = pd.to_numeric(group[col], errors="coerce").dropna()

            if values.empty:
                continue

            records.append(
                {
                    "dataset": dataset_name,
                    **group_info,
                    "ecm_program": program,
                    "score_type": score_suffix,
                    "n_units": n_units,
                    "n_donors": n_donors,
                    "n_cells": n_cells,
                    "mean_score": values.mean(),
                    "median_score": values.median(),
                    "std_score": values.std(ddof=0),
                    "q25_score": values.quantile(0.25),
                    "q75_score": values.quantile(0.75),
                }
            )

    return pd.DataFrame(records)


def compare_with_mean_scores(
    rank_scores_with_meta: pd.DataFrame,
    mean_scores_with_meta_path: Path,
    id_col: str,
    dataset_name: str,
) -> pd.DataFrame:
    if not mean_scores_with_meta_path.exists():
        print(f"[WARNING] Missing previous mean score file: {mean_scores_with_meta_path}")
        return pd.DataFrame()

    mean_df = pd.read_csv(mean_scores_with_meta_path)
    mean_df[id_col] = mean_df[id_col].astype(str)

    rank_df = rank_scores_with_meta.copy()
    rank_df[id_col] = rank_df[id_col].astype(str)

    merged = rank_df.merge(mean_df, on=id_col, how="inner", suffixes=("_rank", "_mean"))

    records = []

    for program in PROGRAM_ORDER:
        rank_col = f"{program}__rank_percentile"

        if rank_col not in merged.columns:
            continue

        if program not in merged.columns:
            continue

        x = pd.to_numeric(merged[rank_col], errors="coerce")
        y = pd.to_numeric(merged[program], errors="coerce")

        valid = x.notna() & y.notna()

        if valid.sum() < 3:
            continue

        pearson = pearsonr(x[valid], y[valid])
        spearman = spearmanr(x[valid], y[valid])

        records.append(
            {
                "dataset": dataset_name,
                "ecm_program": program,
                "n_samples": int(valid.sum()),
                "pearson_r": float(pearson.statistic),
                "pearson_p": float(pearson.pvalue),
                "spearman_r": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue),
            }
        )

    return pd.DataFrame(records)


def make_heatmap_from_summary(
    summary_df: pd.DataFrame,
    row_col: str,
    column_col: str,
    dataset_name: str,
    score_type: str,
    name: str,
    title: str,
) -> None:
    subset = summary_df[summary_df["score_type"].eq(score_type)].copy()

    if subset.empty:
        print(f"[SKIP] Empty summary for {name}")
        return

    matrix = subset.pivot_table(
        index="ecm_program",
        columns=column_col,
        values="mean_score",
        aggfunc="mean",
        fill_value=0.0,
    )

    matrix = matrix.loc[[p for p in PROGRAM_ORDER if p in matrix.index]]

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns.tolist(),
            y=matrix.index.tolist(),
            colorscale="Viridis",
            colorbar=dict(title="Mean score"),
            hovertemplate=(
                "Program: %{y}<br>"
                f"{column_col}: " + "%{x}<br>"
                "Score: %{z:.3f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(l=310, r=60, t=100, b=150),
        xaxis=dict(tickangle=35),
    )

    save_figure(fig, name, width=1450, height=850)


def plot_correlation_heatmap(correlation_df: pd.DataFrame, dataset_name: str) -> None:
    if correlation_df.empty:
        return

    matrix = correlation_df.pivot_table(
        index="ecm_program",
        columns="dataset",
        values="spearman_r",
        aggfunc="mean",
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns.tolist(),
            y=matrix.index.tolist(),
            text=[[f"{v:.2f}" for v in row] for row in matrix.values],
            texttemplate="%{text}",
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="Spearman r"),
            hovertemplate=(
                "Program: %{y}<br>"
                "Dataset: %{x}<br>"
                "Spearman r: %{z:.3f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Rank-based score versus mean-expression score correlation",
        template="plotly_white",
        margin=dict(l=310, r=60, t=100, b=100),
    )

    save_figure(fig, "rank_vs_mean_score_spearman_correlation", width=1100, height=760)


def write_summary_report(
    gtex_corr: pd.DataFrame,
    tabula_corr: pd.DataFrame,
    gtex_availability: pd.DataFrame,
    tabula_availability: pd.DataFrame,
) -> None:
    report_path = REPORT_DIR / "rank_based_ecm_program_scoring_summary.md"

    lines = []

    lines.append("# Rank-Based ECM Program Scoring Robustness Summary\n")
    lines.append("## Purpose\n")
    lines.append(
        "This analysis tests whether the nine curated ECM programs remain stable when scored using within-sample rank-based scoring rather than mean expression."
    )

    lines.append("\n## Scoring definitions\n")
    lines.append("- **rank_percentile_score**: mean within-sample percentile rank of genes in an ECM program.\n")
    lines.append("- **top10_fraction_score**: fraction of program genes ranked in the top 10% of Matrisome genes in that sample.\n")
    lines.append("- **top20_fraction_score**: fraction of program genes ranked in the top 20% of Matrisome genes in that sample.\n")

    lines.append("\n## GTEx V11 rank-vs-mean correlations\n")
    if not gtex_corr.empty:
        for row in gtex_corr.itertuples():
            lines.append(
                f"- **{row.ecm_program}**: Spearman r = {row.spearman_r:.3f}, Pearson r = {row.pearson_r:.3f}"
            )
    else:
        lines.append("No GTEx correlation table was generated.\n")

    lines.append("\n## Tabula Sapiens rank-vs-mean correlations\n")
    if not tabula_corr.empty:
        for row in tabula_corr.itertuples():
            lines.append(
                f"- **{row.ecm_program}**: Spearman r = {row.spearman_r:.3f}, Pearson r = {row.pearson_r:.3f}"
            )
    else:
        lines.append("No Tabula Sapiens correlation table was generated.\n")

    lines.append("\n## Interpretation\n")
    lines.append(
        "High correlations indicate that ECM program activity is robust to the choice of scoring method. Discrepant programs should be interpreted carefully, as they may depend on absolute expression rather than relative within-sample gene ranking."
    )

    lines.append("\n## Methodological relevance\n")
    lines.append(
        "This robustness analysis is inspired by rank-based gene-set scoring approaches such as UCell, used by MatriSpace for spatial matrisome scoring. It strengthens the framework by showing whether the ECM programs remain stable under a scoring scheme less sensitive to library size and expression scale."
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAVED] {report_path}")


def main() -> None:
    ensure_dirs()

    program_df = load_curated_program_gene_sets(CURATED_PROGRAM_FILE)
    program_df.to_csv(TABLE_DIR / "rank_based_reference_ecm_program_gene_sets.csv", index=False)

    # GTEx V11
    gtex_matrix = load_matrix(GTEX_MATRIX, dataset_name="gtex_v11")
    gtex_metadata = load_metadata(GTEX_METADATA, id_col="sample_id")

    gtex_rank, gtex_top10, gtex_top20, gtex_availability = compute_rank_based_scores(
        matrix=gtex_matrix,
        program_df=program_df,
        dataset_name="gtex_v11",
    )

    gtex_scores_meta = merge_scores_with_metadata(
        rank_scores=gtex_rank,
        top10_scores=gtex_top10,
        top20_scores=gtex_top20,
        metadata=gtex_metadata,
        id_col="sample_id",
    )

    gtex_scores_meta.to_csv(TABLE_DIR / "gtex_v11_rank_based_program_scores_with_metadata.csv", index=False)
    gtex_availability.to_csv(TABLE_DIR / "gtex_v11_rank_based_program_gene_availability.csv", index=False)

    gtex_tissue_summary = summarize_by_group(
        scores_with_meta=gtex_scores_meta,
        group_cols=["tissue"],
        id_col="sample_id",
        score_suffix="rank_percentile",
        dataset_name="gtex_v11",
    )

    gtex_tissue_detail_summary = summarize_by_group(
        scores_with_meta=gtex_scores_meta,
        group_cols=["tissue", "tissue_detail"],
        id_col="sample_id",
        score_suffix="rank_percentile",
        dataset_name="gtex_v11",
    )

    gtex_tissue_summary.to_csv(TABLE_DIR / "gtex_v11_rank_based_tissue_summary.csv", index=False)
    gtex_tissue_detail_summary.to_csv(TABLE_DIR / "gtex_v11_rank_based_tissue_detail_summary.csv", index=False)

    gtex_corr = compare_with_mean_scores(
        rank_scores_with_meta=gtex_scores_meta,
        mean_scores_with_meta_path=GTEX_MEAN_SCORE_WITH_METADATA,
        id_col="sample_id",
        dataset_name="gtex_v11",
    )

    gtex_corr.to_csv(TABLE_DIR / "gtex_v11_rank_vs_mean_score_correlation.csv", index=False)

    make_heatmap_from_summary(
        summary_df=gtex_tissue_summary,
        row_col="ecm_program",
        column_col="tissue",
        dataset_name="gtex_v11",
        score_type="rank_percentile",
        name="gtex_v11_rank_based_tissue_program_heatmap",
        title="GTEx V11 rank-based ECM program scores by tissue",
    )

    # Tabula Sapiens
    tabula_matrix = load_matrix(TABULA_MATRIX, dataset_name="tabula_sapiens")
    tabula_metadata = load_metadata(TABULA_METADATA, id_col="pseudobulk_id")

    tabula_rank, tabula_top10, tabula_top20, tabula_availability = compute_rank_based_scores(
        matrix=tabula_matrix,
        program_df=program_df,
        dataset_name="tabula_sapiens",
    )

    tabula_scores_meta = merge_scores_with_metadata(
        rank_scores=tabula_rank,
        top10_scores=tabula_top10,
        top20_scores=tabula_top20,
        metadata=tabula_metadata,
        id_col="pseudobulk_id",
    )

    tabula_scores_meta.to_csv(TABLE_DIR / "tabula_sapiens_rank_based_program_scores_with_metadata.csv", index=False)
    tabula_availability.to_csv(TABLE_DIR / "tabula_sapiens_rank_based_program_gene_availability.csv", index=False)

    tabula_compartment_summary = summarize_by_group(
        scores_with_meta=tabula_scores_meta,
        group_cols=["method", "compartment"],
        id_col="pseudobulk_id",
        score_suffix="rank_percentile",
        dataset_name="tabula_sapiens",
    )

    tabula_celltype_summary = summarize_by_group(
        scores_with_meta=tabula_scores_meta,
        group_cols=["method", "cell_type", "compartment"],
        id_col="pseudobulk_id",
        score_suffix="rank_percentile",
        dataset_name="tabula_sapiens",
    )

    tabula_organ_celltype_summary = summarize_by_group(
        scores_with_meta=tabula_scores_meta,
        group_cols=["method", "organ", "cell_type", "compartment"],
        id_col="pseudobulk_id",
        score_suffix="rank_percentile",
        dataset_name="tabula_sapiens",
    )

    tabula_compartment_summary.to_csv(TABLE_DIR / "tabula_sapiens_rank_based_compartment_summary.csv", index=False)
    tabula_celltype_summary.to_csv(TABLE_DIR / "tabula_sapiens_rank_based_celltype_summary.csv", index=False)
    tabula_organ_celltype_summary.to_csv(TABLE_DIR / "tabula_sapiens_rank_based_organ_celltype_summary.csv", index=False)

    tabula_corr = compare_with_mean_scores(
        rank_scores_with_meta=tabula_scores_meta,
        mean_scores_with_meta_path=TABULA_MEAN_SCORE_WITH_METADATA,
        id_col="pseudobulk_id",
        dataset_name="tabula_sapiens",
    )

    tabula_corr.to_csv(TABLE_DIR / "tabula_sapiens_rank_vs_mean_score_correlation.csv", index=False)

    for method in sorted(tabula_compartment_summary["method"].dropna().astype(str).unique()):
        method_summary = tabula_compartment_summary[tabula_compartment_summary["method"].astype(str).eq(method)]

        make_heatmap_from_summary(
            summary_df=method_summary,
            row_col="ecm_program",
            column_col="compartment",
            dataset_name="tabula_sapiens",
            score_type="rank_percentile",
            name=f"tabula_sapiens_rank_based_compartment_heatmap_{method}",
            title=f"Tabula Sapiens rank-based ECM program scores by compartment<br><sup>{method}</sup>",
        )

    all_corr = pd.concat([gtex_corr, tabula_corr], ignore_index=True)
    all_corr.to_csv(TABLE_DIR / "rank_based_all_rank_vs_mean_score_correlation.csv", index=False)
    plot_correlation_heatmap(all_corr, dataset_name="combined")

    write_summary_report(
        gtex_corr=gtex_corr,
        tabula_corr=tabula_corr,
        gtex_availability=gtex_availability,
        tabula_availability=tabula_availability,
    )

    print("\n[DONE]")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Tables: {TABLE_DIR}")
    print(f"Figures HTML: {HTML_DIR}")
    print(f"Figures PNG: {PNG_DIR}")


if __name__ == "__main__":
    main()