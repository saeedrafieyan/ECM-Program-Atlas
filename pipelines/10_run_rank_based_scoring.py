from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import pearsonr, spearmanr

from ecm_program_atlas.scoring import (
    ProgramGeneSet,
    load_programs_from_curated_table,
    program_gene_availability,
    score_programs,
)


DEFAULT_PROGRAM_TABLE = Path(
    "results/tables/frozen/combined_nmf_module_annotations_curated_programs.csv"
)

DEFAULT_GTEX_MATRIX = Path(
    "data/processed/gtex_v11_sample_level/gtex_v11_matrisome_expression_log2.parquet"
)

DEFAULT_GTEX_METADATA = Path(
    "data/processed/gtex_v11_sample_level/gtex_v11_sample_metadata.csv"
)

DEFAULT_GTEX_MEAN_SCORE_WITH_METADATA = Path(
    "data/processed/gtex_v11_sample_level/gtex_v11_program_scores_zscore_mean_with_metadata.csv"
)

DEFAULT_TABULA_MATRIX = Path(
    "data/processed/tabula_sapiens/tabula_sapiens_pseudobulk_log_cpm.csv.gz"
)

DEFAULT_TABULA_METADATA = Path(
    "data/processed/tabula_sapiens/tabula_sapiens_pseudobulk_metadata.csv"
)

DEFAULT_TABULA_MEAN_SCORE_WITH_METADATA = Path(
    "data/processed/tabula_sapiens/tabula_sapiens_ecm_program_scores_zscore_with_metadata.csv"
)

DEFAULT_OUTPUT_DIR = Path("results/rank_based_scoring")


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
    width: int = 1350,
    height: int = 850,
) -> None:
    html_path = html_dir / f"{name}.html"
    png_path = png_dir / f"{name}.png"

    fig.update_layout(width=width, height=height)
    fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)

    try:
        fig.write_image(str(png_path), scale=3)
        print(f"[SAVED] {png_path}")
    except Exception as exc:
        print(f"[WARNING] Could not export PNG for {name}: {exc}")

    print(f"[SAVED] {html_path}")


def load_matrix(path: Path, dataset_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {dataset_name} matrix:\n{path}\n\n"
            "If you are running from the clean repo, copy or symlink the processed data "
            "from the old experimental project into the matching data/processed folder."
        )

    print(f"[INFO] Loading {dataset_name} matrix: {path}")

    if path.suffix == ".parquet":
        matrix = pd.read_parquet(path)
    else:
        matrix = pd.read_csv(path, index_col=0)

    matrix.index = matrix.index.astype(str)
    matrix.columns = [str(col).strip().upper() for col in matrix.columns]
    matrix = matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    print(f"[INFO] {dataset_name} matrix shape: {matrix.shape}")

    return matrix


def load_metadata(path: Path, id_col: str, dataset_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {dataset_name} metadata:\n{path}")

    metadata = pd.read_csv(path)

    if id_col not in metadata.columns:
        raise ValueError(
            f"Expected ID column '{id_col}' in {path}. "
            f"Available columns: {metadata.columns.tolist()}"
        )

    metadata[id_col] = metadata[id_col].astype(str)

    return metadata


def score_dataset(
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    id_col: str,
    programs: List[ProgramGeneSet],
    dataset_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Compute all rank-based and mean-based program scores for one dataset.
    """
    methods = [
        "mean",
        "zscore_mean",
        "rank_percentile",
        "top10_fraction",
        "top20_fraction",
    ]

    scores = score_programs(matrix=matrix, programs=programs, methods=methods)

    availability = program_gene_availability(matrix=matrix, programs=programs)
    availability.insert(0, "dataset", dataset_name)

    merged = metadata.copy()

    for method, score_df in scores.items():
        method_scores = score_df.copy()
        method_scores.index = method_scores.index.astype(str)
        method_scores = method_scores.reset_index().rename(columns={"index": id_col})

        if id_col not in method_scores.columns:
            method_scores = method_scores.rename(columns={method_scores.columns[0]: id_col})

        rename_map = {
            program: f"{program}__{method}"
            for program in method_scores.columns
            if program != id_col
        }

        method_scores = method_scores.rename(columns=rename_map)
        method_scores[id_col] = method_scores[id_col].astype(str)

        merged = merged.merge(method_scores, on=id_col, how="right")

    return merged, availability, scores


def summarize_by_group(
    scores_with_meta: pd.DataFrame,
    group_cols: List[str],
    id_col: str,
    dataset_name: str,
    score_method: str = "rank_percentile",
) -> pd.DataFrame:
    score_cols = [
        f"{program}__{score_method}"
        for program in PROGRAM_ORDER
        if f"{program}__{score_method}" in scores_with_meta.columns
    ]

    records = []

    for group_values, group in scores_with_meta.groupby(group_cols, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        group_info = dict(zip(group_cols, group_values))

        n_units = group[id_col].nunique()

        if "subject_id" in group.columns:
            n_donors = group["subject_id"].nunique()
        elif "donor" in group.columns:
            n_donors = group["donor"].nunique()
        else:
            n_donors = np.nan

        n_cells = group["n_cells"].sum() if "n_cells" in group.columns else np.nan

        for col in score_cols:
            program = col.replace(f"__{score_method}", "")

            values = pd.to_numeric(group[col], errors="coerce").dropna()

            if values.empty:
                continue

            records.append(
                {
                    "dataset": dataset_name,
                    **group_info,
                    "ecm_program": program,
                    "score_type": score_method,
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


def compare_rank_with_existing_mean_scores(
    rank_scores_with_meta: pd.DataFrame,
    mean_scores_with_meta_path: Path,
    id_col: str,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Compare new rank-percentile scores to old zscore-mean scores.

    Old score files contain columns named by program, without suffix.
    New score columns are named '{program}__rank_percentile'.
    """
    if not mean_scores_with_meta_path.exists():
        print(f"[WARNING] Missing previous mean score file: {mean_scores_with_meta_path}")
        return pd.DataFrame()

    old_scores = pd.read_csv(mean_scores_with_meta_path)

    if id_col not in old_scores.columns:
        raise ValueError(
            f"Expected ID column '{id_col}' in old score file: {mean_scores_with_meta_path}"
        )

    old_scores[id_col] = old_scores[id_col].astype(str)
    new_scores = rank_scores_with_meta.copy()
    new_scores[id_col] = new_scores[id_col].astype(str)

    merged = new_scores.merge(old_scores, on=id_col, how="inner", suffixes=("_rank", "_old"))

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
    column_col: str,
    score_type: str,
    title: str,
    name: str,
    html_dir: Path,
    png_dir: Path,
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

    matrix = matrix.loc[[program for program in PROGRAM_ORDER if program in matrix.index]]

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

    save_figure(fig, name=name, html_dir=html_dir, png_dir=png_dir, width=1450, height=850)


def plot_rank_mean_correlation_heatmap(
    correlation_df: pd.DataFrame,
    html_dir: Path,
    png_dir: Path,
) -> None:
    if correlation_df.empty:
        print("[SKIP] Correlation heatmap, empty correlation table.")
        return

    matrix = correlation_df.pivot_table(
        index="ecm_program",
        columns="dataset",
        values="spearman_r",
        aggfunc="mean",
    )

    matrix = matrix.loc[[program for program in PROGRAM_ORDER if program in matrix.index]]

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns.tolist(),
            y=matrix.index.tolist(),
            text=[[f"{value:.2f}" for value in row] for row in matrix.values],
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

    save_figure(
        fig,
        name="rank_vs_mean_score_spearman_correlation",
        html_dir=html_dir,
        png_dir=png_dir,
        width=1150,
        height=760,
    )


def write_report(
    gtex_corr: pd.DataFrame,
    tabula_corr: pd.DataFrame,
    report_dir: Path,
) -> None:
    report_path = report_dir / "rank_based_ecm_program_scoring_summary.md"

    lines = []
    lines.append("# Rank-Based ECM Program Scoring Robustness Summary\n")
    lines.append("## Purpose\n")
    lines.append(
        "This analysis tests whether the nine curated ECM programs remain stable when "
        "scored using within-sample rank-based scoring rather than mean expression."
    )

    lines.append("\n## Scoring definitions\n")
    lines.append(
        "- **rank_percentile_score**: mean within-sample percentile rank of genes in an ECM program."
    )
    lines.append(
        "- **top10_fraction_score**: fraction of program genes ranked in the top 10% of Matrisome genes."
    )
    lines.append(
        "- **top20_fraction_score**: fraction of program genes ranked in the top 20% of Matrisome genes."
    )

    lines.append("\n## GTEx V11 rank-vs-mean correlations\n")
    if gtex_corr.empty:
        lines.append("No GTEx V11 correlation table was generated.")
    else:
        for row in gtex_corr.itertuples():
            lines.append(
                f"- **{row.ecm_program}**: "
                f"Spearman r = {row.spearman_r:.3f}, Pearson r = {row.pearson_r:.3f}"
            )

    lines.append("\n## Tabula Sapiens rank-vs-mean correlations\n")
    if tabula_corr.empty:
        lines.append("No Tabula Sapiens correlation table was generated.")
    else:
        for row in tabula_corr.itertuples():
            lines.append(
                f"- **{row.ecm_program}**: "
                f"Spearman r = {row.spearman_r:.3f}, Pearson r = {row.pearson_r:.3f}"
            )

    lines.append("\n## Interpretation\n")
    lines.append(
        "High correlations indicate that ECM program activity is robust to the scoring method. "
        "Discrepant programs should be interpreted cautiously, as they may depend more strongly "
        "on absolute expression than on relative within-sample gene ranking."
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAVED] {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--program-table", type=Path, default=DEFAULT_PROGRAM_TABLE)
    parser.add_argument("--gtex-matrix", type=Path, default=DEFAULT_GTEX_MATRIX)
    parser.add_argument("--gtex-metadata", type=Path, default=DEFAULT_GTEX_METADATA)
    parser.add_argument("--gtex-mean-scores", type=Path, default=DEFAULT_GTEX_MEAN_SCORE_WITH_METADATA)
    parser.add_argument("--tabula-matrix", type=Path, default=DEFAULT_TABULA_MATRIX)
    parser.add_argument("--tabula-metadata", type=Path, default=DEFAULT_TABULA_METADATA)
    parser.add_argument("--tabula-mean-scores", type=Path, default=DEFAULT_TABULA_MEAN_SCORE_WITH_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    args = parser.parse_args()

    table_dir, html_dir, png_dir, report_dir = ensure_dirs(args.output_dir)

    programs = load_programs_from_curated_table(
        str(args.program_table),
        program_col="ecm_program_curated",
        genes_col="top_genes",
    )

    # Keep program order if possible.
    program_lookup = {program.name: program for program in programs}
    programs = [program_lookup[name] for name in PROGRAM_ORDER if name in program_lookup]

    pd.DataFrame(
        [
            {
                "ecm_program": program.name,
                "n_genes": program.n_genes,
                "genes": ", ".join(program.genes),
            }
            for program in programs
        ]
    ).to_csv(table_dir / "rank_based_reference_ecm_program_gene_sets.csv", index=False)

    # GTEx V11
    gtex_matrix = load_matrix(args.gtex_matrix, dataset_name="gtex_v11")
    gtex_metadata = load_metadata(args.gtex_metadata, id_col="sample_id", dataset_name="gtex_v11")

    gtex_scores, gtex_availability, _ = score_dataset(
        matrix=gtex_matrix,
        metadata=gtex_metadata,
        id_col="sample_id",
        programs=programs,
        dataset_name="gtex_v11",
    )

    gtex_scores.to_csv(table_dir / "gtex_v11_rank_based_program_scores_with_metadata.csv", index=False)
    gtex_availability.to_csv(table_dir / "gtex_v11_rank_based_program_gene_availability.csv", index=False)

    gtex_tissue_summary = summarize_by_group(
        scores_with_meta=gtex_scores,
        group_cols=["tissue"],
        id_col="sample_id",
        dataset_name="gtex_v11",
        score_method="rank_percentile",
    )

    gtex_tissue_detail_summary = summarize_by_group(
        scores_with_meta=gtex_scores,
        group_cols=["tissue", "tissue_detail"],
        id_col="sample_id",
        dataset_name="gtex_v11",
        score_method="rank_percentile",
    )

    gtex_tissue_summary.to_csv(table_dir / "gtex_v11_rank_based_tissue_summary.csv", index=False)
    gtex_tissue_detail_summary.to_csv(table_dir / "gtex_v11_rank_based_tissue_detail_summary.csv", index=False)

    gtex_corr = compare_rank_with_existing_mean_scores(
        rank_scores_with_meta=gtex_scores,
        mean_scores_with_meta_path=args.gtex_mean_scores,
        id_col="sample_id",
        dataset_name="gtex_v11",
    )
    gtex_corr.to_csv(table_dir / "gtex_v11_rank_vs_mean_score_correlation.csv", index=False)

    make_heatmap_from_summary(
        summary_df=gtex_tissue_summary,
        column_col="tissue",
        score_type="rank_percentile",
        title="GTEx V11 rank-based ECM program scores by tissue",
        name="gtex_v11_rank_based_tissue_program_heatmap",
        html_dir=html_dir,
        png_dir=png_dir,
    )

    # Tabula Sapiens
    tabula_matrix = load_matrix(args.tabula_matrix, dataset_name="tabula_sapiens")
    tabula_metadata = load_metadata(
        args.tabula_metadata,
        id_col="pseudobulk_id",
        dataset_name="tabula_sapiens",
    )

    tabula_scores, tabula_availability, _ = score_dataset(
        matrix=tabula_matrix,
        metadata=tabula_metadata,
        id_col="pseudobulk_id",
        programs=programs,
        dataset_name="tabula_sapiens",
    )

    tabula_scores.to_csv(
        table_dir / "tabula_sapiens_rank_based_program_scores_with_metadata.csv",
        index=False,
    )
    tabula_availability.to_csv(
        table_dir / "tabula_sapiens_rank_based_program_gene_availability.csv",
        index=False,
    )

    tabula_compartment_summary = summarize_by_group(
        scores_with_meta=tabula_scores,
        group_cols=["method", "compartment"],
        id_col="pseudobulk_id",
        dataset_name="tabula_sapiens",
        score_method="rank_percentile",
    )

    tabula_celltype_summary = summarize_by_group(
        scores_with_meta=tabula_scores,
        group_cols=["method", "cell_type", "compartment"],
        id_col="pseudobulk_id",
        dataset_name="tabula_sapiens",
        score_method="rank_percentile",
    )

    tabula_organ_celltype_summary = summarize_by_group(
        scores_with_meta=tabula_scores,
        group_cols=["method", "organ", "cell_type", "compartment"],
        id_col="pseudobulk_id",
        dataset_name="tabula_sapiens",
        score_method="rank_percentile",
    )

    tabula_compartment_summary.to_csv(
        table_dir / "tabula_sapiens_rank_based_compartment_summary.csv",
        index=False,
    )
    tabula_celltype_summary.to_csv(
        table_dir / "tabula_sapiens_rank_based_celltype_summary.csv",
        index=False,
    )
    tabula_organ_celltype_summary.to_csv(
        table_dir / "tabula_sapiens_rank_based_organ_celltype_summary.csv",
        index=False,
    )

    tabula_corr = compare_rank_with_existing_mean_scores(
        rank_scores_with_meta=tabula_scores,
        mean_scores_with_meta_path=args.tabula_mean_scores,
        id_col="pseudobulk_id",
        dataset_name="tabula_sapiens",
    )
    tabula_corr.to_csv(
        table_dir / "tabula_sapiens_rank_vs_mean_score_correlation.csv",
        index=False,
    )

    for method in sorted(tabula_compartment_summary["method"].dropna().astype(str).unique()):
        method_summary = tabula_compartment_summary[
            tabula_compartment_summary["method"].astype(str).eq(method)
        ]

        make_heatmap_from_summary(
            summary_df=method_summary,
            column_col="compartment",
            score_type="rank_percentile",
            title=f"Tabula Sapiens rank-based ECM program scores by compartment<br><sup>{method}</sup>",
            name=f"tabula_sapiens_rank_based_compartment_heatmap_{method}",
            html_dir=html_dir,
            png_dir=png_dir,
        )

    all_corr = pd.concat([gtex_corr, tabula_corr], ignore_index=True)
    all_corr.to_csv(table_dir / "rank_based_all_rank_vs_mean_score_correlation.csv", index=False)

    plot_rank_mean_correlation_heatmap(
        correlation_df=all_corr,
        html_dir=html_dir,
        png_dir=png_dir,
    )

    write_report(
        gtex_corr=gtex_corr,
        tabula_corr=tabula_corr,
        report_dir=report_dir,
    )

    print("\n[DONE]")
    print(f"Output folder: {args.output_dir}")
    print(f"Tables:        {table_dir}")
    print(f"HTML figures:  {html_dir}")
    print(f"PNG figures:   {png_dir}")


if __name__ == "__main__":
    main()