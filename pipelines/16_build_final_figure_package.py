from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


DEFAULT_OUTPUT_DIR = Path("results/final_figure_package")


FIGURE_PLAN = [
    {
        "figure_id": "Figure 1",
        "short_name": "figure_1_workflow",
        "title": "Workflow of the Matrisome-derived ECM representation framework",
        "purpose": "Summarize the full framework from Matrisome filtering to transcriptomic, proteomic, donor-level, cell-type, rank-based, and stability validation.",
        "source_png_candidates": "",
        "source_html_candidates": "",
        "placement": "main",
        "status": "generate",
    },
    {
        "figure_id": "Figure 2",
        "short_name": "figure_2_ecm_tissue_organization",
        "title": "ECM-only expression preserves tissue organization",
        "purpose": "Show PCA/UMAP organization of tissues using Matrisome-filtered expression.",
        "source_png_candidates": "outputs/final_results_v0.1/figures/final/png/figure_2_main_clean.png; outputs/final_results_v0.1/figures/refined/png/figure_2_ecm_only_tissue_space_refined.png; results/final_results_v0.1/figures/final/png/figure_2_main_clean.png",
        "source_html_candidates": "outputs/final_results_v0.1/figures/final/html/figure_2_main_clean.html; outputs/final_results_v0.1/figures/refined/html/figure_2_ecm_only_tissue_space_refined.html; results/final_results_v0.1/figures/final/html/figure_2_main_clean.html",
        "placement": "main",
        "status": "copy_if_available",
    },
    {
        "figure_id": "Figure 3",
        "short_name": "figure_3_matrisome_vs_random",
        "title": "Matrisome genes outperform matched random non-ECM genes for global tissue organization",
        "purpose": "Show specificity of Matrisome genes against matched random non-ECM baselines.",
        "source_png_candidates": "outputs/final_results_v0.1/figures/refined/png/figure_3_ecm_vs_random_non_ecm_refined.png; results/final_results_v0.1/figures/refined/png/figure_3_ecm_vs_random_non_ecm_refined.png",
        "source_html_candidates": "outputs/final_results_v0.1/figures/refined/html/figure_3_ecm_vs_random_non_ecm_refined.html; results/final_results_v0.1/figures/refined/html/figure_3_ecm_vs_random_non_ecm_refined.html",
        "placement": "main",
        "status": "copy_if_available",
    },
    {
        "figure_id": "Figure 4",
        "short_name": "figure_4_matrisome_category_benchmark",
        "title": "Matrisome category benchmark against matched random genes",
        "purpose": "Compare Matrisome categories and identify core Matrisome and ECM glycoproteins as strong broad representation spaces.",
        "source_png_candidates": "outputs/final_results_v0.1/figures/refined/png/figure_4_matrisome_category_benchmark_heatmap_refined.png; results/final_results_v0.1/figures/refined/png/figure_4_matrisome_category_benchmark_heatmap_refined.png",
        "source_html_candidates": "outputs/final_results_v0.1/figures/refined/html/figure_4_matrisome_category_benchmark_heatmap_refined.html; results/final_results_v0.1/figures/refined/html/figure_4_matrisome_category_benchmark_heatmap_refined.html",
        "placement": "main",
        "status": "copy_if_available",
    },
    {
        "figure_id": "Figure 5",
        "short_name": "figure_5_curated_ecm_programs",
        "title": "Curated recurring ECM programs across Matrisome feature sets",
        "purpose": "Show the nine curated NMF-derived ECM programs across core Matrisome, glycoproteins, proteoglycans, and collagens.",
        "source_png_candidates": "outputs/final_results_v0.1/figures/refined/png/figure_5a_curated_ecm_program_presence_refined.png; results/final_results_v0.1/figures/refined/png/figure_5a_curated_ecm_program_presence_refined.png",
        "source_html_candidates": "outputs/final_results_v0.1/figures/refined/html/figure_5a_curated_ecm_program_presence_refined.html; results/final_results_v0.1/figures/refined/html/figure_5a_curated_ecm_program_presence_refined.html",
        "placement": "main",
        "status": "copy_if_available",
    },
    {
        "figure_id": "Figure 6",
        "short_name": "figure_6_external_reproducibility",
        "title": "External transcriptomic reproducibility of ECM programs",
        "purpose": "Show that curated ECM programs reproduce outside the reference dataset.",
        "source_png_candidates": "outputs/final_results_v0.1/figures/refined/png/figure_6a_external_dataset_reproducibility_refined.png; results/final_results_v0.1/figures/refined/png/figure_6a_external_dataset_reproducibility_refined.png",
        "source_html_candidates": "outputs/final_results_v0.1/figures/refined/html/figure_6a_external_dataset_reproducibility_refined.html; results/final_results_v0.1/figures/refined/html/figure_6a_external_dataset_reproducibility_refined.html",
        "placement": "main",
        "status": "copy_if_available",
    },
    {
        "figure_id": "Figure 7",
        "short_name": "figure_7_matrisomedb_validation",
        "title": "MatrisomeDB protein-level validation and random-null support",
        "purpose": "Show MatrisomeDB gene detection, abundance support, and R3 random Matrisome null validation.",
        "source_png_candidates": "outputs/matrisomedb_validation/summary/figures/png/matrisomedb_gene_detection_coverage_clean.png; results/revision_matrisomedb_null_abundance/figures/png/r3_detection_null_all_samples.png; results/revision_matrisomedb_null_abundance/figures/png/r3_abundance_null_all_samples_top3_mean_log_nsaf.png; results/final_results_v0.2/figures/matrisomedb/png/figure_7a_matrisomedb_gene_detection_coverage.png",
        "source_html_candidates": "outputs/matrisomedb_validation/summary/figures/html/matrisomedb_gene_detection_coverage_clean.html; results/revision_matrisomedb_null_abundance/figures/html/r3_detection_null_all_samples.html; results/revision_matrisomedb_null_abundance/figures/html/r3_abundance_null_all_samples_top3_mean_log_nsaf.html",
        "placement": "main",
        "status": "copy_if_available",
    },
    {
        "figure_id": "Figure 8",
        "short_name": "figure_8_gtex_v11_validation",
        "title": "GTEx V11 sample-level ECM program validation",
        "purpose": "Show tissue-level ECM program enrichment and sample-level structure across GTEx V11 donor samples.",
        "source_png_candidates": "outputs/gtex_v11_sample_level_validation/figures/png/gtex_v11_tissue_program_row_zscore_heatmap.png; results/final_results_v0.4/figures/gtex_v11/png/figure_8b_gtex_v11_tissue_program_row_zscore_heatmap.png",
        "source_html_candidates": "outputs/gtex_v11_sample_level_validation/figures/html/gtex_v11_tissue_program_row_zscore_heatmap.html; results/final_results_v0.4/figures/gtex_v11/html/figure_8b_gtex_v11_tissue_program_row_zscore_heatmap.html",
        "placement": "main",
        "status": "copy_if_available",
    },
    {
        "figure_id": "Figure 9",
        "short_name": "figure_9_donor_stratified_classification",
        "title": "Donor-stratified classification using nine ECM program scores",
        "purpose": "Show that ECM program scores classify tissue and tissue-system labels under donor-held-out validation.",
        "source_png_candidates": "outputs/gtex_v11_sample_level_validation/classification/figures/png/gtex_v11_donor_stratified_classification_performance.png; results/final_results_v0.4/figures/gtex_v11/png/figure_9a_gtex_v11_donor_stratified_classification_performance.png",
        "source_html_candidates": "outputs/gtex_v11_sample_level_validation/classification/figures/html/gtex_v11_donor_stratified_classification_performance.html; results/final_results_v0.4/figures/gtex_v11/html/figure_9a_gtex_v11_donor_stratified_classification_performance.html",
        "placement": "main",
        "status": "copy_if_available",
    },
    {
        "figure_id": "Figure 10",
        "short_name": "figure_10_representation_controls",
        "title": "Representation-control benchmark",
        "purpose": "Compare curated ECM programs against Matrisome PCA9, random Matrisome programs, and random non-ECM programs.",
        "source_png_candidates": "results/revision_classification_controls/figures/png/r2_representation_family_benchmark_tissue_system_xgboost.png; results/revision_classification_controls/figures/png/r2_representation_benchmark_tissue_system_xgboost.png",
        "source_html_candidates": "results/revision_classification_controls/figures/html/r2_representation_family_benchmark_tissue_system_xgboost.html; results/revision_classification_controls/figures/html/r2_representation_benchmark_tissue_system_xgboost.html",
        "placement": "main",
        "status": "copy_if_available",
    },
    {
        "figure_id": "Figure 11",
        "short_name": "figure_11_deep_tabular_benchmark",
        "title": "Deep and tabular model benchmark",
        "purpose": "Benchmark dense MLP, residual MLP, FT-Transformer-lite, TabNet, TabICL, and classical models.",
        "source_png_candidates": "outputs/gtex_v11_sample_level_validation/deep_tabular_benchmark/figures/png/gtex_v11_deep_tabular_benchmark_balanced_accuracy.png; results/final_results_v0.4/figures/gtex_v11/png/figure_11_gtex_v11_deep_tabular_benchmark_balanced_accuracy.png",
        "source_html_candidates": "outputs/gtex_v11_sample_level_validation/deep_tabular_benchmark/figures/html/gtex_v11_deep_tabular_benchmark_balanced_accuracy.html; results/final_results_v0.4/figures/gtex_v11/html/figure_11_gtex_v11_deep_tabular_benchmark_balanced_accuracy.html",
        "placement": "main_or_supplementary",
        "status": "copy_if_available",
    },
    {
        "figure_id": "Figure 12",
        "short_name": "figure_12_tabula_sapiens_sources",
        "title": "Tabula Sapiens cell-type source validation",
        "purpose": "Show likely cellular sources of ECM programs across stromal, endothelial, epithelial, immune, and support compartments.",
        "source_png_candidates": "outputs/tabula_sapiens_pseudobulk/final_route_b/figures/png/route_b_compartment_program_heatmap_10X.png; results/final_results_v0.5/figures/tabula_sapiens/png/figure_12a_route_b_compartment_program_heatmap_10X.png",
        "source_html_candidates": "outputs/tabula_sapiens_pseudobulk/final_route_b/figures/html/route_b_compartment_program_heatmap_10X.html; results/final_results_v0.5/figures/tabula_sapiens/html/figure_12a_route_b_compartment_program_heatmap_10X.html",
        "placement": "main",
        "status": "copy_if_available",
    },
    {
        "figure_id": "Figure 13",
        "short_name": "figure_13_rank_based_scoring",
        "title": "Rank-based scoring robustness",
        "purpose": "Show agreement between rank-based and mean-expression ECM program scoring.",
        "source_png_candidates": "outputs/rank_based_ecm_program_scoring/figures/png/rank_vs_mean_score_spearman_correlation.png; results/final_results_v0.6/figures/rank_based_scoring/png/figure_13a_rank_vs_mean_score_spearman_correlation.png",
        "source_html_candidates": "outputs/rank_based_ecm_program_scoring/figures/html/rank_vs_mean_score_spearman_correlation.html; results/final_results_v0.6/figures/rank_based_scoring/html/figure_13a_rank_vs_mean_score_spearman_correlation.html",
        "placement": "main_or_supplementary",
        "status": "copy_if_available",
    },
    {
        "figure_id": "Figure 14",
        "short_name": "figure_14_nmf_rank_stability",
        "title": "NMF rank and stability analysis",
        "purpose": "Show rank selection, component stability, and program recovery across NMF ranks and seeds.",
        "source_png_candidates": "results/revision_nmf_stability/figures/png/nmf_curated_program_recovery_fraction.png; results/revision_nmf_stability/figures/png/nmf_top_gene_jaccard_stability.png; results/revision_nmf_stability/figures/png/nmf_normalized_reconstruction_error.png",
        "source_html_candidates": "results/revision_nmf_stability/figures/html/nmf_curated_program_recovery_fraction.html; results/revision_nmf_stability/figures/html/nmf_top_gene_jaccard_stability.html; results/revision_nmf_stability/figures/html/nmf_normalized_reconstruction_error.html",
        "placement": "main_or_supplementary",
        "status": "copy_if_available",
    },
]


def ensure_dirs(output_dir: Path) -> tuple[Path, Path, Path]:
    png_dir = output_dir / "figures" / "png"
    html_dir = output_dir / "figures" / "html"
    report_dir = output_dir / "reports"
    table_dir = output_dir / "tables"

    for folder in [png_dir, html_dir, report_dir, table_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    return png_dir, html_dir, report_dir, table_dir


def resolve_candidate(
    candidates: str,
    old_root: Path | None,
) -> Path | None:
    if not candidates:
        return None

    for item in [x.strip() for x in candidates.split(";") if x.strip()]:
        path = Path(item)

        if path.exists():
            return path

        if old_root is not None:
            old_path = old_root / item
            if old_path.exists():
                return old_path

    return None


def copy_if_available(src: Path | None, dst: Path) -> str:
    if src is None:
        return "missing"

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return "copied"


def make_workflow_figure(output_dir: Path) -> tuple[Path, Path]:
    png_dir = output_dir / "figures" / "png"
    html_dir = output_dir / "figures" / "html"

    steps = [
        ("Inputs", "HPA, GTEx, Matrisome,<br>MatrisomeDB, Tabula Sapiens"),
        ("ECM matrices", "Matrisome filtering<br>sample × gene matrices"),
        ("Program discovery", "NMF modules<br>curated ECM programs"),
        ("Core validation", "random controls<br>external reproducibility"),
        ("Protein support", "MatrisomeDB detection<br>and NSAF nulls"),
        ("Donor-level validation", "GTEx V11<br>classification benchmarks"),
        ("Cell sources", "Tabula Sapiens<br>pseudobulk mapping"),
        ("Robustness", "rank scoring<br>NMF stability"),
    ]

    fig = go.Figure()

    for i, (title, subtitle) in enumerate(steps):
        x = i
        fig.add_shape(
            type="rect",
            x0=x - 0.42,
            x1=x + 0.42,
            y0=-0.25,
            y1=0.25,
            line=dict(width=2),
            fillcolor="white",
        )
        fig.add_annotation(
            x=x,
            y=0.08,
            text=f"<b>{title}</b>",
            showarrow=False,
            font=dict(size=14),
        )
        fig.add_annotation(
            x=x,
            y=-0.10,
            text=subtitle,
            showarrow=False,
            font=dict(size=10),
        )

        if i < len(steps) - 1:
            fig.add_annotation(
                x=x + 0.78,
                y=0,
                ax=x + 0.45,
                ay=0,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowwidth=2,
            )

    fig.update_xaxes(visible=False, range=[-0.7, len(steps) - 0.3])
    fig.update_yaxes(visible=False, range=[-0.55, 0.55])
    fig.update_layout(
        title=dict(
            text="Figure 1. Workflow of the Matrisome-derived ECM representation framework",
            x=0.5,
            font=dict(size=22),
        ),
        template="plotly_white",
        width=2050,
        height=520,
        margin=dict(l=30, r=30, t=80, b=40),
    )

    html_path = html_dir / "figure_1_workflow.html"
    png_path = png_dir / "figure_1_workflow.png"

    fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)

    try:
        fig.write_image(str(png_path), scale=3)
    except Exception as exc:
        print(f"[WARNING] Could not export workflow PNG: {exc}")

    return png_path, html_path


def write_legends(report_dir: Path, manifest: pd.DataFrame) -> None:
    lines = []
    lines.append("# Figure legends\n")

    for row in manifest.itertuples():
        lines.append(f"## {row.figure_id}. {row.title}\n")
        lines.append(f"{row.purpose}\n")
        lines.append(f"**Placement:** {row.placement}\n")
        lines.append(f"**PNG status:** {row.png_status}\n")
        lines.append(f"**HTML status:** {row.html_status}\n")

    path = report_dir / "figure_legends_v1.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAVED] {path}")


def write_qc_report(report_dir: Path, manifest: pd.DataFrame) -> None:
    missing = manifest[
        (manifest["png_status"].eq("missing"))
        & (manifest["html_status"].eq("missing"))
    ]

    lines = []
    lines.append("# Final figure package QC report\n")
    lines.append(f"- Figures planned: {manifest.shape[0]}\n")
    lines.append(f"- Figures with no available PNG or HTML: {missing.shape[0]}\n")

    if not missing.empty:
        lines.append("## Missing figures\n")
        for row in missing.itertuples():
            lines.append(f"- {row.figure_id}: {row.title}\n")

    lines.append("## Recommendation\n")
    lines.append(
        "Figures marked as missing should either be regenerated or moved to supplementary material before manuscript submission."
    )

    path = report_dir / "figure_qc_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAVED] {path}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--old-root",
        type=Path,
        default=None,
        help="Optional old project root, e.g. E:/Projects/ECM/ecm_latent_space",
    )

    args = parser.parse_args()

    png_dir, html_dir, report_dir, table_dir = ensure_dirs(args.output_dir)

    records = []

    for figure in FIGURE_PLAN:
        figure_id = figure["figure_id"]
        short_name = figure["short_name"]

        if figure["status"] == "generate" and figure_id == "Figure 1":
            png_src, html_src = make_workflow_figure(args.output_dir)
            png_status = "generated"
            html_status = "generated"
            png_final = png_src
            html_final = html_src

        else:
            png_src = resolve_candidate(figure["source_png_candidates"], args.old_root)
            html_src = resolve_candidate(figure["source_html_candidates"], args.old_root)

            png_final = png_dir / f"{short_name}.png"
            html_final = html_dir / f"{short_name}.html"

            png_status = copy_if_available(png_src, png_final)
            html_status = copy_if_available(html_src, html_final)

        records.append(
            {
                "figure_id": figure_id,
                "short_name": short_name,
                "title": figure["title"],
                "purpose": figure["purpose"],
                "placement": figure["placement"],
                "png_status": png_status,
                "html_status": html_status,
                "png_path": str(png_final) if png_status != "missing" else "",
                "html_path": str(html_final) if html_status != "missing" else "",
                "source_png_candidates": figure["source_png_candidates"],
                "source_html_candidates": figure["source_html_candidates"],
            }
        )

    manifest = pd.DataFrame(records)
    manifest_path = table_dir / "final_figure_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"[SAVED] {manifest_path}")

    write_legends(report_dir, manifest)
    write_qc_report(report_dir, manifest)

    print("\n[DONE]")
    print(f"Final figure package: {args.output_dir}")


if __name__ == "__main__":
    main()