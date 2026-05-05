from __future__ import annotations

from pathlib import Path
import pandas as pd


FINAL_DIR = Path("outputs/final_results_v0.1")
TABLE_DIR = FINAL_DIR / "tables"
FIGURE_DIR = FINAL_DIR / "figures"
REPORT_DIR = FINAL_DIR / "reports"
DOCS_DIR = Path("docs")


FIGURE_PLAN = [
    {
        "figure_id": "Figure 1",
        "title": "Workflow of the Matrisome-derived ECM representation framework",
        "purpose": (
            "Show the full computational workflow: HPA/GTEx-derived expression data, "
            "Human Matrisome filtering, ECM matrices, random non-ECM controls, "
            "category benchmarking, latent embeddings, NMF module curation, and external reproducibility."
        ),
        "source_files": "Needs to be created manually or scripted as a schematic.",
        "status": "needs_schematic",
        "notes": "This should be a schematic, not a data plot. Can be made in BioRender, PowerPoint, Illustrator, or generated with Python/Graphviz.",
    },
    {
        "figure_id": "Figure 2",
        "title": "ECM-only tissue organization",
        "purpose": (
            "Show that Matrisome-filtered expression preserves biological tissue organization, "
            "including CNS, immune/lymphoid, epithelial/mucosal, and stromal/connective neighborhoods."
        ),
        "source_files": (
            "figures/eda_tissue_consensus_pca.png; "
            "figures/eda_tissue_consensus_umap.png; "
            "figures/eda_tissue_consensus_heatmap.png"
        ),
        "status": "available",
        "notes": "Use PCA and UMAP as main panels. Heatmap can be supplementary if too dense.",
    },
    {
        "figure_id": "Figure 3",
        "title": "Matrisome genes versus matched random non-ECM genes",
        "purpose": (
            "Show that Matrisome genes outperform matched random non-ECM gene sets "
            "for global tissue-organization metrics."
        ),
        "source_files": "tables/Table_2_ecm_vs_random_specificity.csv",
        "status": "needs_plot_generation",
        "notes": "Generate bar plots or point-range plots for ECM value vs random mean and z-score.",
    },
    {
        "figure_id": "Figure 4",
        "title": "Matrisome category benchmark",
        "purpose": (
            "Compare all Matrisome, core Matrisome, ECM glycoproteins, collagens, "
            "proteoglycans, and other Matrisome categories against matched random baselines."
        ),
        "source_files": "tables/Table_3_matrisome_category_benchmark.csv",
        "status": "needs_plot_generation",
        "notes": "Focus on PCA PC1+PC2 variance and silhouette PCA10. Consider a heatmap of z-scores.",
    },
    {
        "figure_id": "Figure 5",
        "title": "Curated NMF-derived ECM programs",
        "purpose": (
            "Show the nine curated recurring ECM programs derived from NMF modules across "
            "core Matrisome, ECM glycoproteins, proteoglycans, and collagens."
        ),
        "source_files": (
            "tables/Table_4_curated_recurring_ecm_programs.csv; "
            "figures/curated_program_presence_heatmap.html; "
            "figures/curated_program_high_confidence_heatmap.html"
        ),
        "status": "available_html_needs_static_export",
        "notes": "Use curated program presence matrix as the main figure. Export Plotly HTML to PNG/SVG later.",
    },
    {
        "figure_id": "Figure 6",
        "title": "External reproducibility of ECM programs",
        "purpose": (
            "Show that reference-defined ECM programs reproduce across external datasets "
            "after excluding rna_tissue_consensus."
        ),
        "source_files": (
            "tables/Table_5_external_program_reproducibility.csv; "
            "figures/external_dataset_program_presence_heatmap.html; "
            "figures/external_feature_set_program_presence_heatmap.html"
        ),
        "status": "available_html_needs_static_export",
        "notes": "This is one of the strongest figures. It supports non-circular reproducibility.",
    },
]


TABLE_PLAN = [
    {
        "table_id": "Table 1",
        "title": "Processed datasets and Matrisome gene matching",
        "source_file": "tables/Table_1_dataset_summary.csv",
        "purpose": "Summarize datasets, sample counts, labels, and Matrisome gene matching.",
        "status": "available",
        "placement": "main",
    },
    {
        "table_id": "Table 2",
        "title": "Matrisome versus matched random non-ECM specificity metrics",
        "source_file": "tables/Table_2_ecm_vs_random_specificity.csv",
        "purpose": "Show ECM-specificity test metrics, including PCA variance and silhouette scores.",
        "status": "available",
        "placement": "main_or_supplementary",
    },
    {
        "table_id": "Table 3",
        "title": "Matrisome category benchmark",
        "source_file": "tables/Table_3_matrisome_category_benchmark.csv",
        "purpose": "Compare Matrisome categories against matched random baselines.",
        "status": "available",
        "placement": "main_or_supplementary",
    },
    {
        "table_id": "Table 4",
        "title": "Curated recurring ECM programs",
        "source_file": "tables/Table_4_curated_recurring_ecm_programs.csv",
        "purpose": "Summarize curated ECM programs, number of modules, feature sets, and representative genes.",
        "status": "available",
        "placement": "main",
    },
    {
        "table_id": "Table 5",
        "title": "External reproducibility of ECM programs",
        "source_file": "tables/Table_5_external_program_reproducibility.csv",
        "purpose": "Show which ECM programs reproduce after excluding the reference tissue-consensus dataset.",
        "status": "available",
        "placement": "main",
    },
    {
        "table_id": "Supplementary Table 1",
        "title": "Curated NMF module annotations",
        "source_file": "tables/Supplementary_Table_curated_nmf_modules.csv",
        "purpose": "Provide all curated NMF modules, top samples, top genes, confidence, and interpretation.",
        "status": "available",
        "placement": "supplementary",
    },
]


def check_source_exists(source: str) -> str:
    if source.startswith("Needs to be created"):
        return "not_applicable"

    paths = [item.strip() for item in source.split(";")]

    statuses = []

    for path_str in paths:
        path = FINAL_DIR / path_str

        if path.exists():
            statuses.append("exists")
        else:
            statuses.append(f"missing:{path_str}")

    if all(status == "exists" for status in statuses):
        return "exists"

    return " | ".join(statuses)


def create_plan_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    figure_df = pd.DataFrame(FIGURE_PLAN)
    table_df = pd.DataFrame(TABLE_PLAN)

    figure_df["source_check"] = figure_df["source_files"].apply(check_source_exists)
    table_df["source_check"] = table_df["source_file"].apply(check_source_exists)

    return figure_df, table_df


def write_markdown_report(figure_df: pd.DataFrame, table_df: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    lines = []

    lines.append("# Manuscript Figure and Table Plan v0.1\n")
    lines.append("## Project framing\n")
    lines.append(
        "Working title: **A reproducible Matrisome-derived representation framework for mapping human tissue ECM signatures**\n"
    )
    lines.append(
        "This manuscript should be framed as a reproducible ECM representation and benchmarking framework, not as a foundation model or direct scaffold inverse-design system.\n"
    )

    lines.append("## Proposed main figures\n")

    for row in figure_df.itertuples(index=False):
        lines.append(f"### {row.figure_id}: {row.title}\n")
        lines.append(f"**Purpose:** {row.purpose}\n")
        lines.append(f"**Source files:** `{row.source_files}`\n")
        lines.append(f"**Status:** {row.status}\n")
        lines.append(f"**Source check:** {row.source_check}\n")
        lines.append(f"**Notes:** {row.notes}\n")

    lines.append("## Proposed tables\n")

    for row in table_df.itertuples(index=False):
        lines.append(f"### {row.table_id}: {row.title}\n")
        lines.append(f"**Purpose:** {row.purpose}\n")
        lines.append(f"**Source file:** `{row.source_file}`\n")
        lines.append(f"**Placement:** {row.placement}\n")
        lines.append(f"**Status:** {row.status}\n")
        lines.append(f"**Source check:** {row.source_check}\n")

    lines.append("## Immediate next actions\n")
    lines.append("1. Generate static manuscript-ready plots for Figures 3 and 4 from CSV tables.\n")
    lines.append("2. Export Plotly HTML heatmaps for Figures 5 and 6 to PNG or SVG.\n")
    lines.append("3. Create Figure 1 as a workflow schematic.\n")
    lines.append("4. Decide which large tables should stay in the main manuscript and which should move to supplementary material.\n")
    lines.append("5. After figure planning is stable, move to MatrisomeDB proteomics validation.\n")

    report = "\n".join(lines)

    report_paths = [
        REPORT_DIR / "manuscript_figure_table_plan_v0.1.md",
        DOCS_DIR / "manuscript_figure_table_plan_v0.1.md",
    ]

    for path in report_paths:
        path.write_text(report, encoding="utf-8")
        print(f"[SAVED] {path}")


def main() -> None:
    if not FINAL_DIR.exists():
        raise FileNotFoundError(
            f"Final results folder not found: {FINAL_DIR}. "
            "Run src/build_final_results_package.py first."
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    figure_df, table_df = create_plan_tables()

    figure_df.to_csv(
        REPORT_DIR / "manuscript_figure_plan_v0.1.csv",
        index=False,
    )

    table_df.to_csv(
        REPORT_DIR / "manuscript_table_plan_v0.1.csv",
        index=False,
    )

    write_markdown_report(figure_df=figure_df, table_df=table_df)

    print("\n[DONE]")
    print(REPORT_DIR / "manuscript_figure_plan_v0.1.csv")
    print(REPORT_DIR / "manuscript_table_plan_v0.1.csv")
    print(REPORT_DIR / "manuscript_figure_table_plan_v0.1.md")


if __name__ == "__main__":
    main()