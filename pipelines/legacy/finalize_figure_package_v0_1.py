from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


FINAL_RESULTS_DIR = Path("outputs/final_results_v0.1")
EDA_DIR = Path("outputs/eda/rna_tissue_consensus")

REFINED_HTML_DIR = FINAL_RESULTS_DIR / "figures" / "refined" / "html"
REFINED_PNG_DIR = FINAL_RESULTS_DIR / "figures" / "refined" / "png"

FINAL_HTML_DIR = FINAL_RESULTS_DIR / "figures" / "final" / "html"
FINAL_PNG_DIR = FINAL_RESULTS_DIR / "figures" / "final" / "png"

DOCS_DIR = Path("docs")
REPORT_DIR = FINAL_RESULTS_DIR / "reports"


TISSUE_SYSTEM_MAP: Dict[str, str] = {
    "amygdala": "CNS",
    "basal ganglia": "CNS",
    "cerebellum": "CNS",
    "cerebral cortex": "CNS",
    "choroid plexus": "CNS",
    "hippocampal formation": "CNS",
    "hypothalamus": "CNS",
    "midbrain": "CNS",
    "retina": "CNS",
    "spinal cord": "CNS",

    "appendix": "Immune / lymphoid",
    "bone marrow": "Immune / lymphoid",
    "lymph node": "Immune / lymphoid",
    "spleen": "Immune / lymphoid",
    "thymus": "Immune / lymphoid",
    "tonsil": "Immune / lymphoid",

    "colon": "Digestive",
    "duodenum": "Digestive",
    "esophagus": "Digestive",
    "gallbladder": "Digestive",
    "liver": "Digestive",
    "pancreas": "Digestive",
    "rectum": "Digestive",
    "salivary gland": "Digestive",
    "small intestine": "Digestive",
    "stomach": "Digestive",
    "tongue": "Digestive",

    "breast": "Reproductive",
    "cervix": "Reproductive",
    "endometrium": "Reproductive",
    "epididymis": "Reproductive",
    "fallopian tube": "Reproductive",
    "ovary": "Reproductive",
    "placenta": "Reproductive",
    "prostate": "Reproductive",
    "seminal vesicle": "Reproductive",
    "testis": "Reproductive",
    "vagina": "Reproductive",

    "adipose tissue": "Connective / muscle / vascular",
    "blood vessel": "Connective / muscle / vascular",
    "heart muscle": "Connective / muscle / vascular",
    "skeletal muscle": "Connective / muscle / vascular",
    "smooth muscle": "Connective / muscle / vascular",
    "skin": "Connective / muscle / vascular",

    "adrenal gland": "Endocrine",
    "parathyroid gland": "Endocrine",
    "pituitary gland": "Endocrine",
    "thyroid gland": "Endocrine",

    "kidney": "Urinary",
    "urinary bladder": "Urinary",

    "lung": "Respiratory",
}

SELECTED_LABELS = {
    "cerebral cortex",
    "hippocampal formation",
    "cerebellum",
    "spinal cord",
    "bone marrow",
    "lymph node",
    "blood vessel",
    "heart muscle",
    "skin",
    "colon",
    "small intestine",
    "liver",
    "kidney",
    "lung",
    "retina",
}


def ensure_dirs() -> None:
    FINAL_HTML_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_PNG_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig: go.Figure, name: str, width: int = 1650, height: int = 850) -> None:
    html_path = FINAL_HTML_DIR / f"{name}.html"
    png_path = FINAL_PNG_DIR / f"{name}.png"

    fig.update_layout(width=width, height=height)

    fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)

    try:
        fig.write_image(str(png_path), scale=3)
        print(f"[SAVED] {png_path}")
    except Exception as exc:
        print(f"[WARNING] PNG export failed for {name}: {exc}")

    print(f"[SAVED] {html_path}")


def get_tissue_system(tissue: str) -> str:
    return TISSUE_SYSTEM_MAP.get(str(tissue).lower(), "Other")


def build_figure_2_variant(label_mode: str) -> go.Figure:
    pca_path = EDA_DIR / "pca_coordinates.csv"
    umap_path = EDA_DIR / "umap_coordinates.csv"
    explained_path = EDA_DIR / "pca_explained_variance.csv"

    if not pca_path.exists() or not umap_path.exists():
        raise FileNotFoundError("Missing PCA or UMAP coordinate files in outputs/eda/rna_tissue_consensus/")

    pca_df = pd.read_csv(pca_path, index_col=0)
    umap_df = pd.read_csv(umap_path, index_col=0)

    pca_df.insert(0, "sample", pca_df.index.astype(str))
    umap_df.insert(0, "sample", umap_df.index.astype(str))

    pca_df["tissue_system"] = pca_df["sample"].apply(get_tissue_system)
    umap_df["tissue_system"] = umap_df["sample"].apply(get_tissue_system)

    if label_mode == "clean":
        pca_df["label"] = ""
        umap_df["label"] = ""
        figure_title = "Figure 2. ECM-only expression preserves tissue organization"
    elif label_mode == "selected":
        pca_df["label"] = pca_df["sample"].apply(
            lambda x: x if str(x).lower() in SELECTED_LABELS else ""
        )
        umap_df["label"] = umap_df["sample"].apply(
            lambda x: x if str(x).lower() in SELECTED_LABELS else ""
        )
        figure_title = "Figure 2. ECM-only expression preserves tissue organization"
    elif label_mode == "all":
        pca_df["label"] = pca_df["sample"]
        umap_df["label"] = umap_df["sample"]
        figure_title = "Figure 2. ECM-only expression preserves tissue organization, fully labeled view"
    else:
        raise ValueError("label_mode must be one of: clean, selected, all")

    pc1_label = "PC1"
    pc2_label = "PC2"

    if explained_path.exists():
        explained_df = pd.read_csv(explained_path)
        if explained_df.shape[0] >= 2:
            pc1 = explained_df.loc[0, "explained_variance_ratio"] * 100
            pc2 = explained_df.loc[1, "explained_variance_ratio"] * 100
            pc1_label = f"PC1 ({pc1:.2f}% variance)"
            pc2_label = f"PC2 ({pc2:.2f}% variance)"

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("PCA of ECM-only tissue profiles", "UMAP of ECM-only tissue profiles"),
        horizontal_spacing=0.08,
    )

    tissue_systems = sorted(pca_df["tissue_system"].unique().tolist())

    for system in tissue_systems:
        pca_sub = pca_df[pca_df["tissue_system"].eq(system)]
        umap_sub = umap_df[umap_df["tissue_system"].eq(system)]

        fig.add_trace(
            go.Scatter(
                x=pca_sub["PC1"],
                y=pca_sub["PC2"],
                mode="markers+text" if label_mode != "clean" else "markers",
                text=pca_sub["label"],
                textposition="top center",
                textfont=dict(size=9),
                customdata=pca_sub[["sample", "tissue_system"]],
                hovertemplate=(
                    "Tissue: %{customdata[0]}<br>"
                    "System: %{customdata[1]}<br>"
                    "PC1: %{x:.3f}<br>"
                    "PC2: %{y:.3f}<extra></extra>"
                ),
                marker=dict(size=9, opacity=0.85),
                name=system,
                legendgroup=system,
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=umap_sub["UMAP1"],
                y=umap_sub["UMAP2"],
                mode="markers+text" if label_mode != "clean" else "markers",
                text=umap_sub["label"],
                textposition="top center",
                textfont=dict(size=9),
                customdata=umap_sub[["sample", "tissue_system"]],
                hovertemplate=(
                    "Tissue: %{customdata[0]}<br>"
                    "System: %{customdata[1]}<br>"
                    "UMAP1: %{x:.3f}<br>"
                    "UMAP2: %{y:.3f}<extra></extra>"
                ),
                marker=dict(size=9, opacity=0.85),
                name=system,
                legendgroup=system,
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    fig.update_xaxes(title_text=pc1_label, row=1, col=1)
    fig.update_yaxes(title_text=pc2_label, row=1, col=1)
    fig.update_xaxes(title_text="UMAP1", row=1, col=2)
    fig.update_yaxes(title_text="UMAP2", row=1, col=2)

    fig.update_layout(
        title=dict(text=figure_title, x=0.5, font=dict(size=22)),
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.22,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=80, r=40, t=90, b=160),
    )

    return fig


def generate_figure_2_variants() -> None:
    variants = {
        "figure_2_main_clean": "clean",
        "figure_2_selected_labels": "selected",
        "figure_2_all_labels": "all",
    }

    for file_name, label_mode in variants.items():
        fig = build_figure_2_variant(label_mode=label_mode)
        save_figure(fig, file_name, width=1650, height=850)


def build_qc_table() -> pd.DataFrame:
    qc_rows = [
        {
            "figure_id": "Figure 1",
            "panel": "",
            "file_name": "figure_1_workflow_refined.png",
            "folder": str(REFINED_PNG_DIR),
            "recommended_use": "main",
            "status": "",
            "problem": "",
            "required_fix": "",
        },
        {
            "figure_id": "Figure 2",
            "panel": "main",
            "file_name": "figure_2_main_clean.png",
            "folder": str(FINAL_PNG_DIR),
            "recommended_use": "main",
            "status": "",
            "problem": "",
            "required_fix": "",
        },
        {
            "figure_id": "Figure 2",
            "panel": "selected_labels",
            "file_name": "figure_2_selected_labels.png",
            "folder": str(FINAL_PNG_DIR),
            "recommended_use": "supplementary_or_internal",
            "status": "",
            "problem": "",
            "required_fix": "",
        },
        {
            "figure_id": "Figure 2",
            "panel": "all_labels",
            "file_name": "figure_2_all_labels.png",
            "folder": str(FINAL_PNG_DIR),
            "recommended_use": "inspection_only",
            "status": "",
            "problem": "",
            "required_fix": "",
        },
        {
            "figure_id": "Figure 3",
            "panel": "",
            "file_name": "figure_3_ecm_vs_random_non_ecm_refined.png",
            "folder": str(REFINED_PNG_DIR),
            "recommended_use": "main",
            "status": "",
            "problem": "",
            "required_fix": "",
        },
        {
            "figure_id": "Figure 4",
            "panel": "",
            "file_name": "figure_4_matrisome_category_benchmark_heatmap_refined.png",
            "folder": str(REFINED_PNG_DIR),
            "recommended_use": "main",
            "status": "",
            "problem": "",
            "required_fix": "",
        },
        {
            "figure_id": "Figure 5A",
            "panel": "",
            "file_name": "figure_5a_curated_ecm_program_presence_refined.png",
            "folder": str(REFINED_PNG_DIR),
            "recommended_use": "main",
            "status": "",
            "problem": "",
            "required_fix": "",
        },
        {
            "figure_id": "Figure 5B",
            "panel": "",
            "file_name": "figure_5b_curated_ecm_program_high_confidence_refined.png",
            "folder": str(REFINED_PNG_DIR),
            "recommended_use": "main_or_supplementary",
            "status": "",
            "problem": "",
            "required_fix": "",
        },
        {
            "figure_id": "Figure 6A",
            "panel": "",
            "file_name": "figure_6a_external_dataset_reproducibility_refined.png",
            "folder": str(REFINED_PNG_DIR),
            "recommended_use": "main",
            "status": "",
            "problem": "",
            "required_fix": "",
        },
        {
            "figure_id": "Figure 6B",
            "panel": "",
            "file_name": "figure_6b_external_feature_set_reproducibility_refined.png",
            "folder": str(REFINED_PNG_DIR),
            "recommended_use": "main_or_supplementary",
            "status": "",
            "problem": "",
            "required_fix": "",
        },
    ]

    df = pd.DataFrame(qc_rows)

    statuses = []
    problems = []
    fixes = []

    for row in df.itertuples(index=False):
        path = Path(row.folder) / row.file_name
        if path.exists():
            statuses.append("exists")
            problems.append("")
            fixes.append("")
        else:
            statuses.append("missing")
            problems.append("file not found")
            fixes.append("regenerate the figure")
    df["status"] = statuses
    df["problem"] = problems
    df["required_fix"] = fixes

    return df


def write_qc_report(df: pd.DataFrame) -> None:
    csv_path = REPORT_DIR / "figure_qc_v0.1.csv"
    md_path_1 = REPORT_DIR / "figure_qc_v0.1.md"
    md_path_2 = DOCS_DIR / "figure_qc_v0.1.md"

    df.to_csv(csv_path, index=False)

    lines: List[str] = []
    lines.append("# Figure QC v0.1\n")
    lines.append("## Purpose\n")
    lines.append("This report freezes the current manuscript figure package and records which figure versions should be used in the main manuscript, supplementary material, or internal inspection.\n")

    lines.append("## Recommended figure package\n")
    lines.append("- Figure 1: `figure_1_workflow_refined.png`\n")
    lines.append("- Figure 2 (main): `figure_2_main_clean.png`\n")
    lines.append("- Figure 2 (supplementary / discussion): `figure_2_selected_labels.png`\n")
    lines.append("- Figure 2 (inspection only): `figure_2_all_labels.png`\n")
    lines.append("- Figure 3: `figure_3_ecm_vs_random_non_ecm_refined.png`\n")
    lines.append("- Figure 4: `figure_4_matrisome_category_benchmark_heatmap_refined.png`\n")
    lines.append("- Figure 5A: `figure_5a_curated_ecm_program_presence_refined.png`\n")
    lines.append("- Figure 5B: `figure_5b_curated_ecm_program_high_confidence_refined.png`\n")
    lines.append("- Figure 6A: `figure_6a_external_dataset_reproducibility_refined.png`\n")
    lines.append("- Figure 6B: `figure_6b_external_feature_set_reproducibility_refined.png`\n")

    lines.append("## QC table\n")
    lines.append(df.to_markdown(index=False))
    lines.append("\n")

    lines.append("## Notes\n")
    lines.append("1. The recommended main-manuscript version of Figure 2 is the clean no-label version.\n")
    lines.append("2. The selected-label Figure 2 is useful for discussion or supplementary material.\n")
    lines.append("3. The fully labeled Figure 2 is mainly for inspection and should not be used as the primary manuscript figure.\n")
    lines.append("4. If all files exist, the figure package is ready for the next scientific step, which is MatrisomeDB proteomics validation.\n")

    text = "\n".join(lines)

    md_path_1.write_text(text, encoding="utf-8")
    md_path_2.write_text(text, encoding="utf-8")

    print(f"[SAVED] {csv_path}")
    print(f"[SAVED] {md_path_1}")
    print(f"[SAVED] {md_path_2}")


def main() -> None:
    ensure_dirs()
    generate_figure_2_variants()
    qc_df = build_qc_table()
    write_qc_report(qc_df)

    print("\n[DONE]")
    print(f"Final HTML folder: {FINAL_HTML_DIR}")
    print(f"Final PNG folder:  {FINAL_PNG_DIR}")
    print(f"QC report:         {REPORT_DIR / 'figure_qc_v0.1.md'}")


if __name__ == "__main__":
    main()