from __future__ import annotations

from pathlib import Path
import math
import re
from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


FINAL_DIR = Path("outputs/final_results_v0.1")
TABLE_DIR = FINAL_DIR / "tables"

FIGURE_DIR = FINAL_DIR / "figures" / "refined"
HTML_DIR = FIGURE_DIR / "html"
PNG_DIR = FIGURE_DIR / "png"

EDA_DIR = Path("outputs/eda/rna_tissue_consensus")


TISSUE_SYSTEM_MAP: Dict[str, str] = {
    # CNS
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

    # Immune / lymphoid
    "appendix": "Immune / lymphoid",
    "bone marrow": "Immune / lymphoid",
    "lymph node": "Immune / lymphoid",
    "spleen": "Immune / lymphoid",
    "thymus": "Immune / lymphoid",
    "tonsil": "Immune / lymphoid",

    # Digestive
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

    # Reproductive
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

    # Connective / muscle / vascular
    "adipose tissue": "Connective / muscle / vascular",
    "blood vessel": "Connective / muscle / vascular",
    "heart muscle": "Connective / muscle / vascular",
    "skeletal muscle": "Connective / muscle / vascular",
    "smooth muscle": "Connective / muscle / vascular",
    "skin": "Connective / muscle / vascular",

    # Endocrine
    "adrenal gland": "Endocrine",
    "parathyroid gland": "Endocrine",
    "pituitary gland": "Endocrine",
    "thyroid gland": "Endocrine",

    # Urinary
    "kidney": "Urinary",
    "urinary bladder": "Urinary",

    # Respiratory
    "lung": "Respiratory",
}


REPRESENTATIVE_TISSUE_LABELS = {
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
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig: go.Figure, name: str, width: int, height: int) -> None:
    html_path = HTML_DIR / f"{name}.html"
    png_path = PNG_DIR / f"{name}.png"

    fig.update_layout(width=width, height=height)

    fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)

    try:
        fig.write_image(str(png_path), scale=3)
        print(f"[SAVED] {png_path}")
    except Exception as exc:
        print(
            f"[WARNING] PNG export failed for {name}. "
            f"HTML was saved successfully. Error: {exc}"
        )

    print(f"[SAVED] {html_path}")


def wrap_label(text: str, width: int = 20) -> str:
    return "<br>".join(textwrap_line(text, width=width))


def textwrap_line(text: str, width: int) -> List[str]:
    import textwrap
    return textwrap.wrap(str(text), width=width, break_long_words=False)


def clean_metric_name(metric: str) -> str:
    mapping = {
        "pca_pc1_variance": "PCA PC1 variance",
        "pca_pc1_pc2_variance": "PCA PC1 + PC2 variance",
        "pca_pc1_to_pc5_variance": "PCA PC1 to PC5 variance",
        "silhouette_original_space": "Silhouette, original space",
        "silhouette_pca10_space": "Silhouette, PCA10 space",
        "nearest_neighbor_same_system_at_3": "Nearest neighbor, k=3",
        "nearest_neighbor_same_system_at_5": "Nearest neighbor, k=5",
        "ari_agglomerative_vs_system": "ARI, clustering vs tissue system",
        "nmi_agglomerative_vs_system": "NMI, clustering vs tissue system",
    }
    return mapping.get(metric, metric.replace("_", " "))


def short_metric_name(metric: str) -> str:
    mapping = {
        "pca_pc1_pc2_variance": "PCA<br>PC1+PC2",
        "silhouette_pca10_space": "Silhouette<br>PCA10",
        "nearest_neighbor_same_system_at_3": "NN<br>k=3",
        "ari_agglomerative_vs_system": "ARI",
        "nmi_agglomerative_vs_system": "NMI",
    }
    return mapping.get(metric, metric.replace("_", "<br>"))


def clean_feature_set_name(feature_set: str) -> str:
    mapping = {
        "all_matrisome": "All Matrisome",
        "division__core_matrisome": "Core Matrisome",
        "division__matrisome_associated": "Matrisome-associated",
        "category__ecm_glycoproteins": "ECM glycoproteins",
        "category__collagens": "Collagens",
        "category__proteoglycans": "Proteoglycans",
        "category__ecm_affiliated_proteins": "ECM-affiliated proteins",
        "category__ecm_regulators": "ECM regulators",
        "category__secreted_factors": "Secreted factors",
    }
    return mapping.get(feature_set, feature_set.replace("_", " "))


def clean_program_name(program: str) -> str:
    return str(program).replace("/", " / ")


def make_figure_1_workflow() -> None:
    steps = [
        {
            "title": "Data sources",
            "subtitle": "HPA / GTEx-derived RNA<br>Human Matrisome annotations",
            "phase": "Input",
        },
        {
            "title": "ECM matrices",
            "subtitle": "Filter expression by<br>Matrisome genes",
            "phase": "Preprocessing",
        },
        {
            "title": "Specificity test",
            "subtitle": "Matrisome genes vs<br>matched random non-ECM genes",
            "phase": "Validation",
        },
        {
            "title": "Category benchmark",
            "subtitle": "Core Matrisome,<br>glycoproteins, collagens,<br>proteoglycans",
            "phase": "Validation",
        },
        {
            "title": "Latent embeddings",
            "subtitle": "PCA, UMAP, NMF<br>ECM representation spaces",
            "phase": "Representation",
        },
        {
            "title": "Module curation",
            "subtitle": "Top genes + top tissues<br>curated ECM programs",
            "phase": "Interpretation",
        },
        {
            "title": "External validation",
            "subtitle": "Reproducibility outside<br>rna_tissue_consensus",
            "phase": "Validation",
        },
    ]

    fig = go.Figure()

    x_positions = list(range(len(steps)))
    y = 0

    for i, step in enumerate(steps):
        x = x_positions[i]

        fig.add_shape(
            type="rect",
            x0=x - 0.43,
            x1=x + 0.43,
            y0=y - 0.30,
            y1=y + 0.30,
            line=dict(width=2),
            fillcolor="white",
        )

        fig.add_annotation(
            x=x,
            y=y + 0.11,
            text=f"<b>{step['title']}</b>",
            showarrow=False,
            font=dict(size=15),
        )

        fig.add_annotation(
            x=x,
            y=y - 0.10,
            text=step["subtitle"],
            showarrow=False,
            font=dict(size=11),
        )

        fig.add_annotation(
            x=x,
            y=y - 0.42,
            text=step["phase"],
            showarrow=False,
            font=dict(size=10),
        )

        if i < len(steps) - 1:
            fig.add_annotation(
                x=x + 0.82,
                y=y,
                ax=x + 0.45,
                ay=y,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=1.2,
                arrowwidth=2,
            )

    fig.update_xaxes(visible=False, range=[-0.75, len(steps) - 0.25])
    fig.update_yaxes(visible=False, range=[-0.62, 0.55])

    fig.update_layout(
        title=dict(
            text="Figure 1. Workflow of the Matrisome-derived ECM representation framework",
            x=0.5,
            font=dict(size=22),
        ),
        template="plotly_white",
        margin=dict(l=30, r=30, t=80, b=40),
    )

    save_figure(fig, "figure_1_workflow_refined", width=1900, height=520)


def get_tissue_system(tissue: str) -> str:
    return TISSUE_SYSTEM_MAP.get(str(tissue).lower(), "Other")


def make_figure_2_ecm_tissue_space() -> None:
    pca_path = EDA_DIR / "pca_coordinates.csv"
    umap_path = EDA_DIR / "umap_coordinates.csv"
    explained_path = EDA_DIR / "pca_explained_variance.csv"

    if not pca_path.exists() or not umap_path.exists():
        print("[SKIP] Figure 2: missing PCA or UMAP coordinate files.")
        return

    pca_df = pd.read_csv(pca_path, index_col=0)
    umap_df = pd.read_csv(umap_path, index_col=0)

    pca_df.insert(0, "sample", pca_df.index.astype(str))
    umap_df.insert(0, "sample", umap_df.index.astype(str))

    pca_df["tissue_system"] = pca_df["sample"].apply(get_tissue_system)
    umap_df["tissue_system"] = umap_df["sample"].apply(get_tissue_system)

    pca_df["label"] = pca_df["sample"].apply(
        lambda x: x if str(x).lower() in REPRESENTATIVE_TISSUE_LABELS else ""
    )
    umap_df["label"] = umap_df["sample"].apply(
        lambda x: x if str(x).lower() in REPRESENTATIVE_TISSUE_LABELS else ""
    )

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
        subplot_titles=(
            "PCA of ECM-only tissue profiles",
            "UMAP of ECM-only tissue profiles",
        ),
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
                mode="markers+text",
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
                mode="markers+text",
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
        title=dict(
            text="Figure 2. ECM-only expression preserves tissue organization",
            x=0.5,
            font=dict(size=22),
        ),
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

    save_figure(fig, "figure_2_ecm_only_tissue_space_refined", width=1650, height=850)


def make_figure_3_ecm_vs_random() -> None:
    path = TABLE_DIR / "Table_2_ecm_vs_random_specificity.csv"

    if not path.exists():
        print(f"[SKIP] Figure 3: missing {path}")
        return

    df = pd.read_csv(path)

    selected_metrics = [
        "pca_pc1_variance",
        "pca_pc1_pc2_variance",
        "silhouette_original_space",
        "silhouette_pca10_space",
        "nearest_neighbor_same_system_at_3",
        "nearest_neighbor_same_system_at_5",
        "ari_agglomerative_vs_system",
        "nmi_agglomerative_vs_system",
    ]

    df = df[df["metric"].isin(selected_metrics)].copy()
    df["metric_label"] = df["metric"].apply(clean_metric_name)

    df["significant_label"] = df["empirical_p_value_higher_is_better"].apply(
        lambda p: "*" if p < 0.05 else ""
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "A. Observed Matrisome value vs random baseline",
            "B. Z-score vs random baseline",
        ),
        column_widths=[0.58, 0.42],
        horizontal_spacing=0.20,
    )

    fig.add_trace(
        go.Bar(
            y=df["metric_label"],
            x=df["random_mean"],
            orientation="h",
            error_x=dict(
                type="data",
                array=df["random_std"],
                arrayminus=df["random_std"],
                visible=True,
            ),
            name="Random non-ECM mean ± SD",
            opacity=0.65,
            hovertemplate=(
                "Metric: %{y}<br>"
                "Random mean: %{x:.4f}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            y=df["metric_label"],
            x=df["ecm_value"],
            mode="markers",
            name="Matrisome genes",
            marker=dict(size=12, symbol="diamond"),
            customdata=df[
                [
                    "ecm_z_score_vs_random",
                    "empirical_p_value_higher_is_better",
                ]
            ],
            hovertemplate=(
                "Metric: %{y}<br>"
                "Matrisome value: %{x:.4f}<br>"
                "Z-score: %{customdata[0]:.2f}<br>"
                "Empirical p: %{customdata[1]:.4f}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            y=df["metric_label"],
            x=df["ecm_z_score_vs_random"],
            orientation="h",
            name="Matrisome z-score",
            customdata=df[
                [
                    "empirical_p_value_higher_is_better",
                    "significant_label",
                ]
            ],
            text=df["significant_label"],
            textposition="outside",
            hovertemplate=(
                "Metric: %{y}<br>"
                "Z-score: %{x:.2f}<br>"
                "Empirical p: %{customdata[0]:.4f}<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )

    fig.add_vline(x=0, line_dash="dash", line_width=1, row=1, col=2)
    fig.add_vline(x=1.96, line_dash="dot", line_width=1, row=1, col=2)

    fig.update_xaxes(title_text="Metric value", row=1, col=1)
    fig.update_xaxes(title_text="Z-score", row=1, col=2)
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=2)

    fig.update_layout(
        title=dict(
            text="Figure 3. Matrisome genes compared with matched random non-ECM genes",
            x=0.5,
            font=dict(size=22),
        ),
        template="plotly_white",
        barmode="overlay",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.18,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=260, r=60, t=100, b=130),
    )

    save_figure(fig, "figure_3_ecm_vs_random_non_ecm_refined", width=1650, height=900)


def make_figure_4_category_benchmark() -> None:
    path = TABLE_DIR / "Table_3_matrisome_category_benchmark.csv"

    if not path.exists():
        print(f"[SKIP] Figure 4: missing {path}")
        return

    df = pd.read_csv(path)

    selected_metrics = [
        "pca_pc1_pc2_variance",
        "silhouette_pca10_space",
        "nearest_neighbor_same_system_at_3",
        "ari_agglomerative_vs_system",
        "nmi_agglomerative_vs_system",
    ]

    feature_order = [
        "all_matrisome",
        "division__core_matrisome",
        "category__ecm_glycoproteins",
        "category__collagens",
        "category__proteoglycans",
        "category__ecm_affiliated_proteins",
        "division__matrisome_associated",
        "category__ecm_regulators",
        "category__secreted_factors",
    ]

    df = df[df["metric"].isin(selected_metrics)].copy()
    df["metric_label"] = df["metric"].apply(short_metric_name)
    df["feature_label"] = df["feature_set"].apply(clean_feature_set_name)

    ordered_feature_labels = [clean_feature_set_name(x) for x in feature_order]

    df["feature_label"] = pd.Categorical(
        df["feature_label"],
        categories=ordered_feature_labels,
        ordered=True,
    )

    pivot = df.pivot_table(
        index="feature_label",
        columns="metric_label",
        values="z_score_vs_random",
        aggfunc="mean",
        observed=False,
    ).dropna(how="all")

    pval_pivot = df.pivot_table(
        index="feature_label",
        columns="metric_label",
        values="empirical_p_value_higher_is_better",
        aggfunc="mean",
        observed=False,
    )

    hover_text = []
    for row_label in pivot.index:
        row_text = []
        for col_label in pivot.columns:
            z = pivot.loc[row_label, col_label]
            p = (
                pval_pivot.loc[row_label, col_label]
                if row_label in pval_pivot.index and col_label in pval_pivot.columns
                else math.nan
            )
            row_text.append(
                f"Feature set: {row_label}<br>"
                f"Metric: {col_label.replace('<br>', ' ')}<br>"
                f"Z-score: {z:.2f}<br>"
                f"Empirical p: {p:.4f}"
            )
        hover_text.append(row_text)

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.astype(str).tolist(),
            text=[[f"{v:.2f}" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            hovertext=hover_text,
            hovertemplate="%{hovertext}<extra></extra>",
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="Z-score<br>vs random"),
        )
    )

    fig.update_layout(
        title=dict(
            text="Figure 4. Matrisome category benchmark against matched random genes",
            x=0.5,
            font=dict(size=22),
        ),
        template="plotly_white",
        xaxis=dict(title="", tickangle=0),
        yaxis=dict(title=""),
        margin=dict(l=260, r=70, t=100, b=100),
    )

    save_figure(fig, "figure_4_matrisome_category_benchmark_heatmap_refined", width=1450, height=900)


def read_matrix_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[SKIP] Missing matrix: {path}")
        return None

    return pd.read_csv(path, index_col=0)


def make_matrix_heatmap(
    matrix: pd.DataFrame,
    title: str,
    name: str,
    color_title: str,
    width: int = 1400,
    height: int = 850,
) -> None:
    matrix = matrix.copy()
    matrix.index = [clean_program_name(x) for x in matrix.index]

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns.tolist(),
            y=matrix.index.tolist(),
            text=matrix.values,
            texttemplate="%{text}",
            colorscale="Blues",
            colorbar=dict(title=color_title),
            hovertemplate=(
                "Program: %{y}<br>"
                "Column: %{x}<br>"
                "Value: %{z}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=22)),
        template="plotly_white",
        margin=dict(l=320, r=60, t=100, b=110),
        xaxis=dict(tickangle=20),
    )

    save_figure(fig, name, width=width, height=height)


def make_figure_5_curated_programs() -> None:
    presence_path = TABLE_DIR / "curated_program_presence_matrix.csv"
    high_conf_path = TABLE_DIR / "curated_program_high_confidence_matrix.csv"

    presence = read_matrix_csv(presence_path)
    high_conf = read_matrix_csv(high_conf_path)

    if presence is not None:
        make_matrix_heatmap(
            matrix=presence,
            title="Figure 5A. Curated recurring ECM programs across Matrisome feature sets",
            name="figure_5a_curated_ecm_program_presence_refined",
            color_title="N modules",
        )

    if high_conf is not None:
        make_matrix_heatmap(
            matrix=high_conf,
            title="Figure 5B. High-confidence curated ECM programs across Matrisome feature sets",
            name="figure_5b_curated_ecm_program_high_confidence_refined",
            color_title="High-confidence<br>modules",
        )


def make_figure_6_external_reproducibility() -> None:
    dataset_path = TABLE_DIR / "external_dataset_program_presence.csv"
    feature_path = TABLE_DIR / "external_feature_set_program_presence.csv"

    dataset_matrix = read_matrix_csv(dataset_path)
    feature_matrix = read_matrix_csv(feature_path)

    if dataset_matrix is not None:
        make_matrix_heatmap(
            matrix=dataset_matrix,
            title="Figure 6A. External reproducibility of ECM programs across datasets",
            name="figure_6a_external_dataset_reproducibility_refined",
            color_title="N reproduced<br>modules",
            width=1450,
            height=850,
        )

    if feature_matrix is not None:
        make_matrix_heatmap(
            matrix=feature_matrix,
            title="Figure 6B. External reproducibility of ECM programs across feature sets",
            name="figure_6b_external_feature_set_reproducibility_refined",
            color_title="N reproduced<br>modules",
            width=1450,
            height=850,
        )


def main() -> None:
    ensure_dirs()

    make_figure_1_workflow()
    make_figure_2_ecm_tissue_space()
    make_figure_3_ecm_vs_random()
    make_figure_4_category_benchmark()
    make_figure_5_curated_programs()
    make_figure_6_external_reproducibility()

    print("\n[DONE]")
    print(f"HTML figures: {HTML_DIR}")
    print(f"PNG figures:  {PNG_DIR}")


if __name__ == "__main__":
    main()