from __future__ import annotations

from pathlib import Path
import math
import textwrap

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


FINAL_DIR = Path("outputs/final_results_v0.1")
TABLE_DIR = FINAL_DIR / "tables"
FIGURE_DIR = FINAL_DIR / "figures"

HTML_DIR = FIGURE_DIR / "html"
PNG_DIR = FIGURE_DIR / "png"

EDA_DIR = Path("outputs/eda/rna_tissue_consensus")


def ensure_dirs() -> None:
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig: go.Figure, name: str, width: int = 1200, height: int = 800) -> None:
    html_path = HTML_DIR / f"{name}.html"
    png_path = PNG_DIR / f"{name}.png"

    fig.update_layout(width=width, height=height)

    fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)

    try:
        fig.write_image(str(png_path), scale=3)
    except Exception as exc:
        print(
            f"[WARNING] Could not save PNG for {name}. "
            f"HTML was saved. Install/upgrade kaleido if needed. Error: {exc}"
        )

    print(f"[SAVED] {html_path}")
    print(f"[SAVED] {png_path}")


def clean_metric_name(metric: str) -> str:
    mapping = {
        "pca_pc1_variance": "PCA PC1 variance",
        "pca_pc1_pc2_variance": "PCA PC1+PC2 variance",
        "pca_pc1_to_pc5_variance": "PCA PC1-PC5 variance",
        "silhouette_original_space": "Silhouette, original space",
        "silhouette_pca10_space": "Silhouette, PCA10 space",
        "nearest_neighbor_same_system_at_3": "NN same system, k=3",
        "nearest_neighbor_same_system_at_5": "NN same system, k=5",
        "ari_agglomerative_vs_system": "ARI, clustering vs system",
        "nmi_agglomerative_vs_system": "NMI, clustering vs system",
    }
    return mapping.get(metric, metric.replace("_", " "))


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
        ("Data sources", "HPA / GTEx-derived RNA\nHuman Matrisome annotations"),
        ("ECM matrices", "Filter expression by\nMatrisome genes"),
        ("Specificity test", "Matrisome genes vs\nmatched random non-ECM genes"),
        ("Category benchmark", "Core Matrisome,\nglycoproteins, collagens,\nproteoglycans"),
        ("Latent embeddings", "PCA, UMAP, NMF\nECM representation spaces"),
        ("Module curation", "Top genes + top tissues\ncurated ECM programs"),
        ("External validation", "Reproducibility outside\nrna_tissue_consensus"),
    ]

    fig = go.Figure()

    x_positions = list(range(len(steps)))
    y = 0

    for i, (title, subtitle) in enumerate(steps):
        x = x_positions[i]

        fig.add_shape(
            type="rect",
            x0=x - 0.42,
            x1=x + 0.42,
            y0=y - 0.32,
            y1=y + 0.32,
            line=dict(width=2),
            fillcolor="white",
        )

        fig.add_annotation(
            x=x,
            y=y + 0.08,
            text=f"<b>{title}</b>",
            showarrow=False,
            font=dict(size=14),
        )

        fig.add_annotation(
            x=x,
            y=y - 0.12,
            text=subtitle.replace("\n", "<br>"),
            showarrow=False,
            font=dict(size=11),
        )

        if i < len(steps) - 1:
            fig.add_annotation(
                x=x + 0.5,
                y=y,
                ax=x + 0.93,
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

    fig.update_xaxes(visible=False, range=[-0.8, len(steps) - 0.2])
    fig.update_yaxes(visible=False, range=[-0.7, 0.7])

    fig.update_layout(
        title=dict(
            text=(
                "Figure 1. Workflow of the Matrisome-derived ECM representation framework"
            ),
            x=0.5,
        ),
        template="plotly_white",
        margin=dict(l=30, r=30, t=80, b=30),
    )

    save_figure(fig, "figure_1_workflow", width=1800, height=450)


def make_figure_2_ecm_tissue_space() -> None:
    pca_path = EDA_DIR / "pca_coordinates.csv"
    umap_path = EDA_DIR / "umap_coordinates.csv"
    explained_path = EDA_DIR / "pca_explained_variance.csv"

    if not pca_path.exists() or not umap_path.exists():
        print("[SKIP] Figure 2: missing PCA/UMAP coordinate files.")
        return

    pca_df = pd.read_csv(pca_path, index_col=0)
    umap_df = pd.read_csv(umap_path, index_col=0)

    pca_df.insert(0, "sample", pca_df.index.astype(str))
    umap_df.insert(0, "sample", umap_df.index.astype(str))

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
    )

    fig.add_trace(
        go.Scatter(
            x=pca_df["PC1"],
            y=pca_df["PC2"],
            mode="markers",
            text=pca_df["sample"],
            hovertemplate="Tissue: %{text}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>",
            marker=dict(size=9, opacity=0.85),
            name="PCA",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=umap_df["UMAP1"],
            y=umap_df["UMAP2"],
            mode="markers",
            text=umap_df["sample"],
            hovertemplate="Tissue: %{text}<br>UMAP1: %{x:.3f}<br>UMAP2: %{y:.3f}<extra></extra>",
            marker=dict(size=9, opacity=0.85),
            name="UMAP",
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
        ),
        template="plotly_white",
        showlegend=False,
    )

    save_figure(fig, "figure_2_ecm_only_tissue_space", width=1500, height=700)


def make_figure_3_ecm_vs_random() -> None:
    path = TABLE_DIR / "Table_2_ecm_vs_random_specificity.csv"

    if not path.exists():
        print(f"[SKIP] Figure 3: missing {path}")
        return

    df = pd.read_csv(path)

    required = [
        "metric",
        "ecm_value",
        "random_mean",
        "random_std",
        "ecm_z_score_vs_random",
        "empirical_p_value_higher_is_better",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"[SKIP] Figure 3: missing columns {missing}")
        return

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

    df["minus_std"] = df["random_std"]
    df["plus_std"] = df["random_std"]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Observed Matrisome value vs matched random baseline",
            "Z-score of Matrisome value vs random baseline",
        ),
        column_widths=[0.58, 0.42],
    )

    fig.add_trace(
        go.Bar(
            y=df["metric_label"],
            x=df["random_mean"],
            orientation="h",
            error_x=dict(
                type="data",
                array=df["plus_std"],
                arrayminus=df["minus_std"],
                visible=True,
            ),
            name="Random non-ECM mean ± SD",
            opacity=0.65,
            hovertemplate=(
                "Metric: %{y}<br>"
                "Random mean: %{x:.4f}<br>"
                "<extra></extra>"
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
            name="Z-score vs random",
            customdata=df[["empirical_p_value_higher_is_better"]],
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
            text=(
                "Figure 3. Matrisome genes compared with matched random non-ECM genes"
            ),
            x=0.5,
        ),
        template="plotly_white",
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5),
        margin=dict(l=240, r=40, t=90, b=120),
    )

    save_figure(fig, "figure_3_ecm_vs_random_non_ecm", width=1500, height=850)


def make_figure_4_category_benchmark() -> None:
    path = TABLE_DIR / "Table_3_matrisome_category_benchmark.csv"

    if not path.exists():
        print(f"[SKIP] Figure 4: missing {path}")
        return

    df = pd.read_csv(path)

    required = [
        "feature_set",
        "metric",
        "z_score_vs_random",
        "empirical_p_value_higher_is_better",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"[SKIP] Figure 4: missing columns {missing}")
        return

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
        "division__matrisome_associated",
        "category__ecm_glycoproteins",
        "category__collagens",
        "category__proteoglycans",
        "category__ecm_affiliated_proteins",
        "category__ecm_regulators",
        "category__secreted_factors",
    ]

    df = df[df["metric"].isin(selected_metrics)].copy()
    df["metric_label"] = df["metric"].apply(clean_metric_name)
    df["feature_label"] = df["feature_set"].apply(clean_feature_set_name)

    df["feature_label"] = pd.Categorical(
        df["feature_label"],
        categories=[clean_feature_set_name(x) for x in feature_order],
        ordered=True,
    )

    pivot = df.pivot_table(
        index="feature_label",
        columns="metric_label",
        values="z_score_vs_random",
        aggfunc="mean",
        observed=False,
    )

    pval_pivot = df.pivot_table(
        index="feature_label",
        columns="metric_label",
        values="empirical_p_value_higher_is_better",
        aggfunc="mean",
        observed=False,
    )

    pivot = pivot.dropna(how="all")

    hover_text = []
    for row_label in pivot.index:
        row_text = []
        for col_label in pivot.columns:
            z = pivot.loc[row_label, col_label]
            p = pval_pivot.loc[row_label, col_label] if row_label in pval_pivot.index and col_label in pval_pivot.columns else math.nan
            row_text.append(
                f"Feature set: {row_label}<br>"
                f"Metric: {col_label}<br>"
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
            colorbar=dict(title="Z-score vs random"),
        )
    )

    fig.update_layout(
        title=dict(
            text="Figure 4. Matrisome category benchmark against matched random genes",
            x=0.5,
        ),
        template="plotly_white",
        xaxis=dict(title="", tickangle=35),
        yaxis=dict(title=""),
        margin=dict(l=230, r=40, t=90, b=170),
    )

    save_figure(fig, "figure_4_matrisome_category_benchmark_heatmap", width=1350, height=850)


def read_matrix_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[SKIP] Missing matrix: {path}")
        return None

    df = pd.read_csv(path, index_col=0)
    return df


def make_matrix_heatmap(
    matrix: pd.DataFrame,
    title: str,
    name: str,
    color_title: str = "Number of modules",
    colorscale: str = "Blues",
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
            colorscale=colorscale,
            colorbar=dict(title=color_title),
            hovertemplate=(
                "Program: %{y}<br>"
                "Column: %{x}<br>"
                "Value: %{z}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        template="plotly_white",
        margin=dict(l=300, r=40, t=90, b=120),
        xaxis=dict(tickangle=25),
    )

    save_figure(fig, name, width=1350, height=850)


def make_figure_5_curated_programs() -> None:
    presence_path = TABLE_DIR / "curated_program_presence_matrix.csv"
    high_conf_path = TABLE_DIR / "curated_program_high_confidence_matrix.csv"

    presence = read_matrix_csv(presence_path)
    high_conf = read_matrix_csv(high_conf_path)

    if presence is not None:
        make_matrix_heatmap(
            matrix=presence,
            title=(
                "Figure 5A. Curated recurring ECM programs across Matrisome feature sets"
            ),
            name="figure_5a_curated_ecm_program_presence",
            color_title="N modules",
            colorscale="Blues",
        )

    if high_conf is not None:
        make_matrix_heatmap(
            matrix=high_conf,
            title=(
                "Figure 5B. High-confidence curated ECM programs across Matrisome feature sets"
            ),
            name="figure_5b_curated_ecm_program_high_confidence",
            color_title="High-confidence modules",
            colorscale="Greens",
        )


def make_figure_6_external_reproducibility() -> None:
    dataset_path = TABLE_DIR / "external_dataset_program_presence.csv"
    feature_path = TABLE_DIR / "external_feature_set_program_presence.csv"

    dataset_matrix = read_matrix_csv(dataset_path)
    feature_matrix = read_matrix_csv(feature_path)

    if dataset_matrix is not None:
        make_matrix_heatmap(
            matrix=dataset_matrix,
            title=(
                "Figure 6A. External reproducibility of ECM programs across datasets"
            ),
            name="figure_6a_external_dataset_reproducibility",
            color_title="N reproduced modules",
            colorscale="Blues",
        )

    if feature_matrix is not None:
        make_matrix_heatmap(
            matrix=feature_matrix,
            title=(
                "Figure 6B. External reproducibility of ECM programs across feature sets"
            ),
            name="figure_6b_external_feature_set_reproducibility",
            color_title="N reproduced modules",
            colorscale="Purples",
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