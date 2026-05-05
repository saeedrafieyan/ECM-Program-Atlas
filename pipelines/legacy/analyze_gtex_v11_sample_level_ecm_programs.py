from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA


PROCESSED_DIR = Path("data/processed/gtex_v11_sample_level")
OUTPUT_DIR = Path("outputs/gtex_v11_sample_level_validation")

TABLE_DIR = OUTPUT_DIR / "tables"
HTML_DIR = OUTPUT_DIR / "figures" / "html"
PNG_DIR = OUTPUT_DIR / "figures" / "png"


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


TISSUE_SYSTEM_MAP = {
    "Brain": "CNS",
    "Nerve": "CNS / peripheral nerve",
    "Pituitary": "Endocrine / neural-related",

    "Blood Vessel": "Vascular / connective",
    "Adipose Tissue": "Vascular / connective",
    "Muscle": "Muscle",
    "Heart": "Muscle",

    "Lung": "Respiratory",
    "Colon": "Digestive",
    "Small Intestine": "Digestive",
    "Stomach": "Digestive",
    "Esophagus": "Digestive",
    "Liver": "Digestive",
    "Pancreas": "Digestive",

    "Spleen": "Immune / lymphoid",
    "Whole Blood": "Immune / blood",

    "Kidney": "Urinary",
    "Bladder": "Urinary",

    "Breast": "Reproductive",
    "Cervix Uteri": "Reproductive",
    "Uterus": "Reproductive",
    "Ovary": "Reproductive",
    "Fallopian Tube": "Reproductive",
    "Prostate": "Reproductive",
    "Testis": "Reproductive",
    "Vagina": "Reproductive",

    "Thyroid": "Endocrine",
    "Adrenal Gland": "Endocrine",

    "Skin": "Epithelial / barrier",
    "Cells": "Cell culture",
    "Minor Salivary Gland": "Digestive / gland",
}


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig: go.Figure, name: str, width: int = 1300, height: int = 800) -> None:
    html_path = HTML_DIR / f"{name}.html"
    png_path = PNG_DIR / f"{name}.png"

    fig.update_layout(width=width, height=height)
    fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)

    try:
        fig.write_image(str(png_path), scale=3)
        print(f"[SAVED] {png_path}")
    except Exception as exc:
        print(f"[WARNING] PNG export failed for {name}: {exc}")

    print(f"[SAVED] {html_path}")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores_path = PROCESSED_DIR / "gtex_v11_program_scores_zscore_mean_with_metadata.csv"
    tissue_summary_path = PROCESSED_DIR / "gtex_v11_tissue_program_summary.csv"
    availability_path = PROCESSED_DIR / "gtex_v11_program_gene_availability.csv"

    for path in [scores_path, tissue_summary_path, availability_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    scores = pd.read_csv(scores_path)
    tissue_summary = pd.read_csv(tissue_summary_path)
    availability = pd.read_csv(availability_path)

    return scores, tissue_summary, availability


def get_program_columns(df: pd.DataFrame) -> List[str]:
    return [program for program in PROGRAM_ORDER if program in df.columns]


def make_tissue_program_matrix(tissue_summary: pd.DataFrame, score_type: str) -> pd.DataFrame:
    subset = tissue_summary[tissue_summary["score_type"].eq(score_type)].copy()

    matrix = subset.pivot_table(
        index="ecm_program",
        columns="tissue",
        values="mean_score",
        aggfunc="mean",
        fill_value=0.0,
    )

    matrix = matrix.loc[[p for p in PROGRAM_ORDER if p in matrix.index]]

    return matrix


def row_zscore(matrix: pd.DataFrame) -> pd.DataFrame:
    x = matrix.astype(float).copy()

    mean = x.mean(axis=1)
    std = x.std(axis=1, ddof=0).replace(0, np.nan)

    z = x.sub(mean, axis=0).div(std, axis=0)
    z = z.fillna(0.0)

    return z


def plot_tissue_program_heatmap(matrix: pd.DataFrame, name: str, title: str, color_title: str) -> None:
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns.tolist(),
            y=matrix.index.tolist(),
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title=color_title),
            hovertemplate=(
                "Program: %{y}<br>"
                "Tissue: %{x}<br>"
                "Score: %{z:.3f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(l=310, r=60, t=100, b=150),
        xaxis=dict(tickangle=35),
        yaxis=dict(title=""),
    )

    save_figure(fig, name, width=1500, height=850)


def create_top_tissue_table(tissue_summary: pd.DataFrame) -> pd.DataFrame:
    subset = tissue_summary[tissue_summary["score_type"].eq("zscore_mean")].copy()

    records = []

    for program, group in subset.groupby("ecm_program"):
        group = group.sort_values("mean_score", ascending=False)

        for rank, row in enumerate(group.head(10).itertuples(), start=1):
            records.append(
                {
                    "ecm_program": program,
                    "rank": rank,
                    "tissue": row.tissue,
                    "n_samples": row.n_samples,
                    "n_subjects": row.n_subjects,
                    "mean_score": row.mean_score,
                    "median_score": row.median_score,
                    "std_score": row.std_score,
                    "q25_score": row.q25_score,
                    "q75_score": row.q75_score,
                }
            )

    top_df = pd.DataFrame(records)
    return top_df


def plot_top_tissues_per_program(top_tissue_df: pd.DataFrame) -> None:
    # One compact faceted-like plot using subplots would be too crowded.
    # Instead, generate one HTML/PNG per program.
    for program in PROGRAM_ORDER:
        subset = top_tissue_df[top_tissue_df["ecm_program"].eq(program)].copy()

        if subset.empty:
            continue

        subset = subset.sort_values("mean_score", ascending=True)

        fig = go.Figure(
            data=go.Bar(
                x=subset["mean_score"],
                y=subset["tissue"],
                orientation="h",
                customdata=subset[["n_samples", "n_subjects", "std_score"]],
                hovertemplate=(
                    "Tissue: %{y}<br>"
                    "Mean score: %{x:.3f}<br>"
                    "Samples: %{customdata[0]}<br>"
                    "Subjects: %{customdata[1]}<br>"
                    "SD: %{customdata[2]:.3f}<extra></extra>"
                ),
            )
        )

        safe_program = (
            program.lower()
            .replace("/", "_")
            .replace(" ", "_")
            .replace("-", "_")
        )

        fig.update_layout(
            title=f"GTEx V11 top tissues for {program}<br><sup>Mean sample-level ECM program z-score</sup>",
            xaxis_title="Mean ECM program score",
            yaxis_title="",
            template="plotly_white",
            margin=dict(l=240, r=60, t=100, b=80),
        )

        save_figure(
            fig,
            f"gtex_v11_top_tissues__{safe_program}",
            width=1200,
            height=650,
        )


def plot_program_gene_availability(availability: pd.DataFrame) -> None:
    df = availability.copy()
    df["ecm_program"] = pd.Categorical(
        df["ecm_program"],
        categories=PROGRAM_ORDER,
        ordered=True,
    )
    df = df.sort_values("ecm_program")

    fig = go.Figure(
        data=go.Bar(
            x=df["availability_fraction"],
            y=df["ecm_program"],
            orientation="h",
            customdata=df[["n_available_genes", "n_program_genes"]],
            hovertemplate=(
                "Program: %{y}<br>"
                "Availability: %{x:.2f}<br>"
                "Available genes: %{customdata[0]} / %{customdata[1]}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="GTEx V11 program gene availability<br><sup>All curated ECM program genes are matched in GTEx V11</sup>",
        xaxis_title="Availability fraction",
        yaxis_title="",
        template="plotly_white",
        margin=dict(l=310, r=60, t=100, b=80),
    )

    save_figure(
        fig,
        "gtex_v11_program_gene_availability",
        width=1200,
        height=700,
    )


def downsample_scores(scores: pd.DataFrame, max_samples_per_tissue: int = 150) -> pd.DataFrame:
    records = []

    for tissue, group in scores.groupby("tissue"):
        n = min(max_samples_per_tissue, group.shape[0])
        records.append(group.sample(n=n, random_state=42))

    return pd.concat(records, ignore_index=True)


def plot_sample_level_pca(scores: pd.DataFrame) -> None:
    program_cols = get_program_columns(scores)

    if len(program_cols) < 2:
        print("[SKIP] Not enough program columns for PCA.")
        return

    plot_df = downsample_scores(scores, max_samples_per_tissue=150)

    X = plot_df[program_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)

    plot_df["PC1"] = coords[:, 0]
    plot_df["PC2"] = coords[:, 1]
    plot_df["tissue_system"] = plot_df["tissue"].map(TISSUE_SYSTEM_MAP).fillna("Other")

    fig = go.Figure()

    for system, group in plot_df.groupby("tissue_system"):
        fig.add_trace(
            go.Scattergl(
                x=group["PC1"],
                y=group["PC2"],
                mode="markers",
                name=system,
                customdata=group[["sample_id", "subject_id", "tissue", "tissue_detail"]],
                hovertemplate=(
                    "Sample: %{customdata[0]}<br>"
                    "Subject: %{customdata[1]}<br>"
                    "Tissue: %{customdata[2]}<br>"
                    "Detail: %{customdata[3]}<br>"
                    "PC1: %{x:.3f}<br>"
                    "PC2: %{y:.3f}<extra></extra>"
                ),
                marker=dict(size=5, opacity=0.65),
            )
        )

    pc1 = pca.explained_variance_ratio_[0] * 100
    pc2 = pca.explained_variance_ratio_[1] * 100

    fig.update_layout(
        title=(
            "GTEx V11 sample-level PCA of ECM program scores<br>"
            f"<sup>Downsampled to at most 150 samples per tissue; PC1={pc1:.1f}%, PC2={pc2:.1f}%</sup>"
        ),
        xaxis_title=f"PC1 ({pc1:.1f}% variance)",
        yaxis_title=f"PC2 ({pc2:.1f}% variance)",
        template="plotly_white",
        margin=dict(l=80, r=60, t=100, b=120),
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5),
    )

    save_figure(
        fig,
        "gtex_v11_sample_level_ecm_program_pca",
        width=1500,
        height=900,
    )


def create_donor_tissue_summary(scores: pd.DataFrame) -> pd.DataFrame:
    program_cols = get_program_columns(scores)

    records = []

    for tissue, group in scores.groupby("tissue"):
        for program in program_cols:
            values = pd.to_numeric(group[program], errors="coerce").dropna()

            if values.empty:
                continue

            records.append(
                {
                    "tissue": tissue,
                    "ecm_program": program,
                    "n_samples": group["sample_id"].nunique(),
                    "n_subjects": group["subject_id"].nunique(),
                    "mean_score": values.mean(),
                    "std_score": values.std(ddof=0),
                    "coefficient_of_variation_abs": abs(values.std(ddof=0) / values.mean()) if values.mean() != 0 else np.nan,
                    "q25_score": values.quantile(0.25),
                    "q75_score": values.quantile(0.75),
                }
            )

    return pd.DataFrame(records)


def create_report(
    global_summary: pd.DataFrame,
    availability: pd.DataFrame,
    top_tissue_df: pd.DataFrame,
) -> None:
    report_path = OUTPUT_DIR / "gtex_v11_sample_level_validation_summary.md"

    gs = global_summary.iloc[0].to_dict() if not global_summary.empty else {}

    lines = []
    lines.append("# GTEx V11 Sample-Level ECM Program Validation\n")
    lines.append("## Purpose\n")
    lines.append(
        "This analysis validates the curated RNA-derived ECM programs at GTEx V11 sample level, moving beyond tissue-average expression matrices."
    )

    lines.append("\n## Key numbers\n")
    lines.append(f"- Expression samples: {gs.get('n_expression_samples', 'NA')}\n")
    lines.append(f"- Matched Matrisome genes: {gs.get('n_matched_matrisome_genes', 'NA')}\n")
    lines.append(f"- ECM programs scored: {gs.get('n_programs', 'NA')}\n")
    lines.append(f"- Samples with tissue metadata: {gs.get('n_samples_with_tissue_metadata', 'NA')}\n")
    lines.append(f"- Missing tissue metadata: {gs.get('n_samples_missing_tissue_metadata', 'NA')}\n")
    lines.append(f"- Donors/subjects: {gs.get('n_subjects', 'NA')}\n")
    lines.append(f"- Tissues: {gs.get('n_tissues', 'NA')}\n")
    lines.append(f"- Tissue details: {gs.get('n_tissue_details', 'NA')}\n")

    lines.append("\n## Gene availability\n")
    for row in availability.itertuples():
        lines.append(
            f"- **{row.ecm_program}**: {row.n_available_genes}/{row.n_program_genes} genes available "
            f"({row.availability_fraction:.2f})."
        )

    lines.append("\n## Top tissue signals\n")
    for program in PROGRAM_ORDER:
        subset = top_tissue_df[
            (top_tissue_df["ecm_program"].eq(program))
            & (top_tissue_df["rank"] <= 5)
        ]

        if subset.empty:
            continue

        tissues = "; ".join(
            [
                f"{row.tissue} ({row.mean_score:.2f})"
                for row in subset.itertuples()
            ]
        )

        lines.append(f"- **{program}**: {tissues}")

    lines.append("\n## Interpretation\n")
    lines.append(
        "GTEx V11 sample-level data substantially strengthens the project by validating ECM programs across nearly 20,000 individual samples and 946 donors. The CNS/neural ECM program is highest in Brain, the hepatic/plasma-associated ECM program is highest in Liver, and the reproductive-specialized ECM program is highest in reproductive tissues. Vascular/stromal/interstitial ECM is highest in Blood Vessel and connective-rich tissues."
    )

    lines.append("\n## Limitations\n")
    lines.append("1. GTEx is transcriptomic, not protein-level.\n")
    lines.append("2. Tissue-level sample counts are uneven.\n")
    lines.append("3. Some programs, especially retinal/sensory ECM, may require tissue-specific reinterpretation because GTEx lacks a strong retina/eye tissue context.\n")
    lines.append("4. Donor-aware statistical testing and classification should be performed as the next step.\n")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAVED] {report_path}")


def main() -> None:
    ensure_dirs()

    scores, tissue_summary, availability = load_inputs()

    global_summary_path = PROCESSED_DIR / "gtex_v11_global_summary.csv"
    global_summary = pd.read_csv(global_summary_path)

    availability.to_csv(TABLE_DIR / "gtex_v11_program_gene_availability.csv", index=False)

    plot_program_gene_availability(availability)

    z_matrix = make_tissue_program_matrix(tissue_summary, score_type="zscore_mean")
    z_matrix.to_csv(TABLE_DIR / "gtex_v11_tissue_program_zscore_matrix.csv")

    z_matrix_row_z = row_zscore(z_matrix)
    z_matrix_row_z.to_csv(TABLE_DIR / "gtex_v11_tissue_program_row_zscore_matrix.csv")

    plot_tissue_program_heatmap(
        matrix=z_matrix,
        name="gtex_v11_tissue_program_mean_zscore_heatmap",
        title="GTEx V11 tissue-level ECM program scores<br><sup>Mean sample-level program z-score per tissue</sup>",
        color_title="Mean z-score",
    )

    plot_tissue_program_heatmap(
        matrix=z_matrix_row_z,
        name="gtex_v11_tissue_program_row_zscore_heatmap",
        title="GTEx V11 relative tissue enrichment per ECM program<br><sup>Row-wise z-score of tissue mean program scores</sup>",
        color_title="Row z-score",
    )

    top_tissue_df = create_top_tissue_table(tissue_summary)
    top_tissue_df.to_csv(TABLE_DIR / "gtex_v11_top_tissues_per_program.csv", index=False)

    plot_top_tissues_per_program(top_tissue_df)

    plot_sample_level_pca(scores)

    donor_tissue_summary = create_donor_tissue_summary(scores)
    donor_tissue_summary.to_csv(
        TABLE_DIR / "gtex_v11_tissue_program_variability_summary.csv",
        index=False,
    )

    create_report(
        global_summary=global_summary,
        availability=availability,
        top_tissue_df=top_tissue_df,
    )

    print("\n[DONE]")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Tables:        {TABLE_DIR}")
    print(f"HTML figures:  {HTML_DIR}")
    print(f"PNG figures:   {PNG_DIR}")


if __name__ == "__main__":
    main()