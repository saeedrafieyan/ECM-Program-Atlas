from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go


INPUT_DIR = Path("outputs/matrisomedb_validation")
TABLE_DIR = INPUT_DIR / "tables"

OUTPUT_DIR = INPUT_DIR / "summary"
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


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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


def load_required_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gene_support_path = TABLE_DIR / "matrisomedb_program_gene_support_summary.csv"
    scores_path = TABLE_DIR / "matrisomedb_program_tissue_scores.csv"
    top_tissues_path = TABLE_DIR / "matrisomedb_program_top_tissues.csv"

    for path in [gene_support_path, scores_path, top_tissues_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    gene_support = pd.read_csv(gene_support_path)
    scores = pd.read_csv(scores_path)
    top_tissues = pd.read_csv(top_tissues_path)

    return gene_support, scores, top_tissues


def ordered_programs(programs: List[str]) -> List[str]:
    return [p for p in PROGRAM_ORDER if p in programs] + [
        p for p in programs if p not in PROGRAM_ORDER
    ]


def create_gene_support_summary(gene_support: pd.DataFrame) -> pd.DataFrame:
    subset = gene_support[
        (gene_support["condition_group"].eq("all_samples"))
        & (gene_support["matrix_name"].eq("mean_log_nsaf"))
    ].copy()

    subset["support_level"] = pd.cut(
        subset["protein_detection_coverage"],
        bins=[-0.01, 0.50, 0.70, 0.85, 1.01],
        labels=["weak", "moderate", "strong", "very strong"],
    )

    subset["ecm_program"] = pd.Categorical(
        subset["ecm_program"],
        categories=PROGRAM_ORDER,
        ordered=True,
    )

    subset = subset.sort_values("ecm_program")

    columns = [
        "ecm_program",
        "n_program_genes",
        "n_detected_genes",
        "n_missing_genes",
        "protein_detection_coverage",
        "support_level",
        "detected_genes",
        "missing_genes",
    ]

    return subset[columns]


def plot_gene_detection_coverage(gene_support_clean: pd.DataFrame) -> None:
    df = gene_support_clean.copy()
    df = df.sort_values("protein_detection_coverage", ascending=True)

    fig = go.Figure(
        data=go.Bar(
            x=df["protein_detection_coverage"],
            y=df["ecm_program"],
            orientation="h",
            customdata=df[
                [
                    "n_detected_genes",
                    "n_program_genes",
                    "support_level",
                ]
            ],
            hovertemplate=(
                "Program: %{y}<br>"
                "Coverage: %{x:.2f}<br>"
                "Detected genes: %{customdata[0]} / %{customdata[1]}<br>"
                "Support level: %{customdata[2]}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=(
            "MatrisomeDB protein detection coverage of RNA-derived ECM programs"
            "<br><sup>All human MatrisomeDB samples, mean log NSAF matrix</sup>"
        ),
        xaxis_title="Protein detection coverage",
        yaxis_title="",
        template="plotly_white",
        margin=dict(l=310, r=60, t=100, b=80),
    )

    save_figure(
        fig,
        "matrisomedb_gene_detection_coverage_clean",
        width=1300,
        height=760,
    )


def make_score_matrix(
    scores: pd.DataFrame,
    condition_group: str,
    matrix_name: str,
) -> pd.DataFrame:
    subset = scores[
        (scores["condition_group"].eq(condition_group))
        & (scores["matrix_name"].eq(matrix_name))
    ].copy()

    if subset.empty:
        return pd.DataFrame()

    matrix = subset.pivot_table(
        index="ecm_program",
        columns="tissue",
        values="program_score",
        aggfunc="mean",
        fill_value=0.0,
    )

    matrix = matrix.loc[ordered_programs(matrix.index.tolist())]

    return matrix


def row_zscore(matrix: pd.DataFrame) -> pd.DataFrame:
    z = matrix.copy().astype(float)

    means = z.mean(axis=1)
    stds = z.std(axis=1, ddof=0).replace(0, np.nan)

    z = z.sub(means, axis=0).div(stds, axis=0)
    z = z.fillna(0.0)

    return z


def row_minmax(matrix: pd.DataFrame) -> pd.DataFrame:
    x = matrix.copy().astype(float)

    mins = x.min(axis=1)
    maxs = x.max(axis=1)
    ranges = (maxs - mins).replace(0, np.nan)

    x = x.sub(mins, axis=0).div(ranges, axis=0)
    x = x.fillna(0.0)

    return x


def plot_heatmap(
    matrix: pd.DataFrame,
    title: str,
    name: str,
    color_title: str,
    colorscale: str = "Viridis",
    zmid: float | None = None,
) -> None:
    if matrix.empty:
        print(f"[SKIP] Empty heatmap: {name}")
        return

    heatmap_kwargs = dict(
        z=matrix.values,
        x=matrix.columns.tolist(),
        y=matrix.index.tolist(),
        colorscale=colorscale,
        colorbar=dict(title=color_title),
        hovertemplate=(
            "Program: %{y}<br>"
            "Tissue: %{x}<br>"
            "Score: %{z:.3f}<extra></extra>"
        ),
    )

    if zmid is not None:
        heatmap_kwargs["zmid"] = zmid

    fig = go.Figure(data=go.Heatmap(**heatmap_kwargs))

    fig.update_layout(
        title=title,
        xaxis=dict(tickangle=35),
        yaxis=dict(title=""),
        template="plotly_white",
        margin=dict(l=310, r=60, t=105, b=150),
    )

    save_figure(fig, name, width=1350, height=800)


def create_top_tissue_summary(top_tissues: pd.DataFrame) -> pd.DataFrame:
    subset = top_tissues[
        top_tissues["matrix_name"].eq("mean_log_nsaf")
    ].copy()

    keep_cols = [
        "condition_group",
        "ecm_program",
        "n_available_program_genes",
        "top_protein_tissues",
        "top_rna_tissues",
        "n_rna_protein_top_tissue_overlap",
        "rna_protein_top_tissue_overlap",
        "mean_top10_protein_score",
        "max_top10_protein_score",
    ]

    subset = subset[keep_cols]

    subset["ecm_program"] = pd.Categorical(
        subset["ecm_program"],
        categories=PROGRAM_ORDER,
        ordered=True,
    )

    subset = subset.sort_values(["condition_group", "ecm_program"])

    return subset


def assign_validation_strength(row: pd.Series) -> str:
    coverage = row.get("protein_detection_coverage", np.nan)

    if pd.isna(coverage):
        return "uncertain"

    if coverage >= 0.85:
        return "very strong protein gene-level support"

    if coverage >= 0.70:
        return "strong protein gene-level support"

    if coverage >= 0.50:
        return "moderate protein gene-level support"

    return "weak protein gene-level support"


def create_program_validation_summary(
    gene_support_clean: pd.DataFrame,
    top_tissue_summary: pd.DataFrame,
) -> pd.DataFrame:
    records = []

    top_all = top_tissue_summary[
        top_tissue_summary["condition_group"].eq("all_samples")
    ].copy()

    top_normal = top_tissue_summary[
        top_tissue_summary["condition_group"].eq("normal_like")
    ].copy()

    for row in gene_support_clean.itertuples():
        program = row.ecm_program

        all_row = top_all[top_all["ecm_program"].astype(str).eq(str(program))]
        normal_row = top_normal[top_normal["ecm_program"].astype(str).eq(str(program))]

        all_top = all_row["top_protein_tissues"].iloc[0] if not all_row.empty else ""
        normal_top = normal_row["top_protein_tissues"].iloc[0] if not normal_row.empty else ""

        all_overlap = (
            all_row["rna_protein_top_tissue_overlap"].iloc[0]
            if not all_row.empty
            else ""
        )

        normal_overlap = (
            normal_row["rna_protein_top_tissue_overlap"].iloc[0]
            if not normal_row.empty
            else ""
        )

        base = pd.Series(
            {
                "protein_detection_coverage": row.protein_detection_coverage
            }
        )

        records.append(
            {
                "ecm_program": program,
                "n_program_genes": row.n_program_genes,
                "n_detected_genes": row.n_detected_genes,
                "protein_detection_coverage": row.protein_detection_coverage,
                "gene_level_validation_strength": assign_validation_strength(base),
                "all_samples_top_protein_tissues": all_top,
                "normal_like_top_protein_tissues": normal_top,
                "all_samples_rna_protein_overlap": all_overlap,
                "normal_like_rna_protein_overlap": normal_overlap,
                "interpretation_note": "",
            }
        )

    summary = pd.DataFrame(records)

    notes = {
        "Vascular/stromal/interstitial ECM": (
            "Strong protein-level support; high scores in blood vessel and connective-rich tissues are biologically plausible."
        ),
        "Epithelial/mucosal basement membrane ECM": (
            "Good protein-level support; interpretation strongest in stomach/colon/skin-like epithelial contexts when present."
        ),
        "CNS/neural ECM": (
            "Moderate gene-level support, but MatrisomeDB export has limited normal CNS tissue coverage, so tissue-level validation is incomplete."
        ),
        "Retinal/sensory ECM": (
            "Moderate to strong gene-level support; tissue-level interpretation depends heavily on Eye/retina availability."
        ),
        "Immune/lymphoid remodeling ECM": (
            "Strong gene-level support, but tissue-level validation is limited by available immune/lymphoid tissues."
        ),
        "Stromal remodeling ECM": (
            "Moderate to strong support; likely overlaps with vascular/stromal/interstitial ECM."
        ),
        "Renal/endothelial basement membrane ECM": (
            "Strong gene-level support; kidney signal is biologically plausible when kidney samples are available."
        ),
        "Hepatic/plasma-associated ECM": (
            "Good protein-level support, but should be described cautiously because liver/plasma proteins may reflect secreted extracellular proteins rather than deposited ECM architecture."
        ),
        "Reproductive-specialized ECM": (
            "Partial gene-level support; tissue-level validation is currently limited and should be treated as preliminary."
        ),
    }

    summary["interpretation_note"] = summary["ecm_program"].map(notes).fillna("")

    return summary


def write_markdown_report(
    gene_support_clean: pd.DataFrame,
    program_validation_summary: pd.DataFrame,
) -> None:
    report_path = OUTPUT_DIR / "matrisomedb_validation_summary.md"

    lines: List[str] = []
    lines.append("# MatrisomeDB Proteomics Validation Summary\n")
    lines.append("## Purpose\n")
    lines.append(
        "This analysis evaluates whether RNA-derived ECM programs have protein-level support in the MatrisomeDB human export."
    )
    lines.append("\n## Key interpretation\n")
    lines.append(
        "The MatrisomeDB export provides protein-level support for most RNA-derived ECM programs. However, the export contains mixed normal and disease-associated samples, and normal-like tissue coverage is sparse. Therefore, this analysis should be interpreted as protein-level support, not complete healthy-tissue ECM validation."
    )
    lines.append("\n## Gene-level support\n")

    for row in gene_support_clean.itertuples():
        lines.append(
            f"- **{row.ecm_program}**: {row.n_detected_genes}/{row.n_program_genes} genes detected "
            f"({row.protein_detection_coverage:.2f}; {row.support_level})."
        )

    lines.append("\n## Program-level interpretation\n")
    for row in program_validation_summary.itertuples():
        lines.append(f"### {row.ecm_program}\n")
        lines.append(
            f"- Detection coverage: {row.protein_detection_coverage:.2f}\n"
            f"- Gene-level validation: {row.gene_level_validation_strength}\n"
            f"- All-samples top protein tissues: {row.all_samples_top_protein_tissues}\n"
            f"- Normal-like top protein tissues: {row.normal_like_top_protein_tissues}\n"
            f"- Interpretation: {row.interpretation_note}\n"
        )

    lines.append("\n## Limitations\n")
    lines.append("1. The MatrisomeDB export contains mixed normal, cancer, fibrotic, stenotic, and other disease-associated samples.\n")
    lines.append("2. The normal-like subset has limited tissue coverage.\n")
    lines.append("3. NSAF values are semi-quantitative and may not be directly comparable across all studies.\n")
    lines.append("4. Tissue-level protein validation is partial and should be strengthened with a more complete MatrisomeDB export or study-level normalization.\n")
    lines.append("5. CNS/neural and retinal/sensory programs require more targeted protein-level datasets for strong validation.\n")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAVED] {report_path}")


def main() -> None:
    ensure_dirs()

    gene_support, scores, top_tissues = load_required_tables()

    gene_support_clean = create_gene_support_summary(gene_support)
    gene_support_clean.to_csv(
        OUTPUT_DIR / "matrisomedb_gene_support_clean_summary.csv",
        index=False,
    )

    plot_gene_detection_coverage(gene_support_clean)

    top_tissue_summary = create_top_tissue_summary(top_tissues)
    top_tissue_summary.to_csv(
        OUTPUT_DIR / "matrisomedb_top_tissue_summary.csv",
        index=False,
    )

    program_validation_summary = create_program_validation_summary(
        gene_support_clean=gene_support_clean,
        top_tissue_summary=top_tissue_summary,
    )

    program_validation_summary.to_csv(
        OUTPUT_DIR / "matrisomedb_program_validation_summary.csv",
        index=False,
    )

    # Raw and normalized heatmaps for all_samples and normal_like.
    for condition_group in ["all_samples", "normal_like"]:
        matrix = make_score_matrix(
            scores=scores,
            condition_group=condition_group,
            matrix_name="mean_log_nsaf",
        )

        if matrix.empty:
            continue

        matrix.to_csv(OUTPUT_DIR / f"{condition_group}_mean_log_nsaf_program_matrix.csv")

        plot_heatmap(
            matrix=matrix,
            title=(
                f"MatrisomeDB protein-level ECM program scores<br>"
                f"<sup>{condition_group}, raw mean log NSAF</sup>"
            ),
            name=f"matrisomedb_{condition_group}_raw_mean_log_nsaf_heatmap",
            color_title="Program score",
            colorscale="Viridis",
        )

        z = row_zscore(matrix)
        z.to_csv(OUTPUT_DIR / f"{condition_group}_row_zscore_program_matrix.csv")

        plot_heatmap(
            matrix=z,
            title=(
                f"MatrisomeDB relative tissue enrichment per ECM program<br>"
                f"<sup>{condition_group}, row-wise z-score of mean log NSAF</sup>"
            ),
            name=f"matrisomedb_{condition_group}_row_zscore_heatmap",
            color_title="Row z-score",
            colorscale="RdBu",
            zmid=0,
        )

        mm = row_minmax(matrix)
        mm.to_csv(OUTPUT_DIR / f"{condition_group}_row_minmax_program_matrix.csv")

        plot_heatmap(
            matrix=mm,
            title=(
                f"MatrisomeDB relative tissue enrichment per ECM program<br>"
                f"<sup>{condition_group}, row-wise min-max scaled mean log NSAF</sup>"
            ),
            name=f"matrisomedb_{condition_group}_row_minmax_heatmap",
            color_title="Row min-max",
            colorscale="Viridis",
        )

    write_markdown_report(
        gene_support_clean=gene_support_clean,
        program_validation_summary=program_validation_summary,
    )

    print("\n[DONE]")
    print(f"Summary folder: {OUTPUT_DIR}")
    print(f"HTML figures:   {HTML_DIR}")
    print(f"PNG figures:    {PNG_DIR}")


if __name__ == "__main__":
    main()