from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
import plotly.graph_objects as go


PROCESSED_DIR = Path("data/processed/tabula_sapiens")
OUTPUT_DIR = Path("outputs/tabula_sapiens_pseudobulk/analysis")

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


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig: go.Figure, name: str, width: int = 1350, height: int = 850) -> None:
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


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    global_summary = pd.read_csv(PROCESSED_DIR / "tabula_sapiens_global_summary.csv")
    availability = pd.read_csv(PROCESSED_DIR / "tabula_sapiens_program_gene_availability.csv")
    compartment = pd.read_csv(PROCESSED_DIR / "tabula_sapiens_compartment_program_summary.csv")
    celltype = pd.read_csv(PROCESSED_DIR / "tabula_sapiens_celltype_program_summary.csv")
    organ_celltype = pd.read_csv(PROCESSED_DIR / "tabula_sapiens_organ_celltype_program_summary.csv")

    return global_summary, availability, compartment, celltype, organ_celltype


def filter_summary(
    df: pd.DataFrame,
    min_donors: int,
    min_pseudobulk: int,
    min_cells: int,
    score_type: str = "zscore_mean",
) -> pd.DataFrame:
    sub = df[df["score_type"].eq(score_type)].copy()

    sub = sub[
        (sub["n_donors"] >= min_donors)
        & (sub["n_pseudobulk"] >= min_pseudobulk)
        & (sub["n_cells_total"] >= min_cells)
    ].copy()

    return sub


def make_top_celltypes(
    celltype: pd.DataFrame,
    min_donors: int,
    min_pseudobulk: int,
    min_cells: int,
    top_n: int,
) -> pd.DataFrame:
    filtered = filter_summary(
        celltype,
        min_donors=min_donors,
        min_pseudobulk=min_pseudobulk,
        min_cells=min_cells,
    )

    records = []

    for method in sorted(filtered["method"].unique()):
        method_df = filtered[filtered["method"].eq(method)]

        for program in PROGRAM_ORDER:
            group = method_df[method_df["ecm_program"].eq(program)].copy()
            group = group.sort_values("mean_score", ascending=False).head(top_n)

            for rank, row in enumerate(group.itertuples(), start=1):
                records.append(
                    {
                        "method": method,
                        "ecm_program": program,
                        "rank": rank,
                        "cell_type": row.cell_type,
                        "compartment": row.compartment,
                        "n_pseudobulk": row.n_pseudobulk,
                        "n_donors": row.n_donors,
                        "n_cells_total": row.n_cells_total,
                        "mean_score": row.mean_score,
                        "median_score": row.median_score,
                        "std_score": row.std_score,
                    }
                )

    return pd.DataFrame(records)


def make_top_organ_celltypes(
    organ_celltype: pd.DataFrame,
    min_donors: int,
    min_pseudobulk: int,
    min_cells: int,
    top_n: int,
) -> pd.DataFrame:
    filtered = filter_summary(
        organ_celltype,
        min_donors=min_donors,
        min_pseudobulk=min_pseudobulk,
        min_cells=min_cells,
    )

    records = []

    for method in sorted(filtered["method"].unique()):
        method_df = filtered[filtered["method"].eq(method)]

        for program in PROGRAM_ORDER:
            group = method_df[method_df["ecm_program"].eq(program)].copy()
            group = group.sort_values("mean_score", ascending=False).head(top_n)

            for rank, row in enumerate(group.itertuples(), start=1):
                records.append(
                    {
                        "method": method,
                        "ecm_program": program,
                        "rank": rank,
                        "organ": row.organ,
                        "cell_type": row.cell_type,
                        "compartment": row.compartment,
                        "n_pseudobulk": row.n_pseudobulk,
                        "n_donors": row.n_donors,
                        "n_cells_total": row.n_cells_total,
                        "mean_score": row.mean_score,
                        "median_score": row.median_score,
                        "std_score": row.std_score,
                    }
                )

    return pd.DataFrame(records)


def make_compartment_matrix(compartment: pd.DataFrame, method: str) -> pd.DataFrame:
    sub = compartment[
        (compartment["score_type"].eq("zscore_mean"))
        & (compartment["method"].eq(method))
    ].copy()

    matrix = sub.pivot_table(
        index="ecm_program",
        columns="compartment",
        values="mean_score",
        aggfunc="mean",
        fill_value=0.0,
    )

    matrix = matrix.loc[[p for p in PROGRAM_ORDER if p in matrix.index]]

    return matrix


def plot_compartment_heatmap(matrix: pd.DataFrame, method: str) -> None:
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns.tolist(),
            y=matrix.index.tolist(),
            text=[[f"{v:.2f}" for v in row] for row in matrix.values],
            texttemplate="%{text}",
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="Mean z-score"),
            hovertemplate=(
                "Program: %{y}<br>"
                "Compartment: %{x}<br>"
                "Mean score: %{z:.3f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=f"Tabula Sapiens ECM program enrichment by compartment<br><sup>{method}</sup>",
        template="plotly_white",
        margin=dict(l=330, r=60, t=100, b=100),
    )

    save_figure(fig, f"tabula_sapiens_compartment_program_heatmap_{method}")


def plot_top_celltypes(top_df: pd.DataFrame, method: str) -> None:
    method_df = top_df[top_df["method"].eq(method)].copy()

    for program in PROGRAM_ORDER:
        sub = method_df[method_df["ecm_program"].eq(program)].copy()

        if sub.empty:
            continue

        sub = sub.sort_values("mean_score", ascending=True)

        fig = go.Figure(
            data=go.Bar(
                x=sub["mean_score"],
                y=sub["cell_type"],
                orientation="h",
                customdata=sub[["compartment", "n_donors", "n_pseudobulk", "n_cells_total"]],
                hovertemplate=(
                    "Cell type: %{y}<br>"
                    "Mean score: %{x:.3f}<br>"
                    "Compartment: %{customdata[0]}<br>"
                    "Donors: %{customdata[1]}<br>"
                    "Pseudobulk groups: %{customdata[2]}<br>"
                    "Cells: %{customdata[3]}<extra></extra>"
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
            title=f"Top robust Tabula Sapiens cell types for {program}<br><sup>{method}, filtered by donor/cell support</sup>",
            xaxis_title="Mean ECM program z-score",
            yaxis_title="",
            template="plotly_white",
            margin=dict(l=330, r=60, t=100, b=80),
        )

        save_figure(fig, f"tabula_sapiens_top_celltypes_{method}_{safe_program}", width=1300, height=700)


def make_organ_celltype_matrix(
    top_organ_celltypes: pd.DataFrame,
    method: str,
    program: str,
    top_n: int = 30,
) -> pd.DataFrame:
    sub = top_organ_celltypes[
        (top_organ_celltypes["method"].eq(method))
        & (top_organ_celltypes["ecm_program"].eq(program))
    ].copy()

    sub = sub.sort_values("mean_score", ascending=False).head(top_n)

    if sub.empty:
        return pd.DataFrame()

    sub["organ_celltype"] = sub["organ"].astype(str) + " | " + sub["cell_type"].astype(str)

    matrix = sub.set_index("organ_celltype")[["mean_score"]]
    return matrix


def plot_top_organ_celltype_heatmaps(top_organ_celltypes: pd.DataFrame, method: str) -> None:
    for program in PROGRAM_ORDER:
        matrix = make_organ_celltype_matrix(
            top_organ_celltypes=top_organ_celltypes,
            method=method,
            program=program,
            top_n=30,
        )

        if matrix.empty:
            continue

        fig = go.Figure(
            data=go.Heatmap(
                z=matrix.values,
                x=["mean_score"],
                y=matrix.index.tolist(),
                colorscale="Viridis",
                colorbar=dict(title="Mean score"),
                hovertemplate=(
                    "Organ | cell type: %{y}<br>"
                    "Score: %{z:.3f}<extra></extra>"
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
            title=f"Top organ-cell-type sources for {program}<br><sup>{method}</sup>",
            template="plotly_white",
            margin=dict(l=430, r=60, t=100, b=80),
            xaxis=dict(showticklabels=False),
        )

        save_figure(
            fig,
            f"tabula_sapiens_top_organ_celltypes_{method}_{safe_program}",
            width=1100,
            height=950,
        )


def create_interpretation_summary(
    global_summary: pd.DataFrame,
    availability: pd.DataFrame,
    top_celltypes: pd.DataFrame,
    top_organ_celltypes: pd.DataFrame,
) -> None:
    report_path = OUTPUT_DIR / "tabula_sapiens_route_b_summary.md"

    gs = global_summary.iloc[0].to_dict()

    lines: List[str] = []

    lines.append("# Tabula Sapiens Route B Summary\n")
    lines.append("## Purpose\n")
    lines.append(
        "This analysis identifies the cellular sources of the nine curated ECM programs using donor × organ × cell-type pseudobulk profiles from Tabula Sapiens."
    )

    lines.append("\n## Key numbers\n")
    lines.append(f"- Source cells: {gs.get('n_cells_used')}\n")
    lines.append(f"- Matched Matrisome genes: {gs.get('n_matched_matrisome_genes')}\n")
    lines.append(f"- Pseudobulk groups: {gs.get('n_pseudobulk_groups')}\n")
    lines.append(f"- Donors: {gs.get('n_donors')}\n")
    lines.append(f"- Organs: {gs.get('n_organs')}\n")
    lines.append(f"- Cell types: {gs.get('n_cell_types')}\n")
    lines.append(f"- Compartments: {gs.get('n_compartments')}\n")
    lines.append(f"- Methods: {gs.get('n_methods')}\n")

    lines.append("\n## Program gene availability\n")
    for row in availability.itertuples():
        lines.append(
            f"- **{row.ecm_program}**: {row.n_available_genes}/{row.n_program_genes} genes available "
            f"({row.availability_fraction:.2f})."
        )

    lines.append("\n## Robust top cell types per program\n")
    for method in sorted(top_celltypes["method"].unique()):
        lines.append(f"\n### Method: {method}\n")
        method_df = top_celltypes[top_celltypes["method"].eq(method)]

        for program in PROGRAM_ORDER:
            sub = method_df[
                (method_df["ecm_program"].eq(program))
                & (method_df["rank"] <= 5)
            ]

            if sub.empty:
                continue

            entries = [
                f"{row.cell_type} ({row.compartment}, {row.mean_score:.2f})"
                for row in sub.itertuples()
            ]

            lines.append(f"- **{program}**: " + "; ".join(entries))

    lines.append("\n## Interpretation\n")
    lines.append(
        "The major finding is that most ECM programs are enriched in stromal, fibroblastic, mesenchymal, endothelial, or specialized tissue-supporting cell types. This indicates that tissue-level ECM programs reflect biologically plausible cellular ECM sources rather than arbitrary tissue labels."
    )

    lines.append("\n## Important caution\n")
    lines.append(
        "Some high-scoring cell types are supported by few donors or pseudobulk groups. The robust filtered tables should be preferred over unfiltered rankings. Smart-seq2 and 10X should also be interpreted separately because they differ in capture properties."
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAVED] {report_path}")


def main() -> None:
    ensure_dirs()

    global_summary, availability, compartment, celltype, organ_celltype = load_inputs()

    # Robust thresholds, selected to avoid one-donor artifacts while retaining rare ECM cell types.
    min_donors = 2
    min_pseudobulk = 2
    min_cells = 100

    top_celltypes = make_top_celltypes(
        celltype=celltype,
        min_donors=min_donors,
        min_pseudobulk=min_pseudobulk,
        min_cells=min_cells,
        top_n=15,
    )

    top_organ_celltypes = make_top_organ_celltypes(
        organ_celltype=organ_celltype,
        min_donors=min_donors,
        min_pseudobulk=min_pseudobulk,
        min_cells=min_cells,
        top_n=30,
    )

    top_celltypes.to_csv(
        TABLE_DIR / "tabula_sapiens_top_robust_celltypes_per_program.csv",
        index=False,
    )

    top_organ_celltypes.to_csv(
        TABLE_DIR / "tabula_sapiens_top_robust_organ_celltypes_per_program.csv",
        index=False,
    )

    for method in sorted(compartment["method"].unique()):
        matrix = make_compartment_matrix(compartment=compartment, method=method)
        matrix.to_csv(TABLE_DIR / f"tabula_sapiens_compartment_program_matrix_{method}.csv")
        plot_compartment_heatmap(matrix, method=method)

        plot_top_celltypes(top_celltypes, method=method)
        plot_top_organ_celltype_heatmaps(top_organ_celltypes, method=method)

    create_interpretation_summary(
        global_summary=global_summary,
        availability=availability,
        top_celltypes=top_celltypes,
        top_organ_celltypes=top_organ_celltypes,
    )

    print("\n[DONE]")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Tables:        {TABLE_DIR}")
    print(f"HTML figures:  {HTML_DIR}")
    print(f"PNG figures:   {PNG_DIR}")


if __name__ == "__main__":
    main()