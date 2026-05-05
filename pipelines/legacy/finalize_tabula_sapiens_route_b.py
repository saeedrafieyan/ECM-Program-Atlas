from __future__ import annotations

from pathlib import Path
import shutil
import json
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go


ANALYSIS_DIR = Path("outputs/tabula_sapiens_pseudobulk/analysis")
TABLE_DIR = ANALYSIS_DIR / "tables"

PROCESSED_DIR = Path("data/processed/tabula_sapiens")

OUTPUT_DIR = Path("outputs/tabula_sapiens_pseudobulk/final_route_b")
FINAL_TABLE_DIR = OUTPUT_DIR / "tables"
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
    for folder in [FINAL_TABLE_DIR, HTML_DIR, PNG_DIR, REPORT_DIR]:
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
        print(f"[WARNING] PNG export failed for {name}: {exc}")

    print(f"[SAVED] {html_path}")


def load_inputs() -> dict[str, pd.DataFrame]:
    paths = {
        "global_summary": PROCESSED_DIR / "tabula_sapiens_global_summary.csv",
        "availability": PROCESSED_DIR / "tabula_sapiens_program_gene_availability.csv",
        "compartment_10x": TABLE_DIR / "tabula_sapiens_compartment_program_matrix_10X.csv",
        "compartment_smartseq2": TABLE_DIR / "tabula_sapiens_compartment_program_matrix_smartseq2.csv",
        "top_celltypes": TABLE_DIR / "tabula_sapiens_top_robust_celltypes_per_program.csv",
        "top_organ_celltypes": TABLE_DIR / "tabula_sapiens_top_robust_organ_celltypes_per_program.csv",
    }

    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required Route B files:\n" + "\n".join(missing))

    return {
        name: pd.read_csv(path, index_col=0) if "compartment" in name else pd.read_csv(path)
        for name, path in paths.items()
    }


def make_consensus_celltype_table(top_celltypes: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize robust top cell-type sources per ECM program and method.
    Adds a simple cross-method consensus flag if a cell type appears in top ranks for both methods.
    """
    df = top_celltypes.copy()

    # Keep top 10 per program/method for a compact consensus table.
    df = df[df["rank"] <= 10].copy()

    records = []

    for program in PROGRAM_ORDER:
        sub = df[df["ecm_program"].eq(program)].copy()

        if sub.empty:
            continue

        for cell_type, group in sub.groupby("cell_type"):
            methods = sorted(group["method"].unique().tolist())
            compartments = sorted(group["compartment"].astype(str).unique().tolist())

            records.append(
                {
                    "ecm_program": program,
                    "cell_type": cell_type,
                    "methods_detected": "; ".join(methods),
                    "n_methods": len(methods),
                    "compartments": "; ".join(compartments),
                    "best_rank": int(group["rank"].min()),
                    "mean_of_top_scores": float(group["mean_score"].mean()),
                    "max_score": float(group["mean_score"].max()),
                    "max_n_donors": int(group["n_donors"].max()),
                    "max_n_pseudobulk": int(group["n_pseudobulk"].max()),
                    "max_n_cells_total": int(group["n_cells_total"].max()),
                    "cross_method_consensus": len(methods) >= 2,
                }
            )

    consensus = pd.DataFrame(records)

    consensus["ecm_program"] = pd.Categorical(
        consensus["ecm_program"],
        categories=PROGRAM_ORDER,
        ordered=True,
    )

    consensus = consensus.sort_values(
        ["ecm_program", "cross_method_consensus", "best_rank", "max_score"],
        ascending=[True, False, True, False],
    )

    return consensus


def make_program_source_summary(consensus: pd.DataFrame) -> pd.DataFrame:
    records = []

    for program in PROGRAM_ORDER:
        sub = consensus[consensus["ecm_program"].astype(str).eq(program)].copy()

        if sub.empty:
            continue

        consensus_sub = sub[sub["cross_method_consensus"]].head(5)
        top_sub = sub.head(8)

        records.append(
            {
                "ecm_program": program,
                "n_top_celltypes": sub.shape[0],
                "n_cross_method_consensus_celltypes": int(sub["cross_method_consensus"].sum()),
                "cross_method_consensus_celltypes": "; ".join(
                    consensus_sub["cell_type"].astype(str).tolist()
                ),
                "top_celltypes_overall": "; ".join(
                    [
                        f"{row.cell_type} ({row.compartments}, {row.max_score:.2f})"
                        for row in top_sub.itertuples()
                    ]
                ),
            }
        )

    return pd.DataFrame(records)


def plot_compartment_heatmap(matrix: pd.DataFrame, method: str) -> None:
    matrix = matrix.copy()
    matrix = matrix.loc[[p for p in PROGRAM_ORDER if p in matrix.index]]

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
                "Mean z-score: %{z:.3f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=f"Tabula Sapiens ECM program enrichment by compartment<br><sup>{method}</sup>",
        template="plotly_white",
        margin=dict(l=330, r=60, t=100, b=120),
    )

    save_figure(fig, f"route_b_compartment_program_heatmap_{method}", width=1350, height=850)


def plot_consensus_celltypes(consensus: pd.DataFrame) -> None:
    # Plot top 8 cell types per program, using best available score.
    rows = []

    for program in PROGRAM_ORDER:
        sub = consensus[consensus["ecm_program"].astype(str).eq(program)].copy()
        sub = sub.sort_values(
            ["cross_method_consensus", "max_score"],
            ascending=[False, False],
        ).head(8)

        for row in sub.itertuples():
            rows.append(
                {
                    "ecm_program": program,
                    "cell_type": row.cell_type,
                    "score": row.max_score,
                    "consensus": "both methods" if row.cross_method_consensus else "one method",
                    "label": f"{row.cell_type} [{row.consensus if hasattr(row, 'consensus') else ''}]",
                    "methods": row.methods_detected,
                    "compartments": row.compartments,
                    "n_donors": row.max_n_donors,
                    "n_cells": row.max_n_cells_total,
                }
            )

    plot_df = pd.DataFrame(rows)

    # One figure per program, more readable than one huge figure.
    for program in PROGRAM_ORDER:
        sub = plot_df[plot_df["ecm_program"].eq(program)].copy()

        if sub.empty:
            continue

        sub = sub.sort_values("score", ascending=True)

        fig = go.Figure(
            data=go.Bar(
                x=sub["score"],
                y=sub["cell_type"],
                orientation="h",
                marker=dict(
                    opacity=0.9,
                ),
                customdata=sub[["methods", "compartments", "n_donors", "n_cells", "consensus"]],
                hovertemplate=(
                    "Cell type: %{y}<br>"
                    "Score: %{x:.3f}<br>"
                    "Methods: %{customdata[0]}<br>"
                    "Compartments: %{customdata[1]}<br>"
                    "Max donors: %{customdata[2]}<br>"
                    "Max cells: %{customdata[3]}<br>"
                    "Consensus: %{customdata[4]}<extra></extra>"
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
            title=f"Robust cell-type sources for {program}<br><sup>Tabula Sapiens donor × organ × cell-type pseudobulk</sup>",
            xaxis_title="Maximum robust mean ECM program score",
            yaxis_title="",
            template="plotly_white",
            margin=dict(l=330, r=60, t=100, b=80),
        )

        save_figure(fig, f"route_b_consensus_celltypes_{safe_program}", width=1300, height=700)


def write_report(
    global_summary: pd.DataFrame,
    availability: pd.DataFrame,
    program_source_summary: pd.DataFrame,
) -> None:
    gs = global_summary.iloc[0].to_dict()

    report_path = REPORT_DIR / "route_b_final_summary.md"

    lines = []
    lines.append("# Route B Final Summary, Tabula Sapiens Cell-Type Source Validation\n")
    lines.append("## Purpose\n")
    lines.append(
        "Route B tests which cell types contribute to the curated ECM programs using Tabula Sapiens donor × organ × cell-type pseudobulk profiles."
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

    lines.append("\n## Gene availability\n")
    for row in availability.itertuples():
        lines.append(
            f"- **{row.ecm_program}**: {row.n_available_genes}/{row.n_program_genes} genes "
            f"({row.availability_fraction:.2f})."
        )

    lines.append("\n## Main program source findings\n")
    for row in program_source_summary.itertuples():
        lines.append(f"### {row.ecm_program}\n")
        lines.append(f"- Cross-method consensus cell types: {row.cross_method_consensus_celltypes if row.cross_method_consensus_celltypes else 'None detected'}\n")
        lines.append(f"- Top robust cell types: {row.top_celltypes_overall}\n")

    lines.append("\n## Interpretation\n")
    lines.append(
        "Route B supports the biological plausibility of the ECM programs by showing that many programs are enriched in expected ECM-producing compartments, especially stromal, fibroblastic, mesenchymal, endothelial, and tissue-supporting cell types. This addresses the limitation of bulk tissue analysis by identifying likely cellular sources of the tissue-level ECM programs."
    )

    lines.append("\n## Cautions\n")
    lines.append("1. Tabula Sapiens remains transcriptomic, not protein-level ECM deposition.\n")
    lines.append("2. 10X and Smart-seq2 differ and should be interpreted separately.\n")
    lines.append("3. Some rare cell types remain supported by limited donor numbers even after filtering.\n")
    lines.append("4. Spatial localization is still missing, and should be addressed using spatial transcriptomics or MatriSpace-like analysis later.\n")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAVED] {report_path}")


def make_manifest() -> None:
    files = []

    for path in list(FINAL_TABLE_DIR.glob("*")) + list(HTML_DIR.glob("*")) + list(PNG_DIR.glob("*")) + list(REPORT_DIR.glob("*")):
        files.append(
            {
                "file": str(path),
                "size_mb": path.stat().st_size / (1024 ** 2),
            }
        )

    pd.DataFrame(files).to_csv(OUTPUT_DIR / "route_b_final_manifest.csv", index=False)

    metadata = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "route": "B",
        "description": "Tabula Sapiens cell-type source validation of curated ECM programs",
    }

    with open(OUTPUT_DIR / "route_b_final_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main() -> None:
    ensure_dirs()

    data = load_inputs()

    consensus = make_consensus_celltype_table(data["top_celltypes"])
    program_source_summary = make_program_source_summary(consensus)

    consensus.to_csv(FINAL_TABLE_DIR / "route_b_consensus_celltype_sources.csv", index=False)
    program_source_summary.to_csv(FINAL_TABLE_DIR / "route_b_program_source_summary.csv", index=False)

    data["global_summary"].to_csv(FINAL_TABLE_DIR / "route_b_global_summary.csv", index=False)
    data["availability"].to_csv(FINAL_TABLE_DIR / "route_b_program_gene_availability.csv", index=False)
    data["top_celltypes"].to_csv(FINAL_TABLE_DIR / "route_b_top_robust_celltypes_per_program.csv", index=False)
    data["top_organ_celltypes"].to_csv(FINAL_TABLE_DIR / "route_b_top_robust_organ_celltypes_per_program.csv", index=False)

    plot_compartment_heatmap(data["compartment_10x"], method="10X")
    plot_compartment_heatmap(data["compartment_smartseq2"], method="smartseq2")
    plot_consensus_celltypes(consensus)

    write_report(
        global_summary=data["global_summary"],
        availability=data["availability"],
        program_source_summary=program_source_summary,
    )

    make_manifest()

    print("\n[DONE]")
    print(f"Final Route B folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()