from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go


PROJECT_ROOT = Path(".")
MATRISOMEDB_DIR = PROJECT_ROOT / "data" / "processed" / "matrisomedb"

CURATED_PROGRAM_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "latent_baseline_embeddings"
    / "rna_tissue_consensus"
    / "curated_recurring_ecm_programs"
    / "combined_nmf_module_annotations_curated_programs.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "matrisomedb_validation"
TABLE_DIR = OUTPUT_DIR / "tables"
HTML_DIR = OUTPUT_DIR / "figures" / "html"
PNG_DIR = OUTPUT_DIR / "figures" / "png"


CONDITION_GROUPS = [
    "all_samples",
    "normal_like",
    "disease_like",
    "uncertain",
]


MATRIX_FILES = {
    "mean_log_nsaf": "tissue_gene_mean_log_nsaf.csv",
    "max_log_nsaf": "tissue_gene_max_log_nsaf.csv",
    "detection_count": "tissue_gene_detection_count.csv",
    "binary_detection": "tissue_gene_binary_detection.csv",
}


TISSUE_NAME_MAP = {
    "blood vessel": "Blood Vessel",
    "skin": "Skin",
    "stomach": "Stomach",
    "colon": "Colon",
    "kidney": "Kidney",
    "lung": "Lung",
    "liver": "Liver",
    "ovary": "Ovary",
    "prostate": "Prostate",
    "fallopian tube": "Fallopian Tube",
    "breast": "Breast",
    "tooth": "Tooth",
    "omentum": "Omentum",
    "eye": "Eye",
    "retina": "Eye",
}


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


def normalize_gene(gene: str) -> str:
    return str(gene).strip().upper()


def normalize_tissue_name(tissue: str) -> str:
    tissue = str(tissue).strip()
    key = tissue.lower()
    return TISSUE_NAME_MAP.get(key, tissue)


def split_comma_list(value: str) -> List[str]:
    if pd.isna(value):
        return []

    return [
        item.strip()
        for item in str(value).split(",")
        if item.strip()
    ]


def split_semicolon_list(value: str) -> List[str]:
    if pd.isna(value):
        return []

    return [
        item.strip()
        for item in str(value).split(";")
        if item.strip()
    ]


def load_curated_programs(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing curated program annotation file:\n{path}\n"
            "Run src/create_curated_ecm_program_summary.py first."
        )

    df = pd.read_csv(path)

    required = [
        "feature_set",
        "component",
        "module_name",
        "ecm_program_curated",
        "confidence",
        "top_samples",
        "top_genes",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Curated program file is missing columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    return df


def build_program_gene_sets(
    curated_df: pd.DataFrame,
    max_genes_per_module: int | None = None,
) -> pd.DataFrame:
    records = []

    for program, group in curated_df.groupby("ecm_program_curated"):
        genes: List[str] = []
        modules: List[str] = []
        rna_tissues: List[str] = []

        for row in group.itertuples():
            module_genes = split_comma_list(row.top_genes)

            if max_genes_per_module is not None:
                module_genes = module_genes[:max_genes_per_module]

            genes.extend([normalize_gene(gene) for gene in module_genes])

            modules.append(f"{row.feature_set}:{row.component}")

            top_samples = split_semicolon_list(row.top_samples)
            rna_tissues.extend([normalize_tissue_name(t) for t in top_samples])

        unique_genes = sorted(set(genes))
        unique_tissues = sorted(set(rna_tissues))

        records.append(
            {
                "ecm_program": program,
                "n_reference_modules": len(modules),
                "reference_modules": "; ".join(modules),
                "n_program_genes": len(unique_genes),
                "program_genes": ", ".join(unique_genes),
                "n_rna_top_tissues": len(unique_tissues),
                "rna_top_tissues": "; ".join(unique_tissues),
            }
        )

    program_df = pd.DataFrame(records)

    program_df["ecm_program"] = pd.Categorical(
        program_df["ecm_program"],
        categories=PROGRAM_ORDER,
        ordered=True,
    )

    program_df = program_df.sort_values("ecm_program")

    return program_df


def load_matrisomedb_matrix(condition_group: str, matrix_name: str) -> pd.DataFrame | None:
    file_name = MATRIX_FILES[matrix_name]
    path = MATRISOMEDB_DIR / condition_group / file_name

    if not path.exists():
        print(f"[MISSING] {path}")
        return None

    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    df.columns = [normalize_gene(col) for col in df.columns]

    # MatrisomeDB tissues use title case, keep them readable.
    df.index = [normalize_tissue_name(tissue) for tissue in df.index]

    # If tissue names collapse after normalization, aggregate by mean.
    df = df.groupby(df.index).mean()

    return df


def compute_gene_support(
    program_df: pd.DataFrame,
    matrix: pd.DataFrame,
    condition_group: str,
    matrix_name: str,
) -> pd.DataFrame:
    detected_genes = set(matrix.columns)

    records = []

    for row in program_df.itertuples():
        program_genes = set(split_comma_list(row.program_genes))
        detected = sorted(program_genes.intersection(detected_genes))
        missing = sorted(program_genes.difference(detected_genes))

        records.append(
            {
                "condition_group": condition_group,
                "matrix_name": matrix_name,
                "ecm_program": row.ecm_program,
                "n_program_genes": len(program_genes),
                "n_detected_genes": len(detected),
                "n_missing_genes": len(missing),
                "protein_detection_coverage": (
                    len(detected) / len(program_genes)
                    if program_genes else np.nan
                ),
                "detected_genes": ", ".join(detected),
                "missing_genes": ", ".join(missing),
            }
        )

    return pd.DataFrame(records)


def compute_program_tissue_scores(
    program_df: pd.DataFrame,
    matrix: pd.DataFrame,
    condition_group: str,
    matrix_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    score_records = []
    top_records = []

    for row in program_df.itertuples():
        program_genes = set(split_comma_list(row.program_genes))
        available_genes = sorted(program_genes.intersection(set(matrix.columns)))

        if not available_genes:
            continue

        program_scores = matrix[available_genes].mean(axis=1)

        for tissue, score in program_scores.items():
            score_records.append(
                {
                    "condition_group": condition_group,
                    "matrix_name": matrix_name,
                    "ecm_program": row.ecm_program,
                    "tissue": tissue,
                    "program_score": float(score),
                    "n_available_program_genes": len(available_genes),
                }
            )

        top_tissues = program_scores.sort_values(ascending=False).head(10)

        rna_top_tissues = set(split_semicolon_list(row.rna_top_tissues))
        protein_top_tissues = set(top_tissues.index.astype(str).tolist())

        overlap = sorted(rna_top_tissues.intersection(protein_top_tissues))

        top_records.append(
            {
                "condition_group": condition_group,
                "matrix_name": matrix_name,
                "ecm_program": row.ecm_program,
                "n_available_program_genes": len(available_genes),
                "top_protein_tissues": "; ".join(top_tissues.index.astype(str).tolist()),
                "top_rna_tissues": row.rna_top_tissues,
                "n_rna_protein_top_tissue_overlap": len(overlap),
                "rna_protein_top_tissue_overlap": "; ".join(overlap),
                "mean_top10_protein_score": float(top_tissues.mean()),
                "max_top10_protein_score": float(top_tissues.max()),
            }
        )

    score_df = pd.DataFrame(score_records)
    top_df = pd.DataFrame(top_records)

    return score_df, top_df


def plot_program_tissue_heatmap(
    score_df: pd.DataFrame,
    condition_group: str,
    matrix_name: str,
) -> None:
    if score_df.empty:
        return

    pivot = score_df.pivot_table(
        index="ecm_program",
        columns="tissue",
        values="program_score",
        aggfunc="mean",
        fill_value=0.0,
    )

    # Reorder programs.
    available_programs = [p for p in PROGRAM_ORDER if p in pivot.index]
    pivot = pivot.loc[available_programs]

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale="Viridis",
            colorbar=dict(title="Program score"),
            hovertemplate=(
                "Program: %{y}<br>"
                "Tissue: %{x}<br>"
                "Score: %{z:.4f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=(
            f"MatrisomeDB protein-level ECM program scores<br>"
            f"<sup>{condition_group}, {matrix_name}</sup>"
        ),
        template="plotly_white",
        width=1300,
        height=max(650, 55 * pivot.shape[0]),
        margin=dict(l=260, r=60, t=100, b=150),
        xaxis=dict(tickangle=35),
    )

    base_name = f"matrisomedb_program_heatmap__{condition_group}__{matrix_name}"

    html_path = HTML_DIR / f"{base_name}.html"
    png_path = PNG_DIR / f"{base_name}.png"

    fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)

    try:
        fig.write_image(str(png_path), scale=3)
    except Exception as exc:
        print(f"[WARNING] Could not export PNG for {base_name}: {exc}")

    print(f"[SAVED] {html_path}")
    print(f"[SAVED] {png_path}")


def plot_gene_support_barplot(gene_support_df: pd.DataFrame) -> None:
    if gene_support_df.empty:
        return

    # Most useful first view: all_samples, mean_log_nsaf.
    plot_df = gene_support_df[
        (gene_support_df["condition_group"].eq("all_samples"))
        & (gene_support_df["matrix_name"].eq("mean_log_nsaf"))
    ].copy()

    if plot_df.empty:
        return

    plot_df["ecm_program"] = pd.Categorical(
        plot_df["ecm_program"],
        categories=PROGRAM_ORDER,
        ordered=True,
    )
    plot_df = plot_df.sort_values("ecm_program")

    fig = go.Figure(
        data=go.Bar(
            x=plot_df["protein_detection_coverage"],
            y=plot_df["ecm_program"],
            orientation="h",
            customdata=plot_df[
                [
                    "n_detected_genes",
                    "n_program_genes",
                    "detected_genes",
                ]
            ],
            hovertemplate=(
                "Program: %{y}<br>"
                "Coverage: %{x:.2f}<br>"
                "Detected genes: %{customdata[0]} / %{customdata[1]}<br>"
                "Detected: %{customdata[2]}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=(
            "MatrisomeDB protein detection coverage of RNA-derived ECM programs<br>"
            "<sup>All human MatrisomeDB samples, mean log NSAF matrix</sup>"
        ),
        xaxis_title="Protein detection coverage",
        yaxis_title="",
        template="plotly_white",
        width=1200,
        height=700,
        margin=dict(l=280, r=60, t=100, b=80),
    )

    base_name = "matrisomedb_program_gene_detection_coverage"

    html_path = HTML_DIR / f"{base_name}.html"
    png_path = PNG_DIR / f"{base_name}.png"

    fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)

    try:
        fig.write_image(str(png_path), scale=3)
    except Exception as exc:
        print(f"[WARNING] Could not export PNG for {base_name}: {exc}")

    print(f"[SAVED] {html_path}")
    print(f"[SAVED] {png_path}")


def main() -> None:
    ensure_dirs()

    curated_df = load_curated_programs(CURATED_PROGRAM_FILE)
    program_df = build_program_gene_sets(
        curated_df=curated_df,
        max_genes_per_module=30,
    )

    program_df.to_csv(TABLE_DIR / "rna_derived_ecm_program_gene_sets.csv", index=False)

    all_gene_support = []
    all_scores = []
    all_top_tissues = []

    for condition_group in CONDITION_GROUPS:
        for matrix_name in MATRIX_FILES:
            matrix = load_matrisomedb_matrix(
                condition_group=condition_group,
                matrix_name=matrix_name,
            )

            if matrix is None:
                continue

            print(
                f"[PROCESSING] {condition_group} | {matrix_name} | "
                f"matrix={matrix.shape}"
            )

            gene_support = compute_gene_support(
                program_df=program_df,
                matrix=matrix,
                condition_group=condition_group,
                matrix_name=matrix_name,
            )
            all_gene_support.append(gene_support)

            score_df, top_df = compute_program_tissue_scores(
                program_df=program_df,
                matrix=matrix,
                condition_group=condition_group,
                matrix_name=matrix_name,
            )

            all_scores.append(score_df)
            all_top_tissues.append(top_df)

            if matrix_name in ["mean_log_nsaf", "binary_detection"]:
                plot_program_tissue_heatmap(
                    score_df=score_df,
                    condition_group=condition_group,
                    matrix_name=matrix_name,
                )

    gene_support_df = pd.concat(all_gene_support, ignore_index=True)
    scores_df = pd.concat(all_scores, ignore_index=True)
    top_tissues_df = pd.concat(all_top_tissues, ignore_index=True)

    gene_support_df.to_csv(
        TABLE_DIR / "matrisomedb_program_gene_support_summary.csv",
        index=False,
    )

    scores_df.to_csv(
        TABLE_DIR / "matrisomedb_program_tissue_scores.csv",
        index=False,
    )

    top_tissues_df.to_csv(
        TABLE_DIR / "matrisomedb_program_top_tissues.csv",
        index=False,
    )

    plot_gene_support_barplot(gene_support_df)

    print("\n[SAVED TABLES]")
    print(TABLE_DIR / "rna_derived_ecm_program_gene_sets.csv")
    print(TABLE_DIR / "matrisomedb_program_gene_support_summary.csv")
    print(TABLE_DIR / "matrisomedb_program_tissue_scores.csv")
    print(TABLE_DIR / "matrisomedb_program_top_tissues.csv")
    print("\n[SAVED FIGURES]")
    print(HTML_DIR)
    print(PNG_DIR)


if __name__ == "__main__":
    main()