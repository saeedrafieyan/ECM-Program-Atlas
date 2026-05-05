from __future__ import annotations

from pathlib import Path
import pandas as pd
import plotly.express as px


INPUT_PATH = Path(
    "outputs/latent_baseline_embeddings/"
    "rna_tissue_consensus/"
    "combined_nmf_module_annotations.csv"
)

OUTPUT_DIR = Path(
    "outputs/latent_baseline_embeddings/"
    "rna_tissue_consensus/"
    "curated_recurring_ecm_programs"
)


CURATED_PROGRAMS = {
    # Core Matrisome
    ("core_matrisome", "NMF1"): "Vascular/stromal/interstitial ECM",
    ("core_matrisome", "NMF2"): "Retinal/sensory ECM",
    ("core_matrisome", "NMF3"): "Stromal remodeling ECM",
    ("core_matrisome", "NMF4"): "Reproductive-specialized ECM",
    ("core_matrisome", "NMF5"): "Epithelial/mucosal basement membrane ECM",
    ("core_matrisome", "NMF6"): "CNS/neural ECM",
    ("core_matrisome", "NMF7"): "Immune/lymphoid remodeling ECM",
    ("core_matrisome", "NMF8"): "Renal/endothelial basement membrane ECM",
    ("core_matrisome", "NMF9"): "Hepatic/plasma-associated ECM",
    ("core_matrisome", "NMF10"): "Vascular/stromal/interstitial ECM",

    # ECM Glycoproteins
    ("ecm_glycoproteins", "NMF1"): "Epithelial/mucosal basement membrane ECM",
    ("ecm_glycoproteins", "NMF2"): "Retinal/sensory ECM",
    ("ecm_glycoproteins", "NMF3"): "Stromal remodeling ECM",
    ("ecm_glycoproteins", "NMF4"): "Retinal/sensory ECM",
    ("ecm_glycoproteins", "NMF5"): "Vascular/stromal/interstitial ECM",
    ("ecm_glycoproteins", "NMF6"): "Epithelial/mucosal basement membrane ECM",
    ("ecm_glycoproteins", "NMF7"): "Immune/lymphoid remodeling ECM",
    ("ecm_glycoproteins", "NMF8"): "Hepatic/plasma-associated ECM",
    ("ecm_glycoproteins", "NMF9"): "CNS/neural ECM",
    ("ecm_glycoproteins", "NMF10"): "Reproductive-specialized ECM",

    # Proteoglycans
    ("proteoglycans", "NMF1"): "Epithelial/mucosal basement membrane ECM",
    ("proteoglycans", "NMF2"): "CNS/neural ECM",
    ("proteoglycans", "NMF3"): "Immune/lymphoid remodeling ECM",
    ("proteoglycans", "NMF4"): "Vascular/stromal/interstitial ECM",
    ("proteoglycans", "NMF5"): "Vascular/stromal/interstitial ECM",
    ("proteoglycans", "NMF6"): "Epithelial/mucosal basement membrane ECM",
    ("proteoglycans", "NMF7"): "Hepatic/plasma-associated ECM",
    ("proteoglycans", "NMF8"): "Reproductive-specialized ECM",
    ("proteoglycans", "NMF9"): "Retinal/sensory ECM",
    ("proteoglycans", "NMF10"): "Immune/lymphoid remodeling ECM",

    # Collagens
    ("collagens", "NMF1"): "Vascular/stromal/interstitial ECM",
    ("collagens", "NMF2"): "CNS/neural ECM",
    ("collagens", "NMF3"): "Epithelial/mucosal basement membrane ECM",
    ("collagens", "NMF4"): "Renal/endothelial basement membrane ECM",
    ("collagens", "NMF5"): "Immune/lymphoid remodeling ECM",
    ("collagens", "NMF6"): "Vascular/stromal/interstitial ECM",
    ("collagens", "NMF7"): "Retinal/sensory ECM",
    ("collagens", "NMF8"): "Hepatic/plasma-associated ECM",
    ("collagens", "NMF9"): "Retinal/sensory ECM",
    ("collagens", "NMF10"): "Epithelial/mucosal basement membrane ECM",
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


FEATURE_SET_ORDER = [
    "core_matrisome",
    "ecm_glycoproteins",
    "proteoglycans",
    "collagens",
]


def confidence_to_score(confidence: str) -> int:
    mapping = {
        "Low": 1,
        "Moderate": 2,
        "Moderate-high": 3,
        "High": 4,
        "Very high": 5,
    }
    return mapping.get(str(confidence), 0)


def make_curated_long_table(df: pd.DataFrame) -> pd.DataFrame:
    records = []

    for _, row in df.iterrows():
        feature_set = row["feature_set"]
        component = row["component"]
        key = (feature_set, component)

        curated_program = CURATED_PROGRAMS.get(key)

        if curated_program is None:
            curated_program = "Unassigned/specialized ECM"

        records.append(
            {
                "dataset": row["dataset"],
                "feature_set": feature_set,
                "component": component,
                "module_name": row["module_name"],
                "ecm_program_curated": curated_program,
                "confidence": row["confidence"],
                "confidence_score": confidence_to_score(row["confidence"]),
                "short_interpretation": row["short_interpretation"],
                "top_samples": row["top_samples"],
                "top_genes": row["top_genes"],
            }
        )

    curated_df = pd.DataFrame(records)

    curated_df["feature_set"] = pd.Categorical(
        curated_df["feature_set"],
        categories=FEATURE_SET_ORDER,
        ordered=True,
    )

    curated_df["ecm_program_curated"] = pd.Categorical(
        curated_df["ecm_program_curated"],
        categories=PROGRAM_ORDER + ["Unassigned/specialized ECM"],
        ordered=True,
    )

    curated_df = curated_df.sort_values(
        ["ecm_program_curated", "feature_set", "component"]
    )

    return curated_df


def make_presence_matrix(curated_df: pd.DataFrame) -> pd.DataFrame:
    presence = (
        curated_df.groupby(
            ["ecm_program_curated", "feature_set"],
            observed=False,
        )
        .size()
        .reset_index(name="n_modules")
    )

    matrix = presence.pivot_table(
        index="ecm_program_curated",
        columns="feature_set",
        values="n_modules",
        fill_value=0,
        aggfunc="sum",
        observed=False,
    )

    matrix = matrix.reindex(PROGRAM_ORDER)
    matrix = matrix[FEATURE_SET_ORDER]

    return matrix


def make_high_confidence_matrix(curated_df: pd.DataFrame) -> pd.DataFrame:
    high_df = curated_df[curated_df["confidence_score"] >= 4].copy()

    presence = (
        high_df.groupby(
            ["ecm_program_curated", "feature_set"],
            observed=False,
        )
        .size()
        .reset_index(name="n_high_confidence_modules")
    )

    matrix = presence.pivot_table(
        index="ecm_program_curated",
        columns="feature_set",
        values="n_high_confidence_modules",
        fill_value=0,
        aggfunc="sum",
        observed=False,
    )

    matrix = matrix.reindex(PROGRAM_ORDER)
    matrix = matrix[FEATURE_SET_ORDER]

    return matrix


def make_program_summary(curated_df: pd.DataFrame) -> pd.DataFrame:
    records = []

    for program, group in curated_df.groupby("ecm_program_curated", observed=False):
        if pd.isna(program):
            continue

        group = group.copy()

        if group.empty:
            continue

        feature_sets = sorted(
            group["feature_set"].astype(str).unique().tolist(),
            key=lambda x: FEATURE_SET_ORDER.index(x) if x in FEATURE_SET_ORDER else 999,
        )

        high_confidence = group[group["confidence_score"] >= 4]

        modules = [
            f"{row.feature_set}:{row.component} ({row.module_name})"
            for row in group.itertuples()
        ]

        representative_genes = []
        for genes in group["top_genes"].astype(str).tolist():
            representative_genes.extend(
                [gene.strip() for gene in genes.split(",")[:5]]
            )

        # Preserve order while removing duplicates.
        seen = set()
        representative_genes_unique = []
        for gene in representative_genes:
            if gene and gene not in seen:
                seen.add(gene)
                representative_genes_unique.append(gene)

        records.append(
            {
                "ecm_program_curated": program,
                "n_modules": group.shape[0],
                "n_feature_sets": len(feature_sets),
                "feature_sets": "; ".join(feature_sets),
                "n_high_confidence_modules": high_confidence.shape[0],
                "modules": " | ".join(modules),
                "representative_top_genes": ", ".join(
                    representative_genes_unique[:25]
                ),
            }
        )

    summary = pd.DataFrame(records)

    summary["ecm_program_curated"] = pd.Categorical(
        summary["ecm_program_curated"],
        categories=PROGRAM_ORDER,
        ordered=True,
    )

    summary = summary.sort_values(
        ["n_feature_sets", "n_modules", "n_high_confidence_modules"],
        ascending=False,
    )

    return summary


def plot_presence_heatmap(matrix: pd.DataFrame, output_path: Path) -> None:
    fig = px.imshow(
        matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        labels={
            "x": "Matrisome feature set",
            "y": "Curated recurring ECM program",
            "color": "Number of modules",
        },
        title=(
            "Curated recurring ECM programs across Matrisome feature sets<br>"
            "<sup>Values indicate number of NMF modules assigned to each program</sup>"
        ),
    )

    fig.update_layout(
        width=1100,
        height=700,
        template="plotly_white",
    )

    fig.write_html(str(output_path), include_plotlyjs="cdn", full_html=True)


def plot_high_confidence_heatmap(matrix: pd.DataFrame, output_path: Path) -> None:
    fig = px.imshow(
        matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Greens",
        labels={
            "x": "Matrisome feature set",
            "y": "Curated recurring ECM program",
            "color": "High-confidence modules",
        },
        title=(
            "High-confidence curated recurring ECM programs<br>"
            "<sup>High-confidence means confidence = High or Very high</sup>"
        ),
    )

    fig.update_layout(
        width=1100,
        height=700,
        template="plotly_white",
    )

    fig.write_html(str(output_path), include_plotlyjs="cdn", full_html=True)


def plot_program_counts(summary: pd.DataFrame, output_path: Path) -> None:
    plot_df = summary.sort_values("n_modules", ascending=True)

    fig = px.bar(
        plot_df,
        x="n_modules",
        y="ecm_program_curated",
        orientation="h",
        color="n_feature_sets",
        hover_data={
            "n_modules": True,
            "n_feature_sets": True,
            "n_high_confidence_modules": True,
            "feature_sets": True,
        },
        labels={
            "n_modules": "Number of modules",
            "ecm_program_curated": "Curated ECM program",
            "n_feature_sets": "Feature sets",
        },
        title="Curated recurring ECM program counts",
        template="plotly_white",
    )

    fig.update_layout(
        width=1100,
        height=700,
    )

    fig.write_html(str(output_path), include_plotlyjs="cdn", full_html=True)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    required_cols = [
        "dataset",
        "feature_set",
        "component",
        "module_name",
        "confidence",
        "short_interpretation",
        "top_samples",
        "top_genes",
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Input annotation file is missing columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    curated_df = make_curated_long_table(df)
    presence_matrix = make_presence_matrix(curated_df)
    high_confidence_matrix = make_high_confidence_matrix(curated_df)
    program_summary = make_program_summary(curated_df)

    curated_df.to_csv(
        OUTPUT_DIR / "combined_nmf_module_annotations_curated_programs.csv",
        index=False,
    )

    presence_matrix.to_csv(
        OUTPUT_DIR / "curated_ecm_program_presence_matrix.csv",
    )

    high_confidence_matrix.to_csv(
        OUTPUT_DIR / "curated_ecm_program_high_confidence_matrix.csv",
    )

    program_summary.to_csv(
        OUTPUT_DIR / "curated_recurring_ecm_program_summary.csv",
        index=False,
    )

    plot_presence_heatmap(
        matrix=presence_matrix,
        output_path=OUTPUT_DIR / "curated_ecm_program_presence_heatmap.html",
    )

    plot_high_confidence_heatmap(
        matrix=high_confidence_matrix,
        output_path=OUTPUT_DIR / "curated_ecm_program_high_confidence_heatmap.html",
    )

    plot_program_counts(
        summary=program_summary,
        output_path=OUTPUT_DIR / "curated_ecm_program_counts.html",
    )

    print("[SAVED]")
    print(OUTPUT_DIR / "combined_nmf_module_annotations_curated_programs.csv")
    print(OUTPUT_DIR / "curated_ecm_program_presence_matrix.csv")
    print(OUTPUT_DIR / "curated_ecm_program_high_confidence_matrix.csv")
    print(OUTPUT_DIR / "curated_recurring_ecm_program_summary.csv")
    print(OUTPUT_DIR / "curated_ecm_program_presence_heatmap.html")
    print(OUTPUT_DIR / "curated_ecm_program_high_confidence_heatmap.html")
    print(OUTPUT_DIR / "curated_ecm_program_counts.html")

    print("\n[CURATED PROGRAM SUMMARY]")
    print(program_summary)


if __name__ == "__main__":
    main()