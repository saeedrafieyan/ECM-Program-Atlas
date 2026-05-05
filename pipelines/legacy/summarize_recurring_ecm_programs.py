from __future__ import annotations

from pathlib import Path
import re
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
    "recurring_ecm_programs"
)


def assign_ecm_program(module_name: str, interpretation: str) -> str:
    """
    Map curated NMF module names to broader recurring ECM programs.
    This is a manually curated biological grouping.
    """
    text = f"{module_name} {interpretation}".lower()

    if any(term in text for term in ["cns", "neural", "retinal"]):
        if "retinal" in text:
            return "Retinal/sensory ECM"
        return "CNS/neural ECM"

    if any(term in text for term in ["epithelial", "mucosal", "gi-mucosal", "basement membrane"]):
        if any(term in text for term in ["renal", "endothelial", "kidney"]):
            return "Renal/endothelial basement membrane ECM"
        if "reproductive" in text:
            return "Reproductive-specialized ECM"
        return "Epithelial/mucosal basement membrane ECM"

    if any(term in text for term in ["vascular", "connective", "stromal", "interstitial", "fibroblastic"]):
        if "remodeling" in text:
            return "Stromal remodeling ECM"
        return "Vascular/stromal/interstitial ECM"

    if any(term in text for term in ["immune", "lymphoid", "hematopoietic"]):
        return "Immune/lymphoid remodeling ECM"

    if any(term in text for term in ["hepatic", "plasma", "liver"]):
        return "Hepatic/plasma-associated ECM"

    if any(term in text for term in ["reproductive", "secretory", "epididymis", "seminal"]):
        return "Reproductive-specialized ECM"

    if any(term in text for term in ["placental", "placenta"]):
        return "Placental/stromal ECM"

    if any(term in text for term in ["endocrine"]):
        return "Endocrine-associated ECM"

    return "Other/specialized ECM"


def confidence_to_score(confidence: str) -> int:
    mapping = {
        "Low": 1,
        "Moderate": 2,
        "Moderate-high": 3,
        "High": 4,
        "Very high": 5,
    }
    return mapping.get(str(confidence), 0)


def make_presence_matrix(df: pd.DataFrame) -> pd.DataFrame:
    presence = (
        df.groupby(["ecm_program", "feature_set"])
        .size()
        .reset_index(name="n_modules")
    )

    matrix = presence.pivot_table(
        index="ecm_program",
        columns="feature_set",
        values="n_modules",
        fill_value=0,
        aggfunc="sum",
    )

    preferred_order = [
        "core_matrisome",
        "ecm_glycoproteins",
        "proteoglycans",
        "collagens",
    ]

    existing_order = [col for col in preferred_order if col in matrix.columns]
    matrix = matrix[existing_order]

    return matrix


def make_program_summary(df: pd.DataFrame) -> pd.DataFrame:
    records = []

    for program, group in df.groupby("ecm_program"):
        feature_sets = sorted(group["feature_set"].unique().tolist())
        modules = [
            f"{row.feature_set}:{row.component} ({row.module_name})"
            for row in group.itertuples()
        ]

        high_confidence = group[group["confidence_score"] >= 4]

        records.append(
            {
                "ecm_program": program,
                "n_modules": group.shape[0],
                "n_feature_sets": len(feature_sets),
                "feature_sets": "; ".join(feature_sets),
                "n_high_confidence_modules": high_confidence.shape[0],
                "modules": " | ".join(modules),
            }
        )

    summary = pd.DataFrame(records)
    summary = summary.sort_values(
        ["n_feature_sets", "n_modules", "n_high_confidence_modules"],
        ascending=False,
    )

    return summary


def make_feature_set_program_long_table(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        [
            "feature_set",
            "component",
            "ecm_program",
            "module_name",
            "confidence",
            "short_interpretation",
            "top_samples",
            "top_genes",
        ]
    ].sort_values(["ecm_program", "feature_set", "component"])


def plot_presence_heatmap(matrix: pd.DataFrame, output_path: Path) -> None:
    plot_df = matrix.copy()

    fig = px.imshow(
        plot_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        labels=dict(
            x="Feature set",
            y="Recurring ECM program",
            color="Number of modules",
        ),
        title=(
            "Recurring ECM programs across Matrisome feature sets<br>"
            "<sup>Values indicate number of NMF modules mapped to each program</sup>"
        ),
    )

    fig.update_layout(
        width=1000,
        height=max(550, 38 * plot_df.shape[0]),
        template="plotly_white",
    )

    fig.write_html(str(output_path), include_plotlyjs="cdn", full_html=True)


def plot_program_counts(summary: pd.DataFrame, output_path: Path) -> None:
    plot_df = summary.sort_values("n_modules", ascending=True)

    fig = px.bar(
        plot_df,
        x="n_modules",
        y="ecm_program",
        orientation="h",
        color="n_feature_sets",
        hover_data={
            "n_modules": True,
            "n_feature_sets": True,
            "n_high_confidence_modules": True,
            "feature_sets": True,
        },
        title="Number of NMF modules assigned to each recurring ECM program",
        labels={
            "n_modules": "Number of modules",
            "ecm_program": "ECM program",
            "n_feature_sets": "Feature sets",
        },
        template="plotly_white",
    )

    fig.update_layout(
        width=1000,
        height=max(550, 38 * plot_df.shape[0]),
    )

    fig.write_html(str(output_path), include_plotlyjs="cdn", full_html=True)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input annotation file not found: {INPUT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    required = [
        "dataset",
        "feature_set",
        "component",
        "module_name",
        "confidence",
        "short_interpretation",
        "top_samples",
        "top_genes",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Input file is missing columns: {missing}")

    df["confidence_score"] = df["confidence"].apply(confidence_to_score)
    df["ecm_program"] = df.apply(
        lambda row: assign_ecm_program(
            module_name=row["module_name"],
            interpretation=row["short_interpretation"],
        ),
        axis=1,
    )

    annotated_long = make_feature_set_program_long_table(df)
    program_summary = make_program_summary(df)
    presence_matrix = make_presence_matrix(df)

    annotated_long.to_csv(
        OUTPUT_DIR / "ecm_program_module_long_table.csv",
        index=False,
    )

    program_summary.to_csv(
        OUTPUT_DIR / "recurring_ecm_program_summary.csv",
        index=False,
    )

    presence_matrix.to_csv(
        OUTPUT_DIR / "recurring_ecm_program_presence_matrix.csv",
    )

    plot_presence_heatmap(
        matrix=presence_matrix,
        output_path=OUTPUT_DIR / "recurring_ecm_program_presence_heatmap.html",
    )

    plot_program_counts(
        summary=program_summary,
        output_path=OUTPUT_DIR / "recurring_ecm_program_counts.html",
    )

    print("[SAVED]")
    print(OUTPUT_DIR / "ecm_program_module_long_table.csv")
    print(OUTPUT_DIR / "recurring_ecm_program_summary.csv")
    print(OUTPUT_DIR / "recurring_ecm_program_presence_matrix.csv")
    print(OUTPUT_DIR / "recurring_ecm_program_presence_heatmap.html")
    print(OUTPUT_DIR / "recurring_ecm_program_counts.html")

    print("\n[PROGRAM SUMMARY]")
    print(program_summary)


if __name__ == "__main__":
    main()