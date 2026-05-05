from __future__ import annotations

from pathlib import Path
import pandas as pd
import plotly.express as px


INPUT_DIR = Path("outputs/latent_baseline_embeddings/cross_dataset_nmf_program_reproducibility")

BEST_MATCHES_PATH = INPUT_DIR / "best_module_reference_matches.csv"

OUTPUT_DIR = INPUT_DIR / "external_only"


REFERENCE_DATASET = "rna_tissue_consensus"


def make_program_summary(df: pd.DataFrame) -> pd.DataFrame:
    reproduced = df[df["is_reproduced_match"].astype(bool)].copy()

    if reproduced.empty:
        return pd.DataFrame()

    records = []

    for program, group in reproduced.groupby("assigned_ecm_program"):
        datasets = sorted(group["dataset"].unique().tolist())
        feature_sets = sorted(group["feature_set"].unique().tolist())

        records.append(
            {
                "ecm_program": program,
                "n_reproduced_modules": group.shape[0],
                "n_external_datasets": len(datasets),
                "external_datasets": "; ".join(datasets),
                "n_feature_sets": len(feature_sets),
                "feature_sets": "; ".join(feature_sets),
                "mean_overlap_coefficient": group["overlap_coefficient"].mean(),
                "mean_jaccard": group["jaccard"].mean(),
                "mean_n_overlapping_genes": group["n_overlapping_genes"].mean(),
            }
        )

    summary = pd.DataFrame(records)

    summary = summary.sort_values(
        ["n_external_datasets", "n_feature_sets", "n_reproduced_modules"],
        ascending=False,
    )

    return summary


def make_dataset_presence_matrix(df: pd.DataFrame) -> pd.DataFrame:
    reproduced = df[df["is_reproduced_match"].astype(bool)].copy()

    if reproduced.empty:
        return pd.DataFrame()

    presence = (
        reproduced.groupby(["assigned_ecm_program", "dataset"])
        .size()
        .reset_index(name="n_modules")
    )

    matrix = presence.pivot_table(
        index="assigned_ecm_program",
        columns="dataset",
        values="n_modules",
        fill_value=0,
        aggfunc="sum",
    )

    return matrix


def make_feature_set_presence_matrix(df: pd.DataFrame) -> pd.DataFrame:
    reproduced = df[df["is_reproduced_match"].astype(bool)].copy()

    if reproduced.empty:
        return pd.DataFrame()

    presence = (
        reproduced.groupby(["assigned_ecm_program", "feature_set"])
        .size()
        .reset_index(name="n_modules")
    )

    matrix = presence.pivot_table(
        index="assigned_ecm_program",
        columns="feature_set",
        values="n_modules",
        fill_value=0,
        aggfunc="sum",
    )

    return matrix


def plot_heatmap(matrix: pd.DataFrame, title: str, output_path: Path) -> None:
    if matrix.empty:
        return

    fig = px.imshow(
        matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        labels={
            "x": "Dataset / feature set",
            "y": "ECM program",
            "color": "Number of reproduced modules",
        },
        title=title,
    )

    fig.update_layout(
        width=1150,
        height=max(600, 40 * matrix.shape[0]),
        template="plotly_white",
    )

    fig.write_html(str(output_path), include_plotlyjs="cdn", full_html=True)


def main() -> None:
    if not BEST_MATCHES_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {BEST_MATCHES_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(BEST_MATCHES_PATH)

    external_df = df[df["dataset"].astype(str).ne(REFERENCE_DATASET)].copy()

    external_best_path = OUTPUT_DIR / "external_best_module_reference_matches.csv"
    external_df.to_csv(external_best_path, index=False)

    summary = make_program_summary(external_df)
    dataset_matrix = make_dataset_presence_matrix(external_df)
    feature_matrix = make_feature_set_presence_matrix(external_df)

    summary.to_csv(
        OUTPUT_DIR / "external_program_reproducibility_summary.csv",
        index=False,
    )

    dataset_matrix.to_csv(
        OUTPUT_DIR / "external_dataset_program_presence_matrix.csv",
    )

    feature_matrix.to_csv(
        OUTPUT_DIR / "external_feature_set_program_presence_matrix.csv",
    )

    plot_heatmap(
        matrix=dataset_matrix,
        title=(
            "External cross-dataset reproducibility of ECM programs<br>"
            "<sup>Reference dataset rna_tissue_consensus excluded</sup>"
        ),
        output_path=OUTPUT_DIR / "external_dataset_program_presence_heatmap.html",
    )

    plot_heatmap(
        matrix=feature_matrix,
        title=(
            "External cross-feature-set reproducibility of ECM programs<br>"
            "<sup>Reference dataset rna_tissue_consensus excluded</sup>"
        ),
        output_path=OUTPUT_DIR / "external_feature_set_program_presence_heatmap.html",
    )

    print("[SAVED]")
    print(external_best_path)
    print(OUTPUT_DIR / "external_program_reproducibility_summary.csv")
    print(OUTPUT_DIR / "external_dataset_program_presence_matrix.csv")
    print(OUTPUT_DIR / "external_feature_set_program_presence_matrix.csv")


if __name__ == "__main__":
    main()