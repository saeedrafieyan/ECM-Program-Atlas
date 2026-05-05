from __future__ import annotations

from pathlib import Path
import pandas as pd


BASE_DIR = Path("outputs/latent_baseline_embeddings")
DATASET = "rna_tissue_consensus"

FEATURE_SETS = [
    "core_matrisome",
    "ecm_glycoproteins",
    "proteoglycans",
    "collagens",
]


def main() -> None:
    records = []

    for feature_set in FEATURE_SETS:
        path = (
            BASE_DIR
            / DATASET
            / feature_set
            / "nmf_module_activity"
            / "nmf_module_summary.csv"
        )

        if not path.exists():
            print(f"[MISSING] {path}")
            continue

        df = pd.read_csv(path)
        df.insert(0, "dataset", DATASET)
        df.insert(1, "feature_set", feature_set)

        records.append(df)

    if not records:
        raise FileNotFoundError("No NMF module summary files were found.")

    combined = pd.concat(records, ignore_index=True)

    output_path = (
        BASE_DIR
        / DATASET
        / "combined_nmf_module_summary_core_glycoproteins_proteoglycans_collagens.csv"
    )

    combined.to_csv(output_path, index=False)

    print(f"[SAVED] {output_path}")
    print(combined.head())


if __name__ == "__main__":
    main()