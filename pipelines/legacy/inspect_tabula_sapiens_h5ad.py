from __future__ import annotations

from pathlib import Path
import pandas as pd
import anndata as ad


RAW_DIR = Path("data/raw/tabula_sapiens")
OUTPUT_DIR = Path("outputs/tabula_sapiens_pseudobulk/inspection")


def inspect_metadata_csv() -> None:
    metadata_files = sorted(RAW_DIR.glob("*metadata*.csv")) + sorted(RAW_DIR.glob("*Metadata*.csv"))

    for path in metadata_files:
        print("=" * 100)
        print(f"Metadata CSV: {path}")
        print(f"Size: {path.stat().st_size / (1024 ** 2):.2f} MB")

        df = pd.read_csv(path, nrows=10)
        print("Columns:")
        print(df.columns.tolist())
        print("\nPreview:")
        print(df.head())

        full_cols = pd.read_csv(path, nrows=0).columns.tolist()
        pd.DataFrame({"column": full_cols}).to_csv(
            OUTPUT_DIR / f"{path.stem}_columns.csv",
            index=False,
        )


def inspect_h5ad(path: Path) -> dict:
    print("=" * 100)
    print(f"File: {path}")
    print(f"Size: {path.stat().st_size / (1024 ** 3):.2f} GB")

    try:
        adata = ad.read_h5ad(path, backed="r")
    except Exception as exc:
        print(f"[ERROR] Could not read {path}: {exc}")
        return {
            "file": str(path),
            "status": "failed",
            "error": str(exc),
        }

    print(f"Shape: {adata.n_obs} cells × {adata.n_vars} genes")

    obs_cols = adata.obs.columns.astype(str).tolist()
    var_cols = adata.var.columns.astype(str).tolist()

    print("\nobs columns:")
    print(obs_cols)

    print("\nvar columns:")
    print(var_cols)

    print("\nobs preview:")
    print(adata.obs.head())

    print("\nvar preview:")
    print(adata.var.head())

    result = {
        "file": str(path),
        "status": "ok",
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "obs_columns": "; ".join(obs_cols),
        "var_columns": "; ".join(var_cols),
        "var_index_preview": "; ".join(adata.var_names[:20].astype(str)),
    }

    candidate_obs_cols = [
        "organ",
        "tissue",
        "anatomical_information",
        "cell_type",
        "cell_ontology_class",
        "cell_type_tissue",
        "free_annotation",
        "compartment",
        "donor",
        "donor_id",
        "method",
        "assay",
        "sex",
    ]

    for col in candidate_obs_cols:
        if col in adata.obs.columns:
            values = adata.obs[col].dropna().astype(str)
            result[f"n_unique_{col}"] = int(values.nunique())
            result[f"preview_{col}"] = "; ".join(values.unique()[:30])
            print(f"\n{col}: {values.nunique()} unique")
            print(values.value_counts().head(20))

    # Check layers without loading matrices.
    try:
        result["layers"] = "; ".join(list(adata.layers.keys()))
        print("\nlayers:")
        print(list(adata.layers.keys()))
    except Exception:
        result["layers"] = ""

    try:
        result["raw_exists"] = adata.raw is not None
        print("\nraw exists:", adata.raw is not None)
    except Exception:
        result["raw_exists"] = False

    adata.file.close()

    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    inspect_metadata_csv()

    files = sorted(RAW_DIR.glob("*.h5ad"))

    if not files:
        print(f"[ERROR] No .h5ad files found in {RAW_DIR}")
        return

    summaries = []

    for path in files:
        result = inspect_h5ad(path)
        summaries.append(result)

    summary_df = pd.DataFrame(summaries)
    output_path = OUTPUT_DIR / "tabula_sapiens_h5ad_inspection_summary.csv"
    summary_df.to_csv(output_path, index=False)

    print("\n[SAVED]")
    print(output_path)


if __name__ == "__main__":
    main()