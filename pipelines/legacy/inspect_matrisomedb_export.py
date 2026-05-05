from __future__ import annotations

from pathlib import Path
import pandas as pd


RAW_DIR = Path("data/raw/matrisomedb")

CANDIDATE_FILES = [
    RAW_DIR / "matrisomedb_human_export.tsv",
    RAW_DIR / "matrisomedb_human_export.csv",
    RAW_DIR / "matrisomedb_export.tsv",
    RAW_DIR / "matrisomedb_export.csv",
]


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file format: {path}")


def main() -> None:
    existing_files = [path for path in CANDIDATE_FILES if path.exists()]

    if not existing_files:
        print("[ERROR] No MatrisomeDB export found.")
        print("Expected one of:")
        for path in CANDIDATE_FILES:
            print(f"  {path}")
        return

    for path in existing_files:
        print("=" * 100)
        print(f"File: {path}")
        print(f"Size: {path.stat().st_size / (1024 * 1024):.2f} MB")

        df = read_table(path)

        print(f"Shape: {df.shape}")
        print("\nColumns:")
        for col in df.columns:
            print(f"  - {col}")

        print("\nPreview:")
        print(df.head(10))

        print("\nMissing values per column:")
        print(df.isna().sum().sort_values(ascending=False).head(30))

        print("\nUnique values preview:")
        for col in df.columns[:20]:
            try:
                uniques = df[col].dropna().astype(str).unique()[:10]
                print(f"{col}: {uniques}")
            except Exception:
                pass


if __name__ == "__main__":
    main()