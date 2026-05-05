from pathlib import Path
import zipfile
import pandas as pd


RAW_HPA_DIR = Path("data/raw/hpa")
MATRISOME_FILE = Path("data/raw/matrisome/human_matrisome.xlsx")


def inspect_zip_tsv(path: Path, n_rows: int = 5) -> None:
    print("=" * 80)
    print(f"File: {path}")

    if not path.exists():
        print("[MISSING]")
        return

    print(f"Size: {path.stat().st_size / (1024 * 1024):.2f} MB")

    with zipfile.ZipFile(path, "r") as zf:
        names = zf.namelist()
        print(f"Inside zip: {names}")

    df = pd.read_csv(path, sep="\t", compression="zip", nrows=n_rows)
    print("Columns:")
    print(df.columns.tolist())
    print("Preview:")
    print(df.head())


def inspect_matrisome(path: Path) -> None:
    print("=" * 80)
    print(f"Matrisome file: {path}")

    if not path.exists():
        print("[MISSING] Download the Homo sapiens matrisome file manually.")
        return

    df = pd.read_excel(path)
    print(f"Shape: {df.shape}")
    print("Columns:")
    print(df.columns.tolist())
    print("Preview:")
    print(df.head())


def main() -> None:
    hpa_files = sorted(RAW_HPA_DIR.glob("*.tsv.zip"))

    if not hpa_files:
        print("[WARNING] No HPA files found. Run src/download_hpa_data.py first.")

    for path in hpa_files:
        inspect_zip_tsv(path)

    inspect_matrisome(MATRISOME_FILE)


if __name__ == "__main__":
    main()