from __future__ import annotations

from pathlib import Path
import pandas as pd


RAW_DIR = Path("data/raw/gtex_v11_sample_level")


def find_file(patterns: list[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(RAW_DIR.glob(pattern))
        if matches:
            return matches[0]
    return None


def inspect_parquet(path: Path) -> None:
    print("=" * 100)
    print(f"Parquet file: {path}")
    print(f"Size: {path.stat().st_size / (1024 ** 3):.2f} GB")

    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)

        print(f"Number of row groups: {pf.num_row_groups}")
        print("\nSchema:")
        print(pf.schema)

        table = pf.read_row_group(0)
        df = table.to_pandas()

        print("\nFirst row group shape:", df.shape)
        print("\nColumns preview:")
        print(df.columns[:30].tolist())

        print("\nPreview:")
        print(df.head())

    except ImportError:
        print("[ERROR] pyarrow is not installed. Run: pip install pyarrow")
    except Exception as exc:
        print(f"[ERROR] Could not inspect parquet: {exc}")


def inspect_txt(path: Path, n_rows: int = 5) -> None:
    print("=" * 100)
    print(f"Metadata file: {path}")
    print(f"Size: {path.stat().st_size / (1024 ** 2):.2f} MB")

    df = pd.read_csv(path, sep="\t", nrows=n_rows)

    print("\nShape preview:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nPreview:")
    print(df.head())


def main() -> None:
    gene_tpm_file = find_file([
        "gtex_v11_gene_tpm.parquet",
        "*gene_tpm.parquet",
        "*RNASeQCv2.4.3_gene_tpm.parquet",
    ])

    sample_attr_file = find_file([
        "gtex_v11_sample_attributes.txt",
        "*SampleAttributesDS.txt",
    ])

    subject_pheno_file = find_file([
        "gtex_v11_subject_phenotypes.txt",
        "*SubjectPhenotypesDS.txt",
    ])

    files = {
        "gene_tpm": gene_tpm_file,
        "sample_attributes": sample_attr_file,
        "subject_phenotypes": subject_pheno_file,
    }

    print("[GTEx V11 file detection]")
    for name, path in files.items():
        print(f"{name}: {path if path is not None else 'MISSING'}")

    if gene_tpm_file is not None:
        inspect_parquet(gene_tpm_file)

    if sample_attr_file is not None:
        inspect_txt(sample_attr_file)

    if subject_pheno_file is not None:
        inspect_txt(subject_pheno_file)

    missing = [name for name, path in files.items() if path is None]
    if missing:
        print("\n[WARNING] Missing files:")
        for name in missing:
            print(f"  - {name}")


if __name__ == "__main__":
    main()