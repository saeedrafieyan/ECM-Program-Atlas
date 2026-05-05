from pathlib import Path
import requests


HPA_BASE_URL = "https://www.proteinatlas.org/download/tsv"

HPA_FILES = {
    # Minimum required for first experiment
    "rna_tissue_consensus.tsv.zip": f"{HPA_BASE_URL}/rna_tissue_consensus.tsv.zip",

    # Useful comparison datasets
    "rna_tissue_hpa.tsv.zip": f"{HPA_BASE_URL}/rna_tissue_hpa.tsv.zip",
    "rna_tissue_gtex.tsv.zip": f"{HPA_BASE_URL}/rna_tissue_gtex.tsv.zip",
    "rna_tissue_detail_gtex.tsv.zip": f"{HPA_BASE_URL}/rna_tissue_detail_gtex.tsv.zip",

    # Brain-specific datasets, useful later
    "rna_brain_hpa.tsv.zip": f"{HPA_BASE_URL}/rna_brain_hpa.tsv.zip",
    "rna_pfc_brain_hpa.tsv.zip": f"{HPA_BASE_URL}/rna_pfc_brain_hpa.tsv.zip",

    # Single-cell dataset, useful later
    "rna_single_cell_type.tsv.zip": f"{HPA_BASE_URL}/rna_single_cell_type.tsv.zip",
}


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"[SKIP] Already exists: {output_path}")
        return

    print(f"[DOWNLOAD] {url}")

    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()

        with output_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)

    print(f"[SAVED] {output_path}")


def main() -> None:
    output_dir = Path("data/raw/hpa")

    for filename, url in HPA_FILES.items():
        output_path = output_dir / filename
        download_file(url, output_path)

    print("\nHPA download completed.")


if __name__ == "__main__":
    main()