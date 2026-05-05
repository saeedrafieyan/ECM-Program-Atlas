from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


OUTPUT_ROOT = Path("outputs")
BASE_PACKAGE_DIR = OUTPUT_ROOT / "final_results_v0.4"
V05_DIR = OUTPUT_ROOT / "final_results_v0.5"

ROUTE_B_DIR = OUTPUT_ROOT / "tabula_sapiens_pseudobulk" / "final_route_b"

ROUTE_B_TABLES = {
    "Table_21_route_b_global_summary": ROUTE_B_DIR / "tables" / "route_b_global_summary.csv",
    "Table_22_route_b_program_gene_availability": ROUTE_B_DIR / "tables" / "route_b_program_gene_availability.csv",
    "Table_23_route_b_program_source_summary": ROUTE_B_DIR / "tables" / "route_b_program_source_summary.csv",
    "Table_24_route_b_consensus_celltype_sources": ROUTE_B_DIR / "tables" / "route_b_consensus_celltype_sources.csv",
    "Supplementary_Table_route_b_top_robust_celltypes": ROUTE_B_DIR / "tables" / "route_b_top_robust_celltypes_per_program.csv",
    "Supplementary_Table_route_b_top_robust_organ_celltypes": ROUTE_B_DIR / "tables" / "route_b_top_robust_organ_celltypes_per_program.csv",
}

ROUTE_B_REPORTS = {
    "route_b_final_summary": ROUTE_B_DIR / "reports" / "route_b_final_summary.md",
}

ROUTE_B_FIGURES_PNG = {
    "figure_12a_route_b_compartment_program_heatmap_10X": ROUTE_B_DIR / "figures" / "png" / "route_b_compartment_program_heatmap_10X.png",
    "figure_12b_route_b_compartment_program_heatmap_smartseq2": ROUTE_B_DIR / "figures" / "png" / "route_b_compartment_program_heatmap_smartseq2.png",
}

ROUTE_B_FIGURES_HTML = {
    "figure_12a_route_b_compartment_program_heatmap_10X": ROUTE_B_DIR / "figures" / "html" / "route_b_compartment_program_heatmap_10X.html",
    "figure_12b_route_b_compartment_program_heatmap_smartseq2": ROUTE_B_DIR / "figures" / "html" / "route_b_compartment_program_heatmap_smartseq2.html",
}


def reset_v05_dir() -> None:
    if V05_DIR.exists():
        shutil.rmtree(V05_DIR)

    if not BASE_PACKAGE_DIR.exists():
        raise FileNotFoundError(
            f"Missing base package: {BASE_PACKAGE_DIR}. "
            "Run src/build_final_results_package_v0_4.py first."
        )

    shutil.copytree(BASE_PACKAGE_DIR, V05_DIR)

    for folder in [
        V05_DIR / "tables" / "tabula_sapiens",
        V05_DIR / "figures" / "tabula_sapiens" / "png",
        V05_DIR / "figures" / "tabula_sapiens" / "html",
        V05_DIR / "reports" / "tabula_sapiens",
        V05_DIR / "metadata",
    ]:
        folder.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path) -> str:
    if not src.exists():
        print(f"[MISSING] {src}")
        return "missing"

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[COPIED] {src} -> {dst}")
    return "copied"


def collect_group(files: dict[str, Path], output_subdir: Path, suffix: str, file_type: str) -> list[dict]:
    records = []

    for name, src in files.items():
        dst = output_subdir / f"{name}{suffix}"
        status = copy_file(src, dst)

        records.append(
            {
                "logical_name": name,
                "type": file_type,
                "source_path": str(src),
                "final_path": str(dst) if status == "copied" else "",
                "status": status,
            }
        )

    return records


def collect_v05_outputs() -> pd.DataFrame:
    records = []

    records.extend(
        collect_group(
            ROUTE_B_TABLES,
            V05_DIR / "tables" / "tabula_sapiens",
            ".csv",
            "route_b_table",
        )
    )

    records.extend(
        collect_group(
            ROUTE_B_REPORTS,
            V05_DIR / "reports" / "tabula_sapiens",
            ".md",
            "route_b_report",
        )
    )

    records.extend(
        collect_group(
            ROUTE_B_FIGURES_PNG,
            V05_DIR / "figures" / "tabula_sapiens" / "png",
            ".png",
            "route_b_figure_png",
        )
    )

    records.extend(
        collect_group(
            ROUTE_B_FIGURES_HTML,
            V05_DIR / "figures" / "tabula_sapiens" / "html",
            ".html",
            "route_b_figure_html",
        )
    )

    manifest = pd.DataFrame(records)
    manifest.to_csv(V05_DIR / "metadata" / "route_b_v0.5_manifest.csv", index=False)

    return manifest


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[WARNING] Could not read {path}: {exc}")
        return None


def make_key_numbers(manifest: pd.DataFrame) -> dict:
    numbers = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_version": "v0.5",
        "description": "v0.4 plus Tabula Sapiens cell-type source validation of curated ECM programs",
        "route_b_files_copied": int((manifest["status"] == "copied").sum()),
        "route_b_files_missing": int((manifest["status"] == "missing").sum()),
    }

    global_summary = safe_read_csv(V05_DIR / "tables" / "tabula_sapiens" / "Table_21_route_b_global_summary.csv")
    availability = safe_read_csv(V05_DIR / "tables" / "tabula_sapiens" / "Table_22_route_b_program_gene_availability.csv")
    source_summary = safe_read_csv(V05_DIR / "tables" / "tabula_sapiens" / "Table_23_route_b_program_source_summary.csv")

    if global_summary is not None and not global_summary.empty:
        row = global_summary.iloc[0]
        numbers["tabula_sapiens_source_cells"] = int(row.get("n_cells_used", 0))
        numbers["tabula_sapiens_matched_matrisome_genes"] = int(row.get("n_matched_matrisome_genes", 0))
        numbers["tabula_sapiens_pseudobulk_groups"] = int(row.get("n_pseudobulk_groups", 0))
        numbers["tabula_sapiens_donors"] = int(row.get("n_donors", 0))
        numbers["tabula_sapiens_organs"] = int(row.get("n_organs", 0))
        numbers["tabula_sapiens_cell_types"] = int(row.get("n_cell_types", 0))
        numbers["tabula_sapiens_compartments"] = int(row.get("n_compartments", 0))
        numbers["tabula_sapiens_methods"] = int(row.get("n_methods", 0))

    if availability is not None:
        numbers["tabula_sapiens_min_gene_availability"] = float(availability["availability_fraction"].min())
        numbers["tabula_sapiens_mean_gene_availability"] = float(availability["availability_fraction"].mean())

    if source_summary is not None:
        numbers["route_b_programs"] = source_summary["ecm_program"].astype(str).tolist()

    with open(V05_DIR / "metadata" / "key_result_numbers_v0.5.json", "w", encoding="utf-8") as f:
        json.dump(numbers, f, indent=2)

    return numbers


def write_v05_report(numbers: dict) -> None:
    programs = numbers.get("route_b_programs", [])

    report = f"""# ECM Latent Space Project, Final Results Package v0.5

## Purpose

This package extends v0.4 by adding Tabula Sapiens cell-type source validation of the curated ECM programs.

## Package creation

- Created at: {numbers.get("created_at")}
- Version: {numbers.get("project_version")}
- Route B files copied: {numbers.get("route_b_files_copied")}
- Route B files missing: {numbers.get("route_b_files_missing")}

## What changed from v0.4 to v0.5

v0.4 established a Matrisome-derived ECM representation framework with transcriptomic reproducibility, MatrisomeDB protein-level validation, GTEx V11 sample-level validation, donor-stratified classification, and deep/tabular model benchmarking.

v0.5 adds Tabula Sapiens donor × organ × cell-type pseudobulk analysis to identify likely cellular sources of the curated ECM programs.

## Key Tabula Sapiens numbers

- Source cells: {numbers.get("tabula_sapiens_source_cells", "NA")}
- Matched Matrisome genes: {numbers.get("tabula_sapiens_matched_matrisome_genes", "NA")}
- Pseudobulk groups: {numbers.get("tabula_sapiens_pseudobulk_groups", "NA")}
- Donors: {numbers.get("tabula_sapiens_donors", "NA")}
- Organs: {numbers.get("tabula_sapiens_organs", "NA")}
- Cell types: {numbers.get("tabula_sapiens_cell_types", "NA")}
- Compartments: {numbers.get("tabula_sapiens_compartments", "NA")}
- Methods: {numbers.get("tabula_sapiens_methods", "NA")}

## Program gene availability

- Minimum availability fraction: {numbers.get("tabula_sapiens_min_gene_availability", "NA")}
- Mean availability fraction: {numbers.get("tabula_sapiens_mean_gene_availability", "NA")}

## ECM programs evaluated

{chr(10).join([f"- {program}" for program in programs]) if programs else "NA"}

## Main interpretation

Route B supports the biological plausibility of the ECM programs by showing that many programs are enriched in expected ECM-producing cell types and compartments, including stromal, fibroblastic, mesenchymal, endothelial, pericyte, hepatocyte, and tissue-supporting cell populations.

This directly addresses a major limitation of tissue-level transcriptomics: bulk ECM programs are partly driven by cell-type composition. The Tabula Sapiens analysis makes this cell-type contribution explicit.

## Important limitations

1. Tabula Sapiens is transcriptomic, not protein-level ECM deposition.
2. 10X and Smart-seq2 differ and should be interpreted separately.
3. Some rare cell types remain supported by limited donor or pseudobulk counts.
4. Spatial localization is still missing.
5. This analysis improves biological interpretability but does not yet close the scaffold-design actionability gap.

## Suggested manuscript addition

Add this as Figure 12:

- Figure 12A: Tabula Sapiens compartment-level ECM program enrichment, 10X.
- Figure 12B: Tabula Sapiens compartment-level ECM program enrichment, Smart-seq2.
- Optional supplementary panels: top robust cell-type sources for each ECM program.

## Next recommended step

The project is now strong enough to begin manuscript consolidation. Further expansion should be strategic, not open-ended. The next possible biological extension would be spatial validation using MatriSpace-like spatial transcriptomics analysis.
"""

    report_path = V05_DIR / "reports" / "final_results_v0.5_summary.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"[SAVED] {report_path}")


def main() -> None:
    reset_v05_dir()
    manifest = collect_v05_outputs()
    numbers = make_key_numbers(manifest)
    write_v05_report(numbers)

    print("\n[DONE]")
    print(f"Final v0.5 package: {V05_DIR}")
    print(f"Manifest: {V05_DIR / 'metadata' / 'route_b_v0.5_manifest.csv'}")
    print(f"Report: {V05_DIR / 'reports' / 'final_results_v0.5_summary.md'}")


if __name__ == "__main__":
    main()