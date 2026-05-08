from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path("results/final_results_v0.7")

SOURCE_GROUPS = {
    "frozen_reports": {
        "source": Path("results/reports/frozen"),
        "target": OUTPUT_DIR / "reports" / "frozen",
        "patterns": ["*.md"],
    },
    "frozen_tables": {
        "source": Path("results/tables/frozen"),
        "target": OUTPUT_DIR / "tables" / "frozen",
        "patterns": ["*.csv"],
    },
    "r1_nmf_stability_tables": {
        "source": Path("results/revision_nmf_stability/tables"),
        "target": OUTPUT_DIR / "tables" / "r1_nmf_stability",
        "patterns": ["*.csv"],
    },
    "r1_nmf_stability_reports": {
        "source": Path("results/revision_nmf_stability/reports"),
        "target": OUTPUT_DIR / "reports" / "r1_nmf_stability",
        "patterns": ["*.md"],
    },
    "r1_nmf_stability_figures_png": {
        "source": Path("results/revision_nmf_stability/figures/png"),
        "target": OUTPUT_DIR / "figures" / "png" / "r1_nmf_stability",
        "patterns": ["*.png"],
    },
    "r1_nmf_stability_figures_html": {
        "source": Path("results/revision_nmf_stability/figures/html"),
        "target": OUTPUT_DIR / "figures" / "html" / "r1_nmf_stability",
        "patterns": ["*.html"],
    },
    "r2_representation_controls_tables": {
        "source": Path("results/revision_classification_controls/tables"),
        "target": OUTPUT_DIR / "tables" / "r2_representation_controls",
        "patterns": ["*.csv"],
    },
    "r2_representation_controls_reports": {
        "source": Path("results/revision_classification_controls/reports"),
        "target": OUTPUT_DIR / "reports" / "r2_representation_controls",
        "patterns": ["*.md"],
    },
    "r2_representation_controls_figures_png": {
        "source": Path("results/revision_classification_controls/figures/png"),
        "target": OUTPUT_DIR / "figures" / "png" / "r2_representation_controls",
        "patterns": ["*.png"],
    },
    "r2_representation_controls_figures_html": {
        "source": Path("results/revision_classification_controls/figures/html"),
        "target": OUTPUT_DIR / "figures" / "html" / "r2_representation_controls",
        "patterns": ["*.html"],
    },
    "r3_matrisomedb_tables": {
        "source": Path("results/revision_matrisomedb_null_abundance/tables"),
        "target": OUTPUT_DIR / "tables" / "r3_matrisomedb",
        "patterns": ["*.csv"],
    },
    "r3_matrisomedb_reports": {
        "source": Path("results/revision_matrisomedb_null_abundance/reports"),
        "target": OUTPUT_DIR / "reports" / "r3_matrisomedb",
        "patterns": ["*.md"],
    },
    "r3_matrisomedb_figures_png": {
        "source": Path("results/revision_matrisomedb_null_abundance/figures/png"),
        "target": OUTPUT_DIR / "figures" / "png" / "r3_matrisomedb",
        "patterns": ["*.png"],
    },
    "r3_matrisomedb_figures_html": {
        "source": Path("results/revision_matrisomedb_null_abundance/figures/html"),
        "target": OUTPUT_DIR / "figures" / "html" / "r3_matrisomedb",
        "patterns": ["*.html"],
    },
    "r4_r6_reproducibility_tables": {
        "source": Path("results/reproducibility_package/tables"),
        "target": OUTPUT_DIR / "tables" / "r4_r6_reproducibility",
        "patterns": ["*.csv"],
    },
    "r4_r6_reproducibility_reports": {
        "source": Path("results/reproducibility_package/reports"),
        "target": OUTPUT_DIR / "reports" / "r4_r6_reproducibility",
        "patterns": ["*.md"],
    },
    "r4_r6_reproducibility_metadata": {
        "source": Path("results/reproducibility_package/metadata"),
        "target": OUTPUT_DIR / "metadata" / "r4_r6_reproducibility",
        "patterns": ["*.json"],
    },
    "r5_final_figure_tables": {
        "source": Path("results/final_figure_package/tables"),
        "target": OUTPUT_DIR / "tables" / "r5_final_figures",
        "patterns": ["*.csv"],
    },
    "r5_final_figure_reports": {
        "source": Path("results/final_figure_package/reports"),
        "target": OUTPUT_DIR / "reports" / "r5_final_figures",
        "patterns": ["*.md"],
    },
    "r5_final_figures_png": {
        "source": Path("results/final_figure_package/figures/png"),
        "target": OUTPUT_DIR / "figures" / "png" / "r5_final_figures",
        "patterns": ["*.png"],
    },
    "r5_final_figures_html": {
        "source": Path("results/final_figure_package/figures/html"),
        "target": OUTPUT_DIR / "figures" / "html" / "r5_final_figures",
        "patterns": ["*.html"],
    },
    "r7_spatial_tables": {
        "source": Path("results/revision_spatial_validation/tables"),
        "target": OUTPUT_DIR / "tables" / "r7_spatial_validation",
        "patterns": ["*.csv", "*.json"],
    },
    "r7_spatial_reports": {
        "source": Path("results/revision_spatial_validation/reports"),
        "target": OUTPUT_DIR / "reports" / "r7_spatial_validation",
        "patterns": ["*.md"],
    },
    "r7_spatial_figures_png": {
        "source": Path("results/revision_spatial_validation/figures/png"),
        "target": OUTPUT_DIR / "figures" / "png" / "r7_spatial_validation",
        "patterns": ["*.png"],
    },
    "r7_spatial_figures_html": {
        "source": Path("results/revision_spatial_validation/figures/html"),
        "target": OUTPUT_DIR / "figures" / "html" / "r7_spatial_validation",
        "patterns": ["*.html"],
    },
    "rank_based_tables": {
        "source": Path("results/rank_based_scoring/tables"),
        "target": OUTPUT_DIR / "tables" / "rank_based_scoring",
        "patterns": ["*.csv"],
    },
    "rank_based_reports": {
        "source": Path("results/rank_based_scoring/reports"),
        "target": OUTPUT_DIR / "reports" / "rank_based_scoring",
        "patterns": ["*.md"],
    },
    "rank_based_figures_png": {
        "source": Path("results/rank_based_scoring/figures/png"),
        "target": OUTPUT_DIR / "figures" / "png" / "rank_based_scoring",
        "patterns": ["*.png"],
    },
    "rank_based_figures_html": {
        "source": Path("results/rank_based_scoring/figures/html"),
        "target": OUTPUT_DIR / "figures" / "html" / "rank_based_scoring",
        "patterns": ["*.html"],
    },
}


CRITICAL_FILES = [
    "reports/r1_nmf_stability/nmf_rank_stability_summary.md",
    "reports/r2_representation_controls/r2_representation_control_summary.md",
    "reports/r3_matrisomedb/r3_matrisomedb_null_abundance_summary.md",
    "reports/r4_r6_reproducibility/reproducibility_package_summary.md",
    "reports/r5_final_figures/figure_qc_report.md",
    "reports/r7_spatial_validation/r7_spatial_validation_summary.md",
    "tables/r4_r6_reproducibility/Supplementary_Table_S1_curated_ecm_program_gene_sets.csv",
    "tables/r4_r6_reproducibility/Supplementary_Table_S3_pipeline_execution_order.csv",
    "tables/r4_r6_reproducibility/Supplementary_Table_S4_data_manifest.csv",
    "tables/r4_r6_reproducibility/Supplementary_Table_S6_exact_method_parameters.csv",
    "tables/r4_r6_reproducibility/Supplementary_Table_S7_model_parameters.csv",
    "tables/r4_r6_reproducibility/Supplementary_Table_S8_reviewer_criticism_response_map.csv",
    "tables/r5_final_figures/final_figure_manifest.csv",
]


def reset_output_dir() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    for folder in [
        OUTPUT_DIR / "tables",
        OUTPUT_DIR / "figures" / "png",
        OUTPUT_DIR / "figures" / "html",
        OUTPUT_DIR / "reports",
        OUTPUT_DIR / "metadata",
    ]:
        folder.mkdir(parents=True, exist_ok=True)


def iter_files(source: Path, patterns: list[str]) -> list[Path]:
    files: list[Path] = []

    if not source.exists():
        return files

    for pattern in patterns:
        files.extend(sorted(source.glob(pattern)))

    return sorted(set(files))


def copy_group(group_name: str, spec: dict) -> list[dict]:
    source = spec["source"]
    target = spec["target"]
    patterns = spec["patterns"]

    records: list[dict] = []

    if not source.exists():
        records.append(
            {
                "group": group_name,
                "source": str(source),
                "target": str(target),
                "file": "",
                "status": "missing_source_folder",
                "size_mb": None,
            }
        )
        print(f"[MISSING FOLDER] {source}")
        return records

    files = iter_files(source, patterns)

    if not files:
        records.append(
            {
                "group": group_name,
                "source": str(source),
                "target": str(target),
                "file": "",
                "status": "no_matching_files",
                "size_mb": None,
            }
        )
        print(f"[NO FILES] {source} patterns={patterns}")
        return records

    target.mkdir(parents=True, exist_ok=True)

    for src in files:
        dst = target / src.name
        shutil.copy2(src, dst)

        size_mb = dst.stat().st_size / (1024 ** 2)

        records.append(
            {
                "group": group_name,
                "source": str(src),
                "target": str(dst),
                "file": src.name,
                "status": "copied",
                "size_mb": size_mb,
            }
        )

        print(f"[COPIED] {src} -> {dst}")

    return records


def build_manifest() -> pd.DataFrame:
    all_records = []

    for group_name, spec in SOURCE_GROUPS.items():
        all_records.extend(copy_group(group_name, spec))

    manifest = pd.DataFrame(all_records)
    manifest_path = OUTPUT_DIR / "metadata" / "final_results_v0.7_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"[SAVED] {manifest_path}")

    return manifest


def check_critical_files() -> pd.DataFrame:
    records = []

    for relative in CRITICAL_FILES:
        path = OUTPUT_DIR / relative
        records.append(
            {
                "relative_path": relative,
                "exists": path.exists(),
                "size_mb": path.stat().st_size / (1024 ** 2) if path.exists() else None,
            }
        )

    df = pd.DataFrame(records)
    path = OUTPUT_DIR / "metadata" / "critical_file_check.csv"
    df.to_csv(path, index=False)
    print(f"[SAVED] {path}")

    return df


def count_files_by_suffix() -> dict:
    counts = {}
    for path in OUTPUT_DIR.rglob("*"):
        if path.is_file():
            suffix = path.suffix.lower() or "no_suffix"
            counts[suffix] = counts.get(suffix, 0) + 1
    return counts


def write_summary_report(manifest: pd.DataFrame, critical: pd.DataFrame) -> None:
    copied = int((manifest["status"] == "copied").sum())
    missing_folders = int((manifest["status"] == "missing_source_folder").sum())
    no_matching = int((manifest["status"] == "no_matching_files").sum())
    critical_missing = critical[~critical["exists"]]

    suffix_counts = count_files_by_suffix()

    report = f"""# Final Results Package v0.7

## Purpose

This package freezes the current complete ECM Program Atlas revision state.

It includes:

- R1: NMF rank and stability analysis
- R2: representation-control classification benchmark
- R3: MatrisomeDB detection and abundance null validation
- R4/R6: reproducibility package, supplementary tables, exact methods and parameters
- R5: final integrated figure package
- R7: focused spatial validation using public Visium datasets
- rank-based scoring robustness outputs
- frozen prior summary reports and tables

## Package creation

- Created at: {datetime.now().isoformat(timespec="seconds")}
- Files copied: {copied}
- Missing source folders: {missing_folders}
- Source folders with no matching files: {no_matching}

## File type counts

{json.dumps(suffix_counts, indent=2)}

## Critical file check

- Critical files expected: {critical.shape[0]}
- Critical files missing: {critical_missing.shape[0]}

"""

    if not critical_missing.empty:
        report += "\n## Missing critical files\n"
        for row in critical_missing.itertuples():
            report += f"- {row.relative_path}\n"

    report += """
## Interpretation

v0.7 is the final analysis freeze before manuscript writing. It consolidates the core ECM representation framework and all reviewer-driven revision tracks.

## Recommended next step

Stop expanding the analysis and write the manuscript using this v0.7 package as the definitive source of tables, figures, and methods documentation.
"""

    path = OUTPUT_DIR / "reports" / "final_results_v0.7_summary.md"
    path.write_text(report, encoding="utf-8")
    print(f"[SAVED] {path}")


def write_readme() -> None:
    readme = """# Final Results v0.7

This folder contains the final frozen analysis package for the ECM Program Atlas project.

## Main folders

- `tables/`: final tables and supplementary tables
- `figures/png/`: static manuscript-ready figures
- `figures/html/`: interactive figures
- `reports/`: summary reports for each revision track
- `metadata/`: manifests and critical file checks

## Important note

Large raw and processed datasets are not included. This folder contains small reports, figures, and tables only.
"""

    path = OUTPUT_DIR / "README.md"
    path.write_text(readme, encoding="utf-8")
    print(f"[SAVED] {path}")


def main() -> None:
    reset_output_dir()
    manifest = build_manifest()
    critical = check_critical_files()
    write_summary_report(manifest, critical)
    write_readme()

    print("\n[DONE]")
    print(f"Final package: {OUTPUT_DIR}")
    print(f"Summary: {OUTPUT_DIR / 'reports' / 'final_results_v0.7_summary.md'}")
    print(f"Manifest: {OUTPUT_DIR / 'metadata' / 'final_results_v0.7_manifest.csv'}")


if __name__ == "__main__":
    main()