from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(".")
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

V01_DIR = OUTPUT_ROOT / "final_results_v0.1"
V02_DIR = OUTPUT_ROOT / "final_results_v0.2"

MATRISOMEDB_SUMMARY_DIR = OUTPUT_ROOT / "matrisomedb_validation" / "summary"
MATRISOMEDB_TABLE_DIR = MATRISOMEDB_SUMMARY_DIR
MATRISOMEDB_HTML_DIR = MATRISOMEDB_SUMMARY_DIR / "figures" / "html"
MATRISOMEDB_PNG_DIR = MATRISOMEDB_SUMMARY_DIR / "figures" / "png"


MATRISOMEDB_TABLES = {
    "Table_6_matrisomedb_gene_support_clean_summary": MATRISOMEDB_TABLE_DIR / "matrisomedb_gene_support_clean_summary.csv",
    "Table_7_matrisomedb_program_validation_summary": MATRISOMEDB_TABLE_DIR / "matrisomedb_program_validation_summary.csv",
    "Table_8_matrisomedb_top_tissue_summary": MATRISOMEDB_TABLE_DIR / "matrisomedb_top_tissue_summary.csv",
    "Supplementary_Table_matrisomedb_all_samples_raw_matrix": MATRISOMEDB_TABLE_DIR / "all_samples_mean_log_nsaf_program_matrix.csv",
    "Supplementary_Table_matrisomedb_all_samples_row_zscore": MATRISOMEDB_TABLE_DIR / "all_samples_row_zscore_program_matrix.csv",
    "Supplementary_Table_matrisomedb_normal_like_row_zscore": MATRISOMEDB_TABLE_DIR / "normal_like_row_zscore_program_matrix.csv",
}


MATRISOMEDB_FIGURES_PNG = {
    "figure_7a_matrisomedb_gene_detection_coverage": MATRISOMEDB_PNG_DIR / "matrisomedb_gene_detection_coverage_clean.png",
    "figure_7b_matrisomedb_all_samples_row_zscore_heatmap": MATRISOMEDB_PNG_DIR / "matrisomedb_all_samples_row_zscore_heatmap.png",
    "figure_7c_matrisomedb_normal_like_row_zscore_heatmap": MATRISOMEDB_PNG_DIR / "matrisomedb_normal_like_row_zscore_heatmap.png",
}


MATRISOMEDB_FIGURES_HTML = {
    "figure_7a_matrisomedb_gene_detection_coverage": MATRISOMEDB_HTML_DIR / "matrisomedb_gene_detection_coverage_clean.html",
    "figure_7b_matrisomedb_all_samples_row_zscore_heatmap": MATRISOMEDB_HTML_DIR / "matrisomedb_all_samples_row_zscore_heatmap.html",
    "figure_7c_matrisomedb_normal_like_row_zscore_heatmap": MATRISOMEDB_HTML_DIR / "matrisomedb_normal_like_row_zscore_heatmap.html",
}


MATRISOMEDB_REPORTS = {
    "matrisomedb_validation_summary": MATRISOMEDB_TABLE_DIR / "matrisomedb_validation_summary.md",
}


def reset_v02_dir() -> None:
    if V02_DIR.exists():
        shutil.rmtree(V02_DIR)

    V02_DIR.mkdir(parents=True, exist_ok=True)

    if not V01_DIR.exists():
        raise FileNotFoundError(
            f"Missing v0.1 package: {V01_DIR}. "
            "Run src/build_final_results_package.py first."
        )

    # Copy v0.1 package into v0.2 as starting point.
    for item in V01_DIR.iterdir():
        src = item
        dst = V02_DIR / item.name

        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    # Create dedicated MatrisomeDB folders.
    (V02_DIR / "tables" / "matrisomedb").mkdir(parents=True, exist_ok=True)
    (V02_DIR / "figures" / "matrisomedb" / "png").mkdir(parents=True, exist_ok=True)
    (V02_DIR / "figures" / "matrisomedb" / "html").mkdir(parents=True, exist_ok=True)
    (V02_DIR / "reports" / "matrisomedb").mkdir(parents=True, exist_ok=True)
    (V02_DIR / "metadata").mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path) -> str:
    if not src.exists():
        print(f"[MISSING] {src}")
        return "missing"

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[COPIED] {src} -> {dst}")
    return "copied"


def collect_matrisomedb_outputs() -> pd.DataFrame:
    records = []

    for name, src in MATRISOMEDB_TABLES.items():
        dst = V02_DIR / "tables" / "matrisomedb" / f"{name}{src.suffix}"
        status = copy_file(src, dst)

        records.append(
            {
                "logical_name": name,
                "type": "table",
                "source_path": str(src),
                "final_path": str(dst) if status == "copied" else "",
                "status": status,
            }
        )

    for name, src in MATRISOMEDB_FIGURES_PNG.items():
        dst = V02_DIR / "figures" / "matrisomedb" / "png" / f"{name}.png"
        status = copy_file(src, dst)

        records.append(
            {
                "logical_name": name,
                "type": "figure_png",
                "source_path": str(src),
                "final_path": str(dst) if status == "copied" else "",
                "status": status,
            }
        )

    for name, src in MATRISOMEDB_FIGURES_HTML.items():
        dst = V02_DIR / "figures" / "matrisomedb" / "html" / f"{name}.html"
        status = copy_file(src, dst)

        records.append(
            {
                "logical_name": name,
                "type": "figure_html",
                "source_path": str(src),
                "final_path": str(dst) if status == "copied" else "",
                "status": status,
            }
        )

    for name, src in MATRISOMEDB_REPORTS.items():
        dst = V02_DIR / "reports" / "matrisomedb" / f"{name}.md"
        status = copy_file(src, dst)

        records.append(
            {
                "logical_name": name,
                "type": "report",
                "source_path": str(src),
                "final_path": str(dst) if status == "copied" else "",
                "status": status,
            }
        )

    manifest = pd.DataFrame(records)
    manifest.to_csv(
        V02_DIR / "metadata" / "matrisomedb_results_manifest.csv",
        index=False,
    )

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
        "project_version": "v0.2",
        "description": "v0.1 transcriptomic ECM representation package plus MatrisomeDB protein-level validation",
        "matrisomedb_files_copied": int((manifest["status"] == "copied").sum()),
        "matrisomedb_files_missing": int((manifest["status"] == "missing").sum()),
    }

    gene_support_path = V02_DIR / "tables" / "matrisomedb" / "Table_6_matrisomedb_gene_support_clean_summary.csv"
    validation_path = V02_DIR / "tables" / "matrisomedb" / "Table_7_matrisomedb_program_validation_summary.csv"

    gene_support = safe_read_csv(gene_support_path)
    validation = safe_read_csv(validation_path)

    if gene_support is not None:
        numbers["matrisomedb_program_count"] = int(gene_support.shape[0])
        numbers["mean_protein_detection_coverage"] = float(
            gene_support["protein_detection_coverage"].mean()
        )
        numbers["min_protein_detection_coverage"] = float(
            gene_support["protein_detection_coverage"].min()
        )
        numbers["max_protein_detection_coverage"] = float(
            gene_support["protein_detection_coverage"].max()
        )

        if "support_level" in gene_support.columns:
            numbers["protein_support_level_counts"] = (
                gene_support["support_level"].value_counts().to_dict()
            )

    if validation is not None and "ecm_program" in validation.columns:
        numbers["matrisomedb_validated_programs"] = validation["ecm_program"].astype(str).tolist()

    with open(V02_DIR / "metadata" / "key_result_numbers_v0.2.json", "w", encoding="utf-8") as f:
        json.dump(numbers, f, indent=2)

    return numbers


def write_v02_report(numbers: dict) -> None:
    programs = numbers.get("matrisomedb_validated_programs", [])

    report = f"""# ECM Latent Space Project, Final Results Package v0.2

## Purpose

This package extends v0.1 by adding MatrisomeDB protein-level validation of the RNA-derived ECM programs.

## Package creation

- Created at: {numbers.get("created_at")}
- Version: {numbers.get("project_version")}
- MatrisomeDB files copied: {numbers.get("matrisomedb_files_copied")}
- MatrisomeDB files missing: {numbers.get("matrisomedb_files_missing")}

## What changed from v0.1 to v0.2

v0.1 established a transcriptomic Matrisome-derived ECM representation and benchmarking framework.

v0.2 adds protein-level support using a human MatrisomeDB export. This validation evaluates whether genes defining the RNA-derived ECM programs are detected at the ECM protein level and whether protein-level program scores show plausible tissue enrichment.

## Key MatrisomeDB validation numbers

- ECM programs evaluated: {numbers.get("matrisomedb_program_count", "NA")}
- Mean protein detection coverage: {numbers.get("mean_protein_detection_coverage", "NA")}
- Minimum protein detection coverage: {numbers.get("min_protein_detection_coverage", "NA")}
- Maximum protein detection coverage: {numbers.get("max_protein_detection_coverage", "NA")}

## Protein support level counts

{numbers.get("protein_support_level_counts", "NA")}

## Programs evaluated

{chr(10).join([f"- {program}" for program in programs]) if programs else "NA"}

## Main interpretation

Most RNA-derived ECM programs have protein-level support in the MatrisomeDB export. The strongest gene-level support is observed for renal/endothelial basement membrane ECM, vascular/stromal/interstitial ECM, immune/lymphoid remodeling ECM, epithelial/mucosal basement membrane ECM, hepatic/plasma-associated ECM, and retinal/sensory ECM.

CNS/neural ECM and reproductive-specialized ECM show weaker protein detection coverage, likely because the available MatrisomeDB export has limited normal CNS and reproductive tissue coverage.

## Important limitations

1. The MatrisomeDB export contains mixed normal and disease-associated samples.
2. Normal-like tissue coverage is sparse.
3. NSAF is semi-quantitative and may not be directly comparable across studies.
4. Tissue-level validation is partial.
5. This validation supports the biological plausibility of RNA-derived ECM programs, but does not fully validate healthy tissue ECM architecture.

## Suggested manuscript addition

Add this as Figure 7:

- Figure 7A: MatrisomeDB protein detection coverage of RNA-derived ECM programs.
- Figure 7B: All-samples row-normalized MatrisomeDB ECM program enrichment heatmap.
- Figure 7C: Normal-like row-normalized MatrisomeDB ECM program enrichment heatmap.

## Next recommended step

Update the manuscript draft to include a MatrisomeDB validation section in Methods, Results, Discussion, and Limitations.
"""

    report_path = V02_DIR / "reports" / "final_results_v0.2_summary.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"[SAVED] {report_path}")


def main() -> None:
    reset_v02_dir()

    manifest = collect_matrisomedb_outputs()
    numbers = make_key_numbers(manifest)
    write_v02_report(numbers)

    print("\n[DONE]")
    print(f"Final v0.2 package: {V02_DIR}")
    print(f"MatrisomeDB manifest: {V02_DIR / 'metadata' / 'matrisomedb_results_manifest.csv'}")
    print(f"v0.2 report: {V02_DIR / 'reports' / 'final_results_v0.2_summary.md'}")


if __name__ == "__main__":
    main()