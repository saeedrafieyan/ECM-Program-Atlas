from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(".")
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

BASE_PACKAGE_DIR = OUTPUT_ROOT / "final_results_v0.2"
V04_DIR = OUTPUT_ROOT / "final_results_v0.4"

GTEX_VALIDATION_DIR = OUTPUT_ROOT / "gtex_v11_sample_level_validation"
GTEX_CLASSIFICATION_DIR = GTEX_VALIDATION_DIR / "classification"
GTEX_MODEL_BENCHMARK_DIR = GTEX_VALIDATION_DIR / "model_benchmark"
GTEX_DEEP_BENCHMARK_DIR = GTEX_VALIDATION_DIR / "deep_tabular_benchmark"

PROCESSED_GTEX_DIR = PROJECT_ROOT / "data" / "processed" / "gtex_v11_sample_level"


GTEX_SAMPLE_LEVEL_TABLES = {
    "Table_9_gtex_v11_global_summary": PROCESSED_GTEX_DIR / "gtex_v11_global_summary.csv",
    "Table_10_gtex_v11_program_gene_availability": PROCESSED_GTEX_DIR / "gtex_v11_program_gene_availability.csv",
    "Table_11_gtex_v11_tissue_program_summary": PROCESSED_GTEX_DIR / "gtex_v11_tissue_program_summary.csv",
    "Table_12_gtex_v11_tissue_detail_program_summary": PROCESSED_GTEX_DIR / "gtex_v11_tissue_detail_program_summary.csv",
    "Table_13_gtex_v11_top_tissues_per_program": GTEX_VALIDATION_DIR / "tables" / "gtex_v11_top_tissues_per_program.csv",
    "Table_14_gtex_v11_tissue_program_variability_summary": GTEX_VALIDATION_DIR / "tables" / "gtex_v11_tissue_program_variability_summary.csv",
    "Supplementary_Table_gtex_v11_tissue_program_zscore_matrix": GTEX_VALIDATION_DIR / "tables" / "gtex_v11_tissue_program_zscore_matrix.csv",
    "Supplementary_Table_gtex_v11_tissue_program_row_zscore_matrix": GTEX_VALIDATION_DIR / "tables" / "gtex_v11_tissue_program_row_zscore_matrix.csv",
}

GTEX_SAMPLE_LEVEL_REPORTS = {
    "gtex_v11_sample_level_validation_summary": GTEX_VALIDATION_DIR / "gtex_v11_sample_level_validation_summary.md",
}

GTEX_SAMPLE_LEVEL_FIGURES_PNG = {
    "figure_8a_gtex_v11_tissue_program_mean_zscore_heatmap": GTEX_VALIDATION_DIR / "figures" / "png" / "gtex_v11_tissue_program_mean_zscore_heatmap.png",
    "figure_8b_gtex_v11_tissue_program_row_zscore_heatmap": GTEX_VALIDATION_DIR / "figures" / "png" / "gtex_v11_tissue_program_row_zscore_heatmap.png",
    "figure_8c_gtex_v11_sample_level_ecm_program_pca": GTEX_VALIDATION_DIR / "figures" / "png" / "gtex_v11_sample_level_ecm_program_pca.png",
    "figure_8d_gtex_v11_program_gene_availability": GTEX_VALIDATION_DIR / "figures" / "png" / "gtex_v11_program_gene_availability.png",
}

GTEX_SAMPLE_LEVEL_FIGURES_HTML = {
    "figure_8a_gtex_v11_tissue_program_mean_zscore_heatmap": GTEX_VALIDATION_DIR / "figures" / "html" / "gtex_v11_tissue_program_mean_zscore_heatmap.html",
    "figure_8b_gtex_v11_tissue_program_row_zscore_heatmap": GTEX_VALIDATION_DIR / "figures" / "html" / "gtex_v11_tissue_program_row_zscore_heatmap.html",
    "figure_8c_gtex_v11_sample_level_ecm_program_pca": GTEX_VALIDATION_DIR / "figures" / "html" / "gtex_v11_sample_level_ecm_program_pca.html",
    "figure_8d_gtex_v11_program_gene_availability": GTEX_VALIDATION_DIR / "figures" / "html" / "gtex_v11_program_gene_availability.html",
}

GTEX_CLASSIFICATION_TABLES = {
    "Table_15_gtex_v11_donor_stratified_classification_summary": GTEX_CLASSIFICATION_DIR / "tables" / "gtex_v11_donor_stratified_classification_summary.csv",
    "Table_16_gtex_v11_donor_stratified_fold_metrics": GTEX_CLASSIFICATION_DIR / "tables" / "gtex_v11_donor_stratified_fold_metrics.csv",
}

GTEX_CLASSIFICATION_REPORTS = {
    "gtex_v11_donor_stratified_classification_summary": GTEX_CLASSIFICATION_DIR / "gtex_v11_donor_stratified_classification_summary.md",
}

GTEX_CLASSIFICATION_FIGURES_PNG = {
    "figure_9a_gtex_v11_donor_stratified_classification_performance": GTEX_CLASSIFICATION_DIR / "figures" / "png" / "gtex_v11_donor_stratified_classification_performance.png",
    "figure_9b_gtex_v11_confusion_matrix_tissue_system": GTEX_CLASSIFICATION_DIR / "figures" / "png" / "gtex_v11_confusion_matrix_tissue_system.png",
    "figure_9c_gtex_v11_confusion_matrix_tissue": GTEX_CLASSIFICATION_DIR / "figures" / "png" / "gtex_v11_confusion_matrix_tissue.png",
}

GTEX_CLASSIFICATION_FIGURES_HTML = {
    "figure_9a_gtex_v11_donor_stratified_classification_performance": GTEX_CLASSIFICATION_DIR / "figures" / "html" / "gtex_v11_donor_stratified_classification_performance.html",
    "figure_9b_gtex_v11_confusion_matrix_tissue_system": GTEX_CLASSIFICATION_DIR / "figures" / "html" / "gtex_v11_confusion_matrix_tissue_system.html",
    "figure_9c_gtex_v11_confusion_matrix_tissue": GTEX_CLASSIFICATION_DIR / "figures" / "html" / "gtex_v11_confusion_matrix_tissue.html",
}

GTEX_MODEL_BENCHMARK_TABLES = {
    "Table_17_gtex_v11_ecm_classifier_benchmark_summary": GTEX_MODEL_BENCHMARK_DIR / "tables" / "gtex_v11_ecm_classifier_benchmark_summary.csv",
    "Table_18_gtex_v11_ecm_classifier_benchmark_fold_metrics": GTEX_MODEL_BENCHMARK_DIR / "tables" / "gtex_v11_ecm_classifier_benchmark_fold_metrics.csv",
}

GTEX_MODEL_BENCHMARK_REPORTS = {
    "gtex_v11_ecm_classifier_benchmark_summary": GTEX_MODEL_BENCHMARK_DIR / "gtex_v11_ecm_classifier_benchmark_summary.md",
}

GTEX_MODEL_BENCHMARK_FIGURES_PNG = {
    "figure_10a_gtex_v11_ecm_classifier_benchmark_balanced_accuracy": GTEX_MODEL_BENCHMARK_DIR / "figures" / "png" / "gtex_v11_ecm_classifier_benchmark_balanced_accuracy.png",
    "figure_10b_gtex_v11_classifier_metric_heatmap_tissue_system": GTEX_MODEL_BENCHMARK_DIR / "figures" / "png" / "gtex_v11_classifier_metric_heatmap_tissue_system.png",
    "figure_10c_gtex_v11_classifier_metric_heatmap_tissue": GTEX_MODEL_BENCHMARK_DIR / "figures" / "png" / "gtex_v11_classifier_metric_heatmap_tissue.png",
}

GTEX_MODEL_BENCHMARK_FIGURES_HTML = {
    "figure_10a_gtex_v11_ecm_classifier_benchmark_balanced_accuracy": GTEX_MODEL_BENCHMARK_DIR / "figures" / "html" / "gtex_v11_ecm_classifier_benchmark_balanced_accuracy.html",
    "figure_10b_gtex_v11_classifier_metric_heatmap_tissue_system": GTEX_MODEL_BENCHMARK_DIR / "figures" / "html" / "gtex_v11_classifier_metric_heatmap_tissue_system.html",
    "figure_10c_gtex_v11_classifier_metric_heatmap_tissue": GTEX_MODEL_BENCHMARK_DIR / "figures" / "html" / "gtex_v11_classifier_metric_heatmap_tissue.html",
}

GTEX_DEEP_BENCHMARK_TABLES = {
    "Table_19_gtex_v11_deep_tabular_benchmark_summary": GTEX_DEEP_BENCHMARK_DIR / "tables" / "gtex_v11_deep_tabular_benchmark_summary.csv",
    "Table_20_gtex_v11_deep_tabular_benchmark_fold_metrics": GTEX_DEEP_BENCHMARK_DIR / "tables" / "gtex_v11_deep_tabular_benchmark_fold_metrics.csv",
}

GTEX_DEEP_BENCHMARK_REPORTS = {
    "gtex_v11_deep_tabular_benchmark_summary": GTEX_DEEP_BENCHMARK_DIR / "gtex_v11_deep_tabular_benchmark_summary.md",
}

GTEX_DEEP_BENCHMARK_FIGURES_PNG = {
    "figure_11_gtex_v11_deep_tabular_benchmark_balanced_accuracy": GTEX_DEEP_BENCHMARK_DIR / "figures" / "png" / "gtex_v11_deep_tabular_benchmark_balanced_accuracy.png",
}

GTEX_DEEP_BENCHMARK_FIGURES_HTML = {
    "figure_11_gtex_v11_deep_tabular_benchmark_balanced_accuracy": GTEX_DEEP_BENCHMARK_DIR / "figures" / "html" / "gtex_v11_deep_tabular_benchmark_balanced_accuracy.html",
}


def reset_v04_dir() -> None:
    if V04_DIR.exists():
        shutil.rmtree(V04_DIR)

    if not BASE_PACKAGE_DIR.exists():
        raise FileNotFoundError(
            f"Missing base package: {BASE_PACKAGE_DIR}. "
            "Run src/build_final_results_package_v0_2.py first."
        )

    shutil.copytree(BASE_PACKAGE_DIR, V04_DIR)

    for folder in [
        V04_DIR / "tables" / "gtex_v11",
        V04_DIR / "figures" / "gtex_v11" / "png",
        V04_DIR / "figures" / "gtex_v11" / "html",
        V04_DIR / "reports" / "gtex_v11",
        V04_DIR / "metadata",
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


def collect_group(files: dict[str, Path], output_subdir: Path, extension: str | None, file_type: str) -> list[dict]:
    records = []

    for name, src in files.items():
        suffix = extension if extension is not None else src.suffix
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


def collect_v04_outputs() -> pd.DataFrame:
    records = []

    table_dir = V04_DIR / "tables" / "gtex_v11"
    png_dir = V04_DIR / "figures" / "gtex_v11" / "png"
    html_dir = V04_DIR / "figures" / "gtex_v11" / "html"
    report_dir = V04_DIR / "reports" / "gtex_v11"

    records.extend(collect_group(GTEX_SAMPLE_LEVEL_TABLES, table_dir, ".csv", "gtex_sample_level_table"))
    records.extend(collect_group(GTEX_SAMPLE_LEVEL_REPORTS, report_dir, ".md", "gtex_sample_level_report"))
    records.extend(collect_group(GTEX_SAMPLE_LEVEL_FIGURES_PNG, png_dir, ".png", "gtex_sample_level_figure_png"))
    records.extend(collect_group(GTEX_SAMPLE_LEVEL_FIGURES_HTML, html_dir, ".html", "gtex_sample_level_figure_html"))

    records.extend(collect_group(GTEX_CLASSIFICATION_TABLES, table_dir, ".csv", "gtex_classification_table"))
    records.extend(collect_group(GTEX_CLASSIFICATION_REPORTS, report_dir, ".md", "gtex_classification_report"))
    records.extend(collect_group(GTEX_CLASSIFICATION_FIGURES_PNG, png_dir, ".png", "gtex_classification_figure_png"))
    records.extend(collect_group(GTEX_CLASSIFICATION_FIGURES_HTML, html_dir, ".html", "gtex_classification_figure_html"))

    records.extend(collect_group(GTEX_MODEL_BENCHMARK_TABLES, table_dir, ".csv", "gtex_ml_benchmark_table"))
    records.extend(collect_group(GTEX_MODEL_BENCHMARK_REPORTS, report_dir, ".md", "gtex_ml_benchmark_report"))
    records.extend(collect_group(GTEX_MODEL_BENCHMARK_FIGURES_PNG, png_dir, ".png", "gtex_ml_benchmark_figure_png"))
    records.extend(collect_group(GTEX_MODEL_BENCHMARK_FIGURES_HTML, html_dir, ".html", "gtex_ml_benchmark_figure_html"))

    records.extend(collect_group(GTEX_DEEP_BENCHMARK_TABLES, table_dir, ".csv", "gtex_deep_benchmark_table"))
    records.extend(collect_group(GTEX_DEEP_BENCHMARK_REPORTS, report_dir, ".md", "gtex_deep_benchmark_report"))
    records.extend(collect_group(GTEX_DEEP_BENCHMARK_FIGURES_PNG, png_dir, ".png", "gtex_deep_benchmark_figure_png"))
    records.extend(collect_group(GTEX_DEEP_BENCHMARK_FIGURES_HTML, html_dir, ".html", "gtex_deep_benchmark_figure_html"))

    manifest = pd.DataFrame(records)
    manifest.to_csv(V04_DIR / "metadata" / "gtex_v11_v0.4_manifest.csv", index=False)

    return manifest


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[WARNING] Could not read {path}: {exc}")
        return None


def parse_best_model(summary_df: pd.DataFrame, label_col: str, metric_col: str = "balanced_accuracy_mean") -> dict:
    subset = summary_df[summary_df["label_col"].astype(str).eq(label_col)].copy()

    if subset.empty or metric_col not in subset.columns:
        return {}

    subset = subset.sort_values(metric_col, ascending=False)
    row = subset.iloc[0]

    return {
        "task": label_col,
        "best_model": row["model"],
        "balanced_accuracy_mean": float(row.get("balanced_accuracy_mean", float("nan"))),
        "balanced_accuracy_std": float(row.get("balanced_accuracy_std", float("nan"))),
        "macro_f1_mean": float(row.get("macro_f1_mean", float("nan"))),
        "macro_f1_std": float(row.get("macro_f1_std", float("nan"))),
        "accuracy_mean": float(row.get("accuracy_mean", float("nan"))),
        "accuracy_std": float(row.get("accuracy_std", float("nan"))),
    }


def make_key_numbers(manifest: pd.DataFrame) -> dict:
    numbers = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_version": "v0.4",
        "description": "v0.2 plus GTEx V11 sample-level validation, donor-stratified ML classification, and deep/tabular benchmark",
        "gtex_v11_files_copied": int((manifest["status"] == "copied").sum()),
        "gtex_v11_files_missing": int((manifest["status"] == "missing").sum()),
    }

    global_summary = safe_read_csv(V04_DIR / "tables" / "gtex_v11" / "Table_9_gtex_v11_global_summary.csv")
    deep_summary = safe_read_csv(V04_DIR / "tables" / "gtex_v11" / "Table_19_gtex_v11_deep_tabular_benchmark_summary.csv")
    classical_summary = safe_read_csv(V04_DIR / "tables" / "gtex_v11" / "Table_17_gtex_v11_ecm_classifier_benchmark_summary.csv")

    if global_summary is not None and not global_summary.empty:
        row = global_summary.iloc[0]
        numbers["gtex_v11_expression_samples"] = int(row.get("n_expression_samples", 0))
        numbers["gtex_v11_matched_matrisome_genes"] = int(row.get("n_matched_matrisome_genes", 0))
        numbers["gtex_v11_programs_scored"] = int(row.get("n_programs", 0))
        numbers["gtex_v11_subjects"] = int(row.get("n_subjects", 0))
        numbers["gtex_v11_tissues"] = int(row.get("n_tissues", 0))
        numbers["gtex_v11_tissue_details"] = int(row.get("n_tissue_details", 0))

    if deep_summary is not None:
        numbers["deep_tabular_best_tissue"] = parse_best_model(deep_summary, "tissue")
        numbers["deep_tabular_best_tissue_system"] = parse_best_model(deep_summary, "tissue_system")

    if classical_summary is not None:
        numbers["classical_best_tissue"] = parse_best_model(classical_summary, "tissue")
        numbers["classical_best_tissue_system"] = parse_best_model(classical_summary, "tissue_system")

    with open(V04_DIR / "metadata" / "key_result_numbers_v0.4.json", "w", encoding="utf-8") as f:
        json.dump(numbers, f, indent=2)

    return numbers


def write_v04_report(numbers: dict) -> None:
    tissue_best = numbers.get("deep_tabular_best_tissue", {})
    system_best = numbers.get("deep_tabular_best_tissue_system", {})

    report = f"""# ECM Latent Space Project, Final Results Package v0.4

## Purpose

This package extends v0.2 by adding GTEx V11 sample-level validation, donor-stratified tissue classification, and deep/tabular model benchmarking.

## Package creation

- Created at: {numbers.get("created_at")}
- Version: {numbers.get("project_version")}
- GTEx V11 files copied: {numbers.get("gtex_v11_files_copied")}
- GTEx V11 files missing: {numbers.get("gtex_v11_files_missing")}

## What changed from v0.2 to v0.4

v0.2 established a Matrisome-derived ECM representation framework with MatrisomeDB protein-level validation.

v0.4 adds GTEx V11 sample-level validation across individual donor samples and benchmarks classical, ensemble, deep learning, and tabular models for tissue classification using only nine curated ECM program scores.

## Key GTEx V11 sample-level numbers

- Expression samples: {numbers.get("gtex_v11_expression_samples", "NA")}
- Matched Matrisome genes: {numbers.get("gtex_v11_matched_matrisome_genes", "NA")}
- ECM programs scored: {numbers.get("gtex_v11_programs_scored", "NA")}
- Donors / subjects: {numbers.get("gtex_v11_subjects", "NA")}
- Tissues: {numbers.get("gtex_v11_tissues", "NA")}
- Tissue details: {numbers.get("gtex_v11_tissue_details", "NA")}

## Best deep/tabular benchmark results

### Tissue classification

- Best model: {tissue_best.get("best_model", "NA")}
- Balanced accuracy: {tissue_best.get("balanced_accuracy_mean", "NA")} ± {tissue_best.get("balanced_accuracy_std", "NA")}
- Macro-F1: {tissue_best.get("macro_f1_mean", "NA")} ± {tissue_best.get("macro_f1_std", "NA")}
- Accuracy: {tissue_best.get("accuracy_mean", "NA")} ± {tissue_best.get("accuracy_std", "NA")}

### Tissue-system classification

- Best model: {system_best.get("best_model", "NA")}
- Balanced accuracy: {system_best.get("balanced_accuracy_mean", "NA")} ± {system_best.get("balanced_accuracy_std", "NA")}
- Macro-F1: {system_best.get("macro_f1_mean", "NA")} ± {system_best.get("macro_f1_std", "NA")}
- Accuracy: {system_best.get("accuracy_mean", "NA")} ± {system_best.get("accuracy_std", "NA")}

## Main interpretation

The nine curated ECM program scores contain strong donor-generalizable tissue information. Classical models already classify GTEx tissues with high accuracy, but deep/tabular models further improve performance.

Dense MLP achieved the best balanced accuracy for tissue classification, while TabICL achieved the best balanced accuracy for tissue-system classification in the deep/tabular benchmark.

This supports the claim that the curated ECM programs are compact but highly informative descriptors of tissue identity.

## Important limitations

1. GTEx V11 is transcriptomic, not protein-level.
2. The classifier uses nine curated ECM program scores, not full Matrisome expression.
3. Deep and tabular models may exploit nonlinear program interactions, but interpretation should still prioritize the curated ECM programs themselves.
4. TabPFN failed for the full tasks because the number of classes exceeded its supported limit.
5. This benchmark validates tissue-discriminative information, not direct scaffold design.

## Suggested manuscript additions

Add or update the following figures:

- Figure 8: GTEx V11 sample-level ECM program validation.
- Figure 9: Donor-stratified tissue classification from ECM program scores.
- Figure 10: Classical ML classifier benchmark.
- Figure 11: Deep and tabular model benchmark.

## Next recommended step

Update the manuscript draft to include the GTEx V11 sample-level validation and deep/tabular benchmark sections. Then decide whether to proceed to Tabula Sapiens pseudobulk expansion or start manuscript polishing.
"""

    report_path = V04_DIR / "reports" / "final_results_v0.4_summary.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"[SAVED] {report_path}")


def main() -> None:
    reset_v04_dir()
    manifest = collect_v04_outputs()
    numbers = make_key_numbers(manifest)
    write_v04_report(numbers)

    print("\n[DONE]")
    print(f"Final v0.4 package: {V04_DIR}")
    print(f"Manifest: {V04_DIR / 'metadata' / 'gtex_v11_v0.4_manifest.csv'}")
    print(f"Report: {V04_DIR / 'reports' / 'final_results_v0.4_summary.md'}")


if __name__ == "__main__":
    main()