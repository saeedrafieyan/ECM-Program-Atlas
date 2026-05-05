from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(".")
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
FINAL_DIR = OUTPUT_ROOT / "final_results_v0.1"


FILES_TO_COLLECT = {
    # Stage 1, data construction
    "data_matrix_summary": OUTPUT_ROOT / ".." / "data" / "processed" / "all_dataset_summary.csv",

    # Stage 2, exploratory analysis
    "eda_tissue_consensus_pca": OUTPUT_ROOT / "eda" / "rna_tissue_consensus" / "pca_pc1_pc2.png",
    "eda_tissue_consensus_umap": OUTPUT_ROOT / "eda" / "rna_tissue_consensus" / "umap.png",
    "eda_tissue_consensus_heatmap": OUTPUT_ROOT / "eda" / "rna_tissue_consensus" / "tissue_correlation_heatmap.png",
    "eda_tissue_consensus_clusters": OUTPUT_ROOT / "eda" / "rna_tissue_consensus" / "agglomerative_clusters.csv",
    "eda_tissue_consensus_neighbors": OUTPUT_ROOT / "eda" / "rna_tissue_consensus" / "nearest_tissue_neighbors.csv",

    # Stage 3, ECM specificity
    "ecm_specificity_metric_summary": OUTPUT_ROOT / "specificity" / "rna_tissue_consensus" / "metric_comparison_summary.csv",
    "ecm_specificity_reference_metrics": OUTPUT_ROOT / "specificity" / "rna_tissue_consensus" / "reference_metrics.csv",

    # Stage 4, Matrisome category analysis
    "category_gene_counts": OUTPUT_ROOT / "category_analysis" / "rna_tissue_consensus" / "category_gene_counts.csv",
    "category_observed_metrics": OUTPUT_ROOT / "category_analysis" / "rna_tissue_consensus" / "observed_category_metrics.csv",
    "category_metric_comparison": OUTPUT_ROOT / "category_analysis" / "rna_tissue_consensus" / "category_metric_comparison_summary.csv",

    # Stage 5, cross-dataset category reproducibility
    "cross_dataset_summary": OUTPUT_ROOT / "cross_dataset_reproducibility" / "cross_dataset_summary.csv",
    "cross_dataset_key_metrics": OUTPUT_ROOT / "cross_dataset_reproducibility" / "cross_dataset_key_metric_summary.csv",
    "cross_dataset_metric_comparison": OUTPUT_ROOT / "cross_dataset_reproducibility" / "cross_dataset_metric_comparison.csv",

    # Stage 6, baseline latent embeddings
    "latent_embedding_global_summary": OUTPUT_ROOT / "latent_baseline_embeddings" / "latent_embedding_global_summary.csv",

    # Stage 7, curated NMF module annotations
    "combined_nmf_module_annotations": OUTPUT_ROOT / "latent_baseline_embeddings" / "rna_tissue_consensus" / "combined_nmf_module_annotations.csv",
    "curated_nmf_module_annotations": OUTPUT_ROOT / "latent_baseline_embeddings" / "rna_tissue_consensus" / "curated_recurring_ecm_programs" / "combined_nmf_module_annotations_curated_programs.csv",
    "curated_program_summary": OUTPUT_ROOT / "latent_baseline_embeddings" / "rna_tissue_consensus" / "curated_recurring_ecm_programs" / "curated_recurring_ecm_program_summary.csv",
    "curated_program_presence_matrix": OUTPUT_ROOT / "latent_baseline_embeddings" / "rna_tissue_consensus" / "curated_recurring_ecm_programs" / "curated_ecm_program_presence_matrix.csv",
    "curated_program_high_confidence_matrix": OUTPUT_ROOT / "latent_baseline_embeddings" / "rna_tissue_consensus" / "curated_recurring_ecm_programs" / "curated_ecm_program_high_confidence_matrix.csv",
    "curated_program_presence_heatmap": OUTPUT_ROOT / "latent_baseline_embeddings" / "rna_tissue_consensus" / "curated_recurring_ecm_programs" / "curated_ecm_program_presence_heatmap.html",
    "curated_program_high_confidence_heatmap": OUTPUT_ROOT / "latent_baseline_embeddings" / "rna_tissue_consensus" / "curated_recurring_ecm_programs" / "curated_ecm_program_high_confidence_heatmap.html",

    # Stage 8, cross-dataset NMF program reproducibility
    "cross_dataset_nmf_program_summary": OUTPUT_ROOT / "latent_baseline_embeddings" / "cross_dataset_nmf_program_reproducibility" / "cross_dataset_program_reproducibility_summary.csv",
    "cross_dataset_nmf_program_presence": OUTPUT_ROOT / "latent_baseline_embeddings" / "cross_dataset_nmf_program_reproducibility" / "cross_dataset_program_presence_matrix.csv",
    "cross_feature_set_nmf_program_presence": OUTPUT_ROOT / "latent_baseline_embeddings" / "cross_dataset_nmf_program_reproducibility" / "cross_feature_set_program_presence_matrix.csv",
    "best_module_reference_matches": OUTPUT_ROOT / "latent_baseline_embeddings" / "cross_dataset_nmf_program_reproducibility" / "best_module_reference_matches.csv",

    # Stage 9, external-only reproducibility
    "external_program_reproducibility_summary": OUTPUT_ROOT / "latent_baseline_embeddings" / "cross_dataset_nmf_program_reproducibility" / "external_only" / "external_program_reproducibility_summary.csv",
    "external_dataset_program_presence": OUTPUT_ROOT / "latent_baseline_embeddings" / "cross_dataset_nmf_program_reproducibility" / "external_only" / "external_dataset_program_presence_matrix.csv",
    "external_feature_set_program_presence": OUTPUT_ROOT / "latent_baseline_embeddings" / "cross_dataset_nmf_program_reproducibility" / "external_only" / "external_feature_set_program_presence_matrix.csv",
}


FIGURE_FILES_TO_COLLECT = {
    # Interactive latent-space figures, tissue consensus, core Matrisome
    "core_matrisome_pca_plotly": OUTPUT_ROOT / "latent_baseline_embeddings" / "rna_tissue_consensus" / "core_matrisome" / "pca_pc1_pc2.html",
    "core_matrisome_umap_plotly": OUTPUT_ROOT / "latent_baseline_embeddings" / "rna_tissue_consensus" / "core_matrisome" / "umap.html",
    "core_matrisome_nmf_plotly": OUTPUT_ROOT / "latent_baseline_embeddings" / "rna_tissue_consensus" / "core_matrisome" / "nmf_component1_component2.html",

    # Interactive module activity figures
    "core_matrisome_nmf_activity_heatmap": OUTPUT_ROOT / "latent_baseline_embeddings" / "rna_tissue_consensus" / "core_matrisome" / "nmf_module_activity" / "nmf_module_activity_heatmap.html",
    "curated_program_presence_heatmap": OUTPUT_ROOT / "latent_baseline_embeddings" / "rna_tissue_consensus" / "curated_recurring_ecm_programs" / "curated_ecm_program_presence_heatmap.html",
    "curated_program_high_confidence_heatmap": OUTPUT_ROOT / "latent_baseline_embeddings" / "rna_tissue_consensus" / "curated_recurring_ecm_programs" / "curated_ecm_program_high_confidence_heatmap.html",

    # External reproducibility figures
    "external_dataset_program_presence_heatmap": OUTPUT_ROOT / "latent_baseline_embeddings" / "cross_dataset_nmf_program_reproducibility" / "external_only" / "external_dataset_program_presence_heatmap.html",
    "external_feature_set_program_presence_heatmap": OUTPUT_ROOT / "latent_baseline_embeddings" / "cross_dataset_nmf_program_reproducibility" / "external_only" / "external_feature_set_program_presence_heatmap.html",
}


def reset_final_dir() -> None:
    if FINAL_DIR.exists():
        shutil.rmtree(FINAL_DIR)

    (FINAL_DIR / "tables").mkdir(parents=True, exist_ok=True)
    (FINAL_DIR / "figures").mkdir(parents=True, exist_ok=True)
    (FINAL_DIR / "reports").mkdir(parents=True, exist_ok=True)
    (FINAL_DIR / "metadata").mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path) -> bool:
    src = src.resolve()

    if not src.exists():
        print(f"[MISSING] {src}")
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[COPIED] {src} -> {dst}")
    return True


def collect_files() -> pd.DataFrame:
    records = []

    for logical_name, src in FILES_TO_COLLECT.items():
        suffix = src.suffix.lower()

        if suffix in [".csv", ".tsv", ".xlsx"]:
            dst = FINAL_DIR / "tables" / f"{logical_name}{suffix}"
        elif suffix in [".png", ".jpg", ".jpeg", ".svg", ".html"]:
            dst = FINAL_DIR / "figures" / f"{logical_name}{suffix}"
        else:
            dst = FINAL_DIR / "metadata" / f"{logical_name}{suffix}"

        copied = copy_file(src, dst)

        records.append(
            {
                "logical_name": logical_name,
                "source_path": str(src),
                "final_path": str(dst) if copied else "",
                "status": "copied" if copied else "missing",
            }
        )

    for logical_name, src in FIGURE_FILES_TO_COLLECT.items():
        suffix = src.suffix.lower()
        dst = FINAL_DIR / "figures" / f"{logical_name}{suffix}"

        copied = copy_file(src, dst)

        records.append(
            {
                "logical_name": logical_name,
                "source_path": str(src),
                "final_path": str(dst) if copied else "",
                "status": "copied" if copied else "missing",
            }
        )

    manifest = pd.DataFrame(records)
    manifest.to_csv(FINAL_DIR / "metadata" / "final_results_file_manifest.csv", index=False)

    return manifest


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[WARNING] Failed to read {path}: {exc}")
        return None


def create_key_result_tables() -> None:
    """
    Creates smaller manuscript-friendly tables from large pipeline outputs.
    """
    table_dir = FINAL_DIR / "tables"

    # 1. Dataset summary
    dataset_summary = safe_read_csv(table_dir / "cross_dataset_summary.csv")
    if dataset_summary is not None:
        dataset_summary.to_csv(table_dir / "Table_1_dataset_summary.csv", index=False)

    # 2. ECM specificity, key metrics only
    specificity = safe_read_csv(table_dir / "ecm_specificity_metric_summary.csv")
    if specificity is not None:
        key_metrics = [
            "pca_pc1_variance",
            "pca_pc1_pc2_variance",
            "silhouette_original_space",
            "silhouette_pca10_space",
            "nearest_neighbor_same_system_at_3",
            "nearest_neighbor_same_system_at_5",
            "ari_agglomerative_vs_system",
            "nmi_agglomerative_vs_system",
        ]

        if "metric" in specificity.columns:
            specificity_key = specificity[specificity["metric"].isin(key_metrics)].copy()
        else:
            specificity_key = specificity.copy()

        specificity_key.to_csv(table_dir / "Table_2_ecm_vs_random_specificity.csv", index=False)

    # 3. Matrisome category benchmark
    category_metrics = safe_read_csv(table_dir / "category_metric_comparison.csv")
    if category_metrics is not None:
        key_metrics = [
            "pca_pc1_pc2_variance",
            "silhouette_pca10_space",
            "nearest_neighbor_same_system_at_3",
            "ari_agglomerative_vs_system",
            "nmi_agglomerative_vs_system",
        ]

        if "metric" in category_metrics.columns:
            category_key = category_metrics[
                category_metrics["metric"].isin(key_metrics)
            ].copy()
        else:
            category_key = category_metrics.copy()

        category_key.to_csv(table_dir / "Table_3_matrisome_category_benchmark.csv", index=False)

    # 4. Curated ECM programs
    curated_summary = safe_read_csv(table_dir / "curated_program_summary.csv")
    if curated_summary is not None:
        curated_summary.to_csv(table_dir / "Table_4_curated_recurring_ecm_programs.csv", index=False)

    # 5. External reproducibility
    external_summary = safe_read_csv(table_dir / "external_program_reproducibility_summary.csv")
    if external_summary is not None:
        external_summary.to_csv(table_dir / "Table_5_external_program_reproducibility.csv", index=False)

    # 6. Curated NMF annotations
    curated_annotations = safe_read_csv(table_dir / "curated_nmf_module_annotations.csv")
    if curated_annotations is not None:
        curated_annotations.to_csv(table_dir / "Supplementary_Table_curated_nmf_modules.csv", index=False)

    print("[INFO] Manuscript-friendly tables created.")


def summarize_key_numbers() -> dict:
    table_dir = FINAL_DIR / "tables"

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_version": "v0.1",
        "project_scope": "Matrisome-derived ECM representation and reproducibility framework",
    }

    dataset_summary = safe_read_csv(table_dir / "data_matrix_summary.csv")
    if dataset_summary is not None:
        summary["processed_dataset_count"] = int(dataset_summary.shape[0])

        if "n_matched_ecm_genes" in dataset_summary.columns:
            summary["max_matched_ecm_genes"] = int(dataset_summary["n_matched_ecm_genes"].max())

        if "n_missing_matrisome_genes" in dataset_summary.columns:
            summary["min_missing_matrisome_genes"] = int(dataset_summary["n_missing_matrisome_genes"].min())

    latent_summary = safe_read_csv(table_dir / "latent_embedding_global_summary.csv")
    if latent_summary is not None:
        summary["latent_embedding_runs"] = int(latent_summary.shape[0])

        if "dataset" in latent_summary.columns:
            summary["latent_embedding_datasets"] = sorted(latent_summary["dataset"].dropna().unique().tolist())

        if "feature_set" in latent_summary.columns:
            summary["latent_embedding_feature_sets"] = sorted(latent_summary["feature_set"].dropna().unique().tolist())

    curated_program_summary = safe_read_csv(table_dir / "curated_program_summary.csv")
    if curated_program_summary is not None:
        summary["curated_ecm_program_count"] = int(curated_program_summary.shape[0])

        if "ecm_program_curated" in curated_program_summary.columns:
            program_col = "ecm_program_curated"
        elif "ecm_program" in curated_program_summary.columns:
            program_col = "ecm_program"
        else:
            program_col = None

        if program_col is not None:
            summary["curated_ecm_programs"] = curated_program_summary[program_col].astype(str).tolist()

    external_summary = safe_read_csv(table_dir / "external_program_reproducibility_summary.csv")
    if external_summary is not None:
        summary["externally_reproduced_program_count"] = int(external_summary.shape[0])

        if "ecm_program" in external_summary.columns:
            summary["externally_reproduced_programs"] = external_summary["ecm_program"].astype(str).tolist()

    with open(FINAL_DIR / "metadata" / "key_result_numbers.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def write_markdown_report(key_numbers: dict, manifest: pd.DataFrame) -> None:
    missing_count = int((manifest["status"] == "missing").sum())
    copied_count = int((manifest["status"] == "copied").sum())

    externally_reproduced = key_numbers.get("externally_reproduced_programs", [])
    curated_programs = key_numbers.get("curated_ecm_programs", [])

    report = f"""# ECM Latent Space Project, Final Results Package v0.1

## Purpose

This folder consolidates the current validated results for the Matrisome-derived ECM representation project.

The current project is not a foundation model. It is a reproducible ECM representation and benchmarking framework built from Human Protein Atlas, GTEx-derived expression resources, and Human Matrisome annotations.

## Package creation

- Created at: {key_numbers.get("created_at", "unknown")}
- Files copied: {copied_count}
- Files missing: {missing_count}

## Current scope

The current analysis supports a tissue-level and category-level ECM representation framework. It does not yet support direct biomaterial inverse design.

## Main completed stages

1. Human Matrisome gene matching to HPA/GTEx-derived expression resources.
2. ECM-only tissue representation using PCA, UMAP, clustering, and nearest-neighbor analysis.
3. Matrisome genes versus matched random non-ECM baseline testing.
4. Matrisome category-level benchmarking.
5. Cross-dataset category reproducibility analysis.
6. Baseline PCA, NMF, and UMAP ECM latent embeddings.
7. Curated NMF module annotation.
8. Cross-dataset NMF program reproducibility.
9. External-only reproducibility excluding the reference tissue-consensus dataset.

## Key numbers

- Processed dataset count: {key_numbers.get("processed_dataset_count", "NA")}
- Maximum matched ECM genes: {key_numbers.get("max_matched_ecm_genes", "NA")}
- Minimum missing Matrisome genes: {key_numbers.get("min_missing_matrisome_genes", "NA")}
- Latent embedding runs: {key_numbers.get("latent_embedding_runs", "NA")}
- Curated ECM program count: {key_numbers.get("curated_ecm_program_count", "NA")}
- Externally reproduced ECM program count: {key_numbers.get("externally_reproduced_program_count", "NA")}

## Curated ECM programs

{chr(10).join([f"- {program}" for program in curated_programs]) if curated_programs else "NA"}

## Externally reproduced ECM programs

These programs reproduced after excluding the reference dataset `rna_tissue_consensus`:

{chr(10).join([f"- {program}" for program in externally_reproduced]) if externally_reproduced else "NA"}

## Main interpretation

The strongest current result is that Matrisome-derived expression decomposes into recurring ECM programs. The most reproducible programs include vascular/stromal/interstitial ECM, epithelial/mucosal basement membrane ECM, CNS/neural ECM, and hepatic/plasma-associated extracellular protein signatures.

The external-only reproducibility analysis is especially important because it tests whether reference-defined ECM programs are recovered outside the dataset used to define them.

## Important limitations

1. Current data are mainly transcriptomic and aggregated by tissue, source tissue, brain subregion, or cell type.
2. RNA expression does not directly equal deposited ECM protein abundance.
3. The current representation does not capture spatial ECM architecture, fiber organization, mechanical properties, or remodeling dynamics.
4. NMF program matching uses top-gene overlap, which is interpretable but relatively simple.
5. Direct ECM-to-biomaterial mapping has not been performed yet.

## Recommended next steps

1. Prepare manuscript-ready figures and tables from this final results package.
2. Validate transcriptomic ECM programs using MatrisomeDB proteomics.
3. Add GTEx sample-level expression to increase sample count and donor-level variability.
4. Add Tabula Sapiens or other single-cell atlases as pseudobulk organ-cell-type ECM profiles.
5. Later, connect the ECM embedding framework to scaffold and biomaterial datasets.
"""

    report_path = FINAL_DIR / "reports" / "final_results_v0.1_summary.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"[SAVED] {report_path}")


def main() -> None:
    reset_final_dir()

    print("[INFO] Collecting result files...")
    manifest = collect_files()

    print("[INFO] Creating manuscript-friendlyINFO] Collecting result files...")
    manifest = collect_files()

    print("[INFO] Creating manuscript-friendly tables...")
    create_key_result_tables()

    print("[INFO] Summarizing key numbers...")
    key_numbers = summarize_key_numbers()

    print("[INFO] Writing markdown report...")
    write_markdown_report(key_numbers=key_numbers, manifest=manifest)

    print("\n[DONE]")
    print(f"Final results package created at: {FINAL_DIR}")
    print(f"Manifest: {FINAL_DIR / 'metadata' / 'final_results_file_manifest.csv'}")
    print(f"Report: {FINAL_DIR / 'reports' / 'final_results_v0.1_summary.md'}")


if __name__ == "__main__":
    main()