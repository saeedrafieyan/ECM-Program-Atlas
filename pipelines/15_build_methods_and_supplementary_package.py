from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path("results/methods_and_supplementary_package")
TABLE_DIR = OUTPUT_DIR / "tables"
REPORT_DIR = OUTPUT_DIR / "reports"
METADATA_DIR = OUTPUT_DIR / "metadata"


REQUIRED_REVIEW_TRACKS = [
    {
        "track": "R1",
        "name": "NMF rank and stability analysis",
        "main_file": "results/revision_nmf_stability/reports/nmf_rank_stability_summary.md",
        "purpose": "Justify NMF rank selection, component stability, and recovery of curated ECM programs.",
    },
    {
        "track": "R2",
        "name": "Representation-control classification benchmark",
        "main_file": "results/revision_classification_controls/reports/r2_representation_control_summary.md",
        "purpose": "Compare curated ECM programs against PCA, random Matrisome, and random non-ECM compact representations.",
    },
    {
        "track": "R3",
        "name": "MatrisomeDB null and abundance validation",
        "main_file": "results/revision_matrisomedb_null_abundance/reports/r3_matrisomedb_null_abundance_summary.md",
        "purpose": "Compare protein detection and NSAF abundance support against matched random Matrisome nulls.",
    },
    {
        "track": "R4",
        "name": "Reproducibility package",
        "main_file": "results/reproducibility_package/reports/reproducibility_package_summary.md",
        "purpose": "Export gene lists, data manifest, pipeline order, and code/data availability statement.",
    },
    {
        "track": "R5",
        "name": "Final figure integration",
        "main_file": "docs/figures/figure_plan.md",
        "purpose": "Ensure no placeholder figures remain in final manuscript.",
    },
    {
        "track": "R6",
        "name": "Supplementary tables and exact methods",
        "main_file": "results/methods_and_supplementary_package/reports/r6_methods_and_supplementary_summary.md",
        "purpose": "Create exact parameter and supplementary table package.",
    },
    {
        "track": "R7",
        "name": "Focused spatial validation",
        "main_file": "not_started",
        "purpose": "Apply ECM programs to one public spatial transcriptomics dataset.",
    },
]


SUPPLEMENTARY_TABLES = [
    {
        "table_id": "Supplementary Table S1",
        "title": "Curated ECM program gene sets",
        "file": "results/reproducibility_package/tables/Supplementary_Table_S1_curated_ecm_program_gene_sets.csv",
        "purpose": "Defines all nine ECM programs and their genes.",
        "required_for_submission": "yes",
    },
    {
        "table_id": "Supplementary Table S2",
        "title": "Curated ECM program genes, long format",
        "file": "results/reproducibility_package/tables/Supplementary_Table_S2_curated_ecm_program_genes_long.csv",
        "purpose": "One row per ECM program-gene pair.",
        "required_for_submission": "yes",
    },
    {
        "table_id": "Supplementary Table S3",
        "title": "Pipeline execution order",
        "file": "results/reproducibility_package/tables/Supplementary_Table_S3_pipeline_execution_order.csv",
        "purpose": "Documents script order for reproducibility.",
        "required_for_submission": "yes",
    },
    {
        "table_id": "Supplementary Table S4",
        "title": "Data manifest",
        "file": "results/reproducibility_package/tables/Supplementary_Table_S4_data_manifest.csv",
        "purpose": "Lists all datasets, sources, expected local paths, and usage.",
        "required_for_submission": "yes",
    },
    {
        "table_id": "Supplementary Table S5",
        "title": "Key output manifest",
        "file": "results/reproducibility_package/tables/Supplementary_Table_S5_key_output_manifest.csv",
        "purpose": "Lists key result files and their meanings.",
        "required_for_submission": "yes",
    },
    {
        "table_id": "Supplementary Table S6",
        "title": "NMF rank stability metrics",
        "file": "results/revision_nmf_stability/tables/nmf_rank_stability_metrics.csv",
        "purpose": "Reports NMF reconstruction error, stability, cophenetic correlation, and recovery metrics.",
        "required_for_submission": "yes",
    },
    {
        "table_id": "Supplementary Table S7",
        "title": "NMF curated program recovery summary",
        "file": "results/revision_nmf_stability/tables/nmf_curated_program_recovery_summary.csv",
        "purpose": "Shows recovery of curated ECM programs across ranks and feature sets.",
        "required_for_submission": "yes",
    },
    {
        "table_id": "Supplementary Table S8",
        "title": "R2 representation-control benchmark summary",
        "file": "results/revision_classification_controls/tables/r2_representation_control_summary.csv",
        "purpose": "Compares curated ECM programs against compact alternative representations.",
        "required_for_submission": "yes",
    },
    {
        "table_id": "Supplementary Table S9",
        "title": "R2 representation-family summary",
        "file": "results/revision_classification_controls/tables/r2_representation_family_summary.csv",
        "purpose": "Summarizes performance by representation family.",
        "required_for_submission": "yes",
    },
    {
        "table_id": "Supplementary Table S10",
        "title": "R3 MatrisomeDB detection null summary",
        "file": "results/revision_matrisomedb_null_abundance/tables/r3_detection_null_summary.csv",
        "purpose": "Compares protein detection coverage against matched random Matrisome gene sets.",
        "required_for_submission": "yes",
    },
    {
        "table_id": "Supplementary Table S11",
        "title": "R3 MatrisomeDB abundance null summary",
        "file": "results/revision_matrisomedb_null_abundance/tables/r3_abundance_null_summary.csv",
        "purpose": "Compares NSAF-based abundance support against matched random Matrisome gene sets.",
        "required_for_submission": "yes",
    },
    {
        "table_id": "Supplementary Table S12",
        "title": "R3 RNA-protein tissue correlation",
        "file": "results/revision_matrisomedb_null_abundance/tables/r3_rna_protein_tissue_correlation.csv",
        "purpose": "Reports tissue-level RNA-protein correlation where overlapping tissues are available.",
        "required_for_submission": "yes",
    },
    {
        "table_id": "Supplementary Table S13",
        "title": "GTEx V11 deep/tabular benchmark",
        "file": "results/tables/frozen/gtex_v11_deep_tabular_benchmark_summary.csv",
        "purpose": "Reports deep and tabular classifier performance.",
        "required_for_submission": "yes",
    },
    {
        "table_id": "Supplementary Table S14",
        "title": "Tabula Sapiens cell-source summary",
        "file": "results/tables/frozen/route_b_program_source_summary.csv",
        "purpose": "Reports likely cell-type sources of ECM programs.",
        "required_for_submission": "yes",
    },
    {
        "table_id": "Supplementary Table S15",
        "title": "Rank-based scoring correlation",
        "file": "results/tables/frozen/rank_based_all_rank_vs_mean_score_correlation.csv",
        "purpose": "Compares rank-based and mean-expression program scores.",
        "required_for_submission": "yes",
    },
]


METHOD_PARAMETERS = [
    {
        "analysis": "Matrisome filtering",
        "parameter": "gene matching",
        "value": "HGNC gene symbol, upper-case normalized",
        "reason": "Consistent gene matching across HPA, GTEx, MatrisomeDB, and Tabula Sapiens.",
    },
    {
        "analysis": "Expression transformation",
        "parameter": "bulk/tissue expression transform",
        "value": "log2(expression + 1)",
        "reason": "Reduces skew and keeps zero values defined.",
    },
    {
        "analysis": "Z-score scoring",
        "parameter": "gene scaling",
        "value": "gene-wise z-score across samples",
        "reason": "Normalizes gene-level variation before averaging program genes.",
    },
    {
        "analysis": "NMF",
        "parameter": "rank range",
        "value": "k = 2 to 20",
        "reason": "Evaluates factorization granularity and stability across ranks.",
    },
    {
        "analysis": "NMF",
        "parameter": "random seeds",
        "value": "30",
        "reason": "Evaluates component stability across initialization.",
    },
    {
        "analysis": "NMF",
        "parameter": "top genes per component",
        "value": "30",
        "reason": "Used for component comparison and program recovery.",
    },
    {
        "analysis": "NMF",
        "parameter": "recovery threshold",
        "value": ">=4 overlapping genes and overlap coefficient >=0.13",
        "reason": "Defines whether a curated program is recovered by an NMF component.",
    },
    {
        "analysis": "MatrisomeDB null",
        "parameter": "random repeats",
        "value": "1000",
        "reason": "Empirical null distribution for detection and abundance support.",
    },
    {
        "analysis": "GTEx V11 classification",
        "parameter": "cross-validation",
        "value": "GroupKFold by subject_id",
        "reason": "Prevents donor leakage across train/test folds.",
    },
    {
        "analysis": "GTEx V11 classification",
        "parameter": "minimum samples per class",
        "value": "100",
        "reason": "Removes very small classes from donor-stratified benchmark.",
    },
    {
        "analysis": "Tabula Sapiens pseudobulk",
        "parameter": "pseudobulk unit",
        "value": "donor × organ × cell type × method",
        "reason": "Avoids treating cells from the same donor as independent biological replicates.",
    },
    {
        "analysis": "Tabula Sapiens pseudobulk",
        "parameter": "minimum cells per pseudobulk group",
        "value": "20",
        "reason": "Filters unstable low-cell-count pseudobulk groups.",
    },
    {
        "analysis": "Rank-based scoring",
        "parameter": "rank_percentile_score",
        "value": "mean within-sample percentile rank of program genes",
        "reason": "Tests robustness to expression scale and normalization.",
    },
    {
        "analysis": "Rank-based scoring",
        "parameter": "top10_fraction_score",
        "value": "fraction of program genes in top 10% of Matrisome genes",
        "reason": "Alternative rank-based enrichment metric.",
    },
    {
        "analysis": "Rank-based scoring",
        "parameter": "top20_fraction_score",
        "value": "fraction of program genes in top 20% of Matrisome genes",
        "reason": "Alternative rank-based enrichment metric.",
    },
]


MODEL_PARAMETERS = [
    {
        "model": "Logistic regression",
        "parameters": "StandardScaler; class_weight=balanced; solver=lbfgs; max_iter=5000",
        "role": "Interpretable linear baseline.",
    },
    {
        "model": "XGBoost",
        "parameters": "n_estimators=500; max_depth=3; learning_rate=0.05; subsample=0.85; colsample_bytree=0.9",
        "role": "Nonlinear ensemble benchmark.",
    },
    {
        "model": "ExtraTrees",
        "parameters": "n_estimators=500; min_samples_leaf=2; class_weight=balanced",
        "role": "Tree-ensemble benchmark.",
    },
    {
        "model": "Dense MLP",
        "parameters": "2 hidden layers; hidden_dim=128; batch normalization; ReLU; dropout=0.15",
        "role": "Dense deep-learning benchmark.",
    },
    {
        "model": "Residual MLP",
        "parameters": "hidden_dim=128; 3 residual blocks; dropout=0.15",
        "role": "Residual deep-learning benchmark.",
    },
    {
        "model": "FT-Transformer-lite",
        "parameters": "d_token=64; n_heads=4; n_layers=3; dropout=0.15",
        "role": "Lightweight tabular transformer benchmark.",
    },
    {
        "model": "TabNet",
        "parameters": "n_d=16; n_a=16; n_steps=3; gamma=1.3; mask_type=entmax",
        "role": "Attention-based tabular benchmark.",
    },
    {
        "model": "TabICL",
        "parameters": "default TabICLClassifier",
        "role": "Tabular in-context learning benchmark.",
    },
    {
        "model": "TabPFN",
        "parameters": "default TabPFNClassifier",
        "role": "Attempted tabular foundation model; failed for tasks with >10 classes.",
    },
]


REVIEWER_CRITICISM_MAP = [
    {
        "criticism": "NMF rank is arbitrary and manual curation is biased.",
        "response": "Added NMF rank and stability analysis across k=2 to 20 and 30 seeds. Programs are now described as recurring biological classes consolidated across ranks and feature spaces, not as one k=9 NMF solution.",
        "supporting_output": "nmf_rank_stability_summary.md; nmf_rank_stability_metrics.csv; nmf_curated_program_recovery_summary.csv",
    },
    {
        "criticism": "Classification is trivial and dummy baseline is insufficient.",
        "response": "Added R2 representation controls comparing curated ECM programs to Matrisome PCA9, random Matrisome gene sets, and random non-ECM gene sets under the same donor-stratified classification setup.",
        "supporting_output": "r2_representation_control_summary.md; r2_representation_control_summary.csv",
    },
    {
        "criticism": "MatrisomeDB validation is too weak because it only uses detection coverage.",
        "response": "Added matched random Matrisome null models for detection coverage and NSAF abundance support, plus RNA-protein tissue-profile correlation analysis.",
        "supporting_output": "r3_matrisomedb_null_abundance_summary.md; r3_detection_null_summary.csv; r3_abundance_null_summary.csv",
    },
    {
        "criticism": "Code and data availability are insufficient.",
        "response": "Built reproducibility package with program gene lists, data manifest, pipeline order, parameter summaries, and code/data availability statement.",
        "supporting_output": "Supplementary Tables S1-S5; reproducibility_checklist.md; code_and_data_availability_statement.md",
    },
    {
        "criticism": "Bulk transcriptomics may reflect cell composition.",
        "response": "Added Tabula Sapiens donor × organ × cell-type pseudobulk analysis to identify likely cellular sources of ECM programs.",
        "supporting_output": "route_b_final_summary.md; route_b_program_source_summary.csv",
    },
    {
        "criticism": "Program scoring may depend on arbitrary mean-expression scoring.",
        "response": "Added rank-based scoring robustness analysis inspired by UCell-like scoring. Most programs remain stable under rank-based scoring.",
        "supporting_output": "rank_based_ecm_program_scoring_summary.md; rank_based_all_rank_vs_mean_score_correlation.csv",
    },
    {
        "criticism": "Spatial ECM organization is missing.",
        "response": "Acknowledged as remaining limitation and planned R7 focused spatial validation using one public Visium dataset.",
        "supporting_output": "Pending R7.",
    },
]


def ensure_dirs(output_dir: Path) -> tuple[Path, Path, Path]:
    table_dir = output_dir / "tables"
    report_dir = output_dir / "reports"
    metadata_dir = output_dir / "metadata"

    for folder in [table_dir, report_dir, metadata_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    return table_dir, report_dir, metadata_dir


def run_command(command: list[str]) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        output = result.stdout.strip()

        if result.stderr.strip():
            output += "\nSTDERR:\n" + result.stderr.strip()

        return output.strip()
    except Exception as exc:
        return f"FAILED: {exc}"


def get_git_info() -> dict:
    return {
        "git_commit": run_command(["git", "rev-parse", "HEAD"]),
        "git_branch": run_command(["git", "branch", "--show-current"]),
        "git_status_short": run_command(["git", "status", "--short"]),
        "git_remote": run_command(["git", "remote", "-v"]),
    }


def get_environment_info() -> dict:
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "pip_freeze": run_command([sys.executable, "-m", "pip", "freeze"]),
    }


def write_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"[SAVED] {path}")


def build_program_gene_tables(program_table: Path, table_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not program_table.exists():
        raise FileNotFoundError(f"Missing curated program table: {program_table}")

    programs = load_programs_from_curated_table(
        str(program_table),
        program_col="ecm_program_curated",
        genes_col="top_genes",
    )

    program_rows = []
    long_rows = []

    for program in programs:
        program_rows.append(
            {
                "ecm_program": program.name,
                "n_genes": program.n_genes,
                "genes": ", ".join(program.genes),
            }
        )

        for rank, gene in enumerate(program.genes, start=1):
            long_rows.append(
                {
                    "ecm_program": program.name,
                    "gene_rank_alphabetical": rank,
                    "gene_symbol": gene,
                }
            )

    program_gene_table = pd.DataFrame(program_rows)
    program_gene_long = pd.DataFrame(long_rows)

    program_gene_table.to_csv(
        table_dir / "Supplementary_Table_S1_curated_ecm_program_gene_sets.csv",
        index=False,
    )

    program_gene_long.to_csv(
        table_dir / "Supplementary_Table_S2_curated_ecm_program_genes_long.csv",
        index=False,
    )

    print(f"[SAVED] {table_dir / 'Supplementary_Table_S1_curated_ecm_program_gene_sets.csv'}")
    print(f"[SAVED] {table_dir / 'Supplementary_Table_S2_curated_ecm_program_genes_long.csv'}")

    return program_gene_table, program_gene_long


def build_dataframe_table(data: list[dict], output_path: Path) -> pd.DataFrame:
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"[SAVED] {output_path}")
    return df


def write_code_data_availability(report_dir: Path, repo_url: str | None) -> None:
    repo_text = repo_url if repo_url else "[GitHub repository URL to be added]"

    text = f"""# Code and Data Availability Statement

All analysis code is available at:

{repo_text}

Large raw datasets are not distributed in this repository due to file size and third-party data-use restrictions. The repository provides:

1. Pipeline scripts required to reproduce the analysis.
2. Configuration files describing expected input paths.
3. Curated ECM program gene sets.
4. Frozen summary tables and reports.
5. Reproducibility metadata, including software environment information.

## External data sources

The analysis uses public datasets from:

- Human Protein Atlas
- GTEx Portal
- Human Matrisome annotations
- MatrisomeDB
- Tabula Sapiens

Users should download the source datasets from their original providers and place them in the expected `data/raw/` subdirectories.

## Processed outputs

Processed matrices are excluded from Git due to size. Small summary tables and supplementary gene lists are provided in the reproducibility package.

## Reproducibility

Pipeline scripts are located in `pipelines/` and should be executed in numerical order. The exact pipeline order is provided in:

`Supplementary_Table_S3_pipeline_execution_order.csv`
"""

    path = report_dir / "code_and_data_availability_statement.md"
    path.write_text(text, encoding="utf-8")
    print(f"[SAVED] {path}")


def write_reproducibility_checklist(report_dir: Path) -> None:
    text = """# Reproducibility Checklist

## Code

- [x] Analysis scripts are organized in `pipelines/`.
- [x] Reusable utilities are organized in `src/ecm_program_atlas/`.
- [x] Core scoring utilities are tested.
- [x] Environment files are provided.

## Data

- [x] Raw data sources are listed.
- [x] Large raw files are excluded from Git.
- [x] Expected local data paths are documented.
- [x] Curated ECM program gene lists are exported.

## Methods

- [x] NMF rank and stability analysis is included.
- [x] Matched random non-ECM controls are included.
- [x] Matched random Matrisome controls are included.
- [x] MatrisomeDB detection and abundance null models are included.
- [x] GTEx donor-stratified validation is included.
- [x] Tabula Sapiens pseudobulk source validation is included.
- [x] Rank-based scoring robustness is included.

## Manuscript requirements

- [ ] Insert final figures.
- [ ] Add exact software versions from environment report.
- [ ] Add data accession/download dates.
- [ ] Add final code repository URL.
- [ ] Add supplementary tables to submission package.
"""

    path = report_dir / "reproducibility_checklist.md"
    path.write_text(text, encoding="utf-8")
    print(f"[SAVED] {path}")


def write_methods_parameter_summary(report_dir: Path) -> None:
    text = """# Methods and Parameter Summary

## ECM program definition

Curated ECM programs were derived from NMF modules across Matrisome feature spaces and consolidated based on gene composition, tissue enrichment, and cross-dataset recovery.

## NMF stability

- NMF ranks: k = 2 to 20
- Random seeds: 30
- Top genes per component: 30
- Metrics: reconstruction error, top-gene Jaccard stability, consensus cophenetic correlation, program recovery fraction

## GTEx donor-stratified classification

- Cross-validation: GroupKFold by subject_id
- Tissue-system task: 13 classes
- Tissue task: 27 classes after minimum sample filtering
- Features: 9 ECM program scores unless otherwise stated
- Main metrics: accuracy, balanced accuracy, macro-F1, weighted-F1

## Representation controls

Compared:
- Curated 9 ECM programs
- Matrisome PCA9
- Random Matrisome gene-set programs
- Random non-ECM gene-set programs

## MatrisomeDB validation

Compared curated ECM programs against random Matrisome gene sets using:
- Protein detection coverage
- NSAF-based top-tissue abundance scores
- RNA-protein tissue correlation

## Tabula Sapiens pseudobulk

Pseudobulk unit:
- donor × organ × cell type × method

Default filter:
- minimum 20 cells per pseudobulk group

## Rank-based scoring

Scores:
- rank_percentile_score
- top10_fraction_score
- top20_fraction_score
"""

    path = report_dir / "methods_parameter_summary.md"
    path.write_text(text, encoding="utf-8")
    print(f"[SAVED] {path}")


def write_summary_report(
    report_dir: Path,
    program_gene_table: pd.DataFrame,
    pipeline_df: pd.DataFrame,
    data_df: pd.DataFrame,
) -> None:
    text = f"""# Reproducibility Package Summary

## Purpose

This package provides the core reproducibility materials for the ECM Program Atlas project.

## Contents

- Curated ECM program gene lists
- Long-form program gene table
- Pipeline execution order
- Data manifest
- Key output manifest
- Code and data availability statement
- Environment report
- Git metadata
- Reproducibility checklist
- Methods parameter summary
- Reviewer criticism response map

## Key numbers

- Curated ECM programs: {program_gene_table.shape[0]}
- Total program-gene rows: {program_gene_table['n_genes'].sum()}
- Pipeline steps documented: {pipeline_df.shape[0]}
- External datasets documented: {data_df.shape[0]}

## Important interpretation

This reproducibility package supports the manuscript claim that the ECM programs are explicitly defined, reusable, and computationally reproducible from documented inputs and scripts.

## Important limitation

Large raw and processed datasets are not included in Git. Users must download them from the original providers and place them in the documented local paths.
"""

    path = report_dir / "reproducibility_package_summary.md"
    path.write_text(text, encoding="utf-8")
    print(f"[SAVED] {path}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--program-table", type=Path, default=DEFAULT_PROGRAM_TABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repo-url", type=str, default=None)

    args = parser.parse_args()

    table_dir, report_dir, metadata_dir = ensure_dirs(args.output_dir)

    program_gene_table, program_gene_long = build_program_gene_tables(
        program_table=args.program_table,
        table_dir=table_dir,
    )

    pipeline_df = build_dataframe_table(
        PIPELINE_ORDER,
        table_dir / "Supplementary_Table_S3_pipeline_execution_order.csv",
    )

    data_df = build_dataframe_table(
        DATA_MANIFEST,
        table_dir / "Supplementary_Table_S4_data_manifest.csv",
    )

    build_dataframe_table(
        [
            {
                "output": "combined_nmf_module_annotations_curated_programs.csv",
                "description": "Curated NMF module annotations and ECM program labels.",
                "recommended_location": "results/tables/frozen/",
            },
            {
                "output": "nmf_rank_stability_summary.md",
                "description": "NMF rank and stability analysis report.",
                "recommended_location": "results/revision_nmf_stability/reports/",
            },
            {
                "output": "r2_representation_control_summary.md",
                "description": "Representation control benchmark report.",
                "recommended_location": "results/revision_classification_controls/reports/",
            },
            {
                "output": "r3_matrisomedb_null_abundance_summary.md",
                "description": "MatrisomeDB null and abundance validation report.",
                "recommended_location": "results/revision_matrisomedb_null_abundance/reports/",
            },
            {
                "output": "rank_based_ecm_program_scoring_summary.md",
                "description": "Rank-based scoring robustness report.",
                "recommended_location": "results/rank_based_scoring/reports/",
            },
            {
                "output": "route_b_final_summary.md",
                "description": "Tabula Sapiens cell-source validation report.",
                "recommended_location": "results/reports/frozen/",
            },
        ],
        table_dir / "Supplementary_Table_S5_key_output_manifest.csv",
    )

    build_dataframe_table(
        METHOD_PARAMETERS,
        table_dir / "Supplementary_Table_S6_exact_method_parameters.csv",
    )

    build_dataframe_table(
        MODEL_PARAMETERS,
        table_dir / "Supplementary_Table_S7_model_parameters.csv",
    )

    build_dataframe_table(
        REVIEWER_CRITICISM_MAP,
        table_dir / "Supplementary_Table_S8_reviewer_criticism_response_map.csv",
    )

    write_json(get_git_info(), metadata_dir / "git_info.json")
    write_json(get_environment_info(), metadata_dir / "environment_info.json")

    write_code_data_availability(report_dir, args.repo_url)
    write_reproducibility_checklist(report_dir)
    write_methods_parameter_summary(report_dir)

    write_summary_report(
        report_dir=report_dir,
        program_gene_table=program_gene_table,
        pipeline_df=pipeline_df,
        data_df=data_df,
    )

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "program_table": str(args.program_table),
        "output_dir": str(args.output_dir),
        "n_programs": int(program_gene_table.shape[0]),
        "n_program_gene_rows": int(program_gene_long.shape[0]),
    }

    write_json(metadata, metadata_dir / "reproducibility_package_metadata.json")

    print("\n[DONE]")
    print(f"Output folder: {args.output_dir}")
    print(f"Tables: {table_dir}")
    print(f"Reports: {report_dir}")
    print(f"Metadata: {metadata_dir}")


if __name__ == "__main__":
    main()