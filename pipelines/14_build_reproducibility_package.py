from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

from ecm_program_atlas.scoring import load_programs_from_curated_table


DEFAULT_PROGRAM_TABLE = Path(
    "results/tables/frozen/combined_nmf_module_annotations_curated_programs.csv"
)

DEFAULT_OUTPUT_DIR = Path("results/reproducibility_package")


PIPELINE_ORDER = [
    {
        "step": "00",
        "script": "pipelines/00_inspect_inputs.py",
        "purpose": "Inspect downloaded raw input files and metadata columns.",
        "main_outputs": "Input inspection reports.",
    },
    {
        "step": "01",
        "script": "pipelines/01_build_matrisome_matrices.py",
        "purpose": "Build Matrisome-filtered expression matrices.",
        "main_outputs": "Processed sample × Matrisome gene matrices.",
    },
    {
        "step": "02",
        "script": "pipelines/02_run_ecm_specificity_controls.py",
        "purpose": "Compare Matrisome genes against matched random non-ECM baselines.",
        "main_outputs": "ECM specificity metrics and random baseline comparisons.",
    },
    {
        "step": "03",
        "script": "pipelines/03_run_category_benchmark.py",
        "purpose": "Benchmark Matrisome divisions and categories against matched random controls.",
        "main_outputs": "Matrisome category benchmark tables and plots.",
    },
    {
        "step": "04",
        "script": "pipelines/04_run_nmf_program_discovery.py",
        "purpose": "Generate baseline PCA, UMAP, and NMF embeddings and initial ECM modules.",
        "main_outputs": "Latent embeddings, NMF gene weights, and module summaries.",
    },
    {
        "step": "05",
        "script": "pipelines/05_run_external_reproducibility.py",
        "purpose": "Evaluate cross-dataset reproducibility of ECM programs.",
        "main_outputs": "External reproducibility tables and heatmaps.",
    },
    {
        "step": "06",
        "script": "pipelines/06_run_matrisomedb_validation.py",
        "purpose": "Build MatrisomeDB protein-level matrices and initial validation outputs.",
        "main_outputs": "MatrisomeDB protein matrices and validation summaries.",
    },
    {
        "step": "07",
        "script": "pipelines/07_run_gtex_v11_validation.py",
        "purpose": "Build GTEx V11 sample-level Matrisome matrix and ECM program scores.",
        "main_outputs": "GTEx V11 ECM program scores and tissue summaries.",
    },
    {
        "step": "08",
        "script": "pipelines/08_run_gtex_classifier_benchmarks.py",
        "purpose": "Run donor-stratified tissue classification from ECM program scores.",
        "main_outputs": "Classification metrics, predictions, and confusion matrices.",
    },
    {
        "step": "09",
        "script": "pipelines/09_run_tabula_sapiens_pseudobulk.py",
        "purpose": "Build Tabula Sapiens donor × organ × cell-type pseudobulk ECM scores.",
        "main_outputs": "Pseudobulk ECM program scores and cell-type summaries.",
    },
    {
        "step": "10",
        "script": "pipelines/10_run_rank_based_scoring.py",
        "purpose": "Run rank-based ECM program scoring robustness analysis.",
        "main_outputs": "Rank-vs-mean correlations and rank-based heatmaps.",
    },
    {
        "step": "11",
        "script": "pipelines/11_run_nmf_stability_analysis.py",
        "purpose": "Evaluate NMF rank selection, component stability, and program recovery.",
        "main_outputs": "NMF rank stability metrics and recovery summaries.",
    },
    {
        "step": "12",
        "script": "pipelines/12_run_representation_controls.py",
        "purpose": "Compare curated ECM programs against PCA and random compact representations.",
        "main_outputs": "Representation-control classification benchmarks.",
    },
    {
        "step": "13",
        "script": "pipelines/13_run_matrisomedb_null_abundance_validation.py",
        "purpose": "Compare MatrisomeDB detection and NSAF abundance support against random Matrisome nulls.",
        "main_outputs": "Protein detection nulls, abundance nulls, and RNA-protein correlations.",
    },
    {
        "step": "14",
        "script": "pipelines/14_build_reproducibility_package.py",
        "purpose": "Build final reproducibility package, gene lists, manifests, and code/data statements.",
        "main_outputs": "Supplementary gene lists, manifests, reproducibility checklist.",
    },
]


DATA_MANIFEST = [
    {
        "dataset": "Human Matrisome annotations",
        "source": "Matrisome Project",
        "local_expected_path": "data/raw/matrisome/human_matrisome.xlsx",
        "used_for": "Defines ECM/matrisome gene universe and categories.",
        "tracked_in_git": "No",
        "reason_not_tracked": "External data file.",
    },
    {
        "dataset": "HPA tissue consensus RNA",
        "source": "Human Protein Atlas",
        "local_expected_path": "data/raw/hpa/rna_tissue_consensus.tsv.zip",
        "used_for": "Reference tissue-level ECM matrix and initial program discovery.",
        "tracked_in_git": "No",
        "reason_not_tracked": "External data file.",
    },
    {
        "dataset": "HPA tissue RNA",
        "source": "Human Protein Atlas",
        "local_expected_path": "data/raw/hpa/rna_tissue_hpa.tsv.zip",
        "used_for": "External transcriptomic reproducibility.",
        "tracked_in_git": "No",
        "reason_not_tracked": "External data file.",
    },
    {
        "dataset": "GTEx tissue RNA",
        "source": "Human Protein Atlas / GTEx-derived",
        "local_expected_path": "data/raw/hpa/rna_tissue_gtex.tsv.zip",
        "used_for": "External transcriptomic reproducibility.",
        "tracked_in_git": "No",
        "reason_not_tracked": "External data file.",
    },
    {
        "dataset": "GTEx tissue-detail RNA",
        "source": "Human Protein Atlas / GTEx-derived",
        "local_expected_path": "data/raw/hpa/rna_tissue_detail_gtex.tsv.zip",
        "used_for": "External transcriptomic reproducibility with finer tissue labels.",
        "tracked_in_git": "No",
        "reason_not_tracked": "External data file.",
    },
    {
        "dataset": "HPA brain RNA",
        "source": "Human Protein Atlas",
        "local_expected_path": "data/raw/hpa/rna_brain_hpa.tsv.zip",
        "used_for": "Brain-region ECM program reproducibility.",
        "tracked_in_git": "No",
        "reason_not_tracked": "External data file.",
    },
    {
        "dataset": "HPA single-cell-type RNA",
        "source": "Human Protein Atlas",
        "local_expected_path": "data/raw/hpa/rna_single_cell_type.tsv.zip",
        "used_for": "Cell-type-level expression validation.",
        "tracked_in_git": "No",
        "reason_not_tracked": "External data file.",
    },
    {
        "dataset": "MatrisomeDB human export",
        "source": "MatrisomeDB",
        "local_expected_path": "data/raw/matrisomedb/matrisomedb_human_export.tsv",
        "used_for": "Protein-level ECM support and NSAF abundance validation.",
        "tracked_in_git": "No",
        "reason_not_tracked": "External exported data file.",
    },
    {
        "dataset": "GTEx V11 sample-level gene TPM",
        "source": "GTEx Portal",
        "local_expected_path": "data/raw/gtex_v11_sample_level/GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_tpm.parquet",
        "used_for": "Donor-level ECM program scoring and classification.",
        "tracked_in_git": "No",
        "reason_not_tracked": "Large external file.",
    },
    {
        "dataset": "GTEx V11 sample attributes",
        "source": "GTEx Portal",
        "local_expected_path": "data/raw/gtex_v11_sample_level/GTEx_Analysis_v11_Annotations_SampleAttributesDS.txt",
        "used_for": "Sample-to-tissue metadata.",
        "tracked_in_git": "No",
        "reason_not_tracked": "External metadata file.",
    },
    {
        "dataset": "GTEx V11 subject phenotypes",
        "source": "GTEx Portal",
        "local_expected_path": "data/raw/gtex_v11_sample_level/GTEx_Analysis_v11_Annotations_SubjectPhenotypesDS.txt",
        "used_for": "Donor-level metadata and grouped cross-validation.",
        "tracked_in_git": "No",
        "reason_not_tracked": "External metadata file.",
    },
    {
        "dataset": "Tabula Sapiens h5ad",
        "source": "Tabula Sapiens",
        "local_expected_path": "data/raw/tabula_sapiens/TabulaSapiens.h5ad",
        "used_for": "Donor × organ × cell-type pseudobulk ECM source validation.",
        "tracked_in_git": "No",
        "reason_not_tracked": "Large external single-cell file.",
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
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )

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


def build_pipeline_table(table_dir: Path) -> pd.DataFrame:
    pipeline_df = pd.DataFrame(PIPELINE_ORDER)
    pipeline_df.to_csv(table_dir / "Supplementary_Table_S3_pipeline_execution_order.csv", index=False)
    print(f"[SAVED] {table_dir / 'Supplementary_Table_S3_pipeline_execution_order.csv'}")
    return pipeline_df


def build_data_manifest(table_dir: Path) -> pd.DataFrame:
    data_df = pd.DataFrame(DATA_MANIFEST)
    data_df.to_csv(table_dir / "Supplementary_Table_S4_data_manifest.csv", index=False)
    print(f"[SAVED] {table_dir / 'Supplementary_Table_S4_data_manifest.csv'}")
    return data_df


def build_key_output_manifest(table_dir: Path) -> pd.DataFrame:
    key_outputs = [
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
    ]

    out_df = pd.DataFrame(key_outputs)
    out_df.to_csv(table_dir / "Supplementary_Table_S5_key_output_manifest.csv", index=False)
    print(f"[SAVED] {table_dir / 'Supplementary_Table_S5_key_output_manifest.csv'}")
    return out_df


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

Recommended revision analysis:
- NMF ranks: k = 2 to 20
- Random seeds: 30
- Top genes per component: 30
- Metrics: reconstruction error, top-gene Jaccard stability, consensus cophenetic correlation, program recovery fraction

## GTEx donor-stratified classification

- Cross-validation: GroupKFold by subject_id
- Tissue-system task: 13 classes
- Tissue task: 27 classes after minimum sample filtering
- Features: 9 ECM program scores
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

    pipeline_df = build_pipeline_table(table_dir)
    data_df = build_data_manifest(table_dir)
    build_key_output_manifest(table_dir)

    git_info = get_git_info()
    env_info = get_environment_info()

    write_json(git_info, metadata_dir / "git_info.json")
    write_json(env_info, metadata_dir / "environment_info.json")

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