# Code and Data Availability Statement

All analysis code is available at:

[GitHub repository URL to be added]

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
