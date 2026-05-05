# ECM Program Atlas

A reproducible Matrisome-derived ECM representation framework for mapping human tissue extracellular matrix programs.

## Scope

This repository contains code and documentation for:

1. Matrisome-filtered expression matrix construction
2. ECM-specificity controls against matched random non-ECM genes
3. Matrisome category benchmarking
4. NMF-derived ECM program discovery and curation
5. External transcriptomic reproducibility
6. MatrisomeDB protein-level validation
7. GTEx V11 donor-level validation
8. Classical, ensemble, deep, and tabular classifier benchmarking
9. Tabula Sapiens cell-type source validation
10. Rank-based ECM program scoring robustness

## Important statement

This project is not a scaffold inverse-design model and not a foundation model. It is a validated ECM representation framework intended to support future ECM-informed biomaterial design.

## Data

Large datasets are not tracked in Git. Download instructions are provided in `scripts/download_instructions.md`.

## Reproducibility

Run the pipeline scripts in the `pipelines/` folder in numerical order.

## Repository status

This repository is the cleaned and reproducible version of the ECM Program Atlas project. The original exploratory workspace is not included. Large raw and processed datasets are excluded from Git and must be downloaded separately using the instructions in `scripts/download_instructions.md`.

Current migrated components include:
- validated pipeline scripts
- curated ECM program annotations
- frozen summary tables
- frozen analysis reports
- manuscript planning documents
