# Reproducibility Checklist

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
