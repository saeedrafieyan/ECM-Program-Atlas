# Freeze note, v0.4

The ECM latent space project is frozen at v0.4.

## Frozen scope

v0.4 includes:
- Matrisome-derived transcriptomic ECM representation
- Matched random non-ECM baseline testing
- Matrisome category benchmarking
- Curated NMF-derived ECM programs
- External transcriptomic reproducibility
- MatrisomeDB protein-level validation
- GTEx V11 sample-level validation
- Donor-stratified tissue classification
- Classical ML, ensemble, deep learning, and tabular model benchmarks

## Frozen package

outputs/final_results_v0.4/

## Main result

Nine curated ECM program scores contain strong donor-generalizable tissue information across GTEx V11 samples.

## Best benchmark results

Tissue classification:
- Best model: Dense MLP
- Balanced accuracy: approximately 0.925
- Macro-F1: approximately 0.906

Tissue-system classification:
- Best model: TabICL
- Balanced accuracy: approximately 0.967
- Macro-F1: approximately 0.968

## Current limitation

This is still an ECM representation framework, not a scaffold inverse-design model or a foundation model.