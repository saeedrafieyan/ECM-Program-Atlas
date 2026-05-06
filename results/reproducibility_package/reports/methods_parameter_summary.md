# Methods and Parameter Summary

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
