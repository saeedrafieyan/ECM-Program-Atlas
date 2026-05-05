# NMF Rank and Stability Analysis

## Purpose

This analysis evaluates whether NMF-derived ECM modules are stable across ranks and random seeds, and whether curated ECM programs can be recovered across a range of NMF ranks.

## Metrics

- **Normalized reconstruction error:** lower values indicate better matrix reconstruction.

- **Top-gene Jaccard stability:** matched similarity of top component genes across random seeds.

- **Consensus cophenetic correlation:** stability of sample co-clustering across seeds.

- **Program recovery fraction:** fraction of curated ECM programs recovered by NMF components at a given rank.


## Recommended ranks by composite stability score

- **all_matrisome**: recommended k = 20; score = 0.913; recovery fraction = 0.967; top-gene stability = 0.804.
- **collagens**: recommended k = 20; score = 0.936; recovery fraction = 0.778; top-gene stability = 0.973.
- **core_matrisome**: recommended k = 18; score = 0.906; recovery fraction = 1.000; top-gene stability = 0.784.
- **ecm_glycoproteins**: recommended k = 18; score = 0.942; recovery fraction = 1.000; top-gene stability = 0.889.
- **proteoglycans**: recommended k = 20; score = 0.944; recovery fraction = 0.778; top-gene stability = 1.000.

## Interpretation guidance

The curated ECM programs should not be described as arising from one arbitrary NMF run. They should be described as recurring biological programs consolidated from NMF modules across Matrisome feature spaces and evaluated through cross-rank recovery, external reproducibility, MatrisomeDB protein support, GTEx donor-level prediction, Tabula Sapiens cell-type source mapping, and rank-based scoring robustness.

## Manuscript wording recommendation

Use: 'NMF modules were evaluated across ranks and random seeds, and recurrent modules were consolidated into curated ECM programs based on gene composition, tissue enrichment, and cross-dataset recovery.'