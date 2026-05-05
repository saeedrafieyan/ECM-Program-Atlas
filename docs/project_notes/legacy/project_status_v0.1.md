# ECM Latent Space Project, Status v0.1

## Current project goal

The goal is to build a reproducible ECM-specific representation framework using Matrisome-filtered transcriptomic data from HPA and GTEx-derived resources.

This project is not currently a foundation model. It is a validated ECM representation and benchmarking framework.

## Completed steps

1. Downloaded HPA/GTEx-derived RNA expression datasets.
2. Downloaded Human Matrisome Project annotations.
3. Built Matrisome-filtered ECM expression matrices.
4. Matched 1009 out of 1027 Human Matrisome genes.
5. Ran exploratory PCA, UMAP, correlation, and clustering.
6. Compared Matrisome genes against matched random non-ECM genes.
7. Ran Matrisome category-level analysis.
8. Ran cross-dataset reproducibility analysis.

## Main finding so far

Structural Matrisome categories, especially core matrisome and ECM glycoproteins, reproducibly encode low-dimensional biological organization across human tissues, brain regions, and cell types.

## Strongest feature sets

1. all_matrisome
2. core_matrisome
3. ecm_glycoproteins
4. collagens
5. proteoglycans

## Next step

Build baseline ECM latent embeddings using PCA, NMF, and UMAP across selected datasets and selected Matrisome feature sets.

## Later validation

Use MatrisomeDB as a protein-level validation dataset after transcriptomic ECM embeddings are built.