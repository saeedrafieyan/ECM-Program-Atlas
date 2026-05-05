# Stage 2: Interpretable ECM Module Analysis

## Objective

The goal of this stage was to transform baseline ECM latent embeddings into biologically interpretable ECM programs.

## Input

The analysis used the HPA tissue consensus dataset and four Matrisome feature sets:

1. core_matrisome
2. ecm_glycoproteins
3. proteoglycans
4. collagens

## Method

For each feature set, NMF was applied to the log2-transformed expression matrix. Each NMF module was interpreted using:

1. top-scoring tissues
2. top-weighted genes
3. Matrisome category context
4. manually curated biological interpretation

## Main finding

NMF decomposition revealed recurring ECM programs across Matrisome feature spaces.

## Major recurring ECM programs

1. Vascular/stromal/interstitial ECM
2. Epithelial/mucosal basement membrane ECM
3. CNS/neural ECM
4. Retinal/sensory ECM
5. Immune/lymphoid remodeling ECM
6. Hepatic/plasma-associated ECM
7. Reproductive-specialized ECM
8. Stromal remodeling ECM
9. Renal/endothelial basement membrane ECM

## Strongest programs

The strongest recurring programs were vascular/stromal/interstitial ECM and epithelial/mucosal basement membrane ECM. Both appeared across all four feature sets and all detected modules were high-confidence.

CNS/neural ECM was also highly robust, appearing across all four feature sets with high-confidence module assignments.

## Key output files

- outputs/latent_baseline_embeddings/rna_tissue_consensus/curated_recurring_ecm_programs/combined_nmf_module_annotations_curated_programs.csv
- outputs/latent_baseline_embeddings/rna_tissue_consensus/curated_recurring_ecm_programs/curated_recurring_ecm_program_summary.csv
- outputs/latent_baseline_embeddings/rna_tissue_consensus/curated_recurring_ecm_programs/curated_ecm_program_presence_matrix.csv
- outputs/latent_baseline_embeddings/rna_tissue_consensus/curated_recurring_ecm_programs/curated_ecm_program_high_confidence_matrix.csv

## Interpretation

This result shows that Matrisome-derived tissue expression is not only separable in low-dimensional space, but also decomposes into biologically interpretable ECM programs. These programs correspond to known tissue ECM biology, including stromal/interstitial ECM, epithelial basement membrane, CNS ECM, retinal ECM, immune/lymphoid remodeling ECM, renal/endothelial basement membrane ECM, and hepatic/plasma-associated ECM.

## Current conclusion

The project has produced a reproducible and interpretable ECM representation layer for human tissue-level Matrisome expression.