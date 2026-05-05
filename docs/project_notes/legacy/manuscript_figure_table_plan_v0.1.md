# Manuscript Figure and Table Plan v0.1

## Project framing

Working title: **A reproducible Matrisome-derived representation framework for mapping human tissue ECM signatures**

This manuscript should be framed as a reproducible ECM representation and benchmarking framework, not as a foundation model or direct scaffold inverse-design system.

## Proposed main figures

### Figure 1: Workflow of the Matrisome-derived ECM representation framework

**Purpose:** Show the full computational workflow: HPA/GTEx-derived expression data, Human Matrisome filtering, ECM matrices, random non-ECM controls, category benchmarking, latent embeddings, NMF module curation, and external reproducibility.

**Source files:** `Needs to be created manually or scripted as a schematic.`

**Status:** needs_schematic

**Source check:** not_applicable

**Notes:** This should be a schematic, not a data plot. Can be made in BioRender, PowerPoint, Illustrator, or generated with Python/Graphviz.

### Figure 2: ECM-only tissue organization

**Purpose:** Show that Matrisome-filtered expression preserves biological tissue organization, including CNS, immune/lymphoid, epithelial/mucosal, and stromal/connective neighborhoods.

**Source files:** `figures/eda_tissue_consensus_pca.png; figures/eda_tissue_consensus_umap.png; figures/eda_tissue_consensus_heatmap.png`

**Status:** available

**Source check:** exists

**Notes:** Use PCA and UMAP as main panels. Heatmap can be supplementary if too dense.

### Figure 3: Matrisome genes versus matched random non-ECM genes

**Purpose:** Show that Matrisome genes outperform matched random non-ECM gene sets for global tissue-organization metrics.

**Source files:** `tables/Table_2_ecm_vs_random_specificity.csv`

**Status:** needs_plot_generation

**Source check:** exists

**Notes:** Generate bar plots or point-range plots for ECM value vs random mean and z-score.

### Figure 4: Matrisome category benchmark

**Purpose:** Compare all Matrisome, core Matrisome, ECM glycoproteins, collagens, proteoglycans, and other Matrisome categories against matched random baselines.

**Source files:** `tables/Table_3_matrisome_category_benchmark.csv`

**Status:** needs_plot_generation

**Source check:** exists

**Notes:** Focus on PCA PC1+PC2 variance and silhouette PCA10. Consider a heatmap of z-scores.

### Figure 5: Curated NMF-derived ECM programs

**Purpose:** Show the nine curated recurring ECM programs derived from NMF modules across core Matrisome, ECM glycoproteins, proteoglycans, and collagens.

**Source files:** `tables/Table_4_curated_recurring_ecm_programs.csv; figures/curated_program_presence_heatmap.html; figures/curated_program_high_confidence_heatmap.html`

**Status:** available_html_needs_static_export

**Source check:** exists

**Notes:** Use curated program presence matrix as the main figure. Export Plotly HTML to PNG/SVG later.

### Figure 6: External reproducibility of ECM programs

**Purpose:** Show that reference-defined ECM programs reproduce across external datasets after excluding rna_tissue_consensus.

**Source files:** `tables/Table_5_external_program_reproducibility.csv; figures/external_dataset_program_presence_heatmap.html; figures/external_feature_set_program_presence_heatmap.html`

**Status:** available_html_needs_static_export

**Source check:** exists

**Notes:** This is one of the strongest figures. It supports non-circular reproducibility.

## Proposed tables

### Table 1: Processed datasets and Matrisome gene matching

**Purpose:** Summarize datasets, sample counts, labels, and Matrisome gene matching.

**Source file:** `tables/Table_1_dataset_summary.csv`

**Placement:** main

**Status:** available

**Source check:** exists

### Table 2: Matrisome versus matched random non-ECM specificity metrics

**Purpose:** Show ECM-specificity test metrics, including PCA variance and silhouette scores.

**Source file:** `tables/Table_2_ecm_vs_random_specificity.csv`

**Placement:** main_or_supplementary

**Status:** available

**Source check:** exists

### Table 3: Matrisome category benchmark

**Purpose:** Compare Matrisome categories against matched random baselines.

**Source file:** `tables/Table_3_matrisome_category_benchmark.csv`

**Placement:** main_or_supplementary

**Status:** available

**Source check:** exists

### Table 4: Curated recurring ECM programs

**Purpose:** Summarize curated ECM programs, number of modules, feature sets, and representative genes.

**Source file:** `tables/Table_4_curated_recurring_ecm_programs.csv`

**Placement:** main

**Status:** available

**Source check:** exists

### Table 5: External reproducibility of ECM programs

**Purpose:** Show which ECM programs reproduce after excluding the reference tissue-consensus dataset.

**Source file:** `tables/Table_5_external_program_reproducibility.csv`

**Placement:** main

**Status:** available

**Source check:** exists

### Supplementary Table 1: Curated NMF module annotations

**Purpose:** Provide all curated NMF modules, top samples, top genes, confidence, and interpretation.

**Source file:** `tables/Supplementary_Table_curated_nmf_modules.csv`

**Placement:** supplementary

**Status:** available

**Source check:** exists

## Immediate next actions

1. Generate static manuscript-ready plots for Figures 3 and 4 from CSV tables.

2. Export Plotly HTML heatmaps for Figures 5 and 6 to PNG or SVG.

3. Create Figure 1 as a workflow schematic.

4. Decide which large tables should stay in the main manuscript and which should move to supplementary material.

5. After figure planning is stable, move to MatrisomeDB proteomics validation.
