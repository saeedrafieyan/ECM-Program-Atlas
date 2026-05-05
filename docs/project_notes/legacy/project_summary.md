# Project Context: ECM-Based Representation Learning for Tissue Engineering

## 1. High-level motivation

The long-term goal is to rethink scaffold and biomaterial design in tissue engineering. Current tissue engineering often relies on empirical combinations of a few biomaterials, such as collagen, gelatin, alginate, PEG, PLGA, or hyaluronic acid, to approximate the complexity of native extracellular matrix (ECM). This strategy is useful but limited because native ECM is not simply a mixture of materials. It is a complex, tissue-specific, dynamic, spatially organized system involving structural proteins, glycoproteins, proteoglycans, regulators, and signaling-associated molecules.

The original ambition was to develop something conceptually similar to a “foundation model” for tissue engineering or ECM-guided scaffold design. However, after evaluating the available data, the current realistic goal is more focused:

> Build a rigorous ECM representation and benchmarking framework that learns and validates biologically meaningful tissue, region, and cell-type representations from matrisome-derived expression data.

This project should not yet be claimed as a true foundation model. The available sample size is too small for that. Instead, the project should be framed as a computational biology and tissue engineering bridge: a validated ECM-specific representation layer that may later support biomaterial and scaffold design.

---

## 2. Core scientific question

The main question is:

> Do ECM-related genes, especially Matrisome categories, encode reproducible biological organization across human tissues, brain regions, and cell types?

More specifically:

1. Can tissues be represented using only ECM/matrisome genes?
2. Do ECM-only profiles preserve biologically meaningful tissue structure?
3. Are ECM genes more informative than matched random non-ECM gene sets?
4. Which Matrisome categories drive tissue organization?
5. Are these patterns reproducible across multiple datasets?
6. Can this representation later become the first layer of an ECM-informed biomaterial design framework?

---

## 3. Current positioning

The project should be positioned as:

> A reproducible matrisome-derived representation framework for mapping human tissue ECM signatures.

It should not currently be positioned as:

> A foundation model for tissue engineering.

The reason is simple: the current datasets contain tens to hundreds of aggregated samples, not millions of samples. This is enough for representation benchmarking, PCA, UMAP, clustering, tissue similarity analysis, random-gene controls, and cross-dataset validation. It is not enough for a large-scale deep generative model.

---

## 4. Data sources used so far

The project currently uses public human expression and ECM annotation resources.

### 4.1 Human Protein Atlas and GTEx-derived files

The following HPA-downloadable RNA expression datasets were downloaded and processed:

```text
rna_tissue_consensus.tsv.zip
rna_tissue_hpa.tsv.zip
rna_tissue_gtex.tsv.zip
rna_tissue_detail_gtex.tsv.zip
rna_brain_hpa.tsv.zip
rna_pfc_brain_hpa.tsv.zip
rna_single_cell_type.tsv.zip
```

These datasets provide expression values such as `nTPM`, `TPM`, `pTPM`, or `nCPM`.

### 4.2 Matrisome Project annotation

The human Matrisome master list was downloaded as an Excel file:

```text
human_matrisome.xlsx
```

It contains 1027 human Matrisome genes and the following key columns:

```text
Matrisome Division
Matrisome Category
Gene Symbol
Gene Name
Synonyms
HGNC_IDs
UniProt_IDs
Refseq_IDs
Notes
```

The relevant Matrisome divisions and categories are:

```text
Core matrisome:
    ECM glycoproteins
    Collagens
    Proteoglycans

Matrisome-associated:
    ECM-affiliated proteins
    ECM regulators
    Secreted factors
```

---

## 5. Processed datasets already created

The data preprocessing pipeline created ECM-specific expression matrices for each dataset.

Each matrix has:

```text
rows    = tissues, source tissues, brain subregions, or cell types
columns = matched Matrisome genes
values  = normalized expression, usually log2(expression + 1), then z-scored per gene
```

The main processed datasets are:

```text
rna_tissue_consensus:      51 samples × 1009 matched ECM genes
rna_tissue_hpa:            40 samples × 1009 matched ECM genes
rna_tissue_gtex:           36 samples × 1009 matched ECM genes
rna_tissue_detail_gtex:    49 samples × 1009 matched ECM genes
rna_brain_hpa:            193 samples × 1009 matched ECM genes
rna_pfc_brain_hpa:         20 samples × 1009 matched ECM genes
rna_single_cell_type:     154 samples × 1009 matched ECM genes
```

After removing zero-variance genes in some validation scripts, the usable gene count becomes slightly smaller, for example around 1002 genes in the tissue consensus dataset.

The match quality is very strong:

```text
1009 / 1027 Matrisome genes matched in HPA expression data.
Only 18 Matrisome genes were missing.
```

This confirms that the data construction pipeline is technically sound.

---

## 6. Completed analyses and findings

### 6.1 Exploratory ECM tissue representation

The first analysis used:

```text
data/processed/rna_tissue_consensus/ecm_expression_log2_zscore.csv
```

Methods used:

```text
PCA
UMAP
Pearson correlation heatmap
Cosine similarity nearest neighbors
Agglomerative clustering
```

Key findings:

1. CNS tissues clustered together using only Matrisome gene expression.
2. Immune and lymphoid tissues formed a biologically coherent group.
3. Digestive, epithelial, reproductive, and stromal/connective tissues showed partial but meaningful organization.
4. PCA showed that the first two components captured around 30 percent of variance in the tissue consensus ECM matrix.
5. UMAP showed clearer nonlinear tissue organization than PCA.

Interpretation:

> ECM-only expression profiles preserve meaningful tissue-level biological organization.

This was the first evidence that ECM genes are not simply random background genes.

---

### 6.2 ECM genes versus matched random non-ECM genes

A specificity test compared:

```text
A. Matrisome ECM genes
B. Random non-ECM gene sets of the same size
C. Full transcriptome reference
```

The random non-ECM baseline was repeated 1000 times.

Metrics included:

```text
PCA PC1 variance
PCA PC1 + PC2 variance
PCA PC1 to PC5 variance
Silhouette score in original space
Silhouette score in PCA10 space
Nearest-neighbor same tissue-system score at k=3 and k=5
Adjusted Rand Index, ARI
Normalized Mutual Information, NMI
```

Main result:

> Matrisome genes outperformed matched random non-ECM genes for several global tissue-organization metrics, especially PCA variance and silhouette scores.

However, the result was not uniformly positive for all metrics. Local-neighborhood and clustering metrics were mixed.

Accurate interpretation:

> Matrisome genes contain enriched global tissue-organization signal compared with random non-ECM genes, but they do not universally outperform random genes for every metric.

This is scientifically stronger and more honest than claiming that all ECM genes are always superior.

---

### 6.3 Matrisome category analysis

The next analysis tested individual Matrisome feature sets:

```text
all_matrisome
division__core_matrisome
division__matrisome_associated
category__ecm_glycoproteins
category__collagens
category__proteoglycans
category__ecm_affiliated_proteins
category__ecm_regulators
category__secreted_factors
```

Each category was compared against random non-ECM gene sets of the same size.

Gene counts were approximately:

```text
All matrisome:             1002 genes
Matrisome-associated:       733 genes
Secreted factors:           328 genes
Core matrisome:             269 genes
ECM regulators:             238 genes
ECM glycoproteins:          190 genes
ECM-affiliated proteins:    167 genes
Collagens:                   44 genes
Proteoglycans:               35 genes
```

Main findings:

1. **Core matrisome** generated strong global low-dimensional tissue structure.
2. **ECM glycoproteins** produced strong and reproducible PCA-based separation.
3. **Collagens** generated strong dominant structural/stromal axes.
4. **Proteoglycans** were biologically interesting and showed strong signals in some datasets.
5. **ECM-affiliated proteins** performed well for local tissue-system neighborhood structure in the tissue consensus analysis.
6. **Secreted factors** and **ECM regulators** sometimes aligned well with unsupervised clustering, but they may reflect broader signaling or cell-type composition rather than structural ECM.

Accurate interpretation:

> Different Matrisome categories encode different aspects of tissue organization. Structural ECM categories drive global ECM geometry, while matrisome-associated categories may capture remodeling, signaling, immune, or cell-composition-related variation.

---

### 6.4 Cross-dataset reproducibility analysis

The most important validation tested whether the category-level patterns reproduced across multiple datasets.

Datasets analyzed:

```text
rna_tissue_consensus
rna_tissue_hpa
rna_tissue_gtex
rna_tissue_detail_gtex
rna_brain_hpa
rna_single_cell_type
```

One dataset was skipped:

```text
rna_pfc_brain_hpa
```

Reason:

```text
The current label mapper collapsed the PFC brain data into only one label class, so supervised label-based metrics were not meaningful.
```

The processed cross-dataset summary was approximately:

```text
rna_tissue_consensus:      51 original samples, 50 labeled samples, 7 label classes
rna_tissue_hpa:            40 original samples, 39 labeled samples, 7 label classes
rna_tissue_gtex:           36 original samples, 34 labeled samples, 6 label classes
rna_tissue_detail_gtex:    49 original samples, 44 labeled samples, 6 label classes
rna_brain_hpa:            193 original samples, 77 labeled samples, 8 label classes
rna_single_cell_type:     154 original samples, 67 labeled samples, 9 label classes
```

Main reproducibility finding:

> Core matrisome and ECM glycoproteins were the most reproducible Matrisome feature sets across datasets.

The most reproducible categories were:

```text
division__core_matrisome
category__ecm_glycoproteins
all_matrisome
category__collagens
category__proteoglycans
```

The strongest reproducible metrics were:

```text
pca_pc1_pc2_variance
silhouette_pca10_space
```

Weaker and less consistent metrics were:

```text
nearest_neighbor_same_label_at_3
ARI
NMI
```

Accurate interpretation:

> ECM categories reproducibly improve global and medium-scale biological organization, but they do not always improve local-neighborhood or clustering-label recovery.

This suggests that ECM expression is most useful as a **global representation manifold**, not necessarily as a perfect clustering system.

---

## 7. Current scientific conclusion

The current evidence supports the following claim:

> Across multiple transcriptomic datasets, structural Matrisome categories, particularly core matrisome and ECM glycoproteins, reproducibly produce stronger low-dimensional biological organization than matched random non-ECM gene sets.

A more detailed claim:

> Matrisome-derived expression profiles preserve tissue, region, and cell-type organization in a category-dependent manner. Core matrisome and ECM glycoproteins provide the most reproducible global ECM representation, while collagens and proteoglycans capture more specialized structural axes. Matrisome-associated categories may encode remodeling, signaling, immune, or cell-composition-related variation.

Claims to avoid:

```text
We built a foundation model.
We can design scaffolds directly from this model.
ECM genes always outperform random genes.
This is sufficient for biomaterial inverse design.
```

These claims would be premature.

---

## 8. Why the project still matters despite limited data

The available data is not enough for a foundation model, but it is enough for a meaningful benchmarking and representation paper.

The literature already has:

```text
Matrisome gene/protein annotations
ECM proteomics databases
Human tissue transcriptomic atlases
Single-cell atlases
ML models for ECM protein classification
ML models for biomaterial/scaffold optimization
```

But the missing layer is:

> A validated ECM-specific representation framework that benchmarks which Matrisome categories reproducibly encode biological organization across tissues, regions, and cell types.

This project fills that gap.

The value is not only in training a model. The value is in creating a reliable ECM representation layer that could later support:

```text
ECM tissue similarity analysis
ECM category-specific embeddings
native ECM-informed biomaterial selection
scaffold design constraints
cell-type-specific ECM interpretation
proteomics validation
spatial ECM modeling
```

---

## 9. Practical outputs already produced

The project has already generated:

```text
1. Clean Matrisome-filtered expression matrices.
2. Matched ECM gene metadata.
3. Missing Matrisome gene lists.
4. PCA, UMAP, and clustering plots.
5. Tissue cosine similarity matrices.
6. Nearest-neighbor tissue maps.
7. Random non-ECM baseline comparisons.
8. Category-level Matrisome benchmark results.
9. Cross-dataset reproducibility metrics.
10. FDR-corrected metric summaries.
```

Important output folders include:

```text
data/processed/
outputs/eda/
outputs/specificity/
outputs/category_analysis/
outputs/cross_dataset_reproducibility/
```

---

## 10. Main limitations

The major limitations are:

### 10.1 Low sample size

Most datasets are aggregated tissue-level or category-level summaries:

```text
51 tissue consensus samples
40 HPA tissue samples
36 GTEx tissue samples
49 GTEx detailed tissue samples
193 brain subregions
154 single-cell type averages
```

This is not enough for a large deep generative model.

### 10.2 Bulk expression is not ECM composition

Transcriptomic ECM gene expression does not directly equal deposited ECM protein abundance. ECM proteins are secreted, crosslinked, degraded, remodeled, and spatially organized.

### 10.3 Lack of spatial structure

Current representations do not capture:

```text
fiber orientation
matrix topology
pore structure
spatial gradients
mechanical properties
viscoelasticity
regional ECM architecture
```

### 10.4 Tissue labels are coarse

Manual tissue-system labels are useful for validation, but they may not reflect true ECM similarity. Some ECM categories may encode biologically meaningful axes that do not align with the manually defined labels.

### 10.5 Current data does not connect directly to biomaterial design

The current representation describes native ECM signatures. It does not yet map those signatures to synthetic biomaterials, scaffold formulations, fabrication parameters, or cell-response outcomes.

---

## 11. What should be added next

To strengthen the project, the following datasets or data layers should be added.

### 11.1 GTEx sample-level expression

This is the highest-priority next dataset.

Purpose:

```text
Increase sample count from tissue averages to donor-level tissue samples.
Capture inter-individual variability.
Enable more reliable modeling and validation.
```

Expected structure:

```text
rows    = individual GTEx tissue samples
columns = Matrisome genes
labels  = tissue, donor, sex, age group, tissue site
```

### 11.2 Tabula Sapiens pseudobulk profiles

Purpose:

```text
Create organ × cell-type ECM profiles.
Identify which cell types contribute to tissue ECM signatures.
```

Example pseudobulk samples:

```text
lung_fibroblast
lung_endothelial
heart_fibroblast
skin_keratinocyte
liver_endothelial
colon_epithelial
```

This would add a mechanistic single-cell layer.

### 11.3 MatrisomeDB proteomics

Purpose:

```text
Validate whether transcriptomic ECM embeddings agree with protein-level ECM composition.
```

This is critical because ECM is ultimately a protein-level system.

### 11.4 Spatial transcriptomics or spatial proteomics

Purpose:

```text
Move from whole-tissue ECM profiles to tissue-zone-specific ECM representations.
```

This would make the project much more relevant to scaffold design because engineered scaffolds should often mimic specific tissue zones, not whole organs.

### 11.5 Scaffold and biomaterial datasets

Purpose:

```text
Eventually connect native ECM signatures to engineered biomaterial/scaffold design.
```

This could use the user’s existing MLATE/scaffold dataset, including:

```text
biomaterial composition
cell type
cell density
fabrication method
printing parameters
printability
cell response
scaffold quality score
```

This would allow future joint embedding:

```text
native ECM embedding ↔ scaffold/material embedding ↔ biological response
```

---

## 12. Recommended next technical step

The next modeling step should not be a VAE yet.

The recommended next step is:

> Build baseline ECM latent embeddings using PCA, NMF, and UMAP for selected feature sets, then evaluate reproducibility and interpretability.

Feature sets to compare:

```text
all_matrisome
core_matrisome
ecm_glycoproteins
proteoglycans
collagens
```

For each feature set, generate:

```text
PCA coordinates
NMF coordinates
UMAP coordinates
explained variance
top loading genes
nearest-neighbor tissue maps
category-specific tissue similarity matrices
cross-dataset reproducibility scores
```

Only after this should a VAE or contrastive model be considered.

Reason:

```text
The current sample size is small.
Classical representation methods are more defensible.
A neural model may overfit and produce misleading embeddings.
```

---

## 13. Potential first-paper framing

A realistic first paper could be framed as:

```text
Title idea:
A reproducible matrisome-derived representation framework for mapping human tissue ECM signatures
```

Possible aims:

```text
Aim 1:
Construct Matrisome-filtered ECM expression matrices from human tissue, brain-region, and cell-type transcriptomic resources.

Aim 2:
Benchmark ECM and Matrisome-category feature spaces against matched random non-ECM gene sets.

Aim 3:
Test cross-dataset reproducibility of ECM-derived tissue organization.

Aim 4:
Release ECM embeddings, tissue similarity maps, and category-specific benchmark results as a reusable resource.
```

Main contribution:

> This study establishes a reproducible computational framework for evaluating how native ECM gene programs organize human tissues, regions, and cell types.

Potential long-term impact:

> The framework provides the first layer toward ECM-informed biomaterial and scaffold design, where engineered materials may eventually be compared against native tissue ECM signatures.

---

## 14. Final concise summary

This project aims to build a rigorous ECM-specific representation framework, not yet a foundation model. Using Human Protein Atlas, GTEx-derived expression resources, and Matrisome Project annotations, we created clean ECM expression matrices across tissues, brain subregions, and cell types. We showed that ECM-only expression preserves biological tissue organization. Matrisome genes outperform matched random non-ECM genes for several global tissue-organization metrics. Category-level analysis showed that core matrisome and ECM glycoproteins are the most reproducible sources of low-dimensional ECM structure across datasets, while collagens and proteoglycans capture specialized structural axes.

The current data is not enough for a deep foundation model or direct scaffold inverse design. However, it is enough to support a meaningful computational biology contribution: a benchmarked and reproducible ECM representation framework. The next major improvement should be adding GTEx sample-level data, Tabula Sapiens pseudobulk profiles, and MatrisomeDB proteomics validation.
