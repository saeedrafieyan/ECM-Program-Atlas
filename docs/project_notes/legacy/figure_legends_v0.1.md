# Figure Legends v0.1

## Figure 1. Workflow of the Matrisome-derived ECM representation framework

Schematic overview of the computational workflow. Human transcriptomic resources from HPA and GTEx-derived datasets were integrated with Human Matrisome annotations to construct ECM-focused expression matrices. These matrices were used for exploratory tissue organization analysis, matched random non-ECM baseline testing, Matrisome category benchmarking, baseline latent embedding generation, NMF module curation, and external reproducibility analysis.

## Figure 2. ECM-only expression preserves tissue organization

PCA and UMAP projections of tissue samples represented only by Matrisome-filtered ECM gene expression. The PCA panel shows broad linear organization of tissue ECM profiles, while the UMAP panel shows nonlinear tissue neighborhoods. The separation of CNS-related tissues and other tissue groups indicates that ECM-only expression preserves biologically meaningful tissue organization.

## Figure 3. Matrisome genes compared with matched random non-ECM genes

Comparison of Matrisome genes against matched random non-ECM gene sets. The left panel shows observed Matrisome metric values relative to the random non-ECM mean ± standard deviation. The right panel shows z-scores of Matrisome values relative to the random baseline. Matrisome genes outperform random non-ECM genes for global tissue-organization metrics, especially PCA variance and silhouette scores, while nearest-neighbor and clustering agreement metrics are more mixed.

## Figure 4. Matrisome category benchmark against matched random genes

Heatmap of z-scores comparing Matrisome feature categories against matched random non-ECM gene sets. Core Matrisome and ECM glycoproteins show strong enrichment for low-dimensional tissue organization, especially PCA PC1+PC2 variance. All Matrisome and ECM-affiliated proteins show strong silhouette-based tissue-system separation. These results indicate that different Matrisome categories encode distinct aspects of tissue organization.

## Figure 5. Curated recurring ECM programs across Matrisome feature sets

Heatmaps summarizing curated NMF-derived ECM programs across core Matrisome, ECM glycoproteins, proteoglycans, and collagens. Figure 5A shows the number of modules assigned to each curated ECM program. Figure 5B shows only high-confidence module assignments. Vascular/stromal/interstitial ECM, epithelial/mucosal basement membrane ECM, and CNS/neural ECM are among the strongest recurring programs.

## Figure 6. External reproducibility of ECM programs

External-only reproducibility analysis of curated ECM programs after excluding the reference tissue-consensus dataset. Figure 6A shows reproducibility across external datasets, including HPA tissue, GTEx tissue, GTEx detailed tissue, HPA brain, and single-cell-type data. Figure 6B shows reproducibility across Matrisome feature sets. Vascular/stromal/interstitial ECM, epithelial/mucosal basement membrane ECM, CNS/neural ECM, and hepatic/plasma-associated ECM are the most reproducible programs.