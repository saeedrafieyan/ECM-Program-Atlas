# Manuscript Draft v0.2

## Working title

A reproducible Matrisome-derived representation framework for mapping human tissue extracellular matrix signatures

---

## Abstract

The extracellular matrix (ECM) is a tissue-specific biological system that regulates cell behavior, tissue architecture, remodeling, and disease progression. However, tissue engineering and scaffold design still often rely on empirical biomaterial combinations rather than systematic native ECM representations. Here, we developed a reproducible Matrisome-derived ECM representation framework using Human Protein Atlas and GTEx-derived transcriptomic resources integrated with Human Matrisome annotations. Across seven processed datasets, up to 1009 of 1027 Human Matrisome genes were matched, enabling tissue-level, brain-region-level, and cell-type-level ECM representation analysis. Matrisome-filtered expression preserved biologically meaningful tissue organization, including central nervous system, immune/lymphoid, epithelial/mucosal, stromal, and vascular tissue neighborhoods. Matched random non-ECM gene-set controls showed that Matrisome-derived features provide enriched global tissue organization compared with random genes. Category-level benchmarking identified core Matrisome and ECM glycoproteins as particularly reproducible feature spaces for low-dimensional tissue organization. Non-negative matrix factorization identified recurring ECM programs, including vascular/stromal/interstitial ECM, epithelial/mucosal basement membrane ECM, CNS/neural ECM, retinal/sensory ECM, immune/lymphoid remodeling ECM, renal/endothelial basement membrane ECM, hepatic/plasma-associated ECM, reproductive-specialized ECM, and stromal remodeling ECM. These programs reproduced across external datasets after excluding the reference tissue-consensus dataset. Finally, MatrisomeDB proteomics validation showed that most RNA-derived ECM programs have protein-level support, with a mean protein detection coverage of approximately 0.72 across nine curated programs. These results establish a reproducible ECM representation and benchmarking framework that can support future ECM-informed biomaterial design, while highlighting the need for deeper proteomic, spatial, and mechanical validation before direct scaffold inverse design.

---

## 1. Introduction

The extracellular matrix is a dynamic, tissue-specific system composed of structural proteins, glycoproteins, proteoglycans, ECM-affiliated proteins, ECM regulators, and secreted factors. Far from acting as a passive scaffold, the ECM regulates cell adhesion, migration, proliferation, differentiation, mechanotransduction, tissue morphogenesis, and remodeling. Each tissue contains a distinct ECM profile that reflects its developmental origin, mechanical demands, cellular composition, vascularization, and functional state.

Despite this biological complexity, scaffold design in tissue engineering often remains empirical. Researchers commonly select a small number of biomaterials, such as collagen, gelatin, alginate, hyaluronic acid, polyethylene glycol, polycaprolactone, or other synthetic and natural polymers, then optimize scaffold properties experimentally. This approach has produced many useful systems, but it does not fully exploit the information encoded in native ECM composition across tissues. As a result, there remains a gap between native ECM biology and engineered biomaterial design.

Recent biological atlases and computational methods provide an opportunity to rethink this problem. If tissue ECM signatures can be represented computationally, then tissues can be compared quantitatively, ECM modules can be identified, and native ECM programs can eventually inform biomaterial selection or scaffold design. However, a direct foundation-model-style system for tissue engineering is not yet justified because current ECM-related datasets are relatively small, heterogeneous, and incomplete. Before building generative or inverse-design models, it is necessary to test whether ECM-specific features encode reproducible biological organization.

This study therefore focuses on a more realistic and necessary objective: constructing and validating a reproducible Matrisome-derived ECM representation framework. The central hypothesis is that Matrisome-filtered expression profiles contain structured, reproducible, and biologically meaningful tissue information. To test this, we integrated Human Protein Atlas and GTEx-derived transcriptomic resources with Human Matrisome annotations, benchmarked Matrisome features against matched random non-ECM gene sets, evaluated Matrisome categories, extracted interpretable ECM modules using non-negative matrix factorization, tested cross-dataset reproducibility, and added MatrisomeDB proteomics validation.

The resulting framework does not claim to be a foundation model or a direct scaffold-design system. Instead, it establishes a validated computational layer for mapping native ECM signatures. This layer can serve as a basis for future integration with ECM proteomics, spatial transcriptomics or proteomics, mechanical data, and scaffold or biomaterial response datasets.

---

## 2. Methods

### 2.1 Data sources

Transcriptomic data were obtained from Human Protein Atlas and GTEx-derived resources. The processed datasets included tissue-consensus expression, HPA tissue expression, GTEx tissue expression, GTEx detailed tissue expression, HPA brain-region expression, prefrontal cortex brain-region expression, and HPA single-cell-type expression. These datasets represented tissue-level, source-tissue-level, brain-region-level, and cell-type-level contexts.

Human Matrisome annotations were obtained from the Human Matrisome master list. The Matrisome annotations included 1027 human Matrisome genes divided into core Matrisome and Matrisome-associated categories. Core Matrisome included ECM glycoproteins, collagens, and proteoglycans. Matrisome-associated categories included ECM-affiliated proteins, ECM regulators, and secreted factors.

Across processed transcriptomic datasets, up to 1009 of 1027 Human Matrisome genes were matched, with 18 Matrisome genes missing. The final v0.1 results package included seven processed datasets, 30 latent embedding runs, nine curated ECM programs, and nine externally reproduced ECM programs. The v0.2 package extended this by adding MatrisomeDB proteomics validation, with 13 MatrisomeDB files copied and no missing files. :contentReference[oaicite:0]{index=0}

### 2.2 ECM expression matrix construction

For each transcriptomic dataset, gene symbols were standardized and matched against the Human Matrisome gene list. Expression matrices were constructed with samples as rows and Matrisome genes as columns. Depending on the dataset, samples represented tissues, source tissues, brain subregions, or cell types.

Expression values were transformed using:

```text
log2(expression + 1)