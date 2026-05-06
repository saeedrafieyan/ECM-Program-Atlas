# R3 MatrisomeDB Null and Abundance Validation

## Purpose

This analysis strengthens protein-level validation by comparing curated ECM programs against matched random Matrisome gene-set null models for detection coverage and NSAF-based abundance metrics.

## Detection coverage null model

- **Vascular/stromal/interstitial ECM**: observed = 0.818, random mean = 0.426, z = 5.29, empirical p = 0.0010.
- **Epithelial/mucosal basement membrane ECM**: observed = 0.768, random mean = 0.426, z = 5.40, empirical p = 0.0010.
- **CNS/neural ECM**: observed = 0.567, random mean = 0.422, z = 1.64, empirical p = 0.0829.
- **Retinal/sensory ECM**: observed = 0.723, random mean = 0.428, z = 4.13, empirical p = 0.0010.
- **Immune/lymphoid remodeling ECM**: observed = 0.780, random mean = 0.429, z = 4.65, empirical p = 0.0010.
- **Stromal remodeling ECM**: observed = 0.692, random mean = 0.428, z = 1.91, empirical p = 0.0659.
- **Renal/endothelial basement membrane ECM**: observed = 0.833, random mean = 0.429, z = 3.56, empirical p = 0.0010.
- **Hepatic/plasma-associated ECM**: observed = 0.750, random mean = 0.430, z = 3.74, empirical p = 0.0010.
- **Reproductive-specialized ECM**: observed = 0.522, random mean = 0.429, z = 0.91, empirical p = 0.2408.

## Abundance null model

- **Vascular/stromal/interstitial ECM**: observed top3 score = 4.456, random mean = 2.542, z = 5.57, empirical p = 0.0010.
- **Epithelial/mucosal basement membrane ECM**: observed top3 score = 4.159, random mean = 2.533, z = 5.16, empirical p = 0.0010.
- **CNS/neural ECM**: observed top3 score = 3.490, random mean = 2.613, z = 2.02, empirical p = 0.0280.
- **Retinal/sensory ECM**: observed top3 score = 3.818, random mean = 2.558, z = 3.55, empirical p = 0.0010.
- **Immune/lymphoid remodeling ECM**: observed top3 score = 4.021, random mean = 2.556, z = 4.22, empirical p = 0.0010.
- **Stromal remodeling ECM**: observed top3 score = 4.496, random mean = 2.754, z = 2.49, empirical p = 0.0190.
- **Renal/endothelial basement membrane ECM**: observed top3 score = 4.888, random mean = 2.714, z = 3.81, empirical p = 0.0020.
- **Hepatic/plasma-associated ECM**: observed top3 score = 4.256, random mean = 2.604, z = 4.09, empirical p = 0.0010.
- **Reproductive-specialized ECM**: observed top3 score = 3.917, random mean = 2.637, z = 2.55, empirical p = 0.0150.

## RNA-protein tissue correlation

- **CNS/neural ECM**, all_samples: Spearman r = -0.491, Pearson r = -0.275, n tissues = 11.
- **Epithelial/mucosal basement membrane ECM**, all_samples: Spearman r = -0.155, Pearson r = 0.068, n tissues = 11.
- **Hepatic/plasma-associated ECM**, all_samples: Spearman r = -0.155, Pearson r = -0.220, n tissues = 11.
- **Immune/lymphoid remodeling ECM**, all_samples: Spearman r = 0.145, Pearson r = 0.207, n tissues = 11.
- **Renal/endothelial basement membrane ECM**, all_samples: Spearman r = -0.145, Pearson r = -0.102, n tissues = 11.
- **Reproductive-specialized ECM**, all_samples: Spearman r = 0.318, Pearson r = 0.361, n tissues = 11.
- **Retinal/sensory ECM**, all_samples: Spearman r = -0.318, Pearson r = 0.087, n tissues = 11.
- **Stromal remodeling ECM**, all_samples: Spearman r = 0.209, Pearson r = 0.192, n tissues = 11.
- **Vascular/stromal/interstitial ECM**, all_samples: Spearman r = 0.245, Pearson r = 0.433, n tissues = 11.
- **CNS/neural ECM**, normal_like: Spearman r = -0.500, Pearson r = -0.245, n tissues = 5.
- **Epithelial/mucosal basement membrane ECM**, normal_like: Spearman r = 0.200, Pearson r = 0.315, n tissues = 5.
- **Hepatic/plasma-associated ECM**, normal_like: Spearman r = -0.200, Pearson r = -0.072, n tissues = 5.
- **Immune/lymphoid remodeling ECM**, normal_like: Spearman r = 0.200, Pearson r = 0.352, n tissues = 5.
- **Renal/endothelial basement membrane ECM**, normal_like: Spearman r = 0.100, Pearson r = 0.033, n tissues = 5.
- **Reproductive-specialized ECM**, normal_like: Spearman r = 0.600, Pearson r = 0.579, n tissues = 5.
- **Retinal/sensory ECM**, normal_like: Spearman r = -0.500, Pearson r = -0.197, n tissues = 5.
- **Stromal remodeling ECM**, normal_like: Spearman r = 0.400, Pearson r = 0.134, n tissues = 5.
- **Vascular/stromal/interstitial ECM**, normal_like: Spearman r = 0.400, Pearson r = 0.448, n tissues = 5.
- **CNS/neural ECM**, disease_like: Spearman r = 0.095, Pearson r = -0.004, n tissues = 8.
- **Epithelial/mucosal basement membrane ECM**, disease_like: Spearman r = 0.119, Pearson r = 0.501, n tissues = 8.
- **Hepatic/plasma-associated ECM**, disease_like: Spearman r = -0.548, Pearson r = -0.589, n tissues = 8.
- **Immune/lymphoid remodeling ECM**, disease_like: Spearman r = 0.238, Pearson r = 0.479, n tissues = 8.
- **Renal/endothelial basement membrane ECM**, disease_like: Spearman r = -0.143, Pearson r = 0.083, n tissues = 8.
- **Reproductive-specialized ECM**, disease_like: Spearman r = 0.310, Pearson r = 0.388, n tissues = 8.
- **Retinal/sensory ECM**, disease_like: Spearman r = 0.071, Pearson r = 0.266, n tissues = 8.
- **Stromal remodeling ECM**, disease_like: Spearman r = 0.310, Pearson r = 0.458, n tissues = 8.
- **Vascular/stromal/interstitial ECM**, disease_like: Spearman r = 0.548, Pearson r = 0.628, n tissues = 8.
- **CNS/neural ECM**, uncertain: Spearman r = -0.486, Pearson r = -0.419, n tissues = 6.
- **Epithelial/mucosal basement membrane ECM**, uncertain: Spearman r = -0.029, Pearson r = -0.267, n tissues = 6.
- **Hepatic/plasma-associated ECM**, uncertain: Spearman r = 0.200, Pearson r = 0.111, n tissues = 6.
- **Immune/lymphoid remodeling ECM**, uncertain: Spearman r = -0.029, Pearson r = -0.152, n tissues = 6.
- **Renal/endothelial basement membrane ECM**, uncertain: Spearman r = -0.200, Pearson r = -0.466, n tissues = 6.
- **Reproductive-specialized ECM**, uncertain: Spearman r = 0.086, Pearson r = 0.238, n tissues = 6.
- **Retinal/sensory ECM**, uncertain: Spearman r = -0.257, Pearson r = 0.166, n tissues = 6.
- **Stromal remodeling ECM**, uncertain: Spearman r = 0.143, Pearson r = -0.056, n tissues = 6.
- **Vascular/stromal/interstitial ECM**, uncertain: Spearman r = 0.657, Pearson r = 0.467, n tissues = 6.

## Interpretation guidance

Detection coverage alone is weak evidence. Stronger support is obtained when a curated ECM program exceeds matched random Matrisome gene sets for detection and abundance metrics, and when RNA-derived tissue profiles correlate with MatrisomeDB protein-level tissue profiles.

## Limitations

- MatrisomeDB contains mixed normal and disease-associated samples.

- NSAF is semi-quantitative and may not be directly comparable across all studies.

- Tissue overlap with GTEx may be limited.

- Study/repository-aware normalization should be added if enough metadata are available.
