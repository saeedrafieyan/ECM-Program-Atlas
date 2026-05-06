# R2 Representation Control Benchmark

## Purpose

This analysis tests whether the nine curated ECM programs provide stronger or more interpretable donor-stratified tissue classification than alternative compact representations.

## Representations compared

- Curated 9 ECM programs

- 9 PCA components from the full Matrisome expression matrix

- Random Matrisome gene-set programs matched to curated program sizes

- Random non-ECM gene-set programs matched to curated program sizes, if full GTEx expression is supplied


## Models compared

- dense_mlp
- extra_trees
- ft_transformer_lite
- logistic_regression
- residual_mlp
- tabicl
- tabnet
- xgboost

## Best results by task and model

- **tissue, dense_mlp**: best representation = matrisome_pca9 (matrisome_pca), balanced accuracy = 0.958 ± 0.004, macro-F1 = 0.947.
- **tissue, extra_trees**: best representation = matrisome_pca9 (matrisome_pca), balanced accuracy = 0.935 ± 0.005, macro-F1 = 0.938.
- **tissue, ft_transformer_lite**: best representation = matrisome_pca9 (matrisome_pca), balanced accuracy = 0.944 ± 0.008, macro-F1 = 0.932.
- **tissue, logistic_regression**: best representation = matrisome_pca9 (matrisome_pca), balanced accuracy = 0.933 ± 0.005, macro-F1 = 0.916.
- **tissue, residual_mlp**: best representation = matrisome_pca9 (matrisome_pca), balanced accuracy = 0.955 ± 0.005, macro-F1 = 0.941.
- **tissue, tabicl**: best representation = matrisome_pca9 (matrisome_pca), balanced accuracy = 0.952 ± 0.005, macro-F1 = 0.956.
- **tissue, tabnet**: best representation = matrisome_pca9 (matrisome_pca), balanced accuracy = 0.950 ± 0.006, macro-F1 = 0.941.
- **tissue, xgboost**: best representation = matrisome_pca9 (matrisome_pca), balanced accuracy = 0.938 ± 0.005, macro-F1 = 0.936.
- **tissue_system, dense_mlp**: best representation = matrisome_pca9 (matrisome_pca), balanced accuracy = 0.977 ± 0.002, macro-F1 = 0.964.
- **tissue_system, extra_trees**: best representation = matrisome_pca9 (matrisome_pca), balanced accuracy = 0.956 ± 0.007, macro-F1 = 0.964.
- **tissue_system, ft_transformer_lite**: best representation = matrisome_pca9 (matrisome_pca), balanced accuracy = 0.972 ± 0.004, macro-F1 = 0.950.
- **tissue_system, logistic_regression**: best representation = matrisome_pca9 (matrisome_pca), balanced accuracy = 0.913 ± 0.005, macro-F1 = 0.883.
- **tissue_system, residual_mlp**: best representation = matrisome_pca9 (matrisome_pca), balanced accuracy = 0.978 ± 0.004, macro-F1 = 0.965.
- **tissue_system, tabicl**: best representation = matrisome_pca9 (matrisome_pca), balanced accuracy = 0.975 ± 0.004, macro-F1 = 0.978.
- **tissue_system, tabnet**: best representation = matrisome_pca9 (matrisome_pca), balanced accuracy = 0.974 ± 0.003, macro-F1 = 0.967.
- **tissue_system, xgboost**: best representation = matrisome_pca9 (matrisome_pca), balanced accuracy = 0.962 ± 0.006, macro-F1 = 0.961.

## Interpretation guidance

If curated ECM programs outperform random Matrisome and random non-ECM representations, this supports their biological specificity. If PCA performs similarly or better, the nine curated programs should be framed primarily as an interpretable representation rather than the most predictive representation.