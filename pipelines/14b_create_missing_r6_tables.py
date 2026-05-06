from __future__ import annotations

from pathlib import Path
import pandas as pd


TABLE_DIR = Path("results/reproducibility_package/tables")


METHOD_PARAMETERS = [
    {
        "analysis": "Matrisome filtering",
        "parameter": "gene matching",
        "value": "HGNC gene symbol, upper-case normalized",
        "reason": "Consistent gene matching across HPA, GTEx, MatrisomeDB, and Tabula Sapiens.",
    },
    {
        "analysis": "Expression transformation",
        "parameter": "bulk/tissue expression transform",
        "value": "log2(expression + 1)",
        "reason": "Reduces skew and keeps zero values defined.",
    },
    {
        "analysis": "Z-score scoring",
        "parameter": "gene scaling",
        "value": "gene-wise z-score across samples",
        "reason": "Normalizes gene-level variation before averaging program genes.",
    },
    {
        "analysis": "NMF stability",
        "parameter": "rank range",
        "value": "k = 2 to 20",
        "reason": "Evaluates factorization granularity and stability across ranks.",
    },
    {
        "analysis": "NMF stability",
        "parameter": "random seeds",
        "value": "30",
        "reason": "Evaluates component stability across initialization.",
    },
    {
        "analysis": "NMF stability",
        "parameter": "top genes per component",
        "value": "30",
        "reason": "Used for component comparison and program recovery.",
    },
    {
        "analysis": "NMF stability",
        "parameter": "recovery threshold",
        "value": ">=4 overlapping genes and overlap coefficient >=0.13",
        "reason": "Defines whether a curated program is recovered by an NMF component.",
    },
    {
        "analysis": "MatrisomeDB null validation",
        "parameter": "random repeats",
        "value": "1000",
        "reason": "Empirical null distribution for detection and abundance support.",
    },
    {
        "analysis": "GTEx V11 classification",
        "parameter": "cross-validation",
        "value": "GroupKFold by subject_id",
        "reason": "Prevents donor leakage across train/test folds.",
    },
    {
        "analysis": "GTEx V11 classification",
        "parameter": "minimum samples per class",
        "value": "100",
        "reason": "Removes very small classes from donor-stratified benchmark.",
    },
    {
        "analysis": "Tabula Sapiens pseudobulk",
        "parameter": "pseudobulk unit",
        "value": "donor × organ × cell type × method",
        "reason": "Avoids treating cells from the same donor as independent biological replicates.",
    },
    {
        "analysis": "Tabula Sapiens pseudobulk",
        "parameter": "minimum cells per pseudobulk group",
        "value": "20",
        "reason": "Filters unstable low-cell-count pseudobulk groups.",
    },
    {
        "analysis": "Rank-based scoring",
        "parameter": "rank_percentile_score",
        "value": "mean within-sample percentile rank of genes in an ECM program",
        "reason": "Tests robustness to expression scale and normalization.",
    },
    {
        "analysis": "Rank-based scoring",
        "parameter": "top10_fraction_score",
        "value": "fraction of program genes in top 10% of Matrisome genes",
        "reason": "Alternative rank-based enrichment metric.",
    },
    {
        "analysis": "Spatial validation",
        "parameter": "datasets",
        "value": "10x Visium breast cancer CytAssist FFPE and healthy human lymph node",
        "reason": "Focused spatial transcriptomic validation in diseased and healthy tissue contexts.",
    },
]


MODEL_PARAMETERS = [
    {
        "model": "Logistic regression",
        "parameters": "StandardScaler; class_weight=balanced; solver=lbfgs; max_iter=5000",
        "role": "Interpretable linear baseline.",
    },
    {
        "model": "XGBoost",
        "parameters": "n_estimators=500; max_depth=3; learning_rate=0.05; subsample=0.85; colsample_bytree=0.9",
        "role": "Nonlinear ensemble benchmark.",
    },
    {
        "model": "ExtraTrees",
        "parameters": "n_estimators=500; min_samples_leaf=2; class_weight=balanced",
        "role": "Tree-ensemble benchmark.",
    },
    {
        "model": "Random forest",
        "parameters": "n_estimators=500; min_samples_leaf=2; class_weight=balanced_subsample",
        "role": "Tree-ensemble benchmark.",
    },
    {
        "model": "HistGradientBoosting",
        "parameters": "max_iter=400; learning_rate=0.05; max_leaf_nodes=31; l2_regularization=0.01",
        "role": "Gradient boosting benchmark.",
    },
    {
        "model": "Dense MLP",
        "parameters": "2 hidden layers; hidden_dim=128; batch normalization; ReLU; dropout=0.15",
        "role": "Dense deep-learning benchmark.",
    },
    {
        "model": "Residual MLP",
        "parameters": "hidden_dim=128; 3 residual blocks; dropout=0.15",
        "role": "Residual deep-learning benchmark.",
    },
    {
        "model": "FT-Transformer-lite",
        "parameters": "d_token=64; n_heads=4; n_layers=3; dropout=0.15",
        "role": "Lightweight tabular transformer benchmark.",
    },
    {
        "model": "TabNet",
        "parameters": "n_d=16; n_a=16; n_steps=3; gamma=1.3; lambda_sparse=1e-4; mask_type=entmax",
        "role": "Attention-based tabular benchmark.",
    },
    {
        "model": "TabICL",
        "parameters": "default TabICLClassifier",
        "role": "Tabular in-context learning benchmark.",
    },
    {
        "model": "TabPFN",
        "parameters": "default TabPFNClassifier",
        "role": "Attempted tabular foundation model; failed for tasks with >10 classes.",
    },
]


REVIEWER_CRITICISM_MAP = [
    {
        "criticism": "NMF rank is arbitrary and manual curation is biased.",
        "response": "Added NMF rank and stability analysis across k=2 to 20 and 30 seeds. Programs are described as recurring biological classes consolidated across ranks and feature spaces, not as one k=9 solution.",
        "supporting_output": "nmf_rank_stability_summary.md; nmf_rank_stability_metrics.csv; nmf_curated_program_recovery_summary.csv",
    },
    {
        "criticism": "Classification is trivial and dummy baseline is insufficient.",
        "response": "Added representation controls comparing curated ECM programs to Matrisome PCA9, random Matrisome gene sets, and random non-ECM gene sets under donor-stratified classification.",
        "supporting_output": "r2_representation_control_summary.md; r2_representation_control_summary.csv",
    },
    {
        "criticism": "MatrisomeDB validation is weak because it only uses detection coverage.",
        "response": "Added matched random Matrisome null models for detection coverage and NSAF abundance support, plus RNA-protein tissue-profile correlation analysis.",
        "supporting_output": "r3_matrisomedb_null_abundance_summary.md; r3_detection_null_summary.csv; r3_abundance_null_summary.csv",
    },
    {
        "criticism": "Bulk transcriptomics may reflect cell composition.",
        "response": "Added Tabula Sapiens donor × organ × cell-type pseudobulk analysis to identify likely cellular sources of ECM programs.",
        "supporting_output": "route_b_final_summary.md; route_b_program_source_summary.csv",
    },
    {
        "criticism": "Program scoring may depend on arbitrary mean-expression scoring.",
        "response": "Added rank-based scoring robustness analysis inspired by UCell-like scoring. Most programs remain stable under rank-based scoring.",
        "supporting_output": "rank_based_ecm_program_scoring_summary.md; rank_based_all_rank_vs_mean_score_correlation.csv",
    },
    {
        "criticism": "Spatial ECM organization is missing.",
        "response": "Added focused spatial transcriptomics validation using public 10x Visium breast cancer CytAssist FFPE and healthy human lymph node datasets.",
        "supporting_output": "r7_spatial_validation_summary.md; spatial program maps and correlations",
    },
    {
        "criticism": "Code/data availability and exact parameters are insufficient.",
        "response": "Built reproducibility package with program gene lists, data manifest, pipeline order, exact parameter tables, model settings, and code/data statement.",
        "supporting_output": "Supplementary Tables S1-S8; reproducibility_checklist.md; code_and_data_availability_statement.md",
    },
]


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(METHOD_PARAMETERS).to_csv(
        TABLE_DIR / "Supplementary_Table_S6_exact_method_parameters.csv",
        index=False,
    )

    pd.DataFrame(MODEL_PARAMETERS).to_csv(
        TABLE_DIR / "Supplementary_Table_S7_model_parameters.csv",
        index=False,
    )

    pd.DataFrame(REVIEWER_CRITICISM_MAP).to_csv(
        TABLE_DIR / "Supplementary_Table_S8_reviewer_criticism_response_map.csv",
        index=False,
    )

    print("[SAVED]")
    print(TABLE_DIR / "Supplementary_Table_S6_exact_method_parameters.csv")
    print(TABLE_DIR / "Supplementary_Table_S7_model_parameters.csv")
    print(TABLE_DIR / "Supplementary_Table_S8_reviewer_criticism_response_map.csv")


if __name__ == "__main__":
    main()