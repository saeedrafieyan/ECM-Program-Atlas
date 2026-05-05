from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    BaggingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight


INPUT_PATH = Path(
    "data/processed/gtex_v11_sample_level/"
    "gtex_v11_program_scores_zscore_mean_with_metadata.csv"
)

OUTPUT_DIR = Path("outputs/gtex_v11_sample_level_validation/model_benchmark")
TABLE_DIR = OUTPUT_DIR / "tables"
HTML_DIR = OUTPUT_DIR / "figures" / "html"
PNG_DIR = OUTPUT_DIR / "figures" / "png"


PROGRAM_ORDER = [
    "Vascular/stromal/interstitial ECM",
    "Epithelial/mucosal basement membrane ECM",
    "CNS/neural ECM",
    "Retinal/sensory ECM",
    "Immune/lymphoid remodeling ECM",
    "Stromal remodeling ECM",
    "Renal/endothelial basement membrane ECM",
    "Hepatic/plasma-associated ECM",
    "Reproductive-specialized ECM",
]


TISSUE_SYSTEM_MAP: Dict[str, str] = {
    "Brain": "CNS",
    "Nerve": "Peripheral nerve",
    "Pituitary": "Endocrine / neural-related",

    "Blood Vessel": "Vascular / connective",
    "Adipose Tissue": "Vascular / connective",
    "Muscle": "Muscle",
    "Heart": "Muscle",

    "Lung": "Respiratory",

    "Colon": "Digestive",
    "Small Intestine": "Digestive",
    "Stomach": "Digestive",
    "Esophagus": "Digestive",
    "Liver": "Digestive",
    "Pancreas": "Digestive",
    "Minor Salivary Gland": "Digestive",

    "Spleen": "Immune / blood",
    "Whole Blood": "Immune / blood",

    "Kidney": "Urinary",
    "Bladder": "Urinary",

    "Breast": "Reproductive",
    "Cervix Uteri": "Reproductive",
    "Uterus": "Reproductive",
    "Ovary": "Reproductive",
    "Fallopian Tube": "Reproductive",
    "Prostate": "Reproductive",
    "Testis": "Reproductive",
    "Vagina": "Reproductive",

    "Thyroid": "Endocrine",
    "Adrenal Gland": "Endocrine",

    "Skin": "Epithelial / barrier",
    "Cells": "Cell culture",
}


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig: go.Figure, name: str, width: int = 1450, height: int = 850) -> None:
    html_path = HTML_DIR / f"{name}.html"
    png_path = PNG_DIR / f"{name}.png"

    fig.update_layout(width=width, height=height)
    fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)

    try:
        fig.write_image(str(png_path), scale=3)
        print(f"[SAVED] {png_path}")
    except Exception as exc:
        print(f"[WARNING] PNG export failed for {name}: {exc}")

    print(f"[SAVED] {html_path}")


def load_scores(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    df = pd.read_csv(path)

    required = ["sample_id", "subject_id", "tissue", "tissue_detail"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    missing_programs = [p for p in PROGRAM_ORDER if p not in df.columns]
    if missing_programs:
        raise ValueError(f"Missing ECM program columns: {missing_programs}")

    df["tissue_system"] = df["tissue"].map(TISSUE_SYSTEM_MAP).fillna("Other")

    return df


def filter_classes(df: pd.DataFrame, label_col: str, min_samples_per_class: int) -> pd.DataFrame:
    counts = df[label_col].value_counts()
    valid = counts[counts >= min_samples_per_class].index.tolist()
    return df[df[label_col].isin(valid)].copy()


def get_model_specs() -> Dict[str, object]:
    models: Dict[str, object] = {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),

        "logistic_regression": LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="lbfgs",
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),

        "extra_trees": ExtraTreesClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),

        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=400,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            random_state=42,
        ),

        "bagged_decision_trees": BaggingClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
            ),
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        ),
    }

    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=500,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )
    except Exception:
        print("[INFO] xgboost is not installed or failed to import. Skipping XGBoost.")

    return models


def scale_if_needed(model_name: str, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if model_name in ["logistic_regression"]:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    return X_train, X_test


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }


def run_grouped_benchmark(
    df: pd.DataFrame,
    label_col: str,
    feature_cols: List[str],
    min_samples_per_class: int,
    n_splits: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    filtered = filter_classes(df, label_col=label_col, min_samples_per_class=min_samples_per_class)
    filtered = filtered.dropna(subset=feature_cols + [label_col, "subject_id"]).copy()

    X = filtered[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    y_str = filtered[label_col].astype(str).values
    groups = filtered["subject_id"].astype(str).values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_str)

    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    models = get_model_specs()

    fold_records = []
    prediction_records = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), start=1):
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

        for model_name, model in models.items():
            X_train, X_test = scale_if_needed(model_name, X_train_raw, X_test_raw)

            try:
                if model_name == "xgboost":
                    model.fit(X_train, y_train, sample_weight=sample_weight)
                    y_pred = model.predict(X_test)

                elif model_name in ["hist_gradient_boosting"]:
                    model.fit(X_train, y_train, sample_weight=sample_weight)
                    y_pred = model.predict(X_test)

                elif model_name == "dummy_most_frequent":
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                else:
                    model.fit(X_train, y_train, sample_weight=sample_weight)
                    y_pred = model.predict(X_test)

            except TypeError:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            metrics = evaluate(y_test, y_pred)

            record = {
                "label_col": label_col,
                "model": model_name,
                "fold": fold_idx,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_train_subjects": len(np.unique(groups[train_idx])),
                "n_test_subjects": len(np.unique(groups[test_idx])),
                "n_classes": len(label_encoder.classes_),
            }

            record.update(metrics)
            fold_records.append(record)

            pred_meta = filtered.iloc[test_idx][
                ["sample_id", "subject_id", "tissue", "tissue_detail", "tissue_system"]
            ].copy()

            pred_meta["label_col"] = label_col
            pred_meta["model"] = model_name
            pred_meta["fold"] = fold_idx
            pred_meta["y_true"] = label_encoder.inverse_transform(y_test)
            pred_meta["y_pred"] = label_encoder.inverse_transform(y_pred.astype(int))

            prediction_records.append(pred_meta)

    fold_df = pd.DataFrame(fold_records)
    pred_df = pd.concat(prediction_records, ignore_index=True)

    return fold_df, pred_df


def summarize_folds(fold_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]

    records = []

    for (label_col, model), group in fold_df.groupby(["label_col", "model"]):
        record = {
            "label_col": label_col,
            "model": model,
            "n_folds": group["fold"].nunique(),
            "mean_n_train": group["n_train"].mean(),
            "mean_n_test": group["n_test"].mean(),
            "mean_n_classes": group["n_classes"].mean(),
        }

        for metric in metric_cols:
            record[f"{metric}_mean"] = group[metric].mean()
            record[f"{metric}_std"] = group[metric].std(ddof=0)

        records.append(record)

    summary = pd.DataFrame(records)

    summary = summary.sort_values(
        ["label_col", "balanced_accuracy_mean", "macro_f1_mean"],
        ascending=[True, False, False],
    )

    return summary


def plot_model_benchmark(summary_df: pd.DataFrame) -> None:
    plot_df = summary_df.copy()

    metric = "balanced_accuracy_mean"
    err = "balanced_accuracy_std"

    fig = go.Figure()

    for label_col, group in plot_df.groupby("label_col"):
        group = group.sort_values(metric, ascending=True)

        fig.add_trace(
            go.Bar(
                x=group[metric],
                y=group["model"],
                orientation="h",
                error_x=dict(type="data", array=group[err], visible=True),
                name=label_col,
                customdata=group[["macro_f1_mean", "accuracy_mean"]],
                hovertemplate=(
                    "Model: %{y}<br>"
                    "Balanced accuracy: %{x:.3f}<br>"
                    "Macro-F1: %{customdata[0]:.3f}<br>"
                    "Accuracy: %{customdata[1]:.3f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="GTEx V11 donor-stratified classifier benchmark using 9 ECM program scores",
        xaxis_title="Balanced accuracy",
        yaxis_title="Model",
        template="plotly_white",
        barmode="group",
        margin=dict(l=240, r=60, t=100, b=90),
    )

    save_figure(fig, "gtex_v11_ecm_classifier_benchmark_balanced_accuracy", width=1350, height=850)


def plot_metric_heatmap(summary_df: pd.DataFrame) -> None:
    plot_df = summary_df.copy()

    metrics = [
        "accuracy_mean",
        "balanced_accuracy_mean",
        "macro_f1_mean",
        "weighted_f1_mean",
    ]

    for label_col, group in plot_df.groupby("label_col"):
        matrix = group.set_index("model")[metrics]
        matrix = matrix.sort_values("balanced_accuracy_mean", ascending=False)

        fig = go.Figure(
            data=go.Heatmap(
                z=matrix.values,
                x=[m.replace("_mean", "").replace("_", " ") for m in matrix.columns],
                y=matrix.index.tolist(),
                text=[[f"{v:.3f}" for v in row] for row in matrix.values],
                texttemplate="%{text}",
                colorscale="Viridis",
                colorbar=dict(title="Score"),
                hovertemplate=(
                    "Model: %{y}<br>"
                    "Metric: %{x}<br>"
                    "Score: %{z:.3f}<extra></extra>"
                ),
            )
        )

        fig.update_layout(
            title=f"GTEx V11 ECM classifier benchmark, {label_col}",
            template="plotly_white",
            margin=dict(l=240, r=60, t=100, b=100),
        )

        safe_label = label_col.replace(" ", "_").lower()
        save_figure(fig, f"gtex_v11_classifier_metric_heatmap_{safe_label}", width=1150, height=800)


def write_report(summary_df: pd.DataFrame) -> None:
    report_path = OUTPUT_DIR / "gtex_v11_ecm_classifier_benchmark_summary.md"

    lines = []
    lines.append("# GTEx V11 ECM Classifier Benchmark\n")
    lines.append("## Purpose\n")
    lines.append(
        "This benchmark compares simple and ensemble classifiers for predicting GTEx tissue labels from only 9 curated ECM program scores under donor-stratified cross-validation."
    )

    for label_col in sorted(summary_df["label_col"].unique()):
        lines.append(f"\n## Task: {label_col}\n")

        subset = summary_df[summary_df["label_col"].eq(label_col)]
        subset = subset.sort_values("balanced_accuracy_mean", ascending=False)

        best = subset.iloc[0]

        lines.append(
            f"Best model by balanced accuracy: **{best['model']}** "
            f"({best['balanced_accuracy_mean']:.3f} ± {best['balanced_accuracy_std']:.3f})."
        )

        lines.append("\n| Model | Accuracy | Balanced accuracy | Macro-F1 | Weighted-F1 |")
        lines.append("|---|---:|---:|---:|---:|")

        for row in subset.itertuples():
            lines.append(
                f"| {row.model} | "
                f"{row.accuracy_mean:.3f} ± {row.accuracy_std:.3f} | "
                f"{row.balanced_accuracy_mean:.3f} ± {row.balanced_accuracy_std:.3f} | "
                f"{row.macro_f1_mean:.3f} ± {row.macro_f1_std:.3f} | "
                f"{row.weighted_f1_mean:.3f} ± {row.weighted_f1_std:.3f} |"
            )

    lines.append("\n## Interpretation\n")
    lines.append(
        "If ensemble models outperform logistic regression only marginally, logistic regression should remain the primary manuscript model because it is simpler and more interpretable. If tree ensembles or XGBoost improve performance substantially, they can be reported as complementary nonlinear validation."
    )

    lines.append("\n## Important note\n")
    lines.append(
        "All models are evaluated with GroupKFold by subject_id, so samples from the same donor are not shared between train and test folds."
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAVED] {report_path}")


def main() -> None:
    ensure_dirs()

    df = load_scores(INPUT_PATH)
    feature_cols = PROGRAM_ORDER

    tasks = [
        {
            "label_col": "tissue_system",
            "min_samples_per_class": 100,
            "n_splits": 5,
        },
        {
            "label_col": "tissue",
            "min_samples_per_class": 100,
            "n_splits": 5,
        },
    ]

    all_folds = []
    all_predictions = []

    for task in tasks:
        print(f"[TASK] {task['label_col']}")

        fold_df, pred_df = run_grouped_benchmark(
            df=df,
            label_col=task["label_col"],
            feature_cols=feature_cols,
            min_samples_per_class=task["min_samples_per_class"],
            n_splits=task["n_splits"],
        )

        all_folds.append(fold_df)
        all_predictions.append(pred_df)

    folds = pd.concat(all_folds, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    summary = summarize_folds(folds)

    folds.to_csv(TABLE_DIR / "gtex_v11_ecm_classifier_benchmark_fold_metrics.csv", index=False)
    predictions.to_csv(TABLE_DIR / "gtex_v11_ecm_classifier_benchmark_predictions.csv", index=False)
    summary.to_csv(TABLE_DIR / "gtex_v11_ecm_classifier_benchmark_summary.csv", index=False)

    plot_model_benchmark(summary)
    plot_metric_heatmap(summary)
    write_report(summary)

    print("\n[DONE]")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()