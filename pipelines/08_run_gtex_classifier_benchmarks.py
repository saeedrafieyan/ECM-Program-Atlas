from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


INPUT_PATH = Path(
    "data/processed/gtex_v11_sample_level/"
    "gtex_v11_program_scores_zscore_mean_with_metadata.csv"
)

OUTPUT_DIR = Path("outputs/gtex_v11_sample_level_validation/classification")
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


def save_figure(fig: go.Figure, name: str, width: int = 1200, height: int = 800) -> None:
    html_path = HTML_DIR / f"{name}.html"
    png_path = PNG_DIR / f"{name}.png"

    fig.update_layout(width=width, height=height)

    fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)

    try:
        fig.write_image(str(png_path), scale=3)
        print(f"[SAVED] {png_path}")
    except Exception as exc:
        print(f"[WARNING] Could not export PNG for {name}: {exc}")

    print(f"[SAVED] {html_path}")


def load_scores(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    df = pd.read_csv(path)

    required = ["sample_id", "subject_id", "tissue", "tissue_detail"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required metadata columns: {missing}")

    program_cols = [col for col in PROGRAM_ORDER if col in df.columns]

    if len(program_cols) != len(PROGRAM_ORDER):
        missing_programs = sorted(set(PROGRAM_ORDER).difference(program_cols))
        raise ValueError(f"Missing ECM program score columns: {missing_programs}")

    df["tissue_system"] = df["tissue"].map(TISSUE_SYSTEM_MAP).fillna("Other")

    return df


def filter_classes(
    df: pd.DataFrame,
    label_col: str,
    min_samples_per_class: int,
) -> pd.DataFrame:
    counts = df[label_col].value_counts()
    valid_classes = counts[counts >= min_samples_per_class].index.tolist()

    filtered = df[df[label_col].isin(valid_classes)].copy()

    return filtered


def get_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    solver="lbfgs",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }


def run_grouped_cv(
    df: pd.DataFrame,
    label_col: str,
    feature_cols: List[str],
    n_splits: int,
    min_samples_per_class: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    filtered = filter_classes(
        df=df,
        label_col=label_col,
        min_samples_per_class=min_samples_per_class,
    )

    filtered = filtered.dropna(subset=feature_cols + [label_col, "subject_id"]).copy()

    X = filtered[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    y = filtered[label_col].astype(str).values
    groups = filtered["subject_id"].astype(str).values

    unique_groups = np.unique(groups)
    actual_splits = min(n_splits, len(unique_groups))

    if actual_splits < 2:
        raise ValueError(f"Not enough donor groups for {label_col} classification.")

    group_kfold = GroupKFold(n_splits=actual_splits)

    model = get_model()
    dummy = DummyClassifier(strategy="most_frequent")

    fold_records = []
    prediction_records = []

    for fold_idx, (train_idx, test_idx) in enumerate(group_kfold.split(X, y, groups), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        dummy.fit(X_train, y_train)
        y_dummy = dummy.predict(X_test)

        metrics = evaluate_predictions(y_test, y_pred)
        dummy_metrics = evaluate_predictions(y_test, y_dummy)

        record = {
            "label_col": label_col,
            "fold": fold_idx,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "n_train_subjects": len(np.unique(groups[train_idx])),
            "n_test_subjects": len(np.unique(groups[test_idx])),
            "n_classes_train": len(np.unique(y_train)),
            "n_classes_test": len(np.unique(y_test)),
        }

        for key, value in metrics.items():
            record[f"ecm_program_logistic_{key}"] = value

        for key, value in dummy_metrics.items():
            record[f"dummy_{key}"] = value

        fold_records.append(record)

        test_meta = filtered.iloc[test_idx][
            ["sample_id", "subject_id", "tissue", "tissue_detail", "tissue_system"]
        ].copy()

        test_meta["label_col"] = label_col
        test_meta["y_true"] = y_test
        test_meta["y_pred"] = y_pred
        test_meta["fold"] = fold_idx

        prediction_records.append(test_meta)

    fold_df = pd.DataFrame(fold_records)
    pred_df = pd.concat(prediction_records, ignore_index=True)

    summary_records = []

    metric_cols = [
        col for col in fold_df.columns
        if col.startswith("ecm_program_logistic_") or col.startswith("dummy_")
    ]

    for metric in metric_cols:
        values = fold_df[metric].astype(float)

        summary_records.append(
            {
                "label_col": label_col,
                "metric": metric,
                "mean": values.mean(),
                "std": values.std(ddof=0),
                "min": values.min(),
                "max": values.max(),
                "n_folds": values.shape[0],
                "n_samples_used": filtered.shape[0],
                "n_subjects_used": filtered["subject_id"].nunique(),
                "n_classes": filtered[label_col].nunique(),
                "min_samples_per_class": min_samples_per_class,
            }
        )

    summary_df = pd.DataFrame(summary_records)

    return fold_df, summary_df, pred_df


def plot_performance_summary(summary_df: pd.DataFrame) -> None:
    plot_df = summary_df[
        summary_df["metric"].isin(
            [
                "ecm_program_logistic_accuracy",
                "ecm_program_logistic_balanced_accuracy",
                "ecm_program_logistic_macro_f1",
                "dummy_accuracy",
                "dummy_balanced_accuracy",
                "dummy_macro_f1",
            ]
        )
    ].copy()

    plot_df["model"] = plot_df["metric"].apply(
        lambda x: "ECM program logistic regression" if x.startswith("ecm_") else "Dummy baseline"
    )

    plot_df["metric_clean"] = (
        plot_df["metric"]
        .str.replace("ecm_program_logistic_", "", regex=False)
        .str.replace("dummy_", "", regex=False)
        .str.replace("_", " ")
    )

    fig = go.Figure()

    for model_name, group in plot_df.groupby("model"):
        fig.add_trace(
            go.Bar(
                x=group["label_col"] + " | " + group["metric_clean"],
                y=group["mean"],
                error_y=dict(type="data", array=group["std"], visible=True),
                name=model_name,
                hovertemplate=(
                    "Task / metric: %{x}<br>"
                    "Mean: %{y:.3f}<br>"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Donor-stratified GTEx V11 tissue classification from 9 ECM program scores",
        yaxis_title="Cross-validated performance",
        xaxis_title="Task and metric",
        template="plotly_white",
        barmode="group",
        xaxis=dict(tickangle=35),
        margin=dict(l=80, r=60, t=100, b=180),
    )

    save_figure(fig, "gtex_v11_donor_stratified_classification_performance", width=1450, height=850)


def plot_confusion_matrix(pred_df: pd.DataFrame, label_col: str) -> None:
    subset = pred_df[pred_df["label_col"].eq(label_col)].copy()

    labels = sorted(set(subset["y_true"]).union(set(subset["y_pred"])))

    cm = confusion_matrix(
        subset["y_true"],
        subset["y_pred"],
        labels=labels,
        normalize="true",
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale="Blues",
            colorbar=dict(title="Recall"),
            hovertemplate=(
                "True: %{y}<br>"
                "Predicted: %{x}<br>"
                "Recall fraction: %{z:.3f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=f"Normalized confusion matrix, {label_col}",
        xaxis_title="Predicted label",
        yaxis_title="True label",
        template="plotly_white",
        margin=dict(l=240, r=60, t=100, b=220),
        xaxis=dict(tickangle=45),
    )

    safe_label = label_col.replace(" ", "_").lower()

    save_figure(fig, f"gtex_v11_confusion_matrix_{safe_label}", width=1200, height=1000)


def create_markdown_report(summary_df: pd.DataFrame) -> None:
    report_path = OUTPUT_DIR / "gtex_v11_donor_stratified_classification_summary.md"

    lines = []
    lines.append("# GTEx V11 Donor-Stratified Tissue Classification\n")
    lines.append("## Purpose\n")
    lines.append(
        "This analysis tests whether the 9 curated ECM program scores can classify GTEx tissues or tissue systems under donor-stratified cross-validation."
    )

    lines.append("\n## Main results\n")

    for label_col in sorted(summary_df["label_col"].unique()):
        subset = summary_df[summary_df["label_col"].eq(label_col)]

        lines.append(f"### Task: {label_col}\n")

        for metric_name in [
            "ecm_program_logistic_accuracy",
            "ecm_program_logistic_balanced_accuracy",
            "ecm_program_logistic_macro_f1",
            "dummy_accuracy",
            "dummy_balanced_accuracy",
            "dummy_macro_f1",
        ]:
            row = subset[subset["metric"].eq(metric_name)]

            if row.empty:
                continue

            r = row.iloc[0]

            lines.append(
                f"- {metric_name}: {r['mean']:.3f} ± {r['std']:.3f}"
            )

        n_samples = subset["n_samples_used"].iloc[0]
        n_subjects = subset["n_subjects_used"].iloc[0]
        n_classes = subset["n_classes"].iloc[0]

        lines.append(
            f"- Samples used: {n_samples}; subjects used: {n_subjects}; classes: {n_classes}\n"
        )

    lines.append("\n## Interpretation\n")
    lines.append(
        "If ECM program logistic regression strongly outperforms the dummy baseline, the curated ECM programs contain reproducible tissue information at individual-sample level. Because the split is grouped by subject ID, the model is tested on donors not seen during training."
    )

    lines.append("\n## Limitations\n")
    lines.append("1. The classifier uses only 9 ECM program scores, not the full Matrisome expression matrix.\n")
    lines.append("2. Tissue classes are imbalanced, so balanced accuracy and macro-F1 are more informative than raw accuracy.\n")
    lines.append("3. Some tissues may be biologically similar in ECM space, so misclassification among related tissues is not necessarily a failure.\n")
    lines.append("4. This is a transcriptomic validation, not protein-level validation.\n")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAVED] {report_path}")


def main() -> None:
    ensure_dirs()

    df = load_scores(INPUT_PATH)

    feature_cols = [program for program in PROGRAM_ORDER if program in df.columns]

    print("[INFO] Loaded GTEx V11 ECM program score matrix")
    print(f"Samples: {df.shape[0]}")
    print(f"Subjects: {df['subject_id'].nunique()}")
    print(f"Tissues: {df['tissue'].nunique()}")
    print(f"Tissue details: {df['tissue_detail'].nunique()}")
    print(f"Features: {feature_cols}")

    all_fold_records = []
    all_summary_records = []
    all_prediction_records = []

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

    for task in tasks:
        print(f"\n[CLASSIFICATION TASK] {task['label_col']}")

        fold_df, summary_df, pred_df = run_grouped_cv(
            df=df,
            label_col=task["label_col"],
            feature_cols=feature_cols,
            n_splits=task["n_splits"],
            min_samples_per_class=task["min_samples_per_class"],
        )

        all_fold_records.append(fold_df)
        all_summary_records.append(summary_df)
        all_prediction_records.append(pred_df)

    folds = pd.concat(all_fold_records, ignore_index=True)
    summary = pd.concat(all_summary_records, ignore_index=True)
    predictions = pd.concat(all_prediction_records, ignore_index=True)

    folds.to_csv(TABLE_DIR / "gtex_v11_donor_stratified_fold_metrics.csv", index=False)
    summary.to_csv(TABLE_DIR / "gtex_v11_donor_stratified_classification_summary.csv", index=False)
    predictions.to_csv(TABLE_DIR / "gtex_v11_donor_stratified_predictions.csv", index=False)

    plot_performance_summary(summary)
    plot_confusion_matrix(predictions, label_col="tissue_system")
    plot_confusion_matrix(predictions, label_col="tissue")

    create_markdown_report(summary)

    print("\n[DONE]")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Tables:        {TABLE_DIR}")
    print(f"HTML figures:  {HTML_DIR}")
    print(f"PNG figures:   {PNG_DIR}")


if __name__ == "__main__":
    main()