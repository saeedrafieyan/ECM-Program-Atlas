from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


INPUT_PATH = Path(
    "data/processed/gtex_v11_sample_level/"
    "gtex_v11_program_scores_zscore_mean_with_metadata.csv"
)

OUTPUT_DIR = Path("outputs/gtex_v11_sample_level_validation/deep_tabular_benchmark")
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
        print(f"[WARNING] Could not export PNG for {name}: {exc}")

    print(f"[SAVED] {html_path}")


def load_scores(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    df = pd.read_csv(path)

    required = ["sample_id", "subject_id", "tissue", "tissue_detail"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for program in PROGRAM_ORDER:
        if program not in df.columns:
            raise ValueError(f"Missing ECM program column: {program}")

    df["tissue_system"] = df["tissue"].map(TISSUE_SYSTEM_MAP).fillna("Other")

    return df


def filter_classes(df: pd.DataFrame, label_col: str, min_samples_per_class: int) -> pd.DataFrame:
    counts = df[label_col].value_counts()
    valid = counts[counts >= min_samples_per_class].index.tolist()
    return df[df[label_col].isin(valid)].copy()


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }


class DenseMLP(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden_dim: int = 128, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class ResidualMLP(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden_dim: int = 128, n_blocks: int = 3, dropout: float = 0.15):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim=hidden_dim, dropout=dropout) for _ in range(n_blocks)]
        )
        self.output = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        x = self.blocks(x)
        return self.output(x)


class SimpleFTTransformer(nn.Module):
    """
    Lightweight FT-Transformer-style model for continuous tabular features.

    Each scalar feature is tokenized into an embedding:
        token_j = value_j * weight_j + bias_j

    Then TransformerEncoder layers model feature interactions.
    """
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        d_token: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, d_token) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, d_token))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        cls = self.cls.expand(x.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        encoded = self.encoder(tokens)
        return self.head(encoded[:, 0])


def train_torch_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_classes: int,
    model_name: str,
    class_weight: bool = True,
    max_epochs: int = 250,
    batch_size: int = 512,
    patience: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    # Internal validation split from training data.
    rng = np.random.default_rng(42)
    indices = np.arange(X_train.shape[0])
    rng.shuffle(indices)

    val_size = max(int(0.15 * len(indices)), n_classes)
    val_idx = indices[:val_size]
    tr_idx = indices[val_size:]

    train_ds = TensorDataset(X_train_t[tr_idx], y_train_t[tr_idx])
    val_x = X_train_t[val_idx].to(device)
    val_y = y_train_t[val_idx].to(device)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = model.to(device)

    if class_weight:
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(n_classes),
            y=y_train,
        )
        weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        model.train()

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            val_loss = loss_fn(val_logits, val_y).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = model(X_test_t.to(device))
        y_pred = logits.argmax(dim=1).cpu().numpy()

    return y_pred


def fit_predict_sklearn_model(model_name: str, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    if model_name == "dummy_most_frequent":
        model = DummyClassifier(strategy="most_frequent")
        model.fit(X_train, y_train)
        return model.predict(X_test)

    if model_name == "logistic_regression":
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="lbfgs",
        )
        model.fit(X_train_s, y_train)
        return model.predict(X_test_s)

    if model_name == "extra_trees":
        model = ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        return model.predict(X_test)

    if model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        return model.predict(X_test)

    if model_name == "hist_gradient_boosting":
        weights = compute_sample_weight(class_weight="balanced", y=y_train)
        model = HistGradientBoostingClassifier(
            max_iter=400,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            random_state=42,
        )
        model.fit(X_train, y_train, sample_weight=weights)
        return model.predict(X_test)

    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except Exception as exc:
            raise RuntimeError(f"xgboost is not installed: {exc}")

        weights = compute_sample_weight(class_weight="balanced", y=y_train)
        model = XGBClassifier(
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
        model.fit(X_train, y_train, sample_weight=weights)
        return model.predict(X_test)

    raise ValueError(f"Unknown sklearn model: {model_name}")


def fit_predict_tabnet(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
    except Exception as exc:
        raise RuntimeError(f"pytorch-tabnet is not installed: {exc}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    weights = compute_sample_weight(class_weight="balanced", y=y_train)

    model = TabNetClassifier(
        n_d=16,
        n_a=16,
        n_steps=3,
        gamma=1.3,
        lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=1e-2),
        mask_type="entmax",
        seed=42,
        verbose=0,
    )

    model.fit(
        X_train=X_train_s,
        y_train=y_train,
        weights=weights,
        max_epochs=200,
        patience=30,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
    )

    return model.predict(X_test_s).astype(int)


def fit_predict_tabpfn(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, max_train_samples: int = 10000) -> np.ndarray:
    try:
        from tabpfn import TabPFNClassifier
    except Exception as exc:
        raise RuntimeError(f"tabpfn is not installed: {exc}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # TabPFN is intended for smaller train sets, so subsample if needed.
    if X_train_s.shape[0] > max_train_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(X_train_s.shape[0], size=max_train_samples, replace=False)
        X_train_s = X_train_s[idx]
        y_train = y_train[idx]

    model = TabPFNClassifier()
    model.fit(X_train_s, y_train)
    return model.predict(X_test_s).astype(int)


def fit_predict_tabicl(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    try:
        from tabicl import TabICLClassifier
    except Exception as exc:
        raise RuntimeError(f"tabicl is not installed or API changed: {exc}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = TabICLClassifier()
    model.fit(X_train_s, y_train)
    return model.predict(X_test_s).astype(int)


def run_benchmark(
    df: pd.DataFrame,
    label_col: str,
    feature_cols: List[str],
    model_names: List[str],
    min_samples_per_class: int,
    n_splits: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    filtered = df.copy()
    counts = filtered[label_col].value_counts()
    valid = counts[counts >= min_samples_per_class].index.tolist()
    filtered = filtered[filtered[label_col].isin(valid)].copy()

    filtered = filtered.dropna(subset=feature_cols + [label_col, "subject_id"]).copy()

    X = filtered[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    y_str = filtered[label_col].astype(str).values
    groups = filtered["subject_id"].astype(str).values

    le = LabelEncoder()
    y = le.fit_transform(y_str)
    n_classes = len(le.classes_)

    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))

    fold_records = []
    prediction_records = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), start=1):
        print(f"[{label_col}] Fold {fold_idx}")

        X_train_raw = X[train_idx]
        X_test_raw = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        for model_name in model_names:
            print(f"  Model: {model_name}")

            try:
                if model_name in [
                    "dummy_most_frequent",
                    "logistic_regression",
                    "extra_trees",
                    "random_forest",
                    "hist_gradient_boosting",
                    "xgboost",
                ]:
                    y_pred = fit_predict_sklearn_model(
                        model_name=model_name,
                        X_train=X_train_raw,
                        y_train=y_train,
                        X_test=X_test_raw,
                    )

                elif model_name == "dense_mlp":
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train_raw)
                    X_test = scaler.transform(X_test_raw)

                    model = DenseMLP(
                        n_features=X_train.shape[1],
                        n_classes=n_classes,
                        hidden_dim=128,
                        dropout=0.15,
                    )

                    y_pred = train_torch_model(
                        model=model,
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        n_classes=n_classes,
                        model_name=model_name,
                    )

                elif model_name == "residual_mlp":
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train_raw)
                    X_test = scaler.transform(X_test_raw)

                    model = ResidualMLP(
                        n_features=X_train.shape[1],
                        n_classes=n_classes,
                        hidden_dim=128,
                        n_blocks=3,
                        dropout=0.15,
                    )

                    y_pred = train_torch_model(
                        model=model,
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        n_classes=n_classes,
                        model_name=model_name,
                    )

                elif model_name == "ft_transformer_lite":
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train_raw)
                    X_test = scaler.transform(X_test_raw)

                    model = SimpleFTTransformer(
                        n_features=X_train.shape[1],
                        n_classes=n_classes,
                        d_token=64,
                        n_heads=4,
                        n_layers=3,
                        dropout=0.15,
                    )

                    y_pred = train_torch_model(
                        model=model,
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        n_classes=n_classes,
                        model_name=model_name,
                        lr=1e-3,
                    )

                elif model_name == "tabnet":
                    y_pred = fit_predict_tabnet(X_train_raw, y_train, X_test_raw)

                elif model_name == "tabpfn":
                    y_pred = fit_predict_tabpfn(X_train_raw, y_train, X_test_raw)

                elif model_name == "tabicl":
                    y_pred = fit_predict_tabicl(X_train_raw, y_train, X_test_raw)

                else:
                    raise ValueError(f"Unknown model: {model_name}")

                metrics = evaluate(y_test, y_pred)

                record = {
                    "label_col": label_col,
                    "model": model_name,
                    "fold": fold_idx,
                    "status": "ok",
                    "error": "",
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "n_classes": n_classes,
                    "n_train_subjects": len(np.unique(groups[train_idx])),
                    "n_test_subjects": len(np.unique(groups[test_idx])),
                    **metrics,
                }

                pred_meta = filtered.iloc[test_idx][
                    ["sample_id", "subject_id", "tissue", "tissue_detail", "tissue_system"]
                ].copy()

                pred_meta["label_col"] = label_col
                pred_meta["model"] = model_name
                pred_meta["fold"] = fold_idx
                pred_meta["y_true"] = le.inverse_transform(y_test)
                pred_meta["y_pred"] = le.inverse_transform(y_pred.astype(int))

                prediction_records.append(pred_meta)

            except Exception as exc:
                print(f"  [FAILED] {model_name}: {exc}")

                record = {
                    "label_col": label_col,
                    "model": model_name,
                    "fold": fold_idx,
                    "status": "failed",
                    "error": str(exc),
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "n_classes": n_classes,
                    "n_train_subjects": len(np.unique(groups[train_idx])),
                    "n_test_subjects": len(np.unique(groups[test_idx])),
                    "accuracy": np.nan,
                    "balanced_accuracy": np.nan,
                    "macro_f1": np.nan,
                    "weighted_f1": np.nan,
                }

            fold_records.append(record)

    fold_df = pd.DataFrame(fold_records)

    if prediction_records:
        pred_df = pd.concat(prediction_records, ignore_index=True)
    else:
        pred_df = pd.DataFrame()

    return fold_df, pred_df


def summarize_folds(fold_df: pd.DataFrame) -> pd.DataFrame:
    ok = fold_df[fold_df["status"].eq("ok")].copy()

    records = []

    for (label_col, model), group in ok.groupby(["label_col", "model"]):
        record = {
            "label_col": label_col,
            "model": model,
            "n_successful_folds": group["fold"].nunique(),
            "n_classes": group["n_classes"].mean(),
        }

        for metric in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]:
            record[f"{metric}_mean"] = group[metric].mean()
            record[f"{metric}_std"] = group[metric].std(ddof=0)

        records.append(record)

    summary = pd.DataFrame(records)

    if not summary.empty:
        summary = summary.sort_values(
            ["label_col", "balanced_accuracy_mean", "macro_f1_mean"],
            ascending=[True, False, False],
        )

    return summary


def plot_benchmark(summary: pd.DataFrame) -> None:
    if summary.empty:
        return

    fig = go.Figure()

    for label_col, group in summary.groupby("label_col"):
        group = group.sort_values("balanced_accuracy_mean", ascending=True)

        fig.add_trace(
            go.Bar(
                x=group["balanced_accuracy_mean"],
                y=group["model"],
                orientation="h",
                error_x=dict(
                    type="data",
                    array=group["balanced_accuracy_std"],
                    visible=True,
                ),
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
        title="GTEx V11 classifier benchmark, classical ML vs deep/tabular models",
        xaxis_title="Balanced accuracy",
        yaxis_title="Model",
        template="plotly_white",
        barmode="group",
        margin=dict(l=260, r=60, t=100, b=90),
    )

    save_figure(fig, "gtex_v11_deep_tabular_benchmark_balanced_accuracy")


def write_report(summary: pd.DataFrame, fold_df: pd.DataFrame) -> None:
    report_path = OUTPUT_DIR / "gtex_v11_deep_tabular_benchmark_summary.md"

    lines = []
    lines.append("# GTEx V11 Deep and Tabular Model Benchmark\n")
    lines.append("## Purpose\n")
    lines.append(
        "This benchmark compares classical ML models, dense neural networks, TabNet, a lightweight FT-Transformer-style model, and optional tabular foundation models for predicting GTEx tissue labels from only 9 curated ECM program scores under donor-stratified cross-validation."
    )

    if summary.empty:
        lines.append("\nNo successful model runs were completed.\n")
    else:
        for label_col in sorted(summary["label_col"].unique()):
            subset = summary[summary["label_col"].eq(label_col)]
            subset = subset.sort_values("balanced_accuracy_mean", ascending=False)

            best = subset.iloc[0]

            lines.append(f"\n## Task: {label_col}\n")
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

    failed = fold_df[fold_df["status"].eq("failed")].copy()

    if not failed.empty:
        lines.append("\n## Failed model runs\n")
        failed_summary = (
            failed.groupby(["label_col", "model", "error"])
            .size()
            .reset_index(name="n_failed_folds")
        )

        for row in failed_summary.itertuples():
            lines.append(
                f"- {row.label_col} | {row.model}: {row.n_failed_folds} failed folds. Error: {row.error}"
            )

    lines.append("\n## Interpretation\n")
    lines.append(
        "If XGBoost or ExtraTrees remain best, they should be reported as nonlinear performance benchmarks. If dense or transformer-style models do not outperform tree ensembles, that is not a failure; with only 9 curated ECM program features, classical models are expected to be highly competitive."
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAVED] {report_path}")


def main() -> None:
    ensure_dirs()

    df = load_scores(INPUT_PATH)
    feature_cols = PROGRAM_ORDER

    model_names = [
        "dummy_most_frequent",
        "logistic_regression",
        "extra_trees",
        "random_forest",
        "hist_gradient_boosting",
        "xgboost",
        "dense_mlp",
        "residual_mlp",
        "ft_transformer_lite",
        "tabnet",
        "tabpfn",
        "tabicl",
    ]

    tasks = [
        {"label_col": "tissue_system", "min_samples_per_class": 100, "n_splits": 5},
        {"label_col": "tissue", "min_samples_per_class": 100, "n_splits": 5},
    ]

    all_folds = []
    all_predictions = []

    for task in tasks:
        fold_df, pred_df = run_benchmark(
            df=df,
            label_col=task["label_col"],
            feature_cols=feature_cols,
            model_names=model_names,
            min_samples_per_class=task["min_samples_per_class"],
            n_splits=task["n_splits"],
        )

        all_folds.append(fold_df)

        if not pred_df.empty:
            all_predictions.append(pred_df)

    folds = pd.concat(all_folds, ignore_index=True)
    summary = summarize_folds(folds)

    folds.to_csv(TABLE_DIR / "gtex_v11_deep_tabular_benchmark_fold_metrics.csv", index=False)
    summary.to_csv(TABLE_DIR / "gtex_v11_deep_tabular_benchmark_summary.csv", index=False)

    if all_predictions:
        predictions = pd.concat(all_predictions, ignore_index=True)
        predictions.to_csv(TABLE_DIR / "gtex_v11_deep_tabular_benchmark_predictions.csv", index=False)

    plot_benchmark(summary)
    write_report(summary, folds)

    print("\n[DONE]")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()