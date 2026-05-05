from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from scipy.stats import rankdata

from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    BaggingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ecm_program_atlas.scoring import (
    ProgramGeneSet,
    load_programs_from_curated_table,
)


DEFAULT_PROGRAM_TABLE = Path(
    "results/tables/frozen/combined_nmf_module_annotations_curated_programs.csv"
)

DEFAULT_GTEX_MATRISOME_MATRIX = Path(
    "data/processed/gtex_v11_sample_level/gtex_v11_matrisome_expression_log2.parquet"
)

DEFAULT_GTEX_METADATA = Path(
    "data/processed/gtex_v11_sample_level/gtex_v11_sample_metadata.csv"
)

DEFAULT_GTEX_FULL_EXPRESSION = Path(
    "data/raw/gtex_v11_sample_level/"
    "GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_tpm.parquet"
)

DEFAULT_OUTPUT_DIR = Path("results/revision_classification_controls")


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


@dataclass(frozen=True)
class RepresentationSpec:
    name: str
    family: str
    kind: str
    repeat: int | None = None


class DenseMLP(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.15,
    ):
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
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden_dim: int = 128,
        n_blocks: int = 3,
        dropout: float = 0.15,
    ):
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

    TransformerEncoder layers then model feature interactions.
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


def ensure_dirs(output_dir: Path) -> tuple[Path, Path, Path, Path]:
    table_dir = output_dir / "tables"
    html_dir = output_dir / "figures" / "html"
    png_dir = output_dir / "figures" / "png"
    report_dir = output_dir / "reports"

    for folder in [table_dir, html_dir, png_dir, report_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    return table_dir, html_dir, png_dir, report_dir


def save_figure(
    fig: go.Figure,
    name: str,
    html_dir: Path,
    png_dir: Path,
    width: int = 1450,
    height: int = 850,
) -> None:
    html_path = html_dir / f"{name}.html"
    png_path = png_dir / f"{name}.png"

    fig.update_layout(width=width, height=height)
    fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)

    try:
        fig.write_image(str(png_path), scale=3)
        print(f"[SAVED] {png_path}")
    except Exception as exc:
        print(f"[WARNING] PNG export failed for {name}: {exc}")

    print(f"[SAVED] {html_path}")


def normalize_gene(gene: str) -> str:
    return str(gene).strip().upper()


def load_matrix(path: Path, dataset_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {dataset_name} matrix:\n{path}\n\n"
            "If data live in the old project, pass explicit paths through command-line arguments."
        )

    print(f"[INFO] Loading {dataset_name}: {path}")

    if path.suffix == ".parquet":
        matrix = pd.read_parquet(path)
    else:
        matrix = pd.read_csv(path, index_col=0)

    matrix.index = matrix.index.astype(str)
    matrix.columns = [normalize_gene(col) for col in matrix.columns]
    matrix = matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    print(f"[INFO] {dataset_name} shape: {matrix.shape}")
    return matrix


def load_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing GTEx metadata file:\n{path}")

    metadata = pd.read_csv(path)

    required = ["sample_id", "subject_id", "tissue", "tissue_detail"]
    missing = [col for col in required if col not in metadata.columns]

    if missing:
        raise ValueError(f"GTEx metadata missing columns: {missing}")

    metadata["sample_id"] = metadata["sample_id"].astype(str)
    metadata["subject_id"] = metadata["subject_id"].astype(str)
    metadata["tissue"] = metadata["tissue"].astype(str)
    metadata["tissue_detail"] = metadata["tissue_detail"].astype(str)
    metadata["tissue_system"] = metadata["tissue"].map(TISSUE_SYSTEM_MAP).fillna("Other")

    return metadata


def align_matrix_and_metadata(
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    common = metadata["sample_id"][metadata["sample_id"].isin(matrix.index)].tolist()

    if not common:
        raise ValueError("No overlapping sample IDs between expression matrix and metadata.")

    matrix_aligned = matrix.loc[common].copy()
    metadata_aligned = metadata.set_index("sample_id").loc[common].reset_index()

    return matrix_aligned, metadata_aligned


def program_sizes(programs: Sequence[ProgramGeneSet]) -> list[int]:
    return [len(program.genes) for program in programs]


def sample_random_gene_sets(
    universe: Sequence[str],
    sizes: Sequence[int],
    rng: np.random.Generator,
    prefix: str,
) -> list[ProgramGeneSet]:
    universe = sorted(set(normalize_gene(gene) for gene in universe))
    programs = []

    for i, size in enumerate(sizes, start=1):
        if size > len(universe):
            raise ValueError(
                f"Requested gene-set size {size} exceeds universe size {len(universe)}."
            )

        genes = rng.choice(universe, size=size, replace=False)
        programs.append(
            ProgramGeneSet(
                name=f"{prefix}_{i}",
                genes=tuple(sorted(genes.tolist())),
            )
        )

    return programs


def zscore_train_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    means = train.mean(axis=0)
    stds = train.std(axis=0, ddof=0).replace(0, np.nan)

    train_z = train.sub(means, axis=1).div(stds, axis=1).fillna(0.0)
    test_z = test.sub(means, axis=1).div(stds, axis=1).fillna(0.0)

    return train_z, test_z


def score_gene_sets_foldwise(
    train_matrix: pd.DataFrame,
    test_matrix: pd.DataFrame,
    programs: Sequence[ProgramGeneSet],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Compute gene-set scores using train-fitted gene z-scoring.

    This avoids test-sample leakage in gene scaling.
    """
    train_z, test_z = zscore_train_test(train_matrix, test_matrix)

    train_scores = pd.DataFrame(index=train_matrix.index)
    test_scores = pd.DataFrame(index=test_matrix.index)

    for program in programs:
        genes = [gene for gene in program.genes if gene in train_z.columns]

        if not genes:
            train_scores[program.name] = np.nan
            test_scores[program.name] = np.nan
        else:
            train_scores[program.name] = train_z[genes].mean(axis=1)
            test_scores[program.name] = test_z[genes].mean(axis=1)

    train_scores = train_scores.fillna(0.0)
    test_scores = test_scores.fillna(0.0)

    return train_scores.values, test_scores.values, train_scores.columns.tolist()


def pca_foldwise(
    train_matrix: pd.DataFrame,
    test_matrix: pd.DataFrame,
    n_components: int = 9,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_matrix.values)
    X_test = scaler.transform(test_matrix.values)

    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    feature_names = [f"PC{i + 1}" for i in range(n_components)]

    return X_train_pca, X_test_pca, feature_names


def get_full_expression_gene_symbols(full_expression_path: Path) -> set[str]:
    if not full_expression_path.exists():
        raise FileNotFoundError(f"Missing full GTEx expression file:\n{full_expression_path}")

    print("[INFO] Reading full GTEx gene symbols.")
    expr = pd.read_parquet(full_expression_path, columns=["Description"])
    genes = set(expr["Description"].astype(str).str.strip().str.upper().tolist())
    genes.discard("")
    genes.discard("NAN")

    print(f"[INFO] Full GTEx unique gene symbols: {len(genes)}")
    return genes


def load_full_expression_selected_genes(
    full_expression_path: Path,
    selected_genes: Sequence[str],
    sample_ids: Sequence[str],
) -> pd.DataFrame:
    """
    Load GTEx full expression parquet and return sample × selected genes log2(TPM+1).
    """
    if not full_expression_path.exists():
        raise FileNotFoundError(f"Missing full GTEx expression file:\n{full_expression_path}")

    selected = set(normalize_gene(gene) for gene in selected_genes)
    sample_ids = list(sample_ids)

    print(f"[INFO] Loading full GTEx expression for {len(selected)} selected genes.")
    expr = pd.read_parquet(full_expression_path)

    if "Description" not in expr.columns:
        raise ValueError("Full GTEx expression file must contain a Description column.")

    expr = expr.copy()
    expr["gene_symbol"] = expr["Description"].astype(str).str.strip().str.upper()

    sample_cols = [col for col in expr.columns if str(col).startswith("GTEX-")]
    keep_sample_cols = [col for col in sample_cols if col in sample_ids]

    if not keep_sample_cols:
        raise ValueError("No requested sample IDs found in the full GTEx expression matrix.")

    sub = expr[expr["gene_symbol"].isin(selected)][["gene_symbol"] + keep_sample_cols].copy()

    if sub.empty:
        raise ValueError("No selected non-ECM genes were found in full GTEx expression.")

    grouped = sub.groupby("gene_symbol")[keep_sample_cols].mean()
    sample_gene = grouped.T
    sample_gene.index.name = "sample_id"

    sample_gene = sample_gene.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    sample_gene = np.log2(sample_gene + 1.0)
    sample_gene = sample_gene.loc[sample_ids]

    print(f"[INFO] Full-expression selected matrix shape: {sample_gene.shape}")
    return sample_gene


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }


def train_torch_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_classes: int,
    max_epochs: int = 180,
    batch_size: int = 512,
    patience: int = 25,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    rng = np.random.default_rng(42)
    indices = np.arange(X_train.shape[0])
    rng.shuffle(indices)

    val_size = max(int(0.15 * len(indices)), n_classes)
    val_idx = indices[:val_size]
    tr_idx = indices[val_size:]

    train_ds = TensorDataset(X_train_t[tr_idx], y_train_t[tr_idx])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_x = X_train_t[val_idx].to(device)
    val_y = y_train_t[val_idx].to(device)

    model = model.to(device)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(n_classes),
        y=y_train,
    )
    weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for _epoch in range(max_epochs):
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
            val_loss = loss_fn(model(val_x), val_y).item()

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
        y_pred = model(X_test_t.to(device)).argmax(dim=1).cpu().numpy()

    return y_pred


def fit_predict_model(
    model_name: str,
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_test_raw: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    if model_name == "dummy_most_frequent":
        model = DummyClassifier(strategy="most_frequent")
        model.fit(X_train_raw, y_train)
        return model.predict(X_test_raw)

    if model_name == "logistic_regression":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        model = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="lbfgs",
        )
        model.fit(X_train, y_train)
        return model.predict(X_test)

    if model_name == "extra_trees":
        model = ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train_raw, y_train)
        return model.predict(X_test_raw)

    if model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train_raw, y_train)
        return model.predict(X_test_raw)

    if model_name == "hist_gradient_boosting":
        weights = compute_sample_weight(class_weight="balanced", y=y_train)
        model = HistGradientBoostingClassifier(
            max_iter=400,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            random_state=42,
        )
        model.fit(X_train_raw, y_train, sample_weight=weights)
        return model.predict(X_test_raw)

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
        model.fit(X_train_raw, y_train, sample_weight=weights)
        return model.predict(X_test_raw)

    if model_name == "dense_mlp":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        model = DenseMLP(
            n_features=X_train.shape[1],
            n_classes=n_classes,
            hidden_dim=128,
            dropout=0.15,
        )
        return train_torch_model(model, X_train, y_train, X_test, n_classes)

    if model_name == "residual_mlp":
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
        return train_torch_model(model, X_train, y_train, X_test, n_classes)

    if model_name == "ft_transformer_lite":
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
        return train_torch_model(model, X_train, y_train, X_test, n_classes)

    if model_name == "tabnet":
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier
        except Exception as exc:
            raise RuntimeError(f"pytorch-tabnet is not installed: {exc}")

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
        X_test = scaler.transform(X_test_raw).astype(np.float32)
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
            X_train=X_train,
            y_train=y_train,
            weights=weights,
            max_epochs=160,
            patience=25,
            batch_size=1024,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False,
        )

        return model.predict(X_test).astype(int)

    if model_name == "tabpfn":
        try:
            from tabpfn import TabPFNClassifier
        except Exception as exc:
            raise RuntimeError(f"tabpfn is not installed: {exc}")

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        # TabPFN class-count and sample-count limits are model-dependent.
        if len(np.unique(y_train)) > 10:
            raise RuntimeError(
                "TabPFN skipped because the number of classes exceeds 10."
            )

        if X_train.shape[0] > 10000:
            rng = np.random.default_rng(42)
            idx = rng.choice(X_train.shape[0], size=10000, replace=False)
            X_train = X_train[idx]
            y_train = y_train[idx]

        model = TabPFNClassifier()
        model.fit(X_train, y_train)
        return model.predict(X_test).astype(int)

    if model_name == "tabicl":
        try:
            from tabicl import TabICLClassifier
        except Exception as exc:
            raise RuntimeError(f"tabicl is not installed or API changed: {exc}")

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        model = TabICLClassifier()
        model.fit(X_train, y_train)
        return model.predict(X_test).astype(int)

    raise ValueError(f"Unknown model: {model_name}")


def filter_classes(
    metadata: pd.DataFrame,
    label_col: str,
    min_samples_per_class: int,
) -> pd.DataFrame:
    counts = metadata[label_col].value_counts()
    valid = counts[counts >= min_samples_per_class].index.tolist()
    return metadata[metadata[label_col].isin(valid)].copy()


def run_cv_for_representation(
    representation_name: str,
    representation_family: str,
    model_name: str,
    label_col: str,
    metadata: pd.DataFrame,
    expression_matrix: pd.DataFrame,
    representation_kind: str,
    programs: Sequence[ProgramGeneSet] | None,
    n_splits: int,
    min_samples_per_class: int,
    repeat: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = filter_classes(
        metadata,
        label_col=label_col,
        min_samples_per_class=min_samples_per_class,
    ).copy()

    meta = meta[meta["sample_id"].isin(expression_matrix.index)].copy()

    if meta.empty:
        raise ValueError(f"No samples left for label {label_col} and representation {representation_name}")

    sample_ids = meta["sample_id"].tolist()
    X_expr = expression_matrix.loc[sample_ids].copy()

    y_str = meta[label_col].astype(str).values
    groups = meta["subject_id"].astype(str).values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_str)

    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))

    fold_records = []
    prediction_records = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_expr.values, y, groups), start=1):
        train_samples = [sample_ids[i] for i in train_idx]
        test_samples = [sample_ids[i] for i in test_idx]

        train_matrix = X_expr.loc[train_samples]
        test_matrix = X_expr.loc[test_samples]

        if representation_kind == "gene_set_scores":
            if programs is None:
                raise ValueError("programs must be supplied for gene_set_scores.")

            X_train, X_test, feature_names = score_gene_sets_foldwise(
                train_matrix=train_matrix,
                test_matrix=test_matrix,
                programs=programs,
            )

        elif representation_kind == "pca9":
            X_train, X_test, feature_names = pca_foldwise(
                train_matrix=train_matrix,
                test_matrix=test_matrix,
                n_components=9,
            )

        else:
            raise ValueError(f"Unsupported representation kind: {representation_kind}")

        try:
            y_pred = fit_predict_model(
                model_name=model_name,
                X_train_raw=X_train,
                y_train=y[train_idx],
                X_test_raw=X_test,
                n_classes=len(label_encoder.classes_),
            )

            metrics = evaluate(y[test_idx], y_pred)
            status = "ok"
            error = ""

        except Exception as exc:
            print(f"[FAILED] {representation_name} | {label_col} | {model_name} | fold={fold_idx}: {exc}")
            y_pred = np.full(shape=len(test_idx), fill_value=0, dtype=int)
            metrics = {
                "accuracy": np.nan,
                "balanced_accuracy": np.nan,
                "macro_f1": np.nan,
                "weighted_f1": np.nan,
            }
            status = "failed"
            error = str(exc)

        fold_records.append(
            {
                "representation": representation_name,
                "representation_family": representation_family,
                "representation_kind": representation_kind,
                "model": model_name,
                "label_col": label_col,
                "repeat": repeat,
                "fold": fold_idx,
                "status": status,
                "error": error,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_train_subjects": len(np.unique(groups[train_idx])),
                "n_test_subjects": len(np.unique(groups[test_idx])),
                "n_classes": len(label_encoder.classes_),
                "n_features": X_train.shape[1],
                **metrics,
            }
        )

        if status == "ok":
            pred_meta = meta.iloc[test_idx][
                ["sample_id", "subject_id", "tissue", "tissue_detail", "tissue_system"]
            ].copy()
            pred_meta["representation"] = representation_name
            pred_meta["representation_family"] = representation_family
            pred_meta["model"] = model_name
            pred_meta["label_col"] = label_col
            pred_meta["repeat"] = repeat
            pred_meta["fold"] = fold_idx
            pred_meta["y_true"] = label_encoder.inverse_transform(y[test_idx])
            pred_meta["y_pred"] = label_encoder.inverse_transform(y_pred.astype(int))
            prediction_records.append(pred_meta)

    fold_df = pd.DataFrame(fold_records)

    if prediction_records:
        pred_df = pd.concat(prediction_records, ignore_index=True)
    else:
        pred_df = pd.DataFrame()

    return fold_df, pred_df


def summarize_fold_metrics(fold_df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]

    ok = fold_df[fold_df["status"].eq("ok")].copy()

    records = []

    group_cols = [
        "representation",
        "representation_family",
        "representation_kind",
        "model",
        "label_col",
    ]

    for keys, group in ok.groupby(group_cols):
        record = dict(zip(group_cols, keys))
        record["n_folds"] = group["fold"].nunique()
        record["mean_n_features"] = group["n_features"].mean()

        for metric in metrics:
            record[f"{metric}_mean"] = group[metric].mean()
            record[f"{metric}_std"] = group[metric].std(ddof=0)

        records.append(record)

    summary = pd.DataFrame(records)

    if not summary.empty:
        summary = summary.sort_values(
            ["label_col", "model", "balanced_accuracy_mean"],
            ascending=[True, True, False],
        )

    return summary


def summarize_random_families(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize performance across representation families.

    Example:
        random_matrisome has multiple repeated representations.
        This function summarizes their mean, std, min, and max performance.

    Important:
        Input summary_df already has columns such as:
            accuracy_mean
            balanced_accuracy_mean
            macro_f1_mean
            weighted_f1_mean

        We convert these into cleaner family-level columns:
            accuracy_family_mean
            accuracy_family_std
            accuracy_family_min
            accuracy_family_max
    """
    if summary_df.empty:
        return pd.DataFrame()

    metric_map = {
        "accuracy": "accuracy_mean",
        "balanced_accuracy": "balanced_accuracy_mean",
        "macro_f1": "macro_f1_mean",
        "weighted_f1": "weighted_f1_mean",
    }

    records = []

    for (family, model, label_col), group in summary_df.groupby(
        ["representation_family", "model", "label_col"]
    ):
        record = {
            "representation_family": family,
            "model": model,
            "label_col": label_col,
            "n_representations": group["representation"].nunique(),
            "representations": "; ".join(sorted(group["representation"].astype(str).unique())),
        }

        for metric_name, col in metric_map.items():
            if col not in group.columns:
                continue

            values = pd.to_numeric(group[col], errors="coerce").dropna()

            if values.empty:
                record[f"{metric_name}_family_mean"] = np.nan
                record[f"{metric_name}_family_std"] = np.nan
                record[f"{metric_name}_family_min"] = np.nan
                record[f"{metric_name}_family_max"] = np.nan
            else:
                record[f"{metric_name}_family_mean"] = values.mean()
                record[f"{metric_name}_family_std"] = values.std(ddof=0)
                record[f"{metric_name}_family_min"] = values.min()
                record[f"{metric_name}_family_max"] = values.max()

        records.append(record)

    family_summary = pd.DataFrame(records)

    if not family_summary.empty:
        family_summary = family_summary.sort_values(
            ["label_col", "model", "balanced_accuracy_family_mean"],
            ascending=[True, True, False],
        )

    return family_summary

def plot_summary(summary_df: pd.DataFrame, html_dir: Path, png_dir: Path) -> None:
    if summary_df.empty:
        return

    for label_col in sorted(summary_df["label_col"].unique()):
        for model in sorted(summary_df["model"].unique()):
            sub = summary_df[
                (summary_df["label_col"].eq(label_col))
                & (summary_df["model"].eq(model))
            ].copy()

            if sub.empty:
                continue

            sub = sub.sort_values("balanced_accuracy_mean", ascending=True)

            fig = go.Figure(
                data=go.Bar(
                    x=sub["balanced_accuracy_mean"],
                    y=sub["representation"],
                    orientation="h",
                    error_x=dict(
                        type="data",
                        array=sub["balanced_accuracy_std"],
                        visible=True,
                    ),
                    customdata=sub[
                        [
                            "representation_family",
                            "macro_f1_mean",
                            "accuracy_mean",
                            "mean_n_features",
                        ]
                    ],
                    hovertemplate=(
                        "Representation: %{y}<br>"
                        "Family: %{customdata[0]}<br>"
                        "Balanced accuracy: %{x:.3f}<br>"
                        "Macro-F1: %{customdata[1]:.3f}<br>"
                        "Accuracy: %{customdata[2]:.3f}<br>"
                        "Features: %{customdata[3]:.0f}<extra></extra>"
                    ),
                )
            )

            fig.update_layout(
                title=(
                    f"R2 representation control benchmark<br>"
                    f"<sup>Task: {label_col}, model: {model}</sup>"
                ),
                xaxis_title="Balanced accuracy",
                yaxis_title="Representation",
                template="plotly_white",
                margin=dict(l=340, r=60, t=100, b=90),
            )

            name = f"r2_representation_benchmark_{label_col}_{model}"
            save_figure(fig, name=name, html_dir=html_dir, png_dir=png_dir)


def plot_family_summary(family_summary: pd.DataFrame, html_dir: Path, png_dir: Path) -> None:
    if family_summary.empty:
        print("[SKIP] Empty family summary.")
        return

    required = [
        "label_col",
        "model",
        "representation_family",
        "balanced_accuracy_family_mean",
        "balanced_accuracy_family_std",
        "balanced_accuracy_family_max",
        "balanced_accuracy_family_min",
        "macro_f1_family_mean",
        "n_representations",
    ]

    missing = [col for col in required if col not in family_summary.columns]
    if missing:
        print(f"[SKIP] Family summary missing required columns: {missing}")
        print("Available columns:")
        print(family_summary.columns.tolist())
        return

    for label_col in sorted(family_summary["label_col"].unique()):
        for model in sorted(family_summary["model"].unique()):
            sub = family_summary[
                (family_summary["label_col"].eq(label_col))
                & (family_summary["model"].eq(model))
            ].copy()

            if sub.empty:
                continue

            sub = sub.sort_values("balanced_accuracy_family_mean", ascending=True)

            fig = go.Figure(
                data=go.Bar(
                    x=sub["balanced_accuracy_family_mean"],
                    y=sub["representation_family"],
                    orientation="h",
                    error_x=dict(
                        type="data",
                        array=sub["balanced_accuracy_family_std"],
                        visible=True,
                    ),
                    customdata=sub[
                        [
                            "balanced_accuracy_family_max",
                            "balanced_accuracy_family_min",
                            "macro_f1_family_mean",
                            "n_representations",
                        ]
                    ],
                    hovertemplate=(
                        "Family: %{y}<br>"
                        "Mean balanced accuracy: %{x:.3f}<br>"
                        "Max balanced accuracy: %{customdata[0]:.3f}<br>"
                        "Min balanced accuracy: %{customdata[1]:.3f}<br>"
                        "Mean macro-F1: %{customdata[2]:.3f}<br>"
                        "N representations: %{customdata[3]}<extra></extra>"
                    ),
                )
            )

            fig.update_layout(
                title=(
                    f"R2 representation-family benchmark<br>"
                    f"<sup>Task: {label_col}, model: {model}</sup>"
                ),
                xaxis_title="Balanced accuracy",
                yaxis_title="Representation family",
                template="plotly_white",
                margin=dict(l=300, r=60, t=100, b=90),
            )

            name = f"r2_representation_family_benchmark_{label_col}_{model}"
            save_figure(fig, name=name, html_dir=html_dir, png_dir=png_dir)


def write_report(
    summary_df: pd.DataFrame,
    family_summary_df: pd.DataFrame,
    fold_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    report_path = output_dir / "reports" / "r2_representation_control_summary.md"

    lines = []
    lines.append("# R2 Representation Control Benchmark\n")
    lines.append("## Purpose\n")
    lines.append(
        "This analysis tests whether the nine curated ECM programs provide stronger or more interpretable donor-stratified tissue classification than alternative compact representations."
    )

    lines.append("\n## Representations compared\n")
    lines.append("- Curated 9 ECM programs\n")
    lines.append("- 9 PCA components from the full Matrisome expression matrix\n")
    lines.append("- Random Matrisome gene-set programs matched to curated program sizes\n")
    lines.append("- Random non-ECM gene-set programs matched to curated program sizes, if full GTEx expression is supplied\n")

    lines.append("\n## Models compared\n")
    models = sorted(fold_df["model"].dropna().unique().tolist())
    for model in models:
        lines.append(f"- {model}")

    lines.append("\n## Best results by task and model\n")

    if summary_df.empty:
        lines.append("No successful benchmark results were generated.\n")
    else:
        for (label_col, model), group in summary_df.groupby(["label_col", "model"]):
            best = group.sort_values("balanced_accuracy_mean", ascending=False).iloc[0]
            lines.append(
                f"- **{label_col}, {model}**: best representation = "
                f"{best['representation']} "
                f"({best['representation_family']}), balanced accuracy = "
                f"{best['balanced_accuracy_mean']:.3f} ± {best['balanced_accuracy_std']:.3f}, "
                f"macro-F1 = {best['macro_f1_mean']:.3f}."
            )

    failed = fold_df[fold_df["status"].eq("failed")].copy()
    if not failed.empty:
        lines.append("\n## Failed model runs\n")
        failed_summary = (
            failed.groupby(["label_col", "model", "representation_family", "error"])
            .size()
            .reset_index(name="n_failed_folds")
        )
        for row in failed_summary.itertuples():
            lines.append(
                f"- {row.label_col} | {row.model} | {row.representation_family}: "
                f"{row.n_failed_folds} failed folds. Error: {row.error}"
            )

    lines.append("\n## Interpretation guidance\n")
    lines.append(
        "If curated ECM programs outperform random Matrisome and random non-ECM representations, this supports their biological specificity. If PCA performs similarly or better, the nine curated programs should be framed primarily as an interpretable representation rather than the most predictive representation."
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAVED] {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--program-table", type=Path, default=DEFAULT_PROGRAM_TABLE)
    parser.add_argument("--gtex-matrisome-matrix", type=Path, default=DEFAULT_GTEX_MATRISOME_MATRIX)
    parser.add_argument("--gtex-metadata", type=Path, default=DEFAULT_GTEX_METADATA)
    parser.add_argument("--gtex-full-expression", type=Path, default=DEFAULT_GTEX_FULL_EXPRESSION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--n-random-repeats", type=int, default=10)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--min-samples-per-class", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=42)

    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "logistic_regression",
            "xgboost",
            "extra_trees",
            "dense_mlp",
            "residual_mlp",
            "ft_transformer_lite",
        ],
        help=(
            "Models to run. Options: dummy_most_frequent logistic_regression extra_trees "
            "random_forest hist_gradient_boosting xgboost dense_mlp residual_mlp "
            "ft_transformer_lite tabnet tabpfn tabicl"
        ),
    )

    parser.add_argument(
        "--skip-non-ecm",
        action="store_true",
        help="Skip random non-ECM controls. Useful if full GTEx expression is unavailable.",
    )

    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: fewer random repeats, fewer folds, fewer models.",
    )

    args = parser.parse_args()

    if args.fast:
        args.n_random_repeats = min(args.n_random_repeats, 2)
        args.n_splits = min(args.n_splits, 3)
        args.models = ["logistic_regression", "xgboost"]

    table_dir, html_dir, png_dir, report_dir = ensure_dirs(args.output_dir)

    rng = np.random.default_rng(args.random_seed)

    programs = load_programs_from_curated_table(
        str(args.program_table),
        program_col="ecm_program_curated",
        genes_col="top_genes",
    )

    lookup = {program.name: program for program in programs}
    programs = [lookup[name] for name in PROGRAM_ORDER if name in lookup]
    sizes = [len(program.genes) for program in programs]

    matrisome_matrix = load_matrix(args.gtex_matrisome_matrix, "GTEx V11 Matrisome expression")
    metadata = load_metadata(args.gtex_metadata)
    matrisome_matrix, metadata = align_matrix_and_metadata(matrisome_matrix, metadata)

    matrisome_genes = sorted(matrisome_matrix.columns.tolist())

    random_matrisome_program_sets: list[list[ProgramGeneSet]] = []
    for repeat in range(args.n_random_repeats):
        random_matrisome_program_sets.append(
            sample_random_gene_sets(
                universe=matrisome_genes,
                sizes=sizes,
                rng=rng,
                prefix=f"random_matrisome_r{repeat + 1}",
            )
        )

    random_non_ecm_program_sets: list[list[ProgramGeneSet]] = []
    non_ecm_matrix = None

    if not args.skip_non_ecm:
        full_genes = get_full_expression_gene_symbols(args.gtex_full_expression)
        non_ecm_universe = sorted(full_genes.difference(set(matrisome_genes)))

        print(f"[INFO] Non-ECM candidate genes: {len(non_ecm_universe)}")

        all_non_ecm_selected: set[str] = set()

        for repeat in range(args.n_random_repeats):
            repeat_programs = sample_random_gene_sets(
                universe=non_ecm_universe,
                sizes=sizes,
                rng=rng,
                prefix=f"random_non_ecm_r{repeat + 1}",
            )
            random_non_ecm_program_sets.append(repeat_programs)

            for program in repeat_programs:
                all_non_ecm_selected.update(program.genes)

        non_ecm_matrix = load_full_expression_selected_genes(
            full_expression_path=args.gtex_full_expression,
            selected_genes=sorted(all_non_ecm_selected),
            sample_ids=metadata["sample_id"].tolist(),
        )

    all_fold_records = []
    all_prediction_records = []

    tasks = ["tissue_system", "tissue"]

    for label_col in tasks:
        for model_name in args.models:
            print(f"\n[TASK] {label_col} | model={model_name}")

            representation_jobs = [
                {
                    "representation_name": "curated_9_ecm_programs",
                    "representation_family": "curated_ecm",
                    "representation_kind": "gene_set_scores",
                    "expression_matrix": matrisome_matrix,
                    "programs": programs,
                    "repeat": None,
                },
                {
                    "representation_name": "matrisome_pca9",
                    "representation_family": "matrisome_pca",
                    "representation_kind": "pca9",
                    "expression_matrix": matrisome_matrix,
                    "programs": None,
                    "repeat": None,
                },
            ]

            for repeat, random_programs in enumerate(random_matrisome_program_sets, start=1):
                representation_jobs.append(
                    {
                        "representation_name": f"random_matrisome_programs_r{repeat}",
                        "representation_family": "random_matrisome",
                        "representation_kind": "gene_set_scores",
                        "expression_matrix": matrisome_matrix,
                        "programs": random_programs,
                        "repeat": repeat,
                    }
                )

            if non_ecm_matrix is not None:
                for repeat, random_programs in enumerate(random_non_ecm_program_sets, start=1):
                    representation_jobs.append(
                        {
                            "representation_name": f"random_non_ecm_programs_r{repeat}",
                            "representation_family": "random_non_ecm",
                            "representation_kind": "gene_set_scores",
                            "expression_matrix": non_ecm_matrix,
                            "programs": random_programs,
                            "repeat": repeat,
                        }
                    )

            for job in representation_jobs:
                print(f"  [REPRESENTATION] {job['representation_name']}")

                fold_df, pred_df = run_cv_for_representation(
                    representation_name=job["representation_name"],
                    representation_family=job["representation_family"],
                    model_name=model_name,
                    label_col=label_col,
                    metadata=metadata,
                    expression_matrix=job["expression_matrix"],
                    representation_kind=job["representation_kind"],
                    programs=job["programs"],
                    n_splits=args.n_splits,
                    min_samples_per_class=args.min_samples_per_class,
                    repeat=job["repeat"],
                )

                all_fold_records.append(fold_df)

                if not pred_df.empty:
                    all_prediction_records.append(pred_df)

    fold_metrics = pd.concat(all_fold_records, ignore_index=True)
    predictions = pd.concat(all_prediction_records, ignore_index=True) if all_prediction_records else pd.DataFrame()

    summary = summarize_fold_metrics(fold_metrics)
    family_summary = summarize_random_families(summary)

    fold_metrics.to_csv(table_dir / "r2_representation_control_fold_metrics.csv", index=False)
    predictions.to_csv(table_dir / "r2_representation_control_predictions.csv", index=False)
    summary.to_csv(table_dir / "r2_representation_control_summary.csv", index=False)
    family_summary.to_csv(table_dir / "r2_representation_family_summary.csv", index=False)

    plot_summary(summary, html_dir=html_dir, png_dir=png_dir)
    plot_family_summary(family_summary, html_dir=html_dir, png_dir=png_dir)

    write_report(
        summary_df=summary,
        family_summary_df=family_summary,
        fold_df=fold_metrics,
        output_dir=args.output_dir,
    )

    metadata_json = {
        "program_table": str(args.program_table),
        "gtex_matrisome_matrix": str(args.gtex_matrisome_matrix),
        "gtex_metadata": str(args.gtex_metadata),
        "gtex_full_expression": str(args.gtex_full_expression),
        "skip_non_ecm": args.skip_non_ecm,
        "n_random_repeats": args.n_random_repeats,
        "n_splits": args.n_splits,
        "min_samples_per_class": args.min_samples_per_class,
        "models": args.models,
        "fast": args.fast,
    }

    with (args.output_dir / "r2_representation_control_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata_json, f, indent=2)

    print("\n[DONE]")
    print(f"Output folder: {args.output_dir}")
    print(f"Tables: {table_dir}")
    print(f"Reports: {report_dir}")
    print(f"Figures HTML: {html_dir}")
    print(f"Figures PNG: {png_dir}")


if __name__ == "__main__":
    main()