from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px

from sklearn.decomposition import PCA, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, MinMaxScaler


try:
    import umap

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


PROJECT_ROOT = Path(".")
CONFIG_DIR = PROJECT_ROOT / "configs"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "latent_baseline_embeddings"

DATASET_MANIFEST = CONFIG_DIR / "dataset_manifest.csv"
FEATURE_SET_MANIFEST = CONFIG_DIR / "feature_sets.csv"
MATRISOME_FILE = RAW_DIR / "matrisome" / "human_matrisome.xlsx"


def safe_name(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def normalize_gene_symbol(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .replace({"": np.nan, "NAN": np.nan, "NONE": np.nan})
    )


def load_dataset_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {path}")

    df = pd.read_csv(path)

    required = ["dataset", "path", "sample_col", "expr_col", "modality", "use_now"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Dataset manifest is missing columns: {missing}")

    df = df[df["use_now"].astype(str).str.lower().eq("yes")].copy()

    if df.empty:
        raise ValueError("No datasets marked as use_now=yes in dataset_manifest.csv")

    return df


def load_feature_set_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Feature-set manifest not found: {path}")

    df = pd.read_csv(path)

    required = ["feature_set", "matrisome_level", "matrisome_value", "use_now"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Feature-set manifest is missing columns: {missing}")

    df = df[df["use_now"].astype(str).str.lower().isin(["yes", "true", "1"])].copy()

    if df.empty:
        raise ValueError("No feature sets marked as use_now=yes in feature_sets.csv")

    return df


def load_matrisome_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Matrisome file not found: {path}")

    df = pd.read_excel(path, header=1)
    df.columns = [str(col).strip() for col in df.columns]
    df = df.dropna(how="all")

    required = ["Matrisome Division", "Matrisome Category", "Gene Symbol"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Matrisome file is missing columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    df["_gene_symbol_upper"] = normalize_gene_symbol(df["Gene Symbol"])
    df = df.dropna(subset=["_gene_symbol_upper"])
    df = df.drop_duplicates(subset=["_gene_symbol_upper"])

    return df


def load_expression_matrix(
    path: Path,
    sample_col: str,
    expr_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        matrix_log2:
            Non-negative log2(expression + 1) matrix.
            Use this for NMF.

        matrix_zscore:
            Gene-wise z-scored log2(expression + 1) matrix.
            Use this for PCA and UMAP.
    """
    if not path.exists():
        raise FileNotFoundError(f"Expression file not found: {path}")

    df = pd.read_csv(path, sep="\t", compression="zip")

    required = ["Gene name", sample_col, expr_col]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing columns in {path}: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    df = df.copy()
    df["_gene_symbol_upper"] = normalize_gene_symbol(df["Gene name"])
    df[expr_col] = pd.to_numeric(df[expr_col], errors="coerce")
    df = df.dropna(subset=["_gene_symbol_upper", sample_col, expr_col])

    matrix = df.pivot_table(
        index=sample_col,
        columns="_gene_symbol_upper",
        values=expr_col,
        aggfunc="mean",
    )

    matrix = matrix.fillna(0.0)
    matrix = matrix.sort_index()
    matrix = matrix.reindex(sorted(matrix.columns), axis=1)

    variable_genes = matrix.var(axis=0) > 0
    matrix = matrix.loc[:, variable_genes]

    matrix_log2 = np.log2(matrix + 1.0)

    scaler = StandardScaler()
    z_values = scaler.fit_transform(matrix_log2.values)

    matrix_zscore = pd.DataFrame(
        z_values,
        index=matrix_log2.index,
        columns=matrix_log2.columns,
    )

    return matrix_log2, matrix_zscore


def get_feature_set_genes(
    feature_set_row: pd.Series,
    matrisome_df: pd.DataFrame,
    available_genes: set[str],
) -> List[str]:
    feature_set = str(feature_set_row["feature_set"])
    level = str(feature_set_row["matrisome_level"])
    value = str(feature_set_row["matrisome_value"])

    if level.lower() == "all":
        genes = matrisome_df["_gene_symbol_upper"].tolist()

    elif level in matrisome_df.columns:
        genes = matrisome_df.loc[
            matrisome_df[level].astype(str).eq(value),
            "_gene_symbol_upper",
        ].tolist()

    else:
        raise ValueError(
            f"Unknown matrisome level for feature set {feature_set}: {level}. "
            f"Available Matrisome columns: {matrisome_df.columns.tolist()}"
        )

    genes = sorted(set(genes).intersection(available_genes))

    return genes


def write_plotly_figure(
    fig,
    html_path: Path,
    save_static_png: bool = False,
) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)

    fig.write_html(
        str(html_path),
        include_plotlyjs="cdn",
        full_html=True,
    )

    if save_static_png:
        png_path = html_path.with_suffix(".png")
        try:
            fig.write_image(str(png_path), scale=2)
        except Exception as exc:
            print(
                f"[WARNING] Could not save static PNG for {html_path.name}. "
                f"Install kaleido if needed. Error: {exc}"
            )


def plot_2d_coordinates(
    coords_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    output_path: Path,
    dataset_name: str,
    feature_set_name: str,
    method_name: str,
    explained_x: float | None = None,
    explained_y: float | None = None,
    save_static_png: bool = False,
) -> None:
    """
    Create an interactive 2D Plotly scatter plot.

    Sample names are stored in a dedicated 'sample' column and shown only on hover,
    preventing label overlap in dense plots.
    """
    plot_df = coords_df.copy()

    # Robustly preserve sample names regardless of index name
    plot_df.insert(0, "sample", coords_df.index.astype(str))
    plot_df = plot_df.reset_index(drop=True)

    if explained_x is not None:
        x_label = f"{x_col} ({explained_x * 100:.2f}% variance)"
    else:
        x_label = x_col

    if explained_y is not None:
        y_label = f"{y_col} ({explained_y * 100:.2f}% variance)"
    else:
        y_label = y_col

    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        hover_name="sample",
        hover_data={
            x_col: ":.4f",
            y_col: ":.4f",
        },
        title=title,
        labels={
            x_col: x_label,
            y_col: y_label,
        },
        template="plotly_white",
    )

    fig.update_traces(
        marker=dict(size=9, opacity=0.85),
        mode="markers",
    )

    fig.update_layout(
        title=dict(
            text=(
                f"{title}<br>"
                f"<sup>Dataset: {dataset_name} | Feature set: {feature_set_name} | "
                f"Method: {method_name}</sup>"
            ),
            x=0.5,
        ),
        width=950,
        height=750,
        hovermode="closest",
        legend_title_text="",
    )

    fig.update_xaxes(showgrid=True, zeroline=True)
    fig.update_yaxes(showgrid=True, zeroline=True)

    write_plotly_figure(
        fig=fig,
        html_path=output_path,
        save_static_png=save_static_png,
    )


def run_pca(
    matrix_zscore: pd.DataFrame,
    output_dir: Path,
    n_components: int,
    dataset_name: str,
    feature_set_name: str,
    save_static_png: bool,
) -> Dict[str, float]:
    n_components = min(n_components, matrix_zscore.shape[0], matrix_zscore.shape[1])

    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(matrix_zscore.values)

    coord_cols = [f"PC{i + 1}" for i in range(n_components)]
    coords_df = pd.DataFrame(coords, index=matrix_zscore.index, columns=coord_cols)

    explained_df = pd.DataFrame(
        {
            "component": coord_cols,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )

    coords_df.to_csv(output_dir / "pca_coordinates.csv")
    explained_df.to_csv(output_dir / "pca_explained_variance.csv", index=False)

    save_pca_loading_genes(
        matrix=matrix_zscore,
        pca=pca,
        output_path=output_dir / "pca_top_loading_genes.csv",
        n_top=30,
    )

    if n_components >= 2:
        plot_2d_coordinates(
            coords_df=coords_df,
            x_col="PC1",
            y_col="PC2",
            title="PCA latent space",
            output_path=output_dir / "pca_pc1_pc2.html",
            dataset_name=dataset_name,
            feature_set_name=feature_set_name,
            method_name="PCA",
            explained_x=pca.explained_variance_ratio_[0],
            explained_y=pca.explained_variance_ratio_[1],
            save_static_png=save_static_png,
        )

    summary = {
        "pca_n_components": n_components,
        "pca_pc1_variance": float(pca.explained_variance_ratio_[0]),
        "pca_pc1_pc2_variance": float(np.sum(pca.explained_variance_ratio_[:2])),
        "pca_pc1_to_pc5_variance": float(
            np.sum(pca.explained_variance_ratio_[: min(5, n_components)])
        ),
    }

    return summary


def save_pca_loading_genes(
    matrix: pd.DataFrame,
    pca: PCA,
    output_path: Path,
    n_top: int = 30,
) -> None:
    gene_names = np.array(matrix.columns)
    records = []

    n_components = min(5, pca.components_.shape[0])

    for component_idx in range(n_components):
        loadings = pca.components_[component_idx]

        positive_idx = np.argsort(loadings)[-n_top:][::-1]
        negative_idx = np.argsort(loadings)[:n_top]

        for rank, idx in enumerate(positive_idx, start=1):
            records.append(
                {
                    "component": f"PC{component_idx + 1}",
                    "direction": "positive",
                    "rank": rank,
                    "gene": gene_names[idx],
                    "loading": float(loadings[idx]),
                    "abs_loading": float(abs(loadings[idx])),
                }
            )

        for rank, idx in enumerate(negative_idx, start=1):
            records.append(
                {
                    "component": f"PC{component_idx + 1}",
                    "direction": "negative",
                    "rank": rank,
                    "gene": gene_names[idx],
                    "loading": float(loadings[idx]),
                    "abs_loading": float(abs(loadings[idx])),
                }
            )

    pd.DataFrame(records).to_csv(output_path, index=False)


def run_umap(
    matrix_zscore: pd.DataFrame,
    output_dir: Path,
    n_neighbors: int,
    min_dist: float,
    dataset_name: str,
    feature_set_name: str,
    save_static_png: bool,
) -> Dict[str, float]:
    if not HAS_UMAP:
        print("[WARNING] umap-learn is not installed. Skipping UMAP.")
        return {"umap_completed": 0}

    n_neighbors = min(n_neighbors, max(2, matrix_zscore.shape[0] - 1))

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=42,
    )

    coords = reducer.fit_transform(matrix_zscore.values)

    coords_df = pd.DataFrame(
        coords,
        index=matrix_zscore.index,
        columns=["UMAP1", "UMAP2"],
    )

    coords_df.to_csv(output_dir / "umap_coordinates.csv")

    plot_2d_coordinates(
        coords_df=coords_df,
        x_col="UMAP1",
        y_col="UMAP2",
        title="UMAP latent space",
        output_path=output_dir / "umap.html",
        dataset_name=dataset_name,
        feature_set_name=feature_set_name,
        method_name="UMAP",
        save_static_png=save_static_png,
    )

    return {
        "umap_completed": 1,
        "umap_n_neighbors": n_neighbors,
        "umap_min_dist": min_dist,
    }


def run_nmf(
    matrix_log2: pd.DataFrame,
    output_dir: Path,
    n_components: int,
    dataset_name: str,
    feature_set_name: str,
    save_static_png: bool,
) -> Dict[str, float]:
    """
    NMF requires non-negative values, so use log2(expression + 1), not z-scored data.
    """
    n_components = min(n_components, matrix_log2.shape[0], matrix_log2.shape[1])

    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(matrix_log2.values)

    nmf = NMF(
        n_components=n_components,
        init="nndsvda",
        random_state=42,
        max_iter=3000,
    )

    sample_factors = nmf.fit_transform(x_scaled)
    gene_factors = nmf.components_

    sample_cols = [f"NMF{i + 1}" for i in range(n_components)]

    sample_df = pd.DataFrame(
        sample_factors,
        index=matrix_log2.index,
        columns=sample_cols,
    )

    gene_df = pd.DataFrame(
        gene_factors.T,
        index=matrix_log2.columns,
        columns=sample_cols,
    )

    sample_df.to_csv(output_dir / "nmf_sample_coordinates.csv")
    gene_df.to_csv(output_dir / "nmf_gene_weights.csv")

    save_nmf_top_genes(
        gene_weights=gene_df,
        output_path=output_dir / "nmf_top_genes.csv",
        n_top=30,
    )

    if n_components >= 2:
        plot_2d_coordinates(
            coords_df=sample_df,
            x_col="NMF1",
            y_col="NMF2",
            title="NMF latent space",
            output_path=output_dir / "nmf_component1_component2.html",
            dataset_name=dataset_name,
            feature_set_name=feature_set_name,
            method_name="NMF",
            save_static_png=save_static_png,
        )

    return {
        "nmf_n_components": n_components,
        "nmf_reconstruction_error": float(nmf.reconstruction_err_),
    }


def save_nmf_top_genes(
    gene_weights: pd.DataFrame,
    output_path: Path,
    n_top: int,
) -> None:
    """
    Save top genes for all NMF components, not only the first five.
    """
    records = []

    for component in gene_weights.columns:
        top_genes = gene_weights[component].sort_values(ascending=False).head(n_top)

        for rank, (gene, weight) in enumerate(top_genes.items(), start=1):
            records.append(
                {
                    "component": component,
                    "rank": rank,
                    "gene": gene,
                    "weight": float(weight),
                }
            )

    pd.DataFrame(records).to_csv(output_path, index=False)


def compute_nearest_neighbors(
    matrix_zscore: pd.DataFrame,
    output_path: Path,
    k: int = 10,
) -> None:
    similarity = cosine_similarity(matrix_zscore.values)

    sim_df = pd.DataFrame(
        similarity,
        index=matrix_zscore.index,
        columns=matrix_zscore.index,
    )

    sim_df.to_csv(output_path.parent / "sample_cosine_similarity_matrix.csv")

    records = []

    for sample in sim_df.index:
        neighbors = sim_df.loc[sample].drop(index=sample).sort_values(ascending=False)

        for rank, (neighbor, score) in enumerate(neighbors.head(k).items(), start=1):
            records.append(
                {
                    "sample": sample,
                    "rank": rank,
                    "nearest_sample": neighbor,
                    "cosine_similarity": float(score),
                }
            )

    pd.DataFrame(records).to_csv(output_path, index=False)


def save_matrix_metadata(
    output_dir: Path,
    dataset_name: str,
    feature_set_name: str,
    genes: List[str],
    matrix: pd.DataFrame,
) -> None:
    metadata = {
        "dataset": dataset_name,
        "feature_set": feature_set_name,
        "n_samples": matrix.shape[0],
        "n_genes": len(genes),
    }

    pd.DataFrame([metadata]).to_csv(output_dir / "metadata.csv", index=False)
    pd.DataFrame({"gene": genes}).to_csv(output_dir / "genes_used.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca-components", type=int, default=10)
    parser.add_argument("--nmf-components", type=int, default=10)
    parser.add_argument("--umap-neighbors", type=int, default=8)
    parser.add_argument("--umap-min-dist", type=float, default=0.2)
    parser.add_argument(
        "--save-static-png",
        action="store_true",
        help="Also save PNG versions of Plotly plots. Requires kaleido.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_manifest = load_dataset_manifest(DATASET_MANIFEST)
    feature_manifest = load_feature_set_manifest(FEATURE_SET_MANIFEST)
    matrisome_df = load_matrisome_metadata(MATRISOME_FILE)

    global_records = []

    for _, dataset_row in dataset_manifest.iterrows():
        dataset_name = str(dataset_row["dataset"])
        dataset_path = Path(str(dataset_row["path"]))
        sample_col = str(dataset_row["sample_col"])
        expr_col = str(dataset_row["expr_col"])

        print("\n" + "#" * 100)
        print(f"[DATASET] {dataset_name}")

        matrix_log2, matrix_zscore = load_expression_matrix(
            path=dataset_path,
            sample_col=sample_col,
            expr_col=expr_col,
        )

        available_genes = set(matrix_log2.columns)

        print(f"[INFO] Matrix shape: {matrix_log2.shape}")

        for _, feature_row in feature_manifest.iterrows():
            feature_set_name = str(feature_row["feature_set"])
            feature_output_name = safe_name(feature_set_name)

            genes = get_feature_set_genes(
                feature_set_row=feature_row,
                matrisome_df=matrisome_df,
                available_genes=available_genes,
            )

            if len(genes) < 2:
                print(f"[SKIP] {dataset_name} | {feature_set_name}: too few genes.")
                continue

            print(f"[FEATURE SET] {feature_set_name} | genes={len(genes)}")

            output_dir = OUTPUT_DIR / dataset_name / feature_output_name
            output_dir.mkdir(parents=True, exist_ok=True)

            feature_log2 = matrix_log2.loc[:, genes]
            feature_zscore = matrix_zscore.loc[:, genes]

            save_matrix_metadata(
                output_dir=output_dir,
                dataset_name=dataset_name,
                feature_set_name=feature_set_name,
                genes=genes,
                matrix=feature_log2,
            )

            feature_log2.to_csv(output_dir / "input_matrix_log2.csv")
            feature_zscore.to_csv(output_dir / "input_matrix_log2_zscore.csv")

            compute_nearest_neighbors(
                matrix_zscore=feature_zscore,
                output_path=output_dir / "nearest_neighbors.csv",
                k=10,
            )

            summary = {
                "dataset": dataset_name,
                "feature_set": feature_set_name,
                "n_samples": feature_zscore.shape[0],
                "n_genes": feature_zscore.shape[1],
            }

            pca_summary = run_pca(
                matrix_zscore=feature_zscore,
                output_dir=output_dir,
                n_components=args.pca_components,
                dataset_name=dataset_name,
                feature_set_name=feature_set_name,
                save_static_png=args.save_static_png,
            )
            summary.update(pca_summary)

            nmf_summary = run_nmf(
                matrix_log2=feature_log2,
                output_dir=output_dir,
                n_components=args.nmf_components,
                dataset_name=dataset_name,
                feature_set_name=feature_set_name,
                save_static_png=args.save_static_png,
            )
            summary.update(nmf_summary)

            umap_summary = run_umap(
                matrix_zscore=feature_zscore,
                output_dir=output_dir,
                n_neighbors=args.umap_neighbors,
                min_dist=args.umap_min_dist,
                dataset_name=dataset_name,
                feature_set_name=feature_set_name,
                save_static_png=args.save_static_png,
            )
            summary.update(umap_summary)

            pd.DataFrame([summary]).to_csv(output_dir / "summary.csv", index=False)
            global_records.append(summary)

    global_summary = pd.DataFrame(global_records)
    global_summary.to_csv(
        OUTPUT_DIR / "latent_embedding_global_summary.csv",
        index=False,
    )

    print("\n[DONE]")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print(f"Global summary: {OUTPUT_DIR / 'latent_embedding_global_summary.csv'}")


if __name__ == "__main__":
    main()