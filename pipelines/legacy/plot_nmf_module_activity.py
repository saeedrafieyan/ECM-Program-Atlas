from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


DEFAULT_BASE_DIR = Path("outputs/latent_baseline_embeddings")


def safe_name(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def find_embedding_dirs(base_dir: Path) -> List[Path]:
    """
    Find all embedding directories that contain NMF outputs.
    Expected structure:
        outputs/latent_baseline_embeddings/{dataset}/{feature_set}/
    """
    dirs = []

    for nmf_file in base_dir.glob("*/*/nmf_sample_coordinates.csv"):
        dirs.append(nmf_file.parent)

    return sorted(dirs)


def load_nmf_sample_coordinates(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing NMF sample coordinates file: {path}")

    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)

    nmf_cols = [col for col in df.columns if str(col).startswith("NMF")]

    if not nmf_cols:
        raise ValueError(f"No NMF columns found in {path}")

    df = df[nmf_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return df


def load_nmf_top_genes(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing NMF top genes file: {path}")

    df = pd.read_csv(path)

    required = ["component", "rank", "gene", "weight"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"NMF top genes file is missing columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    return df


def get_top_genes_text(
    top_genes_df: pd.DataFrame,
    component: str,
    n_genes: int = 10,
) -> str:
    subset = (
        top_genes_df[top_genes_df["component"].astype(str).eq(component)]
        .sort_values("rank")
        .head(n_genes)
    )

    if subset.empty:
        return ""

    return ", ".join(subset["gene"].astype(str).tolist())


def make_module_activity_table(
    nmf_df: pd.DataFrame,
    top_genes_df: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    records = []

    for component in nmf_df.columns:
        top_samples = nmf_df[component].sort_values(ascending=False).head(top_n)
        top_genes_text = get_top_genes_text(top_genes_df, component, n_genes=10)

        for rank, (sample, score) in enumerate(top_samples.items(), start=1):
            records.append(
                {
                    "component": component,
                    "sample": sample,
                    "rank": rank,
                    "activity_score": float(score),
                    "top_genes": top_genes_text,
                }
            )

    return pd.DataFrame(records)


def plot_component_barplot(
    module_activity_df: pd.DataFrame,
    component: str,
    dataset_name: str,
    feature_set_name: str,
    output_path: Path,
) -> None:
    df = module_activity_df[module_activity_df["component"].eq(component)].copy()
    df = df.sort_values("activity_score", ascending=True)

    top_genes = df["top_genes"].iloc[0] if not df.empty else ""

    fig = px.bar(
        df,
        x="activity_score",
        y="sample",
        orientation="h",
        hover_data={
            "sample": True,
            "rank": True,
            "activity_score": ":.4f",
            "top_genes": True,
        },
        title=(
            f"{component} activity, top samples<br>"
            f"<sup>Dataset: {dataset_name} | Feature set: {feature_set_name}<br>"
            f"Top genes: {top_genes}</sup>"
        ),
        labels={
            "activity_score": "NMF activity score",
            "sample": "Sample",
        },
        template="plotly_white",
    )

    fig.update_layout(
        width=1000,
        height=max(550, 28 * len(df)),
        yaxis=dict(title=""),
        xaxis=dict(title="NMF activity score"),
    )

    fig.write_html(str(output_path), include_plotlyjs="cdn", full_html=True)


def plot_module_activity_heatmap(
    nmf_df: pd.DataFrame,
    dataset_name: str,
    feature_set_name: str,
    output_path: Path,
) -> None:
    """
    Heatmap of samples x NMF components.
    Values are raw NMF activity scores.
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=nmf_df.values,
            x=nmf_df.columns.tolist(),
            y=nmf_df.index.tolist(),
            colorscale="Viridis",
            colorbar=dict(title="Activity"),
            hovertemplate=(
                "Sample: %{y}<br>"
                "Component: %{x}<br>"
                "Activity: %{z:.4f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=(
            f"NMF module activity heatmap<br>"
            f"<sup>Dataset: {dataset_name} | Feature set: {feature_set_name}</sup>"
        ),
        width=1000,
        height=max(650, 18 * nmf_df.shape[0]),
        xaxis_title="NMF component",
        yaxis_title="Sample",
        template="plotly_white",
    )

    fig.write_html(str(output_path), include_plotlyjs="cdn", full_html=True)


def plot_component_scatter_matrix(
    nmf_df: pd.DataFrame,
    dataset_name: str,
    feature_set_name: str,
    output_path: Path,
    max_components: int = 5,
) -> None:
    """
    Optional overview plot for first few NMF components.
    This can become large, so only first max_components are used.
    """
    selected_cols = nmf_df.columns[:max_components].tolist()
    plot_df = nmf_df[selected_cols].copy()
    plot_df.insert(0, "sample", nmf_df.index.astype(str))

    fig = px.scatter_matrix(
        plot_df,
        dimensions=selected_cols,
        hover_name="sample",
        title=(
            f"NMF component scatter matrix<br>"
            f"<sup>Dataset: {dataset_name} | Feature set: {feature_set_name}</sup>"
        ),
        template="plotly_white",
    )

    fig.update_traces(diagonal_visible=False)
    fig.update_layout(
        width=1100,
        height=1100,
    )

    fig.write_html(str(output_path), include_plotlyjs="cdn", full_html=True)


def summarize_modules(
    module_activity_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Save one row per component:
    - top samples
    - top genes
    """
    records = []

    for component, group in module_activity_df.groupby("component"):
        group = group.sort_values("rank")
        top_samples = "; ".join(group["sample"].astype(str).tolist())
        top_genes = group["top_genes"].iloc[0]

        records.append(
            {
                "component": component,
                "top_samples": top_samples,
                "top_genes": top_genes,
            }
        )

    pd.DataFrame(records).to_csv(output_path, index=False)


def process_embedding_dir(
    embedding_dir: Path,
    top_n: int,
    scatter_matrix_components: int,
) -> None:
    dataset_name = embedding_dir.parent.name
    feature_set_name = embedding_dir.name

    print("=" * 100)
    print(f"[PROCESSING] Dataset: {dataset_name} | Feature set: {feature_set_name}")

    nmf_sample_path = embedding_dir / "nmf_sample_coordinates.csv"
    nmf_top_genes_path = embedding_dir / "nmf_top_genes.csv"

    nmf_df = load_nmf_sample_coordinates(nmf_sample_path)
    top_genes_df = load_nmf_top_genes(nmf_top_genes_path)

    output_dir = embedding_dir / "nmf_module_activity"
    barplot_dir = output_dir / "barplots"

    output_dir.mkdir(parents=True, exist_ok=True)
    barplot_dir.mkdir(parents=True, exist_ok=True)

    module_activity_df = make_module_activity_table(
        nmf_df=nmf_df,
        top_genes_df=top_genes_df,
        top_n=top_n,
    )

    module_activity_df.to_csv(
        output_dir / "nmf_module_activity_top_samples.csv",
        index=False,
    )

    summarize_modules(
        module_activity_df=module_activity_df,
        output_path=output_dir / "nmf_module_summary.csv",
    )

    plot_module_activity_heatmap(
        nmf_df=nmf_df,
        dataset_name=dataset_name,
        feature_set_name=feature_set_name,
        output_path=output_dir / "nmf_module_activity_heatmap.html",
    )

    plot_component_scatter_matrix(
        nmf_df=nmf_df,
        dataset_name=dataset_name,
        feature_set_name=feature_set_name,
        output_path=output_dir / "nmf_component_scatter_matrix.html",
        max_components=scatter_matrix_components,
    )

    for component in nmf_df.columns:
        plot_component_barplot(
            module_activity_df=module_activity_df,
            component=component,
            dataset_name=dataset_name,
            feature_set_name=feature_set_name,
            output_path=barplot_dir / f"{safe_name(component)}_top_samples.html",
        )

    print(f"[SAVED] {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-dir",
        type=str,
        default=str(DEFAULT_BASE_DIR),
        help="Base latent embedding output directory.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name, e.g., rna_tissue_consensus.",
    )

    parser.add_argument(
        "--feature-set",
        type=str,
        default=None,
        help="Feature set folder name, e.g., core_matrisome.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all dataset/feature-set embedding folders.",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Number of top samples to show per NMF component.",
    )

    parser.add_argument(
        "--scatter-matrix-components",
        type=int,
        default=5,
        help="Number of NMF components to include in scatter matrix.",
    )

    args = parser.parse_args()

    base_dir = Path(args.base_dir)

    if args.all:
        embedding_dirs = find_embedding_dirs(base_dir)

        if not embedding_dirs:
            raise FileNotFoundError(
                f"No embedding directories with nmf_sample_coordinates.csv found under {base_dir}"
            )

    else:
        if args.dataset is None or args.feature_set is None:
            raise ValueError(
                "Provide --dataset and --feature-set, or use --all."
            )

        embedding_dirs = [
            base_dir / args.dataset / args.feature_set
        ]

    for embedding_dir in embedding_dirs:
        if not embedding_dir.exists():
            print(f"[SKIP] Missing embedding directory: {embedding_dir}")
            continue

        try:
            process_embedding_dir(
                embedding_dir=embedding_dir,
                top_n=args.top_n,
                scatter_matrix_components=args.scatter_matrix_components,
            )
        except Exception as exc:
            print(f"[ERROR] Failed to process {embedding_dir}: {exc}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()