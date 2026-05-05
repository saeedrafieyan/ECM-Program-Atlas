from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from scipy.cluster.hierarchy import linkage, cophenet
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform

from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler


DEFAULT_INPUT_BASE = Path("results/latent_baseline_embeddings/rna_tissue_consensus")
DEFAULT_CURATED_PROGRAM_FILE = Path(
    "results/tables/frozen/combined_nmf_module_annotations_curated_programs.csv"
)
DEFAULT_OUTPUT_DIR = Path("results/revision_nmf_stability")


DEFAULT_FEATURE_SETS = [
    "all_matrisome",
    "core_matrisome",
    "ecm_glycoproteins",
    "proteoglycans",
    "collagens",
]


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
    width: int = 1300,
    height: int = 800,
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


def split_comma_list(value: str) -> List[str]:
    if pd.isna(value):
        return []

    return [
        normalize_gene(item)
        for item in str(value).split(",")
        if item.strip()
    ]


def load_feature_matrix(input_base: Path, feature_set: str) -> pd.DataFrame:
    path = input_base / feature_set / "input_matrix_log2.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing matrix for feature set '{feature_set}':\n{path}\n\n"
            "Expected file structure:\n"
            "input_base/feature_set/input_matrix_log2.csv\n\n"
            "If the file exists in the old project, pass:\n"
            "--input-base E:/Projects/ECM/ecm_latent_space/outputs/latent_baseline_embeddings/rna_tissue_consensus"
        )

    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    df.columns = [normalize_gene(col) for col in df.columns]
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if df.empty:
        raise ValueError(f"Input matrix is empty: {path}")

    return df


def load_curated_programs(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing curated ECM program file: {path}")

    df = pd.read_csv(path)

    required = [
        "feature_set",
        "component",
        "ecm_program_curated",
        "top_genes",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Curated program file is missing columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    records = []

    for program, group in df.groupby("ecm_program_curated"):
        genes: list[str] = []
        modules: list[str] = []

        for row in group.itertuples():
            genes.extend(split_comma_list(row.top_genes))
            modules.append(f"{row.feature_set}:{row.component}")

        unique_genes = sorted(set(genes))

        records.append(
            {
                "ecm_program": program,
                "n_reference_modules": len(modules),
                "reference_modules": "; ".join(modules),
                "n_program_genes": len(unique_genes),
                "program_genes": ", ".join(unique_genes),
            }
        )

    out = pd.DataFrame(records)

    out["ecm_program"] = pd.Categorical(
        out["ecm_program"],
        categories=PROGRAM_ORDER,
        ordered=True,
    )

    out = out.sort_values("ecm_program")

    return out


def minmax_scale_matrix(matrix: pd.DataFrame) -> np.ndarray:
    scaler = MinMaxScaler()
    return scaler.fit_transform(matrix.values)


def get_top_genes_from_components(
    components: np.ndarray,
    gene_names: list[str],
    top_n: int,
) -> Dict[str, list[str]]:
    top_gene_sets: Dict[str, list[str]] = {}

    for component_idx in range(components.shape[0]):
        weights = components[component_idx]
        idx = np.argsort(weights)[::-1][:top_n]
        genes = [gene_names[i] for i in idx]
        top_gene_sets[f"NMF{component_idx + 1}"] = genes

    return top_gene_sets


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0

    union = a.union(b)

    if not union:
        return 0.0

    return len(a.intersection(b)) / len(union)


def overlap_coefficient(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0

    return len(a.intersection(b)) / min(len(a), len(b))


def matched_component_similarity(
    gene_sets_a: Dict[str, list[str]],
    gene_sets_b: Dict[str, list[str]],
) -> float:
    comps_a = list(gene_sets_a.keys())
    comps_b = list(gene_sets_b.keys())

    n = min(len(comps_a), len(comps_b))

    if n == 0:
        return np.nan

    sim = np.zeros((len(comps_a), len(comps_b)), dtype=float)

    for i, comp_a in enumerate(comps_a):
        a = set(gene_sets_a[comp_a])

        for j, comp_b in enumerate(comps_b):
            b = set(gene_sets_b[comp_b])
            sim[i, j] = jaccard(a, b)

    row_ind, col_ind = linear_sum_assignment(-sim)
    matched = sim[row_ind, col_ind]

    return float(np.mean(matched[:n]))


def consensus_cophenetic_from_assignments(assignments: list[np.ndarray]) -> float:
    """
    Build consensus sample co-clustering matrix from NMF sample-component assignments.
    """
    if len(assignments) < 2:
        return np.nan

    n_samples = len(assignments[0])
    consensus = np.zeros((n_samples, n_samples), dtype=float)

    for labels in assignments:
        same = labels[:, None] == labels[None, :]
        consensus += same.astype(float)

    consensus /= len(assignments)

    distance = 1.0 - consensus
    np.fill_diagonal(distance, 0.0)

    condensed = squareform(distance, checks=False)

    if np.allclose(condensed, condensed[0]):
        return np.nan

    linkage_matrix = linkage(condensed, method="average")
    coph_corr, _ = cophenet(linkage_matrix, condensed)

    return float(coph_corr)


def nmf_sparsity(matrix: np.ndarray) -> float:
    """
    Hoyer-like sparsity over component weights, averaged across components.
    """
    if matrix.size == 0:
        return np.nan

    values = []

    for row in matrix:
        n = len(row)

        if n <= 1:
            values.append(np.nan)
            continue

        l1 = np.sum(np.abs(row))
        l2 = np.sqrt(np.sum(row ** 2))

        if l2 == 0:
            values.append(np.nan)
            continue

        sparsity = (np.sqrt(n) - (l1 / l2)) / (np.sqrt(n) - 1)
        values.append(sparsity)

    return float(np.nanmean(values))


def run_single_nmf(
    X: np.ndarray,
    k: int,
    seed: int,
    max_iter: int,
) -> Tuple[NMF, np.ndarray, np.ndarray]:
    model = NMF(
        n_components=k,
        init="nndsvda",
        random_state=seed,
        max_iter=max_iter,
        solver="cd",
        beta_loss="frobenius",
    )

    W = model.fit_transform(X)
    H = model.components_

    return model, W, H


def evaluate_curated_program_recovery(
    feature_set: str,
    k: int,
    seed: int,
    component_gene_sets: Dict[str, list[str]],
    curated_programs: pd.DataFrame,
    min_overlap_genes: int,
    min_overlap_coeff: float,
) -> list[dict]:
    records: list[dict] = []

    for program_row in curated_programs.itertuples():
        program = str(program_row.ecm_program)
        program_genes = set(split_comma_list(program_row.program_genes))

        best_component = None
        best_overlap_coeff = -1.0
        best_jaccard = -1.0
        best_overlap_genes: list[str] = []

        for component, genes in component_gene_sets.items():
            comp_genes = set(genes)
            overlap = sorted(program_genes.intersection(comp_genes))

            oc = overlap_coefficient(program_genes, comp_genes)
            jc = jaccard(program_genes, comp_genes)

            if (oc > best_overlap_coeff) or (
                oc == best_overlap_coeff and jc > best_jaccard
            ):
                best_component = component
                best_overlap_coeff = oc
                best_jaccard = jc
                best_overlap_genes = overlap

        recovered = (
            len(best_overlap_genes) >= min_overlap_genes
            and best_overlap_coeff >= min_overlap_coeff
        )

        records.append(
            {
                "feature_set": feature_set,
                "k": k,
                "seed": seed,
                "ecm_program": program,
                "best_component": best_component,
                "n_program_genes": len(program_genes),
                "n_overlap_genes": len(best_overlap_genes),
                "best_overlap_coefficient": best_overlap_coeff,
                "best_jaccard": best_jaccard,
                "overlap_genes": ", ".join(best_overlap_genes),
                "recovered": recovered,
            }
        )

    return records


def analyze_feature_set(
    input_base: Path,
    feature_set: str,
    curated_programs: pd.DataFrame,
    k_values: list[int],
    seeds: list[int],
    top_n_genes: int,
    max_iter: int,
    min_overlap_genes: int,
    min_overlap_coeff: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("=" * 100)
    print(f"[FEATURE SET] {feature_set}")

    matrix = load_feature_matrix(input_base=input_base, feature_set=feature_set)
    X = minmax_scale_matrix(matrix)
    gene_names = list(matrix.columns)

    fro_norm = np.linalg.norm(X, ord="fro")

    metrics_records: list[dict] = []
    recovery_records: list[dict] = []
    failed_records: list[dict] = []

    for k in k_values:
        print(f"[INFO] {feature_set} | k={k}")

        run_results = []

        for seed in seeds:
            try:
                model, W, H = run_single_nmf(
                    X=X,
                    k=k,
                    seed=seed,
                    max_iter=max_iter,
                )

                component_gene_sets = get_top_genes_from_components(
                    components=H,
                    gene_names=gene_names,
                    top_n=top_n_genes,
                )

                sample_assignments = np.argmax(W, axis=1)

                run_results.append(
                    {
                        "seed": seed,
                        "model": model,
                        "W": W,
                        "H": H,
                        "component_gene_sets": component_gene_sets,
                        "sample_assignments": sample_assignments,
                    }
                )

                recovery_records.extend(
                    evaluate_curated_program_recovery(
                        feature_set=feature_set,
                        k=k,
                        seed=seed,
                        component_gene_sets=component_gene_sets,
                        curated_programs=curated_programs,
                        min_overlap_genes=min_overlap_genes,
                        min_overlap_coeff=min_overlap_coeff,
                    )
                )

            except Exception as exc:
                print(f"[FAILED] {feature_set} | k={k} | seed={seed}: {exc}")
                failed_records.append(
                    {
                        "feature_set": feature_set,
                        "k": k,
                        "seed": seed,
                        "error": str(exc),
                    }
                )

        if not run_results:
            continue

        reconstruction_errors = [
            float(result["model"].reconstruction_err_)
            for result in run_results
        ]

        normalized_errors = [
            err / fro_norm if fro_norm > 0 else np.nan
            for err in reconstruction_errors
        ]

        component_sparsities = [
            nmf_sparsity(result["H"])
            for result in run_results
        ]

        pairwise_stability = []

        for i in range(len(run_results)):
            for j in range(i + 1, len(run_results)):
                sim = matched_component_similarity(
                    run_results[i]["component_gene_sets"],
                    run_results[j]["component_gene_sets"],
                )
                pairwise_stability.append(sim)

        assignments = [result["sample_assignments"] for result in run_results]
        coph = consensus_cophenetic_from_assignments(assignments)

        recovery_for_k = pd.DataFrame(
            [
                r
                for r in recovery_records
                if r["feature_set"] == feature_set and r["k"] == k
            ]
        )

        if not recovery_for_k.empty:
            mean_recovered_programs = (
                recovery_for_k.groupby("seed")["recovered"]
                .sum()
                .mean()
            )
            mean_recovery_fraction = (
                recovery_for_k.groupby("seed")["recovered"]
                .mean()
                .mean()
            )
        else:
            mean_recovered_programs = np.nan
            mean_recovery_fraction = np.nan

        metrics_records.append(
            {
                "feature_set": feature_set,
                "k": k,
                "n_samples": matrix.shape[0],
                "n_genes": matrix.shape[1],
                "n_seeds_successful": len(run_results),
                "mean_reconstruction_error": float(np.mean(reconstruction_errors)),
                "std_reconstruction_error": float(np.std(reconstruction_errors, ddof=0)),
                "mean_normalized_reconstruction_error": float(np.mean(normalized_errors)),
                "std_normalized_reconstruction_error": float(np.std(normalized_errors, ddof=0)),
                "mean_component_sparsity": float(np.nanmean(component_sparsities)),
                "mean_top_gene_jaccard_stability": float(np.nanmean(pairwise_stability))
                if pairwise_stability
                else np.nan,
                "std_top_gene_jaccard_stability": float(np.nanstd(pairwise_stability, ddof=0))
                if pairwise_stability
                else np.nan,
                "consensus_cophenetic": coph,
                "mean_recovered_programs_per_seed": float(mean_recovered_programs),
                "mean_recovery_fraction": float(mean_recovery_fraction),
            }
        )

    metrics_df = pd.DataFrame(metrics_records)
    recovery_df = pd.DataFrame(recovery_records)
    failed_df = pd.DataFrame(failed_records)

    return metrics_df, recovery_df, failed_df


def summarize_program_recovery(recovery_df: pd.DataFrame) -> pd.DataFrame:
    if recovery_df.empty:
        return pd.DataFrame()

    summary = (
        recovery_df
        .groupby(["feature_set", "k", "ecm_program"])
        .agg(
            recovery_rate=("recovered", "mean"),
            mean_overlap_coefficient=("best_overlap_coefficient", "mean"),
            mean_jaccard=("best_jaccard", "mean"),
            mean_n_overlap_genes=("n_overlap_genes", "mean"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )

    return summary


def make_rank_recommendation(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()

    records = []

    for feature_set, group in metrics_df.groupby("feature_set"):
        g = group.copy()

        err = g["mean_normalized_reconstruction_error"]
        err_score = 1 - ((err - err.min()) / (err.max() - err.min() + 1e-12))

        stability = g["mean_top_gene_jaccard_stability"].fillna(0.0)
        cophenetic = g["consensus_cophenetic"].fillna(0.0)
        recovery = g["mean_recovery_fraction"].fillna(0.0)

        combined = (
            0.25 * err_score
            + 0.30 * stability
            + 0.20 * cophenetic
            + 0.25 * recovery
        )

        g["rank_selection_score"] = combined

        best = g.sort_values("rank_selection_score", ascending=False).iloc[0]

        records.append(
            {
                "feature_set": feature_set,
                "recommended_k": int(best["k"]),
                "rank_selection_score": float(best["rank_selection_score"]),
                "mean_normalized_reconstruction_error": float(
                    best["mean_normalized_reconstruction_error"]
                ),
                "mean_top_gene_jaccard_stability": float(
                    best["mean_top_gene_jaccard_stability"]
                ),
                "consensus_cophenetic": float(best["consensus_cophenetic"])
                if not pd.isna(best["consensus_cophenetic"])
                else np.nan,
                "mean_recovery_fraction": float(best["mean_recovery_fraction"]),
                "mean_recovered_programs_per_seed": float(
                    best["mean_recovered_programs_per_seed"]
                ),
            }
        )

    return pd.DataFrame(records)


def plot_metric_line(
    metrics_df: pd.DataFrame,
    metric: str,
    title: str,
    y_label: str,
    name: str,
    html_dir: Path,
    png_dir: Path,
) -> None:
    if metrics_df.empty or metric not in metrics_df.columns:
        return

    fig = go.Figure()

    for feature_set, group in metrics_df.groupby("feature_set"):
        group = group.sort_values("k")

        fig.add_trace(
            go.Scatter(
                x=group["k"],
                y=group[metric],
                mode="lines+markers",
                name=feature_set,
                hovertemplate=(
                    "Feature set: %{fullData.name}<br>"
                    "k: %{x}<br>"
                    f"{y_label}: " + "%{y:.4f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="NMF rank k",
        yaxis_title=y_label,
        template="plotly_white",
        margin=dict(l=90, r=60, t=100, b=90),
    )

    save_figure(fig, name=name, html_dir=html_dir, png_dir=png_dir)


def plot_program_recovery_heatmap(
    recovery_summary: pd.DataFrame,
    html_dir: Path,
    png_dir: Path,
) -> None:
    if recovery_summary.empty:
        return

    for feature_set, group in recovery_summary.groupby("feature_set"):
        matrix = group.pivot_table(
            index="ecm_program",
            columns="k",
            values="recovery_rate",
            fill_value=0.0,
            aggfunc="mean",
        )

        matrix = matrix.loc[[program for program in PROGRAM_ORDER if program in matrix.index]]

        fig = go.Figure(
            data=go.Heatmap(
                z=matrix.values,
                x=matrix.columns.tolist(),
                y=matrix.index.tolist(),
                text=[[f"{value:.2f}" for value in row] for row in matrix.values],
                texttemplate="%{text}",
                colorscale="Viridis",
                colorbar=dict(title="Recovery rate"),
                hovertemplate=(
                    "Program: %{y}<br>"
                    "k: %{x}<br>"
                    "Recovery rate: %{z:.2f}<extra></extra>"
                ),
            )
        )

        fig.update_layout(
            title=f"Curated ECM program recovery across NMF ranks<br><sup>{feature_set}</sup>",
            xaxis_title="NMF rank k",
            yaxis_title="ECM program",
            template="plotly_white",
            margin=dict(l=330, r=60, t=100, b=90),
        )

        safe_name = feature_set.replace("/", "_").replace(" ", "_")
        save_figure(
            fig,
            name=f"nmf_program_recovery_heatmap_{safe_name}",
            html_dir=html_dir,
            png_dir=png_dir,
            width=1300,
            height=850,
        )


def write_report(
    metrics_df: pd.DataFrame,
    recommendation_df: pd.DataFrame,
    report_dir: Path,
) -> None:
    report_path = report_dir / "nmf_rank_stability_summary.md"

    lines: list[str] = []

    lines.append("# NMF Rank and Stability Analysis\n")
    lines.append("## Purpose\n")
    lines.append(
        "This analysis evaluates whether NMF-derived ECM modules are stable across "
        "ranks and random seeds, and whether curated ECM programs can be recovered "
        "across a range of NMF ranks."
    )

    lines.append("\n## Metrics\n")
    lines.append("- **Normalized reconstruction error:** lower values indicate better matrix reconstruction.\n")
    lines.append("- **Top-gene Jaccard stability:** matched similarity of top component genes across random seeds.\n")
    lines.append("- **Consensus cophenetic correlation:** stability of sample co-clustering across seeds.\n")
    lines.append("- **Program recovery fraction:** fraction of curated ECM programs recovered by NMF components at a given rank.\n")

    lines.append("\n## Recommended ranks by composite stability score\n")

    if recommendation_df.empty:
        lines.append("No rank recommendations were generated.\n")
    else:
        for row in recommendation_df.itertuples():
            lines.append(
                f"- **{row.feature_set}**: recommended k = {row.recommended_k}; "
                f"score = {row.rank_selection_score:.3f}; "
                f"recovery fraction = {row.mean_recovery_fraction:.3f}; "
                f"top-gene stability = {row.mean_top_gene_jaccard_stability:.3f}."
            )

    lines.append("\n## Interpretation guidance\n")
    lines.append(
        "The curated ECM programs should not be described as arising from one arbitrary NMF run. "
        "They should be described as recurring biological programs consolidated from NMF modules "
        "across Matrisome feature spaces and evaluated through cross-rank recovery, external "
        "reproducibility, MatrisomeDB protein support, GTEx donor-level prediction, Tabula Sapiens "
        "cell-type source mapping, and rank-based scoring robustness."
    )

    lines.append("\n## Manuscript wording recommendation\n")
    lines.append(
        "Use: 'NMF modules were evaluated across ranks and random seeds, and recurrent modules "
        "were consolidated into curated ECM programs based on gene composition, tissue enrichment, "
        "and cross-dataset recovery.'"
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAVED] {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-base",
        type=Path,
        default=DEFAULT_INPUT_BASE,
        help="Directory containing feature-set folders with input_matrix_log2.csv.",
    )
    parser.add_argument(
        "--curated-program-file",
        type=Path,
        default=DEFAULT_CURATED_PROGRAM_FILE,
        help="CSV file with curated ECM program module annotations.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=DEFAULT_FEATURE_SETS,
    )
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=20)
    parser.add_argument("--n-seeds", type=int, default=30)
    parser.add_argument("--top-n-genes", type=int, default=30)
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--min-overlap-genes", type=int, default=4)
    parser.add_argument("--min-overlap-coeff", type=float, default=0.13)

    args = parser.parse_args()

    table_dir, html_dir, png_dir, report_dir = ensure_dirs(args.output_dir)

    curated_programs = load_curated_programs(args.curated_program_file)

    k_values = list(range(args.k_min, args.k_max + 1))
    seeds = list(range(args.n_seeds))

    all_metrics = []
    all_recovery = []
    all_failed = []

    for feature_set in args.feature_sets:
        metrics_df, recovery_df, failed_df = analyze_feature_set(
            input_base=args.input_base,
            feature_set=feature_set,
            curated_programs=curated_programs,
            k_values=k_values,
            seeds=seeds,
            top_n_genes=args.top_n_genes,
            max_iter=args.max_iter,
            min_overlap_genes=args.min_overlap_genes,
            min_overlap_coeff=args.min_overlap_coeff,
        )

        all_metrics.append(metrics_df)
        all_recovery.append(recovery_df)
        all_failed.append(failed_df)

    metrics = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    recovery = pd.concat(all_recovery, ignore_index=True) if all_recovery else pd.DataFrame()
    failed = pd.concat(all_failed, ignore_index=True) if all_failed else pd.DataFrame()

    recovery_summary = summarize_program_recovery(recovery)
    recommendation = make_rank_recommendation(metrics)

    metrics.to_csv(table_dir / "nmf_rank_stability_metrics.csv", index=False)
    recovery.to_csv(table_dir / "nmf_curated_program_recovery_raw.csv", index=False)
    recovery_summary.to_csv(table_dir / "nmf_curated_program_recovery_summary.csv", index=False)
    recommendation.to_csv(table_dir / "nmf_rank_recommendation_summary.csv", index=False)
    failed.to_csv(table_dir / "nmf_failed_runs.csv", index=False)

    plot_metric_line(
        metrics,
        metric="mean_normalized_reconstruction_error",
        title="NMF normalized reconstruction error across ranks",
        y_label="Normalized reconstruction error",
        name="nmf_normalized_reconstruction_error",
        html_dir=html_dir,
        png_dir=png_dir,
    )

    plot_metric_line(
        metrics,
        metric="mean_top_gene_jaccard_stability",
        title="NMF top-gene component stability across random seeds",
        y_label="Mean top-gene Jaccard stability",
        name="nmf_top_gene_jaccard_stability",
        html_dir=html_dir,
        png_dir=png_dir,
    )

    plot_metric_line(
        metrics,
        metric="consensus_cophenetic",
        title="NMF sample-assignment consensus cophenetic correlation",
        y_label="Consensus cophenetic correlation",
        name="nmf_consensus_cophenetic",
        html_dir=html_dir,
        png_dir=png_dir,
    )

    plot_metric_line(
        metrics,
        metric="mean_recovery_fraction",
        title="Curated ECM program recovery across NMF ranks",
        y_label="Mean recovery fraction",
        name="nmf_curated_program_recovery_fraction",
        html_dir=html_dir,
        png_dir=png_dir,
    )

    plot_program_recovery_heatmap(
        recovery_summary,
        html_dir=html_dir,
        png_dir=png_dir,
    )

    write_report(
        metrics_df=metrics,
        recommendation_df=recommendation,
        report_dir=report_dir,
    )

    metadata = {
        "input_base": str(args.input_base),
        "curated_program_file": str(args.curated_program_file),
        "k_min": args.k_min,
        "k_max": args.k_max,
        "n_seeds": args.n_seeds,
        "top_n_genes": args.top_n_genes,
        "max_iter": args.max_iter,
        "min_overlap_genes": args.min_overlap_genes,
        "min_overlap_coeff": args.min_overlap_coeff,
        "feature_sets": args.feature_sets,
    }

    with (args.output_dir / "nmf_rank_stability_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n[DONE]")
    print(f"Output folder: {args.output_dir}")
    print(f"Tables: {table_dir}")
    print(f"Figures HTML: {html_dir}")
    print(f"Figures PNG: {png_dir}")
    print(f"Report: {report_dir / 'nmf_rank_stability_summary.md'}")


if __name__ == "__main__":
    main()