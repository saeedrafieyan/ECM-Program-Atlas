from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.stats import rankdata, spearmanr
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler

from ecm_program_atlas.scoring import ProgramGeneSet, load_programs_from_curated_table


DEFAULT_INPUT_BASE = Path(
    "results/latent_baseline_embeddings/rna_tissue_consensus"
)

DEFAULT_CURATED_PROGRAM_FILE = Path(
    "results/tables/frozen/combined_nmf_module_annotations_curated_programs.csv"
)

DEFAULT_OUTPUT_DIR = Path("results/revision_module_consolidation")


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


DEFAULT_RECOMMENDED_RANKS = {
    "all_matrisome": 20,
    "core_matrisome": 18,
    "ecm_glycoproteins": 18,
    "proteoglycans": 20,
    "collagens": 20,
}


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


def split_gene_string(value: str) -> list[str]:
    if pd.isna(value):
        return []

    return [
        normalize_gene(item)
        for item in str(value).split(",")
        if item.strip()
    ]


def load_matrix(input_base: Path, feature_set: str) -> pd.DataFrame:
    path = input_base / feature_set / "input_matrix_log2.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing input matrix:\n{path}\n\n"
            "Pass --input-base pointing to the old latent_baseline_embeddings/rna_tissue_consensus folder if needed."
        )

    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    df.columns = [normalize_gene(col) for col in df.columns]
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return df


def load_programs(path: Path) -> list[ProgramGeneSet]:
    if not path.exists():
        raise FileNotFoundError(f"Missing curated program table:\n{path}")

    programs = load_programs_from_curated_table(
        str(path),
        program_col="ecm_program_curated",
        genes_col="top_genes",
    )

    lookup = {program.name: program for program in programs}
    ordered = [lookup[name] for name in PROGRAM_ORDER if name in lookup]

    if len(ordered) != len(PROGRAM_ORDER):
        missing = sorted(set(PROGRAM_ORDER).difference(lookup))
        raise ValueError(f"Missing curated programs: {missing}")

    return ordered


def minmax_scale_matrix(matrix: pd.DataFrame) -> np.ndarray:
    return MinMaxScaler().fit_transform(matrix.values)


def normalize_vector_0_1(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo = np.nanmin(x)
    hi = np.nanmax(x)

    if hi == lo:
        return np.zeros_like(x)

    return (x - lo) / (hi - lo)


def top_items(values: np.ndarray, labels: Sequence[str], n: int) -> list[str]:
    idx = np.argsort(values)[::-1][:n]
    return [str(labels[i]) for i in idx]


def run_nmf_one_seed(
    feature_set: str,
    matrix: pd.DataFrame,
    k: int,
    seed: int,
    top_n_genes: int,
    top_n_tissues: int,
    max_iter: int,
    init: str,
) -> list[dict]:
    X = minmax_scale_matrix(matrix)
    genes = list(matrix.columns)
    tissues = list(matrix.index)

    model = NMF(
        n_components=k,
        init=init,
        random_state=seed,
        max_iter=max_iter,
        solver="cd",
        beta_loss="frobenius",
    )

    W = model.fit_transform(X)
    H = model.components_

    records = []

    for component_idx in range(k):
        weights = H[component_idx]
        activities = W[:, component_idx]

        top_genes = top_items(weights, genes, top_n_genes)
        top_tissues = top_items(activities, tissues, top_n_tissues)

        records.append(
            {
                "module_id": f"{feature_set}|k{k}|seed{seed}|NMF{component_idx + 1}",
                "feature_set": feature_set,
                "k": k,
                "seed": seed,
                "component": f"NMF{component_idx + 1}",
                "reconstruction_error": float(model.reconstruction_err_),
                "top_genes": ", ".join(top_genes),
                "top_tissues": "; ".join(top_tissues),
                "activity_vector": normalize_vector_0_1(activities).tolist(),
            }
        )

    return records


def collect_candidate_modules(
    input_base: Path,
    feature_ranks: dict[str, int],
    seeds: Sequence[int],
    top_n_genes: int,
    top_n_tissues: int,
    max_iter: int,
    init: str,
    n_jobs: int,
) -> pd.DataFrame:
    tasks = []

    for feature_set, k in feature_ranks.items():
        matrix = load_matrix(input_base=input_base, feature_set=feature_set)

        for seed in seeds:
            tasks.append(
                (feature_set, matrix, k, seed)
            )

    print(f"[INFO] Running {len(tasks)} NMF jobs.")

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_nmf_one_seed)(
            feature_set=feature_set,
            matrix=matrix,
            k=k,
            seed=seed,
            top_n_genes=top_n_genes,
            top_n_tissues=top_n_tissues,
            max_iter=max_iter,
            init=init,
        )
        for feature_set, matrix, k, seed in tasks
    )

    records = [item for sublist in results for item in sublist]
    modules = pd.DataFrame(records)

    return modules


def build_gene_index(modules: pd.DataFrame) -> tuple[list[str], dict[str, int]]:
    all_genes = set()

    for value in modules["top_genes"].tolist():
        all_genes.update(split_gene_string(value))

    genes = sorted(all_genes)
    gene_to_idx = {gene: i for i, gene in enumerate(genes)}

    return genes, gene_to_idx


def build_module_gene_binary_matrix(modules: pd.DataFrame) -> tuple[csr_matrix, list[str]]:
    genes, gene_to_idx = build_gene_index(modules)

    rows = []
    cols = []
    data = []

    for module_idx, value in enumerate(modules["top_genes"].tolist()):
        module_genes = split_gene_string(value)

        for gene in module_genes:
            if gene in gene_to_idx:
                rows.append(module_idx)
                cols.append(gene_to_idx[gene])
                data.append(1.0)

    M = csr_matrix(
        (data, (rows, cols)),
        shape=(modules.shape[0], len(genes)),
        dtype=np.float32,
    )

    return M, genes


def compute_gene_jaccard_matrix(M: csr_matrix) -> np.ndarray:
    intersection = (M @ M.T).toarray().astype(float)
    sizes = np.asarray(M.sum(axis=1)).ravel()
    union = sizes[:, None] + sizes[None, :] - intersection

    with np.errstate(divide="ignore", invalid="ignore"):
        jacc = intersection / union

    jacc[~np.isfinite(jacc)] = 0.0
    np.fill_diagonal(jacc, 1.0)

    return jacc


def compute_activity_spearman_matrix(modules: pd.DataFrame) -> np.ndarray:
    activity = np.array(modules["activity_vector"].tolist(), dtype=float)

    if activity.ndim != 2:
        raise ValueError("Activity matrix is not two-dimensional.")

    ranked = np.apply_along_axis(rankdata, 1, activity)
    corr = np.corrcoef(ranked)

    corr[~np.isfinite(corr)] = 0.0
    np.fill_diagonal(corr, 1.0)

    # Negative correlations do not support consolidation.
    corr = np.maximum(corr, 0.0)

    return corr


def compute_combined_similarity(
    modules: pd.DataFrame,
    gene_weight: float,
    activity_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    M, _genes = build_module_gene_binary_matrix(modules)

    gene_jaccard = compute_gene_jaccard_matrix(M)
    activity_spearman = compute_activity_spearman_matrix(modules)

    combined = gene_weight * gene_jaccard + activity_weight * activity_spearman
    combined[~np.isfinite(combined)] = 0.0
    np.fill_diagonal(combined, 1.0)

    return combined, gene_jaccard, activity_spearman


def cluster_modules(
    similarity: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0.0)

    try:
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="precomputed",
            linkage="average",
        )
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            linkage="average",
        )

    labels = model.fit_predict(distance)

    return labels


def program_gene_sets(programs: Sequence[ProgramGeneSet]) -> dict[str, set[str]]:
    return {
        program.name: set(map(normalize_gene, program.genes))
        for program in programs
    }


def overlap_coefficient(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0

    return len(a.intersection(b)) / min(len(a), len(b))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0

    union = a.union(b)
    if not union:
        return 0.0

    return len(a.intersection(b)) / len(union)


def cluster_gene_frequency(modules: pd.DataFrame, cluster_id: int) -> Counter:
    sub = modules[modules["cluster_id"].eq(cluster_id)]

    counter: Counter = Counter()

    for value in sub["top_genes"].tolist():
        counter.update(split_gene_string(value))

    return counter


def cluster_tissue_frequency(modules: pd.DataFrame, cluster_id: int) -> Counter:
    sub = modules[modules["cluster_id"].eq(cluster_id)]

    counter: Counter = Counter()

    for value in sub["top_tissues"].tolist():
        tissues = [item.strip() for item in str(value).split(";") if item.strip()]
        counter.update(tissues)

    return counter


def score_clusters_against_programs(
    modules: pd.DataFrame,
    programs: Sequence[ProgramGeneSet],
) -> pd.DataFrame:
    program_sets = program_gene_sets(programs)
    records = []

    for cluster_id in sorted(modules["cluster_id"].unique()):
        sub = modules[modules["cluster_id"].eq(cluster_id)]
        gene_counter = cluster_gene_frequency(modules, cluster_id)
        cluster_genes = set(gene_counter.keys())

        for program in PROGRAM_ORDER:
            pgenes = program_sets[program]

            module_level_overlaps = []
            module_level_jaccards = []

            for value in sub["top_genes"].tolist():
                mgenes = set(split_gene_string(value))
                module_level_overlaps.append(overlap_coefficient(mgenes, pgenes))
                module_level_jaccards.append(jaccard(mgenes, pgenes))

            records.append(
                {
                    "cluster_id": cluster_id,
                    "ecm_program": program,
                    "n_modules": sub.shape[0],
                    "cluster_unique_top_genes": len(cluster_genes),
                    "cluster_program_overlap_coefficient": overlap_coefficient(cluster_genes, pgenes),
                    "cluster_program_jaccard": jaccard(cluster_genes, pgenes),
                    "mean_module_program_overlap_coefficient": float(np.mean(module_level_overlaps)),
                    "mean_module_program_jaccard": float(np.mean(module_level_jaccards)),
                }
            )

    return pd.DataFrame(records)


def assign_clusters_to_programs(cluster_program_scores: pd.DataFrame) -> pd.DataFrame:
    clusters = sorted(cluster_program_scores["cluster_id"].unique())
    programs = PROGRAM_ORDER

    score_matrix = np.zeros((len(clusters), len(programs)), dtype=float)

    for i, cluster_id in enumerate(clusters):
        for j, program in enumerate(programs):
            row = cluster_program_scores[
                (cluster_program_scores["cluster_id"].eq(cluster_id))
                & (cluster_program_scores["ecm_program"].eq(program))
            ].iloc[0]

            score_matrix[i, j] = (
                0.60 * row["mean_module_program_overlap_coefficient"]
                + 0.25 * row["cluster_program_overlap_coefficient"]
                + 0.15 * row["mean_module_program_jaccard"]
            )

    row_ind, col_ind = linear_sum_assignment(-score_matrix)

    records = []

    for r, c in zip(row_ind, col_ind):
        records.append(
            {
                "cluster_id": clusters[r],
                "assigned_ecm_program": programs[c],
                "assignment_score": float(score_matrix[r, c]),
            }
        )

    return pd.DataFrame(records)


def build_cluster_summary(
    modules: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    records = []

    for row in assignments.itertuples():
        cluster_id = row.cluster_id
        assigned_program = row.assigned_ecm_program

        sub = modules[modules["cluster_id"].eq(cluster_id)]

        gene_counter = cluster_gene_frequency(modules, cluster_id)
        tissue_counter = cluster_tissue_frequency(modules, cluster_id)

        top_genes = [gene for gene, _count in gene_counter.most_common(30)]
        top_tissues = [tissue for tissue, _count in tissue_counter.most_common(15)]

        records.append(
            {
                "cluster_id": cluster_id,
                "assigned_ecm_program": assigned_program,
                "assignment_score": row.assignment_score,
                "n_modules": sub.shape[0],
                "n_feature_sets": sub["feature_set"].nunique(),
                "feature_sets": "; ".join(sorted(sub["feature_set"].unique())),
                "n_seeds": sub["seed"].nunique(),
                "seeds": "; ".join(map(str, sorted(sub["seed"].unique()))),
                "n_unique_top_genes": len(gene_counter),
                "top_genes_by_frequency": ", ".join(top_genes),
                "top_tissues_by_frequency": "; ".join(top_tissues),
            }
        )

    return pd.DataFrame(records).sort_values("assigned_ecm_program")


def attach_assignments_to_modules(
    modules: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    out = modules.merge(assignments, on="cluster_id", how="left")

    return out[
        [
            "module_id",
            "feature_set",
            "k",
            "seed",
            "component",
            "cluster_id",
            "assigned_ecm_program",
            "assignment_score",
            "reconstruction_error",
            "top_genes",
            "top_tissues",
        ]
    ].copy()


def plot_cluster_program_heatmap(
    cluster_program_scores: pd.DataFrame,
    html_dir: Path,
    png_dir: Path,
) -> None:
    plot_df = cluster_program_scores.copy()
    plot_df["combined_score"] = (
        0.60 * plot_df["mean_module_program_overlap_coefficient"]
        + 0.25 * plot_df["cluster_program_overlap_coefficient"]
        + 0.15 * plot_df["mean_module_program_jaccard"]
    )

    matrix = plot_df.pivot_table(
        index="cluster_id",
        columns="ecm_program",
        values="combined_score",
        fill_value=0.0,
        aggfunc="mean",
    )

    matrix = matrix[[program for program in PROGRAM_ORDER if program in matrix.columns]]

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns.tolist(),
            y=[f"Cluster {x}" for x in matrix.index],
            colorscale="Viridis",
            colorbar=dict(title="Similarity"),
            text=[[f"{v:.2f}" for v in row] for row in matrix.values],
            texttemplate="%{text}",
            hovertemplate=(
                "Cluster: %{y}<br>"
                "Program: %{x}<br>"
                "Similarity: %{z:.3f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Algorithmic NMF module-cluster to ECM-program similarity",
        template="plotly_white",
        margin=dict(l=120, r=60, t=100, b=250),
        xaxis=dict(tickangle=45),
    )

    save_figure(
        fig,
        "module_cluster_to_program_similarity_heatmap",
        html_dir=html_dir,
        png_dir=png_dir,
        width=1400,
        height=900,
    )


def plot_cluster_size_bar(
    cluster_summary: pd.DataFrame,
    html_dir: Path,
    png_dir: Path,
) -> None:
    df = cluster_summary.sort_values("n_modules", ascending=True)

    fig = go.Figure(
        data=go.Bar(
            x=df["n_modules"],
            y=df["assigned_ecm_program"],
            orientation="h",
            customdata=df[["n_feature_sets", "n_seeds", "assignment_score"]],
            hovertemplate=(
                "Program: %{y}<br>"
                "Modules: %{x}<br>"
                "Feature sets: %{customdata[0]}<br>"
                "Seeds: %{customdata[1]}<br>"
                "Assignment score: %{customdata[2]:.3f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Number of NMF modules assigned to each curated ECM program",
        xaxis_title="Number of NMF modules",
        yaxis_title="Assigned ECM program",
        template="plotly_white",
        margin=dict(l=330, r=60, t=100, b=80),
    )

    save_figure(
        fig,
        "module_cluster_size_by_ecm_program",
        html_dir=html_dir,
        png_dir=png_dir,
        width=1300,
        height=850,
    )


def write_report(
    feature_ranks: dict[str, int],
    modules: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    report_dir: Path,
    n_clusters: int,
    gene_weight: float,
    activity_weight: float,
) -> None:
    report_path = report_dir / "module_consolidation_summary.md"

    lines = []
    lines.append("# Algorithmic NMF Module Consolidation Summary\n")
    lines.append("## Purpose\n")
    lines.append(
        "This analysis makes the consolidation of higher-rank NMF modules into nine curated ECM programs explicit and reproducible."
    )

    lines.append("\n## Input ranks\n")
    for feature_set, k in feature_ranks.items():
        lines.append(f"- {feature_set}: k = {k}")

    lines.append("\n## Consolidation algorithm\n")
    lines.append("1. Run NMF for each Matrisome feature set at its recommended rank across multiple random seeds.")
    lines.append("2. Represent each NMF component using its top genes and tissue activity vector.")
    lines.append("3. Compute pairwise module similarity:")
    lines.append(f"   - combined_similarity = {gene_weight} × top-gene Jaccard + {activity_weight} × positive Spearman tissue-activity similarity")
    lines.append("4. Cluster all candidate modules into nine clusters using average-linkage agglomerative clustering.")
    lines.append("5. Match the nine module clusters to the nine curated ECM programs using a Hungarian assignment on cluster-program gene-overlap scores.")
    lines.append("6. Assign biological names to clusters using the matched ECM program labels.")

    lines.append("\n## Key numbers\n")
    lines.append(f"- Candidate NMF modules: {modules.shape[0]}")
    lines.append(f"- Module clusters: {n_clusters}")
    lines.append(f"- Feature sets represented: {modules['feature_set'].nunique()}")
    lines.append(f"- Random seeds represented: {modules['seed'].nunique()}")

    lines.append("\n## Cluster summary\n")
    for row in cluster_summary.itertuples():
        lines.append(f"### {row.assigned_ecm_program}\n")
        lines.append(f"- Cluster ID: {row.cluster_id}")
        lines.append(f"- Assignment score: {row.assignment_score:.3f}")
        lines.append(f"- Modules: {row.n_modules}")
        lines.append(f"- Feature sets: {row.feature_sets}")
        lines.append(f"- Top genes: {row.top_genes_by_frequency}")
        lines.append(f"- Top tissues: {row.top_tissues_by_frequency}\n")

    lines.append("\n## Manuscript wording recommendation\n")
    lines.append(
        "Use: 'Higher-rank NMF modules were consolidated into nine ECM programs using an explicit module-similarity procedure based on top-gene overlap and tissue-activity similarity. Biological labels were assigned after algorithmic clustering and cluster-program matching.'"
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAVED] {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-base", type=Path, default=DEFAULT_INPUT_BASE)
    parser.add_argument("--program-table", type=Path, default=DEFAULT_CURATED_PROGRAM_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--top-n-genes", type=int, default=30)
    parser.add_argument("--top-n-tissues", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--init", type=str, default="nndsvdar")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--n-clusters", type=int, default=9)
    parser.add_argument("--gene-weight", type=float, default=0.70)
    parser.add_argument("--activity-weight", type=float, default=0.30)

    args = parser.parse_args()

    table_dir, html_dir, png_dir, report_dir = ensure_dirs(args.output_dir)

    programs = load_programs(args.program_table)

    feature_ranks = DEFAULT_RECOMMENDED_RANKS.copy()
    seeds = list(range(args.n_seeds))

    modules = collect_candidate_modules(
        input_base=args.input_base,
        feature_ranks=feature_ranks,
        seeds=seeds,
        top_n_genes=args.top_n_genes,
        top_n_tissues=args.top_n_tissues,
        max_iter=args.max_iter,
        init=args.init,
        n_jobs=args.n_jobs,
    )

    print(f"[INFO] Candidate modules: {modules.shape[0]}")

    combined, gene_jaccard, activity_spearman = compute_combined_similarity(
        modules=modules,
        gene_weight=args.gene_weight,
        activity_weight=args.activity_weight,
    )

    cluster_labels = cluster_modules(
        similarity=combined,
        n_clusters=args.n_clusters,
    )

    modules = modules.copy()
    modules["cluster_id"] = cluster_labels.astype(int)

    cluster_program_scores = score_clusters_against_programs(
        modules=modules,
        programs=programs,
    )

    assignments = assign_clusters_to_programs(cluster_program_scores)
    cluster_summary = build_cluster_summary(modules, assignments)
    module_assignments = attach_assignments_to_modules(modules, assignments)

    modules_for_csv = modules.drop(columns=["activity_vector"]).copy()

    modules_for_csv.to_csv(
        table_dir / "candidate_nmf_modules.csv",
        index=False,
    )

    cluster_program_scores.to_csv(
        table_dir / "module_cluster_program_similarity_scores.csv",
        index=False,
    )

    assignments.to_csv(
        table_dir / "cluster_to_ecm_program_assignments.csv",
        index=False,
    )

    cluster_summary.to_csv(
        table_dir / "ecm_program_cluster_summary.csv",
        index=False,
    )

    module_assignments.to_csv(
        table_dir / "module_to_program_assignment_table.csv",
        index=False,
    )

    # Save compact similarity matrices as compressed numpy files, not huge CSV.
    np.savez_compressed(
        args.output_dir / "module_similarity_matrices.npz",
        combined_similarity=combined,
        gene_jaccard=gene_jaccard,
        activity_spearman=activity_spearman,
    )

    plot_cluster_program_heatmap(
        cluster_program_scores=cluster_program_scores,
        html_dir=html_dir,
        png_dir=png_dir,
    )

    plot_cluster_size_bar(
        cluster_summary=cluster_summary,
        html_dir=html_dir,
        png_dir=png_dir,
    )

    write_report(
        feature_ranks=feature_ranks,
        modules=modules,
        cluster_summary=cluster_summary,
        report_dir=report_dir,
        n_clusters=args.n_clusters,
        gene_weight=args.gene_weight,
        activity_weight=args.activity_weight,
    )

    metadata = {
        "input_base": str(args.input_base),
        "program_table": str(args.program_table),
        "output_dir": str(args.output_dir),
        "feature_ranks": feature_ranks,
        "n_seeds": args.n_seeds,
        "top_n_genes": args.top_n_genes,
        "top_n_tissues": args.top_n_tissues,
        "max_iter": args.max_iter,
        "init": args.init,
        "n_jobs": args.n_jobs,
        "n_clusters": args.n_clusters,
        "gene_weight": args.gene_weight,
        "activity_weight": args.activity_weight,
    }

    with (args.output_dir / "module_consolidation_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n[DONE]")
    print(f"Output folder: {args.output_dir}")
    print(f"Report: {report_dir / 'module_consolidation_summary.md'}")


if __name__ == "__main__":
    main()