from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import pearsonr, spearmanr

from ecm_program_atlas.scoring import ProgramGeneSet, load_programs_from_curated_table


DEFAULT_PROGRAM_TABLE = Path(
    "results/tables/frozen/combined_nmf_module_annotations_curated_programs.csv"
)

DEFAULT_MATRISOME_FILE = Path(
    "data/raw/matrisome/human_matrisome.xlsx"
)

DEFAULT_MATRISOMEDB_CLEANED = Path(
    "data/processed/matrisomedb/matrisomedb_cleaned_long_table.csv"
)

DEFAULT_MATRISOMEDB_PROCESSED_DIR = Path(
    "data/processed/matrisomedb"
)

DEFAULT_RNA_TISSUE_PROGRAM_MATRIX = Path(
    "results/tables/frozen/Supplementary_Table_gtex_v11_tissue_program_zscore_matrix.csv"
)

DEFAULT_OUTPUT_DIR = Path("results/revision_matrisomedb_null_abundance")


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


CONDITION_GROUPS = [
    "all_samples",
    "normal_like",
    "disease_like",
    "uncertain",
]


MATRIX_NAMES = [
    "mean_log_nsaf",
    "max_log_nsaf",
    "detection_count",
    "binary_detection",
]


MATRIX_FILE_MAP = {
    "mean_log_nsaf": "tissue_gene_mean_log_nsaf.csv",
    "max_log_nsaf": "tissue_gene_max_log_nsaf.csv",
    "detection_count": "tissue_gene_detection_count.csv",
    "binary_detection": "tissue_gene_binary_detection.csv",
}


TISSUE_NAME_MAP = {
    "blood vessel": "Blood Vessel",
    "blood vessels": "Blood Vessel",
    "skin": "Skin",
    "stomach": "Stomach",
    "colon": "Colon",
    "kidney": "Kidney",
    "lung": "Lung",
    "liver": "Liver",
    "ovary": "Ovary",
    "prostate": "Prostate",
    "fallopian tube": "Fallopian Tube",
    "breast": "Breast",
    "tooth": "Tooth",
    "omentum": "Omentum",
    "eye": "Eye",
    "retina": "Eye",
    "adipose tissue": "Adipose Tissue",
    "muscle": "Muscle",
    "heart": "Heart",
    "brain": "Brain",
    "nerve": "Nerve",
    "bladder": "Bladder",
    "thyroid": "Thyroid",
    "pancreas": "Pancreas",
    "uterus": "Uterus",
    "cervix uteri": "Cervix Uteri",
    "small intestine": "Small Intestine",
    "esophagus": "Esophagus",
    "spleen": "Spleen",
    "testis": "Testis",
    "vagina": "Vagina",
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
    width: int = 1350,
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


def normalize_tissue(tissue: str) -> str:
    key = str(tissue).strip().lower()
    return TISSUE_NAME_MAP.get(key, str(tissue).strip())


def split_gene_string(value: str) -> list[str]:
    if pd.isna(value):
        return []

    return [
        normalize_gene(item)
        for item in str(value).split(",")
        if item.strip()
    ]


def load_human_matrisome_genes(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Human Matrisome file:\n{path}")

    df = pd.read_excel(path, header=1)
    df.columns = [str(col).strip() for col in df.columns]

    if "Gene Symbol" not in df.columns:
        raise ValueError(
            f"Expected 'Gene Symbol' column in Matrisome file. "
            f"Available columns: {df.columns.tolist()}"
        )

    genes = (
        df["Gene Symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({"": np.nan, "NAN": np.nan, "NONE": np.nan})
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    return sorted(genes)


def load_curated_programs(path: Path) -> list[ProgramGeneSet]:
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


def load_cleaned_matrisomedb(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing cleaned MatrisomeDB table:\n{path}\n"
            "Run the MatrisomeDB processing pipeline first."
        )

    df = pd.read_csv(path)

    required = ["gene_symbol", "tissue", "condition_group", "sample_id", "nsaf", "log1p_nsaf"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Cleaned MatrisomeDB table missing columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    df = df.copy()
    df["gene_symbol"] = df["gene_symbol"].astype(str).str.upper()
    df["tissue"] = df["tissue"].astype(str).map(normalize_tissue)
    df["condition_group"] = df["condition_group"].astype(str)
    df["nsaf"] = pd.to_numeric(df["nsaf"], errors="coerce")
    df["log1p_nsaf"] = pd.to_numeric(df["log1p_nsaf"], errors="coerce")

    df = df.dropna(subset=["gene_symbol", "tissue", "condition_group", "log1p_nsaf"])

    return df


def condition_subset(df: pd.DataFrame, condition_group: str) -> pd.DataFrame:
    if condition_group == "all_samples":
        return df.copy()

    return df[df["condition_group"].eq(condition_group)].copy()


def empirical_p_higher(observed: float, random_values: np.ndarray) -> float:
    random_values = random_values[~np.isnan(random_values)]

    if len(random_values) == 0 or np.isnan(observed):
        return np.nan

    return float((np.sum(random_values >= observed) + 1) / (len(random_values) + 1))


def z_score_against_random(observed: float, random_values: np.ndarray) -> float:
    random_values = random_values[~np.isnan(random_values)]

    if len(random_values) < 2 or np.isnan(observed):
        return np.nan

    sd = np.std(random_values, ddof=1)

    if sd == 0:
        return np.nan

    return float((observed - np.mean(random_values)) / sd)


def sample_random_gene_set(
    universe: Sequence[str],
    size: int,
    rng: np.random.Generator,
) -> list[str]:
    universe = sorted(set(universe))

    if size > len(universe):
        raise ValueError(f"Requested size {size} exceeds universe size {len(universe)}")

    return sorted(rng.choice(universe, size=size, replace=False).tolist())


def detection_fraction(genes: Sequence[str], detected_genes: set[str]) -> float:
    genes = sorted(set(genes))

    if not genes:
        return np.nan

    return len(set(genes).intersection(detected_genes)) / len(genes)


def run_detection_null(
    cleaned: pd.DataFrame,
    programs: Sequence[ProgramGeneSet],
    matrisome_universe: Sequence[str],
    n_repeats: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    observed_records = []
    random_records = []
    summary_records = []

    for condition_group in CONDITION_GROUPS:
        subset = condition_subset(cleaned, condition_group)
        detected_genes = set(subset["gene_symbol"].dropna().astype(str).str.upper().unique())

        if not detected_genes:
            print(f"[WARNING] No detected genes for condition group: {condition_group}")
            continue

        for program in programs:
            observed = detection_fraction(program.genes, detected_genes)

            observed_records.append(
                {
                    "condition_group": condition_group,
                    "ecm_program": program.name,
                    "n_program_genes": len(program.genes),
                    "n_detected_program_genes": len(set(program.genes).intersection(detected_genes)),
                    "observed_detection_fraction": observed,
                    "detected_program_genes": ", ".join(sorted(set(program.genes).intersection(detected_genes))),
                    "missing_program_genes": ", ".join(sorted(set(program.genes).difference(detected_genes))),
                }
            )

            random_values = []

            for repeat in range(1, n_repeats + 1):
                random_genes = sample_random_gene_set(
                    universe=matrisome_universe,
                    size=len(program.genes),
                    rng=rng,
                )

                value = detection_fraction(random_genes, detected_genes)
                random_values.append(value)

                random_records.append(
                    {
                        "condition_group": condition_group,
                        "ecm_program": program.name,
                        "repeat": repeat,
                        "random_detection_fraction": value,
                    }
                )

            random_array = np.array(random_values, dtype=float)

            summary_records.append(
                {
                    "condition_group": condition_group,
                    "ecm_program": program.name,
                    "observed_detection_fraction": observed,
                    "random_mean": float(np.nanmean(random_array)),
                    "random_std": float(np.nanstd(random_array, ddof=1)),
                    "z_score_vs_random": z_score_against_random(observed, random_array),
                    "empirical_p_higher_is_better": empirical_p_higher(observed, random_array),
                    "n_random_repeats": n_repeats,
                    "n_program_genes": len(program.genes),
                }
            )

    return (
        pd.DataFrame(observed_records),
        pd.DataFrame(random_records),
        pd.DataFrame(summary_records),
    )


def load_matrisomedb_matrix(
    processed_dir: Path,
    condition_group: str,
    matrix_name: str,
) -> pd.DataFrame | None:
    file_name = MATRIX_FILE_MAP[matrix_name]
    path = processed_dir / condition_group / file_name

    if not path.exists():
        print(f"[MISSING] {path}")
        return None

    matrix = pd.read_csv(path, index_col=0)
    matrix.index = [normalize_tissue(tissue) for tissue in matrix.index]
    matrix.columns = [normalize_gene(col) for col in matrix.columns]
    matrix = matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # If tissue names collapse after normalization, average.
    matrix = matrix.groupby(matrix.index).mean()

    return matrix


def score_program_on_matrix(matrix: pd.DataFrame, genes: Sequence[str]) -> pd.Series:
    available = [gene for gene in genes if gene in matrix.columns]

    if not available:
        return pd.Series(np.nan, index=matrix.index)

    return matrix[available].mean(axis=1)


def abundance_metrics(scores: pd.Series) -> dict[str, float | str]:
    values = pd.to_numeric(scores, errors="coerce").dropna()

    if values.empty:
        return {
            "mean_tissue_score": np.nan,
            "max_tissue_score": np.nan,
            "top3_mean_tissue_score": np.nan,
            "variance_tissue_score": np.nan,
            "top_tissue": "",
        }

    top3 = values.sort_values(ascending=False).head(3)

    return {
        "mean_tissue_score": float(values.mean()),
        "max_tissue_score": float(values.max()),
        "top3_mean_tissue_score": float(top3.mean()),
        "variance_tissue_score": float(values.var(ddof=0)),
        "top_tissue": str(values.sort_values(ascending=False).index[0]),
    }


def run_abundance_null(
    processed_dir: Path,
    programs: Sequence[ProgramGeneSet],
    matrisome_universe: Sequence[str],
    n_repeats: int,
    rng: np.random.Generator,
    matrix_name: str = "mean_log_nsaf",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    observed_records = []
    random_records = []
    summary_records = []

    metric_names = [
        "mean_tissue_score",
        "max_tissue_score",
        "top3_mean_tissue_score",
        "variance_tissue_score",
    ]

    for condition_group in CONDITION_GROUPS:
        matrix = load_matrisomedb_matrix(
            processed_dir=processed_dir,
            condition_group=condition_group,
            matrix_name=matrix_name,
        )

        if matrix is None or matrix.empty:
            continue

        for program in programs:
            available = [gene for gene in program.genes if gene in matrix.columns]
            missing = sorted(set(program.genes).difference(matrix.columns))

            observed_scores = score_program_on_matrix(matrix, program.genes)
            observed_metrics = abundance_metrics(observed_scores)

            observed_records.append(
                {
                    "condition_group": condition_group,
                    "matrix_name": matrix_name,
                    "ecm_program": program.name,
                    "n_program_genes": len(program.genes),
                    "n_available_genes": len(available),
                    "availability_fraction": len(available) / len(program.genes) if program.genes else np.nan,
                    "available_genes": ", ".join(sorted(available)),
                    "missing_genes": ", ".join(missing),
                    **observed_metrics,
                }
            )

            random_metric_values = {metric: [] for metric in metric_names}

            for repeat in range(1, n_repeats + 1):
                random_genes = sample_random_gene_set(
                    universe=matrisome_universe,
                    size=len(program.genes),
                    rng=rng,
                )

                random_scores = score_program_on_matrix(matrix, random_genes)
                random_metrics = abundance_metrics(random_scores)

                random_record = {
                    "condition_group": condition_group,
                    "matrix_name": matrix_name,
                    "ecm_program": program.name,
                    "repeat": repeat,
                }

                for metric in metric_names:
                    random_record[metric] = random_metrics[metric]
                    random_metric_values[metric].append(random_metrics[metric])

                random_records.append(random_record)

            for metric in metric_names:
                observed_value = observed_metrics[metric]
                random_array = np.array(random_metric_values[metric], dtype=float)

                summary_records.append(
                    {
                        "condition_group": condition_group,
                        "matrix_name": matrix_name,
                        "ecm_program": program.name,
                        "metric": metric,
                        "observed_value": observed_value,
                        "random_mean": float(np.nanmean(random_array)),
                        "random_std": float(np.nanstd(random_array, ddof=1)),
                        "z_score_vs_random": z_score_against_random(observed_value, random_array),
                        "empirical_p_higher_is_better": empirical_p_higher(observed_value, random_array),
                        "n_random_repeats": n_repeats,
                        "n_program_genes": len(program.genes),
                    }
                )

    return (
        pd.DataFrame(observed_records),
        pd.DataFrame(random_records),
        pd.DataFrame(summary_records),
    )


def row_zscore(matrix: pd.DataFrame) -> pd.DataFrame:
    x = matrix.astype(float).copy()
    means = x.mean(axis=1)
    stds = x.std(axis=1, ddof=0).replace(0, np.nan)

    z = x.sub(means, axis=0).div(stds, axis=0)
    return z.fillna(0.0)


def load_rna_tissue_matrix(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[WARNING] RNA tissue program matrix not found: {path}")
        return None

    matrix = pd.read_csv(path, index_col=0)
    matrix.index = matrix.index.astype(str)
    matrix.columns = [normalize_tissue(col) for col in matrix.columns]
    matrix = matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # Collapse duplicated tissue names after normalization.
    matrix = matrix.T.groupby(level=0).mean().T

    return matrix


def compute_protein_program_matrix(
    matrix: pd.DataFrame,
    programs: Sequence[ProgramGeneSet],
) -> pd.DataFrame:
    records = {}

    for program in programs:
        scores = score_program_on_matrix(matrix, program.genes)
        records[program.name] = scores

    out = pd.DataFrame(records).T
    out.index.name = "ecm_program"

    return out


def compute_rna_protein_correlations(
    rna_matrix: pd.DataFrame | None,
    processed_dir: Path,
    programs: Sequence[ProgramGeneSet],
    matrix_name: str,
) -> pd.DataFrame:
    if rna_matrix is None or rna_matrix.empty:
        return pd.DataFrame()

    records = []

    for condition_group in CONDITION_GROUPS:
        protein_matrix_raw = load_matrisomedb_matrix(
            processed_dir=processed_dir,
            condition_group=condition_group,
            matrix_name=matrix_name,
        )

        if protein_matrix_raw is None or protein_matrix_raw.empty:
            continue

        protein_program_matrix = compute_protein_program_matrix(
            protein_matrix_raw,
            programs,
        )

        # Compare relative tissue profiles.
        rna_rel = row_zscore(rna_matrix)
        protein_rel = row_zscore(protein_program_matrix)

        common_programs = sorted(set(rna_rel.index).intersection(protein_rel.index))
        common_tissues = sorted(set(rna_rel.columns).intersection(protein_rel.columns))

        if len(common_tissues) < 3:
            print(
                f"[WARNING] Too few overlapping tissues for {condition_group}: "
                f"{common_tissues}"
            )
            continue

        for program in common_programs:
            x = pd.to_numeric(rna_rel.loc[program, common_tissues], errors="coerce")
            y = pd.to_numeric(protein_rel.loc[program, common_tissues], errors="coerce")

            valid = x.notna() & y.notna()

            if valid.sum() < 3:
                continue

            sp = spearmanr(x[valid], y[valid])
            pr = pearsonr(x[valid], y[valid])

            records.append(
                {
                    "condition_group": condition_group,
                    "matrix_name": matrix_name,
                    "ecm_program": program,
                    "n_common_tissues": int(valid.sum()),
                    "common_tissues": "; ".join(common_tissues),
                    "spearman_r": float(sp.statistic),
                    "spearman_p": float(sp.pvalue),
                    "pearson_r": float(pr.statistic),
                    "pearson_p": float(pr.pvalue),
                }
            )

    return pd.DataFrame(records)


def plot_detection_null(summary: pd.DataFrame, html_dir: Path, png_dir: Path) -> None:
    if summary.empty:
        return

    for condition_group, group in summary.groupby("condition_group"):
        group = group.copy()
        group["ecm_program"] = pd.Categorical(
            group["ecm_program"],
            categories=PROGRAM_ORDER,
            ordered=True,
        )
        group = group.sort_values("ecm_program")

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=group["observed_detection_fraction"],
                y=group["ecm_program"],
                orientation="h",
                name="Observed",
                customdata=group[
                    [
                        "random_mean",
                        "z_score_vs_random",
                        "empirical_p_higher_is_better",
                    ]
                ],
                hovertemplate=(
                    "Program: %{y}<br>"
                    "Observed coverage: %{x:.3f}<br>"
                    "Random mean: %{customdata[0]:.3f}<br>"
                    "Z-score: %{customdata[1]:.2f}<br>"
                    "Empirical p: %{customdata[2]:.4f}<extra></extra>"
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=group["random_mean"],
                y=group["ecm_program"],
                mode="markers",
                name="Random Matrisome mean",
                marker=dict(size=11, symbol="diamond"),
                hovertemplate=(
                    "Program: %{y}<br>"
                    "Random mean: %{x:.3f}<extra></extra>"
                ),
            )
        )

        fig.update_layout(
            title=(
                f"MatrisomeDB protein detection coverage vs random Matrisome null<br>"
                f"<sup>{condition_group}</sup>"
            ),
            xaxis_title="Detection coverage",
            yaxis_title="",
            template="plotly_white",
            margin=dict(l=330, r=60, t=100, b=90),
        )

        save_figure(
            fig,
            name=f"r3_detection_null_{condition_group}",
            html_dir=html_dir,
            png_dir=png_dir,
        )


def plot_abundance_null(summary: pd.DataFrame, html_dir: Path, png_dir: Path) -> None:
    if summary.empty:
        return

    # Most useful manuscript view: all_samples, mean_log_nsaf, top3 score.
    subset = summary[
        (summary["condition_group"].eq("all_samples"))
        & (summary["matrix_name"].eq("mean_log_nsaf"))
        & (summary["metric"].eq("top3_mean_tissue_score"))
    ].copy()

    if subset.empty:
        return

    subset["ecm_program"] = pd.Categorical(
        subset["ecm_program"],
        categories=PROGRAM_ORDER,
        ordered=True,
    )
    subset = subset.sort_values("ecm_program")

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=subset["observed_value"],
            y=subset["ecm_program"],
            orientation="h",
            name="Observed",
            customdata=subset[
                [
                    "random_mean",
                    "z_score_vs_random",
                    "empirical_p_higher_is_better",
                ]
            ],
            hovertemplate=(
                "Program: %{y}<br>"
                "Observed top3 score: %{x:.3f}<br>"
                "Random mean: %{customdata[0]:.3f}<br>"
                "Z-score: %{customdata[1]:.2f}<br>"
                "Empirical p: %{customdata[2]:.4f}<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=subset["random_mean"],
            y=subset["ecm_program"],
            mode="markers",
            name="Random Matrisome mean",
            marker=dict(size=11, symbol="diamond"),
        )
    )

    fig.update_layout(
        title=(
            "MatrisomeDB abundance support vs random Matrisome null<br>"
            "<sup>All samples, mean log NSAF, top-3 tissue score</sup>"
        ),
        xaxis_title="Top-3 mean tissue program score",
        yaxis_title="",
        template="plotly_white",
        margin=dict(l=330, r=60, t=100, b=90),
    )

    save_figure(
        fig,
        name="r3_abundance_null_all_samples_top3_mean_log_nsaf",
        html_dir=html_dir,
        png_dir=png_dir,
    )


def plot_rna_protein_correlation(corr: pd.DataFrame, html_dir: Path, png_dir: Path) -> None:
    if corr.empty:
        return

    matrix = corr.pivot_table(
        index="ecm_program",
        columns="condition_group",
        values="spearman_r",
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
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="Spearman r"),
            hovertemplate=(
                "Program: %{y}<br>"
                "Condition: %{x}<br>"
                "Spearman r: %{z:.3f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=(
            "RNA-protein tissue profile correlation<br>"
            "<sup>GTEx RNA program scores vs MatrisomeDB protein NSAF program scores</sup>"
        ),
        template="plotly_white",
        margin=dict(l=330, r=60, t=100, b=90),
    )

    save_figure(
        fig,
        name="r3_rna_protein_tissue_correlation",
        html_dir=html_dir,
        png_dir=png_dir,
        width=1150,
        height=780,
    )


def write_report(
    detection_summary: pd.DataFrame,
    abundance_summary: pd.DataFrame,
    corr: pd.DataFrame,
    report_dir: Path,
) -> None:
    report_path = report_dir / "r3_matrisomedb_null_abundance_summary.md"

    lines = []
    lines.append("# R3 MatrisomeDB Null and Abundance Validation\n")
    lines.append("## Purpose\n")
    lines.append(
        "This analysis strengthens protein-level validation by comparing curated ECM programs "
        "against matched random Matrisome gene-set null models for detection coverage and "
        "NSAF-based abundance metrics."
    )

    lines.append("\n## Detection coverage null model\n")
    if detection_summary.empty:
        lines.append("No detection summary was generated.\n")
    else:
        subset = detection_summary[detection_summary["condition_group"].eq("all_samples")].copy()
        for row in subset.itertuples():
            lines.append(
                f"- **{row.ecm_program}**: observed = {row.observed_detection_fraction:.3f}, "
                f"random mean = {row.random_mean:.3f}, z = {row.z_score_vs_random:.2f}, "
                f"empirical p = {row.empirical_p_higher_is_better:.4f}."
            )

    lines.append("\n## Abundance null model\n")
    if abundance_summary.empty:
        lines.append("No abundance summary was generated.\n")
    else:
        subset = abundance_summary[
            (abundance_summary["condition_group"].eq("all_samples"))
            & (abundance_summary["matrix_name"].eq("mean_log_nsaf"))
            & (abundance_summary["metric"].eq("top3_mean_tissue_score"))
        ].copy()

        for row in subset.itertuples():
            lines.append(
                f"- **{row.ecm_program}**: observed top3 score = {row.observed_value:.3f}, "
                f"random mean = {row.random_mean:.3f}, z = {row.z_score_vs_random:.2f}, "
                f"empirical p = {row.empirical_p_higher_is_better:.4f}."
            )

    lines.append("\n## RNA-protein tissue correlation\n")
    if corr.empty:
        lines.append("No RNA-protein correlation was generated, usually because tissue overlap was insufficient.\n")
    else:
        for row in corr.itertuples():
            lines.append(
                f"- **{row.ecm_program}**, {row.condition_group}: "
                f"Spearman r = {row.spearman_r:.3f}, "
                f"Pearson r = {row.pearson_r:.3f}, "
                f"n tissues = {row.n_common_tissues}."
            )

    lines.append("\n## Interpretation guidance\n")
    lines.append(
        "Detection coverage alone is weak evidence. Stronger support is obtained when a curated ECM "
        "program exceeds matched random Matrisome gene sets for detection and abundance metrics, "
        "and when RNA-derived tissue profiles correlate with MatrisomeDB protein-level tissue profiles."
    )

    lines.append("\n## Limitations\n")
    lines.append("- MatrisomeDB contains mixed normal and disease-associated samples.\n")
    lines.append("- NSAF is semi-quantitative and may not be directly comparable across all studies.\n")
    lines.append("- Tissue overlap with GTEx may be limited.\n")
    lines.append("- Study/repository-aware normalization should be added if enough metadata are available.\n")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAVED] {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--program-table", type=Path, default=DEFAULT_PROGRAM_TABLE)
    parser.add_argument("--matrisome-file", type=Path, default=DEFAULT_MATRISOME_FILE)
    parser.add_argument("--matrisomedb-cleaned", type=Path, default=DEFAULT_MATRISOMEDB_CLEANED)
    parser.add_argument("--matrisomedb-processed-dir", type=Path, default=DEFAULT_MATRISOMEDB_PROCESSED_DIR)
    parser.add_argument("--rna-tissue-program-matrix", type=Path, default=DEFAULT_RNA_TISSUE_PROGRAM_MATRIX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--n-random-repeats", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--abundance-matrix", type=str, default="mean_log_nsaf")

    args = parser.parse_args()

    table_dir, html_dir, png_dir, report_dir = ensure_dirs(args.output_dir)

    rng = np.random.default_rng(args.random_seed)

    programs = load_curated_programs(args.program_table)
    matrisome_universe = load_human_matrisome_genes(args.matrisome_file)
    cleaned = load_cleaned_matrisomedb(args.matrisomedb_cleaned)

    pd.DataFrame(
        [
            {
                "ecm_program": program.name,
                "n_genes": len(program.genes),
                "genes": ", ".join(program.genes),
            }
            for program in programs
        ]
    ).to_csv(table_dir / "r3_curated_program_gene_sets.csv", index=False)

    # Detection null.
    det_obs, det_rand, det_summary = run_detection_null(
        cleaned=cleaned,
        programs=programs,
        matrisome_universe=matrisome_universe,
        n_repeats=args.n_random_repeats,
        rng=rng,
    )

    det_obs.to_csv(table_dir / "r3_observed_detection_support.csv", index=False)
    det_rand.to_csv(table_dir / "r3_random_detection_null.csv", index=False)
    det_summary.to_csv(table_dir / "r3_detection_null_summary.csv", index=False)

    # Abundance null.
    abd_obs, abd_rand, abd_summary = run_abundance_null(
        processed_dir=args.matrisomedb_processed_dir,
        programs=programs,
        matrisome_universe=matrisome_universe,
        n_repeats=args.n_random_repeats,
        rng=rng,
        matrix_name=args.abundance_matrix,
    )

    abd_obs.to_csv(table_dir / "r3_observed_abundance_support.csv", index=False)
    abd_rand.to_csv(table_dir / "r3_random_abundance_null.csv", index=False)
    abd_summary.to_csv(table_dir / "r3_abundance_null_summary.csv", index=False)

    # RNA-protein tissue correlation.
    rna_matrix = load_rna_tissue_matrix(args.rna_tissue_program_matrix)

    corr = compute_rna_protein_correlations(
        rna_matrix=rna_matrix,
        processed_dir=args.matrisomedb_processed_dir,
        programs=programs,
        matrix_name=args.abundance_matrix,
    )
    corr.to_csv(table_dir / "r3_rna_protein_tissue_correlation.csv", index=False)

    # Figures.
    plot_detection_null(det_summary, html_dir=html_dir, png_dir=png_dir)
    plot_abundance_null(abd_summary, html_dir=html_dir, png_dir=png_dir)
    plot_rna_protein_correlation(corr, html_dir=html_dir, png_dir=png_dir)

    write_report(
        detection_summary=det_summary,
        abundance_summary=abd_summary,
        corr=corr,
        report_dir=report_dir,
    )

    metadata = {
        "program_table": str(args.program_table),
        "matrisome_file": str(args.matrisome_file),
        "matrisomedb_cleaned": str(args.matrisomedb_cleaned),
        "matrisomedb_processed_dir": str(args.matrisomedb_processed_dir),
        "rna_tissue_program_matrix": str(args.rna_tissue_program_matrix),
        "n_random_repeats": args.n_random_repeats,
        "random_seed": args.random_seed,
        "abundance_matrix": args.abundance_matrix,
    }

    with (args.output_dir / "r3_matrisomedb_null_abundance_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n[DONE]")
    print(f"Output folder: {args.output_dir}")
    print(f"Tables: {table_dir}")
    print(f"Reports: {report_dir}")
    print(f"Figures HTML: {html_dir}")
    print(f"Figures PNG: {png_dir}")


if __name__ == "__main__":
    main()