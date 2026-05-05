from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProgramGeneSet:
    """Container for one curated ECM program gene set."""

    name: str
    genes: tuple[str, ...]

    @property
    def n_genes(self) -> int:
        return len(self.genes)


def normalize_gene_name(gene: str) -> str:
    """Normalize a gene symbol for matching across datasets."""
    return str(gene).strip().upper()


def normalize_gene_list(genes: Iterable[str]) -> list[str]:
    """Normalize and deduplicate gene symbols while preserving sorted order."""
    clean = {normalize_gene_name(gene) for gene in genes if str(gene).strip()}
    return sorted(clean)


def parse_gene_string(gene_string: str, sep: str = ",") -> list[str]:
    """Parse a comma-separated gene list into normalized symbols."""
    if pd.isna(gene_string):
        return []

    return normalize_gene_list(
        gene.strip()
        for gene in str(gene_string).split(sep)
        if gene.strip()
    )


def ensure_numeric_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Return a numeric copy of a sample × gene matrix.

    Rows are samples. Columns are genes.
    """
    if matrix.empty:
        raise ValueError("Input matrix is empty.")

    numeric = matrix.copy()
    numeric.columns = [normalize_gene_name(col) for col in numeric.columns]
    numeric = numeric.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return numeric


def zscore_columns(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score each gene across samples.

    Constant genes are set to zero.
    """
    matrix = ensure_numeric_matrix(matrix)

    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0, ddof=0).replace(0, np.nan)

    z = matrix.sub(means, axis=1).div(stds, axis=1)
    return z.fillna(0.0)


def available_and_missing_genes(
    matrix: pd.DataFrame,
    genes: Sequence[str],
) -> tuple[list[str], list[str]]:
    """Return available and missing genes for a program."""
    matrix_genes = {normalize_gene_name(col) for col in matrix.columns}
    program_genes = normalize_gene_list(genes)

    available = sorted(set(program_genes).intersection(matrix_genes))
    missing = sorted(set(program_genes).difference(matrix_genes))

    return available, missing


def mean_program_score(
    matrix: pd.DataFrame,
    genes: Sequence[str],
) -> pd.Series:
    """
    Compute mean expression score for a gene program.

    Parameters
    ----------
    matrix:
        Sample × gene matrix.
    genes:
        Gene symbols defining the program.

    Returns
    -------
    pd.Series
        One score per sample.
    """
    matrix = ensure_numeric_matrix(matrix)
    available, _ = available_and_missing_genes(matrix, genes)

    if not available:
        return pd.Series(np.nan, index=matrix.index, name="mean_score")

    return matrix[available].mean(axis=1).rename("mean_score")


def zscore_mean_program_score(
    matrix: pd.DataFrame,
    genes: Sequence[str],
) -> pd.Series:
    """
    Compute mean score using gene-wise z-scored expression.
    """
    z = zscore_columns(matrix)
    available, _ = available_and_missing_genes(z, genes)

    if not available:
        return pd.Series(np.nan, index=z.index, name="zscore_mean_score")

    return z[available].mean(axis=1).rename("zscore_mean_score")


def rank_percentile_program_score(
    matrix: pd.DataFrame,
    genes: Sequence[str],
) -> pd.Series:
    """
    Compute rank-percentile score for a gene program.

    Within each sample, all genes are ranked. Higher expression gets a higher
    percentile. The program score is the mean percentile rank of available
    program genes.
    """
    matrix = ensure_numeric_matrix(matrix)
    available, _ = available_and_missing_genes(matrix, genes)

    if not available:
        return pd.Series(np.nan, index=matrix.index, name="rank_percentile_score")

    ranks = matrix.rank(axis=1, method="average", ascending=True)
    n_genes = matrix.shape[1]

    if n_genes <= 1:
        percentile = ranks * 0.0
    else:
        percentile = (ranks - 1.0) / (n_genes - 1.0)

    return percentile[available].mean(axis=1).rename("rank_percentile_score")


def top_fraction_program_score(
    matrix: pd.DataFrame,
    genes: Sequence[str],
    top_fraction: float,
) -> pd.Series:
    """
    Compute the fraction of program genes found in the top fraction of genes.

    Example:
        top_fraction=0.10 means "fraction of program genes in top 10%".
    """
    if not 0 < top_fraction < 1:
        raise ValueError("top_fraction must be between 0 and 1.")

    matrix = ensure_numeric_matrix(matrix)
    available, _ = available_and_missing_genes(matrix, genes)

    if not available:
        return pd.Series(np.nan, index=matrix.index, name=f"top{top_fraction}_score")

    quantile_threshold = matrix.quantile(1.0 - top_fraction, axis=1)
    score = matrix[available].ge(quantile_threshold, axis=0).mean(axis=1)

    return score.rename(f"top{int(top_fraction * 100)}_fraction_score")


def program_gene_availability(
    matrix: pd.DataFrame,
    programs: Sequence[ProgramGeneSet],
) -> pd.DataFrame:
    """
    Summarize gene availability for each ECM program.
    """
    matrix = ensure_numeric_matrix(matrix)
    records = []

    for program in programs:
        available, missing = available_and_missing_genes(matrix, program.genes)

        records.append(
            {
                "ecm_program": program.name,
                "n_program_genes": len(program.genes),
                "n_available_genes": len(available),
                "n_missing_genes": len(missing),
                "availability_fraction": (
                    len(available) / len(program.genes)
                    if program.genes
                    else np.nan
                ),
                "available_genes": ", ".join(available),
                "missing_genes": ", ".join(missing),
            }
        )

    return pd.DataFrame(records)


def score_programs(
    matrix: pd.DataFrame,
    programs: Sequence[ProgramGeneSet],
    methods: Sequence[str] = (
        "mean",
        "zscore_mean",
        "rank_percentile",
        "top10_fraction",
        "top20_fraction",
    ),
) -> dict[str, pd.DataFrame]:
    """
    Score multiple ECM programs using one or more methods.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are method names. Values are sample × program score matrices.
    """
    matrix = ensure_numeric_matrix(matrix)
    results: dict[str, pd.DataFrame] = {}

    for method in methods:
        scores = pd.DataFrame(index=matrix.index)

        for program in programs:
            if method == "mean":
                score = mean_program_score(matrix, program.genes)

            elif method == "zscore_mean":
                score = zscore_mean_program_score(matrix, program.genes)

            elif method == "rank_percentile":
                score = rank_percentile_program_score(matrix, program.genes)

            elif method == "top10_fraction":
                score = top_fraction_program_score(matrix, program.genes, top_fraction=0.10)

            elif method == "top20_fraction":
                score = top_fraction_program_score(matrix, program.genes, top_fraction=0.20)

            else:
                raise ValueError(f"Unsupported scoring method: {method}")

            scores[program.name] = score

        results[method] = scores

    return results


def load_programs_from_curated_table(
    path: str,
    program_col: str = "ecm_program_curated",
    genes_col: str = "top_genes",
) -> list[ProgramGeneSet]:
    """
    Load curated ECM programs from a table containing module-level gene lists.

    The input table may contain multiple rows per program. Genes are unioned
    across rows belonging to the same program.
    """
    df = pd.read_csv(path)

    missing = [col for col in [program_col, genes_col] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    programs: list[ProgramGeneSet] = []

    for program_name, group in df.groupby(program_col):
        genes: list[str] = []

        for value in group[genes_col].tolist():
            genes.extend(parse_gene_string(value))

        programs.append(
            ProgramGeneSet(
                name=str(program_name),
                genes=tuple(normalize_gene_list(genes)),
            )
        )

    return programs