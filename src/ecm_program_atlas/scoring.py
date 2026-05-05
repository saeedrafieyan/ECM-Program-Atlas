import numpy as np
import pandas as pd


def mean_program_score(matrix: pd.DataFrame, genes: list[str]) -> pd.Series:
    available = [g for g in genes if g in matrix.columns]
    if not available:
        return pd.Series(np.nan, index=matrix.index)
    return matrix[available].mean(axis=1)


def rank_percentile_program_score(matrix: pd.DataFrame, genes: list[str]) -> pd.Series:
    available = [g for g in genes if g in matrix.columns]
    if not available:
        return pd.Series(np.nan, index=matrix.index)

    ranks = matrix.rank(axis=1, method="average", ascending=True)
    n = matrix.shape[1]
    percentile = (ranks - 1.0) / max(n - 1, 1)
    return percentile[available].mean(axis=1)
