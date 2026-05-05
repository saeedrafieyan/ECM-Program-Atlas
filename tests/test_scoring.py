import numpy as np
import pandas as pd

from ecm_program_atlas.scoring import (
    ProgramGeneSet,
    available_and_missing_genes,
    mean_program_score,
    program_gene_availability,
    rank_percentile_program_score,
    score_programs,
    top_fraction_program_score,
    zscore_columns,
    zscore_mean_program_score,
)


def make_matrix():
    return pd.DataFrame(
        {
            "A": [1.0, 4.0, 10.0],
            "B": [3.0, 2.0, 0.0],
            "C": [2.0, 1.0, 5.0],
            "D": [0.0, 8.0, 1.0],
        },
        index=["s1", "s2", "s3"],
    )


def test_available_and_missing_genes():
    matrix = make_matrix()
    available, missing = available_and_missing_genes(matrix, ["a", "b", "x"])

    assert available == ["A", "B"]
    assert missing == ["X"]


def test_mean_program_score():
    matrix = make_matrix()
    score = mean_program_score(matrix, ["A", "B"])

    assert score.loc["s1"] == 2.0
    assert score.loc["s2"] == 3.0
    assert score.loc["s3"] == 5.0


def test_zscore_columns_constant_gene_is_zero():
    matrix = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": [5, 5, 5],
        },
        index=["s1", "s2", "s3"],
    )

    z = zscore_columns(matrix)

    assert np.isclose(z["B"].abs().sum(), 0.0)


def test_zscore_mean_program_score_shape():
    matrix = make_matrix()
    score = zscore_mean_program_score(matrix, ["A", "B"])

    assert len(score) == matrix.shape[0]
    assert score.index.tolist() == matrix.index.tolist()


def test_rank_percentile_program_score_range():
    matrix = make_matrix()
    score = rank_percentile_program_score(matrix, ["A", "B"])

    assert len(score) == 3
    assert score.min() >= 0.0
    assert score.max() <= 1.0


def test_top_fraction_program_score():
    matrix = make_matrix()
    score = top_fraction_program_score(matrix, ["A", "B"], top_fraction=0.50)

    assert len(score) == 3
    assert score.min() >= 0.0
    assert score.max() <= 1.0


def test_program_gene_availability():
    matrix = make_matrix()

    programs = [
        ProgramGeneSet("program_1", ("A", "B")),
        ProgramGeneSet("program_2", ("A", "X")),
    ]

    availability = program_gene_availability(matrix, programs)

    assert availability.shape[0] == 2
    assert availability.loc[availability["ecm_program"] == "program_1", "availability_fraction"].iloc[0] == 1.0
    assert availability.loc[availability["ecm_program"] == "program_2", "availability_fraction"].iloc[0] == 0.5


def test_score_programs_multiple_methods():
    matrix = make_matrix()

    programs = [
        ProgramGeneSet("program_1", ("A", "B")),
        ProgramGeneSet("program_2", ("C", "D")),
    ]

    results = score_programs(
        matrix,
        programs,
        methods=["mean", "rank_percentile", "top10_fraction", "top20_fraction"],
    )

    assert set(results.keys()) == {
        "mean",
        "rank_percentile",
        "top10_fraction",
        "top20_fraction",
    }

    for method, scores in results.items():
        assert scores.shape == (3, 2)
        assert scores.columns.tolist() == ["program_1", "program_2"]