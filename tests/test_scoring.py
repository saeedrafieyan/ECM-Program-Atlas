import pandas as pd

from ecm_program_atlas.scoring import mean_program_score, rank_percentile_program_score


def test_mean_program_score():
    matrix = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=["s1", "s2"])
    score = mean_program_score(matrix, ["A", "B"])
    assert score.loc["s1"] == 2
    assert score.loc["s2"] == 3


def test_rank_percentile_program_score():
    matrix = pd.DataFrame({"A": [1, 4], "B": [3, 2], "C": [2, 1]}, index=["s1", "s2"])
    score = rank_percentile_program_score(matrix, ["A", "B"])
    assert len(score) == 2
