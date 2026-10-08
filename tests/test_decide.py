import numpy as np
import polars as pl
from src.decide import _best_k, expected_f05_select


def test_single_candidate_threshold_behaviour():
    assert _best_k(np.array([0.9]), 0.0)[0] == 1
    assert _best_k(np.array([0.3]), 0.0)[0] == 0
    k, v = _best_k(np.array([0.9]), 0.0)
    assert abs(v - 0.9) < 1e-12          # E[F] = P(match) when predicting the single candidate


def test_empty_expected_value_is_prob_all_false():
    k, v = _best_k(np.array([0.2, 0.1]), 0.0)
    assert k == 0 and abs(v - 0.8 * 0.9) < 1e-12


def test_selection_frame():
    df = pl.DataFrame({"s1": [1, 1, 1, 2], "rec": [10, 11, 12, 20], "q": [0.99, 0.95, 0.2, 0.4]})
    sel = expected_f05_select(df)
    assert set(sel["rec"].to_list()) == {10, 11}
