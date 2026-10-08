import numpy as np
import polars as pl
from src import calibrate


def test_apply_rules():
    a = pl.DataFrame({"s1": [1, 1, 2], "s1_eid": ["A", "A", "B"], "rec_eid": ["x", "y", "z"]})
    q = np.array([0.95, 0.2, 0.6])
    thr = calibrate._apply(a, q, "thr", 0.5)
    assert set(thr["rec_eid"]) == {"x", "z"}
    ef0 = calibrate._apply(a, q, "ef", 0.0)
    assert "x" in set(ef0["rec_eid"]) and "y" not in set(ef0["rec_eid"])
    ef_strict = calibrate._apply(a, q, "ef", 2.0)      # delta shifts logits down -> stricter
    assert set(ef_strict["rec_eid"]) <= set(ef0["rec_eid"])
