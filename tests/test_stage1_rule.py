import numpy as np
import polars as pl

from src.stage1 import KEEP_RULE, prune


def _frame(n_q=200, per_q=30, seed=0):
    rng = np.random.default_rng(seed)
    q = np.repeat(np.arange(n_q), per_q)
    s1 = np.tile(np.arange(per_q), n_q)
    p = rng.beta(0.3, 3.0, size=n_q * per_q)
    return pl.DataFrame({"q": q, "s1": s1}), p


def test_and_rule_keeps_top_k_above_floor():
    df, p = _frame()
    k, floor, cap, mode = KEEP_RULE
    assert mode == "and"
    out = prune(df, p, KEEP_RULE)
    assert (out["p1"] >= np.float32(floor)).all()
    assert out.group_by("q").len()["len"].max() <= min(k, cap)
    # every pair that is in the record's top-k AND above the floor is kept
    ref = df.with_columns(pl.Series("p1", p.astype(np.float32))).with_columns(
        pl.col("p1").rank("ordinal", descending=True).over("q").alias("r"))
    expect = ref.filter((pl.col("r") <= k) & (pl.col("p1") >= floor))
    assert out.height == expect.height


def test_and_rule_is_subset_of_or_rule():
    df, p = _frame(seed=1)
    new = prune(df, p, KEEP_RULE).select("q", "s1")
    old = prune(df, p, (12, 0.001, 16, "or")).select("q", "s1")
    assert new.join(old, on=["q", "s1"], how="anti").height == 0
