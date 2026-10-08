import numpy as np
import polars as pl
from src import features


def test_compute_keeps_rows_aligned_after_join():
    # norm rows in scrambled order; pair features must belong to their own (q, s1)
    n = 200
    rng = np.random.default_rng(0)
    idx = rng.permutation(np.arange(2 * n))
    cols = {c: [""] * (2 * n) for c in features.NORM_COLS}
    norm = pl.DataFrame({"idx": idx, **cols}).with_columns(
        pl.col("idx").cast(pl.Utf8).alias("name_core"), pl.lit(None, pl.Int64).alias("pno"),
        pl.lit(0).alias("n_comps"), pl.lit(False).alias("addr_missing"), pl.lit(1).alias("n_core_tok"),
        pl.lit(False).alias("was_indic"), pl.lit(0).alias("indic_oov"), pl.lit(False).alias("is_domain"),
        pl.lit(False).alias("had_junk"), pl.lit(False).alias("had_honorific"), pl.lit(False).alias("upper_style"))
    pairs = pl.DataFrame({"q": np.arange(n), "s1": np.arange(n, 2 * n)})
    pairs = pairs.vstack(pl.DataFrame({"q": np.arange(n), "s1": np.arange(n)}))  # half: q == s1 -> identical names
    info = pl.DataFrame({"idx": np.arange(2 * n), "src": 2, "dup_count": 1})
    out = features.compute(pairs, norm, info)
    same = (out["q"] == out["s1"]).to_numpy()
    assert (out["nm_ratio"].to_numpy()[same] == 100).all()
    assert (out["nm_ratio"].to_numpy()[~same] < 100).mean() > 0.9


def test_feature_cols_rejects_label_leaks():
    import pytest
    from src.train import feature_cols
    ok = pl.DataFrame({"q": [1], "s1": [2], "y": [1], "nm_ratio": [0.5]})
    assert feature_cols(ok) == ["nm_ratio"]
    with pytest.raises(AssertionError):
        feature_cols(pl.DataFrame({"q": [1], "s1": [2], "nm_ratio": [0.5], "p1_right": [0.1]}))
    assert "y_right" not in feature_cols(pl.DataFrame({"q": [1], "s1": [2], "y_right": [1], "a": [0]}))


def test_fold_of_is_row_aligned():
    import numpy as np
    from src import config
    from src.stage1 import fold_of, N_FOLDS
    rng = np.random.default_rng(1)
    q = rng.permutation(np.arange(50_000)).astype(np.int32)
    df = pl.DataFrame({"q": q, "s1": q})
    pos = pl.DataFrame({"q": q[::2], "s1": (q[::2] + 7).astype(np.int32)})    # owners for half of the rows
    f = fold_of(df, pos).to_numpy()
    owner = dict(zip(pos["q"].to_list(), pos["s1"].to_list()))
    g = pl.Series([owner.get(int(x), int(x)) for x in q], dtype=pl.Int32)
    expected = (g.hash(config.SEED) % N_FOLDS).cast(pl.Int8).to_numpy()
    assert (f == expected).all()
