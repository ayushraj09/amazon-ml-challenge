"""Stage D: per-S1 decision maximising expected F0.5 (IMPLEMENTATION_PLAN §7.2, Jansche 2007 / Ye et al. 2012).

For one S1 with candidate probabilities q (records whose argmax is this S1), sorted descending, predicting
the top-k gives   E[F0.5] = sum_{t,f} P(TP=t) P(FN=f) * F(t, f, k)
with TP ~ PoissonBinomial(q[:k]), FN ~ PoissonBinomial(q[k:] + [miss_rate]) and
F = 1.25 t / (1.25 t + 0.25 f + (k - t)); an S1 with no true match and an empty prediction scores 1.
Top-k by q is optimal among subsets under independence, so only k = 0..n is searched.
"""
import numpy as np
import polars as pl
from numba import njit


@njit(cache=True)
def _pb_pmf(p):
    n = p.shape[0]
    pmf = np.zeros(n + 1)
    pmf[0] = 1.0
    for i in range(n):
        for j in range(i + 1, 0, -1):
            pmf[j] = pmf[j] * (1.0 - p[i]) + pmf[j - 1] * p[i]
        pmf[0] *= 1.0 - p[i]
    return pmf


@njit(cache=True)
def _best_k(q, miss):
    n = q.shape[0]
    best_k, best_v = 0, -1.0
    for k in range(n + 1):
        tp = _pb_pmf(q[:k])
        rest = np.empty(n - k + 1)
        rest[: n - k] = q[k:]
        rest[n - k] = miss
        fn = _pb_pmf(rest)
        v = 0.0
        for t in range(k + 1):
            if tp[t] == 0.0:
                continue
            for f in range(n - k + 2):
                if fn[f] == 0.0:
                    continue
                if k == 0:
                    val = 1.0 if f == 0 else 0.0
                elif t == 0:
                    val = 0.0
                else:
                    val = 1.25 * t / (1.25 * t + 0.25 * f + (k - t))
                v += tp[t] * fn[f] * val
        if v > best_v:
            best_v, best_k = v, k
    return best_k, best_v


@njit(cache=True)
def _decide_all(offsets, q, miss):
    m = offsets.shape[0] - 1
    ks = np.zeros(m, dtype=np.int32)
    ev = np.zeros(m)
    for i in range(m):
        k, v = _best_k(q[offsets[i]:offsets[i + 1]], miss)
        ks[i], ev[i] = k, v
    return ks, ev


def expected_f05_select(cands: pl.DataFrame, prob: str = "q", miss_rate: float = 0.0) -> pl.DataFrame:
    """cands: s1, rec, <prob> (one row per record already assigned to its argmax S1).
    Returns the selected (s1, rec) rows plus the expected F0.5 of each S1 decision in column 'ef'."""
    c = cands.sort(["s1", prob], descending=[False, True])
    sizes = c.group_by("s1", maintain_order=True).len()
    offsets = np.concatenate([[0], np.cumsum(sizes["len"].to_numpy())]).astype(np.int64)
    ks, ev = _decide_all(offsets, c[prob].to_numpy().astype(np.float64), float(miss_rate))
    rank = c.with_columns(pl.int_range(pl.len()).over("s1").alias("_r"))
    kk = sizes.select("s1").with_columns(pl.Series("k", ks), pl.Series("ef", ev))
    return rank.join(kk, on="s1").filter(pl.col("_r") < pl.col("k")).drop("_r", "k")


def threshold_select(cands: pl.DataFrame, prob: str = "q", t: float = 0.5) -> pl.DataFrame:
    return cands.filter(pl.col(prob) >= t)
