"""Open-set features from the literature (IMPLEMENTATION_PLAN §5.1, E1/E2).

E1 Meta-recognition (Scheirer et al., TPAMI 2011): per record, fit a Weibull with libMR's FitHigh to the
   tail of its candidate scores *excluding the top score* and evaluate the top score's CDF (w_score).
   Exactly the libMR usage in OSDN/evt_fitting.py and libMR examples/mr-test.cpp; tail size 20 (OSDN default).
E2 Max-logit / margin (Vaze et al., ICLR 2022): unnormalised top logit of the record and each pair's logit
   gap to the record's best.
"""
import libmr
import numpy as np
import polars as pl
from joblib import Parallel, delayed

from . import config

TAILSIZE = 20
MIN_TAIL = 5


def _evt_chunk(offsets: np.ndarray, scores: np.ndarray) -> np.ndarray:
    out = np.full((len(offsets) - 1, 2), np.nan)
    for i in range(len(offsets) - 1):
        s = scores[offsets[i]:offsets[i + 1]]
        if s.shape[0] < MIN_TAIL + 1:
            continue
        mr = libmr.MR()
        tail = s[1:1 + TAILSIZE]
        mr.fit_high(tail.astype(np.float64), len(tail))
        w = mr.w_score(float(s[0]))
        out[i] = (w, np.log1p(-min(w, 1 - 1e-12)))
    return out


def record_features(df: pl.DataFrame, logit_col: str) -> pl.DataFrame:
    """df: q, <logit_col> for ALL blocked candidates. Returns one row per q."""
    g = df.select("q", logit_col).sort(["q", logit_col], descending=[False, True])
    sizes = g.group_by("q", maintain_order=True).len()
    offsets = np.concatenate([[0], np.cumsum(sizes["len"].to_numpy())]).astype(np.int64)
    scores = g[logit_col].to_numpy().astype(np.float64)
    n = len(offsets) - 1
    step = max(1, n // (config.N_THREADS * 4))
    parts = Parallel(n_jobs=config.N_THREADS)(
        delayed(_evt_chunk)(offsets[i:min(i + step, n) + 1] - offsets[i], scores[offsets[i]:offsets[min(i + step, n)]])
        for i in range(0, n, step))
    evt = np.vstack(parts)
    top = scores[offsets[:-1]]
    second = np.where(np.diff(offsets) > 1, scores[np.minimum(offsets[:-1] + 1, len(scores) - 1)], np.nan)
    return sizes.select("q").with_columns(
        pl.Series("evt_w", evt[:, 0].astype(np.float32)), pl.Series("evt_logsurv", evt[:, 1].astype(np.float32)),
        pl.Series("e2_top_logit", top.astype(np.float32)), pl.Series("e2_margin", (top - second).astype(np.float32)))
