"""E4: Extreme Value Machine probability of inclusion per (record, S1) pair (IMPLEMENTATION_PLAN §5.1).

Rudd et al., TPAMI 2018; reference code EMRResearch/ExtremeValueMachine (defaults used verbatim):
  tailsize 50, margin_scale 0.5, per-point Weibull via libMR fit_low on the margins to the nearest
  points of OTHER classes, inclusion probability = w_score(distance).
Here each S1 entity is one extreme vector (one exemplar per class, so nfuse / set-cover are not needed) and
all other S1s of the same country are other classes (S1 is deduplicated). Distances are cosine distances in
the bi-encoder space of the fold model that retrieved the record (biencoder.py cross-fitting), so a separate
Weibull is fitted per (S1, model).
Outputs: work/<split>/evm_params.parquet (s1, model, mr) and work/<split>/evm.parquet (q, s1, evm_psi).
"""
import libmr
import numpy as np
import polars as pl
import torch
from joblib import Parallel, delayed

from . import config
from .biencoder import N_FOLDS, SEARCH_CHUNK, _assign_model, _text, load

TAILSIZE = 50
MARGIN_SCALE = 0.5


def _fit_chunk(margins: np.ndarray) -> list[str]:
    out = []
    for row in margins:
        mr = libmr.MR()
        mr.fit_low(row.astype(np.float64), TAILSIZE)
        out.append(str(mr))
    return out


def fit(split: str) -> pl.DataFrame:
    rec = pl.read_parquet(config.split_dir(split) / "records.parquet")
    s1_all = rec.filter(pl.col("src") == 1).select("idx", "country", "name", "addr")
    parts = []
    for f in range(N_FOLDS):
        enc = load(f)
        for ctry in s1_all["country"].unique().sort().to_list():
            s1 = s1_all.filter(pl.col("country") == ctry)
            S = enc.encode(_text(s1))
            nn = []
            for i in range(0, S.shape[0], SEARCH_CHUNK):
                sims = S[i:i + SEARCH_CHUNK] @ S.T
                sims[torch.arange(sims.shape[0]), torch.arange(i, i + sims.shape[0])] = -2.0   # exclude self
                nn.append(sims.topk(TAILSIZE, dim=1).values.float().cpu())
            margins = MARGIN_SCALE * (1.0 - torch.cat(nn).numpy())        # nearest other-class margins
            step = max(1, len(margins) // (config.N_THREADS * 8))
            strs = Parallel(n_jobs=config.N_THREADS)(delayed(_fit_chunk)(margins[i:i + step]) for i in range(0, len(margins), step))
            parts.append(pl.DataFrame({"s1": s1["idx"].to_numpy(), "model": np.full(s1.height, f, np.int8),
                                       "mr": [x for c in strs for x in c]}))
            print(f"[evm] {split} model{f} {ctry}: {s1.height:,} Weibulls", flush=True)
        del enc; torch.cuda.empty_cache()
    out = pl.concat(parts)
    out.write_parquet(config.split_dir(split) / "evm_params.parquet")
    return out


def _eval_chunk(mrs: list[str], offsets: np.ndarray, dists: np.ndarray) -> np.ndarray:
    out = np.empty(len(dists))
    for i, s in enumerate(mrs):
        mr = libmr.load_from_string(s)
        a, b = offsets[i], offsets[i + 1]
        out[a:b] = mr.w_score_vector(dists[a:b].astype(np.float64))
    return out


def evaluate(split: str) -> pl.DataFrame:
    """Psi for every pruned pair that has a dense cosine (pairs found only by other channels stay null)."""
    d = config.split_dir(split)
    rec = pl.read_parquet(d / "records.parquet")
    pruned = pl.read_parquet(d / "pruned.parquet", columns=["q", "s1", "dense"]).filter(pl.col("dense") > 0)
    model = _assign_model(split, rec).select(pl.col("idx").alias("q"), "model")
    params = pl.read_parquet(d / "evm_params.parquet")
    x = pruned.join(model, on="q").join(params, on=["s1", "model"]).with_columns(
        (1.0 - pl.col("dense")).alias("dist")).sort("s1", "model")
    grp = x.group_by(["s1", "model"], maintain_order=True).agg(pl.col("mr").first(), pl.len().alias("n"))
    offsets = np.concatenate([[0], np.cumsum(grp["n"].to_numpy())]).astype(np.int64)
    mrs, dists = grp["mr"].to_list(), x["dist"].to_numpy()
    n = len(mrs); step = max(1, n // (config.N_THREADS * 8))
    parts = Parallel(n_jobs=config.N_THREADS)(
        delayed(_eval_chunk)(mrs[i:i + step], offsets[i:min(i + step, n) + 1] - offsets[i],
                             dists[offsets[i]:offsets[min(i + step, n)]]) for i in range(0, n, step))
    out = x.select("q", "s1").with_columns(pl.Series("evm_psi", np.concatenate(parts).astype(np.float32)))
    out.write_parquet(d / "evm.parquet")
    print(f"[evm] {split}: psi for {out.height:,} pairs", flush=True)
    return out


def run() -> None:
    for split in config.SPLITS:
        if not (config.split_dir(split) / "evm_params.parquet").exists():
            fit(split)
        evaluate(split)


if __name__ == "__main__":
    run()
