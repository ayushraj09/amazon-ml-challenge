"""Ensemble of two pipeline runs that differ only in the cross-encoder backbone (e.g. mmBERT-small and Qwen3-1.7B).

    ER_WORK_DIR=<ens dir> python -m src.ensemble --runs <run A work dir> <run B work dir>

Countries present in train: the two runs' stacked (full-feature) probabilities are averaged; the decision rule is
re-tuned on the averaged out-of-fold validation probabilities (calibrate.tune, tag stage2x). Countries absent from
train: run A's transfer model and decision rule (default, --transfer a), because the larger cross-encoder's France
changes alone failed the audit (brand / name-only / low-truth additions); --transfer mean averages the two runs'
transfer-model probabilities as well and re-tunes that decision (tag stage2x_tr). The concept-conflict guard in
write_outputs._write applies to unseen countries in both cases. Each run must have
run its `predx` stage. The ensemble dir needs train/ and test/ links to run A's and run A's fillers.json.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import polars as pl

from . import calibrate, config
from .write_outputs import _write


def _mean(a: Path, b: Path, name: str, wb: float = 0.5) -> pl.DataFrame:
    ta = pl.read_parquet(a / "test" / name)
    tb = pl.read_parquet(b / "test" / name).rename({"p": "pb"})
    m = ta.join(tb, on=["q", "s1"], how="left", maintain_order="left")
    assert m["pb"].null_count() == 0, f"test pairs differ between runs ({name})"
    return m.with_columns(((1 - wb) * pl.col("p") + wb * pl.col("pb")).alias("p")).drop("pb")


AVG = "prob"


def _lg(e: pl.Expr) -> pl.Expr:
    e = e.clip(1e-6, 1 - 1e-6)
    return (e / (1 - e)).log() if AVG == "logit" else e


def _avg_np(arrs: list, weights: list) -> np.ndarray:
    if AVG == "logit":
        z = sum(w * np.log(np.clip(o, 1e-6, 1 - 1e-6) / (1 - np.clip(o, 1e-6, 1 - 1e-6))) for w, o in zip(weights, arrs))
        return (1 / (1 + np.exp(-z))).astype(np.float32)
    return sum(w * o for w, o in zip(weights, arrs)).astype(np.float32)


def _mean_n(runs: list, weights: list, name: str) -> pl.DataFrame:
    """Weighted mean of several runs' test probabilities (or of their log-odds with --avg logit)."""
    out = pl.read_parquet(runs[0] / "test" / name).with_columns(_lg(pl.col("p")) * weights[0])
    for r, w in zip(runs[1:], weights[1:]):
        o = pl.read_parquet(r / "test" / name).rename({"p": "pb"})
        out = out.join(o, on=["q", "s1"], how="left", maintain_order="left")
        assert out["pb"].null_count() == 0, f"test pairs differ between runs ({name}: {r})"
        out = out.with_columns(pl.col("p") + w * _lg(pl.col("pb"))).drop("pb")
    if AVG == "logit":
        out = out.with_columns((1 / (1 + (-pl.col("p")).exp())).alias("p"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="runs averaged on the full route; two runs use --w-full, "
                                                                "three or more are averaged with equal weights")
    ap.add_argument("--transfer", choices=["a", "mean"], default="a")
    ap.add_argument("--transfer-runs", nargs="+", default=None,
                    help="with --transfer mean: the runs averaged on the transfer route (default: --runs); two runs use "
                         "--w-transfer, three or more are averaged with equal weights")
    ap.add_argument("--avg", choices=["prob", "logit"], default="prob", help="average probabilities or log-odds (N-run path)")
    ap.add_argument("--weights-full", nargs="+", type=float, default=None, help="explicit weights for --runs (normalised)")
    ap.add_argument("--weights-transfer", nargs="+", type=float, default=None, help="explicit weights for --transfer-runs")
    ap.add_argument("--w-full", type=float, default=0.5, help="weight of run B in the full (train-country) average")
    ap.add_argument("--w-transfer", type=float, default=0.5, help="weight of run B in the transfer average (--transfer mean)")
    args = ap.parse_args()
    global AVG
    AVG = args.avg
    runs = [Path(r) for r in args.runs]
    assert len(runs) >= 2, "--runs needs at least two runs"
    a, b = runs[0], runs[1]
    fw = [1 - args.w_full, args.w_full] if len(runs) == 2 else [1 / len(runs)] * len(runs)
    if args.weights_full:
        assert len(args.weights_full) == len(runs); fw = [x / sum(args.weights_full) for x in args.weights_full]
    tr = pl.read_parquet(config.split_dir("train") / "pairs_feat.parquet", columns=["q", "s1"])
    for r in runs:
        ftr = pl.read_parquet(r / "train" / "pairs_feat.parquet", columns=["q", "s1"])
        assert ftr.height == tr.height and (ftr["q"] == tr["q"]).all() and (ftr["s1"] == tr["s1"]).all(), f"train pairs differ: {r}"
    oofs_f = [np.load(r / "oof_stage2x.npy") for r in runs]
    oa, ob = oofs_f[0], oofs_f[1]
    p = _avg_np(oofs_f, fw)
    np.save(config.WORK_DIR / "oof_stage2x.npy", p)
    print(f"[ensemble] OOF averaged: corr {np.corrcoef(oa, ob)[0, 1]:.4f}", flush=True)
    calibrate.tune(tr, p, "stage2x")
    truns = [Path(r) for r in args.transfer_runs] if args.transfer_runs else [a, b]
    assert len(truns) >= 2, "--transfer-runs needs at least two runs"
    tw = [1 - args.w_transfer, args.w_transfer] if len(truns) == 2 else [1 / len(truns)] * len(truns)
    if args.weights_transfer:
        assert len(args.weights_transfer) == len(truns); tw = [x / sum(args.weights_transfer) for x in args.weights_transfer]
    if args.transfer == "mean":
        for r in truns:
            ftr = pl.read_parquet(r / "train" / "pairs_feat.parquet", columns=["q", "s1"])
            assert ftr.height == tr.height and (ftr["q"] == tr["q"]).all(), f"train pairs differ: {r}"
        oofs = [np.load(r / "oof_stage2x_tr.npy") for r in truns]
        pt = _avg_np(oofs, tw)
        np.save(config.WORK_DIR / "oof_stage2x_tr.npy", pt)
        print(f"[ensemble] transfer OOF averaged over {len(truns)} runs: corr(first two) {np.corrcoef(oofs[0], oofs[1])[0, 1]:.4f}", flush=True)
        calibrate.tune(tr, pt, "stage2x_tr")
    else:
        for f in ("decision_stage2x_tr.json", "iso_stage2x_tr.npz"):
            shutil.copy(a / f, config.WORK_DIR / f)
    full = _mean(a, b, "pred_stage2x.parquet", args.w_full) if (len(runs) == 2 and not args.weights_full) else _mean_n(runs, fw, "pred_stage2x.parquet")
    transfer = _mean_n(truns, tw, "pred_stage2x_tr.parquet") if args.transfer == "mean" else \
        pl.read_parquet(a / "test" / "pred_stage2x_tr.parquet")
    (config.WORK_DIR / "ensemble.json").write_text(json.dumps({"runs": [str(r) for r in runs], "full": "mean",
                                                               "transfer": [str(r) for r in truns] if args.transfer == "mean" else str(a),
                                                               "w_full": args.w_full, "w_transfer": args.w_transfer}))
    _write({"full": full, "transfer": transfer}, "stage2x")


if __name__ == "__main__":
    main()
