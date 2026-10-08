"""Decision tuning on test-like validation (IMPLEMENTATION_PLAN §7.2 steps 3, 5, 6).

Inputs: OOF stage-2 (or stage-2+X) probabilities on train pairs. Only validation S1s (is_val_s1) are scored,
with synthetic near-misses included (F_shift) and re-weighted to the test country mix.
  step 3  isotonic calibration of the record-level assignment probability (label: the record's argmax S1 is
          its true S1), CROSS-FITTED over two halves of the validation S1s so the reported score is honest
  step 5  global logit offset delta swept on F_shift
  step 6  compared against a plain global threshold rule
The winning rule and a calibrator fitted on all validation S1s are saved to work/decision.json / .npz and are
applied to test by write_outputs.
"""
import json

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

from . import config, decide
from .evaluate import macro_f05
from .train import assemble_predictions, eval_groups

DELTAS = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 1.0, 1.5, 2.0]
# fine steps around the optimum: on v2 the F_shift(delta) curve peaks sharply near 0.5 (isotonic plateaus cross the
# single-candidate decision point sigmoid(delta)), and the 0.25-step grid could not locate the peak (experiments/candidate_size/calib_diag.py)
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def _logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def _sig(x):
    return 1 / (1 + np.exp(-x))


_POP = None   # optional restriction of the validation S1 population (rehearsals only)


def _val_frame(df: pl.DataFrame, p: np.ndarray) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    d = config.split_dir("train")
    rec = pl.read_parquet(d / "records.parquet")
    gt = pl.read_parquet(d / "gt.parquet")
    s1 = rec.filter((pl.col("src") == 1) & pl.col("is_val_s1")).select(pl.col("eid").alias("s1_eid"), "country")
    if _POP is not None:
        s1 = s1.join(_POP, on="s1_eid", how="semi")
    a = assemble_predictions(df, p, "train").join(s1, on="s1_eid", how="semi")
    truth = gt.select("s1_eid", "rec_eid").with_columns(pl.lit(1).alias("hit"))
    a = a.join(truth, on=["s1_eid", "rec_eid"], how="left", maintain_order="left").with_columns(pl.col("hit").fill_null(0))
    half = (pl.col("s1_eid").hash(config.SEED + 11) % 2).alias("half")
    return a.with_columns(half), gt, s1.with_columns(half)


def _score(sel: pl.DataFrame, gt: pl.DataFrame, s1: pl.DataFrame) -> float:
    by, mixw = eval_groups(s1.select("s1_eid", "country"))
    _, per = macro_f05(sel.select("s1_eid", "rec_eid"), gt, s1["s1_eid"], by=by)
    by_c = {c: float(g["f05"].mean()) for (c,), g in per.group_by("group")}
    return sum(mixw[c] * by_c[c] for c in mixw if c in by_c) / sum(mixw[c] for c in mixw if c in by_c)


def _apply(a: pl.DataFrame, q: np.ndarray, rule: str, param: float) -> pl.DataFrame:
    x = a.with_columns(pl.Series("q", q))
    if rule == "ef":
        x = x.with_columns(pl.Series("q", _sig(_logit(q) - param)))
        return decide.expected_f05_select(x.select("s1", "s1_eid", "rec_eid", "q"), "q")
    return x.filter(pl.col("q") >= param)


def tune(df: pl.DataFrame, p: np.ndarray, tag: str) -> dict:
    a, gt, s1 = _val_frame(df, p)
    results = []
    for use_iso in (False, True):
        q = np.zeros(a.height)
        for h in (0, 1):   # cross-fitted calibration: fit on one half of val S1s, apply to the other
            fit_m, app_m = (a["half"] != h).to_numpy(), (a["half"] == h).to_numpy()
            if use_iso:
                iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
                iso.fit(a["p"].to_numpy()[fit_m], a["hit"].to_numpy()[fit_m])
                q[app_m] = iso.predict(a["p"].to_numpy()[app_m])
            else:
                q[app_m] = a["p"].to_numpy()[app_m]
        for rule, grid in (("ef", DELTAS), ("thr", THRESHOLDS)):
            for param in grid:
                f = _score(_apply(a, q, rule, param), gt, s1)
                results.append({"iso": use_iso, "rule": rule, "param": param, "F_shift": f})
                print(f"[calibrate:{tag}] iso={use_iso} rule={rule} param={param}: F_shift {f:.5f}", flush=True)
    best = select_rule(results)
    iso_all = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(a["p"].to_numpy(), a["hit"].to_numpy())
    np.savez(config.WORK_DIR / f"iso_{tag}.npz", x=iso_all.X_thresholds_, y=iso_all.y_thresholds_)
    global _TAG
    _TAG = tag
    base = _score(_apply(a, np.interp(a["p"].to_numpy(), iso_all.X_thresholds_, iso_all.y_thresholds_) if best["iso"] else a["p"].to_numpy(),
                         best["rule"], best["param"]), gt, s1)
    v1 = v1_heldout_gallery(df, p, best["rule"], best["param"], best["iso"])
    print(f"[calibrate:{tag}] V1 held-out gallery (15% of val S1 removed): F_shift {v1:.5f} vs {base:.5f} "
          f"(drop {base - v1:+.5f}; plan limit 0.01)", flush=True)
    out = {"tag": tag, **best, "v1_F_shift": v1, "v1_base": base, "all": results}
    (config.WORK_DIR / f"decision_{tag}.json").write_text(json.dumps(out, indent=1))
    print(f"[calibrate:{tag}] BEST {best}", flush=True)
    return out


TIE_EPS = 2e-5


def select_rule(results: list) -> dict:
    """Best rule by F_shift; among rules within TIE_EPS of the best (indistinguishable on validation), prefer no
    isotonic calibration: isotonic step maps made a decision fragile before (sharp F(delta) peak on v2), and on the
    v5 transfer model an isotonic rule with delta -0.25 tied the plain rule at 1e-5 while matching more liberally."""
    top = max(r["F_shift"] for r in results)
    near = [r for r in results if r["F_shift"] >= top - TIE_EPS]
    plain = [r for r in near if not r["iso"]]
    return max(plain or near, key=lambda r: r["F_shift"])


def v1_heldout_gallery(df: pl.DataFrame, p: np.ndarray, rule: str, param: float, use_iso: bool, frac: float = 0.15) -> float:
    """V1 (IMPLEMENTATION_PLAN §6.2): drop `frac` of validation S1s from the candidate index; their records
    become non-mated probes whose next-best candidate is usually a chain-mate. Candidate lists already hold
    the next candidates (dense top-50), so removing pairs before the argmax reproduces the index removal."""
    d = config.split_dir("train")
    rec = pl.read_parquet(d / "records.parquet")
    val_s1 = rec.filter((pl.col("src") == 1) & pl.col("is_val_s1")).select(pl.col("idx").alias("s1"), pl.col("eid").alias("s1_eid"))
    removed = val_s1.filter(pl.col("s1_eid").hash(config.SEED + 13) % 1000 < int(frac * 1000))
    keep = ~df["s1"].is_in(removed["s1"].implode()).to_numpy()
    df2, p2 = df.filter(pl.Series(keep)), p[keep]
    a, gt, s1 = _val_frame(df2, p2)
    s1 = s1.join(removed.select("s1_eid"), on="s1_eid", how="anti")
    q = a["p"].to_numpy()
    if use_iso:
        z = np.load(config.WORK_DIR / f"iso_{_TAG}.npz")
        q = np.interp(q, z["x"], z["y"])
    return _score(_apply(a, q, rule, param).join(s1.select("s1_eid"), on="s1_eid", how="semi"), gt, s1)


_TAG = "stage2"


def apply_saved(p: np.ndarray, tag: str) -> tuple[np.ndarray, str, float]:
    """Test side: calibrated probabilities + rule + parameter chosen on validation."""
    dec = json.loads((config.WORK_DIR / f"decision_{tag}.json").read_text())
    if dec["iso"]:
        z = np.load(config.WORK_DIR / f"iso_{tag}.npz")
        p = np.interp(p, z["x"], z["y"])
    return p, dec["rule"], dec["param"]
