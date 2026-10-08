"""Validation error analysis for the methodology document (IMPLEMENTATION_PLAN §12, template §5).

Uses the OOF probabilities of a trained stage (tag) and the decision rule chosen on validation.
FP = predicted (S1, record) pair that is not in the ground truth; FN = true pair not predicted.
Each error is bucketed by its dominant cause; examples are written to work/errors_<tag>.csv.
"""
import sys

import numpy as np
import polars as pl

from . import calibrate, config
from .train import assemble_predictions


def run(tag: str = "stage2") -> None:
    d = config.split_dir("train")
    df = pl.read_parquet(d / "pairs_feat.parquet", columns=["q", "s1"])
    oof = np.load(config.WORK_DIR / f"oof_{tag}.npy")
    rec = pl.read_parquet(d / "records.parquet")
    norm = pl.read_parquet(d / "norm.parquet").select("idx", "was_indic", "addr_missing", "is_domain", "name_core", "pno", "legal")
    gt = pl.read_parquet(d / "gt.parquet")
    val = rec.filter((pl.col("src") == 1) & pl.col("is_val_s1")).select(pl.col("eid").alias("s1_eid"), "country")
    a = assemble_predictions(df, oof, "train").join(val, on="s1_eid", how="semi")
    q, rule, param = calibrate.apply_saved(a["p"].to_numpy(), tag)
    sel = calibrate._apply(a.select("s1", "s1_eid", "rec_eid"), q, rule, param).select("s1_eid", "rec_eid")
    truth = gt.join(val, on="s1_eid", how="semi")
    fp = sel.join(truth, on=["s1_eid", "rec_eid"], how="anti").with_columns(pl.lit("FP").alias("err"))
    fn = truth.join(sel, on=["s1_eid", "rec_eid"], how="anti").with_columns(pl.lit("FN").alias("err"))
    e = pl.concat([fp, fn])
    ids = rec.select(pl.col("eid"), "idx", "rep", "name", "addr")
    e = (e.join(ids.rename({"eid": "rec_eid", "idx": "r_idx", "rep": "r_rep", "name": "r_name", "addr": "r_addr"}), on="rec_eid")
          .join(ids.rename({"eid": "s1_eid", "idx": "s_idx", "name": "s_name", "addr": "s_addr"}).drop("rep"), on="s1_eid")
          .join(norm.rename({c: "r_" + c for c in norm.columns if c != "idx"}), left_on="r_rep", right_on="idx")
          .join(norm.select("idx", pl.col("name_core").alias("s_core"), pl.col("pno").alias("s_pno"), pl.col("legal").alias("s_legal")),
                left_on="s_idx", right_on="idx"))
    e = e.with_columns(
        pl.when(pl.col("rec_eid").str.contains("-SYN")).then(pl.lit("synthetic near-miss"))
          .when(pl.col("r_addr_missing")).then(pl.lit("address missing"))
          .when((pl.col("r_name_core") == pl.col("s_core")) & (pl.col("r_pno") != pl.col("s_pno"))).then(pl.lit("same name, number differs (near-miss / chain)"))
          .when(pl.col("r_was_indic")).then(pl.col("err") + pl.lit(" Indic name"))
          .when(pl.col("r_is_domain")).then(pl.lit("domain-style name"))
          .when(pl.col("r_pno") == pl.col("s_pno")).then(pl.lit("number equal, name differs"))
          .otherwise(pl.lit("other")).alias("bucket"))
    summary = e.group_by("err", "bucket").len().sort("err", "len", descending=[False, True])
    print(f"[errors:{tag}] FP {fp.height:,}  FN {fn.height:,}  (val S1 {val.height:,})")
    with pl.Config(tbl_rows=40, fmt_str_lengths=60):
        print(summary)
    e.select("err", "bucket", "r_name", "r_addr", "s_name", "s_addr").sample(min(3000, e.height), seed=0) \
     .write_csv(config.WORK_DIR / f"errors_{tag}.csv")
    summary.write_csv(config.WORK_DIR / f"errors_summary_{tag}.csv")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "stage2")
