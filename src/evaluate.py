"""Exact challenge metric: macro F0.5 over Source-1 entities, singletons included.

Per S1: F0.5 = 1.25*P*R / (0.25*P + R).
  no true matches & empty prediction -> 1.0 ; no true matches & any prediction -> 0.0
  true matches & empty prediction   -> 0.0 (P undefined, R = 0)
"""
import polars as pl


def macro_f05(pred: pl.DataFrame, gt: pl.DataFrame, s1_ids: pl.Series, by: pl.DataFrame | None = None):
    """pred, gt: long pairs (s1_eid, rec_eid). s1_ids: the S1 population to average over.

    Returns (score, per_s1 frame). `by` optionally carries extra S1 columns (e.g. country) for breakdowns.
    """
    base = pl.DataFrame({"s1_eid": s1_ids}).unique()
    p = pred.unique(["s1_eid", "rec_eid"]).join(base, on="s1_eid", how="semi")
    g = gt.unique(["s1_eid", "rec_eid"]).join(base, on="s1_eid", how="semi")
    tp = p.join(g, on=["s1_eid", "rec_eid"], how="inner").group_by("s1_eid").len("tp")
    per = (base.join(p.group_by("s1_eid").len("n_pred"), on="s1_eid", how="left", maintain_order="left")
               .join(g.group_by("s1_eid").len("n_true"), on="s1_eid", how="left", maintain_order="left")
               .join(tp, on="s1_eid", how="left", maintain_order="left")
               .with_columns(pl.col("n_pred", "n_true", "tp").fill_null(0).cast(pl.Float64)))
    prec = pl.col("tp") / pl.col("n_pred")
    rec = pl.col("tp") / pl.col("n_true")
    f = (pl.when((pl.col("n_true") == 0) & (pl.col("n_pred") == 0)).then(1.0)
           .when(pl.col("tp") == 0).then(0.0)
           .otherwise(1.25 * prec * rec / (0.25 * prec + rec)))
    per = per.with_columns(f.alias("f05"))
    if by is not None:
        per = per.join(by, on="s1_eid", how="left", maintain_order="left")
    return float(per["f05"].mean()), per


def report(per: pl.DataFrame, cols=("country",)) -> pl.DataFrame:
    """Breakdown table: mean F0.5 by column(s) and by singleton / non-singleton."""
    per = per.with_columns((pl.col("n_true") == 0).alias("singleton"))
    out = []
    for c in [*[c for c in cols if c in per.columns], "singleton"]:
        out.append(per.group_by(c).agg(pl.len().alias("n"), pl.col("f05").mean().round(5))
                      .rename({c: "value"}).with_columns(pl.col("value").cast(pl.Utf8), pl.lit(c).alias("by")))
    return pl.concat(out).select("by", "value", "n", "f05").sort("by", "value")
