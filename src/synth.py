"""Synthetic near-miss distractors for test-like validation and augmentation (IMPLEMENTATION_PLAN §6.2).

The generator's own operators, estimated from REAL train distractors (FINDINGS §16.4):
  * house number shifted by a signed delta drawn from the empirical distractor delta distribution
    (real near-miss look-alikes: same core-name key + same street word as an S1, first number differs),
  * legal form transformed per the empirical (S1 legal -> distractor legal) transition distribution.
Operators are applied to a REAL S2/S3 record of the S1 (so source-style noise is realistic).
Rates per country lift near-misses per S1 from the train rate to the measured test rate (FINDINGS §16.5):
  US 0.303 -> 0.550, India 0.136 -> 0.170; France (unlabelled) is proxied by US at 0.702.
Output: work/train/syn_records.parquet with the records.parquet schema, src kept, eid 'S2-SYN..'/'S3-SYN..'.
"""
import numpy as np
import polars as pl

from . import config
from .normalize import basic_name, tokens, LEGAL_SEED

TRAIN_RATE = {"US": 0.303, "India": 0.136}
TEST_RATE = {"US": 0.550, "India": 0.170}
FRANCE_PROXY_RATE = 0.702  # France's measured test rate, applied to half of the US validation S1s
PROXY_HALF = 1             # US val S1 with hash(eid) % 2 == PROXY_HALF form the France-proxy group


def _legal_of(col: str) -> pl.Expr:
    return tokens(basic_name(col)).list.eval(
        pl.element().replace_strict(LEGAL_SEED, default=None).drop_nulls()).list.unique().list.sort().list.join(" ")


def empirical_operators(rec: pl.DataFrame, norm: pl.DataFrame, gt_rep: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Signed number deltas and legal transitions of real near-miss distractors (train)."""
    n = norm.select("idx", "name_core", "street", "pno", "legal")
    r = rec.filter(pl.col("idx") == pl.col("rep")).select("idx", "src", "country").join(n, on="idx")
    s1 = r.filter(pl.col("src") == 1)
    q = r.filter(pl.col("src") > 1).join(gt_rep.select(pl.col("q").alias("idx")), on="idx", how="anti")  # unmatched
    look = q.join(s1, on=["country", "name_core", "street"], suffix="_s").filter(
        pl.col("pno").is_not_null() & pl.col("pno_s").is_not_null() & (pl.col("pno") != pl.col("pno_s")))
    look = look.with_columns((pl.col("pno") - pl.col("pno_s")).alias("delta"))
    deltas = look.select("country", "delta")
    legal_tr = look.group_by("country", pl.col("legal_s").alias("from"), pl.col("legal").alias("to")).len("n")
    return deltas, legal_tr


def _shift_number(addr: str, old: int, new: int) -> tuple[str, bool]:
    """Replace the first WHOLE-number occurrence of `old` (not a substring of a longer number) by `new`.
    Returns (address, changed)."""
    import re
    m = re.search(rf"(?<!\d){old}(?!\d)", addr)
    if m is None or old == new:
        return addr, False
    return addr[:m.start()] + str(new) + addr[m.end():], True


def _valid_deltas(dl: np.ndarray, old: np.ndarray, rng) -> np.ndarray:
    """Draw from the empirical delta distribution conditioned on old + delta > 0 (rejection sampling).
    Without the condition, large negative deltas on small house numbers produced NO change and the record
    became an exact copy of a true match labelled as a distractor (bug found in error analysis, fixed)."""
    d = rng.choice(dl, size=len(old))
    for _ in range(50):
        bad = (old + d) <= 0
        if not bad.any():
            break
        d[bad] = rng.choice(dl, size=int(bad.sum()))
    return d


LEGAL_TAIL_RE = r"(?i)((?:[\s,]+\(?(?:" + "|".join(sorted(LEGAL_SEED, key=len, reverse=True)) + r")\.?\)?)+)\s*$"


def surface_forms(rec: pl.DataFrame) -> dict:
    """Canonical legal string -> (raw tails, probabilities), from how real records spell legal suffixes."""
    t = (rec.filter(pl.col("src") > 1).select("country", pl.col("name").str.extract(LEGAL_TAIL_RE, 1).str.strip_chars(" ,").alias("tail"))
            .drop_nulls().with_columns(_legal_of("tail").alias("canon")))
    c = t.group_by("country", "canon", "tail").len("n").sort("n", descending=True)
    out = {}
    for (ctry, canon), g in c.group_by("country", "canon"):
        g = g.head(10)
        out[(ctry, canon)] = (g["tail"].to_list(), (g["n"] / g["n"].sum()).to_numpy())
    return out


def _swap_legal(name: str, new_tail: str) -> str:
    """Remove legal words from the raw name and append a real-world spelling of the new legal form."""
    words = name.split()
    keep = [w for w in words if w.strip("[]().,").lower().replace(".", "") not in LEGAL_SEED]
    return " ".join(keep + ([new_tail] if new_tail else []))


def generate(rate_add: dict, seed: int = config.SEED) -> pl.DataFrame:
    d = config.split_dir("train")
    rec = pl.read_parquet(d / "records.parquet")
    if "is_syn" in rec.columns:               # never estimate operators from previously generated synthetics
        rec = rec.filter(~pl.col("is_syn"))
    norm = pl.read_parquet(d / "norm.parquet").join(rec.select("idx"), on="idx", how="semi")
    gt = pl.read_parquet(d / "gt.parquet")
    s1 = rec.filter(pl.col("src") == 1).select(pl.col("eid").alias("s1_eid"), pl.col("idx").alias("s1"), "is_val_s1", "country")
    oth = rec.filter(pl.col("src") > 1).select(pl.col("eid").alias("rec_eid"), pl.col("rep").alias("q"), "idx")
    gt_rep = gt.join(s1, on="s1_eid").join(oth, on="rec_eid").select("q", "s1", "idx", "is_val_s1", "country")
    deltas, legal_tr = empirical_operators(rec, norm, gt_rep)
    rng = np.random.default_rng(seed)
    n_s1 = norm.select("idx", pl.col("pno").alias("pno_s"), pl.col("legal").alias("legal_s"))
    out = []
    for grp, add in rate_add.items():
        ctry = "US" if grp == "France-proxy" else grp
        val = gt_rep.filter(pl.col("is_val_s1") & (pl.col("country") == ctry)).join(n_s1, left_on="s1", right_on="idx")
        if ctry == "US":
            s1e = s1.select(pl.col("s1_eid").alias("_e"), pl.col("s1"))
            val = val.join(s1e, on="s1").with_columns(
                (pl.col("_e").hash(config.SEED + 17) % 2 == PROXY_HALF).alias("_proxy")).filter(
                pl.col("_proxy") == (grp == "France-proxy")).drop("_e", "_proxy")
        val = val.filter(pl.col("pno_s").is_not_null())
        # the operator shifts the S1's number, so the source record must actually contain it
        src_addr = rec.select("idx", "addr", "name")
        val = val.join(src_addr, on="idx").filter(
            pl.col("addr").fill_null("").str.contains(pl.col("pno_s").cast(pl.Utf8), literal=True) &
            ~pl.col("name").str.contains(r"^[A-Za-z0-9-]+\.(?:com|net|org|in|co|fr|biz|io|us)$")).drop("addr", "name")
        vs = s1.filter(pl.col("is_val_s1") & (pl.col("country") == ctry))
        if ctry == "US":
            vs = vs.filter((pl.col("s1_eid").hash(config.SEED + 17) % 2 == PROXY_HALF) == (grp == "France-proxy"))
        n_val_s1 = vs.height
        k = int(round(add * n_val_s1))
        k_draw = int(k * 1.2) + 10          # draw extra, keep exactly k verified near-misses below
        base = val.sample(k_draw, with_replacement=k_draw > val.height, seed=seed)
        dl = deltas.filter(pl.col("country") == ctry)["delta"].to_numpy()
        base = base.with_columns(pl.Series("delta", _valid_deltas(dl, base["pno_s"].to_numpy(), rng)))
        tr = legal_tr.filter(pl.col("country") == ctry)
        # sample a target legal form conditioned on the S1's legal form (fallback: marginal)
        tr_by = {f: (g["to"].to_list(), (g["n"] / g["n"].sum()).to_numpy()) for (f,), g in tr.group_by("from")}
        marg = tr.group_by("to").agg(pl.col("n").sum())
        mto, mp = marg["to"].to_list(), (marg["n"] / marg["n"].sum()).to_numpy()
        targets = []
        for f in base["legal_s"].fill_null("").to_list():
            to, p = tr_by.get(f, (mto, mp))
            targets.append(to[rng.choice(len(to), p=p)])
        forms = surface_forms(rec.filter(pl.col("country") == ctry))
        tails = []
        for t in targets:
            opts, pr = forms.get((ctry, t), ([t.upper()], np.array([1.0])))
            tails.append(opts[rng.choice(len(opts), p=pr)] if t else "")
        base = base.with_columns(pl.Series("to_legal", tails))
        src_rows = rec.select("idx", "src", "name", "addr", "dup_count")
        b = base.join(src_rows, on="idx")
        names = [_swap_legal(nm, t) for nm, t in zip(b["name"].to_list(), b["to_legal"].to_list())]
        shifted = [_shift_number(a or "", int(o), int(o + dlt))
                   for a, o, dlt in zip(b["addr"].to_list(), b["pno_s"].to_list(), b["delta"].to_list())]
        addrs = [a for a, _ in shifted]
        g = b.select("src", "country", "pno_s").with_columns(
            pl.Series("name", names), pl.Series("addr", addrs), pl.Series("_ok", [c for _, c in shifted]),
            pl.lit(ctry).alias("_c"))
        # verify: the shift happened AND the record's parsed primary number now differs from the S1's
        from .normalize import load_dicts, normalize
        chk = normalize(g.with_row_index("idx").select("idx", "name", "addr"), load_dicts(), ctry).sort("idx")
        g = g.with_columns((pl.col("_ok") & (chk["pno"] != g["pno_s"]).fill_null(True)).alias("_ok"))
        n_bad = g.height - int(g["_ok"].sum())
        g = g.filter(pl.col("_ok")).head(k)
        print(f"[synth] {grp}: {g.height:,}/{k:,} verified near-misses ({n_bad:,} rejected draws)", flush=True)
        out.append(g.drop("_ok", "pno_s"))
    syn = pl.concat(out).with_row_index("i")
    syn = syn.with_columns(
        (pl.lit("S") + pl.col("src").cast(pl.Utf8) + pl.lit("-SYN") + pl.col("i").cast(pl.Utf8)).alias("eid"),
        pl.lit(1, pl.Int32).alias("dup_count"), pl.lit(True).alias("is_syn"))
    return syn.drop("i", "_c")


def append_to_train(syn: pl.DataFrame) -> None:
    """Append synthetic records to train records/norm (idempotent: previous synthetic rows are replaced)."""
    from .normalize import load_dicts, normalize
    d = config.split_dir("train")
    rec = pl.read_parquet(d / "records.parquet")
    if "is_syn" in rec.columns:
        old = rec.filter(pl.col("is_syn"))["idx"]
        rec = rec.filter(~pl.col("is_syn"))
    else:
        old = pl.Series("idx", [], dtype=pl.Int32)
        rec = rec.with_columns(pl.lit(False).alias("is_syn"))
    start = int(rec["idx"].max()) + 1
    syn = syn.with_row_index("_i").with_columns((pl.col("_i") + start).cast(pl.Int32).alias("idx")).drop("_i")
    syn = syn.with_columns(pl.col("idx").alias("rep"), pl.lit(False).alias("is_val_s1"))
    rec = pl.concat([rec, syn.select(rec.columns)], how="vertical_relaxed")
    rec.write_parquet(d / "records.parquet")
    dicts = load_dicts()
    nsyn = pl.concat([normalize(syn.filter(pl.col("country") == c), dicts, c) for c in syn["country"].unique().to_list()])
    norm = pl.read_parquet(d / "norm.parquet").filter(~pl.col("idx").is_in(old.implode()))
    pl.concat([norm, nsyn], how="vertical_relaxed").write_parquet(d / "norm.parquet")


def run() -> None:
    add = {"US": TEST_RATE["US"] - TRAIN_RATE["US"], "India": TEST_RATE["India"] - TRAIN_RATE["India"],
           "France-proxy": FRANCE_PROXY_RATE - TRAIN_RATE["US"]}
    syn = generate(add)
    import json
    (config.split_dir("train") / "syn_meta.json").write_text(json.dumps({"france_proxy": True, "rates_added": add}))
    syn.write_parquet(config.split_dir("train") / "syn_records.parquet")
    append_to_train(syn)
    print(f"[synth] {syn.height:,} synthetic near-miss records appended: {syn.group_by('country').len().rows()}")


if __name__ == "__main__":
    run()
