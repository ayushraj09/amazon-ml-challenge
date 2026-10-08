"""Mine normalisation dictionaries from the provided data (IMPLEMENTATION_PLAN §3.2). No external data.

  indic : Indic-script token -> Latin token. Train positive pairs with an Indic-script name are word-by-word
          transliterations (99.998% equal token count), so tokens are aligned by position.
  comp  : whole address component -> canonical component (states: 'tn'/'tamil nadu'/Indic script, US state
          names/codes). From train positive pairs whose addresses differ in exactly one component.
  addr  : address token -> canonical token (street types LN/Lane, ST/Street/SAINT, ...). From positive pairs
          whose addresses differ in exactly one token. France (test only) is mined from high-precision
          pseudo-pairs (same core name + same house number + unique S1), i.e. transductively from test data.
  legal : legal-form variants that substitute a seed legal token in positive pairs.
Output: work/dicts.json (+ counts in work/dict_counts/ for the methodology document).
"""
import json

import polars as pl

from . import config
from .normalize import INDIC_RE, LEGAL_SEED, basic_addr, basic_name, tokens

MIN_TOKEN_SUPPORT = 50   # one-token substitutions must be seen at least this often ...
MIN_SHARE = 0.5          # ... and be the majority substitution of that token
MIN_COMP_SUPPORT = 20


def _close(m: dict) -> dict:
    """Resolve chains a->b->c to a fixed point (within one country's dictionary)."""
    out = {}
    for k in m:
        v, seen = m[k], {k}
        while v in m and v not in seen:
            seen.add(v)
            v = m[v]
        out[k] = v
    return {k: v for k, v in out.items() if k != v}


def _pairs_frame() -> pl.DataFrame:
    rec = pl.read_parquet(config.split_dir("train") / "records.parquet").select("idx", "eid", "src", "country", "name", "addr")
    gt = pl.read_parquet(config.split_dir("train") / "gt.parquet")
    s1 = rec.filter(pl.col("src") == 1).select(pl.col("eid").alias("s1_eid"), pl.col("name").alias("n1"), pl.col("addr").alias("a1"))
    oth = rec.filter(pl.col("src") > 1).select(pl.col("eid").alias("rec_eid"), "country", pl.col("name").alias("n2"), pl.col("addr").alias("a2"))
    return gt.join(s1, on="s1_eid").join(oth, on="rec_eid")


def _majority(counts: pl.DataFrame, src: str, dst: str, min_support: int, min_share: float) -> pl.DataFrame:
    tot = counts.group_by(src).agg(pl.col("n").sum().alias("tot"))
    best = counts.sort("n", descending=True).group_by(src, maintain_order=True).first()
    return (best.join(tot, on=src).with_columns((pl.col("n") / pl.col("tot")).alias("share"))
                .filter((pl.col("n") >= min_support) & (pl.col("share") >= min_share) & (pl.col(src) != pl.col(dst))))


def mine_indic(p: pl.DataFrame) -> dict:
    x = p.filter(pl.col("n2").str.contains(INDIC_RE)).select(
        tokens(basic_name("n2")).alias("t2"), tokens(basic_name("n1")).alias("t1"))
    x = x.filter(pl.col("t1").list.len() == pl.col("t2").list.len()).with_row_index("r")
    ex = x.explode("t1", "t2").filter(pl.col("t2").str.contains(INDIC_RE))
    counts = ex.group_by("t2", "t1").len("n")
    best = _majority(counts, "t2", "t1", 1, 0.5)
    return dict(zip(best["t2"], best["t1"])), counts


def _one_diff(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return a.list.set_difference(b)


def _street_tokens(c: str) -> pl.Expr:
    """Tokens of the street component: the first comma component holding a number followed by letters."""
    comp = basic_addr(c).str.split(",").list.eval(
        pl.element().str.strip_chars(" ").filter(pl.element().str.contains(r"\d+\s*\S*\s+\p{L}"))).list.first()
    return comp.fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique()


def mine_addr_tokens(p: pl.DataFrame, min_support=MIN_TOKEN_SUPPORT) -> tuple[dict, pl.DataFrame]:
    """p has a1 (reference/S1 address) and a2 (other address); only street-component tokens are compared,
    so state / city tokens never enter this map (they are handled by the component map)."""
    tok = _street_tokens
    x = p.filter(pl.col("a2").is_not_null()).select(tok("a1").alias("t1"), tok("a2").alias("t2"))
    x = x.with_columns(_one_diff(pl.col("t2"), pl.col("t1")).alias("d2"), _one_diff(pl.col("t1"), pl.col("t2")).alias("d1"))
    x = x.filter((pl.col("d1").list.len() == 1) & (pl.col("d2").list.len() == 1)).select(
        pl.col("d2").list.first().alias("from"), pl.col("d1").list.first().alias("to"))
    word = r"^\p{L}[\p{L}.'-]*$"
    x = x.filter(pl.col("from").str.contains(word) & pl.col("to").str.contains(word) &
                 (pl.col("to").str.len_chars() >= 2))
    counts = x.group_by("from", "to").len("n")
    best = _majority(counts, "from", "to", min_support, MIN_SHARE)
    return dict(zip(best["from"], best["to"])), counts


def mine_components(p: pl.DataFrame) -> tuple[dict, pl.DataFrame]:
    comp = lambda c: basic_addr(c).str.split(",").list.eval(pl.element().str.strip_chars(" .").filter(pl.element() != ""))
    x = p.filter(pl.col("a2").is_not_null()).select(comp("a1").alias("c1"), comp("a2").alias("c2"))
    x = x.with_columns(pl.col("c2").list.set_difference(pl.col("c1")).alias("d2"),
                       pl.col("c1").list.set_difference(pl.col("c2")).alias("d1"),
                       pl.col("c1").list.last().alias("last1"))
    x = x.filter((pl.col("d1").list.len() == 1) & (pl.col("d2").list.len() == 1) &
                 (pl.col("d1").list.first() == pl.col("last1")))          # the S1 side differs only in its state slot
    x = x.select(pl.col("d2").list.first().alias("from"), pl.col("d1").list.first().alias("to"))
    x = x.filter(~pl.col("from").str.contains(r"\d") & ~pl.col("to").str.contains(r"\d"))
    counts = x.group_by("from", "to").len("n")
    best = _majority(counts, "from", "to", MIN_COMP_SUPPORT, 0.6)
    return dict(zip(best["from"], best["to"])), counts


def mine_legal(p: pl.DataFrame) -> dict:
    x = p.select(tokens(basic_name("n1")).list.unique().alias("t1"), tokens(basic_name("n2")).list.unique().alias("t2"))
    x = x.with_columns(pl.col("t2").list.set_difference(pl.col("t1")).alias("d2"),
                       pl.col("t1").list.set_difference(pl.col("t2")).alias("d1"))
    x = x.filter((pl.col("d1").list.len() == 1) & (pl.col("d2").list.len() == 1)).select(
        pl.col("d2").list.first().alias("from"), pl.col("d1").list.first().alias("to"))
    x = x.filter(pl.col("to").is_in(list(LEGAL_SEED)) & ~pl.col("from").is_in(list(LEGAL_SEED)) &
                 ~pl.col("from").str.contains(INDIC_RE))
    counts = x.group_by("from", "to").len("n")
    best = _majority(counts, "from", "to", MIN_TOKEN_SUPPORT, 0.8)
    from rapidfuzz.distance import Levenshtein
    out = {}
    for k, v in zip(best["from"], best["to"]):
        variants = [s for s, c in LEGAL_SEED.items() if c == LEGAL_SEED[v]]
        if max(Levenshtein.normalized_similarity(k, w) for w in variants) >= 0.6:  # typo / leet variant only
            out[k] = LEGAL_SEED[v]
    return out


def unlabelled_pseudo_pairs(country: str) -> pl.DataFrame:
    """High-precision pseudo-pairs from TEST (unlabelled) for a country absent from train (France in this data):
    same core-name key + same first number, and the key+number identifies exactly one S1. Used only to mine
    address abbreviations."""
    rec = pl.read_parquet(config.split_dir("test") / "records.parquet").filter(pl.col("country") == country)
    legal = list(LEGAL_SEED)
    rec = rec.with_columns(
        tokens(basic_name("name")).list.eval(pl.element().filter(~pl.element().is_in(legal))).list.sort().list.join(" ").alias("k"),
        pl.col("addr").fill_null("").str.extract(r"(\d+)", 1).alias("num"))
    rec = rec.filter((pl.col("k") != "") & pl.col("num").is_not_null())
    s1 = rec.filter(pl.col("src") == 1)
    uniq = s1.group_by("k", "num").len("m").filter(pl.col("m") == 1)
    s1 = s1.join(uniq, on=["k", "num"], how="semi").select("k", "num", pl.col("addr").alias("a1"))
    oth = rec.filter(pl.col("src") > 1).select("k", "num", pl.col("addr").alias("a2"))
    return oth.join(s1, on=["k", "num"])


def run() -> None:
    out_dir = config.WORK_DIR / "dict_counts"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = _pairs_frame()
    print(f"[mine] train positive pairs {p.height:,}")
    indic, ic = mine_indic(p)
    ic.sort("n", descending=True).write_csv(out_dir / "indic_counts.csv")
    legal = mine_legal(p)
    comp, addr = {}, {}
    for ctry in p["country"].unique().sort().to_list():          # labelled countries (train)
        pc = p.filter(pl.col("country") == ctry)
        cm, cc = mine_components(pc)
        am, ac = mine_addr_tokens(pc)
        comp[ctry], addr[ctry] = _close(cm), _close(am)
        cc.sort("n", descending=True).write_csv(out_dir / f"comp_counts_{ctry}.csv")
        ac.sort("n", descending=True).write_csv(out_dir / f"addr_counts_{ctry}.csv")
    # countries seen only in test (country is an open set of labels): mine from test pseudo-pairs
    test_ctry = pl.read_parquet(config.split_dir("test") / "records.parquet", columns=["country"])["country"].unique()
    n_pseudo = {}
    for ctry in sorted(set(test_ctry.drop_nulls().to_list()) - set(comp)):
        fr = unlabelled_pseudo_pairs(ctry)
        n_pseudo[ctry] = fr.height
        if fr.height:
            cm, cc = mine_components(fr)
            am, ac = mine_addr_tokens(fr)
            comp[ctry], addr[ctry] = _close(cm), _close(am)
            ac.sort("n", descending=True).write_csv(out_dir / f"addr_counts_{ctry}.csv")
    dicts = {"indic": indic, "legal": legal, "comp": comp, "addr": addr}
    (config.WORK_DIR / "dicts.json").write_text(json.dumps(dicts, ensure_ascii=False, indent=0))
    print(f"[mine] indic {len(indic):,} | legal {len(legal)} | " +
          " | ".join(f"{c}: comp {len(comp[c])} addr {len(addr[c])}" for c in comp) +
          f" | test-only pseudo-pairs {n_pseudo}")


if __name__ == "__main__":
    run()
