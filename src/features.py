"""Pair features (IMPLEMENTATION_PLAN §5.1). Input: candidate pairs (q = S2/S3 representative idx, s1 = S1 idx)
plus the normalised table. All string similarities use rapidfuzz.process.cpdist (element-wise, multi-threaded).

Feature groups: name, near-miss (house number / legal form), address, context flags. Competition features
(rank / gap within a record's candidate list and within an S1's list) are added by add_competition().
"""
import numpy as np
import polars as pl
from rapidfuzz import distance, fuzz, process

from . import config

NORM_COLS = ["name_core", "name_clean", "name_nospace", "acronym", "legal", "pno", "psuffix", "ptuple", "street",
             "sec_nos", "all_nums", "addr_clean", "addr_first", "addr_last", "n_comps", "addr_missing", "n_core_tok",
             "was_indic", "indic_oov", "is_domain", "had_junk", "had_honorific", "upper_style", "alias"]
LEGAL_FORMS = ["", "co", "corp", "inc", "llc", "llp", "lp", "ltd", "pc", "pllc", "plc", "pvt", "sa", "sarl", "sas",
               "sasu", "eurl", "sci", "ei", "cie", "ets"]


def _cp(scorer, a, b, **kw) -> np.ndarray:
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32, **kw)


def attach(pairs: pl.DataFrame, norm: pl.DataFrame, extra: pl.DataFrame | None = None) -> pl.DataFrame:
    """Join normalised columns for both sides: suffix _q (record) and _s (S1)."""
    n = norm.select("idx", *NORM_COLS)
    out = (pairs.join(n.rename({c: c + "_q" for c in NORM_COLS}), left_on="q", right_on="idx", how="left", maintain_order="left")
                .join(n.rename({c: c + "_s" for c in NORM_COLS}), left_on="s1", right_on="idx", how="left", maintain_order="left"))
    if extra is not None:
        out = out.join(extra, left_on="q", right_on="idx", how="left", maintain_order="left")
    return out


def _nums(col: pl.Series) -> tuple[np.ndarray, np.ndarray]:
    """'12 305 7' strings -> flat int64 values + offsets."""
    lst = col.fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != "").str.slice(0, 9).cast(pl.Int64))
    lens = lst.list.len().to_numpy().astype(np.int64)
    flat = lst.explode().drop_nulls().to_numpy() if lens.sum() else np.zeros(0, dtype=np.int64)
    return flat, lens


def closest_number(all_nums_q: pl.Series, pno_s: pl.Series) -> tuple[np.ndarray, np.ndarray]:
    """min |n - pno_s| over the record's numbers, and whether pno_s occurs among them (junk-prefix robust)."""
    flat, lens = _nums(all_nums_q)
    target = np.repeat(pno_s.fill_null(-10**9).to_numpy(), lens)
    d = np.abs(flat - target).astype(np.float64)
    out = np.full(len(lens), np.nan)
    has = lens > 0
    if d.size:
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        out[has] = np.minimum.reduceat(d, starts[has])
    out[pno_s.is_null().to_numpy()] = np.nan
    return out.astype(np.float32), (out == 0)


def name_features(df: pl.DataFrame) -> dict:
    a, b = df["name_core_q"].fill_null("").to_list(), df["name_core_s"].fill_null("").to_list()
    ac, bc = df["name_clean_q"].fill_null("").to_list(), df["name_clean_s"].fill_null("").to_list()
    an, bn = df["name_nospace_q"].fill_null("").to_list(), df["name_nospace_s"].fill_null("").to_list()
    f = {
        "nm_ratio": _cp(fuzz.ratio, a, b),
        "nm_partial": _cp(fuzz.partial_ratio, a, b),
        "nm_tsort": _cp(fuzz.token_sort_ratio, a, b),
        "nm_tset": _cp(fuzz.token_set_ratio, a, b),
        "nm_jw": _cp(distance.JaroWinkler.normalized_similarity, a, b),
        "nm_lev": _cp(distance.Levenshtein.normalized_similarity, a, b),
        "nm_clean_tset": _cp(fuzz.token_set_ratio, ac, bc),
        "nm_nospace_ratio": _cp(fuzz.ratio, an, bn),
        "nm_nospace_partial": _cp(fuzz.partial_ratio, an, bn),
    }
    f["nm_first_eq"] = (df["name_core_q"].str.split(" ").list.first() == df["name_core_s"].str.split(" ").list.first()).fill_null(False).to_numpy()
    f["nm_acr_eq"] = (df["acronym_q"] == df["acronym_s"]).fill_null(False).to_numpy()
    # the record's whole name IS the S1's initials ("BT" <- "Bordeaux Theatre"): all core tokens, and tokens of >= 3
    # letters (skips short function words without a language-specific stop list). Frequent in France test names
    # (acronym names 1.66% of records vs 0.09-0.24% in train); measured neutral-to-positive on leave-one-country-out.
    ns_q = df["name_nospace_q"].fill_null("")
    acr3_s = df["name_core_s"].fill_null("").str.split(" ").list.eval(
        pl.element().filter(pl.element().str.len_chars() >= 3).str.slice(0, 1)).list.join("")
    ok = ns_q.str.len_chars() >= 2
    f["nm_is_acr_of_s"] = (ok & (ns_q == df["acronym_s"].fill_null(""))).to_numpy().astype(np.float32)
    f["nm_is_acr3_of_s"] = (ok & (ns_q == acr3_s)).to_numpy().astype(np.float32)
    f["nm_len_diff"] = (df["name_core_q"].str.len_chars() - df["name_core_s"].str.len_chars()).cast(pl.Float32).to_numpy()
    f["nm_ntok_q"] = df["n_core_tok_q"].cast(pl.Float32).to_numpy()
    f["nm_ntok_s"] = df["n_core_tok_s"].cast(pl.Float32).to_numpy()
    # alias (dba / formerly): best of primary and alias against the S1 core name
    al = df["alias_q"].fill_null("").to_list()
    f["nm_alias_tset"] = np.where(df["alias_q"].fill_null("").str.len_chars().to_numpy() > 0, _cp(fuzz.token_set_ratio, al, b), np.nan)
    return f


def near_miss_features(df: pl.DataFrame) -> dict:
    """The main precision block (FINDINGS §16.4): arithmetic number shifts and legal-form swaps."""
    pq, ps = df["pno_q"].cast(pl.Float64).to_numpy(), df["pno_s"].cast(pl.Float64).to_numpy()
    delta = pq - ps
    f = {"pno_eq": (delta == 0).astype(np.float32), "pno_signed": delta.astype(np.float32),
         "pno_abs": np.abs(delta).astype(np.float32),
         "pno_rel": (np.abs(delta) / np.maximum(np.maximum(pq, ps), 1)).astype(np.float32),
         "pno_log_abs": np.log1p(np.abs(delta)).astype(np.float32)}
    sq = df["pno_q"].cast(pl.Utf8).fill_null("").to_list()
    ss = df["pno_s"].cast(pl.Utf8).fill_null("").to_list()
    lev = _cp(distance.Levenshtein.distance, sq, ss)
    lq, ls = np.array([len(x) for x in sq]), np.array([len(x) for x in ss])
    both = (lq > 0) & (ls > 0)
    ad = np.abs(delta)
    # edit type: 0 missing, 1 equal, 2 digit drop/insert, 3 substitute, 4 arithmetic small (same length, |d|<=50), 5 other
    et = np.full(len(sq), 5, dtype=np.int8)
    et[(lev == 1) & (lq != ls)] = 2
    et[(lev == 1) & (lq == ls)] = 3
    et[(lq == ls) & (ad <= 50) & (ad > 0) & ~((lev == 1) & (lq == ls) & (ad >= 10))] = 4
    et[delta == 0] = 1
    et[~both] = 0
    f["pno_edit_type"] = et
    f["pno_lev"] = np.where(both, lev, np.nan).astype(np.float32)
    f["pno_len_diff"] = np.where(both, lq - ls, np.nan).astype(np.float32)
    cd, inq = closest_number(df["all_nums_q"], df["pno_s"])
    f["num_closest_abs"], f["num_s_in_q"] = cd, inq.astype(np.float32)
    tq, ts = df["ptuple_q"].fill_null(""), df["ptuple_s"].fill_null("")
    f["ptuple_eq"] = ((tq == ts) & (tq != "")).to_numpy().astype(np.float32)
    lq_t, ls_t = tq.str.split("-"), ts.str.split("-")
    f["ptuple_len_eq"] = (lq_t.list.len() == ls_t.list.len()).to_numpy().astype(np.float32)
    f["ptuple_n_common"] = lq_t.list.set_intersection(ls_t).list.len().cast(pl.Float32).to_numpy()
    last_q = lq_t.list.last().str.slice(0, 9).cast(pl.Int64, strict=False).cast(pl.Float64).to_numpy()
    last_s = ls_t.list.last().str.slice(0, 9).cast(pl.Int64, strict=False).cast(pl.Float64).to_numpy()
    f["ptuple_last_abs"] = np.abs(last_q - last_s).astype(np.float32)
    f["psuffix_eq"] = (df["psuffix_q"].fill_null("") == df["psuffix_s"].fill_null("")).to_numpy().astype(np.float32)
    sq2, ss2 = df["sec_nos_q"].fill_null("").str.split("-"), df["sec_nos_s"].fill_null("").str.split("-")
    f["sec_common"] = sq2.list.set_intersection(ss2).list.eval(pl.element().filter(pl.element() != "")).list.len().cast(pl.Float32).to_numpy()
    f["sec_both"] = ((df["sec_nos_q"].fill_null("") != "") & (df["sec_nos_s"].fill_null("") != "")).to_numpy().astype(np.float32)
    # legal form relation + unordered pair id (categorical)
    lq_, ls_ = df["legal_q"].fill_null(""), df["legal_s"].fill_null("")
    rel = np.where(lq_.to_numpy() == ls_.to_numpy(), np.where(lq_.to_numpy() == "", 0, 1),
                   np.where((lq_.to_numpy() == "") | (ls_.to_numpy() == ""), 2, 3)).astype(np.int8)
    f["legal_rel"] = rel  # 0 both missing, 1 same, 2 one missing, 3 both present & different
    main = lambda s: s.str.split(" ").list.first().fill_null("")
    code = {v: i for i, v in enumerate(LEGAL_FORMS)}
    a = main(lq_).replace_strict(code, default=len(LEGAL_FORMS)).to_numpy()
    b = main(ls_).replace_strict(code, default=len(LEGAL_FORMS)).to_numpy()
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    f["legal_pair_id"] = (lo * (len(LEGAL_FORMS) + 1) + hi).astype(np.int16)
    return f


def address_features(df: pl.DataFrame) -> dict:
    sa, sb = df["street_q"].fill_null("").to_list(), df["street_s"].fill_null("").to_list()
    aa, ab = df["addr_clean_q"].fill_null("").to_list(), df["addr_clean_s"].fill_null("").to_list()
    strip_num = lambda s: s.str.replace_all(r"\d+", " ")
    na, nb = strip_num(df["addr_clean_q"].fill_null("")).to_list(), strip_num(df["addr_clean_s"].fill_null("")).to_list()
    f = {"st_ratio": _cp(fuzz.ratio, sa, sb), "st_tset": _cp(fuzz.token_set_ratio, sa, sb),
         "st_eq": (df["street_q"] == df["street_s"]).fill_null(False).to_numpy().astype(np.float32),
         "ad_tset": _cp(fuzz.token_set_ratio, aa, ab), "ad_words_tset": _cp(fuzz.token_set_ratio, na, nb),
         "ad_words_partial": _cp(fuzz.partial_token_set_ratio, na, nb)}
    comps_q = df["addr_clean_q"].fill_null("").str.split(", ")
    comps_s = df["addr_clean_s"].fill_null("").str.split(", ")
    inter = comps_q.list.set_intersection(comps_s).list.len()
    f["ad_comp_common"] = inter.cast(pl.Float32).to_numpy()
    f["ad_comp_jacc"] = (inter / comps_q.list.set_union(comps_s).list.len()).cast(pl.Float32).to_numpy()
    f["ad_state_in"] = comps_q.list.contains(df["addr_last_s"]).fill_null(False).to_numpy().astype(np.float32)
    f["ad_ncomp_q"] = df["n_comps_q"].cast(pl.Float32).to_numpy()
    f["ad_missing_q"] = df["addr_missing_q"].cast(pl.Float32).to_numpy()
    return f


def flag_features(df: pl.DataFrame) -> dict:
    f = {k: df[k + "_q"].cast(pl.Float32).to_numpy() for k in ["was_indic", "indic_oov", "is_domain", "had_junk",
                                                                "had_honorific", "upper_style"]}
    f["src"] = df["src"].cast(pl.Float32).to_numpy()
    f["dup_count"] = df["dup_count"].cast(pl.Float32).to_numpy()
    return f


def compute(pairs: pl.DataFrame, norm: pl.DataFrame, rec_info: pl.DataFrame) -> pl.DataFrame:
    """pairs: q, s1 (+ blocking columns). rec_info: idx, src, dup_count."""
    df = attach(pairs, norm, rec_info)
    feats = {}
    for fn in (name_features, near_miss_features, address_features, flag_features):
        feats.update(fn(df))
    # feature arrays are aligned to the JOINED frame `df` (a polars join does not guarantee row order),
    # so the output is built from df itself, never from `pairs`
    return df.select(pairs.columns).with_columns([pl.Series(k, v) for k, v in feats.items()])


def add_competition(df: pl.DataFrame, score: str) -> pl.DataFrame:
    """Rank / gap of this S1 within the record's candidate list and of this record within the S1's list."""
    return df.with_columns(
        pl.col(score).rank("ordinal", descending=True).over("q").cast(pl.Int16).alias(f"{score}_rank_q"),
        (pl.col(score) - pl.col(score).max().over("q")).alias(f"{score}_gap_best_q"),
        (pl.col(score) - pl.col(score).top_k(2).min().over("q")).alias(f"{score}_gap_2nd_q"),
        pl.len().over("q").cast(pl.Int16).alias("n_cand_q"),
        pl.col(score).rank("ordinal", descending=True).over("s1").cast(pl.Int32).alias(f"{score}_rank_s"),
        pl.len().over("s1").cast(pl.Int32).alias("n_cand_s"),
    )
