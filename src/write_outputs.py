"""Predict test and write output/matching_results.tsv + output/candidate_pairs.tsv (IMPLEMENTATION_PLAN §4.3, §7).

candidate_pairs.tsv is written from the exact pruned candidate set that the stage-2 model scores
(compliance rule C5), expanded from duplicate-group representatives to every member ID.
"""
import subprocess
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

from . import config, decide
from .io_utils import write_id_lists
from .train import CAT_FEATS, N_FOLDS, SUFFIX, add_x, assemble_predictions, build, feature_cols


def _predict_xgb(df: pl.DataFrame, tag: str) -> np.ndarray:
    import xgboost as xgb
    ps, X = [], None
    for k in range(N_FOLDS):
        m = xgb.XGBClassifier()
        m.load_model(str(config.WORK_DIR / f"{tag}_fold{k}.json"))
        if X is None:                      # the model's own feature list (full or transfer feature set)
            X = df.select(m.get_booster().feature_names).to_pandas()
            for c in CAT_FEATS:
                if c in X:
                    X[c] = X[c].astype("category")
        ps.append(m.predict_proba(X)[:, 1])
    return np.mean(ps, axis=0)


def _has(tag: str) -> bool:
    return (config.WORK_DIR / f"{tag}_fold0.json").exists()


def run_final() -> None:
    """Final submission: blend of LightGBM and XGBoost stacked models (weights from validation log-loss),
    decision rule tuned on test-like validation (tag 'final')."""
    import json
    from . import calibrate
    d = config.split_dir("test")
    bl = json.loads((config.WORK_DIR / "blend.json").read_text())
    df = pl.read_parquet(d / "pairs_feat.parquet")
    p2 = pl.read_parquet(d / "pred_stage2.parquet")["p"].to_numpy()
    dfx = add_x(df, "test", p2)
    pa = _predict(dfx, bl["a"])
    pb = _predict(dfx, bl["b"])
    p = bl["w_a"] * pa + (1 - bl["w_a"]) * pb
    pred = dfx.select("q", "s1").with_columns(pl.Series("p", p))   # p is aligned to dfx (post-join order)
    pred.write_parquet(d / "pred_final.parquet")
    _write({"full": pred}, "final")


def _predict(df: pl.DataFrame, tag: str) -> np.ndarray:
    """Average the fold models saved under <tag>; the learner type is taken from the saved files."""
    if (config.WORK_DIR / f"{tag}_fold0.json").exists():
        return _predict_xgb(df, tag)
    if (config.WORK_DIR / f"{tag}_fold0.cbm").exists():
        from catboost import CatBoostClassifier
        X = df.select(feature_cols(df)).to_pandas()
        for c in CAT_FEATS:
            if c in X:
                X[c] = X[c].fillna(-1).astype("int64")
        ps = []
        for k in range(N_FOLDS):
            m = CatBoostClassifier()
            m.load_model(str(config.WORK_DIR / f"{tag}_fold{k}.cbm"))
            ps.append(m.predict_proba(X)[:, 1])
        return np.mean(ps, axis=0)
    cols = feature_cols(df)
    X = df.select(cols).to_pandas()
    for c in CAT_FEATS:
        if c in X:
            X[c] = X[c].astype("category")
    return np.mean([lgb.Booster(model_file=str(config.WORK_DIR / f"{tag}_fold{k}.txt")).predict(X)
                    for k in range(N_FOLDS)], axis=0)


def predict_test(delta: float = 0.0, use_x: bool = False) -> dict:
    """Test probabilities per trained feature set: {"full": frame, "transfer": frame (if trained)}."""
    d = config.split_dir("test")
    # reuse cached test features only if they have exactly the train feature columns (a stale cache from an
    # earlier feature set caused a feature-name mismatch; measured failure, fixed)
    train_cols = [c for c in pl.read_parquet_schema(config.split_dir("train") / "pairs_feat.parquet") if c != "y"]
    cache = d / "pairs_feat.parquet"
    df = pl.read_parquet(cache) if cache.exists() else None
    if df is None or df.columns != train_cols:
        df = build("test")
        assert df.columns == train_cols, "test/train feature columns differ"
        df.write_parquet(cache)
    out = {}
    for fs in ("full", "transfer"):
        sfx = SUFFIX[fs]
        if not _has("stage2" + sfx):
            continue
        p = _predict(df, "stage2" + sfx)
        df.select("q", "s1").with_columns(pl.Series("p", p.astype(np.float32))).write_parquet(d / f"pred_stage2{sfx}.parquet")
        dfx = df
        if use_x:
            dfx = add_x(df, "test", p)
            p = _predict(dfx, "stage2x" + sfx)
        out[fs] = _finish(dfx, p, delta)
    return out


def _finish(df: pl.DataFrame, p: np.ndarray, delta: float) -> pl.DataFrame:
    if delta:
        logit = np.log(p / (1 - p + 1e-12) + 1e-12) - delta
        p = 1 / (1 + np.exp(-logit))
    return df.select("q", "s1").with_columns(pl.Series("p", p))


def save_stacked_preds() -> None:
    """Save this run's stacked test probabilities per route (test/pred_stage2x{,_tr}.parquet) for src.ensemble."""
    d = config.split_dir("test")
    for fs, pred in predict_test(0.0, use_x=True).items():
        pred.write_parquet(d / f"pred_stage2x{SUFFIX[fs]}.parquet")
        print(f"[predx] saved {fs} stacked test probabilities: {pred.height:,} pairs", flush=True)


def run(delta: float = 0.0, use_x: bool = False, miss_rate: float = 0.0) -> None:
    _write(predict_test(delta, use_x), "stage2x" if use_x else "stage2")


GUARD_BREAK_EVEN, GUARD_CONFLICT = 0.8, 0.5


def _abbrev(w: str, u: str) -> bool:
    """w abbreviates u: same first letter, shorter, and w's letters appear in u in order ("frs" / "freres")."""
    if not w or not u or w[0] != u[0] or len(w) >= len(u):
        return False
    it = iter(u)
    return all(ch in it for ch in w)
GUARD_COLS = ["q", "s1", "nm_tset", "nm_first_eq", "pno_eq", "pno_abs", "addr_missing", "nm_fill_extra", "nm_extra_real",
              "nm_is_acr_of_s", "nm_is_acr3_of_s"]


def _pattern():
    """Name/address patterns (the generator's operators) used by the concept-conflict guard."""
    same_num, first = pl.col("pno_eq") == 1, pl.col("nm_first_eq") == 1
    return (pl.when(pl.col("addr_missing") == 1).then(pl.lit("no_address"))
              .when((pl.col("nm_is_acr_of_s") == 1) | (pl.col("nm_is_acr3_of_s") == 1)).then(pl.lit("acronym"))
              .when((pl.col("nm_tset") >= 90) & (pl.col("pno_abs") > 0) & (pl.col("pno_abs") <= 50)).then(pl.lit("near_miss"))
              .when((pl.col("nm_tset") >= 90) & same_num).then(pl.lit("same_name_same_number"))
              .when(pl.col("nm_fill_extra") == 1).then(pl.lit("filler"))
              .when((pl.col("nm_extra_real") == 1) & first & same_num).then(pl.lit("real_word_swap_same_number"))
              .when((pl.col("nm_extra_real") == 1) & first).then(pl.lit("real_word_swap_other_number"))
              .otherwise(pl.lit("other")))


def conflict_guard(sel: pl.DataFrame, rec: pl.DataFrame) -> pl.DataFrame:
    """Concept-conflict guard for countries ABSENT from train. A pattern whose truth rate differs strongly between the
    train countries (> GUARD_CONFLICT apart) and stays below the F0.5 break-even (~0.8: a false positive costs about
    4x a missed match) in every train country is not matched in an unseen country: no train evidence supports it.
    Found on the public leaderboard analysis: the real-word swap at the same address (sibling business) is 77% true
    in US but 8% in India; France matched it 11x more often than either train country (1.0% of its matches vs
    0.07-0.09%), and every model change that accepted more of it lost score (v4 -> v5: -0.000385)."""
    tr = pl.read_parquet(config.split_dir("train") / "pairs_feat.parquet", columns=GUARD_COLS + ["y"])
    trec = pl.read_parquet(config.split_dir("train") / "records.parquet", columns=["idx", "eid", "src", "country"])
    syn = trec.filter(pl.col("eid").str.contains("-SYN")).select(pl.col("idx").alias("q"))
    tr = tr.join(syn, on="q", how="anti").join(trec.filter(pl.col("src") == 1).select(pl.col("idx").alias("s1"), "country"), on="s1")
    rates = tr.with_columns(_pattern().alias("pat")).group_by("pat", "country").agg(pl.col("y").mean().alias("r"))
    stat = rates.group_by("pat").agg(pl.col("r").min().alias("lo"), pl.col("r").max().alias("hi"))
    guarded = stat.filter((pl.col("hi") < GUARD_BREAK_EVEN) & (pl.col("hi") - pl.col("lo") > GUARD_CONFLICT))["pat"].to_list()
    if not guarded:
        return sel
    te = pl.read_parquet(config.split_dir("test") / "pairs_feat.parquet", columns=GUARD_COLS)
    q_of = rec.filter(pl.col("src") > 1).select(pl.col("eid").alias("rec_eid"), pl.col("rep").alias("q"))
    x = sel.join(q_of, on="rec_eid", how="left").join(te.with_columns(_pattern().alias("pat")).select("q", "s1", "pat"),
                                                     on=["q", "s1"], how="left")
    drop = x.filter(pl.col("pat").is_in(guarded))
    # not a sibling swap if every extra record word is a filler in ANY country's mined list ("OEJ Service" <- "OEJ
    # Fetes") or an abbreviation of a word only the S1 has ("Frs" <- "Freres", "Cie" <- "Compagnie"): both are noise
    # operators on the same business (found in the qualitative audit of v4g)
    import json
    fl = config.WORK_DIR / "fillers.json"
    union = sorted(set(sum(json.loads(fl.read_text()).values(), []))) if fl.exists() else []
    norm = pl.read_parquet(config.split_dir("test") / "norm.parquet", columns=["idx", "name_core"])
    d2 = drop.join(norm.rename({"idx": "q", "name_core": "cq"}), on="q", how="left").join(
        norm.rename({"idx": "s1", "name_core": "cs"}), on="s1", how="left")
    keep_pairs = []
    for r in d2.select("q", "s1", "cq", "cs").iter_rows():
        tq, ts = set((r[2] or "").split()), set((r[3] or "").split())
        extra, only_s = tq - ts, ts - tq
        if extra and all(w in union or any(_abbrev(w, u) for u in only_s) for w in extra):
            keep_pairs.append((r[0], r[1]))
    if keep_pairs:
        kp = pl.DataFrame(keep_pairs, schema=["q", "s1"], orient="row").with_columns(pl.lit(True).alias("_keep"))
        x = x.join(kp, on=["q", "s1"], how="left").with_columns(
            pl.when(pl.col("_keep").fill_null(False)).then(pl.lit("filler_or_abbreviation")).otherwise(pl.col("pat")).alias("pat")).drop("_keep")
        drop = x.filter(pl.col("pat").is_in(guarded))
    print(f"[write] concept-conflict guard (unseen countries): patterns {guarded} "
          f"(train truth {stat.filter(pl.col('pat').is_in(guarded)).rows()}): {drop.height:,} pairs not matched", flush=True)
    return x.filter(~pl.col("pat").is_in(guarded).fill_null(False)).select(sel.columns)


GUARD_HARD_NEG = 0.1


def _legal_swap_pattern(split: str, pairs: pl.DataFrame) -> pl.DataFrame:
    """Same name (token-set >= 90) with DISJOINT legal forms (a clean swap: SARL -> SASU, LLC -> Ltd; not a subset
    variant like "Private Limited" -> "Ltd"), split by what the address does. Adds column `lsw` (None otherwise)."""
    d = config.split_dir(split)
    lg = pl.read_parquet(d / "norm.parquet", columns=["idx", "legal"])
    f = pl.read_parquet(d / "pairs_feat.parquet", columns=["q", "s1", "nm_tset", "pno_abs", "addr_missing"])
    x = pairs.join(f, on=["q", "s1"], how="left").join(lg.rename({"idx": "q", "legal": "lq"}), on="q", how="left") \
             .join(lg.rename({"idx": "s1", "legal": "ls"}), on="s1", how="left")
    tok = lambda c: pl.col(c).fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    disjoint = (tok("lq").list.len() > 0) & (tok("ls").list.len() > 0) & (tok("lq").list.set_intersection(tok("ls")).list.len() == 0)
    same = disjoint & (pl.col("nm_tset") >= 90)
    return x.with_columns(pl.when(same & (pl.col("pno_abs") > 0) & (pl.col("pno_abs") <= 50)).then(pl.lit("legal_swap_number_shift"))
                            .when(same & (pl.col("addr_missing") == 1)).then(pl.lit("legal_swap_no_address"))
                            .otherwise(None).alias("lsw")).drop("nm_tset", "pno_abs", "addr_missing", "lq", "ls")


def hard_negative_guard(sel: pl.DataFrame, rec: pl.DataFrame) -> pl.DataFrame:
    """For countries ABSENT from train: a legal-form swap on the same name with the house number shifted by 1-50 (or
    no address) is the generator's near-miss distractor; its train truth is below GUARD_HARD_NEG in every train country
    (India 0.000, US 0.009 -- the few true ones are professional suffixes, PC/LP) and the models accept it at that rate
    there, but the transfer model accepted 6.6% of them in France (qualitative audit: "MX Atelier SASU 37" -> "MX
    Atelier SNC 39"). Patterns are re-derived from train at write time; none is guarded if the data disagree."""
    tr = pl.read_parquet(config.split_dir("train") / "pairs_feat.parquet", columns=["q", "s1", "y"])
    trec = pl.read_parquet(config.split_dir("train") / "records.parquet", columns=["idx", "eid", "src", "country"])
    tr = tr.join(trec.filter(pl.col("eid").str.contains("-SYN")).select(pl.col("idx").alias("q")), on="q", how="anti") \
           .join(trec.filter(pl.col("src") == 1).select(pl.col("idx").alias("s1"), "country"), on="s1")
    rates = _legal_swap_pattern("train", tr).filter(pl.col("lsw").is_not_null()).group_by("lsw", "country") \
        .agg(pl.len().alias("n"), pl.col("y").mean().alias("r"))
    stat = rates.group_by("lsw").agg(pl.col("r").max().alias("hi"), pl.col("n").min().alias("n"), pl.len().alias("k"))
    n_ctry = trec["country"].n_unique()
    guarded = stat.filter((pl.col("hi") < GUARD_HARD_NEG) & (pl.col("n") >= 1000) & (pl.col("k") == n_ctry))["lsw"].to_list()
    if not guarded:
        return sel
    q_of = rec.filter(pl.col("src") > 1).select(pl.col("eid").alias("rec_eid"), pl.col("rep").alias("q"))
    x = _legal_swap_pattern("test", sel.join(q_of, on="rec_eid", how="left"))
    drop = x.filter(pl.col("lsw").is_in(guarded))
    print(f"[write] hard-negative guard (unseen countries): patterns {sorted(rates.rows())}: guarded {guarded}; "
          f"{drop.height:,} pairs not matched ({sorted(drop.group_by('lsw').len().rows())})", flush=True)
    return x.filter(~pl.col("lsw").is_in(guarded).fill_null(False)).select(sel.columns)


def _adds_country_word(split: str, pairs: pl.DataFrame) -> pl.DataFrame:
    """Flag pairs (q, s1) whose record name is the S1 name plus the S1 country's own name, i.e. the lower-cased country
    label ("EHPAD Joseph" -> "EHPAD JOSEPH (FRANCE)", "Kamal Impex" -> "Kamal (India) Impex"): token sets of name_core
    (legal forms removed). No country is hard-coded."""
    d = config.split_dir(split)
    norm = pl.read_parquet(d / "norm.parquet", columns=["idx", "name_core"])
    ctry = pl.read_parquet(d / "records.parquet", columns=["idx", "src", "country"]).filter(pl.col("src") == 1)
    x = pairs.join(norm.rename({"idx": "q", "name_core": "cq"}), on="q", how="left") \
             .join(norm.rename({"idx": "s1", "name_core": "cs"}), on="s1", how="left") \
             .join(ctry.select(pl.col("idx").alias("s1"), "country"), on="s1", how="left")
    cw = pl.concat_list(pl.col("country").str.to_lowercase())
    tq = pl.col("cq").fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique().list.sort()
    ts = pl.col("cs").fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique().list.sort()
    x = x.with_columns(tq.alias("_tq"), ts.alias("_ts"), cw.alias("_w"))
    flag = ((pl.col("_tq").list.set_intersection("_w").list.len() > 0) & (pl.col("_ts").list.set_intersection("_w").list.len() == 0)
            & (pl.col("_tq").list.set_difference("_w").list.sort() == pl.col("_ts")) & (pl.col("_ts").list.len() > 0))
    return x.with_columns(flag.fill_null(False).alias("_cw")).drop("_tq", "_ts", "_w")


def country_word_guard(sel: pl.DataFrame, rec: pl.DataFrame) -> pl.DataFrame:
    """A record named exactly like the S1 plus the S1 country's own name is a generated distractor in every train
    country (India 1/452 true, US 0/3; the reverse -- the S1 carries the country name and the record drops it -- is
    79-92% true and is not touched). Applied to every country when the train truth of the pattern is below the F0.5
    break-even in each train country. Found in the qualitative France audit: v4 matched 172 such France pairs."""
    tr = pl.read_parquet(config.split_dir("train") / "pairs_feat.parquet", columns=["q", "s1", "y"])
    trec = pl.read_parquet(config.split_dir("train") / "records.parquet", columns=["idx", "eid"])
    tr = tr.join(trec.filter(pl.col("eid").str.contains("-SYN")).select(pl.col("idx").alias("q")), on="q", how="anti")
    tr = _adds_country_word("train", tr.filter(pl.col("q") >= 0)).filter("_cw")
    rates = tr.group_by("country").agg(pl.len().alias("n"), pl.col("y").mean().alias("r"))
    if tr.height < 100 or rates["r"].max() >= GUARD_BREAK_EVEN:
        print(f"[write] country-word guard off (train {rates.rows()})", flush=True)
        return sel
    q_of = rec.filter(pl.col("src") > 1).select(pl.col("eid").alias("rec_eid"), pl.col("rep").alias("q"))
    x = _adds_country_word("test", sel.join(q_of, on="rec_eid", how="left"))
    print(f"[write] country-word guard (train truth {sorted(rates.rows())}): pairs not matched by country "
          f"{sorted(x.filter('_cw').group_by('country').len().rows())}", flush=True)
    return x.filter(~pl.col("_cw")).select(sel.columns)


GUARD_HARD_POS = 0.95
HARD_POS = __import__("os").environ.get("ER_HARD_POSITIVE", "0") == "1"


def hard_positive_rule(chosen: pl.DataFrame, assigned: pl.DataFrame) -> pl.DataFrame:
    """Mirror of hard_negative_guard, for countries ABSENT from train (env ER_HARD_POSITIVE=1). A record whose name is
    the S1's name with its category word replaced or extended by FILLER words only (the fillers mined from that
    country's own data), at the same house number and street, is a true match in every train country (US 0.995,
    India 0.978). The transfer models accept it far less often in France (e.g. 76% for "Developpement" vs 98-99% for
    other fillers), because the cross-encoders never saw French fillers. Such a record is matched to its best S1.
    The pattern is re-derived from train labels at write time; nothing is forced if the data disagree."""
    cols = ["q", "s1", "nm_fill_extra", "pno_eq", "st_tset"]
    tr = pl.read_parquet(config.split_dir("train") / "pairs_feat.parquet", columns=cols + ["y"])
    trec = pl.read_parquet(config.split_dir("train") / "records.parquet", columns=["idx", "eid", "src", "country"])
    tr = tr.join(trec.filter(pl.col("eid").str.contains("-SYN")).select(pl.col("idx").alias("q")), on="q", how="anti") \
           .join(trec.filter(pl.col("src") == 1).select(pl.col("idx").alias("s1"), "country"), on="s1")
    pat = (pl.col("nm_fill_extra") == 1) & (pl.col("pno_eq") == 1) & (pl.col("st_tset") >= 80)
    rates = tr.filter(pat).group_by("country").agg(pl.len().alias("n"), pl.col("y").mean().alias("r"))
    if rates.height < trec["country"].n_unique() or rates["n"].min() < 1000 or rates["r"].min() < GUARD_HARD_POS:
        print(f"[write] hard-positive rule off (train {sorted(rates.rows())})", flush=True)
        return chosen
    te = pl.read_parquet(config.split_dir("test") / "pairs_feat.parquet", columns=cols)
    add = assigned.join(te, on=["q", "s1"], how="left").filter(pat).select(chosen.columns) \
                  .join(chosen, on=["s1_eid", "rec_eid"], how="anti")
    print(f"[write] hard-positive rule (unseen countries; filler at the same address, train truth {sorted(rates.rows())}): "
          f"{add.height:,} pairs added", flush=True)
    return pl.concat([chosen, add])


def _write(preds: dict, tag: str) -> None:
    """Country routing: S1s of countries present in train are decided by the full-feature model, S1s of any other
    country (absent from train, e.g. France) by the transfer model, each with its own validated decision rule.
    Blocking is per country, so a record's candidates never straddle the two routes."""
    from . import calibrate
    d = config.split_dir("test")
    rec = pl.read_parquet(d / "records.parquet")
    s1_ids = rec.filter(pl.col("src") == 1)["eid"]
    seen = set(pl.read_parquet(config.split_dir("train") / "records.parquet", columns=["country"])["country"].unique().to_list())
    s1_ctry = rec.filter(pl.col("src") == 1).select(pl.col("idx").alias("s1"), pl.col("country").is_in(list(seen)).alias("_seen"))
    routed = "transfer" in preds
    sels = []
    for fs, pred in preds.items():
        assigned = assemble_predictions(pred, pred["p"].to_numpy(), "test")
        if routed:
            assigned = assigned.join(s1_ctry, on="s1").filter(pl.col("_seen") == (fs == "full")).drop("_seen")
        t = tag + SUFFIX[fs]
        q, rule, param = calibrate.apply_saved(assigned["p"].to_numpy(), t)
        import os
        if fs == "transfer" and rule == "ef" and os.environ.get("ER_DELTA_TRANSFER"):
            # prior matching for unseen countries: the logit offset that makes the country's predicted cluster-size
            # distribution match the train countries' (the generator's cluster sizes are country-invariant:
            # US and India agree to 3 decimals); chosen label-free on test, see experiments/bottleneck/prior_match.py
            param = float(os.environ["ER_DELTA_TRANSFER"])
        print(f"[write] {fs} model ({t}) on {'seen' if fs == 'full' else 'unseen'}-country S1s: "
              f"decision rule from validation: rule={rule} param={param}", flush=True)
        chosen = calibrate._apply(assigned.select("s1", "s1_eid", "rec_eid"), q, rule, param).select("s1", "s1_eid", "rec_eid")
        if routed and fs == "transfer":
            if HARD_POS:
                chosen = hard_positive_rule(chosen, assigned)
            chosen = hard_negative_guard(conflict_guard(chosen, rec), rec)
        sels.append(chosen)
    sel = country_word_guard(pl.concat(sels), rec)
    write_id_lists(sel.select("s1_eid", "rec_eid"), s1_ids, config.OUT_DIR / "matching_results.tsv", "matched_entity_ids")
    # candidate set = pruned pairs fed to the model, expanded to duplicate members
    pruned = pl.read_parquet(d / "pruned.parquet").select("q", "s1")
    s1map = rec.filter(pl.col("src") == 1).select(pl.col("idx").alias("s1"), pl.col("eid").alias("s1_eid"))
    members = rec.filter(pl.col("src") > 1).select(pl.col("rep").alias("q"), pl.col("eid").alias("rec_eid"))
    cand = pruned.join(s1map, on="s1").join(members, on="q").select("s1_eid", "rec_eid")
    write_id_lists(cand, s1_ids, config.OUT_DIR / "candidate_pairs.tsv", "candidate_entity_ids")
    n_match = sel.height
    print(f"[write] matched pairs {n_match:,}; S1 with >=1 match {sel['s1_eid'].n_unique():,}/{len(s1_ids):,}; "
          f"candidate pairs {cand.height:,}", flush=True)
    validator = config.DATA_DIR.parent / "utils" / "validate_submission.py"
    subprocess.run([sys.executable, str(validator), "--matching", str(config.OUT_DIR / "matching_results.tsv"),
                    "--candidate", str(config.OUT_DIR / "candidate_pairs.tsv"), "--test-dir", str(config.DATA_DIR / "test")],
                   check=False)


if __name__ == "__main__":
    run()
