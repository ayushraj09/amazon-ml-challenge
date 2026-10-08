"""Stage S1: cheap-feature LightGBM that prunes blocking candidates (IMPLEMENTATION_PLAN §4.2).

Parameters are the Foursquare 4th-place level-1 LightGBM config (TheoViel/kaggle_foursquare,
notebooks/Classification.ipynb, LEVEL == 1): lr 0.05, num_leaves 500, reg_alpha 1, reg_lambda 10,
min_child_samples 1000, min_split_gain 0.01, min_child_weight 0.01, path_smooth 0.1,
n_estimators 8000 with early stopping 100; GroupKFold 5.
Train: out-of-fold p1 for every train pair (so pruned train candidates are not optimistically biased).
Test: average of the 5 fold models.
Keep per record: KEEP_RULE (top-2 by p1 AND p1 >= 0.01) -> work/<split>/pruned.parquet = candidate_pairs.tsv
"""
import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz

from . import config
from .features import _cp, closest_number
from .openset import record_features

LGB_L1 = dict(objective="binary", learning_rate=0.05, num_leaves=500, reg_alpha=1, reg_lambda=10,
              min_child_samples=1000, min_split_gain=0.01, min_child_weight=0.01, path_smooth=0.1,
              n_estimators=8000, n_jobs=config.N_THREADS, random_state=config.SEED, verbose=-1)
EARLY_STOP = 100
N_FOLDS = 5
# Stage-1 learner. Default: XGBoost on GPU with the Foursquare 4th-place LEVEL-1 XGBoost config verbatim
# (Classification.ipynb LEVEL==1 "xgb" params; model_zoo/xgb.py::train_xgb: n_estimators 4000, eval auc,
# EarlyStopping(rounds=100, min_delta=1e-5, save_best=True)). LightGBM (CPU) remains available via
# ER_STAGE1_MODEL=lgbm. GPU is used because CPU LightGBM on 83M rows took ~80 min per fold (measured).
STAGE1_MODEL = __import__("os").environ.get("ER_STAGE1_MODEL", "xgb")
XGB_L1 = dict(learning_rate=0.05, max_depth=10, min_child_weight=0.01, reg_alpha=0.01, reg_lambda=0.1,
              colsample_bytree=0.95, subsample=0.75, n_estimators=4000, objective="binary:logistic",
              eval_metric="auc", tree_method="hist", device="cuda", random_state=42)
TRAIN_SAMPLE = 0.2
# Keep rule (k_keep, p_keep, k_max, mode). "and": top-k_keep by p1 AND p1 >= p_keep; "or": top-k_keep OR p1 >= p_keep,
# capped at k_max. The competition update ("a smaller candidate set per Source 1 entity ranks higher") changed the
# objective from recall-at-any-size to size: the rule below is the smallest one on the measured size / recall-ceiling /
# F_shift frontier (experiments/candidate_size, IMPLEMENTATION_PLAN "Candidate-set size") with no measurable F_shift
# loss: 57.0 -> 6.37 candidates per validation S1, recall ceiling 0.99447 -> 0.99429, F_shift 0.98802 -> 0.98801.
# Second step (after retraining stage 2 on each set, work_c6 vs work_c4): top-2 AND p1 >= 0.01 gives stage-2 F_shift
# 0.98717 vs 0.98712 at 16/0.001 (F_val 0.98785 vs 0.98784) with 38% fewer candidates (test 7.88 -> ~4.9 per S1,
# France 11.5 -> 5.0; validation recall ceiling 0.9943 -> 0.9860, the lost pairs are ones the matcher never selected).
KEEP_RULE = (2, 0.01, 2, "and")
MAX_POS_LOSS = 0.003                          # former acceptance criterion (IMPLEMENTATION_PLAN §4.2), reported only
RULE_GRID = [(k, pk, km, "or") for k in (6, 8, 10, 12) for pk in (0.01, 0.003, 0.001) for km in (max(10, k), max(12, k), 16)] + \
            [(k, pk, k, "and") for k in (1, 2, 3, 4, 8, 16) for pk in (0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001)]
FEATS = ["dense", "r_dense", "b1", "r1", "n_ch", "ek_num_street", "ek_name", "bmax", "bmax_rank_q",
         "bmax_gap_best_q", "n_cand_q", "bmax_rank_s", "n_cand_s", "pno_eq", "pno_abs", "num_closest_abs",
         "legal_rel", "nm_tset", "nm_ratio", "st_tset", "ad_words_tset", "src", "was_indic", "addr_missing", "dup_count"]


def cheap_features(cands: pl.DataFrame, norm: pl.DataFrame, rec: pl.DataFrame) -> pl.DataFrame:
    n = norm.select("idx", "name_core", "street", "addr_clean", "pno", "legal", "all_nums", "was_indic", "addr_missing")
    df = (cands.join(n.rename({c: c + "_q" for c in n.columns if c != "idx"}), left_on="q", right_on="idx")
               .join(n.select("idx", "name_core", "street", "addr_clean", "pno", "legal")
                      .rename({c: c + "_s" for c in ["name_core", "street", "addr_clean", "pno", "legal"]}),
                     left_on="s1", right_on="idx")
               .join(rec.select("idx", "src", "dup_count"), left_on="q", right_on="idx"))
    strip = lambda c: pl.col(c).fill_null("").str.replace_all(r"\d+", " ")
    df = df.with_columns(
        # competition score: dense cosine (primary channel); pairs found only by other channels get 0
        pl.col("dense").fill_null(0.0).alias("bmax"),
        pl.col("dense", "b1").fill_null(0.0), pl.col("r_dense", "r1").fill_null(99),
        (pl.col("pno_q") == pl.col("pno_s")).cast(pl.Float32).alias("pno_eq"),
        (pl.col("pno_q") - pl.col("pno_s")).abs().cast(pl.Float32).alias("pno_abs"),
        pl.when(pl.col("legal_q") == pl.col("legal_s")).then(pl.when(pl.col("legal_q") == "").then(0).otherwise(1))
          .when((pl.col("legal_q") == "") | (pl.col("legal_s") == "")).then(2).otherwise(3).cast(pl.Int8).alias("legal_rel"),
        strip("addr_clean_q").alias("_wq"), strip("addr_clean_s").alias("_ws"),
        pl.col("was_indic_q").cast(pl.Float32).alias("was_indic"), pl.col("addr_missing_q").cast(pl.Float32).alias("addr_missing"),
        pl.col("ek_num_street", "ek_name").cast(pl.Float32),
    )
    s = lambda c: df[c].fill_null("").to_list()
    cd, _ = closest_number(df["all_nums_q"], df["pno_s"])
    df = df.with_columns(
        pl.Series("nm_tset", _cp(fuzz.token_set_ratio, s("name_core_q"), s("name_core_s"))),
        pl.Series("nm_ratio", _cp(fuzz.ratio, s("name_core_q"), s("name_core_s"))),
        pl.Series("st_tset", _cp(fuzz.token_set_ratio, s("street_q"), s("street_s"))),
        pl.Series("ad_words_tset", _cp(fuzz.token_set_ratio, s("_wq"), s("_ws"))),
        pl.Series("num_closest_abs", cd),
    )
    df = df.with_columns(
        pl.col("bmax").rank("ordinal", descending=True).over("q").cast(pl.Int16).alias("bmax_rank_q"),
        (pl.col("bmax") - pl.col("bmax").max().over("q")).alias("bmax_gap_best_q"),
        pl.len().over("q").cast(pl.Int16).alias("n_cand_q"),
        pl.col("bmax").rank("ordinal", descending=True).over("s1").cast(pl.Int32).alias("bmax_rank_s"),
        pl.len().over("s1").cast(pl.Int32).alias("n_cand_s"),
    )
    return df.select("q", "s1", *FEATS)


CHUNK_Q = 1_000_000


def restrict_channels(cands: pl.DataFrame, norm: pl.DataFrame) -> pl.DataFrame:
    """Measured on validation (IMPLEMENTATION_PLAN §4.1): for records WITH an address the word-name and exact-key
    channels add +0.00001 recall over dense (0.99958 -> 0.99959) at ~36% extra volume; for records WITHOUT an
    address they add +1.3 points (0.9337 -> 0.9467). So non-dense candidates are kept only for missing-address
    records; for all other records their scores stay as features on the dense pairs."""
    miss = norm.filter(pl.col("addr_missing")).select(pl.col("idx").alias("q"))
    keep = pl.col("dense").is_not_null() | pl.col("q").is_in(miss["q"].implode())
    return cands.filter(keep).with_columns(
        pl.col("ek_num_street", "ek_name").fill_null(False),
        (pl.col("dense").is_not_null().cast(pl.Int8) + pl.col("b1").is_not_null().cast(pl.Int8) +
         pl.col("ek_num_street").fill_null(False).cast(pl.Int8) + pl.col("ek_name").fill_null(False).cast(pl.Int8)).alias("n_ch"))


def cheap_features_chunked(cands: pl.DataFrame, norm: pl.DataFrame, rec: pl.DataFrame) -> pl.DataFrame:
    """cheap_features over record chunks (bounded memory at ~500M pairs); S1-side competition features are
    recomputed afterwards on the full table because an S1's candidate list spans chunks."""
    qs = cands["q"].unique().sort()
    parts = []
    for i in range(0, len(qs), CHUNK_Q):
        lo, hi = qs[i], qs[min(i + CHUNK_Q, len(qs)) - 1]
        parts.append(cheap_features(cands.filter(pl.col("q").is_between(lo, hi)), norm, rec))
        print(f"[stage1] cheap features {min(i + CHUNK_Q, len(qs)):,}/{len(qs):,} records", flush=True)
    df = pl.concat(parts)
    return df.with_columns(pl.col("bmax").rank("ordinal", descending=True).over("s1").cast(pl.Int32).alias("bmax_rank_s"),
                           pl.len().over("s1").cast(pl.Int32).alias("n_cand_s"))


def labels(split: str) -> pl.DataFrame:
    """(q, s1) positive pairs at representative level."""
    rec = pl.read_parquet(config.split_dir(split) / "records.parquet").select("idx", "eid", "rep", "src")
    gt = pl.read_parquet(config.split_dir(split) / "gt.parquet")
    s1 = rec.filter(pl.col("src") == 1).select(pl.col("eid").alias("s1_eid"), pl.col("idx").alias("s1"))
    oth = rec.filter(pl.col("src") > 1).select(pl.col("eid").alias("rec_eid"), pl.col("rep").alias("q"))
    return gt.join(s1, on="s1_eid").join(oth, on="rec_eid").select("q", "s1").unique().with_columns(pl.lit(1, pl.Int8).alias("y"))


def fold_of(df: pl.DataFrame, pos: pl.DataFrame) -> pl.Series:
    """Group = the record's true S1 if matched, else the record itself (a record never straddles folds)."""
    owner = pos.select("q", pl.col("s1").alias("owner"))
    g = df.select("q").join(owner, on="q", how="left", maintain_order="left").with_columns(pl.coalesce("owner", "q").alias("g"))
    return (g["g"].hash(config.SEED) % N_FOLDS).cast(pl.Int8)


def prune(df: pl.DataFrame, p: np.ndarray, rule=None) -> pl.DataFrame:
    k_keep, p_keep, k_max, mode = rule or KEEP_RULE
    df = df.with_columns(pl.Series("p1", p.astype(np.float32)))
    df = df.with_columns(pl.col("p1").rank("ordinal", descending=True).over("q").alias("_r"))
    return df.filter(_keep_expr(k_keep, p_keep, k_max, mode)).drop("_r")


def _keep_expr(k_keep, p_keep, k_max, mode) -> pl.Expr:
    top, hi = pl.col("_r") <= k_keep, pl.col("p1") >= p_keep
    return ((top & hi) if mode == "and" else (top | hi)) & (pl.col("_r") <= k_max)


def select_rule(q: np.ndarray, y: np.ndarray, p: np.ndarray) -> tuple:
    """Reports pairs and positive loss vs. blocking for every rule in RULE_GRID (measured on OOF p1) and returns
    KEEP_RULE, which was chosen on the downstream size / recall / F_shift frontier (see KEEP_RULE)."""
    t = pl.DataFrame({"q": q, "y": y, "p1": p}).with_columns(
        pl.col("p1").rank("ordinal", descending=True).over("q").alias("_r"))
    npos = int(y.sum())
    rows = []
    for k, pk, km, mode in sorted(set(RULE_GRID)):
        keep = _keep_expr(k, pk, km, mode)
        r = t.select(keep.sum().alias("n"), (keep & (pl.col("y") == 1)).sum().alias("pos")).row(0)
        print(f"[stage1] rule {mode} top{k}/p>={pk}/cap{km}: pairs {r[0]:,} pos loss {1 - r[1] / npos:.5f}", flush=True)
    print(f"[stage1] keep rule {KEEP_RULE}", flush=True)
    return KEEP_RULE




def _finish(split: str, df: pl.DataFrame, p: np.ndarray, rule) -> None:
    """E1/E2 open-set features from each record's FULL candidate score list, then prune and save."""
    d = config.split_dir(split)
    logit = np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
    rf = record_features(df.select("q").with_columns(pl.Series("l1", logit.astype(np.float32))), "l1")
    has_y = "y" in df.columns
    pruned = prune(df.select("q", "s1", *FEATS, *(["y"] if has_y else [])), p, rule)
    pruned = pruned.join(rf, on="q", how="left", maintain_order="left").with_columns(
        (pl.col("p1").log() - (1 - pl.col("p1")).log() - pl.col("e2_top_logit")).alias("e2_gap_to_top"))
    pruned.write_parquet(d / "pruned.parquet")
    msg = f"[stage1] {split}: {df.height:,} -> {pruned.height:,} pairs ({pruned.height / df['q'].n_unique():.2f}/record)"
    if has_y:
        msg += f"; positives kept {pruned['y'].sum() / max(df['y'].sum(), 1):.5f} of blocked positives"
    print(msg, flush=True)


def _load(split: str) -> pl.DataFrame:
    """Cheap features for all candidates; cached to disk so a restart does not recompute them (~40 min)."""
    d = config.split_dir(split)
    cache = d / "stage1_cheap.parquet"
    if cache.exists():
        print(f"[stage1] {split}: loading cached cheap features", flush=True)
        return pl.read_parquet(cache)
    df = _compute(split)
    df.write_parquet(cache)
    return df


def _compute(split: str) -> pl.DataFrame:
    d = config.split_dir(split)
    rec = pl.read_parquet(d / "records.parquet").filter(pl.col("idx") == pl.col("rep"))
    norm = pl.read_parquet(d / "norm.parquet")
    cands = restrict_channels(pl.read_parquet(d / "cands.parquet"), norm)
    print(f"[stage1] {split}: {cands.height:,} candidate pairs after channel restriction", flush=True)
    return cheap_features_chunked(cands, norm, rec)


def run_train() -> None:
    import json
    df = _load("train")
    pos = labels("train")
    df = df.join(pos, on=["q", "s1"], how="left", maintain_order="left").with_columns(pl.col("y").fill_null(0))
    X = df.select(FEATS).to_numpy().astype(np.float32)
    y = df["y"].to_numpy()
    fold = fold_of(df, pos).to_numpy()
    in_sample = np.random.default_rng(config.SEED).random(len(y)) < TRAIN_SAMPLE
    oof = np.zeros(len(y), dtype=np.float32)
    for k in range(N_FOLDS):
        tr, va = in_sample & (fold != k), fold == k
        es = va & in_sample   # full in-sample held-out fold (a 20% subsample made the saturated AUC noisy
                              # enough that early stopping stopped triggering: measured, reverted)
        if STAGE1_MODEL == "xgb":
            import xgboost as xgb
            cb = xgb.callback.EarlyStopping(rounds=100, min_delta=1e-5, save_best=True, maximize=True,
                                            data_name="validation_0", metric_name="auc")
            m = xgb.XGBClassifier(**XGB_L1, callbacks=[cb])
            m.fit(X[tr], y[tr], eval_set=[(X[es], y[es])], verbose=100)
            oof[va] = m.predict_proba(X[va])[:, 1]
            m.save_model(str(config.WORK_DIR / f"stage1_fold{k}.json"))        # saved immediately
            print(f"[stage1] fold {k}: best_iter {m.best_iteration} auc {m.best_score:.5f}", flush=True)
        else:
            m = lgb.LGBMClassifier(**LGB_L1)
            m.fit(X[tr], y[tr], eval_set=[(X[es], y[es])], eval_metric="auc",
                  callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(100)])
            oof[va] = m.predict_proba(X[va])[:, 1]
            m.booster_.save_model(str(config.WORK_DIR / f"stage1_fold{k}.txt"))   # saved immediately
            print(f"[stage1] fold {k}: best_iter {m.best_iteration_} auc {m.best_score_['valid_0']['auc']:.5f}", flush=True)
        del m
    del X
    np.save(config.WORK_DIR / "oof_stage1.npy", oof)
    rule = select_rule(df["q"].to_numpy(), y, oof)
    (config.WORK_DIR / "stage1_rule.json").write_text(json.dumps(dict(zip(["k_keep", "p_keep", "k_max", "mode"], rule))))
    _finish("train", df, oof, rule)


def run_test() -> None:
    import gc
    import json
    gc.collect()
    rule = json.loads((config.WORK_DIR / "stage1_rule.json").read_text())
    rule = (rule["k_keep"], rule["p_keep"], rule["k_max"], rule.get("mode", "or"))
    df = _load("test")
    X = df.select(FEATS).to_numpy().astype(np.float32)
    p = np.zeros(len(X), dtype=np.float64)
    for k in range(N_FOLDS):
        if STAGE1_MODEL == "xgb":
            import xgboost as xgb
            m = xgb.XGBClassifier()
            m.load_model(str(config.WORK_DIR / f"stage1_fold{k}.json"))
            p += m.predict_proba(X)[:, 1] / N_FOLDS
        else:
            p += lgb.Booster(model_file=str(config.WORK_DIR / f"stage1_fold{k}.txt")).predict(X) / N_FOLDS
    del X
    _finish("test", df, p, rule)


def run() -> None:
    """Train and test sides run as separate functions so no large train object is alive during test."""
    run_train()
    import gc
    gc.collect()
    run_test()

if __name__ == "__main__":
    run()
