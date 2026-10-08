"""Stage S2: rich-feature matcher + validation scoring (IMPLEMENTATION_PLAN §5.2, §6, §7.2).

LightGBM parameters are the Foursquare 4th-place level-2 config verbatim (TheoViel/kaggle_foursquare,
notebooks/Classification.ipynb, LEVEL == 2): lr 0.05, num_leaves 511, colsample_bytree 0.5, reg_alpha 1,
reg_lambda 70, min_child_samples 2000, min_split_gain 0.02, min_child_weight 0.03, path_smooth 0.2,
n_estimators 8000, early stopping 100, eval metric AUC. GroupKFold by owner (5 folds).

Validation: S1 entities flagged is_val_s1 (20%). Two scores (IMPLEMENTATION_PLAN §6.2):
  F_val   - real validation records only (synthetic near-misses removed from the candidate pool)
  F_shift - with synthetic near-misses (test near-miss rate), countries re-weighted to the test mix
            (US 0.38 + France-proxy 0.15 on US, India 0.47).
"""
import lightgbm as lgb
import numpy as np
import polars as pl

from . import config, decide, features
from .evaluate import macro_f05
from .stage1 import N_FOLDS, fold_of, labels

LGB_L2 = dict(objective="binary", learning_rate=0.05, num_leaves=511, colsample_bytree=0.5, reg_alpha=1,
              reg_lambda=70, min_child_samples=2000, min_split_gain=0.02, min_child_weight=0.03, path_smooth=0.2,
              n_estimators=8000, n_jobs=config.N_THREADS, random_state=config.SEED, verbose=-1)
EARLY_STOP = 100
# Foursquare 4th-place level-2 XGBoost config verbatim (Classification.ipynb LEVEL==2; model_zoo/xgb.py:
# n_estimators 10000, early_stopping_rounds 20, eval_metric auc, hist on GPU, enable_categorical)
XGB_L2 = dict(learning_rate=0.05, max_depth=15, colsample_bytree=0.5, reg_alpha=1, reg_lambda=10,
              min_child_weight=0.1, gamma=0.1, n_estimators=4000, eval_metric="auc",
              tree_method="hist", device=__import__("os").environ.get("ER_XGB_DEVICE", "cuda"),
              enable_categorical=True, random_state=42, n_jobs=config.N_THREADS)
# ER_XGB_DEVICE=cpu runs the same hist algorithm on CPU (used while the GPU is occupied by another stage)
# training settings of model_zoo/xgb.py::train_xgb: EarlyStopping(rounds=100, min_delta=1e-5, save_best=True)
TEST_MIX = {"US": 0.38 + 0.15, "India": 0.47}          # used when no France-proxy group was generated
TEST_MIX_PROXY = {"US": 0.38, "France-proxy": 0.15, "India": 0.47}


def eval_groups(s1: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """s1: s1_eid, country -> adds 'group' (France-proxy = US val half with synthetics at France's rate)."""
    import json
    meta = config.split_dir("train") / "syn_meta.json"
    if meta.exists() and json.loads(meta.read_text()).get("france_proxy"):
        from .synth import PROXY_HALF
        g = pl.when((pl.col("country") == "US") & (pl.col("s1_eid").hash(config.SEED + 17) % 2 == PROXY_HALF)) \
              .then(pl.lit("France-proxy")).otherwise(pl.col("country"))
        return s1.with_columns(g.alias("group")), TEST_MIX_PROXY
    return s1.with_columns(pl.col("country").alias("group")), TEST_MIX
CAT_FEATS = ["pno_edit_type", "legal_rel"]
# Primary stage-2 learner (drives v1, the cross-encoder selection and the stacked model): XGBoost on GPU
# (measured: GPU XGBoost fits a stage-1 fold in ~2 min vs ~80 min for CPU LightGBM at equal AUC).
# The other learner runs in parallel as the blend partner. Both use the Foursquare level-2 configs verbatim.
PRIMARY = __import__("os").environ.get("ER_STAGE2_MODEL", "xgb")
# blend partner: CatBoost on GPU (measured at 4.5M x 80: 0.20 s/round vs 0.28 CUDA-LightGBM, equal AUC; CPU
# LightGBM on 125M rows needed ~146 GB RSS and ~10+ h)
SECONDARY = __import__("os").environ.get("ER_STAGE2_SECONDARY", "cat")
CHUNK_Q = 1_500_000


def build(split: str) -> pl.DataFrame:
    d = config.split_dir(split)
    pruned = pl.read_parquet(d / "pruned.parquet")
    norm = pl.read_parquet(d / "norm.parquet")
    rec = pl.read_parquet(d / "records.parquet")
    info = rec.filter(pl.col("idx") == pl.col("rep")).select("idx", "src", "dup_count")
    # the stage-1 file carries the label on train; it is re-joined AFTER feature building (never a feature)
    base = pruned.drop([c for c in ["src", "dup_count", "y"] if c in pruned.columns]).sort("q")
    base = _add_name_frequency(base, rec, norm)
    # chunk by record so every record's candidates stay together (competition features need the full list)
    qs = base["q"].unique().sort()
    parts = []
    for i in range(0, len(qs), CHUNK_Q):
        lo, hi = qs[i], qs[min(i + CHUNK_Q, len(qs)) - 1]
        b = base.filter(pl.col("q").is_between(lo, hi))
        parts.append(features.compute(b, norm, info))
        print(f"[features] {split}: {min(i + CHUNK_Q, len(qs)):,}/{len(qs):,} records", flush=True)
    df = pl.concat(parts)
    df = features.add_competition(df, "p1")
    df = _add_filler_features(df, split, rec, norm)
    evm = d / "evm.parquet"
    if evm.exists():   # E4 Extreme Value Machine inclusion probability
        df = df.join(pl.read_parquet(evm), on=["q", "s1"], how="left", maintain_order="left")
    return df


FILL_MIN_SHARE, FILL_MIN_RATIO = 0.01, 0.25
REAL_MIN_LEN = int(__import__("os").environ.get("ER_REAL_MIN_LEN", "1"))


def _mine_fillers(df: pl.DataFrame, s1tok: pl.DataFrame, rec: pl.DataFrame) -> dict:
    """Per country, the generic words the data generator puts in place of a business's category word
    ("Ridge Pediatrics Inc" -> "Ridge Inc Services"; France: "ER Amis" -> "ER & Fils"). Label-free: among candidate
    pairs at the same address (same number and street) whose names share the first token but differ, count the tokens
    only the record has; a filler is frequent there (>= 1% of those tokens) and much rarer in the clean S1 names
    (added-count / S1-name count >= 0.25), which separates fillers from category words (France: club 0.15, ecole 0.14).
    Run unlabelled on the train countries it recovers exactly the fillers mined from labelled true pairs
    (US and India: center, services, service, partners)."""
    m = df.filter((pl.col("nm_tset") >= 40) & (pl.col("nm_tset") < 90) & (pl.col("nm_first_eq") == 1) &
                  (pl.col("pno_eq") == 1) & (pl.col("st_tset") >= 80) & (pl.col("addr_missing") == 0))
    x = m.select("country", pl.col("_tq").list.set_difference(pl.col("_ts")).alias("extra")).explode("extra").drop_nulls()
    ex = x.group_by("country", "extra").len("n_extra").with_columns((pl.col("n_extra") / pl.col("n_extra").sum().over("country")).alias("share"))
    ex = ex.join(s1tok, on=["country", "extra"], how="left").with_columns(pl.col("n_s1").fill_null(0))
    keep = ex.filter((pl.col("share") >= FILL_MIN_SHARE) & (pl.col("n_extra") / (pl.col("n_s1") + 1) >= FILL_MIN_RATIO) &
                     (pl.col("extra").str.len_chars() >= 2))
    return {c: sorted(g["extra"].to_list()) for (c,), g in keep.group_by("country")}


def _add_filler_features(df: pl.DataFrame, split: str, rec: pl.DataFrame, norm: pl.DataFrame) -> pl.DataFrame:
    """nm_fill_extra: every token only the record has is a filler of its country; nm_fill_subset: the record's
    non-filler tokens all occur in the S1 name (record = S1 with its category word replaced by fillers)."""
    import json
    core = norm.select("idx", pl.col("name_core").fill_null("").str.split(" ").alias("_t"))
    ctry = rec.filter(pl.col("src") == 1).select(pl.col("idx").alias("s1"), "country")
    x = df.join(core.rename({"idx": "q", "_t": "_tq"}), on="q", how="left", maintain_order="left") \
          .join(core.rename({"idx": "s1", "_t": "_ts"}), on="s1", how="left", maintain_order="left") \
          .join(ctry, on="s1", how="left", maintain_order="left")
    path = config.WORK_DIR / "fillers.json"
    fillers = json.loads(path.read_text()) if path.exists() else {}
    new = [c for c in x["country"].unique().drop_nulls().to_list() if c not in fillers]
    if new:   # countries first seen in this split are mined from it (train countries from train, others from test)
        s1tok = core.join(rec.filter(pl.col("src") == 1).select("idx", "country"), on="idx") \
                    .select("country", pl.col("_t").list.unique().alias("extra")).explode("extra").group_by("country", "extra").len("n_s1")
        mined = _mine_fillers(x.filter(pl.col("country").is_in(new)), s1tok, rec)
        for c in new:
            fillers[c] = mined.get(c, [])
        path.write_text(json.dumps(fillers, ensure_ascii=False, indent=1))
        print(f"[features] fillers mined from {split}: {{{', '.join(f'{c}: {fillers[c]}' for c in new)}}}", flush=True)
    # real-word vocabulary of each country = words of its clean S1 names (this split's S1s; label-free). A record whose
    # extra words are REAL words is a sibling business (train: India same number 90% -> 9% true, US other number
    # 53% -> 4%), a record whose extra words are garbles is the same business. The model already knows English and
    # Indian words; this makes the rule explicit so it transfers to French names. A word must have >= 4 letters: short
    # garbles ("ina", "oas") collide with real short tokens (found in the qualitative audit of v3). ER_REAL_MIN_LEN=4
    # enables that fix (v5); the default 1 is the v4 definition (v5 lost on the public leaderboard).
    vf = core.join(rec.filter(pl.col("src") == 1).select("idx", "country"), on="idx") \
             .select("country", pl.col("_t").list.unique().alias("_w")).explode("_w").group_by("country", "_w").len("_vf")
    xr = x.select(pl.int_range(pl.len()).alias("_r"), "country", pl.col("_tq").list.set_difference(pl.col("_ts")).alias("_w")) \
          .explode("_w").drop_nulls("_w").join(vf, on=["country", "_w"], how="left").with_columns(pl.col("_vf").fill_null(0)) \
          .group_by("_r").agg(((pl.col("_vf") >= 3) & (pl.col("_w").str.len_chars() >= REAL_MIN_LEN)).all().cast(pl.Float32).alias("nm_extra_real"),
                              pl.col("_vf").min().log1p().cast(pl.Float32).alias("nm_extra_min_vf_log"))
    x = x.with_columns(pl.int_range(pl.len()).alias("_r")).join(xr, on="_r", how="left", maintain_order="left").drop("_r")
    fl = pl.DataFrame({"country": list(fillers), "_fill": list(fillers.values())}, schema={"country": pl.Utf8, "_fill": pl.List(pl.Utf8)})
    x = x.join(fl, on="country", how="left", maintain_order="left").with_columns(pl.col("_fill").fill_null([]))
    extra = pl.col("_tq").list.set_difference(pl.col("_ts"))
    core_q = pl.col("_tq").list.set_difference(pl.col("_fill"))
    x = x.with_columns(
        ((extra.list.len() > 0) & (extra.list.set_difference(pl.col("_fill")).list.len() == 0)).cast(pl.Float32).alias("nm_fill_extra"),
        ((core_q.list.len() > 0) & (core_q.list.set_difference(pl.col("_ts")).list.len() == 0) &
         (extra.list.len() > 0)).cast(pl.Float32).alias("nm_fill_subset"))
    return x.drop("_tq", "_ts", "country", "_fill")


def _add_name_frequency(base: pl.DataFrame, rec: pl.DataFrame, norm: pl.DataFrame) -> pl.DataFrame:
    """Chain-ness (IMPLEMENTATION_PLAN §5.1): how many S1 of the same country share the S1's / the record's
    core name. A unique name can be matched on the name alone (e.g. a record without address); a chain name
    needs the address. Found missing in error analysis: 64% of validation FNs were address-missing records."""
    key = norm.select("idx", "name_core").join(rec.select("idx", "src", "country"), on="idx")
    freq = key.filter(pl.col("src") == 1).group_by("country", "name_core").len("n_s1")
    s1f = key.filter(pl.col("src") == 1).join(freq, on=["country", "name_core"]).select(
        pl.col("idx").alias("s1"), pl.col("n_s1").cast(pl.Float32).alias("s1_name_freq"))
    qf = key.filter(pl.col("src") > 1).join(freq, on=["country", "name_core"], how="left", maintain_order="left").select(
        pl.col("idx").alias("q"), pl.col("n_s1").fill_null(0).cast(pl.Float32).alias("q_name_s1_freq"))
    out = base.join(s1f, on="s1", how="left", maintain_order="left").join(qf, on="q", how="left", maintain_order="left")
    return out.with_columns((pl.col("s1_name_freq") == 1).cast(pl.Float32).alias("s1_name_unique"))


# Computed but not used as model inputs: they do not transfer to a country absent from train (France). Measured with
# leave-one-country-out training (experiments/transfer/loco.py): dropping legal_pair_id (categorical identity of the
# legal-form pair; French pairs are unseen categories) raised the mean US<->India transfer F0.5 0.95995 -> 0.96298;
# also dropping the record-style flags (source id, duplicate count, upper-case style, junk prefix, honorific) -> 0.96331.
# Dropping the stage-1 score features (0.94898) or the signed number shift (0.96267) hurt, so they stay.
# They cost in-distribution accuracy (stage-2 F_shift 0.98717 full vs 0.98577 transfer), so two feature sets are
# trained and ROUTED by country (write_outputs): countries present in train use "full", any other country (France)
# uses "transfer". Tags: stage2 / stage2x (full), stage2_tr / stage2x_tr (transfer).
NON_TRANSFERABLE = {"legal_pair_id", "src", "dup_count", "upper_style", "had_junk", "had_honorific"}
FEATSETS = {"full": set(), "transfer": NON_TRANSFERABLE}
SUFFIX = {"full": "", "transfer": "_tr"}


def feature_cols(df: pl.DataFrame, featset: str = "full") -> list[str]:
    cols = [c for c in df.columns if c not in ("q", "s1", "y", "fold") and not c.startswith("y_") and c not in FEATSETS[featset]]
    leak = [c for c in cols if c in ("y", "label") or c.endswith("_right")]
    assert not leak, f"label/duplicate columns would become features: {leak}"
    return cols


X_COLS = ["x_logit", "x_ova", "x_pi", "x_q"]
SIB_P = 0.5   # a sibling = another record whose argmax is this S1 with stage-2 p2 > SIB_P


def add_siblings(df: pl.DataFrame, split: str) -> pl.DataFrame:
    """IMPLEMENTATION_PLAN §7.1 sibling consensus (uses the 'p2_stack' column = stage-2 OOF/test probability).
    For pair (q, j): other records confidently assigned to S1 j, and how many share q's core name / number.
    Targets the address-missing FNs (a name-only record whose exact name is carried by j's addressed siblings)."""
    norm = pl.read_parquet(config.split_dir(split) / "norm.parquet", columns=["idx", "name_core", "pno"])
    best = (df.select("q", "s1", "p2_stack").sort("p2_stack", descending=True)
              .group_by("q", maintain_order=True).first())
    sib = (best.filter(pl.col("p2_stack") > SIB_P)
               .join(norm.rename({"idx": "q", "name_core": "sib_name", "pno": "sib_pno"}), on="q")
               .select(pl.col("q").alias("sib_q"), "s1", pl.col("p2_stack").alias("sib_p"), "sib_name", "sib_pno"))
    me = df.select("q", "s1").join(norm.rename({"idx": "q", "name_core": "q_name", "pno": "q_pno"}), on="q",
                                   how="left", maintain_order="left")
    parts = []
    qs = me["q"].unique().sort()
    for i in range(0, len(qs), 2_000_000):
        lo, hi = qs[i], qs[min(i + 2_000_000, len(qs)) - 1]
        m = me.filter(pl.col("q").is_between(lo, hi)).join(sib, on="s1").filter(pl.col("sib_q") != pl.col("q"))
        parts.append(m.group_by("q", "s1").agg(
            pl.len().cast(pl.Float32).alias("sib_n"),
            pl.col("sib_p").max().alias("sib_max_p"),
            (pl.col("sib_name") == pl.col("q_name")).sum().cast(pl.Float32).alias("sib_same_name"),
            ((pl.col("sib_pno") == pl.col("q_pno")) & pl.col("q_pno").is_not_null()).sum().cast(pl.Float32).alias("sib_same_pno")))
    agg = pl.concat(parts) if parts else pl.DataFrame(schema={"q": pl.Int32, "s1": pl.Int32})
    out = df.join(agg, on=["q", "s1"], how="left", maintain_order="left")
    return out.with_columns(pl.col("sib_n", "sib_same_name", "sib_same_pno").fill_null(0.0),
                            pl.col("sib_max_p").fill_null(0.0))


def add_x(df: pl.DataFrame, split: str, p2: np.ndarray) -> pl.DataFrame:
    """Stacked cross-encoder features + D2 rank-1 consensus between stage-2 LightGBM and the cross-encoder.
    Pairs of records that were not uncertain enough to be cross-encoded keep nulls (same rule on train/test)."""
    x = pl.read_parquet(config.split_dir(split) / "xenc.parquet")
    # attach p2 BEFORE the join: p2 is aligned to df's current row order
    df = df.with_columns(pl.Series("p2_stack", p2.astype(np.float32))).join(x, on=["q", "s1"], how="left", maintain_order="left")
    df = df.with_columns(
        (pl.col("p2_stack") == pl.col("p2_stack").max().over("q")).alias("_a2"),
        (pl.col("x_logit") == pl.col("x_logit").max().over("q")).alias("_ax"))
    df = df.with_columns(
        pl.when(pl.col("x_logit").is_null()).then(None).otherwise((pl.col("_a2") & pl.col("_ax")).cast(pl.Float32))
          .alias("d2_agree"),
        pl.col("x_logit").rank("ordinal", descending=True).over("q").cast(pl.Float32).alias("x_rank_q"),
        (pl.col("x_logit") - pl.col("x_logit").max().over("q")).alias("x_gap_best_q"))
    df = add_siblings(df, split)
    return df.drop("_a2", "_ax", "p2_stack")


EASY_P1, EASY_KEEP = 0.02, 0.5   # IMPLEMENTATION_PLAN §5.2: keep all positives and 50% of easy negatives
                                 # (stage-1 p1 < 0.02), weighted x2; OOF predictions still cover every row


def _train_mask_weights(df: pl.DataFrame, fold: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    y = df["y"].to_numpy()
    easy_neg = (df["p1"].to_numpy() < EASY_P1) & (y == 0)
    keep = ~easy_neg | (np.random.default_rng(config.SEED + 100 + k).random(len(y)) < EASY_KEEP)
    tr = (fold != k) & keep
    w = np.where(easy_neg, 1.0 / EASY_KEEP, 1.0).astype(np.float32)
    return tr, w


def fit_oof(df: pl.DataFrame) -> tuple[np.ndarray, list]:
    pos = labels("train")
    fold = fold_of(df, pos).to_numpy()
    cols = feature_cols(df)
    X = df.select(cols).to_pandas()
    for c in CAT_FEATS:
        if c in X:
            X[c] = X[c].astype("category")
    y = df["y"].to_numpy()
    oof = np.zeros(len(y), dtype=np.float32)
    models = []
    for k in range(N_FOLDS):
        va = fold == k
        tr, w = _train_mask_weights(df, fold, k)
        es = va   # full held-out fold, as in the Foursquare reference (a subsample made early stopping unreliable)
        m = lgb.LGBMClassifier(**LGB_L2)
        m.fit(X[tr], y[tr], sample_weight=w[tr], eval_set=[(X[es], y[es])], eval_metric="auc",
              callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(500)])
        oof[va] = m.predict_proba(X[va])[:, 1]
        models.append(m)
        print(f"[train] fold {k}: best_iter {m.best_iteration_} auc {m.best_score_['valid_0']['auc']:.6f}", flush=True)
    return oof, models


def fit_oof_xgb(df: pl.DataFrame, featset: str = "full") -> tuple[np.ndarray, list]:
    import xgboost as xgb
    pos = labels("train")
    fold = fold_of(df, pos).to_numpy()
    cols = feature_cols(df, featset)
    X = df.select(cols).to_pandas()
    for c in CAT_FEATS:
        if c in X:
            X[c] = X[c].astype("category")
    y = df["y"].to_numpy()
    oof = np.zeros(len(y), dtype=np.float32)
    models = []
    for k in range(N_FOLDS):
        va = fold == k
        tr, w = _train_mask_weights(df, fold, k)
        es = va   # full held-out fold, as in the Foursquare reference (a subsample made early stopping unreliable)
        cb = xgb.callback.EarlyStopping(rounds=100, min_delta=1e-5, save_best=True, maximize=True,
                                        data_name="validation_0", metric_name="auc")
        m = xgb.XGBClassifier(**XGB_L2, callbacks=[cb])
        m.fit(X[tr], y[tr], sample_weight=w[tr], eval_set=[(X[es], y[es])], verbose=100)
        oof[va] = m.predict_proba(X[va])[:, 1]
        models.append(m)
        print(f"[train] xgb fold {k}: best_iter {m.best_iteration} auc {m.best_score:.6f}", flush=True)
    return oof, models


def _fit_and_save(df: pl.DataFrame, model: str, tag: str, featset: str = "full") -> np.ndarray:
    """Fit 5-fold OOF with the given learner and save the fold models as <tag>_fold<k>.(json|txt)."""
    if model == "xgb":
        oof, models = fit_oof_xgb(df, featset)
    else:
        assert featset == "full", "feature sets are implemented for the XGBoost learner"
        oof, models = {"cat": fit_oof_cat, "lgbm": fit_oof}[model](df)
    np.save(config.WORK_DIR / f"oof_{tag}.npy", oof)
    for i, m in enumerate(models):
        if model == "xgb":
            m.save_model(str(config.WORK_DIR / f"{tag}_fold{i}.json"))
        elif model == "cat":
            m.save_model(str(config.WORK_DIR / f"{tag}_fold{i}.cbm"))
        else:
            m.booster_.save_model(str(config.WORK_DIR / f"{tag}_fold{i}.txt"))
    return oof


CAT_L2 = dict(depth=12, l2_leaf_reg=0.1, min_data_in_leaf=2000,      # Classification.ipynb LEVEL==2 "catboost"
              n_estimators=20000, loss_function="Logloss", eval_metric="AUC", task_type="GPU",
              random_state=42, allow_writing_files=False)            # model_zoo/catboost.py::train_catboost


def fit_oof_cat(df: pl.DataFrame) -> tuple[np.ndarray, list]:
    from catboost import CatBoostClassifier
    pos = labels("train")
    fold = fold_of(df, pos).to_numpy()
    cols = feature_cols(df)
    X = df.select(cols).to_pandas()
    cats = [c for c in CAT_FEATS if c in X]
    for c in cats:
        X[c] = X[c].fillna(-1).astype("int64")
    y = df["y"].to_numpy()
    oof = np.zeros(len(y), dtype=np.float32)
    models = []
    for k in range(N_FOLDS):
        va = fold == k
        tr, w = _train_mask_weights(df, fold, k)
        m = CatBoostClassifier(**CAT_L2)
        m.fit(X[tr], y[tr], sample_weight=w[tr], eval_set=[(X[va], y[va])], cat_features=cats,
              early_stopping_rounds=100, verbose=100)                  # train_catboost: early_stopping_rounds=100
        oof[va] = m.predict_proba(X[va])[:, 1]
        models.append(m)
        print(f"[train] cat fold {k}: best_iter {m.get_best_iteration()} auc {m.get_best_score()['validation']['AUC']:.6f}", flush=True)
    return oof, models


ST_HI, ST_LO, ST_ITERS = 0.99, 0.01, 2


def run_selftrain(featset: str = "transfer") -> None:
    """Self-training for countries ABSENT from train (unsupervised domain adaptation; no manual labels).
    The stacked model of `featset` scores the unseen-country test pairs; confident predictions become pseudo-labels
    (p >= 0.99 -> 1, p <= 0.01 -> 0); the 5 fold models are refit on their train folds plus all pseudo-labelled
    test pairs; repeated twice. Train OOF predictions (and so the decision rule, re-tuned here) stay honest: test
    pairs never enter a validation fold. Measured by leave-one-country-out simulation with the real labels of the
    held-out country used only for scoring (experiments/transfer/selftrain*.py): stage-2 US->India 0.96514 ->
    0.96789, India->US 0.97836 -> 0.98067."""
    import shutil
    import xgboost as xgb
    from . import calibrate
    from .write_outputs import _predict
    sfx = SUFFIX[featset]
    tag = "stage2x" + sfx
    seen = set(pl.read_parquet(config.split_dir("train") / "records.parquet", columns=["country"])["country"].unique().to_list())
    te = pl.read_parquet(config.split_dir("test") / "pairs_feat.parquet")
    p2 = pl.read_parquet(config.split_dir("test") / f"pred_stage2{sfx}.parquet")
    assert te.height == p2.height and (te["q"] == p2["q"]).all() and (te["s1"] == p2["s1"]).all()
    te = add_x(te, "test", p2["p"].to_numpy())
    trec = pl.read_parquet(config.split_dir("test") / "records.parquet").filter(pl.col("src") == 1)
    unseen_s1 = trec.filter(~pl.col("country").is_in(list(seen))).select(pl.col("idx").alias("s1"))
    te = te.join(unseen_s1, on="s1", how="semi")
    if te.height == 0:
        print("[selftrain] no country absent from train: nothing to adapt", flush=True)
        return
    tr = pl.read_parquet(config.split_dir("train") / "pairs_feat.parquet")
    tr = add_x(tr, "train", np.load(config.WORK_DIR / f"oof_stage2{sfx}.npy"))
    cols = feature_cols(tr, featset)
    fold = fold_of(tr, labels("train")).to_numpy()
    y = tr["y"].to_numpy()

    def frame(df):
        X = df.select(cols).to_pandas()
        for c in CAT_FEATS:
            if c in X:
                X[c] = X[c].astype("category")
        return X
    Xtr, Xte = frame(tr), frame(te)
    for k in range(N_FOLDS):                    # keep the un-adapted transfer models
        src = config.WORK_DIR / f"{tag}_fold{k}.json"
        shutil.copy(src, config.WORK_DIR / f"{tag}_noselftrain_fold{k}.json")
    shutil.copy(config.WORK_DIR / f"oof_{tag}.npy", config.WORK_DIR / f"oof_{tag}_noselftrain.npy")
    p = _predict(te, tag)
    for it in range(1, ST_ITERS + 1):
        lab = np.where(p >= ST_HI, 1, np.where(p <= ST_LO, 0, -1))
        use = lab >= 0
        Xp, yp = Xte[use], lab[use]
        print(f"[selftrain] iteration {it}: {use.mean():.3f} of {te.height:,} unseen-country test pairs pseudo-labelled "
              f"({(yp == 1).sum():,} positive, {(yp == 0).sum():,} negative)", flush=True)
        oof = np.zeros(len(y), dtype=np.float32)
        ps = []
        for k in range(N_FOLDS):
            trm, w = _train_mask_weights(tr, fold, k)
            va = fold == k
            import pandas as pd
            Xk = pd.concat([Xtr[trm], Xp], ignore_index=True)
            yk = np.concatenate([y[trm], yp])
            wk = np.concatenate([w[trm], np.ones(len(yp), dtype=np.float32)])
            cb = xgb.callback.EarlyStopping(rounds=100, min_delta=1e-5, save_best=True, maximize=True,
                                            data_name="validation_0", metric_name="auc")
            m = xgb.XGBClassifier(**XGB_L2, callbacks=[cb])
            m.fit(Xk, yk, sample_weight=wk, eval_set=[(Xtr[va], y[va])], verbose=0)
            oof[va] = m.predict_proba(Xtr[va])[:, 1]
            m.save_model(str(config.WORK_DIR / f"{tag}_fold{k}.json"))
            ps.append(m.predict_proba(Xte)[:, 1])
            print(f"[selftrain] iteration {it} fold {k}: best_iter {m.best_iteration} auc {m.best_score:.6f}", flush=True)
        p = np.mean(ps, axis=0)
        np.save(config.WORK_DIR / f"oof_{tag}.npy", oof)
    calibrate.tune(tr.select("q", "s1"), oof, tag)


def run_secondary(use_x: bool = False) -> None:
    """The non-primary learner (blend partner), run in parallel with the main chain."""
    tag = ("stage2x_" if use_x else "stage2_") + SECONDARY
    df = pl.read_parquet(config.split_dir("train") / "pairs_feat.parquet")
    if use_x:
        df = add_x(df, "train", np.load(config.WORK_DIR / "oof_stage2.npy"))
    oof = _fit_and_save(df, SECONDARY, tag)
    assigned = assemble_predictions(df, oof, "train")
    r = evaluate_val(assigned, "ef", 0.5, with_syn=False)
    print(f"[val:{tag}] expected-F0.5 rule, uncalibrated: F_val {r['f05']:.5f} (test-mix {r['f05_test_mix']:.5f})", flush=True)


def run_blend(use_x: bool = True) -> None:
    """Blend LightGBM and XGBoost OOF probabilities; weight chosen by validation log-loss (proper scoring
    rule), then the decision rule is re-tuned on the blend (tag 'final')."""
    from sklearn.metrics import log_loss
    from . import calibrate
    a, b = ("stage2x", "stage2x_" + SECONDARY) if use_x else ("stage2", "stage2_" + SECONDARY)
    pa, pb = np.load(config.WORK_DIR / f"oof_{a}.npy"), np.load(config.WORK_DIR / f"oof_{b}.npy")
    df = pl.read_parquet(config.split_dir("train") / "pairs_feat.parquet", columns=["q", "s1", "y"])
    rec = pl.read_parquet(config.split_dir("train") / "records.parquet")
    val_s1 = rec.filter((pl.col("src") == 1) & pl.col("is_val_s1")).select(pl.col("idx").alias("s1"))
    m = df.select(pl.col("s1").is_in(val_s1["s1"].implode())).to_series().to_numpy()
    y = df["y"].to_numpy()
    best = None
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        pw = np.clip(w * pa + (1 - w) * pb, 1e-7, 1 - 1e-7)
        ll = log_loss(y[m], pw[m])
        print(f"[blend] w_lgbm={w}: val log-loss {ll:.6f}", flush=True)
        if best is None or ll < best[1]:
            best = (w, ll)
    w = best[0]
    (config.WORK_DIR / "blend.json").write_text(__import__("json").dumps({"a": a, "b": b, "w_a": w}))
    p = w * pa + (1 - w) * pb
    np.save(config.WORK_DIR / "oof_final.npy", p.astype(np.float32))
    calibrate.tune(df.select("q", "s1"), p, "final")


def assemble_predictions(df: pl.DataFrame, p: np.ndarray, split: str) -> pl.DataFrame:
    """Representative-level (q, s1, p) -> record-level argmax assignment with ids."""
    d = config.split_dir(split)
    rec = pl.read_parquet(d / "records.parquet")
    x = df.select("q", "s1").with_columns(pl.Series("p", p))
    best = x.sort("p", descending=True).group_by("q", maintain_order=True).first()
    s1ids = rec.filter(pl.col("src") == 1).select(pl.col("idx").alias("s1"), pl.col("eid").alias("s1_eid"))
    members = rec.filter(pl.col("src") > 1).select(pl.col("rep").alias("q"), pl.col("eid").alias("rec_eid"))
    return best.join(s1ids, on="s1").join(members, on="q")   # every duplicate member inherits the decision


def evaluate_val(assigned: pl.DataFrame, rule: str = "ef", t: float = 0.5, with_syn: bool = False) -> dict:
    d = config.split_dir("train")
    rec = pl.read_parquet(d / "records.parquet")
    gt = pl.read_parquet(d / "gt.parquet")
    s1 = rec.filter((pl.col("src") == 1) & pl.col("is_val_s1"))
    a = assigned.join(s1.select(pl.col("eid").alias("s1_eid")), on="s1_eid", how="semi")
    if not with_syn:
        a = a.filter(~pl.col("rec_eid").str.contains("-SYN"))
    a = a.select("s1", "s1_eid", "rec_eid", pl.col("p").alias("q"))
    sel = decide.expected_f05_select(a, "q") if rule == "ef" else a.filter(pl.col("q") >= t)
    pred = sel.select("s1_eid", "rec_eid")
    by, mixw = eval_groups(s1.select(pl.col("eid").alias("s1_eid"), "country"))
    score, per = macro_f05(pred, gt, s1["eid"], by=by)
    by_c = {c: float(g["f05"].mean()) for (c,), g in per.group_by("group")}
    mix = sum(mixw[c] * by_c[c] for c in mixw if c in by_c) / sum(mixw[c] for c in mixw if c in by_c)
    return {"f05": score, "f05_test_mix": mix, **{f"f05_{c}": v for c, v in by_c.items()}}


def run(use_x: bool = False, featset: str = "full") -> None:
    sfx = SUFFIX[featset]
    tag = ("stage2x" if use_x else "stage2") + sfx
    if use_x:
        df = pl.read_parquet(config.split_dir("train") / "pairs_feat.parquet")
        df = add_x(df, "train", np.load(config.WORK_DIR / f"oof_stage2{sfx}.npy"))
    elif featset == "full":
        df = build("train")
        df = df.join(labels("train"), on=["q", "s1"], how="left", maintain_order="left").with_columns(pl.col("y").fill_null(0))
        df.write_parquet(config.split_dir("train") / "pairs_feat.parquet")
        (config.WORK_DIR / "pairs_feat.ready").touch()   # lets the GPU XGBoost chain start in parallel
    else:
        df = pl.read_parquet(config.split_dir("train") / "pairs_feat.parquet")   # same features, fewer used
    oof = _fit_and_save(df, PRIMARY, tag, featset)
    print(f"[train] primary learner {PRIMARY} ({tag})", flush=True)
    assigned = assemble_predictions(df, oof, "train")
    r_val = evaluate_val(assigned, "ef", 0.5, with_syn=False)
    print(f"[val:{tag}] expected-F0.5 rule, uncalibrated: F_val {r_val['f05']:.5f} (test-mix {r_val['f05_test_mix']:.5f}) "
          f"{[(k, round(v, 5)) for k, v in r_val.items() if k.startswith('f05_') and k != 'f05_test_mix']}", flush=True)
    from . import calibrate
    calibrate.tune(df.select("q", "s1"), oof, tag)

if __name__ == "__main__":
    run()
