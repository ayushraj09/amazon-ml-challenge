"""Stage X driver: cross-fitted cross-encoder on hard records, scored on uncertain records (IMPLEMENTATION_PLAN §5.3).

Train side (2 folds by owner hash): model f trains on records of fold f and scores records of fold 1-f.
Selection is identical on train and test and uses only the stage-2 probability (OOF on train):
  uncertain record = its best stage-2 probability lies in (P_LO, P_HI). All its candidates are scored.
Training records for each fold: uncertain records first, then all records with a near-miss-type candidate
(pno_edit_type 4 = small arithmetic shift) and synthetic near-misses, capped at MAX_TRAIN_RECORDS.
Outputs work/<split>/xenc.parquet: q, s1, x_logit, x_ova, x_pi, x_q (= x_pi * x_ova, the D1 fusion).
"""
import os

import numpy as np
import polars as pl
import torch

from . import config
from . import cross_encoder as ce

P_LO, P_HI = 0.01, 0.99
N_FOLDS = int(os.environ.get("ER_XENC_FOLDS", "2"))


def _owner_fold(df: pl.DataFrame) -> pl.DataFrame:
    from .stage1 import labels
    owner = labels("train").select("q", pl.col("s1").alias("owner"))
    g = df.select("q").unique().join(owner, on="q", how="left", maintain_order="left").with_columns(
        (pl.coalesce("owner", "q").hash(config.SEED + 7) % N_FOLDS).cast(pl.Int8).alias("xfold"))
    return df.join(g.select("q", "xfold"), on="q")


def _topk_uncertain(df: pl.DataFrame, unc: pl.DataFrame) -> pl.DataFrame:
    """Candidates to score: each uncertain record's top ce.K_LIST candidates by stage-2 probability. The listwise
    cross-encoder is trained on lists of <= K_LIST (8) candidates, so inference uses the same list size."""
    return (df.join(unc, on="q", how="semi").sort(["q", "p2"], descending=[False, True])
              .group_by("q", maintain_order=True).head(ce.K_LIST).select("q", "s1"))


def _uncertain(df: pl.DataFrame) -> pl.DataFrame:
    best = df.group_by("q").agg(pl.col("p2").max().alias("pmax"))
    return best.filter((pl.col("pmax") > P_LO) & (pl.col("pmax") < P_HI)).select("q")


def train_models(folds=(0, 1), resume: bool = False, patience: int | None = None) -> None:
    """Train both cross-fitted cross-encoders as soon as stage 1 (train) is done, on the GPU in parallel with
    the CPU stages. Hard training records are chosen from stage-1 OOF p1 (uncertain records), near-miss-like
    candidates (same core name, house number shifted by 1-50) and all synthetic near-misses."""
    torch.set_num_threads(4)                       # do not oversubscribe the CPU used by LightGBM
    (config.WORK_DIR / "xenc_models.ready").unlink(missing_ok=True)
    (config.WORK_DIR / "xenc_train.running").touch()
    d_tr = config.split_dir("train")
    tr = pl.read_parquet(d_tr / "pruned.parquet", columns=["q", "s1", "p1", "y", "pno_eq", "legal_rel", "st_tset",
                                                            "nm_tset", "pno_abs"])
    tr = tr.with_columns((pl.col("st_tset") >= 100).cast(pl.Int64).alias("st_eq"),
                         pl.col("pno_eq").fill_nan(0).fill_null(0).cast(pl.Int64),
                         pl.col("legal_rel").cast(pl.Int64), pl.col("y").cast(pl.Int64))
    tr = _owner_fold(tr)
    best = tr.group_by("q").agg(pl.col("p1").max().alias("pmax"))
    unc = best.filter((pl.col("pmax") > P_LO) & (pl.col("pmax") < P_HI)).select("q")
    near = tr.filter((pl.col("nm_tset") >= 100) & (pl.col("pno_abs") > 0) & (pl.col("pno_abs") <= 50)).select("q").unique()
    rec = pl.read_parquet(d_tr / "records.parquet").select("idx", "eid")
    syn = rec.filter(pl.col("eid").str.contains("-SYN")).select(pl.col("idx").alias("q"))
    pri = pl.concat([unc, near, syn]).unique()
    print(f"[xenc] training pool: uncertain {unc.height:,} near-miss-like {near.height:,} synthetic {syn.height:,}", flush=True)
    for f in folds:
        pool = tr.filter(pl.col("xfold") == f)
        qs = pool.join(pri, on="q", how="semi")["q"].unique().sort().sample(fraction=1.0, shuffle=True, seed=config.SEED)
        qs = qs.head(ce.MAX_TRAIN_RECORDS)
        train_pairs = pool.join(qs.to_frame(), on="q", how="semi").select("q", "s1", "p1", "y", "pno_eq", "legal_rel", "st_eq")
        dev_q = tr.filter(pl.col("xfold") == (f + 1) % N_FOLDS).join(unc, on="q", how="semi")["q"].unique().sample(3000, seed=config.SEED)
        dev = tr.join(dev_q.to_frame(), on="q", how="semi").select("q", "s1", "y")
        ce.train_fold(train_pairs, f, dev, resume=resume, patience=patience)
        print(f"[xenc] fold {f}: trained on {qs.len():,} records", flush=True)
    (config.WORK_DIR / "xenc_models.ready").touch()
    (config.WORK_DIR / "xenc_train.running").unlink(missing_ok=True)


def fix_models() -> None:
    """Stage 'xencfix': fold 0 stopped early at dev AP 0.9792 after 3 noisy non-improving evals while fold 1 kept
    improving to 0.9880 (measured). Resume fold 0 from its checkpoint with patience 5, then mark both ready."""
    train_models(folds=(0,), resume=True, patience=5)


def _load_model(f: int):
    m = ce.ListwiseCE(ce.BACKBONE).cuda()
    m.load_state_dict(torch.load(config.WORK_DIR / f"xenc_fold{f}.pt"))
    return m


def run() -> None:
    d_tr, d_te = config.split_dir("train"), config.split_dir("test")
    tr = pl.read_parquet(d_tr / "pairs_feat.parquet", columns=["q", "s1", "p1", "y", "pno_eq", "legal_rel", "st_eq", "pno_edit_type"])
    tr = tr.with_columns(pl.Series("p2", np.load(config.WORK_DIR / "oof_stage2.npy")))
    te = pl.read_parquet(d_te / "pred_stage2.parquet").rename({"p": "p2"})
    tr = _owner_fold(tr)
    unc_tr, unc_te = _uncertain(tr), _uncertain(te)
    rec = pl.read_parquet(d_tr / "records.parquet").select("idx", "eid")
    syn_q = rec.filter(pl.col("eid").str.contains("-SYN")).select(pl.col("idx").alias("q"))
    near_q = tr.filter(pl.col("pno_edit_type") == 4).select("q").unique()
    print(f"[xenc] uncertain records: train {unc_tr.height:,} test {unc_te.height:,}", flush=True)
    outs_tr, models = [], []
    import time
    while (config.WORK_DIR / "xenc_train.running").exists() and not (config.WORK_DIR / "xenc_models.ready").exists():
        print("[xenc] waiting for early cross-encoder training (xenctrain) to finish", flush=True)
        time.sleep(300)
    pretrained = (config.WORK_DIR / "xenc_models.ready").exists()
    for f in range(N_FOLDS):
        if pretrained:                      # trained early by train_models() (stage "xenctrain")
            model = _load_model(f)
            target = _topk_uncertain(tr.filter(pl.col("xfold") == (f + 1) % N_FOLDS), unc_tr)
            outs_tr.append(ce.score(model, target, "train"))
            models.append(model)
            print(f"[xenc] fold {f}: loaded pre-trained model, scored {target.height:,} train pairs", flush=True)
            continue
        pool = tr.filter(pl.col("xfold") == f)
        pri = pl.concat([unc_tr, near_q, syn_q]).unique()
        chosen = pool.join(pri, on="q", how="semi")
        qs = chosen["q"].unique().sample(fraction=1.0, shuffle=True, seed=config.SEED).head(ce.MAX_TRAIN_RECORDS)
        train_pairs = pool.join(qs.to_frame(), on="q", how="semi").with_columns(
            pl.col("pno_eq").fill_nan(0).fill_null(0).cast(pl.Int64), pl.col("st_eq").fill_null(0).cast(pl.Int64),
            pl.col("legal_rel").cast(pl.Int64), pl.col("y").cast(pl.Int64))
        # dev set: 3,000 uncertain records of the OTHER fold (never trained on by this model)
        dev_q = tr.filter(pl.col("xfold") == (f + 1) % N_FOLDS).join(unc_tr, on="q", how="semi")["q"].unique().sample(3000, seed=config.SEED)
        dev = tr.join(dev_q.to_frame(), on="q", how="semi").select("q", "s1", "y")
        model = ce.train_fold(train_pairs.select("q", "s1", "p1", "y", "pno_eq", "legal_rel", "st_eq"), f, dev)
        target = _topk_uncertain(tr.filter(pl.col("xfold") == (f + 1) % N_FOLDS), unc_tr)
        outs_tr.append(ce.score(model, target, "train"))
        models.append(model)
        print(f"[xenc] fold {f}: trained on {qs.len():,} records, scored {target.height:,} train pairs", flush=True)
    pl.concat(outs_tr).with_columns((pl.col("x_pi") * pl.col("x_ova")).alias("x_q")).write_parquet(d_tr / "xenc.parquet")
    target_te = _topk_uncertain(te, unc_te)
    halves = [ce.score(m, target_te.filter((pl.col("q").hash(config.SEED + 9) % N_FOLDS) == f), "test")
              for f, m in enumerate(models)]
    pl.concat(halves).with_columns((pl.col("x_pi") * pl.col("x_ova")).alias("x_q")).write_parquet(d_te / "xenc.parquet")
    print(f"[xenc] test scored {target_te.height:,} pairs", flush=True)


def _part_name(f: int, shard: int, n_shards: int) -> str:
    return f"xenc_part{f}.parquet" if n_shards == 1 else f"xenc_part{f}_{shard}of{n_shards}.parquet"


def score_fold(f: int, shard: int = 0, n_shards: int = 1) -> None:
    """Load model f, score its OOF train fold ((f+1) % N_FOLDS) and its test share (hash % N_FOLDS == f);
    write work/train/xenc_part{f}.parquet and work/test/xenc_part{f}.parquet (same columns as run()).
    Same selection and split as run(), so one process per GPU reproduces run() with N_FOLDS models.
    n_shards > 1 splits model f's pairs by record over several GPUs (whole candidate lists stay together,
    so x_pi is unchanged); each shard writes xenc_part{f}_{shard}of{n_shards}.parquet."""
    d_tr, d_te = config.split_dir("train"), config.split_dir("test")
    tr = pl.read_parquet(d_tr / "pairs_feat.parquet", columns=["q", "s1", "p1", "y", "pno_eq", "legal_rel", "st_eq", "pno_edit_type"])
    tr = tr.with_columns(pl.Series("p2", np.load(config.WORK_DIR / "oof_stage2.npy")))
    te = pl.read_parquet(d_te / "pred_stage2.parquet").rename({"p": "p2"})
    tr = _owner_fold(tr)
    unc_tr, unc_te = _uncertain(tr), _uncertain(te)
    print(f"[xenc f{f}] N_FOLDS {N_FOLDS}, uncertain records: train {unc_tr.height:,} test {unc_te.height:,}", flush=True)
    model = _load_model(f)
    mine = (pl.col("q").hash(config.SEED + 11) % n_shards) == shard
    target = _topk_uncertain(tr.filter(pl.col("xfold") == (f + 1) % N_FOLDS), unc_tr).filter(mine)
    ce.score(model, target, "train").write_parquet(d_tr / _part_name(f, shard, n_shards))
    print(f"[xenc f{f}] scored {target.height:,} train pairs", flush=True)
    target_te = _topk_uncertain(te, unc_te).filter((pl.col("q").hash(config.SEED + 9) % N_FOLDS) == f).filter(mine)
    ce.score(model, target_te, "test").write_parquet(d_te / _part_name(f, shard, n_shards))
    print(f"[xenc f{f}] scored {target_te.height:,} test pairs", flush=True)


def merge_parts(n_shards: int = 1) -> None:
    """Concatenate the N_FOLDS parts per split, add x_q = x_pi * x_ova, write work/{train,test}/xenc.parquet."""
    for split in ("train", "test"):
        d = config.split_dir(split)
        parts = [d / _part_name(f, sh, n_shards) for f in range(N_FOLDS) for sh in range(n_shards)]
        missing = [str(p) for p in parts if not p.exists()]
        if missing:
            raise FileNotFoundError(f"missing parts: {missing}")
        out = pl.concat([pl.read_parquet(p) for p in parts]).with_columns((pl.col("x_pi") * pl.col("x_ova")).alias("x_q"))
        out.write_parquet(d / "xenc.parquet")
        print(f"[xenc] {split}: merged {N_FOLDS} parts, {out.height:,} pairs", flush=True)


if __name__ == "__main__":
    run()
