"""Stage B: candidate generation, per country, S2/S3 representative -> S1 (IMPLEMENTATION_PLAN §4.1).

Channels:
  dense  cross-fitted bi-encoder top-50 (biencoder.py; primary channel)
  b1     word-level TF-IDF on name_core, top-20 (cheap; catches exact/near-exact names)
  exact  joins on (pno, first street word) and (sorted name_core key); blocks larger than
         EXACT_MAX_BLOCK are skipped (chain names / common numbers)
Char-n-gram TF-IDF channels were measured and dropped: on the India 100k sample, recall@20 was
0.72 / 0.85 / 0.90 (name / name+street / address) at an estimated 17-60 min per channel per country,
versus 0.978 zero-shot dense in seconds.
IDF is fitted per country on S1 plus a sample of that country's S2/S3 (transductive use of provided data).
Output: work/<split>/cands.parquet  (q, s1, dense, r_dense, b1, r1, ek_num_street, ek_name, n_ch)
"""
import time

import numpy as np
import polars as pl
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from . import config

K = {"b1": 20}
FIT_SAMPLE = 1_500_000
CHUNK = 250_000
EXACT_MAX_BLOCK = 200

CHANNELS = {"b1": dict(text=pl.col("name_core"), ngram=None)}


def _transform(vec, texts):
    return vec.transform(texts).astype(np.float32)


def _vectorize(vec, texts: list[str]) -> sp.csr_matrix:
    parts = Parallel(n_jobs=config.N_THREADS)(
        delayed(_transform)(vec, texts[i:i + 100_000]) for i in range(0, len(texts), 100_000))
    return sp.vstack(parts).tocsr()


def _topn(Q: sp.csr_matrix, S: sp.csr_matrix, k: int):
    """Row-wise top-k cosine (TF-IDF rows are L2-normalised). Returns (q_row, s_row, score, rank)."""
    ST = S.T.tocsr()
    rows, cols, vals, ranks = [], [], [], []
    for i in range(0, Q.shape[0], CHUNK):
        C = sp_matmul_topn(Q[i:i + CHUNK], ST, top_n=k, threshold=0.0, sort=True, n_threads=config.N_THREADS)
        C = C.tocsr()
        counts = np.diff(C.indptr)
        r = np.repeat(np.arange(C.shape[0]), counts) + i
        rk = np.concatenate([np.arange(c) for c in counts]) if len(counts) else np.array([], dtype=np.int64)
        rows.append(r); cols.append(C.indices); vals.append(C.data); ranks.append(rk)
    return (np.concatenate(rows), np.concatenate(cols), np.concatenate(vals).astype(np.float32),
            np.concatenate(ranks).astype(np.int16))


def _channel(ch: str, s1: pl.DataFrame, q: pl.DataFrame) -> pl.DataFrame:
    spec = CHANNELS[ch]
    ts = s1.select(spec["text"].fill_null("")).to_series().to_list()
    tq = q.select(spec["text"].fill_null("")).to_series().to_list()
    rng = np.random.default_rng(config.SEED)
    fit_q = [tq[i] for i in rng.choice(len(tq), size=min(FIT_SAMPLE, len(tq)), replace=False)]
    vec = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", min_df=1, sublinear_tf=True, dtype=np.float32)
    vec.fit(ts + fit_q)
    S, Q = _vectorize(vec, ts), _vectorize(vec, tq)
    r, c, v, rk = _topn(Q, S, K[ch])
    return pl.DataFrame({"q": q["idx"].to_numpy()[r], "s1": s1["idx"].to_numpy()[c], ch: v,
                         "r" + ch[1]: rk})


def _exact(s1: pl.DataFrame, q: pl.DataFrame) -> pl.DataFrame:
    def keys(df):
        return df.select(
            "idx",
            pl.concat_str([pl.col("pno").cast(pl.Utf8), pl.col("street").str.extract(r"(\p{L}{3,})", 1)],
                          separator="|").alias("k_ns"),
            pl.col("name_core").str.split(" ").list.sort().list.join(" ").alias("k_nm"))
    ks, kq = keys(s1), keys(q)
    out = []
    for key, flag in [("k_ns", "ek_num_street"), ("k_nm", "ek_name")]:
        blk = ks.filter(pl.col(key).is_not_null() & (pl.col(key) != "")).group_by(key).agg(pl.col("idx").alias("s1"))
        blk = blk.filter(pl.col("s1").list.len() <= EXACT_MAX_BLOCK)
        j = kq.filter(pl.col(key).is_not_null()).select(pl.col("idx").alias("q"), key).join(blk, on=key).explode("s1")
        out.append(j.select("q", pl.col("s1").cast(pl.Int32), pl.lit(True).alias(flag)))
    return out[0].join(out[1], on=["q", "s1"], how="full", coalesce=True)


def block_split(split: str) -> pl.DataFrame:
    rec = pl.read_parquet(config.split_dir(split) / "records.parquet").select("idx", "src", "country")
    nm = pl.read_parquet(config.split_dir(split) / "norm.parquet")
    df = nm.join(rec, on="idx")
    dense = pl.read_parquet(config.split_dir(split) / "cands_dense.parquet")
    parts = []
    for ctry in df["country"].unique().sort().to_list():
        t0 = time.time()
        s1 = df.filter((pl.col("country") == ctry) & (pl.col("src") == 1))
        q = df.filter((pl.col("country") == ctry) & (pl.col("src") > 1))
        cands = dense.filter(pl.col("q").is_in(q["idx"].implode()))
        for ch in CHANNELS:
            c = _channel(ch, s1, q)
            cands = cands.join(c, on=["q", "s1"], how="full", coalesce=True)
        cands = cands.join(_exact(s1, q), on=["q", "s1"], how="full", coalesce=True)
        parts.append(cands)
        print(f"[block] {split} {ctry}: S1 {s1.height:,} Q {q.height:,} pairs {cands.height:,} "
              f"({cands.height / max(q.height, 1):.1f}/query) {time.time() - t0:.0f}s", flush=True)
    out = pl.concat(parts, how="diagonal_relaxed").with_columns(
        pl.col("q", "s1").cast(pl.Int32), pl.col("ek_num_street", "ek_name").fill_null(False))
    out = out.with_columns((pl.col("dense").is_not_null().cast(pl.Int8) + pl.col("b1").is_not_null().cast(pl.Int8) +
                            pl.col("ek_num_street").cast(pl.Int8) + pl.col("ek_name").cast(pl.Int8)).alias("n_ch"))
    out.write_parquet(config.split_dir(split) / "cands.parquet")
    return out


def recall_report(split: str = "train") -> pl.DataFrame:
    """Pair recall of the candidate set against GT (train), by country and noise flags. Val S1 only."""
    rec = pl.read_parquet(config.split_dir(split) / "records.parquet")
    nm = pl.read_parquet(config.split_dir(split) / "norm.parquet").select("idx", "was_indic", "addr_missing", "is_domain")
    gt = pl.read_parquet(config.split_dir(split) / "gt.parquet")
    cands = pl.read_parquet(config.split_dir(split) / "cands.parquet").select(
        "q", "s1", pl.col("dense").is_not_null().alias("dense"), pl.col("b1").is_not_null().alias("b1"),
        (pl.col("ek_num_street") | pl.col("ek_name")).alias("ek_any")).with_columns(pl.lit(True).alias("hit"))
    s1 = rec.filter(pl.col("src") == 1).select(pl.col("eid").alias("s1_eid"), pl.col("idx").alias("s1"), "is_val_s1", "country")
    oth = rec.filter(pl.col("src") > 1).select(pl.col("eid").alias("rec_eid"), pl.col("rep").alias("q"), "src")
    x = gt.join(s1, on="s1_eid").join(oth, on="rec_eid").filter(pl.col("is_val_s1"))
    x = (x.join(cands, on=["q", "s1"], how="left", maintain_order="left").with_columns(pl.col("hit", "dense", "b1", "ek_any").fill_null(False))
          .join(nm, left_on="q", right_on="idx"))
    rows = [("all", "all", x.height, x["hit"].mean())]
    for ch, col in [("dense", "dense"), ("b1", "b1"), ("exact", "ek_any")]:
        rows.append(("channel_only", ch, x.height, x[col].mean()))
    for c in ["country", "src", "was_indic", "addr_missing", "is_domain"]:
        for k, g in x.group_by(c):
            rows.append((c, str(k[0]), g.height, g["hit"].mean()))
    return pl.DataFrame(rows, schema=["by", "value", "n", "recall"], orient="row").sort("by", "value")


def run() -> None:
    for split in config.SPLITS:
        block_split(split)
    print(recall_report("train"))


if __name__ == "__main__":
    run()
