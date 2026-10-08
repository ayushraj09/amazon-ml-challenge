"""Dense candidate generation: cross-fitted bi-encoder + exact GPU top-k (replaces sparse TF-IDF channels).

Why: on the India sample, char TF-IDF channels reached recall@20 of 0.72 (name) / 0.85 (name+street) /
0.90 (address) and would take hours at full scale; zero-shot multilingual-e5-small reached 0.978 in seconds
(experiments/biencoder/probe_zero_shot.py).

Training follows BLINK's bi-encoder recipe (facebookresearch/BLINK, examples/zeshel/README.md):
lr 1e-5, train batch 128, max length 128, in-batch negatives with softmax cross-entropy. The similarity is
cosine with scale 20 (sentence-transformers MultipleNegativesRankingLoss default, scale=20.0) because the
e5 backbone is trained on normalised embeddings. One shared encoder embeds both sides ("query: " prefix on
both, the e5 convention for symmetric tasks).

Cross-fitting (no optimistic train similarities): S1 entities are split into 2 folds by hash; model f is
trained on pairs whose S1 is in fold f and retrieves only for records whose true S1 is in the other fold
(unmatched records and all test records are assigned to a model by hash).
"""
import math
import time

import numpy as np
import polars as pl
import torch
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from . import config

BACKBONE = "intfloat/multilingual-e5-small"
BACKBONE_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"   # pinned HF revision (MIT)
LR, BATCH, MAX_LEN, SCALE, WARMUP = 1e-5, 128, 128, 20.0, 0.1
EPOCHS = 1
MAX_TRAIN_PAIRS = 1_000_000  # per fold: with random in-batch negatives the loss is 0.0024 by step 2000 (measured)
TOP_K = 50
N_FOLDS = 2
SEARCH_CHUNK = 2048


def _text(df: pl.DataFrame) -> list[str]:
    return ("query: " + df.select(pl.concat_str([pl.col("name").fill_null(""), pl.col("addr").fill_null("")],
                                                 separator=" | ")).to_series()).to_list()


def _fold(e: pl.Expr) -> pl.Expr:
    return (e.hash(config.SEED + 1) % N_FOLDS).cast(pl.Int8)


class Encoder(torch.nn.Module):
    def __init__(self, name: str):
        super().__init__()
        rev = BACKBONE_REVISION if name == BACKBONE else None
        self.tok = AutoTokenizer.from_pretrained(name, revision=rev)
        self.model = AutoModel.from_pretrained(name, revision=rev)

    def forward(self, texts) -> torch.Tensor:
        if texts and isinstance(texts[0], str):
            b = self.tok(texts, max_length=MAX_LEN, truncation=True, padding=True, return_tensors="pt").to("cuda")
        else:  # pre-tokenised ids
            b = self.tok.pad({"input_ids": texts}, return_tensors="pt").to("cuda")
        h = self.model(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
        return torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1)

    @torch.no_grad()
    def encode(self, texts: list[str], bs: int = 2048) -> torch.Tensor:
        self.eval()
        out = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(texts), bs):
                out.append(self(texts[i:i + bs]).to(torch.float16))
        return torch.cat(out)


def _train_pairs() -> pl.DataFrame:
    d = config.split_dir("train")
    rec = pl.read_parquet(d / "records.parquet")
    gt = pl.read_parquet(d / "gt.parquet")
    s1 = rec.filter(pl.col("src") == 1).select(pl.col("eid").alias("s1_eid"), pl.col("name").alias("n1"), pl.col("addr").alias("a1"))
    oth = rec.filter(pl.col("src") > 1).select(pl.col("eid").alias("rec_eid"), "name", "addr", "country")
    return gt.join(s1, on="s1_eid").join(oth, on="rec_eid").with_columns(_fold(pl.col("s1_eid")).alias("fold"))


def _batches_by_country(pairs: pl.DataFrame) -> pl.DataFrame:
    """Shuffle, then order so every batch holds pairs of a single country (harder, realistic in-batch
    negatives: retrieval is always within one country). Incomplete tail batches are dropped."""
    pairs = pairs.sample(fraction=1.0, shuffle=True, seed=config.SEED)
    pairs = pairs.with_columns((pl.int_range(pl.len()).over("country") // BATCH).alias("_b"))
    full = pairs.group_by("country", "_b").len().filter(pl.col("len") == BATCH).drop("len")
    order = full.sample(fraction=1.0, shuffle=True, seed=config.SEED + 1).with_row_index("_o")
    return pairs.join(order, on=["country", "_b"]).sort("_o", maintain_order=True).drop("_b", "_o")


def train(fold: int) -> Encoder:
    torch.manual_seed(config.SEED)
    pairs = _train_pairs().filter(pl.col("fold") == fold)
    pairs = _batches_by_country(pairs.sample(min(MAX_TRAIN_PAIRS, pairs.height), seed=config.SEED))
    enc = Encoder(BACKBONE).cuda()
    # pre-tokenise once (Rust tokenizer, batched) so the GPU is not waiting on tokenisation
    qa = enc.tok(_text(pairs.select("name", "addr")), max_length=MAX_LEN, truncation=True)["input_ids"]
    sa = enc.tok(_text(pairs.select(pl.col("n1").alias("name"), pl.col("a1").alias("addr"))),
                 max_length=MAX_LEN, truncation=True)["input_ids"]
    opt = torch.optim.AdamW(enc.parameters(), lr=LR)
    steps = math.ceil(len(qa) / BATCH) * EPOCHS
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP * steps), steps)
    enc.train(); t0 = time.time(); step = 0
    labels = torch.arange(BATCH, device="cuda")
    for _ in range(EPOCHS):
        for i in range(0, len(qa) - BATCH + 1, BATCH):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                q, s = enc(qa[i:i + BATCH]), enc(sa[i:i + BATCH])
            logits = SCALE * q.float() @ s.float().T
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            if step % 2000 == 0:
                print(f"[biencoder f{fold}] step {step}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
    path = config.WORK_DIR / f"biencoder_fold{fold}"
    enc.model.save_pretrained(path); enc.tok.save_pretrained(path)
    return enc


def load(fold: int) -> Encoder:
    enc = Encoder(str(config.WORK_DIR / f"biencoder_fold{fold}")).cuda()
    return enc


def _assign_model(split: str, rec: pl.DataFrame) -> pl.DataFrame:
    """Which fold model retrieves for each S2/S3 representative."""
    q = rec.filter((pl.col("src") > 1) & (pl.col("idx") == pl.col("rep"))).select("idx", "eid", "country", "name", "addr")
    if split == "train":
        gt = pl.read_parquet(config.split_dir("train") / "gt.parquet")
        owner = gt.select(pl.col("rec_eid").alias("eid"), _fold(pl.col("s1_eid")).alias("s1_fold"))
        q = q.join(owner, on="eid", how="left", maintain_order="left")
        # model f was trained on S1 fold f -> it serves records whose S1 is in the other fold
        return q.with_columns(pl.coalesce(1 - pl.col("s1_fold"), _fold(pl.col("eid"))).cast(pl.Int8).alias("model"))
    return q.with_columns(_fold(pl.col("eid")).alias("model"))


def retrieve(split: str, top_k: int = TOP_K) -> pl.DataFrame:
    d = config.split_dir(split)
    rec = pl.read_parquet(d / "records.parquet")
    q_all = _assign_model(split, rec)
    s1_all = rec.filter(pl.col("src") == 1).select("idx", "country", "name", "addr")
    out = []
    for f in range(N_FOLDS):
        enc = load(f)
        for ctry in s1_all["country"].unique().sort().to_list():
            t0 = time.time()
            s1 = s1_all.filter(pl.col("country") == ctry)
            q = q_all.filter((pl.col("country") == ctry) & (pl.col("model") == f))
            if q.height == 0:
                continue
            S = enc.encode(_text(s1)); Q = enc.encode(_text(q))
            k = min(top_k, S.shape[0])
            vals, idxs = [], []
            for i in range(0, Q.shape[0], SEARCH_CHUNK):   # chunk x |S1| fp16 scores stay < ~6 GB
                v, ix = (Q[i:i + SEARCH_CHUNK] @ S.T).topk(k, dim=1)
                vals.append(v.float().cpu()); idxs.append(ix.cpu())
            v, ix = torch.cat(vals).numpy(), torch.cat(idxs).numpy()
            out.append(pl.DataFrame({"q": np.repeat(q["idx"].to_numpy(), k), "s1": s1["idx"].to_numpy()[ix.ravel()],
                                     "dense": v.ravel().astype(np.float32),
                                     "r_dense": np.tile(np.arange(k, dtype=np.int16), q.height)}))
            print(f"[dense] {split} model{f} {ctry}: S1 {s1.height:,} Q {q.height:,} {time.time() - t0:.0f}s", flush=True)
        del enc; torch.cuda.empty_cache()
    res = pl.concat(out)
    res.write_parquet(d / "cands_dense.parquet")
    return res


def run() -> None:
    for f in range(N_FOLDS):
        # the bi-encoders depend only on ground-truth pairs (not on synthetic records), so already-trained
        # fold models are reused on a resumed run; a fresh run trains them
        if (config.WORK_DIR / f"biencoder_fold{f}" / "config.json").exists():
            print(f"[biencoder f{f}] reusing trained model", flush=True)
            continue
        train(f)
    for split in config.SPLITS:
        retrieve(split)


if __name__ == "__main__":
    run()
