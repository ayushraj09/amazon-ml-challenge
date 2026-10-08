"""Stage X: listwise cross-encoder with a NIL option (IMPLEMENTATION_PLAN §5.3, X1-X3). Cross-fitted (2 folds).

Per training example = one record with its pruned candidate list (<= K_LIST S1s), each pair encoded as
"record text [SEP] S1 text". Heads on the pooled pair vector:
  score s_j  -> X1 listwise softmax CE over {candidates, NIL}; NIL logit is a learned scalar
               ("NIL as a special entity", Sil & Florian 2016; Kolitsas et al. 2018)
  ova (2)    -> X1 one-vs-all loss with hardest negative, exactly OP_Match utils/misc.py::ova_loss;
               NIL records drop the positive term as OpenLTR mb_sup_loss does. Weight 1.0 (mb_loss_ratio default)
  RTM        -> X2 soft maximum of non-mate match probabilities, sum(softmax(o+) * o+), weight 4
               (opengait/modeling/losses/open_set.py: losses = 1 - DIR + 4 * softmax_scores_nm)
  aux heads  -> X3 pno_eq (2), legal_rel (4), st_eq (2), weight 0.5 each (NIL_EL --cate_*_loss_weight 0.5)
Optimisation: lr 3e-5, AdamW, linear schedule (Ditto train_ditto.py); warmup 0.1, grad clip 1.0, 1 epoch
(BLINK params.py / NIL_EL train_cross_raw.sh); 64 pairs per step (Ditto batch 64) = 8 records x 8 candidates.
Backbone: chosen by the measured bake-off (IMPLEMENTATION_PLAN §5.4).
"""
import math
import time

import numpy as np
import polars as pl
import torch
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from . import config

# pinned HF revisions (both MIT). ER_XENC_BACKBONE selects the backbone; mmBERT-small is the default.
REVISIONS = {"jhu-clsp/mmBERT-small": "abc32620dd4f6ab06f5fbe905dc25f310618e09f",   # 141M, MIT
             "jhu-clsp/mmBERT-base": "c5955035435e2bf121cde7f3c8863ef52ff35d82",    # 307M, MIT
             "Qwen/Qwen3-1.7B": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",         # 1.7B, Apache-2.0 (decoder)
             "Qwen/Qwen3-4B": "1cfa9a7208912126459214e8b04321603b3df60c"}           # 4B, Apache-2.0 (decoder)
BACKBONE = __import__("os").environ.get("ER_XENC_BACKBONE", "jhu-clsp/mmBERT-small")
BACKBONE_REVISION = REVISIONS[BACKBONE]
DECODERS = {"Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"}
IS_DECODER = BACKBONE in DECODERS
LR, WARMUP, CLIP, EPOCHS, MAX_LEN = 3e-5, 0.1, 1.0, 1, 128
K_LIST, RECORDS_PER_STEP = 8, 8
SCORE_BS = 2048
if IS_DECODER:
    # decoder LLM backbone: LoRA (QLoRA settings for <= 13B models, Dettmers et al. 2023: r 64, alpha 16, dropout 0.1,
    # lr 2e-4) on all linear layers + fully trained heads; a pair is an explicit prompt pooled at its last token.
    # 32 records per step ~ 64 pairs on the top-2 candidate lists (Ditto batch 64); capped training records.
    LORA_R, LORA_ALPHA, LORA_DROPOUT = 64, 16, 0.1
    LR, MAX_LEN, RECORDS_PER_STEP, SCORE_BS = 2e-4, 160, 32, 256
    RECORDS_PER_STEP = int(__import__("os").environ.get("ER_XENC_RECORDS_PER_STEP", RECORDS_PER_STEP))
    SCORE_BS = int(__import__("os").environ.get("ER_XENC_SCORE_BS", SCORE_BS))
W_OVA, W_RTM, W_AUX = 1.0, 4.0, 0.5
MAX_TRAIN_RECORDS = int(__import__("os").environ.get("ER_XENC_MAX_RECORDS", 1_000_000))  # per fold upper bound; training stops earlier on a dev plateau (Ditto keeps the
EVAL_EVERY, PATIENCE = 3000, 5  # best dev checkpoint): evaluate every 3000 steps; patience 5 because 3 stopped
                                # fold 0 prematurely on the noisy 3,000-record dev set (fold 1 kept improving
                                # 0.968 -> 0.988 after the point where fold 0 had stopped; measured)
N_FOLDS = 2
if IS_DECODER:
    EVAL_EVERY = 1000            # 1000 steps x 32 records: dev AP checked every ~32k records (cap 300k per fold)


class ListwiseCE(torch.nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.decoder = name in DECODERS
        if self.decoder:
            from peft import LoraConfig, get_peft_model
            enc = AutoModel.from_pretrained(name, revision=REVISIONS.get(name), torch_dtype=torch.bfloat16)
            enc.gradient_checkpointing_enable()
            enc.enable_input_require_grads()
            self.enc = get_peft_model(enc, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                                                      target_modules="all-linear"))
        else:
            self.enc = AutoModel.from_pretrained(name, revision=REVISIONS.get(name))
        h = self.enc.config.hidden_size
        self.score = torch.nn.Linear(h, 1)
        self.ova = torch.nn.Linear(h, 2)
        self.aux_pno, self.aux_legal, self.aux_st = torch.nn.Linear(h, 2), torch.nn.Linear(h, 4), torch.nn.Linear(h, 2)
        self.nil = torch.nn.Parameter(torch.zeros(1))

    def forward(self, batch):
        out = self.enc(**batch).last_hidden_state
        if self.decoder:                                   # last real token (right padding)
            last = batch["attention_mask"].sum(1) - 1
            h = out[torch.arange(out.shape[0], device=out.device), last].float()
        else:
            h = out[:, 0]                                  # [CLS]-position pooling
        return self.score(h).squeeze(-1), self.ova(h), self.aux_pno(h), self.aux_legal(h), self.aux_st(h)


def ova_loss(logits_ova: torch.Tensor, is_mate: torch.Tensor, has_mate: torch.Tensor) -> torch.Tensor:
    """OP_Match ova_loss for one record's list: logits_ova [k, 2] (col 1 = inlier), is_mate [k] bool."""
    p = torch.softmax(logits_ova, dim=1)
    pos = -(torch.log(p[:, 1] + 1e-8) * is_mate).sum() if has_mate else logits_ova.new_zeros(())
    neg_terms = -torch.log(p[:, 0] + 1e-8) * (~is_mate)
    return pos + (neg_terms.max() if (~is_mate).any() else logits_ova.new_zeros(()))


def rtm(logits_ova: torch.Tensor, is_mate: torch.Tensor) -> torch.Tensor:
    """Soft maximum of the non-mate inlier probabilities (relative threshold minimisation)."""
    nm = torch.softmax(logits_ova, dim=1)[:, 1][~is_mate]
    return (torch.softmax(nm, 0) * nm).sum() if nm.numel() else logits_ova.new_zeros(())


def _tokenizer():
    tok = AutoTokenizer.from_pretrained(BACKBONE, revision=BACKBONE_REVISION)
    if IS_DECODER:
        tok.padding_side = "right"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
    return tok


def _encode(tok, a: list, b: list, **kw):
    """Encoder: native sentence pair. Decoder: explicit prompt (pooled at its last token)."""
    if IS_DECODER:
        return tok([f"Business A: {x}\nBusiness B: {y}\nSame business:" for x, y in zip(a, b)],
                   truncation=True, max_length=MAX_LEN, **kw)
    return tok(a, b, truncation=True, max_length=MAX_LEN, **kw)


def _texts(split: str) -> pl.DataFrame:
    rec = pl.read_parquet(config.split_dir(split) / "records.parquet")
    return rec.select("idx", pl.concat_str([pl.col("name").fill_null(""), pl.col("addr").fill_null("")],
                                           separator=" | ").alias("text"))


def _lists(pairs: pl.DataFrame) -> pl.DataFrame:
    """One row per record: candidate lists truncated to K_LIST by p1."""
    return (pairs.sort(["q", "p1"], descending=[False, True]).group_by("q", maintain_order=True)
                 .head(K_LIST).group_by("q", maintain_order=True).agg(pl.all()))


# --- format augmentation for countries absent from train (domain randomisation over surface formats) -------------------
# Train has only US / India formats; test adds France, with legal forms (SARL, SAS, SASU, EURL, SCI, SNC, ...), house
# numbers written "N° 12" / "(12)" and the country word "France". With probability AUG_P a training record and ALL its
# candidates are rewritten with one consistent random mapping of legal-form words and country words (a same-form pair
# stays same-form, a legal-form swap stays a swap, so labels are unchanged), and each text's leading house number gets
# an independent random prefix format (the number itself is kept). Dev AP and all scoring use the original texts.
AUG_P = float(__import__("os").environ.get("ER_XENC_AUG", "0"))
# words of one legal form share one target: "Private Limited" / "Pvt Ltd" / "Ltd" (India's true subset variants) all
# become the same French form, while distinct forms (LLC vs Inc, Ltd vs LLP) become distinct French forms
AUG_SRC_LEGAL = [("pvt", "private", "ltd", "limited"), ("llc",), ("inc", "incorporated"), ("corp", "corporation"),
                 ("co", "company"), ("llp",), ("lp",), ("pc",), ("pllc",)]
AUG_DST_LEGAL = ["SARL", "SAS", "SASU", "EURL", "SA", "SCI", "SNC", "EI", "SELARL", "SCOP", "GIE", "SCA", "SCS", "EIRL",
                 "SEL", "EARL"]
AUG_COUNTRY = {"india": "France", "usa": "France", "us": "France"}
AUG_NUM_FMT = ["N° {n}", "Nº {n}", "({n})", "No {n}", "#{n}", "N°{n}"]
_AUG_WORD = __import__("re").compile(r"[A-Za-z]+")
_AUG_NUM = __import__("re").compile(r"(^|,\s*)(\d+)(?=\s+\D)")


def _augment(texts: list, rng: np.random.Generator) -> list:
    """Rewrite a record's texts (record first, then its candidates) with one consistent format mapping."""
    dst = list(rng.permutation(AUG_DST_LEGAL))
    lmap = {w: dst[i] for i, grp in enumerate(AUG_SRC_LEGAL) for w in grp}

    def word(m):
        w = m.group(0); lw = w.lower()
        t = lmap.get(lw) or AUG_COUNTRY.get(lw)
        if t is None:
            return w
        return t.upper() if w.isupper() and len(w) > 1 else t
    out = []
    for x in texts:
        name, _, addr = x.partition(" | ")
        name = _AUG_WORD.sub(word, name)
        for t in dst[:len(AUG_SRC_LEGAL)]:          # "Private Limited" -> "SARL SARL" -> "SARL"
            name = __import__("re").sub(rf"\b({t})(\W+{t}\b)+", r"\1", name, flags=__import__("re").I)
        addr = _AUG_WORD.sub(lambda m: AUG_COUNTRY.get(m.group(0).lower(), m.group(0)) if m.group(0).lower() in ("india", "usa") else m.group(0), addr)
        if rng.random() < 0.5:
            fmt = AUG_NUM_FMT[int(rng.integers(len(AUG_NUM_FMT)))]
            addr = _AUG_NUM.sub(lambda m: m.group(1) + fmt.format(n=m.group(2)), addr, count=1)
        out.append(f"{name} | {addr}" if _ else name)
    return out


def _dev_ap(model: "ListwiseCE", dev: pl.DataFrame) -> float:
    from sklearn.metrics import average_precision_score
    sc = score(model, dev.select("q", "s1"), "train")
    y = dev.select("q", "s1", "y").join(sc, on=["q", "s1"])
    model.train()
    return float(average_precision_score(y["y"].to_numpy(), (y["x_pi"] * y["x_ova"]).to_numpy()))


def _dist() -> tuple[int, int]:
    """(rank, world size). Under torchrun (WORLD_SIZE > 1) one fold trains data-parallel on several GPUs:
    each rank takes its share of every step's records and gradients are summed across ranks. Else (0, 1)."""
    import os
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world == 1:
        return 0, 1
    import datetime
    import torch.distributed as dist
    if not dist.is_initialized():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=60))   # rank 0 alone runs dev evals
    return dist.get_rank(), world


def train_fold(pairs: pl.DataFrame, fold: int, dev: pl.DataFrame | None = None, resume: bool = False,
               patience: int | None = None) -> ListwiseCE:
    """resume=True continues from the saved checkpoint; the starting 'best' is that checkpoint's dev AP, so the
    checkpoint is only overwritten by a genuine improvement. Under torchrun the global step is still
    RECORDS_PER_STEP records, split across ranks; rank 0 runs dev evals and saves checkpoints."""
    rank, world = _dist()
    if world > 1:
        import torch.distributed as dist
    torch.manual_seed(config.SEED)
    frac = float(__import__("os").environ.get("ER_XENC_GPU_FRACTION", "0.6"))
    torch.cuda.set_per_process_memory_fraction(frac)     # leave GPU memory for the GBDT stages
    patience = patience or PATIENCE
    tx = _texts("train")
    lists = _lists(pairs).sample(fraction=1.0, shuffle=True, seed=config.SEED).head(MAX_TRAIN_RECORDS)
    tmap = dict(zip(tx["idx"].to_list(), tx["text"].to_list()))
    tok = _tokenizer()
    model = ListwiseCE(BACKBONE).cuda()
    if resume:
        model.load_state_dict(torch.load(config.WORK_DIR / f"xenc_fold{fold}.pt", map_location="cuda"))
    trainable = [p for p in model.parameters() if p.requires_grad]
    if world > 1:                                          # every rank must hold the same record order
        import hashlib
        sig = torch.tensor([lists.height, int.from_bytes(hashlib.md5(lists["q"].to_numpy().tobytes()).digest()[:7], "little")],
                           device="cuda")
        sig0 = sig.clone(); dist.broadcast(sig0, 0)
        assert torch.equal(sig, sig0), f"rank {rank}: training records differ from rank 0"
    if world > 1:                                          # identical start on every rank
        for p in trainable:
            dist.broadcast(p.data, 0)
    opt = torch.optim.AdamW(trainable, lr=LR)
    steps = math.ceil(lists.height / RECORDS_PER_STEP) * EPOCHS
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP * steps), steps)
    rows = lists.to_dicts()
    aug_rng = np.random.default_rng(config.SEED + 17 * fold + 1000 * rank)
    if AUG_P > 0 and rank == 0:
        print(f"[xenc f{fold}] format augmentation p={AUG_P}", flush=True)
    log_every = int(__import__("os").environ.get("ER_XENC_LOG_EVERY", "2000"))
    run_loss, run_n = torch.zeros((), device="cuda"), 0
    model.train(); t0 = time.time()
    best_ap, bad = (_dev_ap(model, dev) if (resume and dev is not None and rank == 0) else -1.0), 0
    if resume and rank == 0:
        print(f"[xenc f{fold}] resumed from checkpoint, dev AP {best_ap:.6f}", flush=True)
    for st in range(steps):
        full = rows[st * RECORDS_PER_STEP:(st + 1) * RECORDS_PER_STEP]
        grp = full[rank::world]
        if not grp:                                        # a short last step can leave a rank empty
            pass
        elif AUG_P > 0:
            a, b = [], []
            for r in grp:
                t = [tmap[r["q"]]] + [tmap[x] for x in r["s1"]]
                if aug_rng.random() < AUG_P:
                    t = _augment(t, aug_rng)
                a += [t[0]] * len(r["s1"]); b += t[1:]
        else:
            a = [tmap[r["q"]] for r in grp for _ in r["s1"]]
            b = [tmap[s] for r in grp for s in r["s1"]]
        if grp:
            enc = _encode(tok, a, b, padding=True, return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s, o, ap, al, ast = model(enc)
            s, o, ap, al, ast = s.float(), o.float(), ap.float(), al.float(), ast.float()
        loss, off = torch.zeros((), device="cuda"), 0
        for r in grp:
            k = len(r["s1"]); sl = slice(off, off + k); off += k
            y = torch.tensor(r["y"], device="cuda").bool()
            has = bool(y.any())
            logits = torch.cat([s[sl], model.nil])
            target = torch.tensor(int(np.argmax(r["y"])) if has else k, device="cuda")
            l_list = torch.nn.functional.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
            l_aux = (torch.nn.functional.cross_entropy(ap[sl], torch.tensor(r["pno_eq"], device="cuda").nan_to_num(0).long()) +
                     torch.nn.functional.cross_entropy(al[sl], torch.tensor(r["legal_rel"], device="cuda").long()) +
                     torch.nn.functional.cross_entropy(ast[sl], torch.tensor(r["st_eq"], device="cuda").nan_to_num(0).long()))
            loss = loss + l_list + W_OVA * ova_loss(o[sl], y, has) + W_RTM * rtm(o[sl], y) + W_AUX * l_aux
        loss = loss / len(full)                            # summed over ranks = mean over the full step
        if grp:
            loss.backward()
        if world > 1:
            grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in trainable]
            flat = torch._utils._flatten_dense_tensors(grads)
            dist.all_reduce(flat)
            for p, g in zip(trainable, torch._utils._unflatten_dense_tensors(flat, grads)):
                p.grad = g
            del flat
        torch.nn.utils.clip_grad_norm_(trainable, CLIP)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        run_loss += loss.detach(); run_n += 1
        if (st + 1) % log_every == 0:                      # mean full-step loss since the last log line
            if world > 1:
                dist.all_reduce(run_loss)
            el = time.time() - t0; rate = el / (st + 1)
            if rank == 0:
                print(f"[xenc f{fold}] step {st + 1}/{steps} ({100 * (st + 1) / steps:.1f}%) loss {run_loss.item() / run_n:.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} {rate:.3f}s/step elapsed {el / 60:.1f}m eta {rate * (steps - st - 1) / 60:.1f}m "
                      f"peak_mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G", flush=True)
            run_loss, run_n = torch.zeros((), device="cuda"), 0
        if dev is not None and (st + 1) % EVAL_EVERY == 0:
            stop = False
            if rank == 0:
                ap = _dev_ap(model, dev)
                print(f"[xenc f{fold}] step {st + 1}: dev AP {ap:.6f} (best {best_ap:.6f})", flush=True)
                if ap > best_ap:
                    best_ap, bad = ap, 0
                    torch.save(model.state_dict(), config.WORK_DIR / f"xenc_fold{fold}.pt")
                else:
                    bad += 1
                    stop = bad >= patience
                    if stop:
                        print(f"[xenc f{fold}] early stop at step {st + 1}", flush=True)
            if world > 1:                                  # every rank takes rank 0's early-stop decision
                flag = torch.tensor([int(stop)], device="cuda")
                dist.broadcast(flag, 0)
                stop = bool(flag.item())
            if stop:
                break
    if rank == 0 and (dev is None or best_ap < 0):
        torch.save(model.state_dict(), config.WORK_DIR / f"xenc_fold{fold}.pt")
    if world > 1:
        dist.barrier()
    model.load_state_dict(torch.load(config.WORK_DIR / f"xenc_fold{fold}.pt", map_location="cuda"))
    return model


@torch.no_grad()
def score(model: ListwiseCE, pairs: pl.DataFrame, split: str, bs: int | None = None, chunk: int = 500_000) -> pl.DataFrame:
    """Per-pair outputs: listwise logit, OVA inlier prob; and per-record NIL-inclusive softmax (D1 inputs).
    Throughput: pairs are tokenised in large Rust batches (parallel) and scored in length-sorted batches
    (minimal padding); results are scattered back to the input order. Measured need: 47M uncertain pairs,
    ~3k pairs/s with per-batch tokenisation."""
    import os
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    tx = _texts(split)
    tmap = dict(zip(tx["idx"].to_list(), tx["text"].to_list()))
    tok = _tokenizer()
    bs = bs or SCORE_BS
    model.eval()
    q, s1 = pairs["q"].to_list(), pairs["s1"].to_list()
    out_s = np.empty(len(q), dtype=np.float32)
    out_o = np.empty(len(q), dtype=np.float32)
    for c0 in range(0, len(q), chunk):
        a = [tmap[x] for x in q[c0:c0 + chunk]]
        b = [tmap[x] for x in s1[c0:c0 + chunk]]
        enc = _encode(tok, a, b)                                     # Rust batch, no padding
        ids = enc["input_ids"]
        order = np.argsort([len(x) for x in ids], kind="stable")
        for i in range(0, len(order), bs):
            sel = order[i:i + bs]
            L = max(len(ids[k]) for k in sel)                        # manual padding (8x faster than tok.pad)
            arr = np.full((len(sel), L), tok.pad_token_id, dtype=np.int64)
            msk = np.zeros((len(sel), L), dtype=np.int64)
            for r, k in enumerate(sel):
                arr[r, :len(ids[k])] = ids[k]
                msk[r, :len(ids[k])] = 1
            batch = {"input_ids": torch.from_numpy(arr).cuda(non_blocking=True),
                     "attention_mask": torch.from_numpy(msk).cuda(non_blocking=True)}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                sc, o, *_ = model(batch)
            out_s[c0 + sel] = sc.float().cpu().numpy()
            out_o[c0 + sel] = torch.softmax(o.float(), 1)[:, 1].cpu().numpy()
    res = pairs.select("q", "s1").with_columns(pl.Series("x_logit", out_s), pl.Series("x_ova", out_o))
    nil = float(model.nil.item())
    return res.with_columns(
        (pl.col("x_logit").exp() / (pl.col("x_logit").exp().sum().over("q") + np.exp(nil))).alias("x_pi"))
