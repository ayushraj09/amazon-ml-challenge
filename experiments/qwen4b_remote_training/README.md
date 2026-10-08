# Qwen3-4B cross-encoder run (run D, used in submissions G2 / G3)

The 4B run was trained on a second machine (4 × RTX PRO 4000 Blackwell, 24 GB), with the data-parallel (DDP) training variant of the cross-encoder. This folder holds that exact code and its logs.

| File | What it is |
|---|---|
| `cross_encoder.py`, `xenc.py` | code snapshot used for training. Same model, loss and augmentation as `src/`, plus torchrun/DDP support, `ER_XENC_FOLDS`, sharded `score_fold` / `merge_parts`. |
| `run_train.sh`, `xenc_train_ddp.py` | launch: 2 folds, each on 2 GPUs (torchrun), 352k records per fold, 32 records per step, format augmentation p = 0.5 |
| `xenc_train_ddp_resume.py`, `train_f1_run1_diverged.log` | fold 1 diverged at lr 2e-4 (steps 1,300–1,500). It was resumed from its step-1,000 checkpoint with lr 1e-4 and a fresh 10% warmup. |

**Note on the resume code.** `cross_encoder.py` here is the snapshot taken when training started. The fold-1 resume used a later edit on the training machine, adding a start-step option, which was lost when that machine was recycled.
- What it did is documented in `run_manifest.txt`: resume from the step-1,000 checkpoint; the same 352k record order, continuing at record 32,000; lr 1e-4 with a fresh 10% warmup over the remaining 10,000 steps; the checkpoint overwritten only when dev AP beats 0.990258.
- The final checkpoints of both folds, the resulting scores and all logs are kept. G2 / G3 and the final submission are built from those scores.
