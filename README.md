# Business Entity Resolution: Amazon ML Challenge 2026

This is **Team Mayfair's** solution to the **Amazon ML Challenge 2026**, which finished in the **top 50 teams**. The dataset is the one Amazon provided for the challenge. It is not included in this repository; see *Data* below.

📖 **Write-up:** [How We Finished Top 50 of 32,000+ Teams at the Amazon ML Challenge 2026](https://medium.com/@ayush.ttps/how-we-finished-top-50-of-32-000-teams-at-the-amazon-ml-challenge-2026-a4a28f9e7de8) (Medium) explains the approach from scratch.

This pipeline turns the raw challenge TSVs into `output/matching_results.tsv` and `output/candidate_pairs.tsv`. The method is described in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) (PDF: [`docs/Mayfair_Methodology.pdf`](docs/Mayfair_Methodology.pdf)).

**This repository contains G5, our final submission (public LB 0.990654).** To reproduce it, follow *Environment*, *Data* and *Reproduce the final submission* below.

## Repository layout
```
.
├── src/                    # pipeline stages (python -m src.run_pipeline --stages ...)
├── tests/                  # unit tests (python -m pytest -q tests)
├── scripts/                # end-to-end runners: run_all.sh, run_final.sh, run_ensemble.sh
├── experiments/
│   └── qwen4b_remote_training/   # exact code + logs of the Qwen3-4B DDP training run
├── docs/                   # methodology write-up (Markdown + PDF)
├── requirements.txt
└── LICENSES.md             # model and dependency licences
```
Generated and local-only folders (not tracked by git): `student_resource/` (the dataset), `work*/` (stage caches), `output/` (submission TSVs).

## Environment
- Python 3.11. Install the pinned versions with `pip install -r requirements.txt`.
- For torch on Blackwell GPUs, use the cu128 wheel index: `--index-url https://download.pytorch.org/whl/cu128`.
- `libmr` must be installed with `pip install --no-build-isolation libmr`, after numpy and cython are installed.
- Hardware used: 64 CPU cores, 251 GB RAM, 1× 48 GB GPU.

## Data
Point `ER_DATA_DIR` at the provided dataset folder, the one containing `train/` (`train_source{1,2,3}.tsv`, `train_ground_truth.tsv`) and `test/` (`test_source{1,2,3}.tsv`):
```bash
export ER_DATA_DIR=/path/to/student_resource/dataset
```
Without it, the default is `student_resource/dataset` at the repo root. If the official `utils/validate_submission.py` sits next to the dataset folder (as in `student_resource/`), it is run automatically after each write; otherwise that call only prints an error, and the outputs are still written. Other paths (all relative to the repo root):
- `ER_DATA_DIR`: the input dataset;
- `ER_WORK_DIR`: the cache (default `work/`);
- `ER_OUT_DIR`: the outputs (default `output/`).

## Reproduce the final submission
**Final submission (public LB 0.990654). To regenerate both output files from the data, end to end (from the repo root):**
```bash
./scripts/run_all.sh       # run A: data -> normalisation -> blocking (dense + lexical) -> stage-1 filter (= candidate_pairs.tsv)
                           #        -> stage-2 XGBoost -> mmBERT-small cross-encoder -> stacked models        (work dir work/)
./scripts/run_final.sh  # runs B (Qwen3-1.7B), C (+format aug), V9 (+filler aug), 4B (Qwen3-4B) on run A's candidates,
                           # then the weighted 4-way ensemble + guards + filler rule -> output/{matching,candidate}_*.tsv
```
- `scripts/run_final.sh` documents each run's settings; resume with `START_FROM=B|C|V9|4B|ensemble`.
- Our 4B run used 2 GPUs per fold with DDP; the exact code and logs are in `experiments/qwen4b_remote_training/`. On one GPU the script trains the same configuration sequentially.
- The final weights (US/India 1.5 / 1.5 / 3 / 0.75, France route 1.5 / 2 / 3 / 1 for A / B / 4B / V9) were chosen on validation: all 31 model subsets, then a coordinate search over the weights.
- **Expected reproduction:** GPU-nondeterministic training reproduces the models to within noise, not bit for bit. `candidate_pairs.tsv` depends only on run A and is deterministic.

## Earlier submissions (history, not needed for the final submission)
```bash
./scripts/run_ensemble.sh                     # E2h (LB 0.990013): run B (Qwen3-1.7B cross-encoder) + ensemble of runs A and B + guards
TRANSFER=a ./scripts/run_ensemble.sh          # variant E1h: unseen countries use run A's transfer model only
```

**Earlier submission, E5a (public LB 0.990078).** This is E2h with two additions: the French house-number parser fix, already in `normalize.py`, and France averaged with a third run (run C, a Qwen3-1.7B trained with French-format augmentation). After `./scripts/run_all.sh` (run A) and `./scripts/run_ensemble.sh` (run B):
```bash
# run C: like run B, with format augmentation (link run A's artefacts as run_ensemble.sh does for run B, into $WORK_C)
for st in xenctrain xenc trainx trainx_tr predx; do
  ER_WORK_DIR=$WORK_C ER_XENC_BACKBONE=Qwen/Qwen3-1.7B ER_XENC_MAX_RECORDS=300000 ER_XENC_AUG=0.5 python -m src.run_pipeline --stages $st
done
# E5a: US/India = A + B, France = A + C
ER_WORK_DIR=$WORK_E ER_OUT_DIR=output python -m src.ensemble --runs $WORK_A $WORK_B --transfer mean --transfer-runs $WORK_A $WORK_C
# E6 / E6r: run D = continued training of run C with FILLER augmentation (copy run C's xenc_fold*.pt into $WORK_D first)
for f in 0 1; do ER_WORK_DIR=$WORK_D ER_XENC_BACKBONE=Qwen/Qwen3-1.7B ER_XENC_MAX_RECORDS=100000 ER_XENC_AUG=0.5 ER_XENC_AUG_FILL=1 \
  ER_XENC_LR=1e-4 ER_XENC_SAMPLE_SEED=4242 ER_XENC_FRESH_BEST=1 python -c "from src import xenc; xenc.train_models(folds=($f,), resume=True)"; done
for st in xenc trainx trainx_tr predx; do ER_WORK_DIR=$WORK_D ER_XENC_BACKBONE=Qwen/Qwen3-1.7B python -m src.run_pipeline --stages $st; done
ER_WORK_DIR=$WORK_E ER_OUT_DIR=output python -m src.ensemble --runs $WORK_A $WORK_B --transfer mean --transfer-runs $WORK_A $WORK_D          # E6
ER_HARD_POSITIVE=1 ER_WORK_DIR=$WORK_E ER_OUT_DIR=output python -m src.ensemble --runs $WORK_A $WORK_B --transfer mean --transfer-runs $WORK_A $WORK_D   # E6r
python -m src.run_pipeline            # all stages in one process
python -m src.run_pipeline --stages trainx,writex # re-run selected stages (earlier outputs are cached in work/)
python -m pytest -q tests             # unit tests (scorer, normaliser, decision rule, losses, calibration, alignment, synthetics)
```

| Stage | Module | Device | Output (in `work/`) |
|---|---|---|---|
| prep | `prep.py` | CPU | `records.parquet`: integer ids, duplicate groups, validation S1 flag; `gt.parquet` |
| mine | `mine.py` | CPU | `dicts.json`: Indic→Latin tokens, per-country state/street maps, legal variants, all mined from the provided data |
| normalize | `normalize.py` | CPU | `norm.parquet`: core name, legal form, house number/tuple, street, … |
| synth | `synth.py` | CPU | synthetic near-miss distractors at the measured test rates (train only; `syn_meta.json`) |
| dense | `biencoder.py` | GPU | 2 cross-fitted `multilingual-e5-small` bi-encoders + exact top-50 retrieval → `cands_dense.parquet` |
| block | `blocking.py` | CPU | dense + word TF-IDF + exact-key channels → `cands.parquet`, with a recall report |
| stage1 (`stage1train`, `stage1test`) | `stage1.py` | GPU (XGBoost) | cheap-feature candidate filter, 5-fold OOF; keeps a record's top-2 candidates with p ≥ 0.01 (`KEEP_RULE`) → `pruned.parquet` = `candidate_pairs.tsv` (4.89 per test S1) |
| evm | `evm.py` | GPU + CPU | Extreme Value Machine inclusion probability per pair → `evm.parquet` |
| train | `train.py` | GPU (XGBoost) | pair features (incl. label-free per-country filler and real-word vocabularies → `fillers.json`) + open-set features; stage-2 full-feature model, 5-fold GroupKFold OOF, `F_val` / `F_shift`, decision tuning |
| train_tr | `train.py` | GPU (XGBoost) | stage-2 **transfer** model (no country-specific features), routed to countries absent from train |
| write | `write_outputs.py` | CPU | v1 test prediction, per-S1 expected-F0.5 decision, both TSVs, official validator |
| xenctrain, xenc | `cross_encoder.py`, `xenc.py` | GPU | listwise `mmBERT-small` cross-encoder with a NIL option (2 cross-fitted models), scores of the top-8 candidates of uncertain records → `xenc.parquet` |
| trainx, trainx_tr | `train.py` | GPU (XGBoost) | stacked models (full / transfer) on cross-encoder outputs + sibling consensus |
| writex | `write_outputs.py` | CPU | **final** test prediction, country-routed (train countries → full model, other countries → transfer model): both TSVs + official validator |
| predx | `write_outputs.py` | CPU | stacked test probabilities of a run (`test/pred_stage2x{,_tr}.parquet`), the input of the ensemble |
| `scripts/run_ensemble.sh` → `src.ensemble` | `ensemble.py`, `write_outputs.py` | GPU (run B) + CPU | run B = the same candidates and stage-2 models with a `Qwen/Qwen3-1.7B` + LoRA listwise cross-encoder and its own stacked models; averages the two runs' stacked probabilities, re-tunes the decision on the averaged OOF, writes both TSVs with the unseen-country guards (earlier submissions E1h/E2h; the final submission uses `scripts/run_final.sh`, the same `src.ensemble` with four weighted runs) |
| (optional) xgbx, blend, writefinal | `train.py`, `write_outputs.py` | GPU (CatBoost) | CatBoost secondary + log-loss blend; measured, blend weight 0 → not part of the submission |

**Hard-positive rule** (env `ER_HARD_POSITIVE=1`, submission E5r and later): for unseen countries, a record whose name is the S1 name with its category word replaced or extended by that country's mined **filler** words, at the same house number and street, is matched to its best S1. The pattern is re-derived from train labels (US 0.995, India 0.978 true, n ≥ 29k) and applied only if every train country is ≥ 0.95.

**Unseen-country guards** (in `write_outputs._write`, re-derived from train labels at every write):
- **Concept conflict:** a sibling real-word swap at the same number.
- **Hard negative:** same name, disjoint legal forms, number shifted by 1–50 or no address.
- **Country word:** the record name is the S1 name plus the country's own name.

Pairs matching a guarded pattern are not matched. See the methodology document, §4.5.

## Candidate set (`candidate_pairs.tsv`)
- The candidate set is the output of stage 1: retrieval (dense bi-encoder + lexical channels, about 50 per record) filtered by a cheap learned scorer (`stage1.KEEP_RULE`: a record's 2 best candidates with p ≥ 0.01; 4.89 per test S1, validation recall ceiling 0.986).
- Every later model (stage-2 matcher, cross-encoder, stacked model) runs inference only on this set, so `candidate_pairs.tsv` is exactly what the matching model scores, and every match is a candidate.
- The rule was chosen on the measured size / recall / F0.5 frontier; see the methodology document, section 3.

## Useful environment variables
- `ER_THREADS`: CPU threads (default: all).
- `ER_XGB_DEVICE`: `cuda` (default) or `cpu` for the stage-2 XGBoost models.
- `ER_XENC_GPU_FRACTION`: per-process GPU memory cap for the cross-encoder (default 0.6).
- `START_FROM=<stage> ./scripts/run_all.sh`: resume from a stage; `START_FROM=b|predx|ensemble ./scripts/run_ensemble.sh` likewise.
- `ER_XENC_BACKBONE`: cross-encoder backbone (`jhu-clsp/mmBERT-small` default; `Qwen/Qwen3-1.7B` for run B); `ER_XENC_MAX_RECORDS`: training records per fold.
- `WORK_A`, `WORK_B`, `WORK_E`, `TRANSFER`: work directories and France route of `scripts/run_ensemble.sh`.

## Compliance
- No external data or APIs.
- Every dictionary is mined from the provided train/test files; see `mine.py`.
- Pretrained weights come from MIT or Apache-2.0 models (the largest is Qwen3-4B; the limit is 8B); see `LICENSES.md`.
- The cross-encoder augmentation (`cross_encoder._augment`) contains hand-written French surface forms, which are named in the methodology document's compliance section.
