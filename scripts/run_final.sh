#!/bin/bash
# Regenerates the FINAL submission (public LB 0.990654) after ./scripts/run_all.sh (run A).
#   run B  : Qwen3-1.7B listwise cross-encoder                                   ($WORK_B)
#   run C  : Qwen3-1.7B + French-format augmentation (ER_XENC_AUG=0.5)             ($WORK_C)
#   run V9 : run C continued with French-filler augmentation (ER_XENC_AUG_FILL=1)  ($WORK_V9)
#   run 4B : Qwen3-4B + format augmentation, 2 folds x 352k records               ($WORK_4B)
#            (our run used 2 GPUs per fold with DDP: experiments/qwen4b_remote_training/; on one GPU this script trains the
#             same configuration sequentially, which takes ~8-10 h; fold 1 needed lr 1e-4 after diverging at 2e-4)
# Every run shares run A's candidates and stage-2 models (linked), then: cross-encoder training -> scoring -> stacked
# models (full + transfer) -> test predictions (predx). Final: weighted 4-way ensemble on both routes + the
# train-justified unseen-country guards and the hard-positive filler rule -> output/{matching_results,candidate_pairs}.tsv
# Resume with START_FROM=B|C|V9|4B|ensemble.
set -eu
HERE=$(cd "$(dirname "$0")/.." && pwd)  # repo root
WORK_A=${WORK_A:-$HERE/work}
WORK_B=${WORK_B:-$HERE/work_qwen}
WORK_C=${WORK_C:-$HERE/work_qwen_aug}
WORK_V9=${WORK_V9:-$HERE/work_qwen_fill}
WORK_4B=${WORK_4B:-$HERE/work_qwen4b}
WORK_E=${WORK_E:-$HERE/work_final}
OUT=${ER_OUT_DIR:-$HERE/output}
PY=${PY:-python}
START_FROM=${START_FROM:-B}
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$HERE"
[ -f "$WORK_A/decision_stage2x_tr.json" ] || { echo "run A is not complete in $WORK_A (run ./scripts/run_all.sh first)"; exit 1; }

link_run() {  # new run dir sharing run A's data, candidates and stage-2 models (not its cross-encoder / stacked outputs)
  local W=$1; mkdir -p "$W/train" "$W/test"
  for f in "$WORK_A"/*; do case $(basename "$f") in train|test|xenc_*|stage2x*|oof_stage2x*|decision_stage2x*|iso_stage2x*|errors_*|*.log) ;;
    *) ln -sfn "$f" "$W/" ;; esac; done
  for s in train test; do for f in "$WORK_A/$s"/*; do case $(basename "$f") in xenc.parquet|pred_*) ;; *) ln -sfn "$f" "$W/$s/" ;; esac; done; done
  cp "$WORK_A/test/pred_stage2.parquet" "$WORK_A/test/pred_stage2_tr.parquet" "$W/test/"
}
stages() { local W=$1; shift; for st in "$@"; do echo "=== $(basename "$W"): $st $(date)"; ER_WORK_DIR=$W ER_OUT_DIR=$W/output $PY -m src.run_pipeline --stages $st; done; }
step() { case "$START_FROM" in B) return 0;; C) [ "$1" != B ];; V9) [ "$1" != B ] && [ "$1" != C ];; 4B) [ "$1" = 4B ] || [ "$1" = ensemble ];; ensemble) [ "$1" = ensemble ];; esac; }

if step B; then link_run "$WORK_B"
  ER_XENC_BACKBONE=Qwen/Qwen3-1.7B ER_XENC_MAX_RECORDS=300000 stages "$WORK_B" xenctrain xenc trainx trainx_tr predx; fi
if step C; then link_run "$WORK_C"
  ER_XENC_BACKBONE=Qwen/Qwen3-1.7B ER_XENC_MAX_RECORDS=300000 ER_XENC_AUG=0.5 stages "$WORK_C" xenctrain xenc trainx trainx_tr predx; fi
if step V9; then link_run "$WORK_V9"; cp "$WORK_C"/xenc_fold0.pt "$WORK_C"/xenc_fold1.pt "$WORK_V9/"
  for f in 0 1; do ER_WORK_DIR=$WORK_V9 ER_XENC_BACKBONE=Qwen/Qwen3-1.7B ER_XENC_MAX_RECORDS=100000 ER_XENC_AUG=0.5 ER_XENC_AUG_FILL=1 \
    ER_XENC_LR=1e-4 ER_XENC_SAMPLE_SEED=4242 ER_XENC_FRESH_BEST=1 $PY -c "from src import xenc; xenc.train_models(folds=($f,), resume=True)"; done
  touch "$WORK_V9/xenc_models.ready"
  ER_XENC_BACKBONE=Qwen/Qwen3-1.7B stages "$WORK_V9" xenc trainx trainx_tr predx; fi
if step 4B; then link_run "$WORK_4B"
  ER_XENC_BACKBONE=Qwen/Qwen3-4B ER_XENC_MAX_RECORDS=352000 ER_XENC_AUG=0.5 ER_XENC_SCORE_BS=128 stages "$WORK_4B" xenctrain xenc trainx trainx_tr predx; fi

echo "=== final ensemble $(date)"
mkdir -p "$WORK_E" "$OUT"
for f in train test fillers.json; do ln -sfn "$WORK_A/$f" "$WORK_E/$f"; done
ER_HARD_POSITIVE=1 ER_WORK_DIR=$WORK_E ER_OUT_DIR=$OUT $PY -m src.ensemble \
  --runs "$WORK_A" "$WORK_B" "$WORK_4B" "$WORK_V9" --weights-full 1.5 1.5 3 0.75 \
  --transfer mean --transfer-runs "$WORK_A" "$WORK_B" "$WORK_4B" "$WORK_V9" --weights-transfer 1.5 2 3 1
echo "=== final submission written to $OUT $(date)"
