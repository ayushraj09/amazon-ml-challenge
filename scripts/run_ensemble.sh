#!/bin/bash
# Final submission: ensemble of two runs that differ only in the cross-encoder backbone.
#   run A = ./scripts/run_all.sh (listwise mmBERT-small cross-encoder), work dir $WORK_A
#   run B = the same data, candidates and stage-2 models (linked from A), listwise Qwen3-1.7B + LoRA cross-encoder
#           (Apache-2.0, 1.7B parameters), its own stacked models, work dir $WORK_B
# then src.ensemble averages the two runs' stacked probabilities, re-tunes the per-S1 expected-F0.5 decision on the
# averaged out-of-fold validation probabilities, and writes both TSVs with the unseen-country guards
# (write_outputs: concept-conflict, hard-negative and country-word guards) + the official validator.
#   TRANSFER=a    unseen countries use run A's transfer model   -> submission E1h
#   TRANSFER=mean (default) unseen countries use the average of both transfer models -> E2h (LB 0.990013); add
#                 --transfer-runs <A> <C> in src.ensemble for E5a (final, LB 0.990078: France averaged with the augmented Qwen run)
# Run A first: ./scripts/run_all.sh. Resumable per step with START_FROM=b|predx|ensemble.
set -eu
HERE=$(cd "$(dirname "$0")/.." && pwd)  # repo root
WORK_A=${WORK_A:-$HERE/work}
WORK_B=${WORK_B:-$HERE/work_qwen}
WORK_E=${WORK_E:-$HERE/work_ens}
OUT=${ER_OUT_DIR:-$HERE/output}
TRANSFER=${TRANSFER:-mean}
START_FROM=${START_FROM:-b}
PY=${PY:-python}
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$HERE"
[ -f "$WORK_A/decision_stage2x_tr.json" ] || { echo "run A is not complete in $WORK_A (run ./scripts/run_all.sh first)"; exit 1; }

if [ "$START_FROM" = b ]; then
  echo "=== run B: link run A's artefacts up to stage 2, train the Qwen3-1.7B cross-encoder $(date)"
  mkdir -p "$WORK_B/train" "$WORK_B/test"
  for f in "$WORK_A"/*; do
    case $(basename "$f") in train|test|xenc_*|stage2x*|oof_stage2x*|decision_stage2x*|iso_stage2x*|errors_*|*.log) ;;
      *) ln -sfn "$f" "$WORK_B/" ;; esac
  done
  for s in train test; do
    for f in "$WORK_A/$s"/*; do
      case $(basename "$f") in xenc.parquet|pred_*) ;; *) ln -sfn "$f" "$WORK_B/$s/" ;; esac
    done
  done
  cp "$WORK_A/test/pred_stage2.parquet" "$WORK_A/test/pred_stage2_tr.parquet" "$WORK_B/test/"
  for st in xenctrain xenc trainx trainx_tr; do
    echo "=== START B $st $(date)"
    ER_WORK_DIR=$WORK_B ER_OUT_DIR=$WORK_B/output ER_XENC_BACKBONE=Qwen/Qwen3-1.7B ER_XENC_MAX_RECORDS=300000 \
      $PY -m src.run_pipeline --stages $st
  done
  START_FROM=predx
fi
if [ "$START_FROM" = predx ]; then
  for w in "$WORK_A" "$WORK_B"; do
    echo "=== START predx $w $(date)"
    ER_WORK_DIR=$w ER_OUT_DIR=$w/output $PY -m src.run_pipeline --stages predx
  done
  START_FROM=ensemble
fi
echo "=== START ensemble (transfer=$TRANSFER) $(date)"
mkdir -p "$WORK_E" "$OUT"
for f in train test fillers.json; do ln -sfn "$WORK_A/$f" "$WORK_E/$f"; done
ER_WORK_DIR=$WORK_E ER_OUT_DIR=$OUT $PY -m src.ensemble --runs "$WORK_A" "$WORK_B" --transfer "$TRANSFER"
echo "=== ALL DONE $(date)"
