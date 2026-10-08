#!/bin/bash
# End-to-end run (raw TSVs -> output/matching_results.tsv + output/candidate_pairs.tsv), one stage per process.
# Stages run strictly one after another: running two memory-heavy stages at once caused an out-of-memory kill at 125M pairs.
# Resumable: START_FROM=<stage> skips the stages before it (their outputs are cached in $ER_WORK_DIR).
set -u
HERE=$(cd "$(dirname "$0")/.." && pwd)  # repo root
export ER_WORK_DIR=${ER_WORK_DIR:-$HERE/work} ER_OUT_DIR=${ER_OUT_DIR:-$HERE/output} PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$ER_WORK_DIR" "$ER_OUT_DIR"
cd "$HERE"
PY=${PY:-python}
# The final submission is written by `writex` (stacked XGBoost). The optional CatBoost secondary + blend stages
# (xgbx, blend, writefinal) were measured and not used: the validation log-loss-optimal blend weight was 1.0 for XGBoost.
STAGES=(prep mine normalize synth dense block stage1train stage1test evm
        train train_tr write xenctrain xenc trainx trainx_tr writex)
# optional, measured and not used: selftrain_tr (self-training on unseen countries; accepted more sibling pairs in France)
START_FROM=${START_FROM:-prep}
skip=1
for st in "${STAGES[@]}"; do
  [ "$st" = "$START_FROM" ] && skip=0
  [ $skip = 1 ] && { echo "=== SKIP $st"; continue; }
  echo "=== START $st $(date)"
  $PY -m src.run_pipeline --stages $st || { echo "=== FAILED $st $(date)"; exit 1; }
  echo "=== DONE $st $(date)"
done
echo "=== ALL DONE $(date)"
