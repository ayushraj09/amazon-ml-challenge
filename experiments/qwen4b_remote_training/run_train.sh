#!/bin/bash
# Qwen3-4B cross-encoder: 2 folds x 352k records, each fold data-parallel on 2 GPUs. Isolated work dir.
W=/workspace/MLC/work_qwen4b
CODE=/workspace/MLC/code/business_entity_resolution
cd $CODE && source /workspace/MLC/.venv/bin/activate
unset ER_WORK_DIR
export PYTHONPATH=. ER_WORK_DIR=$W ER_XENC_FOLDS=2 ER_XENC_MAX_RECORDS=352000 ER_XENC_AUG=0.5 \
       ER_XENC_BACKBONE=Qwen/Qwen3-4B ER_XENC_GPU_FRACTION=0.95 ER_XENC_SCORE_BS=128 ER_XENC_LOG_EVERY=100 \
       PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=$W/logs/run_manifest.txt
{
  echo "run started: $(TZ=Asia/Kolkata date '+%Y-%m-%d %H:%M:%S IST')"
  echo "layout: 2 folds, fold 0 on GPUs 0,1 (torchrun port 29501), fold 1 on GPUs 2,3 (port 29502)"
  echo "records per fold: 352000 | records per step: 32 (16 per GPU) | steps per fold: 11000 | epochs: 1"
  echo "dev eval every 1000 steps on 3000 uncertain records of the other fold; patience 5; best checkpoint kept"
  echo "backbone: Qwen/Qwen3-4B revision $(python -c 'from src import cross_encoder as c; print(c.BACKBONE_REVISION)')"
  echo "LoRA r64 alpha16 dropout0.1 all-linear, bf16, grad checkpointing, lr 2e-4 linear warmup 0.1, format aug p=0.5"
  echo "--- env"; env | grep -E '^(ER_|PYTORCH_|CUDA_)' | sort
  echo "--- packages"; pip list 2>/dev/null | grep -iE '^(torch|transformers|peft|tokenizers|accelerate|safetensors|huggingface.hub|polars|pyarrow|numpy|scikit-learn|scipy|xgboost|numba|lightgbm) '
  echo "--- code md5"; md5sum src/cross_encoder.py src/xenc.py
  echo "--- input md5"; cat $W/logs/input_md5.txt
  echo "--- GPUs"; nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
} > $M
mkdir -p $W/code_snapshot && cp src/cross_encoder.py src/xenc.py $W/code_snapshot/
rm -f $W/xenc_fold*.pt $W/xenc_models.ready $W/xenc_train.running
CUDA_VISIBLE_DEVICES=0,1 nohup torchrun --nproc_per_node=2 --master_port=29501 $W/xenc_train_ddp.py 0 > $W/logs/train_f0.log 2>&1 &
CUDA_VISIBLE_DEVICES=2,3 nohup torchrun --nproc_per_node=2 --master_port=29502 $W/xenc_train_ddp.py 1 > $W/logs/train_f1.log 2>&1 &
nohup python $W/progress_monitor.py > $W/logs/progress_monitor.out 2>&1 &
echo "launched at $(TZ=Asia/Kolkata date +%H:%M:%S) IST" | tee -a $M
