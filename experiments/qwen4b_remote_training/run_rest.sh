#!/bin/bash
# Remaining 4B pipeline, run automatically after fold 1 training:
#   1) model 1 scoring on 4 GPUs  2) merge + acceptance checks  3) copy xenc.parquet into work/
#   4) trainx, trainx_tr, predx (stacked models on the 4B scores)  5) runs/D_qwen4b
#   6) ensembles: E5a with both 1.7B slots -> 4B (A+D / A+D) and France-only swap (A+B / A+D)  7) France check vs E5a
# Stops at the first failure. Status lines go to $W/PROGRESS.log and $W/logs/run_rest.log.
set -uo pipefail
W=/workspace/MLC/work_qwen4b; M=/workspace/MLC; CODE=$M/code/business_entity_resolution
LOG=$W/logs/run_rest.log
say() { echo "=== $(TZ=Asia/Kolkata date '+%H:%M:%S IST') $*" | tee -a $LOG >> $W/PROGRESS.log; }
die() { say "FAILED: $*"; exit 1; }
cd $CODE && source $M/.venv/bin/activate
export PYTHONPATH=. PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ER_XENC_BACKBONE=Qwen/Qwen3-4B

# 1) wait for fold 1 training, then score model 1 on all 4 GPUs
while pgrep -f "work_qwen4b/xenc_train_ddp_resume.py" >/dev/null; do sleep 20; done
grep -q "fold 1: trained on" $W/logs/train_f1.log || die "fold 1 training did not finish cleanly (see logs/train_f1.log)"
touch $W/xenc_models.ready
say "fold 1 DONE: $(grep 'dev AP' $W/logs/train_f1.log | tail -1)"
say "model 1 scoring started on GPUs 0-3 (4 shards)"
for s in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$s ER_WORK_DIR=$W ER_XENC_FOLDS=2 ER_XENC_GPU_FRACTION=0.95 ER_XENC_SCORE_BS=128 \
    python -c "from src import xenc; xenc.score_fold(1, $s, 4)" > $W/logs/score_f1_$s.log 2>&1 &
done
wait
for s in 0 1 2 3; do grep -q "scored .* test pairs" $W/logs/score_f1_$s.log || die "model 1 shard $s scoring failed (logs/score_f1_$s.log)"; done
say "model 1 scoring DONE"

# 2) merge (model 0: 2 shards, model 1: 4 shards) + acceptance checks
ER_WORK_DIR=$W python - <<'EOF' 2>&1 | tee -a $LOG >> $W/logs/accept.log || die "merge/acceptance checks failed (logs/accept.log)"
import glob, sys, numpy as np, polars as pl
W = "/workspace/MLC/work_qwen4b"
ok = True
for split, n in (("train", 927_642), ("test", 1_082_365)):
    parts = sorted(glob.glob(f"{W}/{split}/xenc_part0_*of2.parquet")) + sorted(glob.glob(f"{W}/{split}/xenc_part1_*of4.parquet"))
    assert len(parts) == 6, parts
    d = pl.concat([pl.read_parquet(p) for p in parts]).with_columns((pl.col("x_pi") * pl.col("x_ova")).alias("x_q"))
    num = ["x_logit", "x_ova", "x_pi", "x_q"]
    fin = all(np.isfinite(d[c].to_numpy()).all() for c in num)
    dup = d.height - d.unique(["q", "s1"]).height
    rng = all(d[c].min() >= 0 and d[c].max() <= 1 for c in ("x_ova", "x_pi"))
    qsum = d.group_by("q").agg(pl.col("x_pi").sum())["x_pi"].max()
    cols = d.columns == ["q", "s1", "x_logit", "x_ova", "x_pi", "x_q"]
    good = d.height == n and dup == 0 and fin and rng and qsum <= 1 + 1e-5 and cols
    ok &= good
    print(f"{split}: rows {d.height:,} (expect {n:,}) dup {dup} finite {fin} ova/pi in [0,1] {rng} max per-q sum x_pi {qsum:.6f} "
          f"cols {d.columns} x_logit [{d['x_logit'].min():.2f}, {d['x_logit'].max():.2f}] -> {'OK' if good else 'FAIL'}")
    if good:
        d.write_parquet(f"{W}/{split}/xenc.parquet")
sys.exit(0 if ok else 1)
EOF
say "merge + acceptance: $(tail -2 $W/logs/accept.log | tr '\n' ' ')"
grep -q "FAIL" $W/logs/accept.log && die "acceptance checks failed (logs/accept.log)"

# 3) copy into the pipeline work dir
cp $W/train/xenc.parquet $M/work/train/xenc.parquet && cp $W/test/xenc.parquet $M/work/test/xenc.parquet || die "copy to work/"
(cd $M/work && md5sum train/xenc.parquet test/xenc.parquet) >> $W/logs/run_manifest.txt
say "xenc.parquet copied into work/train and work/test"

# 4) stacked models on the 4B scores (one GPU XGBoost job at a time), in work/
unset ER_WORK_DIR; export ER_XGB_DEVICE=cuda
for st in trainx trainx_tr predx; do
  say "stage $st started"
  python -m src.run_pipeline --stages $st > $M/work/$st.log 2>&1 || die "stage $st (work/$st.log)"
  say "stage $st DONE $(grep -h 'BEST' $M/work/$st.log | tail -1)"
done

# 5) run D
mkdir -p $M/runs/D_qwen4b/test $M/runs/D_qwen4b/train
cp $M/work/oof_stage2x.npy $M/work/oof_stage2x_tr.npy $M/runs/D_qwen4b/ || die "copy oof to runs/D"
cp $M/work/test/pred_stage2x.parquet $M/work/test/pred_stage2x_tr.parquet $M/runs/D_qwen4b/test/ || die "copy preds to runs/D"
ln -sfn $M/work/train/pairs_feat.parquet $M/runs/D_qwen4b/train/pairs_feat.parquet
say "runs/D_qwen4b created"

# 6) ensembles (CPU), each writes both TSVs and runs the official validator
R=$M/runs
say "ensemble E5a_4b_both (US/IN A+D, France A+D) started"
ER_WORK_DIR=$M/work_ens ER_OUT_DIR=$M/output_E5a_4b_both python -m src.ensemble --runs $R/A_mmbert_small $R/D_qwen4b \
  --transfer mean --transfer-runs $R/A_mmbert_small $R/D_qwen4b > $M/work_ens/E5a_4b_both.log 2>&1 || die "ensemble E5a_4b_both"
say "E5a_4b_both DONE: $(grep -h 'BEST' $M/work_ens/E5a_4b_both.log | sed 's/.*F_shift/F_shift/' | tr '\n' ' ') | $(grep -hE '^PASS|FAIL' $M/work_ens/E5a_4b_both.log | head -1)"
say "ensemble F1_france_only (US/IN A+B, France A+D) started"
ER_WORK_DIR=$M/work_ens ER_OUT_DIR=$M/output_F1 python -m src.ensemble --runs $R/A_mmbert_small $R/B_qwen17b \
  --transfer mean --transfer-runs $R/A_mmbert_small $R/D_qwen4b > $M/work_ens/F1.log 2>&1 || die "ensemble F1"
say "F1 DONE: $(grep -h 'BEST' $M/work_ens/F1.log | sed 's/.*F_shift/F_shift/' | tr '\n' ' ') | $(grep -hE '^PASS|FAIL' $M/work_ens/F1.log | head -1)"

# 7) France check vs E5a (doc §3.4) + candidate set identity
python - <<'EOF' 2>&1 | tee -a $LOG >> $W/PROGRESS.log
import polars as pl, hashlib
M = "/workspace/MLC"
s1 = pl.read_csv(f"{M}/student_resource/dataset/test/test_source1.tsv", separator="\t", quote_char=None,
                 columns=["entity_id", "country"]).rename({"entity_id": "source1_entity_id"})
def rd(p): return pl.read_csv(p, separator="\t", quote_char=None, schema_overrides={"matched_entity_ids": pl.Utf8}).join(s1, on="source1_entity_id")
ref = rd(f"{M}/reference/E5a_matching_results.tsv")
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
for name in ("output_E5a_4b_both", "output_F1"):
    b = rd(f"{M}/{name}/matching_results.tsv")
    x = ref.join(b.select("source1_entity_id", pl.col("matched_entity_ids").alias("new")), on="source1_entity_id")
    t = x.group_by("country").agg((pl.col("matched_entity_ids").fill_null("") == pl.col("new").fill_null("")).mean().alias("same_as_E5a"),
        pl.col("new").fill_null("").str.split(",").list.eval(pl.element().filter(pl.element() != "")).list.len().mean().alias("matches_per_S1")).sort("country")
    same_cand = md5(f"{M}/{name}/candidate_pairs.tsv") == md5(f"{M}/reference/E5a_candidate_pairs.tsv")
    print(f"--- {name}: candidate_pairs identical to E5a: {same_cand}")
    print(t)
EOF
say "ALL DONE: TSVs in /workspace/MLC/output_E5a_4b_both and /workspace/MLC/output_F1"
