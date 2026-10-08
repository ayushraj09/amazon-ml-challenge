"""End-to-end pipeline: raw TSVs -> output/matching_results.tsv + output/candidate_pairs.tsv.

python -m src.run_pipeline [--stages prep,mine,normalize,synth,dense,block,stage1,...]
Every stage reads/writes parquet under work/ so stages can be re-run individually.
"""
import argparse
import time

STAGES = ["prep", "mine", "normalize", "synth", "dense", "block", "stage1", "evm", "train", "train_tr", "write", "xenctrain", "xenc",
          "trainx", "trainx_tr", "writex"]
# optional: "selftrain_tr" (self-training on unseen countries; measured, not used in the submission)
# optional, measured and not used in the submission: "xgbx" (CatBoost secondary), "blend", "writefinal"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default=",".join(STAGES))
    args = ap.parse_args()
    for st in args.stages.split(","):
        t0 = time.time()
        if st == "prep":
            from . import prep; prep.run()
        elif st == "mine":
            from . import mine; mine.run()
        elif st == "normalize":
            from . import normalize; normalize.run()
        elif st == "synth":
            from . import synth; synth.run()
        elif st == "dense":
            from . import biencoder; biencoder.run()
        elif st == "block":
            from . import blocking; blocking.run()
        elif st == "stage1":
            from . import stage1; stage1.run()
        elif st == "stage1train":
            from . import stage1; stage1.run_train()
        elif st == "stage1test":
            from . import stage1; stage1.run_test()
        elif st == "evm":
            from . import evm; evm.run()
        elif st == "train":
            from . import train; train.run()
        elif st == "train_tr":       # transfer feature set (routed to countries absent from train)
            from . import train; train.run(featset="transfer")
        elif st == "trainx_tr":
            from . import train; train.run(use_x=True, featset="transfer")
        elif st == "selftrain_tr":   # self-training of the transfer model on countries absent from train
            from . import train; train.run_selftrain("transfer")
        elif st == "predx":          # save stacked test probabilities (input of src.ensemble)
            from . import write_outputs; write_outputs.save_stacked_preds()
        elif st == "write":
            from . import write_outputs; write_outputs.run()
        elif st == "xencfix":
            from . import xenc; xenc.fix_models()
        elif st == "xenctrain":
            from . import xenc; xenc.train_models()
        elif st == "xenc":
            from . import xenc; xenc.run()
        elif st == "trainx":
            from . import train; train.run(use_x=True)
        elif st == "writex":
            from . import write_outputs; write_outputs.run(use_x=True)
        elif st == "xgb":      # historical stage name: the SECONDARY (blend-partner) learner, parallel job
            from . import train; train.run_secondary(use_x=False)
        elif st == "xgbx":
            from . import train; train.run_secondary(use_x=True)
        elif st == "blend":
            from . import train; train.run_blend(use_x=True)
        elif st == "writefinal":
            from . import write_outputs; write_outputs.run_final()
        else:
            raise ValueError(st)
        print(f"== stage {st} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
