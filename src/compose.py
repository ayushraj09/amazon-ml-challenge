"""Merge the outputs of pipeline runs by country (used when the cross-encoder backbone differs per country route).

    python -m src.compose --out output --route US=<dirA> India=<dirA> France=<dirB>

Each <dir> holds matching_results.tsv and candidate_pairs.tsv from a full pipeline run (run_all.sh with its own
ER_WORK_DIR / ER_OUT_DIR and, e.g., ER_XENC_BACKBONE). Candidate sets must be identical (the candidate generation does
not depend on the cross-encoder). Countries not listed take the first route's rows. Rows keep the S1 order of the
first route; the official validator is run on the result.
"""
import argparse
import filecmp
import subprocess
import sys
from pathlib import Path

import polars as pl

from . import config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--route", nargs="+", required=True, help="COUNTRY=DIR")
    args = ap.parse_args()
    route = dict(r.split("=", 1) for r in args.route)
    dirs = list(dict.fromkeys(route.values()))
    for d in dirs[1:]:
        assert filecmp.cmp(Path(dirs[0]) / "candidate_pairs.tsv", Path(d) / "candidate_pairs.tsv", shallow=False), \
            f"candidate sets differ between {dirs[0]} and {d}"
    s1 = pl.read_csv(config.DATA_DIR / "test" / "test_source1.tsv", separator="\t", quote_char=None,
                     columns=["entity_id", "country"]).rename({"entity_id": "source1_entity_id"})
    read = {d: pl.read_csv(Path(d) / "matching_results.tsv", separator="\t", quote_char=None,
                           schema_overrides={"matched_entity_ids": pl.Utf8}).join(s1, on="source1_entity_id", maintain_order="left")
            for d in dirs}
    base = read[dirs[0]]
    parts = [read[route.get(c, dirs[0])].filter(pl.col("country") == c) for c in base["country"].unique().to_list()]
    out = base.select("source1_entity_id").join(pl.concat(parts).select("source1_entity_id", "matched_entity_ids"),
                                                on="source1_entity_id", how="left", maintain_order="left")
    assert out.height == base.height
    o = Path(args.out); o.mkdir(parents=True, exist_ok=True)
    out.write_csv(o / "matching_results.tsv", separator="\t", quote_style="never", null_value="")
    subprocess.run(["cp", str(Path(dirs[0]) / "candidate_pairs.tsv"), str(o / "candidate_pairs.tsv")], check=True)
    print(f"[compose] {route} -> {o}", flush=True)
    validator = config.DATA_DIR.parent / "utils" / "validate_submission.py"
    subprocess.run([sys.executable, str(validator), "--matching", str(o / "matching_results.tsv"),
                    "--candidate", str(o / "candidate_pairs.tsv"), "--test-dir", str(config.DATA_DIR / "test")], check=False)


if __name__ == "__main__":
    main()
