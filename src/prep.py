"""Stage P: load raw TSVs, give every record an integer index, collapse exact duplicate S2/S3 rows.

Output per split: work/<split>/records.parquet with
  idx (int32, unique per split), eid, src (1/2/3), country, name, addr (null if missing),
  rep (idx of the duplicate-group representative), dup_count.
Train also gets work/train/gt.parquet (s1_eid, rec_eid) and the S1 validation flag.
Exact duplicate S2/S3 rows always carry the same label in train (FINDINGS §16.3), so scoring one
representative per group and expanding afterwards is lossless.
"""
import polars as pl

from . import config
from .io_utils import read_ground_truth, read_source


def build_records(split: str) -> pl.DataFrame:
    frames = []
    for s in config.SOURCES:
        df = read_source(split, s).rename({"entity_id": "eid", "business_name": "name", "business_address": "addr"})
        frames.append(df.with_columns(pl.lit(s, dtype=pl.Int8).alias("src")))
    rec = pl.concat(frames, how="vertical").with_row_index("idx").with_columns(pl.col("idx").cast(pl.Int32))
    rec = rec.with_columns(pl.col("name").fill_null(""))
    # duplicate groups among S2/S3 only (S1 is deduplicated by definition)
    key = ["src", "country", "name", "addr"]
    grp = (rec.filter(pl.col("src") > 1)
              .group_by(key, maintain_order=True)
              .agg(pl.col("idx").min().alias("rep"), pl.len().alias("dup_count")))
    rec = rec.join(grp, on=key, how="left", nulls_equal=True)
    rec = rec.with_columns(pl.col("rep").fill_null(pl.col("idx")).cast(pl.Int32),
                           pl.col("dup_count").fill_null(1).cast(pl.Int32))
    return rec.select("idx", "eid", "src", "country", "name", "addr", "rep", "dup_count")


def run() -> None:
    for split in config.SPLITS:
        rec = build_records(split)
        if split == "train":
            s1 = rec.filter(pl.col("src") == 1)
            rec = rec.with_columns(
                ((pl.col("src") == 1) & (pl.col("eid").hash(config.SEED) % config.VAL_FRACTION_MOD == 0)).alias("is_val_s1"))
            gt = read_ground_truth()
            gt.write_parquet(config.split_dir(split) / "gt.parquet")
            print(f"[prep] train gt pairs {gt.height:,}  val S1 {rec['is_val_s1'].sum():,}/{s1.height:,}")
        rec.write_parquet(config.split_dir(split) / "records.parquet")
        n_rep = rec.filter(pl.col("src") > 1).select(pl.col("rep").n_unique()).item()
        print(f"[prep] {split}: {rec.height:,} records, S2/S3 representatives {n_rep:,}, "
              f"countries {rec['country'].unique().sort().to_list()}")


if __name__ == "__main__":
    run()
