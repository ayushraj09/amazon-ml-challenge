"""Reading the challenge TSVs and writing submission TSVs."""
from pathlib import Path
import polars as pl

from . import config


def read_source(split: str, source: int) -> pl.DataFrame:
    """Read one source TSV exactly as provided: tab separated, no quoting (names contain quotes)."""
    path = config.DATA_DIR / split / f"{split}_source{source}.tsv"
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False, encoding="utf8")


def read_ground_truth() -> pl.DataFrame:
    """Train ground truth as long pairs (s1_eid, rec_eid)."""
    path = config.DATA_DIR / "train" / "train_ground_truth.tsv"
    gt = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)
    return (gt.with_columns(pl.col("matched_entity_ids").str.split(","))
              .explode("matched_entity_ids")
              .filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
              .select(pl.col("source1_entity_id").alias("s1_eid"), pl.col("matched_entity_ids").alias("rec_eid")))


def write_id_lists(pairs: pl.DataFrame, s1_ids: pl.Series, path: Path, col: str) -> None:
    """Write one row per S1 (empty list allowed), IDs comma-joined, no duplicates, tab separated.

    `pairs` has columns s1_eid, rec_eid. Every id in `s1_ids` gets a row.
    """
    lists = (pairs.unique(["s1_eid", "rec_eid"]).sort("s1_eid", "rec_eid")
                  .group_by("s1_eid", maintain_order=True).agg(pl.col("rec_eid").str.join(",").alias(col)))
    out = (pl.DataFrame({"source1_entity_id": s1_ids}).unique(maintain_order=True)
             .join(lists.rename({"s1_eid": "source1_entity_id"}), on="source1_entity_id", how="left", maintain_order="left")
             .with_columns(pl.col(col).fill_null("")))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(path, separator="\t", quote_style="never")
