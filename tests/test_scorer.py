import polars as pl
from src.evaluate import macro_f05


def P(rows):
    return pl.DataFrame(rows, schema=["s1_eid", "rec_eid"], orient="row")


def test_pdf_example():
    pred = P([("S1-00001", "S2-00047"), ("S1-00001", "S2-00193"), ("S1-00001", "S3-00812")])
    gt = P([("S1-00001", "S2-00047"), ("S1-00001", "S3-00812")])
    score, _ = macro_f05(pred, gt, pl.Series(["S1-00001"]))
    assert abs(score - 0.714) < 1e-3


def test_singletons_and_misses():
    gt = P([("A", "S2-1")])
    ids = pl.Series(["A", "B", "C"])  # B, C are singletons
    pred = P([("A", "S2-1"), ("C", "S3-9")])  # A perfect, B empty (1.0), C false merge (0.0)
    score, per = macro_f05(pred, gt, ids)
    assert abs(score - 2 / 3) < 1e-9
    empty, _ = macro_f05(P([]).cast(pl.Utf8), gt, ids)
    assert abs(empty - 2 / 3) < 1e-9  # A missed (0), B and C correct singletons
