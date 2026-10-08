"""Central configuration: paths, seeds and every tunable constant (see IMPLEMENTATION_PLAN.md)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("ER_DATA_DIR", ROOT / "student_resource" / "dataset"))
WORK_DIR = Path(os.environ.get("ER_WORK_DIR", ROOT / "work"))
OUT_DIR = Path(os.environ.get("ER_OUT_DIR", ROOT / "output"))

SEED = 42
N_THREADS = int(os.environ.get("ER_THREADS", os.cpu_count() or 8))
SPLITS = ("train", "test")
SOURCES = (1, 2, 3)

# validation split (IMPLEMENTATION_PLAN §6.1): 20% of train S1, hashed so it is stable
VAL_FRACTION_MOD = 5  # hash(S1) % 5 == 0  -> validation

def split_dir(split: str) -> Path:
    d = WORK_DIR / split
    d.mkdir(parents=True, exist_ok=True)
    return d
