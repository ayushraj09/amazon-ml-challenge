import numpy as np
from src.synth import _shift_number, _valid_deltas


def test_shift_whole_number_only():
    assert _shift_number("1142 Oak St, 142 Elm", 142, 150) == ("1142 Oak St, 150 Elm", True)
    assert _shift_number("91 Main Rd", 91, 95) == ("95 Main Rd", True)
    assert _shift_number("No number here", 12, 13) == ("No number here", False)
    assert _shift_number("12 A St", 12, 12)[1] is False


def test_valid_deltas_keep_numbers_positive():
    rng = np.random.default_rng(0)
    dl = np.array([-5000, -30, -3, 2, 7, 40])
    old = np.array([2, 20, 38, 91, 5000] * 100)
    d = _valid_deltas(dl, old, rng)
    assert ((old + d) > 0).all() and (d != 0).all()
