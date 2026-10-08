import math
import torch
from src.cross_encoder import ova_loss, rtm


def test_ova_loss_matches_op_match_definition():
    # 3 candidates, mate at index 0; logits [neg, pos]
    lo = torch.tensor([[0.0, 2.0], [1.0, 0.0], [0.0, 1.0]])
    y = torch.tensor([True, False, False])
    p = torch.softmax(lo, 1)
    expected = -math.log(p[0, 1]) + max(-math.log(p[1, 0]), -math.log(p[2, 0]))  # hardest negative
    assert abs(ova_loss(lo, y, True).item() - expected) < 1e-5


def test_ova_loss_nil_record_drops_positive_term():
    lo = torch.tensor([[0.0, 1.0], [2.0, 0.0]])
    y = torch.tensor([False, False])
    p = torch.softmax(lo, 1)
    expected = max(-math.log(p[0, 0]), -math.log(p[1, 0]))
    assert abs(ova_loss(lo, y, False).item() - expected) < 1e-5


def test_rtm_is_soft_max_of_nonmate_scores():
    lo = torch.tensor([[0.0, 5.0], [0.0, -5.0], [0.0, 0.0]])
    y = torch.tensor([True, False, False])
    nm = torch.softmax(lo, 1)[1:, 1]
    expected = (torch.softmax(nm, 0) * nm).sum().item()
    assert abs(rtm(lo, y).item() - expected) < 1e-6 and expected <= nm.max().item()
