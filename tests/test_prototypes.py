"""Prototype k-means++ init and pre-injection bank (CPU)."""

from __future__ import annotations

import torch

from src.models.prototypes import PrototypeModule


def test_init_from_bank_nn_dist_zero_and_distinct() -> None:
    torch.manual_seed(0)
    n_proto, d, n_per = 4, 8, 20
    centers = torch.eye(n_proto, d) * 10.0
    chunks = [centers[i] + 0.01 * torch.randn(n_per, d) for i in range(n_proto)]
    bank = torch.cat(chunks, dim=0)
    proto = PrototypeModule(n_proto, d, d_min=0.5)
    proto.init_from_bank(bank, random_state=0)
    nn_mean = float(proto.nn_dist_mean(bank))
    idx, _dist = proto.project(bank)
    n_distinct = len(set(int(i) for i in idx.tolist()))
    assert nn_mean < 1e-5
    assert n_distinct == n_proto
