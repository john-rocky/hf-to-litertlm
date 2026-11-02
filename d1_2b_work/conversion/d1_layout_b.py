"""The batch graph's mask inputs in numpy alone (the host imports this, never torch): `make_sees_batch` stacks
d1_layout.make_sees per real row; rows past the real ones are
padding rows (sees_q [0, 1, 0..], sees_k 0: they see nothing real and are never read)."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d1_layout import SEES, make_sees  # noqa: E402


def make_sees_batch(rows, B: int, L: int):
    """rows = [(state_len, question tags)] (len <= B) -> sees_q, sees_k float32 [B, L, 66]."""
    assert len(rows) <= B, (len(rows), B)
    q = np.zeros((B, L, SEES), np.float32)
    k = np.zeros((B, L, SEES), np.float32)
    q[:, :, 1] = 1.0
    for b, (state_len, tags) in enumerate(rows):
        qb, kb = make_sees(state_len, tags, L)
        q[b], k[b] = qb[0], kb[0]
    return q, k
