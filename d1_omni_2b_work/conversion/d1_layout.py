"""The graphs' input layouts in numpy alone (the host's LiteRT backends import this, never torch):
the decision graph's two mask inputs (`make_sees`) and the picture tower's canvas (`to_canvas` / `from_canvas`).
d1_graph and d1_tower import these, so the export form and the host share one definition."""
from __future__ import annotations

import numpy as np

NEG = 30000.0          # the additive mask value (exp underflows to 0 in float32; fits float16)
Q_MAX = 64             # question tags 1..64 in one sequence
SEES = 2 + Q_MAX       # sees_q / sees_k width: [is_state, not_state | state_key, one-hot question]


def make_sees(state_len: int, question, L: int, dtype=np.float32) -> tuple[np.ndarray, np.ndarray]:
    """The two mask inputs for one packed sequence of length T = len(question) <= L (`question`: the provider's tags,
    -1 on the state, 1..N on question k), right-padded to L. allowed = sees_q . sees_k^T is exactly 0 / 1: a state
    row sees every state key; a question row sees the state and its own question's keys; padding rows see nothing
    real (their keys are zero) and are never read."""
    T = len(question)
    assert T <= L, (T, L)
    q = np.zeros((1, L, SEES), dtype)
    k = np.zeros((1, L, SEES), dtype)
    tags = np.asarray(question)
    st = np.arange(T) < state_len
    assert np.all((tags == -1) == st), "tags: -1 exactly on the state"
    q[0, :, 1] = 1.0
    q[0, :T, 0] = st
    q[0, :T, 1] = ~st
    k[0, :T, 0] = 1.0
    k[0, :T, 1] = st
    qt = tags[~st]
    assert qt.size == 0 or (qt.min() >= 1 and qt.max() <= Q_MAX), "question tags 1..64"
    idx = np.nonzero(~st)[0]
    q[0, idx, 1 + qt] = 1.0
    k[0, idx, 1 + qt] = 1.0
    return q, k


def to_canvas(pixel_values, gh: int, gw: int, GH: int, GW: int):
    """The processor's pixel_values [gh*gw, 1536] (merge-block order) -> (patches [1, GH*GW, 768], valid
    [1, GH*GW, 1]) on the canvas (raster, top-left), first temporal copy."""
    pv = np.asarray(pixel_values, np.float32)
    assert pv.shape == (gh * gw, 1536) and gh <= GH and gw <= GW
    first = pv.reshape(gh * gw, 3, 2, 256)[:, :, 0, :].reshape(gh * gw, 768)
    rast = first.reshape(gh // 2, gw // 2, 2, 2, 768).transpose(0, 2, 1, 3, 4).reshape(gh, gw, 768)
    patches = np.zeros((1, GH, GW, 768), np.float32)
    patches[0, :gh, :gw] = rast
    valid = np.zeros((1, GH, GW, 1), np.float32)
    valid[0, :gh, :gw] = 1.0
    return patches.reshape(1, GH * GW, 768), valid.reshape(1, GH * GW, 1)


def from_canvas(features, gh: int, gw: int, GH: int, GW: int):
    """[1, GH*GW/4, 2048] block raster of the canvas -> [gh*gw/4, 2048] in the provider's token order."""
    f = np.asarray(features).reshape(GH // 2, GW // 2, -1)
    return f[: gh // 2, : gw // 2].reshape((gh // 2) * (gw // 2), -1)
