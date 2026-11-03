"""r4 host rules of the fused pre-encode + conformer graph (numpy only; shared by graphs_r4.py (torch, the export and
its checks) and stream_lt4.py (the LiteRT host loop in .venv-q, no torch)). The why is in graphs_r4.py's docstring.

Chunk 0 through the fused (steady) graph: mel = 7 zero rows + mel_first's 25 rows, the three pre-encode caches = 0,
pre_m0 = [0, 0, 0, 1 x 13] (conv_in's output rows), pre_m1 = [0, 1 x 7] (stage 1's output rows).
Chunk >= 1: mel = the mel graph's [1, 32, 128], the caches carried from the previous chunk's cache*_out, masks = 1.
"""
import numpy as np

from contract import MEL, T_FIRST, T_NEXT

T_M0 = 16       # steady conv_in output rows ([cache0, mel] = 33 rows, kernel 3, stride 2)
T_M1 = 8        # steady stage-1 output rows ([cache1, 16 rows] = 17 rows)
PAD_ROWS = T_NEXT - T_FIRST      # 7 zero mel rows in front of mel_first's 25 rows
M0_ZERO = 3     # conv_in rows in front of PreFirst's conv_in row 0
M1_ZERO = 1     # stage-1 rows in front of PreFirst's stage-1 row 0


def masks(chunk_idx):
    """(pre_m0 [1,1,16,1], pre_m1 [1,1,8,1]) float32 for a chunk index."""
    m0 = np.ones((1, 1, T_M0, 1), np.float32)
    m1 = np.ones((1, 1, T_M1, 1), np.float32)
    if chunk_idx == 0:
        m0[:, :, :M0_ZERO] = 0.0
        m1[:, :, :M1_ZERO] = 0.0
    return m0, m1


def first_mel(mel_first):
    """mel_first's output [1, 25, 128] -> the fused graph's chunk-0 mel [1, 32, 128]: 7 zero rows in front."""
    m = np.asarray(mel_first, np.float32).reshape(1, T_FIRST, MEL)
    return np.concatenate([np.zeros((1, PAD_ROWS, MEL), np.float32), m], axis=1)
