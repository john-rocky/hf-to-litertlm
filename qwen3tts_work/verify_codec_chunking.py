"""Validate the fixed left-context codec chunking against single-shot decode.

The codec decoder graph is fixed-T (T=64). Utterances longer than T are
decoded in chunks of (T - ctx) new frames with `ctx` frames of discarded left
context. This checks that stitching is seamless: decode a >T code sequence
(a) single-shot with a T=128 graph and (b) chunked with the T=64 graph, and
compare. corr ~1.0 => ctx frames of left context fully cover the receptive
field, so the host-loop chunking is numerically exact.

  .venv/bin/python qwen3tts_work/verify_codec_chunking.py
"""
import numpy as np
from ai_edge_litert.interpreter import Interpreter

CH, CTX, UP = 64, 25, 1920

# A >64-frame code stream: tile the 52-frame reference twice -> 104 frames.
# (Realism is irrelevant; we only test boundary stitching invariance.)
ref = np.load("qwen3tts_work/ref/e2e_ref.npz")
codes = np.concatenate([ref["codes"], ref["codes"]], 0).astype(np.int32)  # [104,16]
T = codes.shape[0]
assert T > CH, T

it64 = Interpreter(model_path="out/qwen3tts-codec/codec_decode_T64.tflite", num_threads=4)
c64 = it64.get_signature_runner()
it128 = Interpreter(model_path="out/qwen3tts-codec/codec_decode_T128.tflite", num_threads=4)
c128 = it128.get_signature_runner()


def single_shot(codes):
    buf = np.zeros((1, 16, 128), np.int32)
    buf[0, :, :codes.shape[0]] = codes.T
    out = c128(args_0=buf)
    return list(out.values())[0][0, 0][: codes.shape[0] * UP]


def chunked(codes):
    T = codes.shape[0]
    pieces = []
    i = 0
    while i < T:
        ctx = min(CTX, i)
        j = min(i + CH - ctx, T)
        chunk = codes[i - ctx: j]
        buf = np.zeros((1, 16, CH), np.int32)
        buf[0, :, : len(chunk)] = chunk.T
        out = c64(args_0=buf)
        wav = list(out.values())[0][0, 0]
        pieces.append(wav[ctx * UP: len(chunk) * UP])
        i = j
    return np.concatenate(pieces)


gt = single_shot(codes)
ck = chunked(codes)
n = min(len(gt), len(ck))
c = float(np.corrcoef(gt[:n], ck[:n])[0, 1])
md = float(np.abs(gt[:n] - ck[:n]).max())
print(f"frames={T} single-shot={len(gt)} chunked={len(ck)} "
      f"corr={c:.8f} max|d|={md:.3e}")
print("CHUNKING_OK" if c > 0.9999 else "CHUNKING_SEAM_ERROR")
