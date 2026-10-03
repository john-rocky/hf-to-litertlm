#!/usr/bin/env python3
"""G2 mel check (copied from confucius4_r2t2_work/r3_mel_check.py): the log-mel the bundle computes (inside its own fp16 audio_encoder_hw section, read as an
intermediate tensor with ai_edge_litert's experimental_preserve_all_tensors) vs Qwen3ASRFeatureExtractor on the same
samples (the wav zero-padded to the window, = the round-2 direct driver's mel), on N clips.

The encoder input is built the way the runtime builds it (GENERIC_CONTRACT.md §2-3): PCM16 / 32768, frames of 160
samples, zero-filled after the clip up to the window. The mel tensor is identified as the [1, 128, T] float tensor
closest to the HF mel (its name is recorded); its consumer chain is the conv stack.

  ~/venvs/ltmain0918/bin/python mel_check.py --encoder out/sections/r3_q17_30s_C_off/enc__fp16.tflite --n 10 --out out/mel_check.json
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, help="the bundle's audio_encoder_hw .tflite")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import transformers
    from ai_edge_litert.interpreter import Interpreter
    proc = transformers.AutoProcessor.from_pretrained(common.MODEL_DIR)
    n = args.window * 16000
    T = n // 160
    it = Interpreter(model_path=args.encoder, num_threads=8, experimental_preserve_all_tensors=True)
    it.allocate_tensors()
    sig = it.get_signature_runner("encode")
    cands = [d for d in it.get_tensor_details() if list(d["shape"]) == [1, 128, T] and d["dtype"] == np.float32]
    clips = []
    for cfg in ("en_us", "cmn_hans_cn", "ja_jp"):
        clips += [c for c in common.load_clips() if c["config"] == cfg][:4]
    clips = clips[:args.n]
    rows = []
    for c in clips:
        wav = common.read_wav(c["path"])
        padded = np.concatenate([wav, np.zeros(n - len(wav), np.float32)])
        hf = proc(text=common.LITERT_TORCH_PROMPT, audio=padded, return_tensors="pt")["input_features"].numpy()
        sig(audio=padded.reshape(1, T, 160))
        best = None
        for d in cands:
            try:
                v = it.get_tensor(d["index"])
            except ValueError:
                continue
            err = float(np.abs(v - hf).max())
            if best is None or err < best[0]:
                best = (err, d["name"], v)
        err, name, v = best
        diff = np.abs(v - hf)
        nf = len(wav) // 160  # frames fully inside the clip
        rows.append({"clip": c["name"], "audio_seconds": round(len(wav) / 16000, 3), "tensor": name,
                     "max_abs": float(diff.max()), "mean_abs": float(diff.mean()),
                     "max_abs_speech_frames": float(diff[..., :nf].max()),
                     "hf_range": [float(hf.min()), float(hf.max())]})
        print(json.dumps(rows[-1]), flush=True)
    doc = {"encoder": os.path.realpath(args.encoder), "window_s": args.window, "candidates": len(cands),
           "rows": rows, "max_abs": max(r["max_abs"] for r in rows),
           "mean_abs_mean": float(np.mean([r["mean_abs"] for r in rows]))}
    json.dump(doc, open(args.out, "w"), indent=1)
    print("MEL max|d|", doc["max_abs"], "mean|d|", doc["mean_abs_mean"])


if __name__ == "__main__":
    main()
