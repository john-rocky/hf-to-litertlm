#!/usr/bin/env python3
"""Window / mask experiments with reference B on LibriSpeech test-clean S32 (82 utterances), CPU fp32, batch 1.

(i)    the audio alone (the processor's own call, its attention mask) -> generate()
(ii)   zeros to 30 s (479,840 samples = 3000 mel frames, as the 30 s export) + the valid-length attention mask
(iii)  zeros to 30 s treated as audio (stats over the whole window), no mask, generate() walks all 375 frames
       = the shape of litert-samples' 30 s file; (iii-s) = the same encoder output, decode stopped at
       min(375, ceil(n / 8) + 4) with n = floor(samples / 160) (the 09-25 patch-06 / ASR-lane L4f rule)
(iv)   standalone 5 s windows (79,840 samples = 500 mel frames, as the 5 s export), consecutive, no overlap, the
       last one zero-padded, each as (iii) (no mask, all 63 frames), window texts joined with a space;
       (iv-s) = each window stopped as (iii-s)
An utterance longer than 30 s is cut into consecutive 30 s windows in (ii) / (iii) / (iii-s) (texts joined).
Writes results/s32_modes.jsonl (one line per utterance and mode, resumable), results/ref_b_s32.jsonl (mode (i):
token ids, emission frames, durations, every decode step) and out/s32_ii/<id>[__w<k>].npz per 30 s window of
mode (ii) (mel [128, 3000] as the export takes it, valid mel frames, B's encoder_projector output [375, 640],
valid encoder frames, the window's token ids; for the encoder smoke).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import redux_ref as R  # noqa: E402

W30 = 479840
W5 = 79840
OUT = "results/s32_modes.jsonl"
REF = "results/ref_b_s32.jsonl"
NPZ_DIR = "out/s32_ii"
MODES = ["i", "ii", "iii", "iii-s", "iv", "iv-s"]


def windows(audio, n):
    return [audio[k:k + n] for k in range(0, len(audio), n)] or [audio]


def stop_frames(n_samples, T):
    return min(T, math.ceil((n_samples // 160) / 8) + 4)


def run_utt(model, proc, r, have):
    audio = R.load_wav(r["path"])
    res = {}
    t0 = time.time()
    if "i" not in have:
        feats, mask = R.features(proc, audio)
        seq, dur = R.gen(model, feats, mask)
        steps = R.steps_from_generate(seq, dur)
        toks = [tk for (_, tk, _) in steps if tk != R.BLANK]
        res["i"] = {"text": R.decode_text(proc, toks), "tokens": toks}
        with open(REF, "a") as f:
            f.write(json.dumps({"id": r["id"], "text": res["i"]["text"], "tokens": toks,
                                "frames": [t for (t, tk, _) in steps if tk != R.BLANK],
                                "durs": [d for (_, tk, d) in steps if tk != R.BLANK],
                                "steps": steps, "mel_frames": int(feats.shape[1]), "mask_frames": int(mask.sum()),
                                "audio_s": len(audio) / R.SR}) + "\n")
    if "ii" not in have:
        texts, toks_all = [], []
        ws = windows(audio, W30)
        for wi, w in enumerate(ws):
            feats, mask = R.features(proc, w, pad_to=W30, mask_padding=True)
            seq, dur = R.gen(model, feats, mask)
            toks = [tk for tk in seq[1:] if tk != R.BLANK]
            texts.append(R.decode_text(proc, toks))
            toks_all += toks
            enc, _, valid = R.encode(model, feats, mask)
            os.makedirs(NPZ_DIR, exist_ok=True)
            stem = r["id"] if len(ws) == 1 else f"{r['id']}__w{wi}"
            np.savez(os.path.join(NPZ_DIR, stem + ".npz"),
                     mel=feats[0].transpose(0, 1).contiguous().numpy().astype(np.float32),
                     valid_mel=np.int32(int(mask.sum())), enc_proj=enc.numpy().astype(np.float32),
                     valid_enc=np.int32(valid), tokens=np.array(toks, np.int32))
        res["ii"] = {"text": " ".join(t for t in texts if t), "tokens": toks_all, "windows": len(texts)}
    if "iii" not in have or "iii-s" not in have:
        tx, tx_s, tk3, tk3s, nw = [], [], [], [], 0
        for w in windows(audio, W30):
            nw += 1
            feats, _ = R.features(proc, w, pad_to=W30, mask_padding=False)
            seq, dur = R.gen(model, feats, None)
            toks = [tk for tk in seq[1:] if tk != R.BLANK]
            tx.append(R.decode_text(proc, toks))
            tk3 += toks
            enc, _, T = R.encode(model, feats, None)
            lr = R.tdt_loop(model, enc, stop_frames(len(w), T))
            tx_s.append(R.decode_text(proc, lr["tokens"]))
            tk3s += lr["tokens"]
        res["iii"] = {"text": " ".join(t for t in tx if t), "tokens": tk3, "windows": nw}
        res["iii-s"] = {"text": " ".join(t for t in tx_s if t), "tokens": tk3s, "windows": nw}
    if "iv" not in have or "iv-s" not in have:
        tx, tx_s, tk4, tk4s, empty, nw = [], [], [], [], 0, 0
        for w in windows(audio, W5):
            nw += 1
            feats, _ = R.features(proc, w, pad_to=W5, mask_padding=False)
            seq, dur = R.gen(model, feats, None)
            toks = [tk for tk in seq[1:] if tk != R.BLANK]
            t = R.decode_text(proc, toks)
            empty += (t.strip() == "")
            tx.append(t)
            tk4 += toks
            enc, _, T = R.encode(model, feats, None)
            lr = R.tdt_loop(model, enc, stop_frames(len(w), T))
            tx_s.append(R.decode_text(proc, lr["tokens"]))
            tk4s += lr["tokens"]
        res["iv"] = {"text": " ".join(t for t in tx if t), "tokens": tk4, "windows": nw, "empty_windows": empty}
        res["iv-s"] = {"text": " ".join(t for t in tx_s if t), "tokens": tk4s, "windows": nw,
                       "empty_windows": sum(1 for t in tx_s if not t.strip())}
    return res, time.time() - t0, len(audio) / R.SR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deadline", type=float, default=0.0)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    rows = R.read_jsonl("data/sets/librispeech-test-clean/manifest-S32.jsonl")
    have = {}
    for x in R.read_jsonl(OUT):
        have.setdefault(x["id"], set()).add(x["mode"])
    todo =[r for r in rows if have.get(r["id"], set()) != set(MODES)]
    if not todo:
        open(OUT + ".DONE", "w").write(time.strftime("%H:%M:%S") + "\n")
        print("s32 modes: nothing to do")
        return 0
    if a.deadline and time.time() > a.deadline - 90:
        return 0
    R.set_threads(a.threads)
    model, rep = R.build("B")
    proc = R.processor()
    with open(OUT + ".meta.json", "a") as f:
        f.write(json.dumps({"load_report": rep, "env": R.env_record()}, default=str) + "\n")
    per = 0.0
    n = 0
    for r in todo:
        if a.deadline and time.time() + max(3 * per, 10) > a.deadline:
            break
        h = set(have.get(r["id"], set()))
        res, dt, sec = run_utt(model, proc, r, h)
        per = dt if per == 0 else 0.8 * per + 0.2 * dt
        with open(OUT, "a") as f:
            for m, v in res.items():
                f.write(json.dumps({"id": r["id"], "mode": m, "audio_s": sec, **v}, ensure_ascii=False) + "\n")
        n += 1
        print(f"{r['id']} {sec:.1f}s {dt:.1f}s", flush=True)
    left = len(todo) - n
    print(f"s32 modes: +{n}, {left} left", flush=True)
    if left == 0:
        open(OUT + ".DONE", "w").write(time.strftime("%H:%M:%S") + "\n")
    return 0


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    sys.exit(main())
