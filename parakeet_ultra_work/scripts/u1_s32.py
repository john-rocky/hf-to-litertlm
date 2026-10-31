#!/usr/bin/env python3
"""Reference U on LibriSpeech test-clean S32 (82 utterances), CPU fp32, batch 1 (venv .venv-ref): modes i and ii of
redux r1 (parakeet_redux_work/scripts/s32_modes.py, the two modes the lane's lines use).

(i)   the audio alone (the processor's own call, its attention mask) -> generate()
(ii)  30 s windows: consecutive 479,840-sample windows from sample 0 (a window under 320 samples dropped), zeros to
      479,840 samples + the valid-length mask (the extractor told the true length) -> generate() per window
Per mode-ii window also: the encoder_projector output, and the greedy TDT loop on it (ultra_ref.tdt_loop, no cap =
generate's rule) with every step's top-2 logit gap of the token head and the duration head; the loop's tokens must
equal generate's (checked, recorded).
-> results/r1/s32_modes.jsonl (id, mode, text, tokens per utterance; resumable), results/r1/ref_u_s32.jsonl (mode i:
   tokens, frames, durations, steps), out/s32_ii/<id>[__w<k>].npz per mode-ii window (mel [128, 3000] as the export
   takes it, valid mel frames, enc_proj [375, 640], valid encoder frames, tokens / frames / durs of generate, the loop's
   steps [n, 3] and gaps [n, 2]), results/r1/s32_modes.meta.json.
CLI: u1_s32.py [--threads 8]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ultra_ref as R  # noqa: E402

W30 = 479840
MIN_SAMPLES = 320
OUT = "results/r1/s32_modes.jsonl"
REF = "results/r1/ref_u_s32.jsonl"
NPZ_DIR = "out/s32_ii"
MODES = ["i", "ii"]


def windows(audio):
    out = []
    for s in range(0, max(len(audio), 1), W30):
        w = audio[s:s + W30]
        if len(w) >= MIN_SAMPLES:
            out.append(w)
    return out


def run_utt(model, proc, r, have):
    audio = R.load_wav(r["path"])
    res = {}
    t0 = time.time()
    if "i" not in have:
        feats, mask = R.features(proc, audio)
        seq, dur = R.gen(model, feats, mask)
        steps = R.steps_from_generate(seq, dur)
        toks = [tk for (_, tk, _) in steps if tk != R.BLANK]
        res["i"] = {"text": R.tdt_text(toks), "tokens": toks}
        with open(REF, "a") as f:
            f.write(json.dumps({"id": r["id"], "text": res["i"]["text"], "tokens": toks,
                                "frames": [t for (t, tk, _) in steps if tk != R.BLANK],
                                "durs": [d for (_, tk, d) in steps if tk != R.BLANK],
                                "steps": steps, "mel_frames": int(feats.shape[1]), "mask_frames": int(mask.sum()),
                                "audio_s": len(audio) / R.SR}) + "\n")
    if "ii" not in have:
        texts, toks_all, loop_eq = [], [], []
        ws = windows(audio)
        for wi, w in enumerate(ws):
            feats, mask = R.features(proc, w, pad_to=W30, mask_padding=True)
            seq, dur = R.gen(model, feats, mask)
            steps = R.steps_from_generate(seq, dur)
            toks = [tk for (_, tk, _) in steps if tk != R.BLANK]
            texts.append(R.tdt_text(toks))
            toks_all += toks
            enc, _, valid = R.encode(model, feats, mask)
            lp = R.tdt_loop(model, enc, valid, gaps=True)
            loop_eq.append(lp["tokens"] == toks and [tuple(s) for s in lp["steps"]] == [tuple(s) for s in steps])
            os.makedirs(NPZ_DIR, exist_ok=True)
            stem = r["id"] if len(ws) == 1 else f"{r['id']}__w{wi}"
            np.savez(os.path.join(NPZ_DIR, stem + ".npz"),
                     mel=feats[0].transpose(0, 1).contiguous().numpy().astype(np.float32),
                     valid_mel=np.int32(int(mask.sum())), enc_proj=enc.numpy().astype(np.float32),
                     valid_enc=np.int32(valid), tokens=np.array(toks, np.int32),
                     frames=np.array([t for (t, tk, _) in steps if tk != R.BLANK], np.int32),
                     durs=np.array([d for (_, tk, d) in steps if tk != R.BLANK], np.int32),
                     steps=np.array(lp["steps"], np.int32).reshape(-1, 3),
                     gaps=np.array(lp["gaps"], np.float32).reshape(-1, 2))
        res["ii"] = {"text": " ".join(t for t in texts if t), "tokens": toks_all, "windows": len(texts),
                     "loop_eq_generate": loop_eq}
    return res, time.time() - t0, len(audio) / R.SR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    rows = R.read_jsonl("data/sets/librispeech-test-clean/manifest-S32.jsonl")
    have = {}
    for x in R.read_jsonl(OUT):
        have.setdefault(x["id"], set()).add(x["mode"])
    todo = [r for r in rows if have.get(r["id"], set()) != set(MODES)]
    if not todo:
        open(OUT + ".DONE", "w").write(time.strftime("%H:%M:%S") + "\n")
        print("s32 modes: nothing to do")
        return 0
    os.makedirs("results/r1", exist_ok=True)
    R.set_threads(a.threads)
    t0 = time.time()
    model, rep = R.build("U")
    proc = R.processor()
    with open(OUT.replace(".jsonl", ".meta.json"), "a") as f:
        f.write(json.dumps({"load_report": rep, "env": R.env_record(), "load_s": round(time.time() - t0, 1)},
                           default=str) + "\n")
    n = 0
    for r in todo:
        h = set(have.get(r["id"], set()))
        res, dt, sec = run_utt(model, proc, r, h)
        with open(OUT, "a") as f:
            for m, v in res.items():
                f.write(json.dumps({"id": r["id"], "mode": m, "audio_s": sec, **v}, ensure_ascii=False) + "\n")
        n += 1
        print(f"{r['id']} {sec:.1f}s {dt:.1f}s", flush=True)
    print(f"s32 modes: +{n}", flush=True)
    open(OUT + ".DONE", "w").write(time.strftime("%H:%M:%S") + "\n")
    return 0


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    sys.exit(main())
