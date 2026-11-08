#!/usr/bin/env python3
"""The reference's window tokens on S4 (the 655 LibriSpeech test-clean utterances of make_subset.py --every 4), which p7_s4new.py reads for the
token check of the 7.5, 15 and 25 s graphs. Venv .venv-ref, CPU fp32, cwd = phonon2_work: Phonon-2's scripts, the checkpoint folders named by
PHONON_HF_DIR / PHONON_UNPACK_DIR (scripts/p1_ref.py reads them). Run it with `python -I`.
An utterance of up to 479,840 samples (30 s) is one window, run whole (the processor on the audio alone); a longer one is cut into 479,840-sample
windows from sample 0, each padded with zeros to 30 s and given the valid-length mask, a window under 320 samples dropped (the runner's rule).
A window's tokens = transformers' generate() output without the blanks (scripts/p1_ref.py: features, gen, steps_from_generate).
-> results/r1/ref_pii_s4.jsonl (one line per utterance in the manifest's order: id, win_tokens), written whole at the end.
CLI: pii_s4.py [--threads 6]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "phonon2_work", "scripts"))
import p1_ref as R  # noqa: E402

W30 = 479840
MIN_SAMPLES = 320
MAN = "data/sets/librispeech-test-clean/manifest-S4.jsonl"
OUT = "results/r1/ref_pii_s4.jsonl"


def window_tokens(model, proc, audio):
    """[[token, ...] per window]"""
    if len(audio) <= W30:
        parts = [(audio, False)]
    else:
        parts = [(audio[s0:s0 + W30], True) for s0 in range(0, len(audio), W30) if len(audio[s0:s0 + W30]) >= MIN_SAMPLES]
    out = []
    for w, padded in parts:
        feats, mask = R.features(proc, w, pad_to=W30, mask_padding=True) if padded else R.features(proc, w)
        seq, dur = R.gen(model, feats, mask)
        out.append([tk for (_, tk, _) in R.steps_from_generate(seq, dur) if tk != R.BLANK])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    rows = R.read_jsonl(MAN)
    R.set_threads(a.threads)
    t0 = time.time()
    model, rep = R.build("P")
    proc = R.processor()
    lines, n_win, n_long = [], 0, 0
    for r in rows:
        audio = R.load_wav(r["path"])
        toks = window_tokens(model, proc, audio)
        n_win += len(toks)
        n_long += len(audio) > W30
        lines.append(json.dumps({"id": r["id"], "win_tokens": toks}) + "\n")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT + ".part", "w") as f:
        f.writelines(lines)
    os.replace(OUT + ".part", OUT)
    print(json.dumps({"utterances": len(rows), "windows": n_win, "over_30s": n_long, "container": os.path.relpath(rep["container"]),
                      "missing": rep["missing"], "unexpected": rep["unexpected"], "threads": a.threads,
                      "seconds": round(time.time() - t0, 1), "out": OUT}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
