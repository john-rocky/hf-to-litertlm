"""Eager oracle gate for granite-speech-5.0-470m-turboctc.

Loads the HF checkpoint (trust_remote_code, fp32, CPU), transcribes the 20
LibriSpeech dummy fixtures, and reports per-clip transcripts + corpus WER
against the reference texts. Also runs the two kickoff checks from
the pre-conversion license/architecture check:

  1. attention_dists persistence (tf5x-meta-load-zeroes-init-buffers):
     assert the loaded buffer is non-trivial and equals a freshly-initialized
     encoder's buffer (catches meta-load zeroing).
  2. input_linear.in_features == 320 (n_mels 80 x 2 (deltas) x stack 2).

Usage: python3 eager_gate.py [--pad-seconds N]
  --pad-seconds N additionally right-pads every clip with zeros to N seconds
  and re-transcribes (mask=None), reporting transcript agreement with the
  unpadded run — the semantic gate for the fixed-shape export.
"""

import argparse
import json
import re
import sys
import wave
from pathlib import Path

import numpy as np
import torch

WORK = Path(__file__).parent
MODEL_DIR = WORK / "hf_model"


def load_wav(path):
    # Minimal RIFF walk: afconvert emits WAVE_FORMAT_EXTENSIBLE (tag 65534),
    # which the stdlib wave module refuses; the data chunk is plain PCM16LE.
    raw = Path(path).read_bytes()
    assert raw[:4] == b"RIFF" and raw[8:12] == b"WAVE", path
    pos, fmt, data = 12, None, None
    while pos + 8 <= len(raw):
        cid = raw[pos:pos + 4]
        size = int.from_bytes(raw[pos + 4:pos + 8], "little")
        body = raw[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt = body
        elif cid == b"data":
            data = body
        pos += 8 + size + (size & 1)
    channels = int.from_bytes(fmt[2:4], "little")
    rate = int.from_bytes(fmt[4:8], "little")
    bits = int.from_bytes(fmt[14:16], "little")
    assert (channels, rate, bits) == (1, 16000, 16), (path, channels, rate, bits)
    x = np.frombuffer(data, dtype=np.int16)
    return x.astype(np.float32) / 32768.0


def norm_text(s):
    return re.sub(r"[^A-Z' ]", "", s.upper().replace("-", " ")).split()


def wer(ref, hyp):
    d = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(ref) + 1)
    d[0, :] = np.arange(len(hyp) + 1)
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    return int(d[-1, -1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pad-seconds", type=float, default=None)
    args = ap.parse_args()

    from load_model import load_eager

    torch.manual_seed(0)
    model, processor = load_eager(dtype=torch.float32)

    # --- kickoff check 2: input feature dim
    in_f = model.encoder.input_linear.in_features
    print(f"input_linear.in_features = {in_f}")
    assert in_f == 320, in_f

    # --- kickoff check 1: attention_dists buffer integrity
    ad = model.encoder.attention_dists
    print(f"attention_dists: shape {tuple(ad.shape)} dtype {ad.dtype} "
          f"min {ad.min().item()} max {ad.max().item()} "
          f"persistent={'attention_dists' in dict(model.encoder.named_buffers())}")
    assert ad.abs().sum().item() > 0, "attention_dists is all-zero (meta-load trap)"
    from granite_encoder_check import fresh_attention_dists
    ref_ad = fresh_attention_dists(model.config)
    assert torch.equal(ad.cpu(), ref_ad), "loaded attention_dists != freshly built"
    print("attention_dists matches fresh init: OK")

    meta = json.load(open(WORK / "fixtures" / "meta.json"))
    results, errs, words = [], 0, 0
    for m in meta:
        audio = load_wav(WORK / m["file"])
        feats = processor([audio])
        out = model.transcribe(feats["input_features"])
        hyp = processor.batch_decode(out.preds)[0]
        row = {"id": m["id"], "sec": round(len(audio) / 16000, 2),
               "ref": m["text"], "hyp": hyp}
        r, h = norm_text(m["text"]), norm_text(hyp)
        row["wer_errs"], row["ref_words"] = wer(r, h), len(r)
        errs += row["wer_errs"]
        words += len(r)

        if args.pad_seconds:
            n = int(args.pad_seconds * 16000)
            if len(audio) > n:
                row["pad"] = "SKIP (clip longer than pad window)"
            else:
                padded = np.zeros(n, dtype=np.float32)
                padded[: len(audio)] = audio
                pf = processor([padded])
                pout = model.transcribe(pf["input_features"])
                phyp = processor.batch_decode(pout.preds)[0]
                row["pad_hyp"] = phyp
                row["pad"] = "MATCH" if phyp == hyp else "DIFF"
        results.append(row)
        print(f"[{m['id']}] {row['sec']}s wer_errs={row['wer_errs']}/{row['ref_words']}"
              + (f" pad={row['pad']}" if 'pad' in row else "")
              + f"\n  hyp: {hyp}")

    print(f"\ncorpus WER = {errs}/{words} = {errs / words:.4f}")
    if args.pad_seconds:
        n_match = sum(1 for r in results if r.get("pad") == "MATCH")
        n_run = sum(1 for r in results if r.get("pad") in ("MATCH", "DIFF"))
        print(f"pad-{args.pad_seconds}s transcript agreement: {n_match}/{n_run}")
    tag = f"_pad{int(args.pad_seconds)}" if args.pad_seconds else ""
    json.dump({"wer_errs": errs, "ref_words": words, "rows": results},
              open(WORK / f"eager_gate{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    sys.path.insert(0, str(WORK))
    main()
