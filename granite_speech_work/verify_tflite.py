#!/usr/bin/env python3
"""Full-fixture verification of the granite-speech CTC LiteRT export.

For each of the 20 LibriSpeech dummy clips: pick the smallest signature window
that fits (5s/10s/30s), zero-pad, run each tflite variant, CTC-greedy-collapse
host-side, decode with the repo's tokenizer.json, and compare
  (a) transcript vs the PADDED eager oracle at the same window — conversion
      parity, the number that must be ~exact;
  (b) WER vs the LibriSpeech reference — end-task sanity per variant.

    python3 verify_tflite.py [--variants wi8fc,fp16,fp32]

Writes verify_report.json next to this script.
"""
import argparse
import json
import os
import sys
import time

WORK = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORK)
sys.path.insert(0, os.path.join(WORK, "..", "scripts"))
import _stub  # noqa: F401

import numpy as np
import torch

from eager_gate import load_wav, norm_text, wer
from load_model import load_eager

SIG_SECONDS = (5, 10, 30)
FILES = {
    "fp32": "out/granite_speech_ctc_fp32.tflite",
    "wi8fc": "out/granite_speech_ctc_wi8fc.tflite",
    "fp16": "out/granite_speech_ctc_fp16.tflite",
}


def ctc_collapse(ids):
    out, prev = [], -1
    for i in ids:
        if i != prev and i != 0:
            out.append(int(i))
        prev = i
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="wi8fc,fp16,fp32")
    args = ap.parse_args()
    variants = args.variants.split(",")

    from tokenizers import Tokenizer
    from ai_edge_litert.interpreter import Interpreter

    tok = Tokenizer.from_file(os.path.join(WORK, "hf_model", "tokenizer.json"))
    model, processor = load_eager(dtype=torch.float32)

    runners = {}
    for v in variants:
        it = Interpreter(model_path=os.path.join(WORK, FILES[v]),
                         num_threads=os.cpu_count())
        runners[v] = {s: it.get_signature_runner(f"transcribe_{s}s")
                      for s in SIG_SECONDS}

    meta = json.load(open(os.path.join(WORK, "fixtures", "meta.json")))
    report = {"clips": [], "variants": {v: {"eager_match": 0, "wer_errs": 0}
                                        for v in variants}}
    eager_errs = words = 0
    for m in meta:
        audio = load_wav(os.path.join(WORK, m["file"]))
        sec = next(s for s in SIG_SECONDS if len(audio) <= s * 16000)
        padded = np.zeros(sec * 16000, dtype=np.float32)
        padded[: len(audio)] = audio
        feats = processor([padded])["input_features"]

        with torch.inference_mode():
            e_logits = model(feats)
        e_ids = e_logits.argmax(-1)[0].numpy()
        e_hyp = tok.decode(ctc_collapse(e_ids)).strip()
        r = norm_text(m["text"])
        e_errs = wer(r, norm_text(e_hyp))
        eager_errs += e_errs
        words += len(r)

        row = {"id": m["id"], "sec": round(len(audio) / 16000, 2), "sig": sec,
               "eager_hyp": e_hyp, "eager_wer_errs": e_errs, "ref_words": len(r)}
        for v in variants:
            t0 = time.perf_counter()
            out = runners[v][sec](input_features=feats.numpy())
            dt = time.perf_counter() - t0
            ids = next(x for x in out.values()
                       if x.dtype in (np.int32, np.int64))[0]
            hyp = tok.decode(ctc_collapse(ids)).strip()
            errs = wer(r, norm_text(hyp))
            row[v] = {"hyp": hyp, "match_eager": hyp == e_hyp,
                      "wer_errs": errs, "sec_wall": round(dt, 3)}
            report["variants"][v]["eager_match"] += hyp == e_hyp
            report["variants"][v]["wer_errs"] += errs
        report["clips"].append(row)
        marks = " ".join(f"{v}:{'=' if row[v]['match_eager'] else 'DIFF'}"
                         for v in variants)
        print(f"[{m['id']}] sig={sec}s {marks}")
        for v in variants:
            if not row[v]["match_eager"]:
                print(f"    eager: {e_hyp}\n    {v}   : {row[v]['hyp']}")

    n = len(meta)
    print(f"\neager (padded windows) corpus WER = {eager_errs}/{words} = "
          f"{eager_errs/words:.4f}")
    report["eager"] = {"wer_errs": eager_errs, "ref_words": words}
    for v in variants:
        s = report["variants"][v]
        print(f"{v}: transcript match vs eager {s['eager_match']}/{n}, "
              f"corpus WER {s['wer_errs']}/{words} = {s['wer_errs']/words:.4f}")
    json.dump(report, open(os.path.join(WORK, "verify_report.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
