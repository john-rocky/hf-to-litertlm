#!/usr/bin/env python3
"""Transcribe with a Speech2TextMetadata .litertlm through MediaPipe's AudioTranscriber (the pip `mediapipe` 1.2.0,
CPU) and score the text against the clip's human transcript.

    uv run --frozen --project asr python asr/smoke.py <bundle.litertlm> [--clips N] [--wav W --ref TEXT]
        [--bar 0.2] [--parquet FILE | --cache DIR] [--json OUT]

Default clips: hf-internal-testing/librispeech_asr_dummy at a pinned commit (73 LibriSpeech dev-clean utterances,
16 kHz FLAC, CC BY 4.0; a 9.2 MB parquet, sha256-checked). scripts/asr_route.py fetches it through the Hugging Face
cache and passes it as --parquet; run alone, this script downloads it into --cache. --clips 1 (the smoke) takes the
first, 1272-128104-0000 (5.86 s: two of the runtime's 5 s windows); --clips 73 takes all of them (the gate).
--wav / --ref scores one file of your own instead. Every clip goes through a fresh transcriber, one call with
end_of_stream=True. The word error rate is over lowercased words with punctuation dropped and the transcript's
spoken-form abbreviations expanded (mr -> mister, ...), summed over the clips (edits / reference words).
Exit 0 = every clip produced text and the WER is at most --bar, 1 = not, 3 = the run failed. One JSON report.
"""
import argparse
import hashlib
import io
import json
import os
import re
import sys
import time
import urllib.request

DUMMY = ("https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy/resolve/"
         "5be91486e11a2d616f4ec5db8d3fd248585ac07a/clean/validation-00000-of-00001.parquet")
DUMMY_SHA256 = "4e69a06fa5edc90921e5e7e39a7084881f8b3ed9c805c574f4f39c6fde27c603"
SPOKEN = {"mr": "mister", "mrs": "missus", "dr": "doctor", "st": "saint", "&": "and"}


def words(text):
    t = text.lower().replace("-", " ")
    t = re.sub(r"[^\w' ]+", " ", t)
    out = []
    for w in t.split():
        w = w.strip("'")
        if w:
            out.append(SPOKEN.get(w, w))
    return out


def edits(ref, hyp):
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[-1]


def dummy_clips(cache, n, parquet=None):
    import pyarrow.parquet as pq
    import soundfile as sf
    if parquet:
        path = parquet
        if hashlib.sha256(open(path, "rb").read()).hexdigest() != DUMMY_SHA256:
            raise RuntimeError(f"{path} is not the pinned LibriSpeech dummy parquet (sha256 differs)")
    else:
        os.makedirs(cache, exist_ok=True)
        path = os.path.join(cache, "librispeech_asr_dummy_clean_validation.parquet")
    if not (os.path.exists(path) and hashlib.sha256(open(path, "rb").read()).hexdigest() == DUMMY_SHA256):
        with urllib.request.urlopen(DUMMY, timeout=120) as r:
            data = r.read()
        if hashlib.sha256(data).hexdigest() != DUMMY_SHA256:
            raise RuntimeError("the LibriSpeech dummy parquet does not match its pinned sha256")
        with open(path + ".tmp", "wb") as f:
            f.write(data)
        os.replace(path + ".tmp", path)
    rows = pq.read_table(path, columns=["id", "audio", "text"]).to_pylist()[:n]
    for r in rows:
        x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        yield r["id"], x, sr, r["text"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bundle")
    ap.add_argument("--clips", type=int, default=1)
    ap.add_argument("--wav")
    ap.add_argument("--ref")
    ap.add_argument("--bar", type=float, default=0.2)
    ap.add_argument("--parquet", help="a local copy of the pinned clip parquet (sha256-checked)")
    ap.add_argument("--cache", default=os.path.join("out", "_asr_clips"))
    ap.add_argument("--json")
    a = ap.parse_args()

    import numpy as np
    import soundfile as sf
    from mediapipe.tasks.python.audio import audio_transcriber
    from mediapipe.tasks.python.core import base_options
    import importlib.metadata as md

    if a.wav:
        x, sr = sf.read(a.wav, dtype="float32")
        clips = [(os.path.basename(a.wav), x, sr, a.ref or "")]
    else:
        clips = list(dummy_clips(a.cache, a.clips, a.parquet))
    rep = {"bundle": os.path.abspath(a.bundle), "runtime": f"mediapipe {md.version('mediapipe')} AudioTranscriber, CPU",
           "bar": a.bar, "clips": []}
    tot_e = tot_w = 0
    all_text = True
    for cid, x, sr, ref in clips:
        if x.ndim > 1:
            x = x.mean(axis=1)
        opts = audio_transcriber.AudioTranscriberOptions(
            base_options=base_options.BaseOptions(model_asset_path=a.bundle,
                                                  delegate=base_options.BaseOptions.Delegate.CPU),
            sample_rate_hz=int(sr))
        t0 = time.time()
        with audio_transcriber.AudioTranscriber.create_from_options(opts) as tr:
            t1 = time.time()
            res = tr.transcribe(np.ascontiguousarray(x, dtype=np.float32), end_of_stream=True)
            t2 = time.time()
        text = res.text
        row = {"id": cid, "seconds": round(len(x) / sr, 3), "text": text, "reference": ref,
               "create_s": round(t1 - t0, 2), "transcribe_s": round(t2 - t1, 2)}
        if ref:
            rw, hw = words(ref), words(text)
            e = edits(rw, hw)
            row["edits"], row["ref_words"] = e, len(rw)
            tot_e, tot_w = tot_e + e, tot_w + len(rw)
        all_text = all_text and bool(text.strip())
        rep["clips"].append(row)
        print(f"{cid} ({row['seconds']} s): {text!r}" + (f"  [{row.get('edits')}/{row.get('ref_words')} edits]"
                                                         if ref else ""), flush=True)
    rep["wer"] = round(tot_e / tot_w, 4) if tot_w else None
    rep["pass"] = bool(all_text and (rep["wer"] is None or rep["wer"] <= a.bar))
    print(f"{'PASS' if rep['pass'] else 'FAIL'}: {len(clips)} clip(s), WER {rep['wer']} (bar {a.bar})", flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rep, f, indent=1, ensure_ascii=False)
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001 — one FAIL line, like scripts/convert.py
        print(f"FAIL smoke: {type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}", file=sys.stderr)
        sys.exit(3)
