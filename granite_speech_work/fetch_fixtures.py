#!/usr/bin/env python3
"""Fetch the 20 LibriSpeech dev-clean fixture clips used by the granite-speech
CTC gates (hf-internal-testing/librispeech_asr_dummy via the datasets-server
rows API), convert to 16 kHz mono wav, and write fixtures/meta.json.

Needs ffmpeg on PATH (or afconvert on macOS) for the flac -> wav step.

    python3 fetch_fixtures.py
"""
import json
import os
import shutil
import subprocess
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "fixtures")
API = ("https://datasets-server.huggingface.co/rows?dataset="
       "hf-internal-testing%2Flibrispeech_asr_dummy&config=clean"
       "&split=validation&offset=0&length=20")


def to_wav(flac, wav):
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", flac,
                        "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", wav],
                       check=True)
    else:  # macOS
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
                        flac, wav], check=True)


def main():
    os.makedirs(FIX, exist_ok=True)
    rows = json.load(urllib.request.urlopen(API))["rows"]
    meta = []
    for i, rr in enumerate(rows):
        r = rr["row"]
        a = r["audio"]
        src = a["src"] if isinstance(a, dict) else a[0]["src"]
        flac = os.path.join(FIX, f"clip{i:02d}.flac")
        wav = os.path.join(FIX, f"clip{i:02d}.wav")
        urllib.request.urlretrieve(src, flac)
        to_wav(flac, wav)
        meta.append({"file": f"fixtures/clip{i:02d}.wav", "text": r["text"],
                     "id": r["id"]})
    json.dump(meta, open(os.path.join(FIX, "meta.json"), "w"), indent=1)
    print(f"fetched {len(meta)} clips into {FIX}")


if __name__ == "__main__":
    main()
