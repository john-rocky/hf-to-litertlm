#!/usr/bin/env python3
"""out/fixtures/long_en.wav = en_clip04 + en_clip09 + en_clip02 back to back (16 kHz mono PCM16, no gap), for the
long-form row of the Mac gate (runtime cuts it into 504-frame chunks, the oracle encodes it in one go).
Pure python (any venv). Prints and returns the per-part sample counts and the sha256 of the result."""
import hashlib
import json
import os
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
PARTS = ["en_clip04", "en_clip09", "en_clip02"]
OUT = os.path.join(HERE, "out", "fixtures", "long_en.wav")


def main():
    pcm, parts = b"", []
    for p in PARTS:
        with wave.open(os.path.join(HERE, "out", "fixtures", f"{p}.wav"), "rb") as w:
            assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2, p
            n = w.getnframes()
            pcm += w.readframes(n)
            parts.append({"id": p, "n_samples": n})
    with wave.open(OUT, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm)
    n = len(pcm) // 2
    rec = {"file": os.path.relpath(OUT, HERE), "parts": parts, "n_samples": n, "duration_s": n / 16000,
           "runtime_frames_960": -(-n // 960), "sha256": hashlib.sha256(open(OUT, "rb").read()).hexdigest()}
    print(json.dumps(rec, indent=1))
    return rec


if __name__ == "__main__":
    main()
