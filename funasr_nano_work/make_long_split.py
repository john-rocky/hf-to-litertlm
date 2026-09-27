#!/usr/bin/env python3
"""Round 3 B: cut out/fixtures/long_en.wav (make_long_fixture.py: en_clip04 + en_clip09 + en_clip02, 60.175 s) at
483,840 samples (= 504 frames x 960 = 30.24 s = one runtime audio window) into long_en_p1.wav / long_en_p2.wav.
This is the same place the runtime cuts the 60 s clip when it is sent as one message (504-frame chunks, no overlap).

Also records where the cut falls: which source clip, how far into it, and the RMS level (20 ms frames) around it, so
the word that is cut can be named from the transcripts. Pure python (any venv). Writes long_split_fixture.json.
"""
import hashlib
import json
import math
import os
import struct
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "out", "fixtures")
PARTS = ["en_clip04", "en_clip09", "en_clip02"]
CUT = 504 * 960  # 483,840 samples


def read_pcm(path):
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2, path
        return w.readframes(w.getnframes())


def write_pcm(path, pcm):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm)
    return {"file": os.path.relpath(path, HERE), "n_samples": len(pcm) // 2, "duration_s": len(pcm) / 2 / 16000,
            "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}


def rms_dbfs(samples):
    if not samples:
        return None
    r = math.sqrt(sum(s * s for s in samples) / len(samples))
    return round(20 * math.log10(r / 32768), 1) if r > 0 else -120.0


def main():
    pcm = read_pcm(os.path.join(FIX, "long_en.wav"))
    n = len(pcm) // 2
    starts, pos = {}, 0
    for p in PARTS:
        k = len(read_pcm(os.path.join(FIX, f"{p}.wav"))) // 2
        starts[p] = (pos, pos + k)
        pos += k
    assert pos == n, (pos, n)
    rec = {"source": "out/fixtures/long_en.wav", "source_n_samples": n, "cut_sample": CUT, "cut_s": CUT / 16000,
           "parts_samples": {p: list(v) for p, v in starts.items()}}
    inside = [p for p, (a, b) in starts.items() if a <= CUT < b][0]
    rec["cut_inside"] = {"clip": inside, "offset_samples": CUT - starts[inside][0],
                         "offset_s": (CUT - starts[inside][0]) / 16000}
    rec["p1"] = write_pcm(os.path.join(FIX, "long_en_p1.wav"), pcm[:CUT * 2])
    rec["p2"] = write_pcm(os.path.join(FIX, "long_en_p2.wav"), pcm[CUT * 2:])
    assert rec["p1"]["n_samples"] + rec["p2"]["n_samples"] == n
    # RMS in 20 ms frames from -600 ms to +600 ms around the cut (frame start relative to the cut, ms)
    sam = struct.unpack(f"<{n}h", pcm)
    prof = []
    for ms in range(-600, 600, 20):
        a = CUT + ms * 16
        prof.append([ms, rms_dbfs(sam[max(a, 0):max(a + 320, 0)])])
    rec["rms_dbfs_20ms_around_cut"] = prof
    # start of speech in en_clip09 (first 20 ms frame above -40 dBFS), for the "how far into the first word" note
    a9 = starts["en_clip09"][0]
    first = None
    for k in range(0, 3 * 16000, 320):
        v = rms_dbfs(sam[a9 + k:a9 + k + 320])
        if v is not None and v > -40:
            first = k / 16000
            break
    rec["en_clip09_first_frame_above_-40dBFS_s"] = first
    with open(os.path.join(HERE, "long_split_fixture.json"), "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps({k: v for k, v in rec.items() if k != "rms_dbfs_20ms_around_cut"}, indent=1))
    print("rms dBFS around the cut (ms from cut: level):",
          " ".join(f"{ms}:{v}" for ms, v in prof if -300 <= ms < 300))


if __name__ == "__main__":
    main()
