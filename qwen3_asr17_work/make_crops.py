#!/usr/bin/env python3
"""5 s crops (copied from confucius4_r2t2_work/make_crops.py; used here by encoder_window_parity.py only): for each
FLEURS config take, in meta.json order (ascending FLEURS id), the first 20 clips with duration >= 5.0 s and write their
first N samples as PCM16 mono WAV, byte-exact copies of the source samples, to <out_dir>/<clip>__crop5s.wav. Writes
<out_dir>/manifest.json and <out_dir>/manifest.tsv (<id>\t<wav path>). N = 79,999 gives 500 mel frames from the HF
feature extractor (one 5 s window).

  python3 make_crops.py --samples 79999 --out_dir out/crops79999
"""
import argparse
import hashlib
import json
import os
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.environ.get("C4_FIXTURES", os.path.expanduser("~/code/coreai/_funasr_nano/fixtures"))
N_PER_CONFIG = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=80000)
    ap.add_argument("--out_dir", default=os.path.join(HERE, "out", "crops"))
    args = ap.parse_args()
    CROP = args.samples
    OUT = os.path.abspath(args.out_dir)
    os.makedirs(OUT, exist_ok=True)
    meta = json.load(open(os.path.join(FIX, "meta.json")))["clips"]
    crops = []
    for cfg in ("en_us", "cmn_hans_cn", "ja_jp"):
        picked = [c for c in meta if c["path"].startswith("fleurs/") and c["source"]["config"] == cfg
                  and c["duration_s"] >= 5.0][:N_PER_CONFIG]
        assert len(picked) == N_PER_CONFIG, (cfg, len(picked))
        for c in picked:
            src = os.path.join(FIX, c["path"])
            with wave.open(src, "rb") as w:
                params = w.getparams()
                assert params.framerate == 16000 and params.nchannels == 1 and params.sampwidth == 2
                frames = w.readframes(CROP)
            assert len(frames) == CROP * 2
            name = c["name"] + "__crop5s"
            dst = os.path.join(OUT, name + ".wav")
            with wave.open(dst, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(frames)
            crops.append({"name": name, "path": dst, "config": cfg, "source_clip": c["name"],
                          "source_duration_s": c["duration_s"], "samples": CROP,
                          "pcm_sha256": hashlib.sha256(frames).hexdigest()})
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump({"rule": f"per config: first {N_PER_CONFIG} FLEURS clips in meta.json order with duration_s >= 5.0; "
                           f"first {CROP} samples", "crops": crops}, f, indent=1)
    with open(os.path.join(OUT, "manifest.tsv"), "w") as f:
        for c in crops:
            f.write(f"{c['name']}\t{c['path']}\n")
    print(len(crops), "crops;", {cfg: [c["source_clip"] for c in crops if c["config"] == cfg][:3] for cfg in
                                 ("en_us", "cmn_hans_cn", "ja_jp")})


if __name__ == "__main__":
    main()
