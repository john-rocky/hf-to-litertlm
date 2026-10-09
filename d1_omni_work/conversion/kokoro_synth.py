"""Round 2 step 1 (audio): Kokoro-82M synthesis of our three audio records -> K/fixtures/audio/<id>.wav,
K/fixtures/audio/kokoro_manifest.json

    cd K
    PYTHONDONTWRITEBYTECODE=1 ~/code/executorch-convert/.venv/bin/python -B scripts/kokoro_synth.py \
        --tmp-dir <dir outside K> [--voice-file af_sarah=<path to voices/af_sarah.pt>]

The Core AI lane d1d's recipe (~/code/coreai/_d1_omni/audio/manifest.json, scripts/synth_aud.py): KPipeline(lang_code
"a", repo_id "hexgrad/Kokoro-82M"), speed 1.0, torch.manual_seed(0) before each clip, the clip's chunks concatenated,
24 kHz float -> clip to [-1, 1] -> int16 (x 32767) WAV -> `ffmpeg -ar 16000 -ac 1 -sample_fmt s16` -> 16 kHz mono
int16 WAV. Offline (HF_HUB_OFFLINE=1): the model and the voices come from the shared HF cache; a voice that is not
cached is passed as a .pt file (--voice-file, fetched by the caller at the pinned revision and sha256-checked) so
nothing is written to the cache or to the venv (run with -B). Transcript and voice per record = fixtures/requests.json
(`media.transcript`, `media.voice` of aud_reservation_01 / aud_weather_02 / aud_food_03).
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import wave
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
sys.dont_write_bytecode = True

K = Path(__file__).resolve().parents[1]
IDS = ("aud_reservation_01", "aud_weather_02", "aud_food_03")
RECIPE = {"model": "hexgrad/Kokoro-82M", "package": "kokoro 0.9.4 + misaki", "lang_code": "a", "speed": 1.0,
          "seed": 0, "native_rate": 24000, "rate": 16000, "resample": "ffmpeg -ar 16000 -ac 1 -sample_fmt s16"}
CACHE = Path.home() / ".cache/huggingface/hub/models--hexgrad--Kokoro-82M"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmp-dir", required=True)
    ap.add_argument("--voice-file", action="append", default=[], help="name=path to a voices/<name>.pt file")
    ap.add_argument("--out-dir", default=str(K / "fixtures/audio"), help="a re-run elsewhere checks reproducibility")
    a = ap.parse_args()
    tmp_dir = Path(a.tmp_dir).resolve()
    assert K not in tmp_dir.parents, "keep temporary files out of K"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    files = dict(v.split("=", 1) for v in a.voice_file)

    import numpy as np
    import torch
    import kokoro
    from kokoro import KPipeline

    doc = json.loads((K / "fixtures/requests.json").read_text())
    recs = {r["id"]: r for r in doc["records"] if r["id"] in IDS}
    assert sorted(recs) == sorted(IDS), sorted(recs)
    rev = (CACHE / "refs/main").read_text().strip()
    t0 = time.time()
    pipe = KPipeline(lang_code=RECIPE["lang_code"], repo_id=RECIPE["model"])
    load_s = time.time() - t0
    ff = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0]
    manifest = {"status": "synthesised", "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "synth": RECIPE,
                "kokoro_version": getattr(kokoro, "__version__", "unknown"), "torch": torch.__version__,
                "python": sys.executable, "hub_revision_in_cache": rev, "ffmpeg": ff, "load_s": round(load_s, 2),
                "recipe_from": "~/code/coreai/_d1_omni/audio/manifest.json (scripts/synth_aud.py)", "clips": {}}
    out_dir = Path(a.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for cid in IDS:
        m = recs[cid]["media"]
        voice = m["voice"]
        if voice in files:
            vpath = files[voice]
            vinfo = {"voice_file": vpath, "voice_sha256": sha256_file(vpath), "voice_source": "--voice-file"}
            vref = vpath
        else:
            vp = CACHE / "snapshots" / rev / "voices" / f"{voice}.pt"
            vinfo = {"voice_file": str(vp), "voice_sha256": sha256_file(vp), "voice_source": "shared HF cache"}
            vref = voice
        torch.manual_seed(RECIPE["seed"])
        t1 = time.time()
        parts, phonemes = [], []
        for _graphemes, ps, audio in pipe(m["transcript"], voice=vref, speed=RECIPE["speed"]):
            parts.append(np.asarray(audio, dtype=np.float32))
            phonemes.append(ps)
        w = np.concatenate(parts)
        tmp = tmp_dir / f".{cid}_24k.wav"
        with wave.open(str(tmp), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(RECIPE["native_rate"])
            f.writeframes((np.clip(w, -1, 1) * 32767).astype(np.int16).tobytes())
        out = out_dir / f"{cid}.wav"
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-ar", str(RECIPE["rate"]), "-ac", "1", "-sample_fmt",
               "s16", str(out)]
        subprocess.run(cmd, check=True)
        os.remove(tmp)
        with wave.open(str(out), "rb") as f:
            rate, channels, width, frames = f.getframerate(), f.getnchannels(), f.getsampwidth(), f.getnframes()
        manifest["clips"][cid] = {
            "file": str(out.relative_to(K)) if K in out.parents else str(out), "voice": voice, **vinfo, "seed": RECIPE["seed"],
            "text": m["transcript"], "phonemes": phonemes, "chunks": len(parts),
            "native_seconds": round(len(w) / RECIPE["native_rate"], 4), "seconds": round(frames / rate, 4),
            "sample_rate": rate, "channels": channels, "sample_width_bytes": width, "frames": frames,
            "sha256": sha256_file(out), "bytes": out.stat().st_size, "synth_s": round(time.time() - t1, 2),
            "ffmpeg_cmd": " ".join(cmd[:5] + ["<24 kHz int16 wav>"] + cmd[6:-1] + [f"fixtures/audio/{cid}.wav"])}
        print(f"{cid}: {frames / rate:.2f} s voice {voice} ({time.time() - t1:.1f} s)", flush=True)
    (out_dir / "kokoro_manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print("manifest written", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
