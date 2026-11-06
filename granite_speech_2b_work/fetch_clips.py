#!/usr/bin/env python3
"""The clips of the transcription check, by reference (no audio is committed here):

  multilingual_sample.wav   the source repo's own sample (ibm-granite/granite-speech-5.0-2b at the pinned revision,
                            IEEE float32 mono 16 kHz, 24.94 s), copied from the snapshot after a sha256 check
  ls_clip00 .. ls_clip04    the first five rows of hf-internal-testing/librispeech_asr_dummy (LibriSpeech dev-clean,
                            CC BY 4.0) at revision 5be91486, `clean/validation-00000-of-00001.parquet` (sha256-checked):
                            the FLAC bytes decoded to int16 with soundfile and written as PCM16 mono 16 kHz WAV

Writes OUT/<id>.wav and OUT/clips.json (id, file, sha256, samples, the dataset's reference text) and exits 1 when a
file's sha256 differs from the table below.

  python fetch_clips.py --src <snapshot dir> --out <dir>
"""
import argparse
import hashlib
import io
import json
import shutil
import sys
import urllib.request
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

LS_REPO = "hf-internal-testing/librispeech_asr_dummy"
LS_REV = "5be91486e11a2d616f4ec5db8d3fd248585ac07a"
LS_FILE = "clean/validation-00000-of-00001.parquet"
LS_SHA256 = "4e69a06fa5edc90921e5e7e39a7084881f8b3ed9c805c574f4f39c6fde27c603"
LS_BYTES = 9_192_059
SAMPLE_SHA256 = "91d243650809c1274141ec20ff23045315eaf27567694002ea3ef390048b7058"
N_LS = 5
# sha256 of each written file
EXPECT = {
    "multilingual_sample": SAMPLE_SHA256,
    "ls_clip00": "799f78ed4beb4de7ceae3a809262d4ce242394342ccd1d58cef7d49dbc2def46",
    "ls_clip01": "473b1bb573e8f7cdcd639c6e318036236190107b66bfe6892286121a92d464c1",
    "ls_clip02": "5f4d8f7331acffc3d2efbe0f764b9eb3447911dc17f9843e38e1083eb1685b9c",
    "ls_clip03": "fc23353b884061a5c5784b0cf6b5c539680c50cfae993882822ebdd4a06ddf8c",
    "ls_clip04": "4798e34d9a323ed835ae6055172236f8965dd5ccf146c25758711023ecc8b2e6",
}


def fetch(url, path, sha, size):
    if path.exists() and path.stat().st_size == size and C.sha256(path) == sha:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    got = C.sha256(tmp)
    if tmp.stat().st_size != size or got != sha:
        raise SystemExit(f"{url}: {tmp.stat().st_size} B sha256 {got}, expected {size} B {sha}")
    tmp.rename(path)


def write_wav(path, pcm_i16):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(C.SR)
        w.writeframes(pcm_i16.astype("<i2").tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="snapshot of ibm-granite/granite-speech-5.0-2b")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clips = []
    sample = Path(args.src) / "multilingual_sample.wav"
    assert C.sha256(sample) == SAMPLE_SHA256, sample
    shutil.copyfile(sample, out / "multilingual_sample.wav")
    clips.append({"id": "multilingual_sample", "path": "multilingual_sample.wav",
                  "source": f"{C.REPO_ID}@{C.REVISION} multilingual_sample.wav", "license": "Apache-2.0"})
    pq_path = out / "librispeech_asr_dummy" / LS_FILE
    fetch(f"https://huggingface.co/datasets/{LS_REPO}/resolve/{LS_REV}/{LS_FILE}", pq_path, LS_SHA256, LS_BYTES)
    rows = pq.read_table(pq_path, columns=["audio", "text", "id"]).slice(0, N_LS).to_pylist()
    for i, r in enumerate(rows):
        data, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="int16")
        assert sr == C.SR and data.ndim == 1, (r["id"], sr, data.shape)
        cid = f"ls_clip{i:02d}"
        write_wav(out / f"{cid}.wav", np.asarray(data))
        clips.append({"id": cid, "path": f"{cid}.wav", "source": f"{LS_REPO}@{LS_REV} {LS_FILE} row {i} ({r['id']})",
                      "license": "CC BY 4.0 (LibriSpeech)", "ref": r["text"]})
    bad = []
    for c in clips:
        p = out / c["path"]
        c["sha256"] = C.sha256(p)
        c["samples"] = len(C.wav_float(p))
        c["seconds"] = round(c["samples"] / C.SR, 3)
        if EXPECT and EXPECT.get(c["id"]) != c["sha256"]:
            bad.append((c["id"], c["sha256"], EXPECT.get(c["id"])))
        print(f"{c['id']:<20} {c['seconds']:6.2f} s {c['sha256']}", flush=True)
    (out / "clips.json").write_text(json.dumps({"clips": clips}, ensure_ascii=False, indent=1))
    if bad:
        raise SystemExit(f"sha256 differs from the table: {bad}")
    print("FETCH_CLIPS_DONE", flush=True)


if __name__ == "__main__":
    main()
