#!/usr/bin/env python3
"""Unpack a google/fleurs test parquet (parquet-data/<lang>_<cc>/test-00000-of-00001.parquet) into WAV files.

The audio bytes are written verbatim (the parquet stores float32 RIFF WAVs; no re-encoding), named by the audio's
own file name, in parquet row order. The leaderboard's text column (data/fleurs/lb_<lang>_text.jsonl, from
lb_text_cols.py) is joined by that file name. manifest.jsonl rows: id (= wav file stem), path, text (the leaderboard
text), raw_transcription and transcription (google/fleurs), fleurs_id, audio_length_s, lb_duration.
SET.json: utterances, total seconds, sha256 chain over the WAV bytes in row order, join checks.
usage: prep_fleurs.py <parquet> <lang2> <out_dir>
"""
import hashlib
import io
import json
import os
import pathlib
import sys

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf


def main(argv):
    pq_path, lang, out = argv[1], argv[2], pathlib.Path(argv[3])
    out.mkdir(parents=True, exist_ok=True)
    lb = {}
    for line in open(f"data/fleurs/lb_{lang}_text.jsonl"):
        r = json.loads(line)
        lb[os.path.basename(r["file_name"])] = r
    t = pq.read_table(pq_path)
    sha = hashlib.sha256()
    total = 0.0
    n = 0
    miss_lb = []
    dur_bad = []
    ns_bad = []
    subtypes = {}
    seen = set()
    with open(out / "manifest.jsonl", "w") as mj:
        for i in range(t.num_rows):
            row = {c: t.column(c)[i].as_py() for c in ("id", "num_samples", "transcription", "raw_transcription")}
            audio = t.column("audio")[i].as_py()
            name = os.path.basename(audio["path"])
            assert name not in seen, name
            seen.add(name)
            b = audio["bytes"]
            x, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
            info = sf.info(io.BytesIO(b))
            subtypes[info.subtype] = subtypes.get(info.subtype, 0) + 1
            if sr != 16000 or x.shape[1] != 1:
                raise SystemExit(f"{name}: sr {sr} ch {x.shape[1]}")
            if x.shape[0] != row["num_samples"]:
                ns_bad.append((name, x.shape[0], row["num_samples"]))
            p = out / name
            p.write_bytes(b)
            sha.update(b)
            sec = x.shape[0] / 16000.0
            total += sec
            lbr = lb.get(name)
            if lbr is None:
                miss_lb.append(name)
            elif abs(np.float32(lbr["duration"]) - np.float32(sec)) > 1e-3:
                dur_bad.append((name, lbr["duration"], sec))
            mj.write(json.dumps({
                "id": pathlib.Path(name).stem, "path": str(p),
                "text": lbr["text"] if lbr else None,
                "raw_transcription": row["raw_transcription"], "transcription": row["transcription"],
                "fleurs_id": row["id"], "audio_length_s": sec, "seconds_written": sec,
                "lb_duration": lbr["duration"] if lbr else None,
            }, ensure_ascii=False) + "\n")
            n += 1
    lb_only = sorted(set(lb) - seen)
    rec = {"source_parquet": pq_path, "lang": lang, "utterances": n, "total_seconds": total,
           "wav_bytes_sha256_chain": sha.hexdigest(), "wav_subtypes": subtypes,
           "joined_leaderboard_text": n - len(miss_lb), "google_rows_without_lb_row": miss_lb[:20],
           "lb_rows_without_google_row": lb_only[:20], "n_lb_rows": len(lb),
           "duration_mismatch_gt_1ms": dur_bad[:20], "n_duration_mismatch": len(dur_bad),
           "num_samples_mismatch": ns_bad[:20]}
    json.dump(rec, open(out / "SET.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in rec.items() if not isinstance(v, list) or v}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
