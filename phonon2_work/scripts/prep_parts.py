#!/usr/bin/env python3
"""Unpack an open-asr-leaderboard split held in several parquet parts into 16 kHz mono PCM16 WAV files: prep_librispeech.py's
row handling, over the parts in the order given. Refuses a sample rate other than 16,000 and a repeated id.

Writes <out>/<id>.wav, <out>/manifest.jsonl ({id, path, text, audio_length_s, seconds_written, dataset}),
<out>/manifest.tsv (id<TAB>path) and <out>/SET.json (the parts, utterances, total seconds, wav_sha256_chain).
CLI: prep_parts.py <out> <part.parquet> [<part.parquet> ...]
"""
import hashlib
import io
import json
import pathlib
import sys

import numpy as np  # noqa: F401
import pyarrow.parquet as pq
import soundfile as sf


def main() -> int:
    out = pathlib.Path(sys.argv[1])
    parts = sys.argv[2:]
    out.mkdir(parents=True, exist_ok=True)
    total = 0.0
    n = 0
    seen = set()
    sha = hashlib.sha256()
    with open(out / "manifest.jsonl", "w") as mj, open(out / "manifest.tsv", "w") as mt:
        for part in parts:
            table = pq.read_table(part)
            cols = table.column_names
            print("part:", part, "columns:", cols, "rows:", table.num_rows, file=sys.stderr)
            for i in range(table.num_rows):
                row = {c: table.column(c)[i].as_py() for c in cols}
                audio = row["audio"]
                data, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=True)
                if data.shape[1] != 1:
                    data = data.mean(axis=1, keepdims=True)
                if sr != 16000:
                    raise SystemExit(f"{row['id']}: sample rate {sr}, expected 16000")
                pcm = data[:, 0]
                utt_id = str(row["id"]).replace("/", "_")
                if utt_id in seen:
                    raise SystemExit(f"repeated id {utt_id}")
                seen.add(utt_id)
                path = out / f"{utt_id}.wav"
                sf.write(path, pcm, 16000, subtype="PCM_16")
                seconds = len(pcm) / 16000.0
                total += seconds
                sha.update(path.read_bytes())
                rec = {"id": utt_id, "path": str(path), "text": row["text"],
                       "audio_length_s": row.get("audio_length_s", seconds), "seconds_written": seconds,
                       "dataset": row.get("dataset")}
                mj.write(json.dumps(rec, ensure_ascii=False) + "\n")
                mt.write(f"{utt_id}\t{path}\n")
                n += 1
    (out / "SET.json").write_text(json.dumps({"source_parquets": [str(pathlib.Path(p).resolve()) for p in parts],
                                              "utterances": n, "total_seconds": total,
                                              "wav_sha256_chain": sha.hexdigest()}, indent=2))
    print(f"done: {n} utterances, {total / 3600:.3f} h -> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
