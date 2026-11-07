#!/usr/bin/env python3
"""Unpack an open-asr-leaderboard parquet split into 16 kHz mono PCM16 WAV files.

Writes <out>/<id>.wav, <out>/manifest.jsonl ({id, path, text, audio_length_s}) and
<out>/manifest.tsv (id<TAB>path, the runner's input). Every utterance keeps the
dataset's own `id`, `text` (raw; the leaderboard normalizes at scoring time) and
`audio_length_s`.
"""
import argparse
import hashlib
import io
import json
import pathlib
import sys

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("out")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(args.parquet)
    cols = table.column_names
    print("columns:", cols, "rows:", table.num_rows, file=sys.stderr)
    n = table.num_rows if args.limit <= 0 else min(args.limit, table.num_rows)
    total = 0.0
    sha = hashlib.sha256()
    with open(out / "manifest.jsonl", "w") as mj, open(out / "manifest.tsv", "w") as mt:
        for i in range(n):
            row = {c: table.column(c)[i].as_py() for c in cols}
            audio = row["audio"]
            data, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=True)
            if data.shape[1] != 1:
                data = data.mean(axis=1, keepdims=True)
            if sr != 16000:
                raise SystemExit(f"{row['id']}: sample rate {sr}, expected 16000")
            pcm = data[:, 0]
            utt_id = str(row["id"]).replace("/", "_")
            path = out / f"{utt_id}.wav"
            sf.write(path, pcm, 16000, subtype="PCM_16")
            seconds = len(pcm) / 16000.0
            total += seconds
            sha.update(path.read_bytes())
            rec = {
                "id": utt_id,
                "path": str(path),
                "text": row["text"],
                "audio_length_s": row.get("audio_length_s", seconds),
                "seconds_written": seconds,
                "dataset": row.get("dataset"),
            }
            mj.write(json.dumps(rec, ensure_ascii=False) + "\n")
            mt.write(f"{utt_id}\t{path}\n")
            if (i + 1) % 500 == 0:
                print(f"{i + 1}/{n} {total / 3600:.2f} h", file=sys.stderr)
    (out / "SET.json").write_text(json.dumps({
        "source_parquet": str(pathlib.Path(args.parquet).resolve()),
        "utterances": n,
        "total_seconds": total,
        "wav_sha256_chain": sha.hexdigest(),
    }, indent=2))
    print(f"done: {n} utterances, {total / 3600:.3f} h -> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
