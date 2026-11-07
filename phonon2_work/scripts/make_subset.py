#!/usr/bin/env python3
"""Cut a fixed-stride subset out of a prep_librispeech.py set (S4 = every 4th utterance, S32 = every 32nd).

S<k> = the rows of <set>/manifest.tsv at index 0, k, 2k, ... in the file's order
(= the parquet's row order). Writes next to the set:
  manifest-S<k>.tsv    id<TAB>path, the subset rows in order
  manifest-S<k>.jsonl  the manifest.jsonl rows of those ids, same order (score.py's --manifest)
  SET-S<k>.json        utterances, total_seconds and wav_sha256_chain over the subset's WAVs
                       in subset order (the computation prep_librispeech.py does for SET.json),
                       plus the same three over the full set recomputed from the WAVs on disk.
"""
import argparse
import hashlib
import json
import pathlib
import sys
import wave


def wav_seconds(path):
    with wave.open(str(path), "rb") as w:
        if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (1, 2, 16000):
            raise SystemExit(f"{path}: not 16 kHz mono PCM16")
        return w.getnframes() / 16000.0


def chain(rows):
    sha = hashlib.sha256()
    total = 0.0
    for _, path in rows:
        p = pathlib.Path(path)
        sha.update(p.read_bytes())
        total += wav_seconds(p)
    return {"utterances": len(rows), "total_seconds": total, "wav_sha256_chain": sha.hexdigest()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("set_dir")
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--parquet-record", default="", help="PARQUET.json written by fetch_set.sh")
    args = ap.parse_args()

    d = pathlib.Path(args.set_dir)
    rows = [line.rstrip("\n").split("\t") for line in open(d / "manifest.tsv") if line.strip()]
    meta = {}
    for line in open(d / "manifest.jsonl"):
        if line.strip():
            r = json.loads(line)
            meta[r["id"]] = r
    if [i for i, _ in rows] != list(meta):
        raise SystemExit("manifest.tsv and manifest.jsonl disagree on ids or order")
    k = args.every
    sub = rows[::k]
    tag = f"S{k}"
    with open(d / f"manifest-{tag}.tsv", "w") as f:
        for i, p in sub:
            f.write(f"{i}\t{p}\n")
    with open(d / f"manifest-{tag}.jsonl", "w") as f:
        for i, _ in sub:
            f.write(json.dumps(meta[i], ensure_ascii=False) + "\n")

    full = chain(rows)
    subset = chain(sub)
    rec = {
        "set": f"librispeech-test-clean-{tag}",
        "rule": f"manifest.tsv rows at index 0, {k}, {2 * k}, ... (every {k}th row from the first, file order)",
        **subset,
        "first_ids": [i for i, _ in sub[:3]],
        "last_id": sub[-1][0],
        "full": full,
    }
    if args.parquet_record:
        rec["parquet"] = json.loads(pathlib.Path(args.parquet_record).read_text())
    (d / f"SET-{tag}.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
