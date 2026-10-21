#!/usr/bin/env python3
"""Read only the metadata columns (file_name, duration, text, source_lang) of the Open ASR Leaderboard's FLEURS
test parquet (hf-audio/open-asr-leaderboard-multilingual-datasets) over HTTP range requests -- the audio column is
not downloaded. Writes data/fleurs/lb_<lang>_text.jsonl + a record of the file, revision and bytes read.
usage: lb_text_cols.py <lang> [<lang> ...]
"""
import json
import sys

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

REPO = "hf-audio/open-asr-leaderboard-multilingual-datasets"
REV = "b791fc9151221e5b7e59c6c2dfa4dee09dda3cb7"


def main(langs):
    fs = HfFileSystem()
    rec = {"repo": REPO, "revision": REV, "files": []}
    for lang in langs:
        path = f"datasets/{REPO}@{REV}/data/fleurs/{lang}_test.parquet"
        with fs.open(path, "rb", block_size=1 << 20) as f:
            pf = pq.ParquetFile(f)
            names = pf.schema_arrow.names
            cols = [c for c in ("file_name", "duration", "source_lang", "target_lang", "text") if c in names]
            t = pf.read(columns=cols)
            nrg = pf.metadata.num_row_groups
            size = pf.metadata.serialized_size
        out = f"data/fleurs/lb_{lang}_text.jsonl"
        with open(out, "w") as fo:
            for i in range(t.num_rows):
                fo.write(json.dumps({c: t.column(c)[i].as_py() for c in cols}, ensure_ascii=False) + "\n")
        rec["files"].append({"lang": lang, "path": path, "rows": t.num_rows, "row_groups": nrg,
                             "columns": cols, "schema": names, "footer_bytes": size, "out": out})
        print(lang, t.num_rows, "rows", cols, "of", names)
    json.dump(rec, open("data/fleurs/lb_text_cols.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
