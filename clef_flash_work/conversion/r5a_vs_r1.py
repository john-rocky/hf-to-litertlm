#!/usr/bin/env python3
"""Round 5a: the shipped host's row-graph inputs against the inputs r1 and r3 actually fed the row graph.

    VECLIB_MAXIMUM_THREADS=1 venv-host/bin/python scripts/r5a_vs_r1.py      (from K) -> results/r5a_vs_r1.json

r1's torch check (results/torch_check_L1024.json, 22 runs = own_text 13 + own_json 3 + SemIf 3 + red arms 3) fed
`scripts/clef_host.py row_inputs(ids, 1024, EmbedTable(<snapshot>))` (the bf16 rows read in place from the checkpoint
shard) to the torch row graph; r3 builds an image record's inputs with `scripts/r3_host.py image_row_inputs` (42 image
runs, the reference's picture rows). Here the host package (host/clef_litert: the out/r5a table file, its own
encoder, positions and tables) builds the same four arrays: embeds / valid / cos / sin must be bit-equal. Both r1 / r3
modules are NumPy-only and are only compared against, never imported by the host package."""
import json
import sys
import time
from pathlib import Path

import numpy as np

K = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K / "host"))
sys.path.insert(0, str(K / "scripts"))
import clef_litert as C  # noqa: E402
import clef_host as R1  # noqa: E402
import r3_host as R3  # noqa: E402

SNAP = K / "hf/hub/models--Cloudflare--clef-flash/snapshots/f0b7d3cd65296abfe459e6c228b618f5d8b26d2f"
NAMES = ("embeds", "valid", "cos", "sin")


def bit_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and bool((a.view(np.uint32) == b.view(np.uint32)).all())


def main() -> int:
    from tokenizers import Tokenizer

    t0 = time.time()
    tok = Tokenizer.from_file(str(SNAP / "tokenizer.json"))
    host = C.ClefHost.from_files(SNAP / "tokenizer.json", K / "out/r5a/embed_tokens.safetensors",
                                 K / "out/r5a/lm_head.safetensors", SNAP / "joint_head.safetensors",
                                 SNAP / "joint_head_config.json")
    r1_table = R1.EmbedTable(SNAP)
    ref = {(r["id"], r["arm"]): r for r in json.loads((K / "ref/records_ref.json").read_text())["rows"]}
    ref.update({(r["id"], r["arm"]): r for r in json.loads((K / "ref/records_ref_red.json").read_text())["rows"]})
    fx = {r["id"]: r for r in json.loads((K / "fixtures/records.json").read_text())["records"]}
    fx.update({r["id"]: r for r in json.loads((K / "fixtures/red_arms.json").read_text())["records"]})
    tc = json.loads((K / "results/torch_check_L1024.json").read_text())
    rows = []
    for x in tc["rows"]:
        row = ref[(x["id"], x["arm"])]
        rec = C.encode_record(tok, fx[x["id"]]["request"])
        assert list(rec.ids) == row["ids"]
        mine = host.row_inputs(C.Encoded(rec, tc["L"]))
        theirs = dict(zip(NAMES, R1.row_inputs(row["ids"], tc["L"], r1_table)))
        rows.append({"run": f"{x['id']}/{x['arm']}", "L": tc["L"], "against": "r1 clef_host.row_inputs (torch check)",
                     **{n: bit_equal(mine[n], theirs[n]) for n in NAMES}})
    for (rid, arm), row in ref.items():
        if arm == "text":
            continue
        z = np.load(K / f"ref/npz/{rid}__{arm}.npz")
        rec = C.encode_record(tok, fx[rid]["request"], image_grid=tuple(row["merged_hw"]))
        assert list(rec.ids) == row["ids"]
        L = C.pick_length(rec.tokens)
        mine = host.row_inputs(C.Encoded(rec, L), z["image_embeds"])
        h, w = row["merged_hw"]
        theirs = dict(zip(NAMES, R3.image_row_inputs(row["ids"], L, r1_table, z["image_embeds"], row["token_offset"], h, w)))
        rows.append({"run": f"{rid}/{arm}", "L": L, "against": "r3 r3_host.image_row_inputs",
                     **{n: bit_equal(mine[n], theirs[n]) for n in NAMES}})
    ok = all(all(r[n] for n in NAMES) for r in rows)
    doc = {"what": "round 5a: host row-graph inputs vs r1's (torch check) and r3's builders, bit for bit",
           "r1_table": r1_table.info, "host_table": host.embed_tokens.info, "runs": len(rows),
           "summary": {n: f"{sum(r[n] for r in rows)}/{len(rows)}" for n in NAMES}, "pass": ok,
           "seconds": round(time.time() - t0, 1), "rows": rows}
    (K / "results/r5a_vs_r1.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: doc[k] for k in ("runs", "summary", "pass", "seconds")}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
