#!/usr/bin/env python3
"""Round 3 host check (no model weights but the bf16 embedding rows): the row graph's `embeds` of an image record,
built by scripts/r3_host.py image_row_inputs from the reference's ids and the reference's own image rows (npz
image_embeds), against the author's layer-0 input of the same run (the reference npz hidden_states[0] of img_01/g256 =
embed_tokens(ids) in float32 with the image rows scattered in, scripts/clef_ref.py HIDDEN_RUNS).

    venv-ref/bin/python scripts/r3_embeds_check.py      -> results/r3_embeds_check.json (refuses to overwrite)

Also for every image run (42): the image rows sit at P+1 .. P+N, the <|vision_start|> / <|vision_end|> rows are the
table's, valid = 1 on the record and 0 after, and the pad rows are 0 (shape checks at the run's smallest bucket)."""
import json
import sys
from pathlib import Path

import numpy as np

K = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K / "scripts"))
import clef_host as CH  # noqa: E402
import r3_host as RH  # noqa: E402

SNAP = K / "hf/hub/models--Cloudflare--clef-flash/snapshots/f0b7d3cd65296abfe459e6c228b618f5d8b26d2f"


def main() -> int:
    out = K / "results/r3_embeds_check.json"
    assert not out.exists(), f"refusing to overwrite {out}"
    ref = json.loads((K / "ref/records_ref.json").read_text())
    table = CH.EmbedTable(SNAP)
    doc = {"what": "round 3: host embeds of image records vs the author's layer-0 input", "table": table.info}
    r = next(x for x in ref["rows"] if x["id"] == "img_01" and x["arm"] == "g256")
    z = np.load(K / "ref/npz/img_01__g256.npz")
    T, P = r["tokens"], r["token_offset"]
    emb, valid, cos, sin = RH.image_row_inputs(r["ids"], 512, table, z["image_embeds"], P, 8, 8)
    l0 = z["hidden_states"][0]
    doc["layer0_img_01_g256"] = {"tokens": T, "bit_equal": bool(np.array_equal(emb[0, :T].view(np.uint32), l0.view(np.uint32))),
                                 "max_abs": float(np.abs(emb[0, :T].astype(np.float64) - l0).max()),
                                 "pad_rows_zero": bool((emb[0, T:] == 0).all()), "valid_sum": float(valid.sum())}
    shapes = []
    for r in ref["rows"]:
        if r["arm"] == "text":
            continue
        z = np.load(K / f"ref/npz/{r['id']}__{r['arm']}.npz")
        T, P = r["tokens"], r["token_offset"]
        gt, gh, gw = r["grid_thw"]
        H, W = gh // 2, gw // 2
        L = RH.bucket(T)
        e, v, c, s = RH.image_row_inputs(r["ids"], L, table, z["image_embeds"], P, H, W)
        ids = np.asarray(r["ids"])
        ok = (np.array_equal(e[0, P + 1:P + 1 + H * W], z["image_embeds"])
              and np.array_equal(e[0, P], table.rows([RH.VISION_START])[0])
              and np.array_equal(e[0, P + 1 + H * W], table.rows([RH.VISION_END])[0])
              and float(v[0, :T].min()) == 1.0 and float(v[0, T:].max(initial=0.0)) == 0.0 and (e[0, T:] == 0).all()
              and e.shape == (1, L, 4096) and c.shape == (1, L, 64))
        shapes.append({"id": r["id"], "arm": r["arm"], "T": T, "L": L, "N": H * W, "ok": bool(ok),
                       "image_pad_ids": int((ids == RH.IMAGE_PAD).sum())})
    doc["runs"] = shapes
    doc["summary"] = {"layer0_bit_equal": doc["layer0_img_01_g256"]["bit_equal"],
                      "runs_ok": f"{sum(x['ok'] for x in shapes)}/{len(shapes)}",
                      "buckets": {str(L): sum(x["L"] == L for x in shapes) for L in RH.BUCKETS}}
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(doc["summary"] | {"layer0": doc["layer0_img_01_g256"]}, indent=1))
    return 0 if doc["summary"]["layer0_bit_equal"] and all(x["ok"] for x in shapes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
