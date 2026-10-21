#!/usr/bin/env python3
"""Round 3 acceptance 1-2 (no model weights): the host's pixels, ids and M-RoPE planes of every reference image run,
and the host's cos / sin against the author's rotary (transformers 5.17.0 = the reference's version).

    HF_HUB_OFFLINE=1 venv-ref/bin/python scripts/r3_pixpos.py      -> results/r3_pixpos.json (refuses to overwrite)

1. Pixels (g256 / g448, 28 runs): scripts/r3_host.py tile_patches(fixtures/images/<file>, tile) vs the reference npz
   `pixel_values` (the processor's output inside systemone()): bit-equal (uint32 views), else the count of differing
   elements, max |diff| and the uint8 tile's own check (the resize vs the normalization).
2. ids (42 runs): encode_image_record(tokenizers.Tokenizer(<snapshot>/tokenizer.json), request, N) vs the reference
   row's ids, token_offset and every question / option span.
3. Planes (42 runs = g256 / g448 / native): image_positions(T, P, H, W) (H, W = the reference's grid_thw // 2) vs the
   npz `rope_pos` [3, T] (what the author's text rotary received): equality.
4. cos / sin (42 runs): clef_host.mrope_tables(planes) vs Qwen3_5TextRotaryEmbedding(text_config).forward(x float32,
   rope_pos[:, None, :]) of transformers 5.17.0 / torch 2.9.0: largest difference in ulps, count of differing elements
   (every real position of the record; the row graph's pad positions are never read)."""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

K = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K / "scripts"))
import clef_host as CH  # noqa: E402
import r3_host as RH  # noqa: E402

REV = "f0b7d3cd65296abfe459e6c228b618f5d8b26d2f"
SNAP = K / "hf/hub/models--Cloudflare--clef-flash/snapshots" / REV


def ulps(a: np.ndarray, b: np.ndarray) -> tuple[int, int]:
    ia = a.astype(np.float32).view(np.int32).astype(np.int64)
    ib = b.astype(np.float32).view(np.int32).astype(np.int64)
    d = np.abs(ia - ib)
    return int(d.max()), int((d > 0).sum())


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    out = K / "results/r3_pixpos.json"
    assert not out.exists(), f"refusing to overwrite {out}"
    import torch
    import transformers
    import PIL
    from tokenizers import Tokenizer
    from transformers import AutoConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M

    t0 = time.time()
    ref = json.loads((K / "ref/records_ref.json").read_text())
    recs = {r["id"]: r for r in json.loads((K / "fixtures/records.json").read_text())["records"]}
    rows = [r for r in ref["rows"] if r["arm"] != "text"]
    assert len(rows) == 42, len(rows)
    tok = Tokenizer.from_file(str(SNAP / "tokenizer.json"))
    cfg = AutoConfig.from_pretrained(str(SNAP)).text_config
    rot = M.Qwen3_5TextRotaryEmbedding(cfg)
    res_pix, res_ids, res_pos, res_cs = [], [], [], []
    for r in rows:
        rid, arm = r["id"], r["arm"]
        z = np.load(K / f"ref/npz/{rid}__{arm}.npz")
        T, P = r["tokens"], r["token_offset"]
        gt, gh, gw = r["grid_thw"]
        H, W = gh // RH.MERGE, gw // RH.MERGE
        N = H * W
        assert gt == 1 and N == r["n_image_tokens"], (rid, arm)
        img = K / "fixtures/images" / recs[rid]["image_files"][0]
        # 1. pixels
        if arm in RH.TILES:
            tile = RH.TILES[arm]
            patches = RH.tile_patches(img, tile)
            pv = z["pixel_values"]
            assert patches.shape == pv.shape, (patches.shape, pv.shape)
            bits = patches.view(np.uint32) == pv.view(np.uint32)
            d = np.abs(patches.astype(np.float64) - pv.astype(np.float64))
            item = {"id": rid, "arm": arm, "tile": tile, "image": img.name, "image_sha256": sha256_file(img),
                    "shape": list(pv.shape), "bit_equal": bool(bits.all()), "elements_differing": int((~bits).sum()),
                    "max_abs_diff": float(d.max()), "pixel_values_sha256_ref": r["pixel_values_sha256"],
                    "patches_sha256": hashlib.sha256(np.ascontiguousarray(patches).tobytes()).hexdigest()}
            if not item["bit_equal"]:
                # which side: the uint8 tile (resize) or the normalization -> recover the reference's uint8 levels
                lv = np.rint((pv[:, :768].astype(np.float64) * 0.5 + 0.5) * 255.0)
                mine = np.rint((patches[:, :768].astype(np.float64) * 0.5 + 0.5) * 255.0)
                item["level_diff_max"] = float(np.abs(lv - mine).max())
            res_pix.append(item)
        # 2. ids
        enc = RH.encode_image_record(tok, recs[rid]["request"], N)
        ids_ok = enc["ids"] == r["ids"]
        off_ok = enc["token_offset"] == P
        spans_ok = all(q["question_id"] == rq["question_id"] and q["question_span"] == rq["question_span"]
                       and q["option_spans"] == rq["option_spans"] and q["option_ids"] == rq["option_ids"]
                       for q, rq in zip(enc["questions"], r["questions"])) and len(enc["questions"]) == len(r["questions"])
        res_ids.append({"id": rid, "arm": arm, "T": T, "N": N, "ids_equal": bool(ids_ok), "token_offset_equal": bool(off_ok),
                        "spans_equal": bool(spans_ok)})
        # 3. planes
        pos = RH.image_positions(T, P, H, W)
        rp = z["rope_pos"].astype(np.int64)
        eq = bool(np.array_equal(pos, rp))
        item = {"id": rid, "arm": arm, "T": T, "P": P, "merged_hw": [H, W], "N": N, "shift": N - max(H, W),
                "planes_equal": eq}
        if not eq:
            bad = np.argwhere(pos != rp)
            item["first_mismatches"] = bad[:8].tolist()
            item["mismatch_count"] = int(len(bad))
        res_pos.append(item)
        # 4. cos / sin on the real positions
        cos_np, sin_np = CH.mrope_tables(pos)
        with torch.no_grad():
            c_t, s_t = rot(torch.zeros(1, T, 8, dtype=torch.float32), torch.from_numpy(rp)[:, None, :])
        uc, nc = ulps(cos_np[0], c_t[0].numpy())
        us, ns = ulps(sin_np[0], s_t[0].numpy())
        res_cs.append({"id": rid, "arm": arm, "T": T, "cos_max_ulps": uc, "cos_elements_differing": nc,
                       "sin_max_ulps": us, "sin_elements_differing": ns, "elements": int(cos_np[0].size)})
    summary = {
        "pixels": {"runs": len(res_pix), "bit_equal": sum(x["bit_equal"] for x in res_pix),
                   "max_abs_diff": max(x["max_abs_diff"] for x in res_pix),
                   "by_arm": {a: f"{sum(x['bit_equal'] for x in res_pix if x['arm'] == a)}/"
                                 f"{sum(x['arm'] == a for x in res_pix)}" for a in RH.TILES}},
        "ids": {"runs": len(res_ids), "all_equal": sum(x["ids_equal"] and x["token_offset_equal"] and x["spans_equal"]
                                                      for x in res_ids)},
        "planes": {"runs": len(res_pos), "equal": sum(x["planes_equal"] for x in res_pos),
                   "by_arm": {a: f"{sum(x['planes_equal'] for x in res_pos if x['arm'] == a)}/"
                                 f"{sum(x['arm'] == a for x in res_pos)}" for a in ("g256", "g448", "native")}},
        "cos_sin": {"runs": len(res_cs), "max_ulps": max(max(x["cos_max_ulps"], x["sin_max_ulps"]) for x in res_cs),
                    "elements": sum(x["elements"] for x in res_cs),
                    "cos_differing": sum(x["cos_elements_differing"] for x in res_cs),
                    "sin_differing": sum(x["sin_elements_differing"] for x in res_cs)},
    }
    summary["pass"] = {"pixels_bit_equal_28": summary["pixels"]["bit_equal"] == 28,
                       "ids_equal_42": summary["ids"]["all_equal"] == 42,
                       "planes_equal_42": summary["planes"]["equal"] == 42,
                       "cos_sin_le_1ulp": summary["cos_sin"]["max_ulps"] <= 1}
    doc = {"what": "round 3: host pixels / ids / M-RoPE planes / cos-sin of the reference image runs (no model weights)",
           "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pillow": PIL.__version__,
                        "torch": torch.__version__, "transformers": transformers.__version__,
                        "tokenizers": __import__("tokenizers").__version__},
           "host_py_sha256": sha256_file(K / "scripts/r3_host.py"), "clef_host_py_sha256": sha256_file(K / "scripts/clef_host.py"),
           "reference": {"file": "ref/records_ref.json", "rows": len(rows)}, "seconds": round(time.time() - t0, 1),
           "summary": summary, "pixels": res_pix, "ids": res_ids, "planes": res_pos, "cos_sin": res_cs}
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0 if all(summary["pass"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
