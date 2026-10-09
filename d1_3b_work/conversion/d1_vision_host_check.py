"""Round 3 acceptance 1-3: the host's picture preprocessing, token ids and position table against transformers 5.14.1.

    $REF scripts/d1_vision_host_check.py

1. Pictures: seed-fixed synthetic RGB pictures drawn with Pillow (gradient, shapes, a noise band; no third-party
   image), CASES below (single tile up / exact / down, split with thumbnail in both orientations, cap_pixels active,
   a thumbnail whose patch grid is below 16 on one side, extreme aspect ratios, 1 x 1). Saved as PNG under
   cache/vision/ (never committed); the raw RGB bytes' sha256 is recorded.
2. Preprocessing (acceptance 1): the provider's `cap_pixels` and the host's give the same picture; then the HF
   processor (`AutoProcessor.from_pretrained(hf_small)`, as `SystemOne._image_inputs` calls it, before the provider's
   cut) and `Lfm2VlImageProcessor(..., return_row_col_info=True)` vs `host/d1_vision.py preprocess`: pixel_values
   (float32 bit patterns), pixel_attention_mask, spatial_shapes, image_rows / image_cols / image_sizes, all equal.
   Also PyTorch's uint8 bicubic antialias resize vs the host's port on random sizes (RESIZE_RANDOM cases).
3. Token ids (acceptance 2): rows of fixture questions with 1 and 2 pictures: the provider's render with
   `SystemOne._image_markup(n)` through the processor (add_special_tokens=False) vs the host's render +
   `row_ids`; the texts are compared too.
4. Position table (acceptance 3): `resize_positions` vs `F.interpolate(bilinear, align_corners=False,
   antialias=True)` for h, w in {2, 4, ..., 64} and four wide grids, on the tiny model's table [16, 16, 32] and a
   seed-fixed N(0, 1) table [16, 16, 1152]; and `tower_inputs` vs `Siglip2VisionEmbeddings.resize_positional_embeddings`
   for every tile of the pictures (pad rows included).
Outputs: results/vision_host_check.json, cache/vision/img_<case>.png, cache/vision/proc_<case>.npz (the processor's
tensors, for the tower check).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

K = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K / "scripts"))
sys.path.insert(0, str(K / "host"))
import d1_litert as H  # noqa: E402
import d1_vision as V  # noqa: E402
from d1_common import HF_SMALL, engine_settings, load_fixtures, load_tokenizer, provider, render_row  # noqa: E402

OUT = K / "results/vision_host_check.json"
IMG_DIR = K / "cache/vision"
CASES = [  # name, width, height, why
    ("s100x60", 100, 60, "single tile, upscaled (64-token floor)"),
    ("s60x100", 60, 100, "rotated"),
    ("s512x512", 512, 512, "single tile, no resize"),
    ("s700x500", 700, 500, "single tile, shrunk"),
    ("s640x480", 640, 480, "single tile, shrunk"),
    ("s1280x853", 1280, 853, "cap_pixels, 2 x 3 tiles + thumbnail"),
    ("s853x1280", 853, 1280, "rotated"),
    ("s600x1600", 600, 1600, "tall, 5 x 2 tiles + thumbnail"),
    ("s1600x600", 1600, 600, "rotated"),
    ("s4000x800", 4000, 800, "cap_pixels, 1 x 5 tiles, thumbnail grid 14 x 70 (shrinks the position table)"),
    ("s800x4000", 800, 4000, "rotated"),
    ("s1100x520", 1100, 520, "1 x 2 tiles + thumbnail (fits a 1024-token row)"),
    ("s1024x1024", 1024, 1024, "cap boundary (not shrunk), 2 x 2 tiles"),
    ("s1025x1025", 1025, 1025, "cap_pixels just above the boundary"),
    ("s3000x20", 3000, 20, "extreme aspect, single tile 2 x 188 patches"),
    ("s20x3000", 20, 3000, "rotated"),
    ("s1x1", 1, 1, "one pixel"),
    ("s333x777", 333, 777, "odd sizes"),
]
PAIRS = [("s100x60", "s512x512"), ("s512x512", "s100x60"), ("s1280x853", "s600x1600"), ("s4000x800", "s1x1"),
         ("s1100x520", "s100x60")]
QUESTION_RECORDS = [("card_cats_001", "cats"), ("tv4_000", None), ("tv4x_qnli_07", None), ("tv4s_00", None)]
RESIZE_RANDOM = 300
WIDE = [(8, 128), (4, 256), (128, 8), (256, 4)]


def make_image(name: str, w: int, h: int) -> Image.Image:
    seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 255, w, dtype=np.float64)[None, :]
    y = np.linspace(0, 255, h, dtype=np.float64)[:, None]
    arr = np.empty((h, w, 3), np.uint8)
    arr[..., 0] = np.clip(x + 0 * y, 0, 255).astype(np.uint8)
    arr[..., 1] = np.clip(0 * x + y, 0, 255).astype(np.uint8)
    arr[..., 2] = np.clip(255 - (x + y) / 2, 0, 255).astype(np.uint8)
    band = slice(int(h * 0.7), max(int(h * 0.7) + 1, int(h * 0.85)))
    arr[band] = rng.integers(0, 256, arr[band].shape, dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    d = ImageDraw.Draw(img)
    d.ellipse([w * 0.1, h * 0.1, w * 0.45, h * 0.5], fill=(240, 30, 30), outline=(0, 0, 0))
    d.rectangle([w * 0.55, h * 0.15, w * 0.9, h * 0.45], fill=(20, 200, 60))
    d.line([0, h - 1, w - 1, 0], fill=(255, 255, 255), width=max(1, min(w, h) // 50))
    return img


def plain(x):
    """Tensors / arrays (and lists of them) -> Python values for the json."""
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    return x


def bits_equal(a: np.ndarray, b: np.ndarray) -> bool:
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    return bool(np.array_equal(a.view(np.uint8), b.view(np.uint8)))


def contract_image_ids(tok) -> dict:
    ids = {"image": tok.single("<image>"), "image_start": tok.single("<|image_start|>"),
           "image_end": tok.single("<|image_end|>"), "img_thumbnail": tok.single("<|img_thumbnail|>"),
           "img_row_col": {f"{r},{c}": tok.single(f"<|img_row_{r}_col_{c}|>") for r in range(1, 11) for c in range(1, 11)}}
    assert all(v is not None for v in ids["img_row_col"].values()) and None not in ids.values(), ids
    return ids


def main() -> int:
    from transformers import AutoProcessor
    from transformers.models.siglip2.modeling_siglip2 import Siglip2VisionEmbeddings

    assert not OUT.exists(), f"refusing to overwrite {OUT}"
    t0 = time.time()
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    runner, prompt = provider("runner"), provider("prompt")
    proc = AutoProcessor.from_pretrained(str(HF_SMALL))
    ip = proc.image_processor
    tok_hf = load_tokenizer()
    tok = H.D1Tokenizer(HF_SMALL / H.TOKENIZER_FILE)
    ids = contract_image_ids(tok)
    doc = {"what": "round 3 acceptance 1-3: host picture path vs transformers 5.14.1", "processor": type(proc).__name__,
           "image_processor": type(ip).__name__, "backend": getattr(ip, "backend", None), "resample": int(ip.resample),
           "torch": torch.__version__, "cpu_capability": torch.backends.cpu.get_cpu_capability(),
           "image_token_ids": {k: v for k, v in ids.items() if k != "img_row_col"},
           "img_row_col_ids": {"first": ids["img_row_col"]["1,1"], "last": ids["img_row_col"]["10,10"],
                               "rule_row_major": all(ids["img_row_col"][f"{r},{c}"] == ids["img_row_col"]["1,1"] + (r - 1) * 10 + c - 1
                                                     for r in range(1, 11) for c in range(1, 11))}}

    # ---- 1/2. pictures and preprocessing
    pics, caps, cases = {}, {}, []
    for name, w, h, why in CASES:
        img = make_image(name, w, h)
        png = IMG_DIR / f"img_{name}.png"
        img.save(png)
        back = Image.open(png)
        assert np.array_equal(np.asarray(back), np.asarray(img)), name
        cap_p, cap_h = runner.cap_pixels(back), V.cap_pixels(back)
        cap_equal = cap_p.size == cap_h.size and np.array_equal(np.asarray(cap_p), np.asarray(cap_h))
        hf = proc(text=[V.IMAGE_MARKER], images=[[cap_p]], return_tensors="pt", add_special_tokens=False)
        info = ip([cap_p], return_row_col_info=True, return_tensors="pt")
        pic = V.preprocess(cap_h)
        pics[name], caps[name] = pic, cap_h
        pv = np.stack([t.pixels for t in pic.tiles])
        mask = np.stack([t.mask for t in pic.tiles])
        spatial = np.array([list(t.grid) for t in pic.tiles], dtype=np.int64)
        np.savez_compressed(IMG_DIR / f"proc_{name}.npz", pixel_values=hf["pixel_values"].numpy(),
                 pixel_attention_mask=hf["pixel_attention_mask"].numpy(), spatial_shapes=hf["spatial_shapes"].numpy())
        row = {"case": name, "why": why, "input_wh": [w, h], "capped_wh": list(cap_h.size),
               "rgb_sha256": hashlib.sha256(np.asarray(back).tobytes()).hexdigest()[:16],
               "cap_pixels_equal": cap_equal, "rows_cols": [pic.rows, pic.cols],
               "tiles": len(pic.tiles), "tile_grids": sorted({tuple(t.grid) for t in pic.tiles[:-1]}) if len(pic.tiles) > 1 else [],
               "thumbnail_or_single_hw": list(pic.size), "thumbnail_or_single_grid": list(pic.tiles[-1].grid),
               "image_tokens": V.tokens_for_size(pic.size) + (pic.rows * pic.cols * 256 if len(pic.tiles) > 1 else 0),
               "pixel_values_bits_equal": bits_equal(hf["pixel_values"].numpy(), pv),
               "pixel_values_shape": list(hf["pixel_values"].shape),
               "pixel_attention_mask_equal": bits_equal(hf["pixel_attention_mask"].numpy(), mask),
               "spatial_shapes_equal": bits_equal(hf["spatial_shapes"].numpy(), spatial),
               "image_rows_cols_sizes_equal": (plain(info["image_rows"]) == [pic.rows] and plain(info["image_cols"]) == [pic.cols]
                                               and plain(info["image_sizes"]) == [list(pic.size)]),
               "hf_rows_cols_sizes": [plain(info["image_rows"]), plain(info["image_cols"]), plain(info["image_sizes"])]}
        row["pass"] = all(row[k] for k in ("cap_pixels_equal", "pixel_values_bits_equal", "pixel_attention_mask_equal",
                                           "spatial_shapes_equal", "image_rows_cols_sizes_equal"))
        cases.append(row)
    doc["preprocess"] = cases
    rng = np.random.default_rng(11)
    bad = []
    for _ in range(RESIZE_RANDOM):
        h, w = int(rng.integers(1, 1400)), int(rng.integers(1, 1400))
        oh, ow = int(rng.integers(1, 1700)), int(rng.integers(1, 1700))
        a = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        t = torch.from_numpy(a.copy()).permute(2, 0, 1)[None]
        ref = F.interpolate(t, size=(oh, ow), mode="bicubic", align_corners=False, antialias=True)[0].permute(1, 2, 0).numpy()
        if not np.array_equal(ref, V.resize_uint8_bicubic_aa(a, oh, ow)):
            bad.append([h, w, oh, ow])
    doc["resize_random"] = {"cases": RESIZE_RANDOM, "sizes": "h, w in 1..1399 -> 1..1699, uint8 RGB noise, seed 11",
                            "mismatch": bad, "pass": not bad}

    # ---- 3. token ids
    settings = engine_settings(tok_hf, prompt)
    fixtures = {r["id"]: r for r in load_fixtures()["records"]}
    dummy = type("Dummy", (), {"processor": proc})()
    markup = {n: runner.SystemOne._image_markup(dummy, n) for n in (1, 2, 3)}
    doc["image_markup"] = {str(n): {"provider": m, "host": V.image_markup(n), "equal": m == V.image_markup(n)}
                           for n, m in markup.items()}
    token_rows = []
    sets = [[c[0]] for c in CASES] + [list(p) for p in PAIRS]
    for names in sets:
        for rid, qname in QUESTION_RECORDS:
            rec = fixtures[rid]
            state = rec["request"].get("state")
            qn = qname or next(iter(rec["request"]["questions"]))
            qd = rec["request"]["questions"][qn]
            q = prompt.as_question(qd)
            text = render_row(prompt, tok_hf, settings, state, q, images=markup[len(names)])
            ref = proc(text=[text], images=[[caps[n] for n in names]], return_tensors="pt",
                       add_special_tokens=False)["input_ids"][0].tolist()
            hq = H.as_question(qd)
            host_text = H.prefix_text(state, V.image_markup(len(names))) + H.suffix_text(tok, hq)
            host_ids = V.row_ids(tok.encode, host_text, [pics[n] for n in names], ids)
            token_rows.append({"pictures": names, "record": rid, "question": qn, "type": qd.get("type", "choice"),
                               "tokens": len(ref), "image_positions": sum(i == ids["image"] for i in ref),
                               "text_equal": host_text == text, "ids_equal": host_ids == ref})
    doc["token_ids"] = {"rows": token_rows, "pass": all(r["text_equal"] and r["ids_equal"] for r in token_rows)}

    # ---- 4. position table
    from safetensors.numpy import load_file

    tiny = load_file(str(K / "cache/vision/tiny_vl_seed0.safetensors"))
    tables = {"tiny_16x16x32": tiny["model.vision_tower.vision_model.embeddings.position_embedding.weight"].reshape(16, 16, -1),
              "random_16x16x1152": np.random.default_rng(5).standard_normal((16, 16, 1152)).astype(np.float32)}
    grids = [(h, w) for h in range(2, 65, 2) for w in range(2, 65, 2)] + WIDE
    pos_rows = {}
    for tname, table in tables.items():
        tt = torch.from_numpy(np.ascontiguousarray(table)).permute(2, 0, 1)[None]
        worst, nbits, per_wide = 0.0, 0, {}
        for (h, w) in grids:
            ref = F.interpolate(tt, size=(h, w), mode="bilinear", align_corners=False, antialias=True)
            ref = ref.reshape(table.shape[2], h * w).transpose(0, 1).numpy()
            mine = V.resize_positions(table, h, w)
            d = float(np.abs(ref.astype(np.float64) - mine).max())
            worst = max(worst, d)
            nbits += bits_equal(ref, mine)
            if (h, w) in WIDE:
                per_wide[f"{h}x{w}"] = d
        pos_rows[tname] = {"grids": len(grids), "max_abs": worst, "bits_equal": nbits, "wide_max_abs": per_wide,
                           "pass": worst <= 1e-6}
    # the transformers function itself on every tile of the pictures (pad rows = row 0, as it fills them)
    tw, tbits, tworst = 0, 0, 0.0
    for name, pic in pics.items():
        for tile in pic.tiles:
            ss = torch.tensor([list(tile.grid)], dtype=torch.long)
            for tname, table in tables.items():
                ref = Siglip2VisionEmbeddings.resize_positional_embeddings(
                    torch.from_numpy(np.ascontiguousarray(table)), ss, max_length=V.SETTINGS.max_num_patches)[0].numpy()
                mine = V.tower_inputs(tile, table)["pos"][0]
                tw += 1
                tbits += bits_equal(ref, mine)
                tworst = max(tworst, float(np.abs(ref.astype(np.float64) - mine).max()))
    pos_rows["transformers_resize_positional_embeddings_on_tiles"] = {"tiles_x_tables": tw, "bits_equal": tbits,
                                                                       "max_abs": tworst, "pass": tworst <= 1e-6}
    doc["position_table"] = pos_rows
    doc["summary"] = {
        "preprocess_cases": len(cases), "preprocess_pass": sum(r["pass"] for r in cases),
        "resize_random_pass": doc["resize_random"]["pass"],
        "token_rows": len(token_rows), "token_rows_pass": sum(r["text_equal"] and r["ids_equal"] for r in token_rows),
        "markup_pass": all(v["equal"] for v in doc["image_markup"].values()),
        "position_max_abs": max(v["max_abs"] for v in pos_rows.values()),
        "pass": (all(r["pass"] for r in cases) and doc["resize_random"]["pass"] and doc["token_ids"]["pass"]
                 and all(v["equal"] for v in doc["image_markup"].values()) and all(v["pass"] for v in pos_rows.values()))}
    doc["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(doc["summary"], indent=1))
    for r in cases:
        print(r["case"], r["capped_wh"], r["rows_cols"], r["tiles"], r["thumbnail_or_single_hw"],
              r["thumbnail_or_single_grid"], r["image_tokens"], "PASS" if r["pass"] else "FAIL")
    return 0 if doc["summary"]["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
