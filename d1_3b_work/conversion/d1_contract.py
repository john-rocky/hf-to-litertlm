"""Round 2 acceptance 7 / round 3 acceptance 9: host/contract.json, the contract between the d1-3B graphs and a host
(any language).

    $EXPORT scripts/d1_contract.py [--replace]

Read from the files, never typed in: token ids from fixtures/token_probes.json (the provider's tokenizer) and checked
again with the host tokenizer (tokenizer.json through `tokenizers`); the read-out id set = every id the host's
`readout_ids` can produce for an ASCII request: the yes / no forms, the digits 0-9, and for every code the provider can
assign (A..Z, a..z, 00..99 and the fallback pool #0..#199, AA..ZZ) the code's id and the id of " " + code when those
are single tokens (`aliases` takes only single-token codes); bucket candidates and how many fixture rows each holds
(fixtures/rows.json); the graph I/O as the tiny exports declare it (results/tiny_export_*.json: names, dtypes, rank;
d = 2048 on d1-3B). The graph files and the read-out table file are cut in round 3 (they need the weights), so
`graphs` is empty here and `readout_table.file` names the file the host will load.
Round 3 adds `vision` (`vision_section`): the picture token ids (every `<|img_row_r_col_c|>` read from the
tokenizer), the preprocessing numbers (read from processor_config.json / config.json), the two picture graphs' I/O
(the checkpoint's sizes, checked against the tiny exports' signatures), the position table file, the unshuffle and
insertion rules, the limits, and the tiny verification records. The prose version is host/VISION_CONTRACT.md.
Round 5 adds `tables` (`tables_section`: the two table files the host loads, their keys / shapes / dtypes, and the
checkpoint tensor each is cut from, read from the Hub header evidence; scripts/d1_tables.py cuts them) and
`storage_variants` (`storage_section`: v1 / v2 / v3 as scripts/d1_storage.py builds and checks them, the recipe read
from its build_recipe, the tiny verification records).
Round 6c fills `graphs` (`files_section`): per bucket the v2 file (fp16 FC + int8 table) with bytes and sha256 from
its quantization record (--rehash: hashed again and compared), the I/O from the float32 export's record (the gate run's
input / output names compared), and the Mac gate of each accelerator (the check jsons: verdict, questions, max / mean
|dp|, argmax, red arms); `embeds_graph` (the embeds variant v2e, its host rule and gate); `readout_table` and
`tables.files` get bytes and sha256, and `tables.files` the bfloat16 embed_table.safetensors. --out <file> writes
elsewhere (a dry run to diff against host/contract.json); --ledger results/real_files.json writes the shipping
candidates into the ledger next to its `deleted` list.
    $EXPORT scripts/d1_contract.py --replace --rehash --ledger results/real_files.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

K = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K / "host"))
sys.path.insert(0, str(K / "scripts"))
import d1_litert as H  # noqa: E402
from d1_common import HF_SMALL, PROBES, REV, ROWS, sha256_file  # noqa: E402

OUT = K / "host/contract.json"
LAUNCH_IDS = {"bos": 124894, "pad": 124893, "im_start": 124899, "im_end": 124900, "image": 124907}
SPECIAL_TEXT = {"bos": "<|startoftext|>", "im_start": "<|im_start|>", "im_end": "<|im_end|>", "image": "<image>",
                "image_start": "<|image_start|>", "image_end": "<|image_end|>", "img_thumbnail": "<|img_thumbnail|>",
                "img_row_1_col_1": "<|img_row_1_col_1|>"}
CANDIDATES = (64, 128, 256, 512, 1024, 2048, 4096)


def vision_section(tok, token_ids: dict) -> dict:
    """The picture path, read from the files (never typed in)."""
    pc = json.loads((HF_SMALL / "processor_config.json").read_text())["image_processor"]
    cfg = json.loads((HF_SMALL / "config.json").read_text())
    vc = cfg["vision_config"]
    rc = {f"{r},{c}": tok.single(f"<|img_row_{r}_col_{c}|>") for r in range(1, 11) for c in range(1, 11)}
    assert all(v is not None for v in rc.values()), rc
    assert all(rc[f"{r},{c}"] == rc["1,1"] + (r - 1) * 10 + (c - 1) for r in range(1, 11) for c in range(1, 11))
    assert token_ids["image"] == cfg["image_token_id"]
    hid, d, side = vc["hidden_size"], cfg["text_config"]["hidden_size"], int(round(vc["num_patches"] ** 0.5))
    n, tile_tokens = pc["max_num_patches"], ((pc["tile_size"] // pc["encoder_patch_size"]) // pc["downsample_factor"]) ** 2
    exports = {}
    for stem in ("tiny_vision_tower_fp32", "tiny_projector_fp32"):
        w = json.loads((K / f"results/{stem}_tflite.json").read_text())
        exports[stem] = {"signature": w["signatures"][0], "sha256": w["sha256"], "operator_count": w["operator_count"],
                         "watched_ops": w["watched_ops"]}
    p, f = pc["encoder_patch_size"], pc["downsample_factor"]
    return {
        "status": "round 3: rules final, verified end to end on the tiny VL model (host/test_d1_vision_tiny.py); the "
                  "real graph files and the position table are cut from the checkpoint later",
        "prose": "host/VISION_CONTRACT.md",
        "token_ids": {"image": token_ids["image"], "image_start": token_ids["image_start"],
                      "image_end": token_ids["image_end"], "img_thumbnail": token_ids["img_thumbnail"],
                      "img_row_col_rule": f"<|img_row_r_col_c|> = {rc['1,1']} + (r - 1) * 10 + (c - 1), r, c in 1..10",
                      "img_row_col": rc},
        "markup": "one '<image>' per picture at the head of the user turn, right after '<|im_start|>user\\n' (what the "
                  "chat template writes; SystemOne._image_markup)",
        "token_sequence": f"per picture: <|image_start|>; split: per tile (row-major) <|img_row_r_col_c|> + "
                          f"{tile_tokens} x <image>, then <|img_thumbnail|> + n x <image>; single tile: n x <image>; "
                          f"then <|image_end|>; n = ceil(h / {p} / {f}) * ceil(w / {p} / {f}) for the thumbnail / "
                          f"single-tile size (h, w)",
        "preprocessing": {
            "cap_pixels": {"max_pixels": 1024 * 1024, "resample": "Pillow BICUBIC",
                           "size": "(max(1, int(w * s)), max(1, int(h * s))), s = sqrt(max_pixels / (w * h))",
                           "source": "runner.cap_pixels (VISION_MAX_PIXELS)"},
            "processor_config": {k: pc[k] for k in ("encoder_patch_size", "downsample_factor", "tile_size", "min_tiles",
                                                    "max_tiles", "min_image_tokens", "max_image_tokens",
                                                    "max_pixels_tolerance", "max_num_patches", "use_thumbnail",
                                                    "do_image_splitting", "image_mean", "image_std", "rescale_factor",
                                                    "resample")},
            "resize": "torchvision resize(uint8, BICUBIC, antialias=True) on the CPU = PyTorch's uint8 separable "
                      "antialias kernel (Keys cubic a = -0.5, float64 weights normalised and scaled to int16 at the "
                      "largest precision p below 2^15, integer sum + 2^(p-1) >> p clamped to 0..255; width first, "
                      "then height; an unchanged side is skipped): host/d1_vision.py resize_uint8_bicubic_aa",
            "normalise": "float32 (x - 127.5) / 127.5",
            "patches": f"{p} x {p} patches in raster order, each flattened (row, column, channel) = {3 * p * p} "
                       f"values, zero patches after the real ones up to {n}",
            "verified": "results/vision_host_check.json (18 pictures, pixel_values bit-equal to Lfm2VlProcessor)"},
        "graphs": {
            "tower": {"inputs": [{"name": "pixels", "dtype": "float32", "shape": [1, n, 3 * p * p]},
                                 {"name": "pos", "dtype": "float32", "shape": [1, n, hid],
                                  "meaning": "position table resized to the tile's patch grid (host); rows past the "
                                             "grid repeat row 0 (any value works: they are masked)"},
                                 {"name": "mask", "dtype": "float32", "shape": [1, n],
                                  "meaning": "1.0 real patch, 0.0 padding"}],
                      "outputs": [{"name": "features", "dtype": "float32", "shape": [1, n, hid],
                                   "meaning": "post_layernorm output; the first h * w rows are the tile's patches"}],
                      "runs": "once per tile (grid tiles + thumbnail, or the single tile)"},
            "projector": {"inputs": [{"name": "soft", "dtype": "float32", "shape": [1, tile_tokens, 4 * hid],
                                      "meaning": "the unshuffled cells in the first (h/2)(w/2) rows, zeros after"}],
                          "outputs": [{"name": "mm", "dtype": "float32", "shape": [1, tile_tokens, d],
                                       "meaning": "the first (h/2)(w/2) rows are the tile's picture tokens"}]},
            "verified_on_tiny_exports": exports},
        "position_table": {"file": "vision_position_table.safetensors",
                           "keys": {"table": f"float32 [{side}, {side}, {hid}]"},
                           "source": f"model.vision_tower.vision_model.embeddings.position_embedding.weight "
                                     f"[{vc['num_patches']}, {hid}] (bfloat16 -> float32) reshaped",
                           "resize": "bilinear, antialias, align_corners False, float32 (host/d1_vision.py "
                                     "resize_positions; bit-equal to F.interpolate on the Mac)"},
        "unshuffle": "features[:h * w] as (h, w, H) -> (h/2, w/2, 4H): channel j * 2H + k * H + c of cell (r, q) = "
                     "input (2r + j, 2q + k, c); cells in raster order (Lfm2VlMultiModalProjector.pixel_unshuffle)",
        "insertion": "the row's embeddings = the text table's rows (tied embed_tokens) at every non-<image> position; "
                     "the k-th <image> position takes the k-th picture token, tiles in order (grid row-major, then the "
                     "thumbnail), pictures in order (get_placeholder_mask + masked_scatter); the counts must agree; "
                     "the row then goes to the embeds variant of the row graph",
        "limits": {"patches_per_tile": n, "tokens_per_tile": tile_tokens,
                   "tiles_per_picture": f"1, or 2..{pc['max_tiles']} + thumbnail",
                   "picture_tokens_max": pc["max_tiles"] * tile_tokens + tile_tokens,
                   "text_rows_needed": "every text id of a picture row (the host needs the tied table's rows, not only "
                                       "the read-out rows)"},
        "verified": {"host": "results/vision_host_check.json", "torch": "results/tiny_vision_torch_check.json",
                     "litert": "results/tiny_vision_tower_fp32_*_check.json, results/tiny_projector_fp32_*_check.json",
                     "end_to_end": "results/tiny_vision_e2e_{torch,cpu,gpu,gpu_default}.json",
                     "header": "results/vision_header_check.json"},
    }


def tables_section(readout_ids: list[int]) -> dict:
    """The host's two table files, from the Hub header evidence and config.json (never typed in)."""
    import d1_tables as TB

    ev = json.loads(TB.EVIDENCE.read_text())
    cfg = json.loads((HF_SMALL / "config.json").read_text())
    d, c = cfg["text_config"]["hidden_size"], cfg["vision_config"]["hidden_size"]
    side = int(round(cfg["vision_config"]["num_patches"] ** 0.5))
    src = {name: {"tensor": name, **ev["header"][name]} for name in (TB.READOUT, TB.POSITION)}
    check = json.loads((K / "results/tables_header_check.json").read_text())
    return {
        "cut_by": "scripts/d1_tables.py <snapshot dir> --out <dir> (reads only the needed bytes; bfloat16 -> float32 "
                  "exact: the 16 bits become the high half)",
        "files": {
            "readout_table.safetensors": {
                "keys": {"ids": {"dtype": "int64", "shape": [len(readout_ids)]},
                         "rows": {"dtype": "float32", "shape": [len(readout_ids), d]}},
                "ids": "readout.table.ids (ascending)", "bytes_rows": len(readout_ids) * d * 4,
                "source": src[TB.READOUT], "bytes_read_from_checkpoint": len(readout_ids) * d * 2},
            "vision_position_table.safetensors": {
                "keys": {"table": {"dtype": "float32", "shape": [side, side, c]}},
                "source": src[TB.POSITION], "bytes_read_from_checkpoint": side * side * c * 2}},
        "header_evidence": {"file": str(TB.EVIDENCE.relative_to(K)), "url": ev["url"], "data_start": 8 + ev["header_bytes"],
                            "check": "results/tables_header_check.json", "check_pass": check["pass"]},
        "file_format": "safetensors without __metadata__ (the sha256 depends on the rows only; provenance in the "
                       "cut's tables_manifest.json)",
        "verified": {"tiny_cut": ["results/tables_cut_tinyvl.json", "results/tables_cut_tinyvl_verify.json (same files, "
                                  "bit-equal to safetensors + torch)", "results/tables_cut_tinyvl_bf16.json (bf16 "
                                  "source, bit-equal to safetensors + torch)", "results/tables_cut_tinyr2.json"],
                     "end_to_end": ["results/tiny_host_e2e_cpu_cut.json", "results/tiny_vision_e2e_cpu_cut.json"]},
        "not_cut": "the full tied table for a picture row's text ids (vision.limits.text_rows_needed): not a file yet",
    }


def storage_section() -> dict:
    """v1 / v2 / v3 as scripts/d1_storage.py builds and checks them."""
    import d1_storage as ST

    meaning = {
        "v1": {"fc": "INT8 weights read by the FC itself (dynamic int8, channelwise, symmetric, integer compute: the "
                     "activations are quantized per call), no DEQUANTIZE", "table": "INT8, one scale per row",
               "for": "the S26 CPU (round 4 estimate: ~2.7 GB, the one form predicted to fit); its option probabilities "
                      "move with the activation quantization, so it goes to a phone only after the |dp| gate on the "
                      "real weights"},
        "v2": {"fc": "FLOAT16 weights through DEQUANTIZE (weight-only FLOAT_CASTING)", "table": "INT8, one scale per row",
               "for": "the GPU candidate (Kev's shipped form); XNNPACK unpacks the fp16 weights to fp32 on a CPU"},
        "v3": {"fc": "FLOAT32 weights", "table": "INT8, one scale per row",
               "for": "the float32 reference a GPU accepts (the in-graph float table is not)"},
        "v2e": {"fc": "FLOAT16 weights through DEQUANTIZE (weight-only FLOAT_CASTING)",
                "table": "none in the graph (D1PrefillEmbeds): the host writes the float32 rows of the bfloat16 table "
                         "embed_table.safetensors",
                "for": "round 6c: the embeds variant (no int8 table error; the picture row's entrance)"},
    }
    out = {}
    for v, name in ST.VARIANTS.items():
        recipe, need_cal = ST.build_recipe(v)
        tiny = K / f"results/tiny_rowprefill_L64_{name}_quant.json"
        rec = json.loads(tiny.read_text()) if tiny.exists() else None
        out[v] = {"name": name, "file": f"<stem>_{name}.tflite (from <stem>_fp32.tflite)", **meaning[v],
                  "expect": ST.EXPECT[v], "recipe": json.loads(json.dumps(recipe, default=str)),
                  "needs_calibration": bool(need_cal),
                  "verified_on_tiny": None if rec is None else {"file": str(tiny.relative_to(K)), "checks": rec["checks"],
                                                                "checks_pass": rec["checks_pass"],
                                                                "bytes": rec["output"]["bytes"]}}
    out["not_made"] = "weight-only int8 (DEQUANTIZE -> float FC): CPU-only and refused by the GPU, not a shipping form"
    out["built_by"] = ("scripts/d1_storage.py --tflite <..._fp32.tflite> --variant v1|v2|v3|v2e (ai-edge-quantizer 0.9.0, "
                       "no calibration)")
    return out


BUCKETS_REAL = (256, 512, 1024, 2048, 4096)
ACCELS = ("cpu", "gpu_f32", "gpu_default")


def gate_line(path: Path) -> dict:
    """One gate run (results/<stem>_<accel>_check.json, `reference_parity`) in one line."""
    d = json.loads(path.read_text())
    rp = (d.get("reference_parity") or {}).get("summary") or {}
    arms = rp.get("red_arms") or {}
    return {"check": str(path.relative_to(K)), "status": d["status"], "accel": d["accel"],
            "questions": rp.get("questions_run", rp.get("questions_fit_L")), "questions_fit_L": rp.get("questions_fit_L"),
            "non_near_tie_argmax": rp.get("non_near_tie_argmax"), "near_tie_argmax": rp.get("near_tie_argmax"),
            "max_abs_dp": rp.get("max_abs_dp"), "mean_abs_dp": rp.get("mean_abs_dp_all_options"),
            "p95_abs_dp": rp.get("p95_abs_dp_all_options"), "questions_over_0_02": len(rp.get("rows_over_max_dp") or []),
            "red_arms": f"{arms.get('red')}/{arms.get('in_reference')}",
            "delegated": [f"{r['delegated']}/{r['total']} {r['delegate']}" for r in d["delegation"]["replacing"]],
            "nonfinite_rows": len(rp.get("nonfinite") or []),
            "verdict": "PASS" if (rp.get("bar_near_tie_apart") and arms.get("all_red")) else "FAIL"}


def shipped_file(stem: str, rehash: bool) -> dict | None:
    """exports/<stem>.tflite with its quantization record (bytes, sha256), or None when the file is not there."""
    f, q = K / f"exports/{stem}.tflite", K / f"results/{stem}_quant.json"
    if not (f.exists() and q.exists()):
        return None
    rec = json.loads(q.read_text())["output"]
    assert f.stat().st_size == rec["bytes"], (stem, f.stat().st_size, rec["bytes"])
    if rehash:
        assert sha256_file(f) == rec["sha256"], f"{stem}: the file's sha256 differs from {q.name}"
    return {"file": f.name, "bytes": rec["bytes"], "sha256": rec["sha256"], "record": str(q.relative_to(K)),
            "sha256_rehashed": rehash}


def io_of(export_record: Path, check: Path | None) -> dict:
    """The graph's signature from the float32 export's record; the storage variant keeps it (the gate run's input and
    output names are compared when a check json is given)."""
    sig = json.loads(export_record.read_text())["signatures"][0]
    io = {"signature": sig.get("key", "serving_default"),
          "inputs": [{"name": i["name"], "dtype": i["dtype"].lower(), "shape": i["shape"]} for i in sig["inputs"]],
          "outputs": [{"name": o["name"], "dtype": o["dtype"].lower(), "shape": o["shape"]} for o in sig["outputs"]],
          "from": str(export_record.relative_to(K))}
    if check is not None and check.exists():
        c = json.loads(check.read_text())
        assert sorted(c["inputs"]) == sorted(i["name"] for i in sig["inputs"]) and list(c["outputs"]) == ["hidden"], check
        io["names_checked_on"] = str(check.relative_to(K))
    return io


def files_section(rehash: bool) -> dict:
    """Round 6c: the shipping candidates of the text path (v2 per bucket, the embeds variant, the two text tables)."""
    graphs, missing = [], []
    for L in BUCKETS_REAL:
        stem = f"real_rowprefill_L{L}_v2_fp16fc_i8emb"
        f = shipped_file(stem, rehash)
        if f is None:
            missing.append(stem)
            continue
        checks = {acc: K / f"results/{stem}_{acc}_check.json" for acc in ACCELS}
        graphs.append({"L": L, **f, "form": "v2_fp16fc_i8emb (fp16 FC + int8 table)",
                       "io": io_of(K / f"results/real_export_L{L}.json", checks["gpu_f32"]),
                       "gate": {acc: gate_line(p) for acc, p in checks.items() if p.exists()},
                       "run_on": "Metal float32 (GpuOptions(enforce_f32=True)) or the CPU; the Metal default precision "
                                 "misses the bar (gate)"})
    emb = None
    stem = "real_rowprefill_embeds_L256_v2e_fp16fc"
    f = shipped_file(stem, rehash)
    if f is not None:
        checks = {acc: K / f"results/{stem}_{acc}_check.json" for acc in ACCELS}
        emb = {"L": 256, **f, "form": "v2e_fp16fc (fp16 FC, no table in the graph)",
               "io": io_of(K / "results/real_export_embeds_L256.json", checks["gpu_f32"]),
               "gate": {acc: gate_line(p) for acc, p in checks.items() if p.exists()},
               "host": "embeds = the float32 rows of embed_table.safetensors (bfloat16) at the row's ids, pads included "
                       "(scripts/d1_tables.py EmbedTable); valid and the read-out as for the ids graphs"}
    else:
        missing.append(stem)
    tm = json.loads((K / "cache/real/tables/tables_manifest.json").read_text())["files"]["readout_table.safetensors"]
    em = json.loads((K / "cache/real/tables/embed_table_manifest.json").read_text())["file"]
    tables = {"readout_table.safetensors": {"bytes": tm["bytes"], "sha256": tm["sha256"], "keys": tm["keys"],
                                            "manifest": "cache/real/tables/tables_manifest.json"},
              "embed_table.safetensors": {"bytes": em["bytes"], "sha256": em["sha256"], "keys": em["keys"],
                                          "manifest": "cache/real/tables/embed_table_manifest.json"}}
    for name, t in tables.items():
        p = K / "cache/real/tables" / name
        assert p.stat().st_size == t["bytes"], name
        if rehash:
            assert sha256_file(p) == t["sha256"], name
    return {"graphs": graphs, "embeds_graph": emb, "tables": tables, "missing": missing}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replace", action="store_true")
    ap.add_argument("--out", default="", help="write here instead of host/contract.json (a dry run to diff)")
    ap.add_argument("--rehash", action="store_true", help="hash the shipped files again (else: size + the records)")
    ap.add_argument("--ledger", default="", help="round 6c: also write the shipping candidates into this ledger json "
                                                 "(results/real_files.json), keeping its `deleted` list")
    ap.add_argument("--keep-vision", action="store_true",
                    help="take the `vision` section verbatim from the existing output file (round 6d owns it since its "
                         "real files: scripts/d1v_contract.py --vision-only writes it); this script then changes only the "
                         "text sections")
    a = ap.parse_args()
    out_path = (Path(a.out) if Path(a.out).is_absolute() else K / a.out) if a.out else OUT
    assert a.replace or not out_path.exists(), f"refusing to overwrite {out_path} (--replace)"
    kept_vision = json.loads(out_path.read_text())["vision"] if a.keep_vision else None
    probes = json.loads(PROBES.read_text())
    tok = H.D1Tokenizer(K / "hf_small" / H.TOKENIZER_FILE)
    by_text = {p["text"]: p for p in probes["probes"]}
    token_ids = {}
    for name, text in SPECIAL_TEXT.items():
        p = by_text[text]
        assert p["single"] and tok.single(text) == p["ids"][0], (name, p, tok.single(text))
        token_ids[name] = p["ids"][0]
    token_ids["pad"] = probes["tokenizer"]["pad"][1]
    assert probes["tokenizer"]["pad"][0] == "<|pad|>" and tok.single("<|pad|>") == token_ids["pad"]
    assert {k: token_ids[k] for k in LAUNCH_IDS} == LAUNCH_IDS, token_ids

    def singles(texts):
        return {t: tok.single(t) for t in texts if tok.single(t) is not None}

    letters = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    lower = [chr(c) for c in range(ord("a"), ord("z") + 1)]
    groups = {
        "noul_yes": [tok.single(t) for t in H.YES_FORMS if tok.single(t) is not None],
        "noul_no": [tok.single(t) for t in H.NO_FORMS if tok.single(t) is not None],
        "digits": singles([str(i) for i in range(10)]),
        "letters_upper": singles(letters), "letters_upper_space": singles([" " + c for c in letters]),
        "letters_lower": singles(lower), "letters_lower_space": singles([" " + c for c in lower]),
    }
    assert groups["noul_yes"] == [11683, 12447] and groups["noul_no"] == [2243, 4547, 19598], groups
    codes = list(dict.fromkeys(letters + lower + [f"{i:02d}" for i in range(100)] + list(H.FALLBACK_POOL)))
    ids = set(groups["noul_yes"]) | set(groups["noul_no"]) | set(groups["digits"].values())
    for c in codes:
        for t in (c, " " + c):
            tid = tok.single(t)
            if tid is not None:
                ids.add(tid)
    ids = sorted(ids)
    # every read-out id of the fixture is in the set
    rows = json.loads(ROWS.read_text())
    fixture_ids = sorted({i for r in rows["rows"] for g in r["readout_ids"] for i in g})
    assert set(fixture_ids) <= set(ids), sorted(set(fixture_ids) - set(ids))[:10]
    lens = [r["row_len"] for r in rows["rows"] if not r.get("image_expansion_pending")]
    smallest = {}
    for n in lens:
        L = next(c for c in CANDIDATES if c >= n)
        smallest[L] = smallest.get(L, 0) + 1
    exports = {p.stem: json.loads(p.read_text()) for p in sorted((K / "results").glob("tiny_export*_L*.json"))}
    verified = {k: {"signature": v["signatures"][0]["key"], "inputs": v["signatures"][0]["inputs"],
                    "outputs": v["signatures"][0]["outputs"], "sha256": v["sha256"]} for k, v in exports.items()}
    doc = {
        "what": "d1-3B on LiteRT: the contract between the row graphs and a host",
        "status": "round 2: token ids, read-out id set, graph I/O and padding rules are final for d1-3B; round 3: the "
                  "picture path (`vision`); round 5: the table files (`tables`) and the storage variants "
                  "(`storage_variants`); the graph files and the tables are cut from the checkpoint later (`graphs` "
                  "empty until then)",
        "model": {"repo": "LiquidAI/d1-3B", "revision": REV, "hidden_size": 2048, "embedding_rows": 128000,
                  "tokenizer_vocab": probes["tokenizer"]["len"], "tied_embeddings": True},
        "tokenizer": {"file": "tokenizer.json", "sha256": sha256_file(K / "hf_small/tokenizer.json"),
                      "encode": "tokenizer.encode(text, add_special_tokens=False); the BOS is part of the text"},
        "token_ids": token_ids,
        "graph_io": {
            "signature": H.SIGNATURE,
            "inputs": [{"name": "ids", "dtype": "int32", "shape": [1, "L"], "meaning": "row token ids, right-padded with token_ids.pad"},
                       {"name": "valid", "dtype": "float32", "shape": [1, "L"], "meaning": "1.0 on real tokens, 0.0 on padding"}],
            "outputs": [{"name": "hidden", "dtype": "float32", "shape": [1, "L", 2048],
                         "meaning": "hidden state after the final RMSNorm (embedding_norm), every position"}],
            "embeds_variant": {"inputs": [{"name": "embeds", "dtype": "float32", "shape": [1, "L", 2048]},
                                          {"name": "valid", "dtype": "float32", "shape": [1, "L"]}],
                               "outputs": "as above", "use": "rows whose embeddings the host supplies (pictures, host lookup)"},
            "positions": "0..L-1, constant in the graph (each row starts at position 0)",
            "attention": "causal; keys at padding masked by valid (they never reach a real token: right padding + causal)",
            "verified_on_tiny_exports": verified,
        },
        "row": {"padding": "right, with token_ids.pad, valid 0.0", "answer_slot": "the row's last real token (n - 1)",
                "bucket": "the smallest graph L that holds the row; a longer row is refused, never truncated",
                "render": "host/d1_litert.py docstring 2 (the provider's prompt.render with SystemOne's defaults)"},
        "buckets": {"candidates": list(CANDIDATES), "host_default": list(H.BUCKETS),
                    "fixture_rows_by_smallest_candidate": {str(k): smallest.get(k, 0) for k in CANDIDATES},
                    "fixture_rows": len(lens), "fixture_max_row": max(lens), "rows_json_sha256": sha256_file(ROWS)},
        "readout": {
            "formula": "logit(id) = hidden[answer_slot] . table[id] (float32); option score = max over its group; "
                       "probs = softmax over the options (float64); equals the provider's readout(log_softmax(lm_head(h)))",
            "groups": groups,
            "group_rules": "noul = [yes forms, no forms]; score level i = [id(str(i))], 2..10 levels; choice option = "
                           "[id(code)] + [id(' ' + code)] when single and different (codes: host/d1_litert.py aliases)",
            "table": {"file": "readout_table.safetensors", "keys": {"ids": "int64 [n]", "rows": "float32 [n, 2048]"},
                      "n_ids": len(ids), "shape": [len(ids), 2048], "bytes_float32": len(ids) * 2048 * 4,
                      "source": "rows of the tied embedding model.language_model.embed_tokens.weight (bfloat16 -> float32)",
                      "ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(), "ids": ids,
                      "covers": "every id readout_ids can give for ASCII option names (single-letter non-ASCII names "
                                "would need more rows: the host refuses a missing row with RequestError)",
                      "fixture_readout_ids": len(fixture_ids)},
        },
        "graphs": [],
        "readout_table": {"file": "readout_table.safetensors"},
        "vision": vision_section(tok, token_ids),
        "tables": tables_section(ids),
        "storage_variants": storage_section(),
    }
    # round 6c: the real files (graphs per bucket, the embeds variant, the two text tables) with bytes, sha256, I/O and
    # the Mac gate of each
    fs = files_section(a.rehash)
    doc["status"] = (doc["status"].replace("; the graph files and the tables are cut from the checkpoint later (`graphs` "
                                           "empty until then)", "")
                     + "; round 6c: the real text graphs (`graphs`: v2 per bucket, `embeds_graph`: the embeds variant) "
                       "and the two text tables with bytes and sha256, gated on the Mac (the picture graphs: round 6d)")
    doc["graphs"] = [{"L": g["L"], "file": g["file"], "bytes": g["bytes"], "sha256": g["sha256"], "form": g["form"],
                      "inputs": g["io"]["inputs"], "outputs": g["io"]["outputs"], "run_on": g["run_on"],
                      "gate": {acc: {k: x[k] for k in ("verdict", "questions", "max_abs_dp", "mean_abs_dp",
                                                        "non_near_tie_argmax", "near_tie_argmax", "red_arms", "check")}
                               for acc, x in g["gate"].items()}} for g in fs["graphs"]]
    if fs["embeds_graph"]:
        e = fs["embeds_graph"]
        doc["embeds_graph"] = {"L": e["L"], "file": e["file"], "bytes": e["bytes"], "sha256": e["sha256"],
                               "form": e["form"], "inputs": e["io"]["inputs"], "outputs": e["io"]["outputs"],
                               "host": e["host"], "table": "embed_table.safetensors",
                               "gate": {acc: {k: x[k] for k in ("verdict", "questions", "max_abs_dp", "mean_abs_dp",
                                                                 "non_near_tie_argmax", "near_tie_argmax", "red_arms",
                                                                 "check")} for acc, x in e["gate"].items()}}
    rt = fs["tables"]["readout_table.safetensors"]
    doc["readout_table"] = {"file": "readout_table.safetensors", "bytes": rt["bytes"], "sha256": rt["sha256"]}
    et = fs["tables"]["embed_table.safetensors"]
    doc["tables"]["files"]["embed_table.safetensors"] = {
        "keys": {"embed_tokens.weight": {"dtype": "bfloat16", "shape": [128000, 2048]}}, "bytes": et["bytes"],
        "sha256": et["sha256"], "source": doc["tables"]["files"]["readout_table.safetensors"]["source"],
        "bytes_read_from_checkpoint": 128000 * 2048 * 2,
        "use": "the embeds variant: the host widens the row of every id to float32 (exact) and writes them as `embeds`",
        "cut_by": "scripts/d1_tables.py <snapshot dir> --out <dir> --full-embed (the snapshot's bytes copied as they are)"}
    doc["tables"]["files"]["readout_table.safetensors"].update(bytes=rt["bytes"], sha256=rt["sha256"])
    doc["tables"]["not_cut"] = "nothing: the full tied table is embed_table.safetensors (round 6c)"
    if kept_vision is not None:
        doc["vision"] = kept_vision
    out_path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"token_ids": token_ids, "n_readout_ids": len(ids), "fixture_readout_ids": len(fixture_ids),
                      "buckets": doc["buckets"]["fixture_rows_by_smallest_candidate"], "verified": list(verified),
                      "graphs": [(g["L"], g["bytes"], g["sha256"][:12]) for g in doc["graphs"]],
                      "embeds_graph": fs["embeds_graph"] and fs["embeds_graph"]["sha256"][:12],
                      "missing": fs["missing"], "out": str(out_path), "sha256": sha256_file(out_path)}, indent=1))
    if a.ledger:
        write_ledger(K / a.ledger if not Path(a.ledger).is_absolute() else Path(a.ledger), fs)
    return 0


def write_ledger(path: Path, fs: dict) -> None:
    """results/real_files.json: the shipping candidates (file, bytes, sha256, bucket, form, the gate's verdicts), next to
    the `deleted` list the deletions wrote; the other kept files are listed with their role."""
    led = json.loads(path.read_text()) if path.exists() else {"deleted": []}
    cands = []
    for g in fs["graphs"]:
        cands.append({"file": f"exports/{g['file']}", "bytes": g["bytes"], "sha256": g["sha256"], "bucket": g["L"],
                      "form": g["form"], "gate": {acc: x["verdict"] for acc, x in g["gate"].items()},
                      "gate_detail": g["gate"], "record": g["record"]})
    e = fs["embeds_graph"]
    if e:
        cands.append({"file": f"exports/{e['file']}", "bytes": e["bytes"], "sha256": e["sha256"], "bucket": e["L"],
                      "form": e["form"], "gate": {acc: x["verdict"] for acc, x in e["gate"].items()},
                      "gate_detail": e["gate"], "record": e["record"]})
    for name, t in fs["tables"].items():
        cands.append({"file": f"cache/real/tables/{name}", "bytes": t["bytes"], "sha256": t["sha256"], "bucket": None,
                      "form": "host table", "gate": "read by every gate run of the graphs above", "record": t["manifest"]})
    kept = []
    v1 = shipped_file("real_rowprefill_L256_v1_wi8fc", False)
    if v1:
        kept.append({"file": f"exports/{v1['file']}", "bytes": v1["bytes"], "sha256": v1["sha256"], "bucket": 256,
                     "form": "v1_wi8fc (dynamic int8 FC + int8 table)",
                     "gate": {acc: gate_line(K / f"results/real_rowprefill_L256_v1_wi8fc_{acc}_check.json")["verdict"]
                              for acc in ACCELS},
                     "why_kept": "the quant lever's evidence (fails the bar) and the S26 CPU form to measure there"})
    led.update(what="d1-3B LiteRT row graph files (real weights): the shipping candidates of the text path with their "
                    "Mac gate, the kept non-candidates, and the deleted files (bytes and sha256 checked against the "
                    "record of their making just before deletion)",
               candidates=cands, kept_not_candidates=kept, missing=fs["missing"],
               written_at=__import__("time").strftime("%Y-%m-%dT%H:%M:%S%z"))
    path.write_text(json.dumps(led, indent=1) + "\n")
    print(f"ledger {path.relative_to(K)}: {len(cands)} candidates, {len(kept)} kept, {len(led.get('deleted', []))} deleted")


if __name__ == "__main__":
    sys.exit(main())
