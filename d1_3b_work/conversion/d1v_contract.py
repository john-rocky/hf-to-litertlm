"""Round 6d copy of d1_contract.py (round 6b's file, copied at sha256
7cda33c7a69a0d7c2b0d72022f0f111bc81668c011f757fd70af5b64e717cc12, HEAD 67d05491): updates ONLY the `vision` section of
host/contract.json with the real picture graphs (round 6d acceptance 6).

    $EXPORT scripts/d1v_contract.py --vision-only

Changes against the copied file: `real_files_section` (the four real picture files: file name, bytes, sha256, the
signature as the file declares it, op counts, how each was made, and the Mac checks: delegation, max |diff| and
relative vs transformers, non-finite values, compile seconds; the position table file the host feeds the tower),
`vision_section(..., real=...)` (status and verified records gain round 6d), and `main` = --vision-only: the existing
contract is read, only `vision` is rebuilt and replaced, every other top-level key is asserted unchanged, and the text
is asserted identical to the old file outside the vision section (the same json.dumps arguments as the original).
Everything is read from the files, never typed in.

Original docstring (round 2 / 3 / 5):
Round 2 acceptance 7 / round 3 acceptance 9: host/contract.json, the contract between the d1-3B graphs and a host
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


REAL_TAG, REAL_RTAG = "real", "realv"
REAL_GRAPHS = {"tower": "vision_tower", "projector": "projector"}
REAL_FORMS = {"fp32": "fp32", "v2_fp16fc": "v2_fp16fc"}
POSITION_TABLE = K / "cache/real/tables/vision_position_table.safetensors"
POSITION_CUT = K / "results/tables_cut_real.json"


def real_files_section() -> dict:
    """Round 6d: the real picture graph files and their Mac checks, read from the round's records."""
    from tflite_scan import scan

    out = {"status": "round 6d: exported from the real weights (LiquidAI/d1-3B at da1fe36a) and checked on the Mac "
                     "(CPU XNNPACK, Metal float32 and default precision) against transformers 5.14.1 on 71 tiles (18 "
                     "synthetic pictures + one photo); not run on a phone yet",
           "made_by": {"fp32": "scripts/d1_vision_graph.py --source <snapshot> --tag real --rtag realv --write tower|projector",
                       "v2_fp16fc": "scripts/d1v_storage.py --tflite exports/real_<graph>_fp32.tflite --variant v2 "
                                    "(weight-only 16-bit FLOAT_CASTING on every FULLY_CONNECTED, one DEQUANTIZE each)",
                       "driver": "scripts/d1_run_vision.sh <snapshot> real (env D1V_RTAG=realv D1V_STORAGE=v2 ...)"},
           "torch_check": None}
    tc = json.loads((K / f"results/{REAL_RTAG}_vision_torch_check.json").read_text())
    out["torch_check"] = {"file": f"results/{REAL_RTAG}_vision_torch_check.json", "pass": tc["summary"]["pass"],
                          "bar": tc["bar"], **{k: tc["summary"][k] for k in (
                              "tiles", "tower_vs_hf_uncut_max", "tower_vs_hf_uncut_rel_max", "projector_alone_vs_hf_rel_max",
                              "projector_chain_vs_hf_rel_max", "pad_bits_equal", "pad_tiles", "unshuffle_bits_equal")}}
    for g, gname in REAL_GRAPHS.items():
        w = json.loads((K / f"results/{REAL_RTAG}_{gname}_fp32_tflite.json").read_text())
        fp32_sig = w["signatures"][0]
        forms = {}
        for form in REAL_FORMS:
            stem = f"{REAL_TAG}_{gname}_{form}"
            path = K / f"exports/{stem}.tflite"
            if form == "fp32":
                rec = {"bytes": w["bytes"], "sha256": w["sha256"], "operator_count": w["operator_count"],
                       "op_histogram": w["op_histogram"], "signature": fp32_sig, "params": w.get("params"),
                       "record": f"results/{REAL_RTAG}_{gname}_fp32_tflite.json"}
            else:
                q = json.loads((K / f"results/{REAL_RTAG}_{gname}_{form}_quant.json").read_text())
                sc = scan(path)
                assert sc["sha256"] == q["output"]["sha256"] and sc["signatures"][0] == fp32_sig, stem
                rec = {"bytes": q["output"]["bytes"], "sha256": q["output"]["sha256"],
                       "operator_count": sc["operator_count"], "op_histogram": sc["op_histogram"],
                       "signature": "same as fp32", "storage_checks_pass": q["checks_pass"],
                       "record": f"results/{REAL_RTAG}_{gname}_{form}_quant.json"}
            assert path.stat().st_size == rec["bytes"], stem
            checks = {}
            for acc in ("cpu", "gpu_f32", "gpu_default"):
                cp = K / f"results/{REAL_RTAG}_{gname}_{form}_{acc}_check.json"
                if not cp.exists():
                    checks[acc] = None
                    continue
                c = json.loads(cp.read_text())
                checks[acc] = {"status": c["status"], "error": c.get("error"),
                               "delegation": [f"{r['delegated']}/{r['total']} {r['delegate']}"
                                              for r in c["delegation"]["replacing"]],
                               "max_abs_vs_hf": c.get("max_abs_vs_hf"), "max_rel_vs_hf": c.get("max_rel_vs_hf"),
                               "nonfinite": c.get("nonfinite_total"), "compile_seconds": c.get("compile_seconds"),
                               "record": str(cp.relative_to(K))}
            forms[form] = {"file": path.name, **rec, "mac_checks": checks}
        out[g] = forms
    vs64 = json.loads((K / f"results/{REAL_RTAG}_tower_vs_fp64.json").read_text())["results"]
    col = json.loads((K / f"results/{REAL_RTAG}_default_collapse.json").read_text())["results"]
    nr = json.loads((K / f"results/{REAL_RTAG}_vision_norm_range.json").read_text())["sites"]
    w16 = json.loads((K / f"results/{REAL_RTAG}_vision_tower_v2_fp16fc_weights.json").read_text())["totals"]
    p16 = json.loads((K / f"results/{REAL_RTAG}_projector_v2_fp16fc_weights.json").read_text())["totals"]
    gd, m32 = col[f"litert_{REAL_RTAG}_vision_tower_fp32_gpu_default"], vs64[f"litert_{REAL_RTAG}_vision_tower_fp32_gpu_f32"]
    out["notes"] = {
        "metal_default_precision": (
            f"not usable for the tower: its LayerNorm inputs reach a sum of squared differences of "
            f"{max(v['max_sum_of_squares_all_positions'] for v in nr.values()):.4g} (fp16 holds 65,504), the variance "
            f"overflows, rsqrt gives 0 and {gd['rows_within_tol_of_beta']:,} of {gd['rows']:,} output rows equal "
            f"post_layernorm.bias (results/realv_vision_norm_range.json, results/realv_default_collapse.json)"),
        "metal_float32": (
            f"{m32['max_rel']:.2e} relative from the float64 truth on its worst tile, the closest of the float32 runs "
            f"(transformers float32 {vs64['transformers float32 SDPA (hf32)']['max_rel']:.2e}, LiteRT CPU "
            f"{vs64[f'litert_{REAL_RTAG}_vision_tower_fp32_cpu']['max_rel']:.2e}; results/realv_tower_vs_fp64.json)"),
        "fp16_fc_weights": (
            f"tower: {w16['exact'] / w16['elements']:.1%} of the FC weight values survive the fp16 cast (the converter "
            f"folds the LayerNorm scale into the next FC, so those weights are no longer bfloat16 values); projector: "
            f"{p16['exact'] / p16['elements']:.2%} (results/realv_*_v2_fp16fc_weights.json)"),
    }
    cut = json.loads(POSITION_CUT.read_text())["files"][POSITION_TABLE.name]
    out["position_table"] = {"file": POSITION_TABLE.name, "bytes": POSITION_TABLE.stat().st_size,
                             "sha256": sha256_file(POSITION_TABLE), "keys": cut["keys"], "source": cut["source"],
                             "cut_by": "scripts/d1_tables.py (round 6b, results/tables_cut_real.json)"}
    assert out["position_table"]["sha256"] == cut["sha256"] and out["position_table"]["bytes"] == cut["bytes"]
    return out


def vision_section(tok, token_ids: dict, real: dict | None = None) -> dict:
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
        "status": ("round 3: rules final, verified end to end on the tiny VL model (host/test_d1_vision_tiny.py); the "
                   "real graph files and the position table are cut from the checkpoint later" if real is None else
                   "round 3: rules final, verified end to end on the tiny VL model (host/test_d1_vision_tiny.py); round "
                   "6d: the real graph files (`real_files`), checked on the Mac against transformers; the picture "
                   "path end to end on the real weights is not run yet (needs the real embeds row graph)"),
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
                     "header": "results/vision_header_check.json",
                     **({} if real is None else {
                         "real_torch": "results/realv_vision_torch_check.json",
                         "real_litert": "results/realv_{vision_tower,projector}_{fp32,v2_fp16fc}_{cpu,gpu_f32,gpu_default}_check.json",
                         "real_float64_noise": "results/realv_tower_fp64_noise.json",
                         "real_extra_picture": "results/realv_extra_pictures.json"})},
        **({} if real is None else {"real_files": real}),
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
    out["built_by"] = "scripts/d1_storage.py --tflite <..._fp32.tflite> --variant v1|v2|v3 (ai-edge-quantizer 0.9.0, no calibration)"
    return out


def vision_only(tok, token_ids: dict) -> int:
    """Round 6d: rebuild and replace `vision` only; everything else must stay byte for byte."""
    old_text = OUT.read_text()
    old = json.loads(old_text)
    assert json.dumps(old, indent=1, ensure_ascii=False) + "\n" == old_text, "contract.json is not in the writer's format"
    new_vision = vision_section(tok, token_ids, real=real_files_section())
    doc = {k: (new_vision if k == "vision" else v) for k, v in old.items()}
    new_text = json.dumps(doc, indent=1, ensure_ascii=False) + "\n"
    back = {k: (old["vision"] if k == "vision" else v) for k, v in doc.items()}
    assert json.dumps(back, indent=1, ensure_ascii=False) + "\n" == old_text, "a key other than vision changed"
    assert list(doc) == list(old) and all(doc[k] == old[k] for k in old if k != "vision")
    changed = sorted(k for k in set(old["vision"]) | set(new_vision) if old["vision"].get(k) != new_vision.get(k))
    OUT.write_text(new_text)
    print(json.dumps({"file": str(OUT.relative_to(K)), "sha256_before": hashlib.sha256(old_text.encode()).hexdigest(),
                      "sha256_after": sha256_file(OUT), "top_level_keys": list(doc), "changed_top_level": ["vision"],
                      "vision_keys_changed_or_added": changed}, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replace", action="store_true")
    ap.add_argument("--vision-only", action="store_true", help="round 6d: replace only the vision section")
    a = ap.parse_args()
    if not a.vision_only:
        raise SystemExit("this copy only updates the vision section: --vision-only (the full writer is d1_contract.py)")
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
    if a.vision_only:
        return vision_only(tok, token_ids)

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
    OUT.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"token_ids": token_ids, "n_readout_ids": len(ids), "fixture_readout_ids": len(fixture_ids),
                      "buckets": doc["buckets"]["fixture_rows_by_smallest_candidate"], "verified": list(verified),
                      "sha256": sha256_file(OUT)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
