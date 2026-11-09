#!/usr/bin/env python3
"""make_int4_qwen3.py EXPORT_DIR TOKENIZER_JSON OUT_DIR NAME [--block i8|b128|b32] [--act fp32|none] [--keep-tflite DIR]

The quantized file from the float export of export_qwen3.py, without a float .litertlm (~131 GB for Qwen3-32B):
  1 quantize EXPORT_DIR's prefill_decode TFLite (export_artifacts.json) with ai-edge-quantizer and the recipe of
    minicpm5_work/quantize_minicpm5.py build_recipe("wi8fc" | "wi4b128_wi8" | "wi4b32_wi8", "minmax", []) (loaded by
    path) = --block i8: int8 per-channel min-max on every FULLY_CONNECTED (lm_head included) and the embedding;
    b128 / b32: int4 blockwise-128 / -32 min-max on every FULLY_CONNECTED, int8 channelwise on the lm_head FC (its
    LM_HEAD_REGEX) and the embedding. Every blockwise 4-bit symmetric scale is raised to at least 2^-14 while it is computed
    (uniform_quantize_tensor.tensor_zp_scale_from_min_max wrapped; a scale below 2^-14 is subnormal in fp16 and XNNPACK
    refuses the tensor); prints CLAMP_TOTAL; `qt.quantize(serialize_to_path=...)` writes the quantized model into the
    output file's memory map instead of a bytearray of the whole output;
  2 pack with litert-lm-builder 0.18.1 the way litert-torch 0.10.0's own packager builds the bundle: system metadata
    Authors = ODML; LlmMetadata = qwen3_LlmMetadataProto_v0.19.1.pbtext (LiteRT-LM v0.19.1's models/qwen3 pbtext byte
    for byte; Q32_PBTEXT overrides); the export's ExecutorMetadata; TOKENIZER_JSON (the checkpoint's tokenizer.json);
    the quantized TFLite as prefill_decode with `prefer_activation_type` = --act (fp32: the GPU runs fp32 activations;
    none: the GPU's fp16 default);
  3 records OUT_DIR/work_NAME/build.json (inputs and their sha256, recipe, clamp counts, seconds, the file's sha256).
The quantized TFLite is removed after the pack unless --keep-tflite DIR (it moves there). Run with the export's Python
(an environment made from mnn_pocket_work/requirements.txt: ai-edge-quantizer 0.10.1, litert-lm-builder 0.18.1)."""
import argparse
import atexit
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PBTEXT = os.environ.get("Q32_PBTEXT", os.path.join(HERE, "qwen3_LlmMetadataProto_v0.19.1.pbtext"))


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(64 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_by_path(name, path):
    """Load a helper by its path; undo any sys.path entry it inserts at import."""
    before = list(sys.path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path[:] = before
    return mod


def clamp_patch(stats):
    from ai_edge_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor as u
    floor = np.float32(2.0 ** -14)
    orig = u.tensor_zp_scale_from_min_max

    def clamped(min_value, max_value, num_bits, symmetric, granularity, clipping_values=None, *args, **kwargs):
        zp, scale = orig(min_value, max_value, num_bits, symmetric, granularity, clipping_values, *args, **kwargs)
        if u.is_blockwise(granularity) and num_bits == 4:
            if not symmetric:
                raise ValueError("make_int4_qwen3: asymmetric blockwise int4 is not covered by the floor")
            low = scale < floor
            k = int(np.count_nonzero(low))
            stats["blockwise_int4_calls"] += 1
            if k:
                scale = np.where(low, floor, scale).astype(scale.dtype)
                stats["tensors"] += 1
                stats["scales"] += k
                print(f"CLAMP floor 2^-14: {k} of {scale.size} block scales", flush=True)
        return zp, scale

    u.tensor_zp_scale_from_min_max = clamped
    atexit.register(lambda: print(f"CLAMP_TOTAL tensors={stats['tensors']} scales={stats['scales']} "
                                  f"(blockwise int4 calls {stats['blockwise_int4_calls']})", flush=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir")
    ap.add_argument("tokenizer_json")
    ap.add_argument("out_dir")
    ap.add_argument("name")
    ap.add_argument("--block", default="b128", choices=("b128", "b32", "i8"),
                    help="b128 / b32 = int4 blockwise FC + int8 lm_head and token table; i8 = int8 per-channel "
                    "for every FC and the token table (recipe kind wi8fc)")
    ap.add_argument("--act", default="fp32", choices=("fp32", "none"))
    ap.add_argument("--keep-tflite", default="")
    ap.add_argument("--qtflite", default="", help="an existing quantized TFLite of the same export (skip step 1)")
    a = ap.parse_args()
    exp, out = os.path.abspath(a.export_dir), os.path.abspath(a.out_dir)
    final = os.path.join(out, a.name + ".litertlm")
    work = os.path.join(out, f"work_{a.name}")
    os.makedirs(work, exist_ok=True)
    if os.path.exists(final):
        raise SystemExit(f"exists: {final}")
    art = json.load(open(os.path.join(exp, "export_artifacts.json")))
    tfl = art["artifacts"]["prefill_decode_model_path"]
    em = os.path.join(exp, "executor_metadata.pb")
    rec = {"export_dir": exp, "tflite": tfl, "tflite_bytes": os.path.getsize(tfl), "block": a.block, "act": a.act,
           "pbtext": PBTEXT, "pbtext_sha256": sha256_file(PBTEXT), "tokenizer": a.tokenizer_json,
           "tokenizer_sha256": sha256_file(a.tokenizer_json),
           "executor_metadata": em if os.path.exists(em) else None,
           "executor_metadata_sha256": sha256_file(em) if os.path.exists(em) else None, "steps": {}}
    t0 = time.time()
    if a.qtflite:
        qpath = os.path.abspath(a.qtflite)
        rec["steps"]["quantize"] = {"reused": qpath, "bytes": os.path.getsize(qpath)}
    else:
        stats = {"tensors": 0, "scales": 0, "blockwise_int4_calls": 0}
        clamp_patch(stats)
        q5 = load_by_path("quantize_minicpm5_qwen3", os.path.join(REPO, "minicpm5_work", "quantize_minicpm5.py"))
        kind = {"b128": "wi4b128_wi8", "b32": "wi4b32_wi8", "i8": "wi8fc"}[a.block]
        recipe = q5.build_recipe(kind, "minmax", [])
        json.dump(recipe, open(os.path.join(work, "recipe.json"), "w"), indent=1, default=str)
        from ai_edge_quantizer import quantizer
        qt = quantizer.Quantizer(tfl, recipe)
        if qt.need_calibration:
            raise SystemExit("recipe unexpectedly needs calibration")
        qpath = os.path.join(work, "model_q.tflite")
        qt.quantize(serialize_to_path=qpath)
        rec["steps"]["quantize"] = {"s": round(time.time() - t0, 1), "bytes": os.path.getsize(qpath), "clamp": dict(stats),
                                    "recipe_kind": kind, "lm_head_regex": q5.LM_HEAD_REGEX}
        print(f"quantized {os.path.getsize(qpath)} B in {time.time() - t0:.0f} s; clamp {stats}", flush=True)
        del qt
    rec["steps"]["quantize"]["tflite_sha256"] = sha256_file(qpath)
    # pack (litert-torch 0.10.0 package_model's order and system metadata)
    t1 = time.time()
    from litert_lm_builder import litertlm_builder as lb
    b = lb.LitertLmFileBuilder()
    b.add_system_metadata(lb.Metadata(key="Authors", value="ODML", dtype=lb.DType.STRING))
    b.add_llm_metadata(PBTEXT)
    if os.path.exists(em):
        b.add_executor_metadata(em)
    b.add_hf_tokenizer(a.tokenizer_json)
    kw = {"prefer_activation_type": "fp32"} if a.act == "fp32" else {}
    b.add_tflite_model(qpath, lb.TfLiteModelType.PREFILL_DECODE, **kw)
    tmp = final + ".partial"
    with open(tmp, "wb") as f:
        b.build(f)
    os.replace(tmp, final)
    rec["steps"]["pack"] = {"s": round(time.time() - t1, 1)}
    rec["final"] = final
    rec["final_bytes"] = os.path.getsize(final)
    rec["final_sha256"] = sha256_file(final)
    if not a.qtflite:
        if a.keep_tflite:
            os.makedirs(a.keep_tflite, exist_ok=True)
            dst = os.path.join(a.keep_tflite, a.name + ".tflite")
            shutil.move(qpath, dst)
            rec["steps"]["quantize"]["tflite_kept"] = dst
        else:
            os.remove(qpath)
            rec["steps"]["quantize"]["tflite_removed"] = qpath
    rec["total_s"] = round(time.time() - t0, 1)
    json.dump(rec, open(os.path.join(work, "build.json"), "w"), indent=1)
    print("DONE:", final, rec["final_bytes"], "B", rec["final_sha256"], f"{rec['total_s']} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
