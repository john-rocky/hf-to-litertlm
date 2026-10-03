#!/usr/bin/env python3
"""Per-section quantization arms from one fp32 export.

export_hf quantizes each float .tflite section after conversion with the same call for every section
(core/export_lib.py maybe_quantize_model -> quantize_model: Quantizer(path) + load_quantization_recipe(recipe) +
quantize().export_model(...)); the float graphs do not depend on the recipe. This script makes that call per section
with its own recipe, on the sections of an unpacked fp32 bundle (litert-lm unpack), so an arm can mix recipes across
sections (embedder fp32 with an int8 LM, an fp16 encoder with an int8 LM). Quantized sections are cached under
out/sections/<tag>/<section>__<recipe>.tflite with the recipe JSON beside them (a .json recipe path is also accepted by
export_hf --quantization_recipe). An arm dir = symlinks to the chosen sections + the fp32 bundle's metadata/tokenizer +
model.toml; --pack also writes the .litertlm (litert-lm pack, ~/venvs/lt0171run).

Recipes:
  none                      the fp32 section as exported
  dynamic_wi8_afp32         the export's named recipe (int8 channelwise weights, fp32 activations, dynamic)
  dynamic_wi8c_hr_afp32     the same with ai-edge-quantizer's decomposed Hadamard rotation (recipe.py alias)
  fp16                      float16 weights, fp32 compute (FLOAT_CASTING, explicit dequantize; funasr_nano_work recipe)
  i8_headf16                LM: dynamic int8 on every FC except the lm_head (decode_logits_output), lm_head fp16
  <name>                    any other ai_edge_quantizer.recipe attribute

  ~/venvs/ltmain0918/bin/python quant_sections.py --src out/unpack/c4r2_f32_5s --tag c4r2_5s --arm a_embf32 \
      --lm dynamic_wi8_afp32 --emb none --enc dynamic_wi8_afp32
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
SECTIONS = {"lm": "Section3_TFLiteModel_tf_lite_prefill_decode.tflite",
            "emb": "Section4_TFLiteModel_tf_lite_embedder.tflite",
            "enc": "Section5_TFLiteModel_tf_lite_audio_encoder_hw.tflite"}
PACK = os.path.expanduser("~/venvs/lt0171run/bin/litert-lm")


def recipe_of(name):
    from ai_edge_quantizer import algorithm_manager, qtyping, recipe as recipe_lib, recipe_manager
    if name == "fp16":
        rm = recipe_manager.RecipeManager()
        rm.add_quantization_config(
            regex=".*", operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
            op_config=qtyping.OpQuantizationConfig(
                weight_tensor_config=qtyping.TensorQuantizationConfig(num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
                compute_precision=qtyping.ComputePrecision.FLOAT, explicit_dequantize=True),
            algorithm_key=algorithm_manager.AlgorithmName.FLOAT_CASTING)
        return rm.get_quantization_recipe()
    if name == "i8_headf16":
        # LM only: dynamic int8 channelwise on every FULLY_CONNECTED except the lm_head (the vocab-size FC whose
        # output tensor is 'decode_logits_output'), which gets fp16 weights by FLOAT_CASTING.
        rm = recipe_manager.RecipeManager()
        rm.add_dynamic_config(regex=r"^(?!.*logits).*$", operation_name=qtyping.TFLOperationName.FULLY_CONNECTED,
                              num_bits=8)
        rm.add_quantization_config(
            regex=r".*logits.*", operation_name=qtyping.TFLOperationName.FULLY_CONNECTED,
            op_config=qtyping.OpQuantizationConfig(
                weight_tensor_config=qtyping.TensorQuantizationConfig(num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
                compute_precision=qtyping.ComputePrecision.FLOAT, explicit_dequantize=True),
            algorithm_key=algorithm_manager.AlgorithmName.FLOAT_CASTING)
        return rm.get_quantization_recipe()
    from litert_torch.generative.export_hf.core import export_lib
    if name in export_lib._LOCAL_QUANTIZATION_RECIPES:
        return export_lib._LOCAL_QUANTIZATION_RECIPES[name]()
    return recipe_lib.__dict__[name]()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def section(src, tag, key, recipe):
    fp32 = os.path.join(src, SECTIONS[key])
    if recipe == "none":
        return os.path.realpath(fp32), 0.0
    d = os.path.join(OUT, "sections", tag)
    os.makedirs(d, exist_ok=True)
    out = os.path.join(d, f"{key}__{recipe}.tflite")
    if os.path.exists(out):
        return out, 0.0
    from ai_edge_quantizer import quantizer as quantizer_lib
    rec = recipe_of(recipe)
    with open(out[:-7] + ".recipe.json", "w") as f:
        json.dump(rec, f, indent=1, default=str)
    t0 = time.time()
    qt = quantizer_lib.Quantizer(fp32)
    qt.load_quantization_recipe(rec)
    qt.quantize().export_model(out + ".part", overwrite=True)
    os.replace(out + ".part", out)
    dt = time.time() - t0
    print(f"quantized {key} {recipe}: {os.path.getsize(fp32):,} -> {os.path.getsize(out):,} B in {dt:.0f} s",
          flush=True)
    return out, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="unpacked fp32 bundle dir (litert-lm unpack)")
    ap.add_argument("--tag", required=True, help="section cache name, one per fp32 export")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--lm", default="dynamic_wi8_afp32")
    ap.add_argument("--emb", default="dynamic_wi8_afp32")
    ap.add_argument("--enc", default="dynamic_wi8_afp32")
    ap.add_argument("--pack", action="store_true")
    args = ap.parse_args()
    dst = os.path.join(OUT, "unpack", args.arm)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    info = {"arm": args.arm, "src": args.src, "recipes": {"lm": args.lm, "emb": args.emb, "enc": args.enc},
            "sections": {}}
    for f in os.listdir(args.src):
        if f.endswith(".tflite"):
            continue
        os.symlink(os.path.realpath(os.path.join(args.src, f)), os.path.join(dst, f))
    for key in ("lm", "emb", "enc"):
        path, dt = section(args.src, args.tag, key, getattr(args, key))
        os.symlink(path, os.path.join(dst, SECTIONS[key]))
        info["sections"][key] = {"path": path, "bytes": os.path.getsize(path), "quantize_seconds": round(dt, 1)}
    info["sections_bytes"] = sum(s["bytes"] for s in info["sections"].values())
    if args.pack:
        out = os.path.join(OUT, "export", args.arm, "model.litertlm")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        t0 = time.time()
        subprocess.run([PACK, "pack", dst, "--output", out, "--allow-overwrite"], check=True)
        info["bundle"] = {"path": out, "bytes": os.path.getsize(out), "sha256": sha256(out),
                          "pack_seconds": round(time.time() - t0, 1)}
    with open(os.path.join(dst, "arm.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
