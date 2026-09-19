#!/usr/bin/env python3
"""Build a LiteRT-LM bundle from a float model.tflite (export_driver.py output):

  int4 blockwise (b32|b128, min-max) on every FULLY_CONNECTED, int8 channelwise on lm_head + embedding lookup,
  the Hadamard rotation FCs left float (NO_QUANTIZE; regex on the tensor name)
  -> LlmMetadata pbtext (stop tokens, simple ChatML template, sampler defaults)
  -> litert-lm-builder pack (hf tokenizer)
  -> zero-scale fix in place (ternary sparsity -> all-zero int4 blocks -> scale 0 -> XNNPACK refuses)
  -> ExecutorMetadata section (state buffers, litert-lm >= 0.15)
  -> prefer_activation_type fp32 (GPU executor; fp16 overflows on this family)

Usage: build_bundle.py <model.tflite> <tokenizer.json> <template.jinja> <out.litertlm> [--block b32|b128]
       [--hadamard-regex REGEX] [--max-tokens 4096] [--sampler topk,topp,temp] [--lm-head-regex REGEX]
Run with a venv that has ai-edge-quantizer + litert-lm(-builder) >= 0.15 on PATH (ltconv040dev).
"""
import argparse, os, shutil, subprocess, sys, tempfile, time

ap = argparse.ArgumentParser()
ap.add_argument("tflite"); ap.add_argument("tokenizer"); ap.add_argument("template"); ap.add_argument("out")
ap.add_argument("--block", default="b32", choices=["b32", "b128"])
ap.add_argument("--hadamard-regex", default="Linear_hadamard_rotation")
ap.add_argument("--lm-head-regex", default="(decode_logits_output|Linear_lm_head|lm_head)")
ap.add_argument("--max-tokens", type=int, default=4096)
ap.add_argument("--sampler", default="20,0.8,0.7")
ap.add_argument("--stop-ids", default="248046,248044")
ap.add_argument("--keep-float-head", action="store_true", help="int4 the lm_head/embedding too (default: int8)")
ap.add_argument("--keep-tflite", default=None, help="also save the quantized tflite here (parity harness input)")
ap.add_argument("--prequantized", action="store_true", help="input tflite is already quantized: skip step 1 (repack with another template)")
ap.add_argument("--channels", action="store_true", help="declare the <think> thought channel (thinking-mode template)")
a = ap.parse_args()
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.dirname(here)
tmp = tempfile.mkdtemp(prefix="bonsai2_build_", dir=os.path.dirname(os.path.abspath(a.out)))
t0 = time.time()

# 1. quantize (unless --prequantized)
qtfl = os.path.join(tmp, "model_q.tflite")
if a.prequantized:
    shutil.copyfile(a.tflite, qtfl)
else:
    from ai_edge_quantizer import quantizer, recipe_manager, qtyping
    from ai_edge_quantizer.algorithm_manager import AlgorithmName
    G, OP = qtyping.QuantGranularity, qtyping.TFLOperationName
    rm = recipe_manager.RecipeManager()
    blk = {"b32": G.BLOCKWISE_32, "b128": G.BLOCKWISE_128}[a.block]
    rm.add_dynamic_config(regex=".*", operation_name=OP.FULLY_CONNECTED, num_bits=4, granularity=blk,
                          algorithm_key=AlgorithmName.MIN_MAX_UNIFORM_QUANT)
    if not a.keep_float_head:
        rm.add_dynamic_config(regex=a.lm_head_regex, operation_name=OP.FULLY_CONNECTED, num_bits=8, granularity=G.CHANNELWISE)
    rm.add_dynamic_config(regex=".*", operation_name=OP.EMBEDDING_LOOKUP, num_bits=8, granularity=G.CHANNELWISE)
    rm.add_quantization_config(regex=a.hadamard_regex, operation_name=OP.FULLY_CONNECTED, algorithm_key=AlgorithmName.NO_QUANTIZE)
    recipe = rm.get_quantization_recipe()
    qt = quantizer.Quantizer(a.tflite, recipe)
    assert not qt.need_calibration
    qt.quantize().export_model(qtfl, overwrite=True)
    print(f"quantized: {os.path.getsize(qtfl)/1e9:.2f} GB  t={time.time()-t0:.0f}s", flush=True)
    if a.keep_tflite:
        shutil.copyfile(qtfl, a.keep_tflite)

# 2. metadata
tmpl = open(a.template).read()
esc = tmpl.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
k, p, temp = a.sampler.split(",")
pb = "".join("stop_tokens {\n  token_ids {\n    ids: %s\n  }\n}\n" % s for s in a.stop_ids.split(","))
pb += f"max_num_tokens: {a.max_tokens}\nllm_model_type {{\n  generic_model {{\n  }}\n}}\n"
pb += f'sampler_params {{\n  type: TOP_P\n  k: {k}\n  p: {p}\n  temperature: {temp}\n}}\n'
if a.channels:
    pb += 'channels {\n  channel_name: "thought"\n  start: "<think>"\n  end: "</think>"\n  is_reasoning_channel: true\n}\nsupports_thinking: true\n'
pb += f'jinja_prompt_template: "{esc}"\n'
pbp = os.path.join(tmp, "LlmMetadataProto.pbtext"); open(pbp, "w").write(pb)

# 3. pack
packed = os.path.join(tmp, "packed.litertlm")
subprocess.run(["litert-lm-builder", "llm_metadata", "--path", pbp, "hf_tokenizer", "--path", a.tokenizer,
                "tflite_model", "--path", qtfl, "--model_type", "prefill_decode", "output", "--path", packed], check=True)
os.remove(qtfl)
# 4. zero scales
fixed = os.path.join(tmp, "fixed.litertlm")
subprocess.run([sys.executable, os.path.join(root, "minicpm52b_work", "fix_zero_scales_inplace.py"), packed, fixed], check=True)
os.remove(packed)
# 5. executor metadata
meta = os.path.join(tmp, "meta.litertlm")
subprocess.run([sys.executable, os.path.join(root, "scripts", "add_executor_metadata.py"), fixed, meta,
                "--litert-lm", shutil.which("litert-lm")], check=True)
os.remove(fixed)
# 6. fp32 activations
subprocess.run([sys.executable, os.path.join(root, "scripts", "set_activation_type.py"), meta, a.out, "--type", "fp32"], check=True)
shutil.rmtree(tmp, ignore_errors=True)
print(f"DONE {a.out} {os.path.getsize(a.out)/1e9:.2f} GB  t={time.time()-t0:.0f}s")
