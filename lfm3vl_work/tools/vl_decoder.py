#!/usr/bin/env python3
"""The LFM3-VL-2B text decoder through the LFM3 decoder exporter of this repository (lfm3_work/lfm3_ext, imported by
path), with the VL checkpoint's weights and text_config.

  PY tools/vl_decoder.py float --src SRC --template VL.jinja --out DEC [--ctx 2048] [--prefill 1024,256,64,16,4,1]
  PY tools/vl_decoder.py quant --form int4|emb_wi8|f16 --out DEC [--ctx 2048]

Exporter entry = lfm3_ext.export_lfm3.export_lfm3(output_dir, *, code_dir, weights, key_prefix, config, tokenizer_dir,
**export_kwargs); signatures prefill_{N} / decode with `embeddings` [1,T,2048], `input_pos`, `mask`, states
kv_cache_c_i [1,2048,6] / kv_cache_k_i [1,4,C,128] / kv_cache_v_i [1,4,128,C]; the separate embedder tflite
token_ids [1,1] -> embeddings [1,1,2048].
float: export_lfm3(DEC/float, code_dir = SRC (the source snapshot: modeling_lfm3.py + configuration_lfm3.py; the folder
    must be named `src`, because transformers names the remote-code module after it and the graph's tensor names carry
    that module path), weights = SRC/model.safetensors, key_prefix "model.language_model.", config = config.json's
    text_config, tokenizer_dir = SRC) with fp32 (recipe 'none'), externalize_embedder=True + single_token_embedder=True
    (the fast_vlm image path feeds the decoder embeddings), use_jinja_template with the VL template,
    tokenizer_path_override = SRC/tokenizer.json, keep_temporary_files (the float tflites stay for the quant step),
    use_rope_composite=False (the LFM3 attention has no rope composite). The .litertlm it writes is the "decoder bundle"
    of tools/build_vl_bundle.py (EMBEDDER + PREFILL_DECODE + ExecutorMetadata); its LlmMetadata is replaced there.
quant: DEC/float/model.tflite or DEC/float/embedder.tflite through litert-torch's quantize_model ->
    DEC/<form>.tflite. int4 = ../recipe_int4_b32_minmax.json (FULLY_CONNECTED int4 blockwise-32 symmetric min-max,
    EMBEDDING_LOOKUP and the tied lm_head FC (output `decode_logits_output`) int8 per channel) with every int4 block
    scale floored at 2^-14 (lfm3_work/quant_pack.py install_int4_floor); f16 = ../recipe_f16.json (every
    FULLY_CONNECTED weight cast to fp16); emb_wi8 = the embedder through dynamic_wi8_afp32 (int8 table).
Each step writes DEC/vl_decoder_<step>.json (versions, seconds, files, sha256).
"""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time

K = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
A = os.path.abspath(os.path.join(K, "..", "lfm3_work"))
RECIPES = {"int4": os.path.join(K, "recipe_int4_b32_minmax.json"), "f16": os.path.join(K, "recipe_f16.json"),
           "emb_wi8": "dynamic_wi8_afp32"}


def sha256(p):
  h = hashlib.sha256()
  with open(p, "rb") as f:
    for c in iter(lambda: f.read(1 << 24), b""):
      h.update(c)
  return h.hexdigest()


def load_by_path(name, path):
  spec = importlib.util.spec_from_file_location(name, path)
  mod = importlib.util.module_from_spec(spec)
  sys.modules[name] = mod
  spec.loader.exec_module(mod)
  return mod


def a_package():
  """The lfm3_ext package (its modules import each other as `from . import ...`): lfm3_work on sys.path."""
  if A not in sys.path:
    sys.path.insert(0, A)
  import lfm3_ext  # noqa: F401
  from lfm3_ext import export_lfm3  # noqa: F401
  return sys.modules["lfm3_ext"]


def run_float(a, out):
  a_package()
  from lfm3_ext import export_lfm3
  src = os.path.abspath(a.src)
  text_config = json.load(open(os.path.join(src, "config.json")))["text_config"]
  t0 = time.time()
  export_lfm3.export_lfm3(
      os.path.join(out, "float"), code_dir=src, weights=os.path.join(src, "model.safetensors"),
      key_prefix="model.language_model.", config=text_config, tokenizer_dir=src,
      quantization_recipe="none", keep_temporary_files=True, cache_length=a.ctx,
      prefill_lengths=[int(x) for x in a.prefill.split(",")], use_jinja_template=True,
      jinja_chat_template_override=os.path.abspath(a.template),
      tokenizer_path_override=os.path.join(src, "tokenizer.json"),
      externalize_embedder=True, single_token_embedder=True,
      use_rope_composite=False)
  float_s = round(time.time() - t0, 1)
  files = {}
  for root, _d, fs in os.walk(os.path.join(out, "float")):
    for f in fs:
      if f.endswith((".tflite", ".litertlm")):
        p = os.path.join(root, f)
        files[os.path.relpath(p, out)] = {"bytes": os.path.getsize(p), "sha256": sha256(p)}
  return {"float_s": float_s, "files": files}


def run_quant(a, out):
  a_package()
  from litert_torch.generative.export_hf.core import quantization
  qp = load_by_path("lfm3_quant_pack", os.path.join(A, "quant_pack.py"))
  fdir = os.path.join(out, "float")
  tfl = sorted(f for f in os.listdir(fdir) if f.endswith(".tflite"))
  emb = [f for f in tfl if "embedder" in f]
  dec = [f for f in tfl if "embedder" not in f]
  assert len(emb) == 1 and len(dec) == 1, tfl
  src = os.path.join(fdir, emb[0] if a.form.startswith("emb_") else dec[0])
  recipe = RECIPES[a.form]
  stats = qp.install_int4_floor() if a.form == "int4" else None
  t0 = time.time()
  qpath = quantization.quantize_model(src, recipe)
  dst = os.path.join(out, f"{a.form}.tflite")
  shutil.move(qpath, dst)
  rec = {"recipe_sha256": sha256(recipe)} if os.sep in recipe else {}
  return {"source": os.path.relpath(src, out), "recipe": os.path.basename(recipe), **rec,
          "quant_s": round(time.time() - t0, 1), "int4_floor": stats, "output": os.path.relpath(dst, out),
          "bytes": os.path.getsize(dst), "sha256": sha256(dst)}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("stage", choices=("float", "quant"))
  ap.add_argument("--out", required=True, help="the decoder work folder (DEC/float/ and DEC/<form>.tflite)")
  ap.add_argument("--src", help="float: the source snapshot folder, named src")
  ap.add_argument("--template", help="float: the VL chat template (tools/vl_template.py)")
  ap.add_argument("--ctx", type=int, default=2048)
  ap.add_argument("--prefill", default="1024,256,64,16,4,1")
  ap.add_argument("--form", choices=sorted(RECIPES))
  a = ap.parse_args()
  if a.stage == "float":
    assert a.src and a.template, "float needs --src and --template"
    assert os.path.basename(os.path.normpath(a.src)) == "src", "the source folder must be named src (tensor names)"
  else:
    assert a.form, "quant needs --form"
  out = os.path.abspath(a.out)
  os.makedirs(out, exist_ok=True)
  res = {"stage": a.stage, "ctx": a.ctx, "form": a.form}
  res.update(run_float(a, out) if a.stage == "float" else run_quant(a, out))
  import importlib.metadata as md
  import litert_torch
  lt_dir = os.path.dirname(os.path.dirname(os.path.abspath(litert_torch.__file__)))
  res["litert_torch_checkout"] = subprocess.run(["git", "-C", lt_dir, "rev-parse", "HEAD"], capture_output=True,
                                                text=True).stdout.strip() or None
  res["versions"] = {p: md.version(p) for p in ("torch", "litert-torch", "transformers")}
  for p in ("ai-edge-quantizer", "ai-edge-quantizer-nightly"):
    try:
      res["versions"][p] = md.version(p)
    except md.PackageNotFoundError:
      pass
  tag = a.form if a.stage == "quant" else "float"
  json.dump(res, open(os.path.join(out, f"vl_decoder_{tag}.json"), "w"), indent=1)
  print("VL_DECODER_DONE", json.dumps(res)[:600], flush=True)
  return 0


if __name__ == "__main__":
  sys.exit(main())
