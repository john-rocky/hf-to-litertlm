#!/usr/bin/env python3
"""quant_pack.py --float GRAPH --recipe JSON --ctx N --code-dir HF --template JINJA --out FILE [--sampler-top-p P]

Quantizes the float LFM3-1.7B graph that `lfm3_ext.export_lfm3` wrote (quantization_recipe 'none',
keep_temporary_files) with an ai-edge-quantizer recipe through litert-torch's own `quantize_model`, and packs it with
litert-torch's own `package_model` and the export settings of the float run: the chat template JINJA, the repository's
tokenizer.json bytes, generic_model, a KV cache of N tokens, prefill signatures 1024 / 256 / 64 / 16 / 4 / 1.
Every blockwise 4-bit symmetric scale is floored at 2^-14 before the weights are rounded (XNNPACK refuses a block
scale under that); a recipe without int4 entries is not touched by the floor. --sampler-top-p sets the sampler the
file declares: with the source's generation_config.json (do_sample, top_k 50, temperature 0.1) and P = 1 the file
declares TOP_P with k 50, p 1, temperature 0.1, which LiteRT-LM's CPU and GPU samplers both run as top-k 50 at
temperature 0.1 (without it the packager writes TOP_K, which the 0.18.1 CPU sampler refuses).
"""
import argparse
import os
import shutil
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
FLOOR = 2.0 ** -14


def install_int4_floor():
  """Floors blockwise 4-bit symmetric scales at 2^-14 inside ai-edge-quantizer (clamp before quantize)."""
  import numpy as np  # pylint: disable=g-import-not-at-top
  from ai_edge_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor as u  # pylint: disable=g-import-not-at-top
  orig = u.tensor_zp_scale_from_min_max
  stats = {"tensors": 0, "scales": 0}

  def clamped(min_value, max_value, num_bits, symmetric, granularity, *args, **kwargs):
    zp, scale = orig(min_value, max_value, num_bits, symmetric, granularity, *args, **kwargs)
    if u.is_blockwise(granularity) and num_bits == 4:
      if not symmetric:
        raise ValueError("int4 floor: asymmetric blockwise int4 is not covered")
      low = scale < np.float32(FLOOR)
      n = int(np.count_nonzero(low))
      if n:
        scale = np.where(low, np.float32(FLOOR), scale).astype(scale.dtype)
        stats["tensors"] += 1
        stats["scales"] += n
    return zp, scale

  u.tensor_zp_scale_from_min_max = clamped
  return stats


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--float", dest="float_graph", required=True)
  ap.add_argument("--recipe", required=True)
  ap.add_argument("--ctx", type=int, required=True)
  ap.add_argument("--code-dir", required=True)
  ap.add_argument("--template", required=True)
  ap.add_argument("--out", required=True, help="the .litertlm file to write")
  ap.add_argument("--prefill", default="1024,256,64,16,4,1")
  ap.add_argument("--sampler-top-p", type=float)
  a = ap.parse_args()
  hf = os.path.abspath(a.code_dir)
  recipe = os.path.abspath(a.recipe)
  work = os.path.splitext(os.path.abspath(a.out))[0] + ".work"
  os.makedirs(work, exist_ok=True)

  import transformers  # pylint: disable=g-import-not-at-top
  from litert_torch.generative.export_hf.core import export_lib  # pylint: disable=g-import-not-at-top
  from litert_torch.generative.export_hf.core import exportable_module_config  # pylint: disable=g-import-not-at-top
  from litert_torch.generative.export_hf.core import litert_lm_builder  # pylint: disable=g-import-not-at-top
  from litert_torch.generative.export_hf.core import quantization  # pylint: disable=g-import-not-at-top
  from lfm3_ext import extension  # pylint: disable=g-import-not-at-top
  from lfm3_ext import loader  # pylint: disable=g-import-not-at-top

  stats = install_int4_floor()
  t0 = time.time()
  qpath = quantization.quantize_model(os.path.abspath(a.float_graph), recipe)
  graph = os.path.join(work, "model.tflite")
  shutil.move(qpath, graph)
  t1 = time.time()
  print(f"quantized with {os.path.basename(recipe)} in {t1 - t0:.1f}s; int4 scales floored at 2^-14: "
        f"{stats['scales']} in {stats['tensors']} tensors", flush=True)
  config = loader.load_lfm3_config(hf)
  gen = transformers.GenerationConfig.from_pretrained(hf)
  tokenizer = transformers.AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
  stand_in = types.SimpleNamespace(config=config, generation_config=gen)  # packaging reads no weights
  artifacts = export_lib.SourceModelArtifacts(
      model=stand_in, model_config=config, text_model_config=config, tokenizer=tokenizer)
  extra = {} if a.sampler_top_p is None else {"sampler_top_p": a.sampler_top_p}
  export_config = exportable_module_config.ExportableModuleConfig(
      model=hf, output_dir=work, work_dir=work, quantization_recipe=recipe,
      model_extension=extension.Lfm3Extension(),
      cache_length=a.ctx, prefill_lengths=[int(x) for x in a.prefill.split(",")],
      use_jinja_template=True, jinja_chat_template_override=os.path.abspath(a.template),
      tokenizer_path_override=os.path.join(hf, "tokenizer.json"),
      externalize_embedder=False, single_token_embedder=False, **extra)
  export_config = export_lib.update_export_config(export_config, artifacts)
  exported = export_lib.ExportedModelArtifacts(
      prefill_decode_model_path=graph, tokenizer_model_path=os.path.join(hf, "tokenizer.json"))
  litert_lm_builder.package_model(artifacts, export_config, exported)
  shutil.move(os.path.join(work, "model.litertlm"), a.out)
  shutil.rmtree(work)
  print(f"packed in {time.time() - t1:.1f}s -> {a.out}", flush=True)


if __name__ == "__main__":
  main()
