"""Run litert-torch export_hf on one model and record where it stops.

  export_one.py <model_dir> <out_dir> <impl> [--quant RECIPE] [--trust-remote-code] [--prefill 32[,1024]] [--cache 256]
                [--clamp-int4-floor] [--jinja-override FILE] [--tokenizer-verbatim]

--prefill takes a comma list (one prefill signature each). --clamp-int4-floor floors every blockwise 4-bit symmetric
scale at 2^-14 before the weights are quantized (an all-zero or near-zero block otherwise gets a zero or subnormal
scale that XNNPACK refuses); the count of floored scales goes to RESULT.json `clamp`; the int8 recipe does not use it.
--jinja-override passes a template file as jinja_chat_template_override. --tokenizer-verbatim packs
<model_dir>/tokenizer.json byte for byte (the exporter's tokenizer step writes a save_pretrained re-serialization).

impl: litert_moe | litert_moe_sequential | none (no moe_exports_implementation kwarg).
Writes <out_dir>/RESULT.json: ok, stage, first error line, the deepest litert_torch / transformers / model-code
frames, seconds, the .litertlm path and size. Exit 0 = bundle written, 1 = export raised.

Stages (the innermost wrapped step active when the exception left): load (transformers config / class import /
from_pretrained inside export_lib.load_model), experts_rewrite (pre_flatten_model_experts / pre_split_model_experts /
bind_per_expert_scale), export_task (prefill/decode export task before the first trace: exportable module set-up and
sample inputs), torch_export (torch.export.export = forward trace), lowering (after a trace returned: MLIR / TFLite
lowering and serialization), quantization, tokenizer, bundle.
"""
import argparse
import functools
import glob
import json
import os
import sys
import time
import traceback

STACK = ["setup"]


def _wrap(mod, name, stage, after=None):
  fn = getattr(mod, name)

  @functools.wraps(fn)
  def w(*a, **k):
    idx = len(STACK)
    STACK.append(stage)
    out = fn(*a, **k)  # on an exception the stage stays on the stack: the innermost entry names where it stopped
    del STACK[idx:]
    if after:
      STACK.append(after)
    return out

  setattr(mod, name, w)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("model")
  ap.add_argument("out")
  ap.add_argument("impl", choices=["litert_moe", "litert_moe_sequential", "none"])
  ap.add_argument("--quant", default=None)
  ap.add_argument("--trust-remote-code", action="store_true")
  ap.add_argument("--prefill", default="32")
  ap.add_argument("--cache", type=int, default=256)
  ap.add_argument("--clamp-int4-floor", action="store_true")
  ap.add_argument("--jinja-override")
  ap.add_argument("--tokenizer-verbatim", action="store_true")
  a = ap.parse_args()
  prefills = [int(x) for x in str(a.prefill).split(",") if x]
  os.makedirs(a.out, exist_ok=True)
  t0 = time.time()
  res = {"model": a.model, "impl": a.impl, "quant": a.quant, "prefill": prefills, "cache": a.cache,
         "use_v2": False, "trust_remote_code": a.trust_remote_code, "ok": False}
  clamp = {"enabled": a.clamp_int4_floor, "tensors_seen": 0, "tensors": 0, "scales": 0}
  res["clamp"] = clamp
  try:
    import torch
    import transformers
    import litert_torch
    from litert_torch.generative.export_hf import export as export_mod
    from litert_torch.generative.export_hf.core import export_lib, litert_lm_builder, quantization
    from litert_torch.generative.layers import moe as moe_layer
    res["versions"] = {"torch": torch.__version__, "transformers": transformers.__version__,
                       "litert_torch_file": litert_torch.__file__}
    _wrap(export_lib, "load_model", "load")
    _wrap(export_lib, "pre_flatten_model_experts", "experts_rewrite")
    if a.tokenizer_verbatim:
      import dataclasses
      import shutil
      src_tok = os.path.join(a.model, "tokenizer.json")
      if not os.path.isfile(src_tok):
        raise FileNotFoundError(f"--tokenizer-verbatim: no {src_tok}")

      @functools.wraps(export_lib.export_tokenizer)
      def verbatim_tokenizer(source_model_artifacts, export_config, exported_model_artifacts):
        dst = os.path.join(export_config.work_dir, "tokenizer.json")
        shutil.copyfile(src_tok, dst)
        res["tokenizer_verbatim"] = {"source": src_tok, "bytes": os.path.getsize(dst)}
        return dataclasses.replace(exported_model_artifacts, tokenizer_model_path=dst)

      export_lib.export_tokenizer = verbatim_tokenizer
    _wrap(export_lib, "pre_split_model_experts", "experts_rewrite")
    _wrap(moe_layer, "bind_per_expert_scale", "experts_rewrite")
    _wrap(export_lib, "export_text_prefill_decode_model", "export_task")
    _wrap(torch.export, "export", "torch_export", after="lowering")
    _wrap(quantization, "quantize_model", "quantization")
    _wrap(export_lib, "export_tokenizer", "tokenizer")
    _wrap(litert_lm_builder, "package_model", "bundle")
    if a.clamp_int4_floor:
      import numpy as np
      from ai_edge_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor as _u
      floor = np.float32(2.0 ** -14)
      orig = _u.tensor_zp_scale_from_min_max

      def clamped(min_value, max_value, num_bits, symmetric, granularity, *args, **kwargs):
        # the nightly's signature is (…, granularity, clipping_values=None, signed=True); pass the rest through
        zp, scale = orig(min_value, max_value, num_bits, symmetric, granularity, *args, **kwargs)
        if _u.is_blockwise(granularity) and num_bits == 4:
          if not symmetric:
            raise ValueError("clamp-int4-floor: asymmetric blockwise int4 is not covered by this floor")
          clamp["tensors_seen"] += 1
          low = scale < floor
          n = int(np.count_nonzero(low))
          if n:
            scale = np.where(low, floor, scale).astype(scale.dtype)
            clamp["tensors"] += 1
            clamp["scales"] += n
            print(f"CLAMP floor 2^-14: {n} of {scale.size} block scales", flush=True)
        return zp, scale

      _u.tensor_zp_scale_from_min_max = clamped
      print("PATCHED uniform_quantize_tensor.tensor_zp_scale_from_min_max: blockwise int4 scales floored at 2^-14",
            flush=True)
    kw = dict(model=a.model, output_dir=a.out, prefill_lengths=prefills, cache_length=a.cache,
              quantization_recipe=a.quant, trust_remote_code=a.trust_remote_code, use_v2=False)
    if a.impl != "none":
      kw["moe_exports_implementation"] = a.impl
    if a.jinja_override:
      kw["jinja_chat_template_override"] = os.path.abspath(a.jinja_override)
      res["jinja_override"] = kw["jinja_chat_template_override"]
    STACK[:] = ["export"]
    export_mod.export(**kw)
    res["ok"] = True
    res["stage"] = "done"
  except BaseException as e:  # noqa: BLE001
    tb = traceback.extract_tb(e.__traceback__)
    msg = str(e).strip()
    res["stage"] = STACK[-1]
    res["stage_stack"] = list(STACK)
    res["error"] = type(e).__name__ + (": " + msg.splitlines()[0][:600] if msg else "")
    res["error_full_head"] = msg[:2000]

    def last(pred):
      fr = [f for f in tb if pred(f.filename)]
      return f"{fr[-1].filename}:{fr[-1].lineno} in {fr[-1].name}" if fr else None

    res["deepest_litert_torch"] = last(lambda p: "/litert_torch/" in p)
    res["deepest_transformers"] = last(lambda p: "/transformers/" in p)
    res["deepest_model_code"] = last(lambda p: "/transformers/models/" in p or "/hf_cache/modules/" in p
                                     or "transformers_modules" in p)
    res["deepest_frame"] = f"{tb[-1].filename}:{tb[-1].lineno} in {tb[-1].name}" if tb else None
    traceback.print_exc()
  res["seconds"] = round(time.time() - t0, 1)
  bundles = sorted(glob.glob(os.path.join(a.out, "*.litertlm")))
  res["bundles"] = [{"path": b, "bytes": os.path.getsize(b)} for b in bundles]
  with open(os.path.join(a.out, "RESULT.json"), "w") as f:
    json.dump(res, f, indent=1)
  print("EXPORT_RESULT", json.dumps({k: res.get(k) for k in ("ok", "stage", "error", "deepest_litert_torch",
                                                             "deepest_model_code", "seconds", "clamp")}))
  sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
  main()
