"""export_dory.py MODEL_DIR OUT_DIR SHIFT_JSON PREFILL_LENGTHS - the float export of NVIDIA-Nemotron-Labs-Dory1.0-4B for
LiteRT-LM with litert-torch's export_hf: the model of dory_modeling.py (registered with transformers' Auto classes, the
looped KV slots of its DoryLoopCache hook), a 2048-token KV cache, the prefill lengths given (comma separated), the
embedding table in its own embedder.tflite, recipe "none" (float; the string: None would fall through to
dynamic_wi8_afp32), the repo's own tokenizer.json and chat_template.jinja, no start token.

The MLP justnorm of the blocks listed in SHIFT_JSON (fp16_shift.json) is rewritten so that a GPU running the graph with
fp16 activations does not overflow its sum of squares:
  down_proj(justnorm(m))  ->  ms = m * 2^-k ; y = ms * minimum(rsqrt(sum(ms * ms) + eps * 4^-k), 2^k / sqrt(eps)) ;
                              down_proj(y)
In fp32 these are the same numbers: scaling by a power of two is exact, the sum of squares scales by 4^-k term by
term, rsqrt(4^-k s) = 2^k rsqrt(s), and the cap is never reached (rsqrt(sum + eps) <= 1 / sqrt(eps)); in fp16 storage
the sum of squares shrinks by 4^k. The cap keeps an all-zero row at 0 where eps * 4^-k rounds to 0 in fp16.

export_hf's package step writes llm_metadata.pb and executor_metadata.pb into OUT_DIR and would then write a ~16 GB
float model.litertlm; that write is skipped (quantize_dory.py packs the quantized bundles from the files kept here:
model.tflite, embedder.tflite, llm_metadata.pb, executor_metadata.pb).

Usage: python dory_work/export_dory.py <model dir> <out dir> dory_work/fp16_shift.json 128,512,1024
"""
import importlib.util
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))


def load_dory_modeling():
  # the module must be importable as `dory_modeling`: its name is part of every tensor name in the graph
  path = os.path.join(HERE, "dory_modeling.py")
  spec = importlib.util.spec_from_file_location("dory_modeling", path)
  mod = importlib.util.module_from_spec(spec)
  sys.modules["dory_modeling"] = mod
  spec.loader.exec_module(mod)
  return mod


def main(model_dir, out_dir, shift_json, prefill):
  lengths = [int(x) for x in prefill.split(",") if x]
  import litert_lm_builder as lb  # pylint: disable=g-import-not-at-top

  def skip_build(self, fh):  # pylint: disable=unused-argument
    print(f"NO_BUNDLE: LitertLmFileBuilder.build skipped ({getattr(fh, 'name', fh)})", flush=True)

  lb.LitertLmFileBuilder.build = skip_build
  dm = load_dory_modeling()
  shifts = {k: int(v) for k, v in json.load(open(shift_json))["shift"].items() if int(v) > 0}
  if not shifts:
    raise SystemExit("export_dory: no block has a shift > 0 = nothing to rewrite")
  seen = {}
  orig_init = dm.DoryBlock.__init__

  def block_init(self, config, layer_idx):
    orig_init(self, config, layer_idx)
    if not self.is_attention:
      self.mixer._fp16_shift = shifts.get(f"L{layer_idx}", 0)
      if self.mixer._fp16_shift:
        seen[f"L{layer_idx}"] = self.mixer._fp16_shift

  def justnorm_shifted(x, k):
    xs = dm._acc(x) * (2.0 ** -k)
    r = torch.rsqrt((xs * xs).sum(dim=-1, keepdim=True) + dm.EPS * (4.0 ** -k))
    return (xs * torch.clamp(r, max=2.0 ** k / math.sqrt(dm.EPS))).to(x.dtype)

  def mlp_forward(self, x):
    # dory_modeling.DoryMLP.forward with the justnorm swapped for blocks with a shift
    g = self.gate_proj(x) * (dm._acc(self.suv_gate) * self._suv_factor).to(x.dtype)
    u = self.up_proj(x) * (dm._acc(self.suv_up) * self._suv_factor).to(x.dtype)
    m = F.silu(g) * u
    k = getattr(self, "_fp16_shift", 0)
    return self.down_proj(justnorm_shifted(m, k) if k else dm.justnorm(m))

  dm.DoryBlock.__init__ = block_init
  dm.DoryMLP.forward = mlp_forward
  cache_impl = dm.install_export_hooks(strip_start_token=True)
  from litert_torch.generative.export_hf.export import export  # pylint: disable=g-import-not-at-top

  t0 = time.time()
  export(
      model=model_dir,
      output_dir=out_dir,
      prefill_lengths=lengths,
      cache_length=2048,
      quantization_recipe="none",
      cache_implementation=cache_impl,
      tokenizer_path_override=os.path.join(model_dir, "tokenizer.json"),
      jinja_chat_template_override=os.path.join(model_dir, "chat_template.jinja"),
      externalize_embedder=True,
      keep_temporary_files=True,
  )
  if seen != shifts:
    raise SystemExit(f"export_dory: rewritten blocks {seen} != requested {shifts}")
  out = os.path.join(out_dir, "model.litertlm")
  if os.path.exists(out) and os.path.getsize(out) == 0:
    os.remove(out)
    print(f"NO_BUNDLE: removed the empty {out}", flush=True)
  missing = []
  for name in ("model.tflite", "embedder.tflite", "llm_metadata.pb", "executor_metadata.pb"):
    p = os.path.join(out_dir, name)
    size = os.path.getsize(p) if os.path.exists(p) else None
    missing += [name] if size is None else []
    print(f"KEPT {p} {size if size is not None else 'MISSING'}", flush=True)
  if missing:
    raise SystemExit(f"export_dory: missing {missing}")
  print(f"EXPORT_DONE {time.time() - t0:.1f} s -> {out_dir}; prefill {lengths}; rewritten MLP blocks {seen}", flush=True)
  return 0


if __name__ == "__main__":
  if len(sys.argv) != 5:
    raise SystemExit(__doc__)
  sys.exit(main(*sys.argv[1:5]))
