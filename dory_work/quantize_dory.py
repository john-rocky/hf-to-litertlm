"""quantize_dory.py FLOAT_DIR OUT_DIR RECIPE TEMPLATE [--int4-floor] - export_hf's own quantize step on the float export's
model.tflite and embedder.tflite (litert-torch 0.9.4: export_lib.maybe_quantize_model = ai-edge-quantizer with RECIPE,
which writes <stem>_quantized.tflite next to the file it is given), plus the LiteRT-LM metadata with the runtime chat
template.

  1. llm_metadata_t.pb = the export's llm_metadata.pb with jinja_prompt_template replaced by TEMPLATE (every other field
     kept).
  2. RECIPE on model.tflite and embedder.tflite: dynamic_wi8_afp32 (int8 weights, float activations) or a recipe JSON
     (recipe_int4_b32.json: int4 blockwise-32 FCs, int8 embedding).
     --int4-floor: every blockwise 4-bit symmetric scale is floored at 2^-14 (the fp16 min normal) before the weights
     are quantized with it (XNNPACK refuses a zero or subnormal fp16 block scale); channelwise / int8 requests pass
     through unchanged. Prints CLAMP_TOTAL at exit.
Writes OUT_DIR/{llm_metadata.pb, executor_metadata.pb, llm_metadata_t.pb, model_quantized.tflite,
embedder_quantized.tflite}.

Usage: python dory_work/quantize_dory.py <float dir> <out dir> <dynamic_wi8_afp32 | recipe.json> <template.jinja>
       [--int4-floor]
"""
import atexit
import hashlib
import os
import shutil
import sys
import time


def sha(path):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for b in iter(lambda: f.read(64 << 20), b""):
      h.update(b)
  return h.hexdigest()


def install_int4_floor():
  import numpy as np  # pylint: disable=g-import-not-at-top
  from ai_edge_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor as u  # pylint: disable=g-import-not-at-top
  floor = np.float32(2.0 ** -14)
  orig = u.tensor_zp_scale_from_min_max
  stats = {"tensors": 0, "scales": 0, "blockwise_calls": 0}

  def clamped(min_value, max_value, num_bits, symmetric, granularity, clipping_values=None):
    zp, scale = orig(min_value, max_value, num_bits, symmetric, granularity, clipping_values)
    if u.is_blockwise(granularity) and num_bits == 4:
      if not symmetric:
        raise ValueError("quantize_dory: asymmetric blockwise int4 is not covered by the floor")
      stats["blockwise_calls"] += 1
      low = scale < floor
      n = int(np.count_nonzero(low))
      if n:
        scale = np.where(low, floor, scale).astype(scale.dtype)
        stats["tensors"] += 1
        stats["scales"] += n
        print(f"CLAMP floor 2^-14: {n} of {scale.size} block scales", flush=True)
    return zp, scale

  u.tensor_zp_scale_from_min_max = clamped
  atexit.register(lambda: print(f"CLAMP_TOTAL tensors={stats['tensors']} scales={stats['scales']} "
                                f"blockwise_int4_calls={stats['blockwise_calls']}", flush=True))
  print("PATCHED uniform_quantize_tensor.tensor_zp_scale_from_min_max: blockwise int4 scales floored at 2^-14",
        flush=True)


def main(argv):
  fdir, out, recipe, template = argv[:4]
  flags = argv[4:]
  if [f for f in flags if f != "--int4-floor"]:
    raise SystemExit(__doc__)
  if "--int4-floor" in flags:
    install_int4_floor()
  from litert_torch.generative.export_hf.core import export_lib  # pylint: disable=g-import-not-at-top
  from litert_lm_builder.runtime.proto import llm_metadata_pb2  # pylint: disable=g-import-not-at-top
  os.makedirs(out, exist_ok=True)
  for name in ("llm_metadata.pb", "executor_metadata.pb"):
    shutil.copyfile(os.path.join(fdir, name), os.path.join(out, name))
  meta = llm_metadata_pb2.LlmMetadata()
  meta.ParseFromString(open(os.path.join(out, "llm_metadata.pb"), "rb").read())
  old_len = len(meta.jinja_prompt_template)
  meta.jinja_prompt_template = open(template).read()
  meta_t = os.path.join(out, "llm_metadata_t.pb")
  open(meta_t, "wb").write(meta.SerializeToString())
  print(f"llm_metadata_t.pb: jinja_prompt_template {old_len} -> {len(meta.jinja_prompt_template)} chars from "
        f"{template}; {os.path.getsize(meta_t)} B sha256 {sha(meta_t)}", flush=True)
  for stem in ("model", "embedder"):
    src = os.path.abspath(os.path.join(fdir, f"{stem}.tflite"))
    link = os.path.join(out, f"{stem}.tflite")
    if os.path.lexists(link):
      os.remove(link)
    os.symlink(src, link)          # quantize_model writes <path>_quantized.tflite next to the path it is given
    t0 = time.time()
    q = export_lib.maybe_quantize_model(link, recipe)
    os.remove(link)
    want = os.path.join(out, f"{stem}_quantized.tflite")
    if os.path.abspath(q) != os.path.abspath(want):
      raise SystemExit(f"quantize_dory: quantize_model wrote {q}, expected {want}")
    print(f"QUANTIZED {stem}: {q} {os.path.getsize(q)} B sha256 {sha(q)} in {time.time() - t0:.1f} s", flush=True)
  print("QUANTIZE_DONE", flush=True)
  return 0


if __name__ == "__main__":
  if len(sys.argv) < 5:
    raise SystemExit(__doc__)
  sys.exit(main(sys.argv[1:]))
