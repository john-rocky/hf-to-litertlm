#!/usr/bin/env python3
"""export_q35.py MODEL_DIR OUT_DIR VARIANT PREFILL CACHE - litert-torch's export_hf with the extension q35moe_ext.py
(each layer's experts as one `moe` custom op), fp32 (no quantization), LiteRT-LM's own qwen3_5 metadata
(template/LlmMetadataProto.pbtext, LiteRT-LM v0.19.1, fetched by reproduce.sh).

VARIANT: plain (no GPU composites: apply_gpu_composites False, rope composite off, GDN short-conv composite off; left =
CUSTOM gated_delta_update and CUSTOM moe) or gpu (main's GPU form, apply_gpu_composites True). PREFILL = comma list of
prefill lengths, CACHE = KV length.
Env defaults set here and recorded: Q35MOE_EXPERTS (graft for the build: 4-element placeholders that graft_moe.py replaces; fp32 writes the experts in
fp32), Q35MOE_FUSION_GUARD=2 (the residual add after each moe op as shared + ((moe * 0.5) + (moe - moe * 0.5)), exact
in fp32, so a GPU delegate does not fuse the add into the moe kernel), Q35MOE_GDN_VALID_LEN=0 (6-input GDN op),
Q35MOE_SHARED_3D=1 for the gpu variant (the shared expert on the rank-3 input) and 0 for plain, Q35_NORM_PERM_ONCE=1
for the gpu variant (the q / k norm weights get litert-torch's partial-rotary permutation once, not twice:
install_norm_perm_once below).
export_hf writes executor_metadata.pb / llm_metadata.pb only inside its bundling step; the bundle itself (a full copy of
model.tflite plus export_hf's rewritten tokenizer, never shipped: make_bundle.py packs the repo's tokenizer bytes)
is skipped here (Q35_SKIP_BUNDLE=1, default): `_build_litert_lm_file` writes nothing, the metadata files stay.
Writes OUT_DIR/export_q35.json (arguments, env, wall time, peak RSS). Exit = 0 on success.
"""
import json
import os
import resource
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(HERE, "q35moe_ext.py")
PBTEXT = os.path.join(HERE, "template", "LlmMetadataProto.pbtext")


def install_norm_perm_once():
  """Makes litert-torch's checkpoint loader return q_norm / k_norm unpermuted (the gpu variant).

  litert-torch 30d0be0 (2026-10-28) permutes q_norm / k_norm by the partial-rotary permutation in
  HFCheckpointWeightsLoader._try_load when use_qkv_norm_rope_composite is set, and the Qwen3.5 patch's
  _build_composite_rms_norm permutes them again when it rebuilds the norm from the loaded weights. On this path (the
  extension's model class has no from_pretrained, so export_lib.load_model copies every parameter from the loader before
  the patch runs) the export held (1 + w)[perm[perm]] in every odml.qkv_norm_rope norm input. With the two norms loaded
  unpermuted, the patch's permutation is the only one, as on the q / k rows.
  """
  from litert_torch.generative.export_hf.core import weights_loader as wl
  orig_try_load = wl.HFCheckpointWeightsLoader._try_load  # pylint: disable=protected-access
  norm_names = (".self_attn.q_norm.weight", ".self_attn.k_norm.weight")

  def _try_load_norm_unpermuted(self, param_name):
    if self._strip_prefixes(param_name, wl._WRAPPER_PREFIXES).endswith(norm_names):  # pylint: disable=protected-access
      self._get_qwen3_5_partial_rope_perm = lambda: None  # the loader's own "no permutation" branch
      try:
        return orig_try_load(self, param_name)
      finally:
        del self._get_qwen3_5_partial_rope_perm
    return orig_try_load(self, param_name)

  wl.HFCheckpointWeightsLoader._try_load = _try_load_norm_unpermuted  # pylint: disable=protected-access


def main():
  model, out, variant = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), sys.argv[3]
  prefill = [int(x) for x in sys.argv[4].split(",")]
  cache = int(sys.argv[5])
  os.makedirs(out, exist_ok=True)
  kw = dict(model=model, output_dir=out, extension_modules=[EXT], prefill_lengths=prefill, cache_length=cache,
            quantization_recipe="none", keep_temporary_files=True, bundle_litert_lm=True, use_jinja_template=True,
            litert_lm_llm_metadata_override=PBTEXT)
  if variant == "plain":
    os.environ["Q35MOE_SHORT_CONV_COMPOSITE"] = "0"
    kw.update(apply_gpu_composites=False, use_rope_composite=False)
  elif variant == "gpu":
    kw.update(apply_gpu_composites=True)
  else:
    raise SystemExit(f"unknown variant {variant}")
  os.environ.setdefault("Q35MOE_GDN_VALID_LEN", "0")
  os.environ.setdefault("Q35MOE_EXPERTS", "fp32")
  os.environ.setdefault("Q35MOE_FUSION_GUARD", "2")
  # the shared expert on the rank-3 input in the GPU form only (q35moe_ext.py docstring); the plain form keeps the
  # flattened input
  os.environ.setdefault("Q35MOE_SHARED_3D", "1" if variant == "gpu" else "0")
  skip = os.environ.setdefault("Q35_SKIP_BUNDLE", "1") == "1"
  from litert_torch.generative.export_hf import export as export_lib  # after the env is set
  if variant == "gpu" and os.environ.setdefault("Q35_NORM_PERM_ONCE", "1") == "1":
    install_norm_perm_once()
  if skip:
    import dataclasses
    from litert_torch.generative.export_hf.core import litert_lm_builder as llb

    def _no_bundle(builder, export_config, artifacts, output_dir, work_dir):
      del builder, export_config, work_dir
      marker = os.path.join(output_dir, "model.litertlm.skipped")
      with open(marker, "w") as f:
        f.write("export_hf's bundle was not written (export_q35.py, Q35_SKIP_BUNDLE=1)\n")
      return dataclasses.replace(artifacts, litert_lm_model_path=None)

    llb._build_litert_lm_file = _no_bundle  # pylint: disable=protected-access
  t0 = time.time()
  export_lib.export(**kw)
  for need in ("model.tflite", "executor_metadata.pb", "llm_metadata.pb"):
    if not os.path.exists(os.path.join(out, need)):
      raise SystemExit(f"missing {need} in {out}")
  rec = {"variant": variant, "kwargs": kw, "env": {k: v for k, v in os.environ.items() if k.startswith(("Q35MOE_", "Q35_"))},
         "wall_s": round(time.time() - t0, 1),
         "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2),
         "tflite_bytes": os.path.getsize(os.path.join(out, "model.tflite"))}
  with open(os.path.join(out, "export_q35.json"), "w") as f:
    json.dump(rec, f, indent=1)
  print("EXPORT_DONE", json.dumps(rec))


if __name__ == "__main__":
  main()
