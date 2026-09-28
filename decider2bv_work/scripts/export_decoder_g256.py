"""Float (fp32) export of decider-2b-vision's text decoder for the fast_vlm contract, with the derived M-RoPE rotary.

  EMBEDDER       : token_ids [1,1] -> embeddings [1,1,2048]                 (single_token_embedder)
  PREFILL_DECODE : embeddings + input_pos (1-D) + mask + 48 states -> states (+ logits on decode)

Converter = the patched clone out/litert-torch-d2bv (john-rocky/litert-torch 115a13607c73 + qwen35_work/
qwen35_hybrid_litert_torch.patch, sha256 0a01e2ae...), put ahead of the installed wheel via PYTHONPATH. The only
change on top of the qwen35 patch is process-local: scripts/mrope_derived.install() swaps the patch's 1-D
PatchedQwen3_5TextRotaryEmbedding.forward for the derived (t, h, w) form before the model is loaded; nothing on
disk is modified. Ladder = the iPhone 6-signature one (qwen35vl_work/FINDINGS.md), cache 4096, no quantization.

    PYTHONPATH=out/litert-torch-d2bv PYTHONDONTWRITEBYTECODE=1 \
      out/venv-export/bin/python -B -u scripts/export_decoder_g256.py [--out DIR] [--step-form relu_diff|clamp]
                                                                              (from decider2bv_work/)

Round 2 = `--step-form clamp --out out/decoder_g256_fp32` (the derived rotary's step as clamp -> RELU_0_TO_1).
Round 4 = `--step-form relu_diff --out out/decoder_g256_r4_fp32` (relu(x) - relu(x - 1), no RELU_0_TO_1); every other
argument is unchanged. The driver refuses an output dir that already holds a .tflite.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
SNAPSHOT = os.path.join(ROOT, "out/src/decider-2b-vision")
CLONE = os.path.join(ROOT, "out/litert-torch-d2bv")
OUT = os.path.join(ROOT, "out/decoder_g256_fp32")
PREFILL = [1024, 256, 64, 16, 4, 1]
CACHE = 4096


def sha256_file(p):
  with open(p, "rb") as f:
    return hashlib.file_digest(f, "sha256").hexdigest()


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", default=OUT)
  ap.add_argument("--step-form", default=None, help="mrope_derived step form (default: the module's STEP_FORM)")
  args = ap.parse_args()
  out_dir = os.path.abspath(args.out)
  t0 = time.monotonic()
  os.makedirs(out_dir, exist_ok=True)
  assert not any(f.endswith(".tflite") for f in os.listdir(out_dir)), f"refusing to overwrite the tflites in {out_dir}"
  rec = {"status": "RUNNING", "snapshot": SNAPSHOT, "prefill_lengths": PREFILL, "cache_length": CACHE,
         "argv": sys.argv[1:]}
  try:
    import numpy as np
    import torch
    import transformers
    import litert_torch
    import importlib.metadata as md
    assert os.path.realpath(litert_torch.__file__).startswith(os.path.realpath(CLONE)), litert_torch.__file__
    rec["versions"] = {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__,
                       **{p: md.version(p) for p in ("litert-torch", "litert-converter", "ai-edge-litert", "ai-edge-quantizer")}}
    rec["litert_torch_imported_from"] = os.path.dirname(litert_torch.__file__)
    rec["clone"] = {"head": subprocess.check_output(["git", "-C", CLONE, "rev-parse", "HEAD"], text=True).strip(),
                    "diff_stat": subprocess.check_output(["git", "-C", CLONE, "diff", "--stat"], text=True).strip().splitlines()[-1],
                    "untracked": subprocess.check_output(["git", "-C", CLONE, "status", "--short"], text=True).strip().splitlines()}
    rec["patch_sha256"] = sha256_file(os.path.join(ROOT, "../qwen35_work/qwen35_hybrid_litert_torch.patch"))
    assert rec["patch_sha256"] == "0a01e2ae9f6bbb0aa79b1ba7de343bad3c4bf17b919a2ccd21c912ede2885aa5"
    rec["mrope_derived_sha256"] = sha256_file(os.path.join(HERE, "mrope_derived.py"))

    from litert_torch.generative.export_hf.model_ext.qwen3_5 import patch as qpatch
    import mrope_derived
    if args.step_form:
      mrope_derived.set_step_form(args.step_form)
    rec["step_form"] = mrope_derived.STEP_FORM
    replaced = mrope_derived.install(qpatch)
    assert qpatch.PatchedQwen3_5TextRotaryEmbedding.forward is mrope_derived.derived_forward
    rec["rotary_replaced"] = f"{replaced.__module__}.{replaced.__qualname__} -> mrope_derived.derived_forward"

    # pre-export check in THIS venv: the patched class (5.14.1 base) now yields the derived cos/sin; compare with the
    # HF 5.17.0 reference written by scripts/test_mrope_derived.py over the whole cache
    cfg = transformers.AutoConfig.from_pretrained(SNAPSHOT)
    rot = qpatch.PatchedQwen3_5TextRotaryEmbedding(config=cfg.text_config)
    inv_ref = np.load(os.path.join(ROOT, "out/mrope_ref_5170_inv_freq.npy"))
    pos = torch.arange(CACHE, dtype=torch.int32)[None, :]
    pos4 = torch.stack([pos, pos, pos], dim=0)
    with torch.no_grad():
      c, s = rot(torch.zeros(1), pos4)
    cref = np.load(os.path.join(ROOT, "out/mrope_ref_5170_cos4096.npy"))
    sref = np.load(os.path.join(ROOT, "out/mrope_ref_5170_sin4096.npy"))
    rec["precheck_vs_5170"] = {
        "inv_freq_equal": bool(np.array_equal(rot.inv_freq.numpy(), inv_ref)),
        "mrope_section": list(rot.mrope_section),
        "cos_max_abs": float(np.abs(c.numpy() - cref).max()), "sin_max_abs": float(np.abs(s.numpy() - sref).max()),
        "cos_bit_equal": bool(np.array_equal(c.numpy(), cref)), "sin_bit_equal": bool(np.array_equal(s.numpy(), sref))}
    print("precheck vs 5.17.0:", rec["precheck_vs_5170"], flush=True)
    assert rec["precheck_vs_5170"]["inv_freq_equal"]
    assert max(rec["precheck_vs_5170"]["cos_max_abs"], rec["precheck_vs_5170"]["sin_max_abs"]) <= 1e-6

    from litert_torch.generative.export_hf.export import export
    rec["export_args"] = dict(model=SNAPSHOT, output_dir=out_dir, quantization_recipe="", externalize_embedder=True,
                              single_token_embedder=True, cache_length=CACHE, prefill_lengths=PREFILL,
                              bundle_litert_lm=False, keep_temporary_files=True, use_jinja_template=False,
                              trust_remote_code=False)
    with open(os.path.join(out_dir, "export_driver.json"), "w") as f:
      json.dump(rec, f, indent=1)
    export(**rec["export_args"])
    rec["tflites"] = {f: {"bytes": os.path.getsize(os.path.join(out_dir, f))} for f in sorted(os.listdir(out_dir)) if f.endswith(".tflite")}
    rec["status"] = "DONE"
  except BaseException as e:  # noqa: BLE001
    rec["status"] = "ERROR"
    rec["error_type"] = type(e).__name__
    rec["traceback"] = traceback.format_exc()
    print(rec["traceback"], flush=True)
  rec["wall_seconds_contended"] = time.monotonic() - t0
  with open(os.path.join(out_dir, "export_driver.json"), "w") as f:
    json.dump(rec, f, indent=1)
  print("EXPORT_DRIVER", rec["status"], f"{rec['wall_seconds_contended']:.0f}s", flush=True)
  if rec["status"] != "DONE":
    sys.exit(1)


if __name__ == "__main__":
  main()
