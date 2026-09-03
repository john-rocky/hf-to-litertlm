#!/usr/bin/env python3
"""Convert Qwen3.5 to a speculative-decoding (MTP) .litertlm — P1: base re-export.

Adds on top of the shipped qwen35 hybrid recipe (convert_qwen35_hybrid.py):
ring-addressed gated-delta state (R = G+2), a `verify` signature (embeddings in,
per-position fp32 logits + activations out), `activations` on decode, an
externalized embedder section, and the drafter's teacher-forced K/V + hidden
ring written by prefill/decode/verify. Design + derivations: P1_DESIGN.md.

  # one-time setup: litert-torch at the qwen35 pin + the COMBINED patch
  # (qwen35_mtp_litert_torch.patch supersedes qwen35_hybrid_litert_torch.patch:
  #  it contains the full hybrid recipe plus the MTP additions — apply INSTEAD
  #  of the qwen35 patch, on a clean 115a136)
  git clone https://github.com/google-ai-edge/litert-torch litert-torch-mtp
  git -C litert-torch-mtp checkout 115a136
  git -C litert-torch-mtp apply "$(pwd)/mtp_work/qwen35_mtp_litert_torch.patch"

  # float export (P1 gates run on the tflite; bundle+int8 after gates pass)
  PYTHONPATH=~/code/litert-torch-mtp ~/venvs/lt094dev/bin/python3 \
      mtp_work/convert_qwen35_mtp.py Qwen/Qwen3.5-0.8B mtp_work/out_08b

Env knobs: MTP_G (default 3), MTP_LADDER (default full ship ladder),
MTP_BUNDLE=1 to also build model.litertlm in the output dir.
"""
import os
import sys

model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-0.8B"
outdir = sys.argv[2] if len(sys.argv) > 2 else "mtp_work/out_qwen35_mtp"

G = int(os.environ.get("MTP_G", "3"))
ladder = [int(x) for x in os.environ.get(
    "MTP_LADDER", "1024,512,256,128,64,32,16,8,4,2,1").split(",")]
bundle = os.environ.get("MTP_BUNDLE", "0") == "1"

from litert_torch.generative.export_hf import export as export_mod  # noqa: E402

export_mod.export(
    model=model,
    output_dir=outdir,
    prefill_lengths=ladder,
    cache_length=4096,
    quantization_recipe="",
    externalize_embedder=True,
    bundle_litert_lm=bundle,
    keep_temporary_files=True,
    mtp_speculative={"num_draft_steps": G, "ring_size": G + 2},
)
print("DONE:", outdir)
