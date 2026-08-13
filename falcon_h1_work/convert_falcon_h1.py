#!/usr/bin/env python3
"""Convert TII Falcon-H1 (attention + Mamba2 in parallel, every layer) to .litertlm.

  # one-time setup: litert-torch at the pinned base + the hybrid patch
  git clone https://github.com/google-ai-edge/litert-torch litert-torch-falcon
  git -C litert-torch-falcon checkout 115a136
  git -C litert-torch-falcon apply "$(pwd)/falcon_h1_litert_torch.patch"

  # convert (float export -> post-hoc int8 on linears+embedding -> executor metadata)
  PYTHONPATH=litert-torch-falcon python convert_falcon_h1.py \
      tiiuae/Falcon-H1-0.5B-Instruct out_falcon_05b

Requires litert-lm >= 0.15 (both to package and to run: the hybrid state
buffers bind through the ExecutorMetadata section) plus litert-torch's deps,
transformers >= 5.14, and ai-edge-quantizer.

Every FalconH1 layer runs a grouped-query attention branch and a Mamba2 SSD
branch in parallel on the same normalized input, so every layer carries KV and
conv/recurrent state at the same index — the patch adds a composite cache
layer that is full-attention and Mamba2 at the same time, flattened as
k_i/v_i/mc_i/mr_i. The scan body is granite's verbatim (asserted at patch
time); the wrapper adds ssm_in_multiplier, the mup_vector buffer
(non-persistent — zero-checked after load), and the conditional gate norm.

Quantization is post-hoc dynamic int8 over linears + embedding ONLY (convs and
the scan stay float) — the same rule as granite/Qwen3.5. For GPU use, declare
fp32 activations afterwards: ../scripts/set_activation_type.py.
"""
import os
import subprocess
import sys

model = sys.argv[1] if len(sys.argv) > 1 else "tiiuae/Falcon-H1-0.5B-Instruct"
outdir = sys.argv[2] if len(sys.argv) > 2 else "out_falcon_05b"
here = os.path.dirname(os.path.abspath(__file__))

# 1. Float export (bundled .litertlm), full prefill ladder, 4096-token KV
#    budget for the attention branches (the SSM state is constant-size).
argv = [
    "litert-torch", "export_hf",
    "--model", model,
    "--output_dir", outdir,
    "--prefill_lengths", "1024,512,256,128,64,32,16,8,4,2,1",
    "--cache_length", "4096",
    "--bundle_litert_lm", "True",
    "--use_jinja_template", "True",
    "--quantization_recipe", "",
]
from litert_torch.cli import main  # noqa: E402  (after PYTHONPATH is set by caller)

sys.argv = argv
rc = main()
if rc:
    sys.exit(rc)

fp = os.path.join(outdir, "model.litertlm")
wi8fc = os.path.join(outdir, "model_wi8fc.litertlm")
final = os.path.join(outdir, os.path.basename(model) + "_int8.litertlm")

# 2. Post-hoc int8 (linears + embedding only; convs/SSM float).
subprocess.run([sys.executable,
                os.path.join(here, "..", "minicpm_work", "quantize_litertlm.py"),
                "apply", fp, wi8fc, "--recipe", "wi8fc"], check=True)

# 3. Append the ExecutorMetadata section litert-lm >= 0.15 binds states with.
subprocess.run([sys.executable,
                os.path.join(here, "..", "lfm_work", "add_executor_metadata.py"),
                wi8fc, final], check=True)
print("DONE:", final)
