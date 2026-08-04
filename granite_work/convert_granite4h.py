#!/usr/bin/env python3
"""Convert IBM granite-4.0-h dense-hybrid models (Mamba2 + attention) to .litertlm.

  # one-time setup: litert-torch at the pinned base + the hybrid patch
  git clone https://github.com/google-ai-edge/litert-torch litert-torch-granite
  git -C litert-torch-granite checkout 115a136
  git -C litert-torch-granite apply "$(pwd)/granite_hybrid_litert_torch.patch"

  # convert (float export -> post-hoc int8 on linears+embedding -> executor metadata)
  PYTHONPATH=litert-torch-granite python convert_granite4h.py \
      ibm-granite/granite-4.0-h-1b out_granite_1b

Requires litert-lm >= 0.15 (both to package and to run: the hybrid conv/SSM
state buffers bind through the ExecutorMetadata section, which the 0.14 engine
does not read) plus litert-torch's deps, transformers >= 5.14, and
ai-edge-quantizer. The patch adds: a Mamba2 export-cache layer, decode-state
continuation + chunked-prefill continuation tracing, and the mamba prefill-pad
guard (the runtime's chunk planner runs partially-filled prefill chunks; the
guard makes pad positions identity steps and gathers the conv window at the
last valid column — without it, generation corrupts at chunk-plan-dependent
prompt lengths).

Quantization is post-hoc dynamic int8 over linears + embedding ONLY (convs and
the SSM stay float): export-time conv-int8 measurably costs quality on this
family (8Q sanity 5/8 vs 6/8 on the 350m), the same rule as the LFM2.5 convs.
"""
import os
import subprocess
import sys

model = sys.argv[1] if len(sys.argv) > 1 else "ibm-granite/granite-4.0-h-1b"
outdir = sys.argv[2] if len(sys.argv) > 2 else "out_granite4h"
here = os.path.dirname(os.path.abspath(__file__))

# 1. Float export (bundled .litertlm), full prefill ladder, 4096-token KV for
#    the 4 attention layers (the 28 mamba layers hold constant-size state).
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
