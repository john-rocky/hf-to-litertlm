#!/usr/bin/env python3
"""Convert NVIDIA Nemotron-H (Mamba2 + MLP + attention hybrid) to .litertlm.

  # one-time setup: litert-torch at the pinned base + the hybrid patch
  git clone https://github.com/google-ai-edge/litert-torch litert-torch-nemotron
  git -C litert-torch-nemotron fetch origin 115a13607c730c81018bb9789138a3e5e5119e3d
  git -C litert-torch-nemotron checkout 115a13607c730c81018bb9789138a3e5e5119e3d
  git -C litert-torch-nemotron apply "$(pwd)/nemotron_h_litert_torch.patch"

  # convert (float export -> post-hoc int8 on linears+embedding -> executor metadata)
  PYTHONPATH=litert-torch-nemotron python convert_nemotron_h.py \
      nvidia/Nemotron-H-4B-Instruct-128K out_nemotron_4b

Requires litert-lm >= 0.15 (both to package and to run: the hybrid state
buffers bind through the ExecutorMetadata section) plus litert-torch's deps,
transformers >= 5.14, and ai-edge-quantizer.

Three-kind hybrid (the 4B = 24 mamba + 24 mlp + 4 attention layers). The folded
scan is a PORT, not blind reuse: NemotronH's torch_forward is an older SSD
spelling with a min-only dt clamp — pads must be forced to dt=0 post-clamp or
partially-filled prefill chunks decay the state. `mlp` layers ride a dedicated
no-state cache layer (zero state tensors, absolute layer indexing preserved).
NemotronHBlock constructs mixers from the module-level MIXER_TYPES dict; the
patch swaps the dict entry and guards loudly — a class-attribute swap alone
silently exports the unrewritten reference scan, and parity cannot tell.

Quantization is post-hoc dynamic int8 over linears + embedding ONLY (convs and
the scan stay float) — the same rule as granite/Qwen3.5. For GPU use, declare
fp32 activations afterwards: ../scripts/set_activation_type.py.
"""
import os
import subprocess
import sys

model = sys.argv[1] if len(sys.argv) > 1 else "nvidia/Nemotron-H-4B-Instruct-128K"
outdir = sys.argv[2] if len(sys.argv) > 2 else "out_nemotron_4b"
here = os.path.dirname(os.path.abspath(__file__))

# 1. Float export (bundled .litertlm), full prefill ladder, 4096-token KV
#    budget for the 4 attention layers (mamba state is constant-size, mlp
#    layers hold no state at all).
argv = [
    "litert-torch", "export_hf",
    "--model", model,
    "--output_dir", outdir,
    # Full 11-length ladder by default; convert.py sets PREFILL_LENGTHS to the
    # reduced 7-signature ladder for ≥3B models (signature-count RAM law).
    "--prefill_lengths", os.environ.get("PREFILL_LENGTHS",
                                        "1024,512,256,128,64,32,16,8,4,2,1"),
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

# 2. Post-hoc int8 (linears + embedding only; convs/scan float).
subprocess.run([sys.executable,
                os.path.join(here, "..", "minicpm_work", "quantize_litertlm.py"),
                "apply", fp, wi8fc, "--recipe", "wi8fc"], check=True)

# 3. Append the ExecutorMetadata section litert-lm >= 0.15 binds states with.
subprocess.run([sys.executable,
                os.path.join(here, "..", "lfm_work", "add_executor_metadata.py"),
                wi8fc, final], check=True)
print("DONE:", final)
