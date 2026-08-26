#!/usr/bin/env python3
"""Convert Zyphra Zamba2 (Mamba2 backbone + shared/tied transformer blocks) to .litertlm.

  PYTHONPATH=~/code/litert-torch .venv-092/bin/python zamba2_work/convert_zamba2.py \
      Zyphra/Zamba2-1.2B-instruct zamba2_work/out_12b

Zamba2 interleaves 'linear_attention' Mamba2 layers with 'hybrid' positions
where a SHARED transformer block (weights tied across positions, per-position
LoRA adapters) runs attention over concat(hidden, embedding) and its output is
projected into the mamba layer input. The scan body is granite's Mamba2 SSD in
a different spelling with NemotronH's min-only dt clamp — the model_ext patch
(model_ext/zamba2/patch.py) reuses NemotronH's folded forward verbatim.

These are Instruct checkpoints (ChatML template): gates are logits parity +
the 8-question instruct gate + the hermetic prompt-length sweep.

Quantization: post-hoc dynamic int8 over linears + embedding only (convs and
the scan stay float) — the standing family rule. For GPU use, declare fp32
activations afterwards: scripts/set_activation_type.py.
"""
import os
import subprocess
import sys

model = sys.argv[1] if len(sys.argv) > 1 else "Zyphra/Zamba2-1.2B-instruct"
outdir = sys.argv[2] if len(sys.argv) > 2 else "zamba2_work/out_zamba2"
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Full 11-length ladder by default; override for memory-limited phones
# (signature-count RAM law: every exported signature costs RAM even unused —
# the 2.7B full ladder jetsams an iPhone 17 Pro at GPU load, and the AGX
# compiled-variants footprint limit trips on 12 signatures).
ladder = (os.environ.get("PREFILL_LENGTHS")            # generic knob convert.py sets
          or os.environ.get("ZAMBA2_PREFILL_LADDER",   # documented family-specific name
                            "1024,512,256,128,64,32,16,8,4,2,1"))

argv = [
    "litert-torch", "export_hf",
    "--model", model,
    "--output_dir", outdir,
    "--prefill_lengths", ladder,
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

subprocess.run([sys.executable,
                os.path.join(repo_root, "minicpm5_work", "quantize_minicpm5.py"),
                "apply", fp, wi8fc, "--recipe", "wi8fc"], check=True)

subprocess.run([sys.executable,
                os.path.join(repo_root, "scripts", "add_executor_metadata.py"),
                wi8fc, final], check=True)

# Zamba2's metaspace tokenizer + the runtime's per-token streaming decode eat
# every interior space unless the Strip decoder is dropped (see
# fix_tokenizer_strip.py for the mechanism).
fixed = final + ".tokfix"
subprocess.run([sys.executable,
                os.path.join(repo_root, "zamba2_work", "fix_tokenizer_strip.py"),
                final, fixed], check=True,
               env={**os.environ,
                    "PATH": os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]})
os.replace(fixed, final)
print("DONE:", final)
