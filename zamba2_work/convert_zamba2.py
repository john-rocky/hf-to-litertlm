#!/usr/bin/env python3
"""Convert Zyphra Zamba2 (Mamba2 backbone + shared/tied transformer blocks) to .litertlm.

  PYTHONPATH=litert-torch-zamba2 python convert_zamba2.py \
      Zyphra/Zamba2-1.2B-instruct out_zamba2_12b

Zamba2 interleaves Mamba2 selective-scan layers with 'hybrid' positions where
a SHARED transformer block (one set of weights tied across positions,
per-position LoRA adapters) runs attention over concat(hidden, embedding) and
its output is projected into the mamba layer input. The patch reuses the
NemotronH folded rank<=4 scan verbatim (same mixer attribute names, same
min-only dt clamp) — see zamba2_litert_torch.patch.

Quantization: post-hoc dynamic int8 over linears + embedding only (convs and
the scan stay float). For GPU use, declare fp32 activations afterwards:
scripts/set_activation_type.py.
"""
import os
import subprocess
import sys

model = sys.argv[1] if len(sys.argv) > 1 else "Zyphra/Zamba2-1.2B-instruct"
outdir = sys.argv[2] if len(sys.argv) > 2 else "out_zamba2"
here = os.path.dirname(os.path.abspath(__file__))

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

# 4. Zamba2's metaspace (SP-BPE) tokenizer loses every interior space under
# the runtime's per-token streaming decode unless the Strip decoder is
# dropped from the bundled tokenizer.json (see fix_tokenizer_strip.py).
fixed = final + ".tokfix"
subprocess.run([sys.executable,
                os.path.join(here, "fix_tokenizer_strip.py"),
                final, fixed], check=True)
os.replace(fixed, final)
print("DONE:", final)
