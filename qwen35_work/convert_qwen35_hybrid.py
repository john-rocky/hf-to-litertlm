#!/usr/bin/env python3
"""Convert Qwen3.5 (GatedDeltaNet + attention hybrid) to .litertlm.

  # one-time setup: litert-torch at the pinned base + the hybrid patch
  git clone https://github.com/google-ai-edge/litert-torch litert-torch-qwen35
  git -C litert-torch-qwen35 checkout 115a136
  git -C litert-torch-qwen35 apply "$(pwd)/qwen35_hybrid_litert_torch.patch"

  # convert (float export -> post-hoc int8 on linears+embedding -> template -> metadata)
  PYTHONPATH=litert-torch-qwen35 python convert_qwen35_hybrid.py \
      Qwen/Qwen3.5-0.8B out_qwen35_08b

Requires litert-lm >= 0.15 (both to package and to run: the gated-delta conv +
recurrent state buffers bind through the ExecutorMetadata section) plus
litert-torch's deps, transformers >= 5.14, and ai-edge-quantizer.

The patch adds: a gated-delta-net export-cache layer wired to
layer_types == "linear_attention", state-continuation tracing (decode traces the
fused single-step branch, prefill the chunked-continuation branch, so
multi-chunk prefill composes), the prefill-pad guard (the runtime's chunk
planner runs prefill chunks partially filled; pads must be identity steps for
the delta rule and the stored conv window must end at the last VALID column),
and a constant-eye vendored chunk kernel (the traced `torch.eye` otherwise
lowers to STABLEHLO_IOTA, which no released TFLite kernel set registers).

The bundled chat template is replaced with a minijinja-safe simple ChatML
template that keeps the empty `<think>` block in BOTH the generation prompt and
history assistant renders — the stock template strips it from history, which
violates the engine's incremental-render prefix contract and kills turn 2.

Quantization is post-hoc dynamic int8 over linears + embedding ONLY (convs and
the delta-rule stay float), the same house rule as LFM2.5 / Granite-4-h convs.
"""
import os
import re
import subprocess
import sys

model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-0.8B"
outdir = sys.argv[2] if len(sys.argv) > 2 else "out_qwen35"
here = os.path.dirname(os.path.abspath(__file__))

# 1. Float export (bundled .litertlm), full prefill ladder, 4096-token KV for
#    the 6 attention layers (the 18 gated-delta layers hold constant-size state).
argv = [
    "litert-torch", "export_hf",
    "--model", model,
    "--output_dir", outdir,
    "--prefill_lengths", "1024,512,256,128,64,32,16,8,4,2,1",
    "--cache_length", "4096",
    "--quantization_recipe", "",
]
from litert_torch.cli import main  # noqa: E402  (after PYTHONPATH is set by caller)

sys.argv = argv
rc = main()
if rc:
    sys.exit(rc)

fp = os.path.join(outdir, "model.litertlm")
wi8fc = os.path.join(outdir, "model_wi8fc.litertlm")
tmpl = os.path.join(outdir, "model_wi8fc_tmpl.litertlm")
final = os.path.join(outdir, os.path.basename(model).replace("/", "_") + "_int8.litertlm")

# 2. Post-hoc int8 (linears + embedding only; convs/delta-rule float).
subprocess.run([sys.executable,
                os.path.join(here, "..", "minicpm_work", "quantize_litertlm.py"),
                "apply", fp, wi8fc, "--recipe", "wi8fc"], check=True)

# 3. Replace the bundled template with the simple ChatML template (see module
#    docstring): unpack -> rewrite jinja_prompt_template in the pbtext -> pack.
unpack_dir = os.path.join(outdir, "wi8fc_unpacked")
subprocess.run(["litert-lm", "unpack", wi8fc, "--output-dir", unpack_dir,
                "--allow-overwrite"], check=True)
template = open(os.path.join(here, "chat_template_simple.jinja")).read()
esc = template.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
pbtext_path = os.path.join(unpack_dir, "LlmMetadataProto.pbtext")
pbtext = open(pbtext_path).read()
pbtext_new = re.sub(r'jinja_prompt_template: ".*"',
                    lambda m: 'jinja_prompt_template: "' + esc + '"',
                    pbtext, count=1)
assert pbtext_new != pbtext, "jinja_prompt_template not found in pbtext"
open(pbtext_path, "w").write(pbtext_new)
subprocess.run(["litert-lm", "pack", unpack_dir,
                "--output", tmpl, "--allow-overwrite"], check=True)

# 4. Append the ExecutorMetadata section litert-lm >= 0.15 binds states with.
meta_script = next(p for p in (
    os.path.join(here, "..", "scripts", "add_executor_metadata.py"),
    os.path.join(here, "..", "lfm_work", "add_executor_metadata.py"),
) if os.path.exists(p))
subprocess.run([sys.executable, meta_script, tmpl, final], check=True)
print("DONE:", final)
