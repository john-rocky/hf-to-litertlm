#!/usr/bin/env python3
"""convert_qwen35_hybrid_lw.py HF_ID OUTDIR - qwen35_work/convert_qwen35_hybrid.py for the 9B-class Qwen3.5 hybrids, with these
changes (everything else as in convert_qwen35_hybrid.py):
  * the float export passes --experimental_lightweight_conversion=True (litert-torch export() argument: float32 / int32 constants of
    >= RESOURCE_CONSTANT_NUMEL_THRESHOLD bytes stay lazy through the MLIR lowering, and the folding skipped there is done by the
    runtime constant-folding stage): the default export path of an 8B model peaked past 92 GiB in the converter passes on a
    128 GB Mac;
  * RESOURCE_CONSTANT_NUMEL_THRESHOLD defaults to 100000000 (bytes; litert-torch's default is 1 MiB, compared as numel *
    element_size) when the caller does not set it: at 1 MiB the lightweight export left `MUL(W, scale)` ops unfolded in front of
    FCs on an 8B model, and those weights were not quantized; weights under 100 MB (q/k/v/o, the GatedDeltaNet in_proj_z /
    out_proj / a / b) are then dense and fold as in the default export;
  * SKIP_EXPORT=1 runs steps 2-4 only, on OUTDIR/model.litertlm from an earlier STOP_AFTER_EXPORT=1 run (convert_qwen35_9b.sh runs
    the float export and the quantize as separate steps);
  * `here` = this script's directory (qwen35_work/), so ../minicpm5_work/quantize_minicpm5.py, chat_template_simple.jinja and
    ../scripts/add_executor_metadata.py resolve to the files convert_qwen35_hybrid.py uses;
  * disk: each intermediate is removed as soon as the next step no longer reads it (quantize_minicpm5.py's mc5q_* scratch dir
    after step 2, model_wi8fc.litertlm + wi8fc_unpacked/ after step 3, model_wi8fc_tmpl.litertlm after step 4); the outputs are
    unchanged.

Convert Qwen3.5 (GatedDeltaNet + attention hybrid) to .litertlm.

  # one-time setup: litert-torch at the pinned base + the hybrid patch
  git clone https://github.com/google-ai-edge/litert-torch litert-torch-qwen35
  git -C litert-torch-qwen35 fetch origin 115a13607c730c81018bb9789138a3e5e5119e3d
  git -C litert-torch-qwen35 checkout 115a13607c730c81018bb9789138a3e5e5119e3d
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
QWEN35_FLOAT_FC_REGEX (optional, ';;'-separated regexes) keeps the matching
FULLY_CONNECTED ops float on top of that: the 2B and the 0.8B ship with
'Linear_in_proj_(a|b);' — their GatedDeltaNet decay/gate heads could not take the
CPU backend's run-time int8 activation quantization and looped on multi-step
arithmetic (2B 2026-10-11, 0.8B 2026-10-12; the 4B 2026-10-13, where the same regex keeps
its 48 decay/gate projections float — 24 linear-attention layers).
"""
import os
import re
import subprocess
import sys

model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-0.8B"
outdir = sys.argv[2] if len(sys.argv) > 2 else "out_qwen35"
here = os.path.dirname(os.path.abspath(__file__))  # qwen35_work/: ../minicpm5_work, chat_template_simple.jinja, ../scripts
os.environ.setdefault("RESOURCE_CONSTANT_NUMEL_THRESHOLD", "100000000")  # bytes; see the module docstring


def _drop(*paths):
    """Remove an intermediate as soon as the next step no longer reads it (convert_qwen35_hybrid.py keeps model_wi8fc.litertlm,
    wi8fc_unpacked/ and model_wi8fc_tmpl.litertlm); the outputs are unchanged. Each removal is logged as
    `removed <bytes> <path>`."""
    import shutil
    for p in paths:
        if os.path.isdir(p):
            n = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs)
            shutil.rmtree(p)
        elif os.path.exists(p):
            n = os.path.getsize(p)
            os.remove(p)
        else:
            continue
        print(f"removed {n} {p}", flush=True)


def _drop_quant_scratch():
    """quantize_minicpm5.py's tempfile.mkdtemp(prefix="mc5q_") (the extracted sections + the quantized tflite) is never removed by
    that script; convert_qwen35_9b.sh runs each stage with a separate TMPDIR, so every mc5q_* there is this run's."""
    import glob
    import tempfile
    _drop(*glob.glob(os.path.join(tempfile.gettempdir(), "mc5q_*")))

# 1. Float export (bundled .litertlm), full prefill ladder, 4096-token KV for
#    the 6 attention layers (the 18 gated-delta layers hold constant-size state).
#    QWEN35_PREFILL_LADDER overrides the ladder: every signature is charged
#    engine RAM even if never called, and on a 12 GB iPhone the 4B dies in
#    jetsam at Metal-init #9 of 12 with the full ladder (the 248320-vocab
#    programs are what pushes it past the nemotron-4B 6.7 GB precedent) —
#    the 4B ships 1024,256,64,16,4,1.
argv = [
    "litert-torch", "export_hf",
    "--model", model,
    "--output_dir", outdir,
    "--prefill_lengths",
    os.environ.get("QWEN35_PREFILL_LADDER", "1024,512,256,128,64,32,16,8,4,2,1"),
    "--cache_length", "4096",
    "--quantization_recipe", "",
    "--experimental_lightweight_conversion=True",
]
if not os.environ.get("SKIP_EXPORT"):  # SKIP_EXPORT=1 = steps 2-4 on an existing float export
    from litert_torch.cli import main  # noqa: E402  (after PYTHONPATH is set by caller)

    print("RESOURCE_CONSTANT_NUMEL_THRESHOLD =", os.environ["RESOURCE_CONSTANT_NUMEL_THRESHOLD"], flush=True)
    sys.argv = argv
    rc = main()
    if rc:
        sys.exit(rc)
if os.environ.get("STOP_AFTER_EXPORT"):
    print("DONE (float export only):", os.path.join(outdir, "model.litertlm"))
    sys.exit(0)

fp = os.path.join(outdir, "model.litertlm")
wi8fc = os.path.join(outdir, "model_wi8fc.litertlm")
tmpl = os.path.join(outdir, "model_wi8fc_tmpl.litertlm")
final = os.path.join(outdir, os.path.basename(model).replace("/", "_") + "_int8.litertlm")

# 2. Post-hoc int8 (linears + embedding only; convs/delta-rule float).
#    QWEN35_FLOAT_FC_REGEX keeps named FCs float on top (the 2B / 0.8B: 36 GDN
#    decay/gate projections, the 4B: 48; see the module docstring). The scope a regex is
#    matched against is the output tensor name + ";" (anchor with ";", not "$").
quant_cmd = [sys.executable,
             os.path.join(here, "..", "minicpm5_work", "quantize_minicpm5.py"),
             "apply", fp, wi8fc, "--recipe", "wi8fc"]
for rx in filter(None, os.environ.get("QWEN35_FLOAT_FC_REGEX", "").split(";;")):
    quant_cmd += ["--float-fc-regex", rx]
subprocess.run(quant_cmd, check=True)
_drop_quant_scratch()  # disk

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
# Qwen3.5 splits its chat turn-end (<|im_end|> = 248046) from config.json's
# eos_token_id (<|endoftext|> = 248044); the exporter declares only the latter,
# so <|im_end|> would leak into `litert-lm run` output as text. Declare both.
if "ids: 248046" not in pbtext_new:
    eos_block = "stop_tokens {\n  token_ids {\n    ids: 248044\n  }\n}\n"
    assert eos_block in pbtext_new, "expected <|endoftext|> stop_tokens block"
    pbtext_new = pbtext_new.replace(
        eos_block,
        eos_block + "stop_tokens {\n  token_ids {\n    ids: 248046\n  }\n}\n",
        1)
open(pbtext_path, "w").write(pbtext_new)
subprocess.run(["litert-lm", "pack", unpack_dir,
                "--output", tmpl, "--allow-overwrite"], check=True)
_drop(wi8fc, unpack_dir)  # disk: step 4 reads only tmpl

# 4. Append the ExecutorMetadata section litert-lm >= 0.15 binds states with.
meta_script = next(p for p in (
    os.path.join(here, "..", "scripts", "add_executor_metadata.py"),
    os.path.join(here, "..", "lfm_work", "add_executor_metadata.py"),
) if os.path.exists(p))
subprocess.run([sys.executable, meta_script, tmpl, final], check=True)
_drop(tmpl)  # disk
print("DONE:", final)
