#!/usr/bin/env python3
"""convert_nemotron_nano_9b.py HF_ID OUTDIR [FINAL] - nvidia/NVIDIA-Nemotron-Nano-9B-v2 (nemotron_h hybrid, 56 layers = 27 Mamba2 +
25 MLP + 4 attention; reasoning switched by `/think` / `/no_think` in the system or user text) to .litertlm on the Nemotron-H rail.
convert_nemotron_nano_9b.sh calls it once for the float export and once per quantized file.

  PYTHONPATH=<litert-torch @ 115a136 + nemotron_h_litert_torch.patch + nemotron_h_vendor_dt.patch> RUN_PY=<litert-lm 0.18.0 python> \\
      HF_HUB_CACHE=<cache with the pinned commit> HF_HUB_OFFLINE=1 python convert_nemotron_nano_9b.py nvidia/NVIDIA-Nemotron-Nano-9B-v2 out

The flow of convert_nemotron_h.py (float export -> post-hoc quantization -> ExecutorMetadata) with the changes a 9B needs:
  1. float export: --prefill_lengths 1024,256,64,16,4,1 (six prefill signatures + decode), --cache_length 4096,
     --experimental_lightweight_conversion=True (constants of >= RESOURCE_CONSTANT_NUMEL_THRESHOLD bytes stay lazy through the MLIR
     lowering and are folded by the LiteRT runtime afterwards; the default export of an 8B model needs well over 90 GB of RAM) with
     RESOURCE_CONSTANT_NUMEL_THRESHOLD=100000000 unless set (smaller constants fold as in a default export; at a 1 MiB threshold a
     lightweight export can leave MUL(W, scale) in front of FCs whose weights then stay unquantized); the vendor template is bundled
     here and replaced in step 3. STOP_AFTER_EXPORT=1 stops after this step.
  2. post-hoc quantization, minicpm5_work/quantize_minicpm5.py apply --recipe RECIPE: wi8fc = dynamic int8 on every
     FULLY_CONNECTED + int8 channelwise embedding (the convolutions and the selective scan stay float), wi4b32_wi8 = int4
     blockwise-32 min-max FCs + int8 lm_head + int8 embedding, run through quantize_clamped.py (blockwise int4 scales floored at
     2^-14 before quantizing; a no-op when every scale is normal; on this model it floors 66,388 zero or subnormal scales).
  3. metadata: make_pbtext.py (jinja_prompt_template = nemotron_nano_dual.jinja, no start token (the tokenizer's
     add_bos_token is false and the template writes no `<s>`; the exporter writes start_token `<s>` from tokenizer.bos_token),
     supports_thinking = true) + granite_work/repack_metadata.py (unpack / pack with the litert-lm 0.18.0 CLI, every non-metadata
     section verified, the changed LlmMetadata fields == the expected ones both ways).
  4. scripts/add_executor_metadata.py (the key / value caches and the Mamba2 conv / SSM state buffers bind through ExecutorMetadata).
  5. scripts/set_activation_type.py --type fp32 (the GPU decoder's default activation type is fp16; with it this model answers
     `<unk>` from the first token).
  6. tools/add_thought_channel.py --start '<think>\\n' --end '\\n</think>' (the template's generation prompt ends with `<think>\\n`
     when thinking is on and `<think></think>` when `/no_think` switched it off; the channel's start is the template's text)
     -> FINAL (default OUTDIR/<NAME>_<int8|int4>.litertlm).
SKIP_EXPORT=1 runs steps 2-6 on OUTDIR/model.litertlm. Intermediates go to NN9B_WORK (default OUTDIR/work_<tag>), one directory per
step, each removed as soon as the next step has read it. Every subprocess command is printed as `### CMD: …` and, when
NN9B_RECIPE_LOG is set, appended to that file.
Env: RECIPE (wi8fc | wi4b32_wi8, default wi8fc), NN9B_NAME (default the HF id's basename), NN9B_TEMPLATE (default
nemotron_nano_dual.jinja next to this file), NO_START_TOKEN (1 | 0, default 1), NN9B_PREFILL (default 1024,256,64,16,4,1),
RUN_PY (a python with litert-lm 0.18.0, default `python`), LITERT_LM (the litert-lm 0.18.0 CLI, default next to RUN_PY).
"""
import glob
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RPY = os.environ.get("RUN_PY") or shutil.which("python") or "python"
LT = os.environ.get("LITERT_LM") or os.path.join(os.path.dirname(RPY), "litert-lm")
CH_START, CH_END = "<think>\n", "\n</think>"

model = sys.argv[1] if len(sys.argv) > 1 else "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
outdir = sys.argv[2] if len(sys.argv) > 2 else "out_nemotron_nano_9b"
recipe = os.environ.get("RECIPE", "wi8fc")
tag = {"wi8fc": "int8", "wi4b32_wi8": "int4"}[recipe]
name = os.environ.get("NN9B_NAME") or os.path.basename(model.rstrip("/"))
final = sys.argv[3] if len(sys.argv) > 3 else os.path.join(outdir, f"{name}_{tag}.litertlm")
work = os.environ.get("NN9B_WORK") or os.path.join(outdir, f"work_{tag}")
template = os.environ.get("NN9B_TEMPLATE") or os.path.join(HERE, "nemotron_nano_dual.jinja")
no_start = os.environ.get("NO_START_TOKEN", "1") == "1"
os.environ.setdefault("RESOURCE_CONSTANT_NUMEL_THRESHOLD", "100000000")


def log_cmd(argv, env_note=""):
    line = "### CMD: " + (env_note + " " if env_note else "") + " ".join(shlex.quote(x) for x in argv)
    print(line, flush=True)
    rl = os.environ.get("NN9B_RECIPE_LOG")
    if rl:
        with open(rl, "a") as f:
            f.write(line + "\n")


def run(argv, env=None, env_note=""):
    log_cmd(argv, env_note)
    subprocess.run(argv, check=True, env=env)


def drop(*paths):
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


# 1. Float export (bundled .litertlm, vendor template, no quantization).
argv = [
    "litert-torch", "export_hf",
    "--model", model,
    "--output_dir", outdir,
    "--prefill_lengths", os.environ.get("NN9B_PREFILL", "1024,256,64,16,4,1"),
    "--cache_length", "4096",
    "--bundle_litert_lm", "True",
    "--use_jinja_template", "True",
    "--quantization_recipe", "",
    "--experimental_lightweight_conversion=True",
]
if not os.environ.get("SKIP_EXPORT"):
    from litert_torch.cli import main  # noqa: E402  (after PYTHONPATH is set by the caller)

    log_cmd(["python", os.path.abspath(__file__)] + sys.argv[1:],
            f"RESOURCE_CONSTANT_NUMEL_THRESHOLD={os.environ['RESOURCE_CONSTANT_NUMEL_THRESHOLD']} STOP_AFTER_EXPORT=1")
    print("export argv:", " ".join(shlex.quote(x) for x in argv[2:]), flush=True)
    sys.argv = argv
    rc = main()
    if rc:
        sys.exit(rc)
if os.environ.get("STOP_AFTER_EXPORT"):
    print("DONE (float export only):", os.path.join(outdir, "model.litertlm"))
    sys.exit(0)

fp = os.path.join(outdir, "model.litertlm")
assert os.path.exists(fp), fp
d1, d2, d2s, d3, d4 = (os.path.join(work, x) for x in ("1_quant", "2_meta", "2_scratch", "3_exec", "4_fp32"))
for d in (d1, d2, d3, d4):
    os.makedirs(d, exist_ok=True)
# steps 4-6 unpack / pack the quantized bundle (5-9 GB): their scratch goes under NN9B_WORK, not the system TMPDIR
wtmp = os.path.join(work, "tmp")
os.makedirs(wtmp, exist_ok=True)
wenv = dict(os.environ, TMPDIR=wtmp)
q = os.path.join(d1, "q.litertlm")
m = os.path.join(d2, "m.litertlm")
e = os.path.join(d3, "e.litertlm")
f32 = os.path.join(d4, "f.litertlm")
for p in (q, m, e, f32, final):
    if os.path.exists(p):
        raise SystemExit(f"refusing to overwrite {p}")

# 2. Post-hoc quantization of the prefill_decode TFLite section (int4 through quantize_clamped.py: blockwise int4 scales floored
#    at 2^-14 before quantizing; the int8 recipe is channelwise).
qscript = (os.path.join(HERE, "quantize_clamped.py") if recipe.startswith("wi4")
           else os.path.join(ROOT, "minicpm5_work", "quantize_minicpm5.py"))
run([sys.executable, qscript, "apply", fp, q, "--recipe", recipe])
drop(*glob.glob(os.path.join(tempfile.gettempdir(), "mc5q_*")))

# 3. LlmMetadata (template, start token, supports_thinking), verified re-pack.
pb = os.path.join(d2, "metadata.pbtext")
run([RPY, "-I", os.path.join(HERE, "make_pbtext.py"), q, template, pb] + (["--no-start-token"] if no_start else []))
chk = json.load(open(pb + ".check.json"))
run([RPY, "-I", os.path.join(ROOT, "granite_work", "repack_metadata.py"), q, m, "--pbtext", pb,
     "--report", os.path.join(d2, "report.json"), "--expect-changed", ",".join(chk["expected_changed_fields"]),
     "--cli", LT, "--scratch-parent", d2s])
drop(q, d1, d2s)

# 4. ExecutorMetadata (state-buffer binding).
run([sys.executable, os.path.join(ROOT, "scripts", "add_executor_metadata.py"), m, e, "--litert-lm", LT], env=wenv)
drop(m)

# 5. fp32 activations declared in the bundle.
run([sys.executable, os.path.join(ROOT, "scripts", "set_activation_type.py"), e, f32, "--type", "fp32", "--litert-lm", LT],
    env=wenv)
drop(e, d3)

# 6. Thought channel (the template opens it with `<think>\n`).
run([sys.executable, os.path.join(ROOT, "tools", "add_thought_channel.py"), f32, final, "--start", CH_START, "--end", CH_END],
    env=dict(wenv, LITERT_LM_BIN=LT), env_note=f"LITERT_LM_BIN={LT}")
drop(f32, d4, wtmp)
print("DONE:", final, os.path.getsize(final))
