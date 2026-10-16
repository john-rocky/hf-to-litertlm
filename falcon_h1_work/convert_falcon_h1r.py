#!/usr/bin/env python3
"""convert_falcon_h1r.py HF_ID OUTDIR [FINAL] - tiiuae/Falcon-H1R-7B (falcon_h1: 44 layers of attention ∥ Mamba2, self-emitted
<think>…</think> reasoning) to .litertlm on the Falcon-H1 rail of this directory.

  PYTHONPATH=<litert-torch @ 115a136 + falcon_h1_litert_torch.patch> RUN_PY=<python with litert-lm 0.18.0> \\
      HF_HUB_CACHE=<cache holding the pinned commit> HF_HUB_OFFLINE=1 python convert_falcon_h1r.py tiiuae/Falcon-H1R-7B out_7b

The flow of convert_falcon_h1.py (float export -> post-hoc int8 -> ExecutorMetadata) and convert_falcon_tinyr.py (the reasoning
ship form: a thought channel added after export, stop [11, 228], the start token decided against the HF token stream), with these
changes for a 7B model:
  1. float export: --prefill_lengths 1024,256,64,16,4,1 (6 signatures), --cache_length 4096,
     --experimental_lightweight_conversion=True (litert-torch 115a136 export() argument: constants of
     >= RESOURCE_CONSTANT_NUMEL_THRESHOLD bytes stay lazy through the MLIR lowering; the default export of an 8B model peaked past
     92 GiB on a 128 GiB Mac) with RESOURCE_CONSTANT_NUMEL_THRESHOLD=100000000 unless set (smaller weights are dense and fold as in
     the default export; at the 1 MiB default a lightweight export left MUL(W, scale) in front of FCs and those weights were not
     quantized); the vendor template is bundled (--use_jinja_template True) and replaced in step 3.
     STOP_AFTER_EXPORT=1 stops here (run the export and each quantization as separate jobs: each needs the free RAM to itself).
  2. post-hoc quantization, ../minicpm5_work/quantize_minicpm5.py apply --recipe RECIPE: wi8fc = dynamic int8 on every
     FULLY_CONNECTED + int8 channelwise embedding (convs and the scan stay float), wi4b32_wi8 = int4 blockwise-32 min-max FCs +
     int8 lm_head + int8 embedding, run through quantize_clamped.py (blockwise int4 scales floored at 2^-14 before quantizing;
     without it this model's int4 file carries zero / subnormal block scales that XNNPACK refuses on the CPU).
  3. LlmMetadata: make_pbtext_h1r.py (jinja_prompt_template = falcon_h1r_dual.jinja, start_token cleared when NO_START_TOKEN=1
     (default; the HF token stream has no BOS), supports_thinking = true) + ../granite_work/repack_metadata.py (unpack / pack with
     the litert-lm CLI, every other section verified, the changed LlmMetadata fields == the expected ones both ways).
  4. ../scripts/add_executor_metadata.py (the k/v + mamba conv / recurrent state buffers bind through ExecutorMetadata).
  5. ../scripts/set_activation_type.py --type fp32 (the GPU decoder's default fp16 activations answer nothing on this model).
  6. ../tools/add_thought_channel.py --start '<think>' --end '</think>' (self-emitted reasoning: the template never pre-opens the
     channel) -> FINAL (default OUTDIR/<NAME>_<int8|int4>.litertlm).
SKIP_EXPORT=1 runs steps 2-6 on OUTDIR/model.litertlm. Intermediates go to FH1R_WORK (default OUTDIR/work_<tag>), one directory
per step, each removed as soon as the next step has read it. Every subprocess command is printed as `### CMD: …` and, when
FH1R_RECIPE_LOG is set, appended to that file.
Env: RECIPE (wi8fc | wi4b32_wi8, default wi8fc), FH1R_NAME (default the HF id's basename), FH1R_TEMPLATE (default
falcon_h1r_dual.jinja next to this script), NO_START_TOKEN (1 | 0), FH1R_PREFILL (default 1024,256,64,16,4,1), RUN_PY (a python
with litert-lm 0.18.0, default `python`), LITERT_LM (the litert-lm 0.18.0 CLI, default `litert-lm`).
"""
import glob
import os
import shlex
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RPY = os.environ.get("RUN_PY", "python")
LT = os.environ.get("LITERT_LM", "litert-lm")

model = sys.argv[1] if len(sys.argv) > 1 else "tiiuae/Falcon-H1R-7B"
outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.getcwd(), "out_7b")
recipe = os.environ.get("RECIPE", "wi8fc")
tag = {"wi8fc": "int8", "wi4b32_wi8": "int4"}[recipe]
name = os.environ.get("FH1R_NAME") or os.path.basename(model.rstrip("/"))
final = sys.argv[3] if len(sys.argv) > 3 else os.path.join(outdir, f"{name}_{tag}.litertlm")
work = os.environ.get("FH1R_WORK") or os.path.join(outdir, f"work_{tag}")
template = os.environ.get("FH1R_TEMPLATE") or os.path.join(HERE, "falcon_h1r_dual.jinja")
no_start = os.environ.get("NO_START_TOKEN", "1") == "1"
os.environ.setdefault("RESOURCE_CONSTANT_NUMEL_THRESHOLD", "100000000")


def log_cmd(argv, env_note=""):
    line = "### CMD: " + (env_note + " " if env_note else "") + " ".join(shlex.quote(a) for a in argv)
    print(line, flush=True)
    rl = os.environ.get("FH1R_RECIPE_LOG")
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
    "--prefill_lengths", os.environ.get("FH1R_PREFILL", "1024,256,64,16,4,1"),
    "--cache_length", "4096",
    "--bundle_litert_lm", "True",
    "--use_jinja_template", "True",
    "--quantization_recipe", "",
    "--experimental_lightweight_conversion=True",
]
if not os.environ.get("SKIP_EXPORT"):
    from litert_torch.cli import main  # noqa: E402  (after PYTHONPATH is set by caller)

    log_cmd(["python", os.path.abspath(__file__)] + sys.argv[1:],
            f"RESOURCE_CONSTANT_NUMEL_THRESHOLD={os.environ['RESOURCE_CONSTANT_NUMEL_THRESHOLD']} STOP_AFTER_EXPORT=1")
    print("export argv:", " ".join(shlex.quote(a) for a in argv[2:]), flush=True)
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
# steps 4-6 unpack / pack the quantized bundle (5-8 GB): their scratch goes under FH1R_WORK, not the system TMPDIR
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

# 2. Post-hoc quantization of the prefill_decode TFLite section. int4: through quantize_clamped.py = the same script with
#    ai-edge-quantizer's blockwise int4 scales floored at 2^-14 before quantizing (a plain build carried 96 zero + 748 subnormal
#    block scales in Mamba-mixer projections, which XNNPACK refuses).
qscript = (os.path.join(HERE, "quantize_clamped.py") if recipe.startswith("wi4")
           else os.path.join(ROOT, "minicpm5_work", "quantize_minicpm5.py"))
run([sys.executable, qscript, "apply", fp, q, "--recipe", recipe])
drop(*glob.glob(os.path.join(tempfile.gettempdir(), "mc5q_*")))

# 3. LlmMetadata (template, start token, supports_thinking), verified re-pack.
pb = os.path.join(d2, "h1r.pbtext")
run([RPY, "-I", os.path.join(HERE, "make_pbtext_h1r.py"), q, template, pb] + (["--no-start-token"] if no_start else []))
import json  # noqa: E402

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

# 6. Thought channel (self-emitted <think>…</think>).
run([sys.executable, os.path.join(ROOT, "tools", "add_thought_channel.py"), f32, final, "--start", "<think>", "--end", "</think>"],
    env=dict(wenv, LITERT_LM_BIN=LT), env_note=f"LITERT_LM_BIN={LT}")
drop(f32, d4, wtmp)
print("DONE:", final, os.path.getsize(final))
