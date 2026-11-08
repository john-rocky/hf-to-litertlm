#!/usr/bin/env python3
"""Ahead-of-time compile one graph for the Hexagon NPU of the Qualcomm SM8850 (Galaxy S26) with ai-edge-litert's Qualcomm
compiler plugin, inside the image of the Dockerfile next to this file (linux/amd64). Extra arguments are passed to the
plugin's apply_plugin_main as --<flag> (qualcomm_log_level=verbose prints the lines aot_check.py reads). Keeps the plugin's
whole output (OUTDIR/apply_plugin.log), writes OUTDIR/aot.json (wall time, peak memory, the compilation report, the output
files) and prints the report.

    docker run --rm --platform linux/amd64 --cpus 4 --memory 28g --memory-swap 28g -v "$PWD/work":/m:ro \
        -v "$PWD/conversion/npu":/s:ro -v "$PWD/work/npu":/w npu-aot-qc:2.3.0-q250 \
        python /s/aot_compile_sm8850.py /m/mnn-pocket-0.3b-dec_L64_fp16w_npu.tflite /w/L64 qualcomm_log_level=verbose

The output is OUTDIR/<model name>_Qualcomm_SM8850_apply_plugin.tflite.
"""
import json
import os
import resource
import subprocess
import sys
import time

from ai_edge_litert.aot.aot_compile import aot_compile
from ai_edge_litert.aot.core import apply_plugin
from ai_edge_litert.aot.vendors.qualcomm import target as qc

model, outdir, flags = sys.argv[1], sys.argv[2], sys.argv[3:]
os.makedirs(outdir, exist_ok=True)
plugin_log = os.path.join(outdir, "apply_plugin.log")
_run = subprocess.run
calls = []


def run(args, **kw):
    """subprocess.run as apply_plugin calls it, plus the extra flags and the whole output kept in a file."""
    if args and str(args[0]).endswith("apply_plugin_main"):
        args = list(args) + [f"--{f}" for f in flags]
        t0 = time.time()
        res = _run(args, **kw)
        wall = time.time() - t0
        with open(plugin_log, "w") as f:
            f.write("ARGS " + " ".join(str(a) for a in args) + "\n")
            f.write(res.stdout or "")
            f.write(f"\nRC {res.returncode} WALL_S {wall:.1f}\n")
        calls.append({"args": [str(a) for a in args], "rc": res.returncode, "wall_s": round(wall, 1)})
        return res
    return _run(args, **kw)


apply_plugin.subprocess.run = run  # aot_compile's Qualcomm backend passes only its libraries to the plugin binary
t0 = time.time()
err = None
try:
    res = aot_compile(model, output_dir=outdir, target=[qc.Target(soc_model=qc.SocModel.SM8850)], keep_going=True)
    report = res.compilation_report()
except Exception as e:  # recorded, not raised
    res, report, err = None, None, f"{type(e).__name__}: {e}"
wall = time.time() - t0
files = {f: os.path.getsize(os.path.join(outdir, f)) for f in sorted(os.listdir(outdir)) if f != "aot.json"}
peak = None
try:
    peak = int(open("/sys/fs/cgroup/memory.peak").read().strip())
except OSError:
    pass
doc = {"model": model, "model_bytes": os.path.getsize(model), "target": "Qualcomm SM8850", "flags": flags,
       "wall_s": round(wall, 1), "peak_rss_children_kB": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
       "cgroup_memory_peak_B": peak, "plugin_calls": calls, "error": err,
       "report": str(report) if report is not None else None, "outputs": files}
with open(os.path.join(outdir, "aot.json"), "w") as f:
    json.dump(doc, f, indent=1)
print(json.dumps({k: v for k, v in doc.items() if k != "report"}), flush=True)
print(report, flush=True)
sys.exit(1 if err or not any(c["rc"] == 0 for c in calls) else 0)
