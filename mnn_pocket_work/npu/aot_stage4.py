#!/usr/bin/env python3
"""Compile MNN-Pocket-0.3B's NPU file ahead of time for the HTP of the Qualcomm SM8850 with litert-torch's LiteRT-LM NPU
compiler, inside the image of lfm3audio_work/npu/Dockerfile (linux/amd64: ai-edge-litert 2.2.0 and its Qualcomm compiler
plugin with QAIRT 2.47.0.260601). aot_sm8850.sh runs it in a container of that image.

    python aot_stage4.py IN.litertlm OUTDIR TREE

TREE holds the three compiler files of litert-torch at commit 6d4c622 at their paths in the litert_torch package
(generative/export_hf/experimental/litert_lm_npu_compiler/: __init__.py, litert_lm_npu_compiler.py,
litert_lm_npu_compiler_configs.py); PYTHONPATH holds litert-lm-builder 0.17.0, absl-py 2.5.0 and tomli 2.4.1. The image
has no torch, so the parent packages of the compiler are made empty modules here (litert_torch's own __init__.py imports
torch); the compiler files are imported unchanged.

The compile: compile_litertlm(backend="qualcomm", soc_model="SM8850") with the compiler's own options for that backend and
disable_aux_compilation=True, so the graphs of the aux section (rotary tables, masks, cache updates) stay as they are and
run on the phone's CPU. Writes OUTDIR/model_qualcomm_SM8850.litertlm, keeps the output of each call of the plugin's
apply_plugin_main in OUTDIR/apply_plugin_<n>.log, prints the partition lines, the result's size and the peak memory;
exit 1 on a failure."""
import os
import resource
import subprocess
import sys
import time
import traceback
import types

IN, OUT_DIR, TREE = sys.argv[1], sys.argv[2], sys.argv[3]
for name in ("litert_torch", "litert_torch.generative", "litert_torch.generative.export_hf",
             "litert_torch.generative.export_hf.experimental"):
  stub = types.ModuleType(name)
  stub.__path__ = [os.path.join(TREE, *name.split("."))]
  sys.modules[name] = stub

from absl import logging as absl_logging  # noqa: E402
from ai_edge_litert.aot.core import apply_plugin  # noqa: E402
from litert_torch.generative.export_hf.experimental.litert_lm_npu_compiler.litert_lm_npu_compiler import compile_litertlm  # noqa: E402

absl_logging.set_verbosity(absl_logging.INFO)
os.makedirs(OUT_DIR, exist_ok=True)
OUT = os.path.join(OUT_DIR, "model_qualcomm_SM8850.litertlm")
_run, _calls = subprocess.run, [0]


def run_logged(args, **kwargs):
  """apply_plugin's subprocess.run: the same call, its whole output kept in a file and its verdict lines printed."""
  _calls[0] += 1
  t = time.time()
  r = _run(args, **kwargs)
  log = os.path.join(OUT_DIR, f"apply_plugin_{_calls[0]}.log")
  with open(log, "w") as fh:
    fh.write(" ".join(str(a) for a in args) + "\n" + (r.stdout or ""))
  print(f"apply_plugin call {_calls[0]}: rc {r.returncode} in {time.time() - t:.0f} s, {len((r.stdout or '').splitlines())} lines"
        f" -> {log}; args {[str(a) for a in args[1:]]}", flush=True)
  for line in (r.stdout or "").splitlines():
    if any(k in line for k in ("Partitioned subgraph", "compiler plugin", "ERROR", "Error", "error:", "FATAL", "QNN")):
      print("  | " + line[:400], flush=True)
  return r


apply_plugin.subprocess = types.SimpleNamespace(run=run_logged, PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT)
print("=== STAGE 4 START ===", IN, os.path.getsize(IN), flush=True)
t = time.time()
rc = 0
try:
  compile_litertlm(input_litertlm=IN, output_litertlm=OUT, backend="qualcomm", soc_model="SM8850", compile_configs=None,
                   intermediate_dir=os.path.join(OUT_DIR, "intermediates"), disable_aux_compilation=True)
  print(f"=== STAGE 4 OK in {time.time() - t:.0f}s ===", flush=True)
  print("artifact:", OUT, os.path.getsize(OUT))
except BaseException as e:  # noqa: BLE001 - the failure line is the result
  traceback.print_exc()
  print(f"=== STAGE 4 FAILED in {time.time() - t:.0f}s: {type(e).__name__}: {e} ===")
  rc = 1
finally:
  inter = os.path.join(OUT_DIR, "intermediates")
  for f in sorted(os.listdir(inter)) if os.path.isdir(inter) else []:
    q = os.path.join(inter, f)
    print(f"intermediate {f} {os.path.getsize(q) if os.path.isfile(q) else 'dir'}")
  print(f"peak RSS: self {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss} kB, children "
        f"{resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss} kB", flush=True)
sys.exit(rc)
