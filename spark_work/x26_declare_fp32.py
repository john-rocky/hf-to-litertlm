#!/usr/bin/env python3
"""Spark-X2.6: a copy of a bundle whose prefill_decode TFLiteModel section declares `prefer_activation_type = fp32`
(the activation type of the GPU executor; LiteRT-LM's default for the text decoder is fp16), every section's bytes
unchanged. The litert-lm 0.19.0 CLI unpacks the declaration as an `additional_metadata` entry of the section, a form
scripts/set_activation_type.py's pattern does not reach (with that CLI, on a file that already carried the item, it
printed OK and the header kept the old value), so this script writes that entry itself into the unpacked model.toml
(refusing a section that already declares one), packs with the 0.19.0 CLI and checks the result with
gpu_graph_retrofit.py's bundle_sections (loaded by path): same section count, every section byte-identical, the
prefill_decode section's items = the source's + prefer_activation_type fp32, nothing else.

  LITERT_LM_CLI=<litert-lm 0.19.0 CLI> python -B spark_work/x26_declare_fp32.py <src.litertlm> <dst.litertlm> <report.json>
"""
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
CLI = os.environ.get("LITERT_LM_CLI", "litert-lm")
TOOL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "gpu_graph", "gpu_graph_retrofit.py")
src, dst, report = sys.argv[1:4]
spec = importlib.util.spec_from_file_location("gpu_graph_retrofit_tool", TOOL)
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)

work = dst + ".unpack"
shutil.rmtree(work, ignore_errors=True)  # the 0.19.0 CLI writes nothing into an existing directory
p = subprocess.run([CLI, "unpack", src, "--output-dir", work], stdin=subprocess.DEVNULL, capture_output=True, text=True)
if p.returncode != 0:
  sys.exit(f"unpack failed: {p.stdout[-500:]} {p.stderr[-500:]}")
toml_path = os.path.join(work, "model.toml")
toml = open(toml_path).read()
blocks = toml.split("[[section]]")
hits = [i for i, b in enumerate(blocks) if re.search(r'model_type = "(?:tf_lite_)?prefill_decode"', b, re.I)]
if len(hits) != 1:
  sys.exit(f"expected one prefill_decode section, found {len(hits)}")
blk = blocks[hits[0]]
if "prefer_activation_type" in blk:
  sys.exit("the prefill_decode section already declares prefer_activation_type: " + blk)
blocks[hits[0]] = ('\nadditional_metadata = [\n  { key = "prefer_activation_type", value_type = "String", value = "fp32" },\n]'
                   + blk)
open(toml_path, "w").write("[[section]]".join(blocks))
if os.path.exists(dst):
  os.remove(dst)
p = subprocess.run([CLI, "pack", toml_path, "--output", dst], stdin=subprocess.DEVNULL, capture_output=True, text=True)
if p.returncode != 0 or not os.path.exists(dst):
  sys.exit(f"pack failed: {p.stdout[-500:]} {p.stderr[-500:]}")
shutil.rmtree(work, ignore_errors=True)
A, B = T.bundle_sections(src), T.bundle_sections(dst)
rows, ok = [], len(A) == len(B)
for a, b in zip(A, B):
  is_pd = a["type"] == "TFLiteModel" and a["items"].get("model_type", "").lower().endswith("prefill_decode")
  want = dict(a["items"], prefer_activation_type="fp32") if is_pd else a["items"]
  good = a["sha256"] == b["sha256"] and b["items"] == want
  ok = ok and good
  rows.append({"type": a["type"], "items_src": a["items"], "items_dst": b["items"], "sha256": a["sha256"],
               "same_bytes": a["sha256"] == b["sha256"], "ok": good})
h = hashlib.sha256()
with open(dst, "rb") as f:
  for chunk in iter(lambda: f.read(1 << 24), b""):
    h.update(chunk)
rec = {"src": os.path.abspath(src), "dst": os.path.abspath(dst), "cli": CLI, "sections": rows, "ok": ok,
       "dst_bytes": os.path.getsize(dst), "dst_sha256": h.hexdigest()}
json.dump(rec, open(report, "w"), indent=1)
for r in rows:
  print(f"{r['type']:<14} {r['sha256'][:12]} same_bytes={r['same_bytes']} items {r['items_dst']} {'ok' if r['ok'] else 'BAD'}")
print(f"DECLARE_FP32 {'OK' if ok else 'FAIL'} {dst} {rec['dst_bytes']:,} B sha256 {rec['dst_sha256']}")
sys.exit(0 if ok else 1)
