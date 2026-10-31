#!/usr/bin/env python3
"""Spark-X2.6: put a rewritten prefill_decode .tflite back into an unpacked bundle, pack it with the litert-lm CLI,
and check the result the way gpu_graph_retrofit.py's own bundle path does (retrofit_bundle): same section count, same
section items (model_type spelling normalized), every section other than prefill_decode byte-identical (sha256).
The tool's bundle_sections() is used, loaded by path (the public tool is read, never written).

  python -B spark_work/x26_pack_check.py <src.litertlm> <unpack dir> <new prefill_decode.tflite> \
      <dst.litertlm> <report.json> [--cli <litert-lm 0.19.0 CLI>]   (default: $LITERT_LM_CLI, else litert-lm on PATH)
"""
import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
TOOL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "gpu_graph", "gpu_graph_retrofit.py")


def sha256(path):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for b in iter(lambda: f.read(1 << 24), b""):
      h.update(b)
  return h.hexdigest()


ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("unpack")
ap.add_argument("tflite")
ap.add_argument("dst")
ap.add_argument("report")
ap.add_argument("--cli", default=os.environ.get("LITERT_LM_CLI", "litert-lm"))
a = ap.parse_args()

spec = importlib.util.spec_from_file_location("gpu_graph_retrofit_tool", TOOL)
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)

toml_path = os.path.join(a.unpack, "model.toml")
toml = open(toml_path).read()
m = re.search(r'model_type = "(?:TF_LITE_)?(?i:prefill_decode)"\s*\nsection_type = "TFLiteModel"\s*\ndata_path = "([^"]+)"', toml)
if not m:
  sys.exit("no prefill_decode TFLiteModel section in model.toml")
pd = os.path.join(a.unpack, m.group(1))
rec = {"src": os.path.abspath(a.src), "dst": os.path.abspath(a.dst), "tflite_in": os.path.abspath(a.tflite),
       "cli": a.cli, "tool": TOOL, "tool_sha256": sha256(TOOL), "toml": toml}
rec["prefill_decode_sha256_before"] = sha256(pd)
rec["tflite_in_sha256"] = sha256(a.tflite)
shutil.copyfile(a.tflite, pd)
if os.path.exists(a.dst):
  os.remove(a.dst)
p = subprocess.run([a.cli, "pack", toml_path, "--output", a.dst], stdin=subprocess.DEVNULL, capture_output=True, text=True)
rec["pack_rc"] = p.returncode
rec["pack_out"] = (p.stdout + p.stderr)[-2000:]
if p.returncode != 0 or not os.path.exists(a.dst):
  json.dump(rec, open(a.report, "w"), indent=1)
  sys.exit(f"pack failed rc={p.returncode}: {rec['pack_out']}")
A, B = T.bundle_sections(a.src), T.bundle_sections(a.dst)
ok = len(A) == len(B)
rows = []
norm = lambda d: {k: (v.lower().replace("tf_lite_", "") if k == "model_type" else v) for k, v in d.items()}
for x, y in zip(A, B):
  is_pd = x["type"] == "TFLiteModel" and x["items"].get("model_type", "").lower().endswith("prefill_decode")
  same_items = norm(x["items"]) == norm(y["items"])
  same_bytes = x["sha256"] == y["sha256"]
  ok = ok and same_items and (is_pd or same_bytes)
  rows.append({"type": x["type"], "items_src": x["items"], "items_dst": y["items"], "bytes_src": x["bytes"],
               "bytes_dst": y["bytes"], "sha256_src": x["sha256"], "sha256_dst": y["sha256"], "replaced": is_pd,
               "same_items": same_items, "same_bytes": same_bytes})
rec["sections"] = rows
rec["ok"] = ok
rec["dst_bytes"] = os.path.getsize(a.dst)
rec["dst_sha256"] = sha256(a.dst)
json.dump(rec, open(a.report, "w"), indent=1)
for r in rows:
  print(f"{r['type']:<14} {str(r['items_dst']):<70.70} {r['bytes_src']:>13,} -> {r['bytes_dst']:>13,} "
        f"{r['sha256_src'][:12]} -> {r['sha256_dst'][:12]} {'REPLACED' if r['replaced'] else ('same' if r['same_bytes'] else 'CHANGED')}")
print(f"PACK_CHECK {'OK' if ok else 'FAIL'} {a.dst} {rec['dst_bytes']:,} B sha256 {rec['dst_sha256']}")
sys.exit(0 if ok else 1)
