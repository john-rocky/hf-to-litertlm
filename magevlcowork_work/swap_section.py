#!/usr/bin/env python3
"""swap_section.py SRC.litertlm NEW.tflite DST.litertlm --model-type TYPE --report R.json [--cli LITERT_LM_CLI] [--work DIR]

Bundle SRC with only its TYPE section (for example vision_encoder) replaced by NEW: `litert-lm unpack` SRC, copy NEW over
the unpacked file, `litert-lm pack`, then the check of gpu_graph_retrofit.py's own bundle path (retrofit_bundle):
same section count, same section items (model_type spelling normalized; the activation declarations stay as SRC has
them), every other section byte-identical (sha256), the replaced section = NEW's bytes.
bundle_sections() is loaded by path from tools/gpu_graph/gpu_graph_retrofit.py. Exit 1 when a check fails (the output
is kept for the record but the report says FAIL). The CLI: --cli, else $LITERT_LM_CLI, else litert-lm on PATH.
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

TOOL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "gpu_graph", "gpu_graph_retrofit.py")


def sha256(path):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for b in iter(lambda: f.read(1 << 24), b""):
      h.update(b)
  return h.hexdigest()


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("src")
  ap.add_argument("tflite")
  ap.add_argument("dst")
  ap.add_argument("--model-type", required=True, help="the section to replace, without TF_LITE_ (vision_encoder, ...)")
  ap.add_argument("--report", required=True)
  ap.add_argument("--cli", default=os.environ.get("LITERT_LM_CLI", "litert-lm"))
  ap.add_argument("--work")
  a = ap.parse_args()
  spec = importlib.util.spec_from_file_location("gpu_graph_retrofit_by_path", TOOL)
  T = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(T)
  work = a.work or (os.path.splitext(a.dst)[0] + "_swapwork")
  unpack = os.path.join(work, "unpack")
  if os.path.exists(unpack):
    shutil.rmtree(unpack)
  os.makedirs(work, exist_ok=True)
  p = subprocess.run([a.cli, "unpack", a.src, "--output-dir", unpack], stdin=subprocess.DEVNULL, capture_output=True,
                     text=True)
  toml_path = os.path.join(unpack, "model.toml")
  if p.returncode != 0 or not os.path.exists(toml_path):
    sys.exit(f"unpack failed rc={p.returncode}: {(p.stdout + p.stderr)[-2000:]}")
  toml = open(toml_path).read()
  m = re.search(rf'model_type = "(?:TF_LITE_)?(?i:{re.escape(a.model_type)})"\s*\n(?:[a-z_]+ = "[^"]*"\s*\n)*?data_path = "([^"]+)"',
                toml)
  if not m:
    sys.exit(f"no {a.model_type} TFLiteModel section in model.toml")
  pd = os.path.join(unpack, m.group(1))
  rec = {"src": os.path.abspath(a.src), "dst": os.path.abspath(a.dst), "tflite_in": os.path.abspath(a.tflite),
         "cli": a.cli, "tool": TOOL, "tool_sha256": sha256(TOOL), "toml": toml}
  rec["section_sha256_before"] = sha256(pd)
  rec["tflite_in_sha256"] = sha256(a.tflite)
  rec["tflite_in_bytes"] = os.path.getsize(a.tflite)
  shutil.copyfile(a.tflite, pd)
  if os.path.exists(a.dst):
    os.remove(a.dst)
  p = subprocess.run([a.cli, "pack", toml_path, "--output", a.dst], stdin=subprocess.DEVNULL, capture_output=True,
                     text=True)
  rec["pack_rc"] = p.returncode
  rec["pack_out"] = (p.stdout + p.stderr)[-2000:]
  if p.returncode != 0 or not os.path.exists(a.dst) or "Error" in rec["pack_out"]:
    json.dump(rec, open(a.report, "w"), indent=1)
    sys.exit(f"pack failed rc={p.returncode}: {rec['pack_out']}")
  shutil.rmtree(work, ignore_errors=True)
  A, B = T.bundle_sections(a.src), T.bundle_sections(a.dst)
  ok = len(A) == len(B)
  rows = []
  norm = lambda d: {k: (v.lower().replace("tf_lite_", "") if k == "model_type" else v) for k, v in d.items()}
  for x, y in zip(A, B):
    is_pd = x["type"] == "TFLiteModel" and x["items"].get("model_type", "").lower().endswith(a.model_type.lower())
    same_items = norm(x["items"]) == norm(y["items"])
    same_bytes = x["sha256"] == y["sha256"]
    new_ok = (not is_pd) or y["sha256"] == rec["tflite_in_sha256"]
    ok = ok and same_items and (is_pd or same_bytes) and new_ok
    rows.append({"type": x["type"], "items_src": x["items"], "items_dst": y["items"], "bytes_src": x["bytes"],
                 "bytes_dst": y["bytes"], "sha256_src": x["sha256"], "sha256_dst": y["sha256"], "replaced": is_pd,
                 "same_items": same_items, "same_bytes": same_bytes, "replaced_is_new_tflite": new_ok if is_pd else None})
  rec["sections"] = rows
  rec["ok"] = ok
  rec["dst_bytes"] = os.path.getsize(a.dst)
  rec["dst_sha256"] = sha256(a.dst)
  json.dump(rec, open(a.report, "w"), indent=1)
  for r in rows:
    print(f"{r['type']:<14} {str(r['items_dst']):<70.70} {r['bytes_src']:>13,} -> {r['bytes_dst']:>13,} "
          f"{r['sha256_src'][:12]} -> {r['sha256_dst'][:12]} "
          f"{'REPLACED' if r['replaced'] else ('same' if r['same_bytes'] else 'CHANGED')}")
  print(f"SWAP_CHECK {'OK' if ok else 'FAIL'} {a.dst} {rec['dst_bytes']:,} B sha256 {rec['dst_sha256']}")
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
