#!/usr/bin/env python3
"""Read one SM8850 compile (aot_compile_sm8850.py with qualcomm_log_level=verbose):

  - the plugin's exit code is 0 (the script's own exit code is not enough: a kill at the memory limit shows as `RC -9` in the log, with no
    output file);
  - every op of the graph is selected, in one partition;
  - a context binary was serialized and the output file is not empty;
  - the SDK version the plugin applied, the sizes the log gives (constSize, DDRTensorSize, the context binary).

QAIRT 2.50's verbose log prints no per-graph memory estimate (no `STAT: alloca_*` lines), so nothing here says whether the context fits the
NPU's session on a phone: the first load and the first run on the phone decide that.

    python3 conversion/npu/aot_check.py work/npu/L64        # standard library only; exit 0 when all of it holds, 1 otherwise

Prints one line and writes OUTDIR/aot_check.json with the values and the output's bytes and SHA-256.
"""
import glob
import hashlib
import json
import os
import re
import sys


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    outdir = sys.argv[1]
    try:
        log = open(os.path.join(outdir, "apply_plugin.log"), errors="replace").read()
    except OSError as e:
        print(f"not ok: no plugin log ({e})")
        sys.exit(1)
    parts = [tuple(int(v) for v in m) for m in re.findall(
        r"Partitioned subgraph<(\d+)>, selected (\d+) ops, from a total of (\d+) ops\. resulted in (\d+) partitions", log)]
    ctx = [int(x) for x in re.findall(r"Serialized a context bin of size \(bytes\): (\d+)", log)]
    const = re.findall(r"constSize=(\d+)", log)
    ddr = re.findall(r"DDRTensorSize=(\d+)", log)
    sdk = re.findall(r"Applying plugin with SDK version: (\S+)", log)
    rc = re.findall(r"^RC (-?\d+) WALL_S ([\d.]+)", log, re.M)
    outs = sorted(glob.glob(os.path.join(outdir, "*_apply_plugin.tflite")))
    reasons = []
    if not rc or int(rc[-1][0]) != 0:
        reasons.append(f"plugin rc {rc[-1][0] if rc else 'missing'}")
    if not parts:
        reasons.append("no partition line")
    for s, sel, tot, n in parts:
        if sel != tot or n != 1:
            reasons.append(f"subgraph {s}: {sel} / {tot} ops in {n} partitions")
    if not ctx:
        reasons.append("no context binary")
    if not outs or os.path.getsize(outs[0]) == 0:
        reasons.append("no output file")
    doc = {"plugin_rc": int(rc[-1][0]) if rc else None, "plugin_wall_s": float(rc[-1][1]) if rc else None,
           "partitions": [{"subgraph": s, "selected": a, "total": b, "partitions": n} for s, a, b, n in parts],
           "sdk": sdk[0] if sdk else None, "constSize_B": int(const[-1]) if const else None,
           "DDRTensorSize_B": int(ddr[-1]) if ddr else None, "context_bin_B": ctx[-1] if ctx else None}
    if outs:
        doc["output"] = os.path.basename(outs[0])
        doc["output_bytes"] = os.path.getsize(outs[0])
        doc["output_sha256"] = sha256(outs[0])
    doc["verdict"] = "ok" if not reasons else "not ok"
    doc["reasons"] = reasons
    with open(os.path.join(outdir, "aot_check.json"), "w") as f:
        json.dump(doc, f, indent=1)
    pp = "; ".join(f"{a} / {b} ops in {n} partition(s)" for _, a, b, n in parts) or "no partition line"
    print(f"{doc['verdict']}: {pp}, SDK {doc['sdk']}, context {doc['context_bin_B'] or 0:,} B, output {doc.get('output_bytes', 0):,} B sha256 "
          f"{doc.get('output_sha256', '-')}" + (f"; {' | '.join(reasons)}" if reasons else ""))
    sys.exit(0 if not reasons else 1)


if __name__ == "__main__":
    main()
