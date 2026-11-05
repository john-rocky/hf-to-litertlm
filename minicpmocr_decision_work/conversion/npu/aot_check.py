#!/usr/bin/env python3
"""Read one SM8850 compile (aot_compile_sm8850.py with qualcomm_log_level=verbose) against what the Galaxy S26's NPU
accepts:

  - the compile: every op of the graph selected, in one partition, a context binary and an output file;
  - (i)  one graph: the plugin's alloca_est_total <= 3,670,016 KiB (its init graph option memory_alloc_limit, 3,584 MiB);
  - (ii) one session: context binary + sum over graphs (alloca_est_total - alloca_pickle_const_extent) + 64 MiB
         <= 3,750,000,000 B (the address space of one compute-DSP session, which a context cannot span).

    python3 conversion/npu/aot_check.py out/npu/s_L256        # standard library only; exit 0 inside, 1 not

Prints one line and writes OUTDIR/aot_check.json with the values and the output's bytes and SHA-256.
"""
import glob
import hashlib
import json
import os
import re
import sys

LINE_GRAPH_KIB = 3_670_016
LINE_VA_B = 3_750_000_000
VA_PAD_B = 64 * 1024 * 1024


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
        print(f"not inside: no plugin log ({e})")
        sys.exit(1)
    parts = [tuple(int(v) for v in m) for m in re.findall(
        r"Partitioned subgraph<(\d+)>, selected (\d+) ops, from a total of (\d+) ops\. resulted in (\d+) partitions", log)]
    est = [int(x) for x in re.findall(r"STAT: alloca_est_total=(\d+)", log)]
    const = [int(x) for x in re.findall(r"STAT: alloca_pickle_const_extent=(\d+)", log)]
    ctx = [int(x) for x in re.findall(r"Serialized a context bin of size \(bytes\): (\d+)", log)]
    rc = re.findall(r"^RC (-?\d+) WALL_S", log, re.M)
    outs = sorted(glob.glob(os.path.join(outdir, "*_apply_plugin.tflite")))
    reasons = []
    if not rc or int(rc[-1]) != 0:
        reasons.append(f"plugin rc {rc[-1] if rc else 'missing'}")
    if not parts:
        reasons.append("no partition line")
    for s, sel, tot, n in parts:
        if sel != tot or n != 1:
            reasons.append(f"subgraph {s}: {sel} / {tot} ops in {n} partitions")
    if not est or len(est) != len(const):
        reasons.append("no (or unpaired) alloca STAT lines: compile with qualcomm_log_level=verbose")
    if not ctx:
        reasons.append("no context binary")
    if not outs or os.path.getsize(outs[0]) == 0:
        reasons.append("no output file")
    doc = {"partitions": [{"subgraph": s, "selected": a, "total": b, "partitions": n} for s, a, b, n in parts],
           "alloca_est_total_B": est, "alloca_pickle_const_extent_B": const, "context_bin_B": ctx}
    if est:
        doc["line_i"] = {"max_alloca_est_total_KiB": max(est) // 1024, "line_KiB": LINE_GRAPH_KIB,
                         "inside": max(est) <= LINE_GRAPH_KIB * 1024}
        if not doc["line_i"]["inside"]:
            reasons.append(f"line (i): {max(est) // 1024} KiB > {LINE_GRAPH_KIB} KiB")
    if est and ctx and len(est) == len(const):
        va = sum(ctx) + sum(e - c for e, c in zip(est, const)) + VA_PAD_B
        doc["line_ii"] = {"session_va_B": va, "line_B": LINE_VA_B, "inside": va <= LINE_VA_B}
        if not doc["line_ii"]["inside"]:
            reasons.append(f"line (ii): {va} B > {LINE_VA_B} B")
    if outs:
        doc["output"] = os.path.basename(outs[0])
        doc["output_bytes"] = os.path.getsize(outs[0])
        doc["output_sha256"] = sha256(outs[0])
    doc["verdict"] = "inside" if not reasons else "not inside"
    doc["reasons"] = reasons
    with open(os.path.join(outdir, "aot_check.json"), "w") as f:
        json.dump(doc, f, indent=1)
    li = f"(i) {doc['line_i']['max_alloca_est_total_KiB']:,} KiB of {LINE_GRAPH_KIB:,}" if "line_i" in doc else "(i) -"
    lii = f"(ii) {doc['line_ii']['session_va_B']:,} B of {LINE_VA_B:,}" if "line_ii" in doc else "(ii) -"
    pp = "; ".join(f"{a} / {b} ops in {n} partition(s)" for _, a, b, n in parts) or "no partition line"
    print(f"{doc['verdict']}: {pp}, {li}, {lii}, output {doc.get('output_bytes', 0):,} B sha256 "
          f"{doc.get('output_sha256', '-')}" + (f"; {' | '.join(reasons)}" if reasons else ""))
    sys.exit(0 if not reasons else 1)


if __name__ == "__main__":
    main()
