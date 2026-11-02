"""Every leg of a phone run's pulled folder as one table row: the status, the backend, the graph, the rows run, the
engine build (compile_ms), the call times (generic: the first call and the warm median of run_ms; generic_timing: each
set's median / min / max per call), the memory (the app's VmHWM / VmRSS at the end, the sampler's VmHWM max and
MemAvailable min), the delegation line of the leg's logcat ("Replacing N out of M node(s) ..."), the leg check's
verdict and its distance from the Mac's Metal fp32 run of the same rows. leg_table_omni.py imports its row.

  python3 conversion/leg_table.py <round dir> [--prefix g32_ ...] [--json out.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.normpath(os.path.join(HERE, ".."))


def read(p, default=""):
    try:
        return open(p, errors="replace").read()
    except OSError:
        return default


def leg_row(R, tag, results_dir):
    rep = {}
    try:
        rep = json.load(open(os.path.join(R, tag + ".json")))
    except Exception:
        pass
    st = read(os.path.join(R, tag + ".leg_status")).strip() or rep.get("status") or "?"
    mem_txt = read(os.path.join(R, tag + ".memory.txt"))
    hwm = re.search(r"VmHWM_max_kB (\d+)", mem_txt)
    mav = re.search(r"MemAvailable_min_kB (\d+)", mem_txt)
    logc = read(os.path.join(R, tag + ".logcat.txt"))
    deleg = re.findall(r"Replacing (\d+) out of (\d+) node\(s\) with delegate \(([^)]*)\) node, yielding (\d+) partitions", logc)
    end = rep.get("memory_at_end") or {}
    row = {"tag": tag, "status": st, "accel": rep.get("accel"), "precision": rep.get("precision"),
           "graph": rep.get("graph"), "rows_file": rep.get("rows_file"), "signature": rep.get("signature"),
           "compile_ms": rep.get("compile_ms"), "VmHWM_end_kb": end.get("VmHWM_kb"), "VmRSS_end_kb": end.get("VmRSS_kb"),
           "sampler_VmHWM_max_kB": int(hwm.group(1)) if hwm else None,
           "MemAvailable_min_kB": int(mav.group(1)) if mav else None,
           "delegation": [{"replaced": int(a), "of": int(b), "delegate": c, "partitions": int(d)} for a, b, c, d in deleg]}
    s = rep.get("summary") or {}
    if rep.get("mode") == "generic":
        # a call = write every input + run + read every output: on the phone's GPU run() only enqueues (0.2-0.9 ms)
        # and the read waits for the GPU, so run_ms alone is not a call's time
        per = [r.get("write_run_read_ms") for r in (rep.get("rows") or []) if r.get("write_run_read_ms") is not None]
        row.update({"rows_ran": s.get("count"), "first_call_ms": s.get("first_call_write_run_read_ms"),
                    "warm_median_ms": s.get("warm_median_write_run_read_ms"), "warm_rows": s.get("warm_rows"),
                    "row_ms_min": min(per) if per else None, "row_ms_max": max(per) if per else None,
                    "first_call_run_only_ms": s.get("first_call_run_ms"), "nonfinite_rows": s.get("nonfinite_rows")})
    if rep.get("mode") == "generic_timing":
        row["timing"] = {k: {"kind": v.get("kind"), "rows": v.get("rows"),
                             "per_call_ms": v.get("per_call_ms_write_run_read"), "run_only_ms": v.get("per_call_ms_run_only"),
                             "finite": v.get("finite_outputs")} for k, v in (rep.get("timing") or {}).items()}
    row["verdict"] = read(os.path.join(R, tag + ".verdict")).split(" ")[0] or None
    lc = sorted(glob.glob(os.path.join(results_dir, f"*_leg_{tag}_*.json")))
    if lc:
        v = (json.load(open(lc[-1])).get("vs_mac_metal_fp32") or {})
        row["vs_mac"] = {"rows": v.get("rows"), "same_input_rows": v.get("same_input_rows"),
                         "max_abs_same_input": v.get("max_abs_same_input"), "cos_min_same_input": v.get("cos_min_same_input")}
    return row


def fmt(x, nd=1):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("round_dir")
    ap.add_argument("--prefix", action="append", default=[])
    ap.add_argument("--results-dir", default=os.path.join(WORK, "results/device"))
    ap.add_argument("--json")
    a = ap.parse_args()
    R = os.path.abspath(a.round_dir)
    tags = sorted({os.path.basename(p)[:-len(".leg_status")] for p in glob.glob(os.path.join(R, "*.leg_status"))}
                  | {os.path.basename(p)[:-len(".attempted")] for p in glob.glob(os.path.join(R, "*.attempted"))})
    if a.prefix:
        tags = [t for t in tags if any(t.startswith(p) for p in a.prefix)]
    att = lambda t: os.path.getmtime(os.path.join(R, t + ".attempted")) if os.path.exists(os.path.join(R, t + ".attempted")) else 0
    tags.sort(key=att)  # the order the legs ran
    rows = [leg_row(R, t, a.results_dir) for t in tags]
    print("| leg | status | backend | rows | compile s | 1st call ms | rows' ms min–max | VmHWM MB | MemAvail min MB |"
          " delegation | verdict | vs Mac max|d| / cos min |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        dl = ";".join(f"{d['replaced']}/{d['of']} {d['delegate']} {d['partitions']}p" for d in r["delegation"]) or "-"
        vm = r.get("vs_mac") or {}
        hw = r["VmHWM_end_kb"] or r["sampler_VmHWM_max_kB"]
        print(f"| {r['tag']} | {r['status']} | {r['accel']} {r['precision']} | {fmt(r.get('rows_ran'))} |"
              f" {fmt(r['compile_ms'] / 1000 if r['compile_ms'] else None)} | {fmt(r.get('first_call_ms'))} |"
              f" {fmt(r.get('row_ms_min'))}–{fmt(r.get('row_ms_max'))} | {fmt(hw / 1024 if hw else None, 0)} |"
              f" {fmt(r['MemAvailable_min_kB'] / 1024 if r['MemAvailable_min_kB'] else None, 0)} | {dl} | {r['verdict'] or '-'} |"
              f" {fmt(vm.get('max_abs_same_input'), 6) if vm else '-'} / {fmt(vm.get('cos_min_same_input'), 7) if vm else '-'} |")
        for k, v in (r.get("timing") or {}).items():
            pc = v.get("per_call_ms") or {}
            print(f"|  ↳ {k} | ({v['kind']}, {v['rows']} rows) | | | | median {fmt(pc.get('median'))} |"
                  f" min {fmt(pc.get('min'))} max {fmt(pc.get('max'))} n {fmt(pc.get('n'))} | | | | finite {v.get('finite')} | |")
    if a.json:
        json.dump({"round_dir": os.path.relpath(R, WORK), "legs": rows}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
