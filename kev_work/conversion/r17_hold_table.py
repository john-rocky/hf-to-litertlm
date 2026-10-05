"""One table of every NPU-runner run in a folder (read only): form, backend, compile s, delegate numbers, timing (median /
min / max of the timed calls), the clock-cap record, parity against the reference and against the same file on the
Mac CPU (the backend's own error), VmHWM.

    python3 r17_hold_table.py [--md] [--dir device/r17/B --out results/r17_device_table_B.json]
        -> results/r17_device_table.json (+ markdown on stdout with --md)"""
import argparse
import json
from pathlib import Path

K = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", action="store_true")
    ap.add_argument("--dir", default="device/r17")
    ap.add_argument("--out", default="results/r17_device_table.json")
    a = ap.parse_args()
    R = K / a.dir
    rows = []
    for rep in sorted(R.glob("kev_s26_r17_*.json")):
        if rep.name.endswith(".caps.json"):
            continue
        tag = rep.stem
        d = json.loads(rep.read_text())
        par = R / f"device_parity_{tag}.json"
        caps = R / f"{tag}.caps.json"
        p = json.loads(par.read_text()) if par.exists() else {}
        c = json.loads(caps.read_text()) if caps.exists() else {}
        s = p.get("summary", {})
        o = s.get("overall", {})
        sf = (p.get("same_file") or {}).get("mac_cpu8_r17") or {}
        lat = d.get("latency") or {}
        lp = c.get("latency_phase") or {}
        rows.append({
            "tag": tag, "graph": d.get("graph"), "accel": d.get("accel"), "precision": d.get("precision"),
            "status": d.get("status"), "compile_s": round(d["compile_ms"] / 1000, 1) if d.get("compile_ms") else None,
            "delegate": d.get("delegate_summary"), "rows": d.get("summary", {}).get("count"),
            "nonfinite_rows": d.get("summary", {}).get("nonfinite_rows"),
            "ms_median": lat.get("median_write_run_read_ms"), "ms_min": lat.get("min_ms"), "ms_max": lat.get("max_ms"),
            "timing_cap": c.get("timing_cap"), "lat_kgsl_min": lp.get("kgsl_max_min"), "lat_cpu_caps": lp.get("cpu_caps"),
            "lat_thermal_max": lp.get("thermal_max"), "ready": c.get("ready_gate"),
            "vmhwm_kb": (c.get("whole_leg") or {}).get("vmhwm_kb_max"),
            "bar_near_tie_apart": s.get("bar_near_tie_apart"), "non_near_tie": s.get("non_near_tie_argmax"),
            "near_tie": s.get("near_tie_argmax"), "max_abs_dp": o.get("max_abs_dp"),
            "mean_abs_dp": o.get("mean_abs_dp_all_options"), "red_arm": (s.get("red_arm") or {}).get("max_abs_dp"),
            "same_file_max": sf.get("max_abs_dp"), "same_file_mean": sf.get("mean_abs_dp_all_options"),
            "same_file_hsel_max": sf.get("h_sel_max_abs"), "same_file_questions": sf.get("questions")})
    (K / a.out).write_text(json.dumps(rows, indent=1) + "\n")
    if a.md:
        f = lambda x, n=4: "—" if x is None else (f"{x:.{n}g}" if isinstance(x, float) else str(x))
        print("| leg | backend | compile s | delegate | ms median (min–max) | cap | parity (322) | vs same file on Mac CPU |")
        print("|---|---|---|---|---|---|---|---|")
        for r in rows:
            dl = r["delegate"] or {}
            dele = "; ".join([f"{x[0]}/{x[1]} ops -> {x[2]} partitions" for x in dl.get("partitioned", [])] +
                             [f"{x[2]} {x[0]}/{x[1]}, {x[3]} partitions" for x in dl.get("replacing", [])])
            if dl.get("jit_from_cache"):
                dele += " (from the JIT cache)"
            ms = f"{f(r['ms_median'], 5)} ({f(r['ms_min'], 5)}–{f(r['ms_max'], 5)})" if r["ms_median"] else "—"
            cap = ("cap" if r["timing_cap"] else "clean" if r["timing_cap"] is False else "—") + \
                  (f" (kgsl ≥ {r['lat_kgsl_min']}, cpu caps {len(r['lat_cpu_caps'] or [])})" if r["lat_kgsl_min"] else "")
            par = (f"{'PASS' if r['bar_near_tie_apart'] else 'FAIL'}: {r['non_near_tie']} + {r['near_tie']}, max "
                   f"{f(r['max_abs_dp'])}, mean {f(r['mean_abs_dp'])}, red {f(r['red_arm'])}") if r["max_abs_dp"] is not None else "—"
            same = f"max {f(r['same_file_max'])}, mean {f(r['same_file_mean'])}, h_sel {f(r['same_file_hsel_max'])}" \
                if r["same_file_max"] is not None else "—"
            print(f"| {r['tag'].replace('kev_s26_r17_', '')} | {r['accel']} {r['precision']} | {r['compile_s']} | {dele} | "
                  f"{ms} | {cap} | {par} | {same} |")


if __name__ == "__main__":
    main()
