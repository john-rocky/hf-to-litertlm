#!/usr/bin/env python3
"""Round 5b: staging/upload/conversion/results/*.json — the published summaries of the checks and timings, taken field by
field from the work directory's records (no path of this machine; every value as recorded).

    python3 scripts/r5b_public_results.py
        -> gates_mac.json     the row graphs (each length, CPU and Metal float32) and the towers against the reference
           e2e_mac.json       this folder end to end (host + graphs) on both backends, per length and per run
           timing_mac.json    the graphs alone (one call), the towers, and one decision request with its head
           host_check.json    the host package's test against the provider's code
           example_mac.json   the commands of examples/README.md as they ran
Missing records (a window not run yet) leave that part out and are listed under "pending"."""
from __future__ import annotations

import json
from pathlib import Path

K = Path(__file__).resolve().parents[1]
OUT = K / "staging/upload/conversion/results"
L_ALL = (256, 512, 1024, 2048, 4096)
ROW = "clef-flash_rowprefill_embeds_L{L}_fp16fc.tflite"
TOWER = "clef-flash_vision_tower_{g}_fp16fc.tflite"
pending: list[str] = []


def load(p: str):
    f = K / p
    if not f.exists():
        pending.append(p)
        return None
    return json.loads(f.read_text())


def mem(m: dict | None):
    return None if not m else {"phys_footprint": m.get("phys_footprint"),
                               "lifetime_max_phys_footprint": m.get("lifetime_max_phys_footprint")}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    env = {"machine": "Apple M4 Max, 128 GB, macOS 27.0", "runtime": "ai-edge-litert 2.2.0 CompiledModel",
           "backends": {"gpu_f32": "Metal, GpuOptions(enforce_f32=True)", "cpu": "XNNPACK, CpuOptions(num_threads=8)"}}
    gates = {"what": "each row graph and tower against the provider's float32 CPU reference (text runs per length; the "
                     "picture records through the towers on the L1024 row graph)", "environment": env,
             "line": {"near_tie": 0.02, "max_abs_dp": 0.02, "mean_of_run_means": 0.002}, "row_graphs": {}, "towers": {}}
    for L in L_ALL:
        for acc in ("gpu_f32", "cpu"):
            chk = load(f"results/r2_check_v2_fp16fc_L{L}_{acc}.json")
            sc = load(f"results/r2_score_v2_fp16fc_L{L}_{acc}.json")
            if not (chk and sc):
                continue
            gates["row_graphs"][f"{ROW.format(L=L)} {acc}"] = {
                "runs": chk["runs"], "set": chk["set"], "all_finite": chk["all_finite"],
                "load_and_compile_seconds": chk["compile_seconds"], "memory_after_compile": [mem(m) for m in chk["memory_after_compile"]],
                "delegation": [d["replacing"] for d in chk["delegation"]], "fully_accelerated": chk["is_fully_accelerated"],
                "hidden_max_abs_vs_reference": chk["hidden_max_abs_vs_reference"],
                "summary": {k: sc["summary"][k] for k in ("runs", "questions", "argmax_non_near_tie", "argmax_near_tie",
                                                          "max_abs_dp", "max_abs_dp_run", "max_abs_dp_question",
                                                          "mean_of_run_means", "line_pass")},
                "red_arms": sc.get("red_arms", {}).get("moved") if isinstance(sc.get("red_arms"), dict) else sc.get("red_arms"),
                "pass": sc["gate_pass"]}
    for g in ("g256", "g448"):
        for acc in ("gpu_f32", "cpu"):
            t = load(f"results/r3_tower_rt_clef_tower_{g}_v2_fp16fc_{acc}.json")
            if t:
                gates["towers"][f"{TOWER.format(g=g)} {acc}"] = {
                    "runs": t["summary"]["runs"], "rows_vs_reference": t["summary"]["vs_author"],
                    "nonfinite": t["summary"]["nonfinite_total"], "compile_seconds": t["compile_seconds"],
                    "delegation": t["delegation"]["replacing"] if isinstance(t["delegation"], dict) else t["delegation"]}
    img = load("results/r3_score_L1024.json")
    if img:
        gates["pictures_on_L1024"] = {k: img["summary"][k] for k in ("fixed_grid", "g256", "g448")} | \
                                     {"red_arms": img["red_arms"]["moved"]}
    (OUT / "gates_mac.json").write_text(json.dumps(gates, indent=1) + "\n")

    e2e = {"what": "this folder end to end: examples' host package (ClefLiteRT.decide) on the folder's graphs, every "
                   "fixture run it can make (text, and pictures at the 256 / 448 px tiles), against the provider's float32 "
                   "CPU reference; GPU = every run, CPU = the first 20 runs of each length and the red arms",
           "environment": env, "backends": {}}
    for acc in ("gpu", "cpu"):
        s = load(f"results/r5b_score_{acc}.json")
        if not s:
            continue
        runs = []
        for L in L_ALL:
            p = K / f"results/r5b_e2e_{acc}_L{L}.jsonl"
            if p.exists():
                for ln in p.read_text().splitlines():
                    r = json.loads(ln)
                    runs.append({k: r[k] for k in ("key", "tokens", "length", "max_abs_dp", "mean_abs_dp",
                                                   "argmax_equal_non_near_tie", "body_structure_equal",
                                                   "decisions_equal_non_near_tie", "ms_decide")}
                                | {"red": r["red"]})
        e2e["backends"][acc] = {k: s[k] for k in ("complete", "line", "red_arms", "response", "nonfinite_real", "groups",
                                                  "pass")} | {"runs": runs}
    (OUT / "e2e_mac.json").write_text(json.dumps(e2e, indent=1) + "\n")

    timing = {"what": "median of 20 timed calls after 5 warm-up calls, inside a machine-wide measurement lock",
              "environment": env, "row_graph_call": {}, "tower_call": {}, "decision_request": {}}
    for L in L_ALL:
        for acc in ("gpu_f32", "cpu"):
            m = load(f"results/r2_ms_v2_fp16fc_L{L}_{acc}.json")
            if m:
                timing["row_graph_call"][f"L{L} {acc}"] = {"ms": m["ms"], "load_and_compile_seconds": m["load_and_compile_seconds"],
                                                            "loadavg_before": m["loadavg_before"], "loadavg_after": m["loadavg_after"],
                                                            "started_at": m["started_at"]}
    for g in ("g256", "g448"):
        for acc in ("gpu_f32", "cpu"):
            t = load(f"results/r3_tower_rt_clef_tower_{g}_v2_fp16fc_{acc}.json")
            if t:
                timing["tower_call"][f"{g} {acc}"] = {"ms": t["ms"]["call"]}
    for acc, Ls in (("gpu", L_ALL), ("cpu", (256, 1024))):
        for L in Ls:
            m = load(f"results/r5b_ms_{acc}_L{L}.json")
            if m:
                timing["decision_request"][f"L{L} {acc}"] = {
                    name: {k: v for k, v in r.items() if k not in ("lock_labels_seen",)} for name, r in m["requests"].items()} | \
                    {"loadavg_before": m["loadavg_before"], "loadavg_after": m["loadavg_after"], "started_at": m["started_at"]}
    (OUT / "timing_mac.json").write_text(json.dumps(timing, indent=1) + "\n")

    t = load("results/r5b_host_test.json")
    if t:
        host = {"what": "host/clef_litert against the provider's code: ids, spans, positions, row inputs, the head in "
                        "NumPy float32 and the response on 257 runs (the provider's reference hidden states as input)",
                "versions": t["versions"], "pass": t["pass"],
                "sections": {k: {kk: vv for kk, vv in v.items() if kk in ("pass", "summary")} for k, v in t["sections"].items()}}
        ref = host["sections"].get("refusals", {}).get("summary", {})
        for k in [k for k in ref if k.endswith("_four")]:    # the four refusals the host test names first
            ref["four_named"] = ref.pop(k)
        (OUT / "host_check.json").write_text(json.dumps(host, indent=1).replace(str(K) + "/", "") + "\n")
    ex = {"what": "the commands of examples/README.md, run as written inside this folder", "runs": {}}
    for mode in ("setup", "gpu", "cpu"):
        e = load(f"results/r5b_example_{mode}.json")
        if e:
            keep = ("#", "refused:", "missing in", "run_example.py:", "PASS", "FAIL", "Traceback")
            ex["runs"][mode] = [{k: c[k] for k in ("command", "rc", "want", "seconds", "peak_memory_footprint", "stdout_tail")}
                                | {"stderr_lines": [ln for ln in c["stderr_tail"] if ln.startswith(keep)]}
                                for c in e["commands"]] + [{"pass": e["pass"]}]
    (OUT / "example_mac.json").write_text(json.dumps(ex, indent=1).replace(str(K) + "/", "") + "\n")
    print(json.dumps({"written": sorted(p.name for p in OUT.glob("*.json")), "pending": pending}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
