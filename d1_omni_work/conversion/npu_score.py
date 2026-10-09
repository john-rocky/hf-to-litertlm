"""Round 10: host scorer of the S26 NPU chain's legs (scripts/s26_npu_chain.sh: the Kev-lane C-API runner, LiteRT 2.2.0,
HTP JIT through the Qualcomm compiler plugin, or the same runner on the GPU), and of its dry run's stand-in.

    cd d1_omni_work; PY=~/venvs/lt094dev/bin/python   (numpy + torch: the host read-out is host/d1_host.py's float32 ops)
    $PY scripts/npu_score.py leg --tag <leg> [--round r10] [--round-dir device/r10] [--results results]
        -> <results>/s26_npu_parity_<leg>_<round>.json (+ s26_npu_timing_<leg>_<round>.json when the latency phase ran)
    $PY scripts/npu_score.py table [--round r10] [--round-dir ...] [--results ...]   # markdown tables from the results
        -> <results>/s26_npu_summary_<round>.json
Round 12 (additions; round 10's legs score as before): any graph of the generalized chain. A leg whose fixture manifest
says kind audio / vision_tower / projector is scored by score_generic_leg (output 0 vs the Mac CPU run of the same file
on the same fixture, vs the oracle / provider references, the prefixes -> <round dir>/<leg>.prefix.npz); its PASS / FAIL
is the end to end on the Mac:
    $Q/quiet_wait.py -- $PY scripts/npu_score.py e2e --tag <leg> --round r12     -> s26_npu_parity_<leg>_e2e_r12.json
The timing json adds the calls' burst / capped split (round 9's rule, by the nearest 2 s sample); the GPU precision
pairs of a round after r10 come from the plan (extras pair=<FP32 leg> on the FP16_WITH_FP32_ACCUM leg).

leg: inputs = what the chain leaves in the round dir for <leg>: leg.json (the plan line, status, rc, the compile watch,
the phone's /data and cache sizes), stdout.txt (the runner's stdout + stderr: its input order, compile stamps, one line
per fixture of the parity pass, the latency phase, and LiteRT's own INFO / VERBOSE lines), dump/<id3>.f32 (the runner's
--dump full: scores float32 [L] per fixture), logcat_all.txt (logcat -b all since the leg's device mark), samples.txt
(the 2 s sampler), state_before / state_after.txt; the fixture manifest device/<round>/<fx>.manifest.json; the Mac CPU
same-file scores device/<round>/mac/<graph>/model0/<id3>.f32 (scripts/npu_fixtures.py check).
Parity (the gate rows of the fixture dir; bar = FACTS §7, as rounds 3-9): the K scores at P + markers -> the host
read-out (scripts/s26_score.py readout = host/d1_host.py readout: / T for a text question, softmax, a noul reversed)
-> vs the oracle (ref/records_ref.json version 2) with scripts/s26_score.py compare (argmax outside near-tie, near-tie
apart, max / p95 / mean |dp|, max |dlogit|, crossings of 0.5 / 0.9, non-finite rows), the control row (red_arm_000 vs the
oracle of tv4_000, must differ by > 0.02), rows whose K marker scores are all equal (reported apart, not part of the bar:
one near-tie row can tie in fp16 storage; `collapse` = every row uniform, the pre-rewrite file's failure under fp16
storage), and the same file's Mac CPU scores (max |dscore| at the markers and at every real position,
bit-equal rows, max |dp|, argmax agreement). Timing rows of the fixture dir that are not gate rows are scored apart.
Delegation: LiteRT's `Partitioned subgraph<0>, selected N ops, from a total of M ops. resulted in P partitions.`, the
`Replacing N out of M node(s) with delegate (X) node, yielding P partitions for subgraph S (name)` lines, the compiler
plugin line, the JIT reserialize / cache lines, the QNN validator's rejected ops; from the runner's stdout and from the
logcat lines of the runner's pid (memory gates-can-lie #11: an NPU row needs the Replacing (DispatchDelegate) line).
The runner's own `input[i] sig_name=... elems=... rank=...` lines are checked against the manifest's subgraph order.
Timing (when the latency phase ran): the runner's calls (write the six inputs + LiteRtRunCompiledModel + read the
scores = 1 call; run-only apart), warm-up calls, median / min / max, the runner's own summary line next to it; the
parity pass's per-row calls (set (b) = its three rows' calls summed, one request); the compile (the runner's
compile_ms and its wall stamps); the phone's state at the latency phase's start (nearest sample), the samples' MemAvailable
minimum and the runner's VmHWM maximum, the state before / after. Output files are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
K = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K / "scripts"))
sys.path.insert(0, str(K / "host"))
import s26_score as SS  # noqa: E402  (rounds 5 / 9: oracle rows, read-out, compare, samples; read only)

PAIRS = (("G1_gpu_fp32_f16s_L128", "G2_gpu_fp16acc32_f16s_L128"), ("G3_gpu_fp32_f16s_L256", "G4_gpu_fp16acc32_f16s_L256"))
GRAPH_FILES = {"L128_f16safe_fp16": "out/d1omni_decide_L128_f16safe_fp16.tflite",
               "L256_f16safe_fp16": "out/d1omni_decide_L256_f16safe_fp16.tflite",
               "L128_fp16": "out/d1omni_decide_L128_fp16.tflite"}


def graph_file(tag: str) -> str:
    """Round 12: the chain's graph tag -> out/<file> (the chain's graph_file rule): L<n>_* = a decision graph
    (out/d1omni_decide_<tag>.tflite), *.tflite = the file itself, else out/d1omni_<tag>.tflite."""
    if tag in GRAPH_FILES:
        return GRAPH_FILES[tag]
    if re.match(r"^L\d+_", tag):
        return f"out/d1omni_decide_{tag}.tflite"
    return f"out/{tag}" if tag.endswith(".tflite") else f"out/d1omni_{tag}.tflite"


def plan_pairs(round_dir: Path, rnd: str):
    """Round 12: the GPU precision pairs declared in the plan (extras pair=<FP32 leg> on the FP16_WITH_FP32_ACCUM leg);
    round 10's pairs are the fixed PAIRS above."""
    if rnd == "r10":
        return PAIRS
    plan = None
    for p in (round_dir / f"plan_{rnd}.tsv", K / "device" / rnd / f"plan_{rnd}.tsv"):
        if p.exists():
            plan = p
            break
    if plan is None:
        return ()
    prec, decl = {}, []
    for line in plan.read_text().splitlines():
        line = line.split("#", 1)[0].split()
        if len(line) < 12:
            continue
        prec[line[0]] = line[2]
        for kv in line[12:]:
            if kv.startswith("pair="):
                decl.append((kv.split("=", 1)[1], line[0]))
    out = []
    for x, y in decl:     # oriented (FP32 leg, FP16_WITH_FP32_ACCUM leg) by the plan's precision column
        out.append((x, y) if prec.get(x) == "fp32" else (y, x))
    return tuple(out)
RE = {
    "header": re.compile(r"^pid=(\d+) accel=(\S+) gpu_precision=(\S+) burst=(\d+) models=(\d+) fixtures=(\d+) warmup=(\d+) "
                         r"rounds=(\d+) ab_fixture=(\S+) .*wall_ms=(\d+)"),
    "input": re.compile(r"^model\[0\] input\[(\d+)\] sig_name=(\S+) logical=(\S+) elems=(\d+) rank=(\d+)"),
    "compile_start": re.compile(r"^compile_start model\[0\] wall_ms=(\d+)"),
    "compile_end": re.compile(r"^compile_end model\[0\] wall_ms=(\d+)"),
    "model": re.compile(r"^model\[0\]=(\S+) inputs=(\d+) outputs=(\d+) out0_elems=(\d+) compile_ms=([\d.]+) "
                        r"fully_accelerated=(true|false)"),
    "parity": re.compile(r"^parity model\[0\] fixture (\d+) ms=([\d.]+) run_ms=([\d.]+) nonfinite=(\d+)"),
    "parity_summary": re.compile(r"^parity_summary model\[0\] \S+ fixtures=(\d+) nonfinite_values=(\d+) first_run_ms=([\d.]+) "
                                 r"mean_ms=([\d.]+)"),
    "latency_start": re.compile(r"^latency_start wall_ms=(\d+)"),
    "warmup": re.compile(r"^warmup (\d+) model\[0\] ([\d.]+) ms run_ms=([\d.]+)"),
    "round": re.compile(r"^round (\d+) model\[0\] write\+run\+readback_ms=([\d.]+) run_ms=([\d.]+)"),
    "summary": re.compile(r"^summary model\[0\] \S+ n=(\d+) median=([\d.]+) min=([\d.]+) max=([\d.]+) nonfinite_rounds=(\d+) "
                          r"compile_ms=([\d.]+) fully_accelerated=(\w+)"),
    "done": re.compile(r"^done wall_ms=(\d+)"),
}
PARTITIONED = re.compile(r"Partitioned subgraph<(\d+)>, selected (\d+) ops, from a total of (\d+) ops\. resulted in (\d+) partitions")
REPLACING = re.compile(r"Replacing (\d+) out of (\d+) node\(s\) with delegate \(([^)]*)\) node, yielding (\d+) partitions"
                       r"(?: for subgraph (\d+) \(([^)]*)\))?")
KEY_LINE = re.compile(r"(?i)partitioned subgraph|replacing \d+ out of|compiler plugins? were applied|reserializ|cached model|"
                      r"caching enabled|context binary|qnn context|failed to validate op|validateopconfig failed|"
                      r"unsupported input/output|op validation failed|not supported|unsupported|fatal|abort|"
                      r"litertcreatecompiledmodel|status=|scudo|out of memory|signal \d+|npu accelerator|htpperformancemode|"
                      r"created tensorflow lite xnnpack|dispatchdelegate|litert_cl|lowmemorykiller|lmkd")
REJECT = re.compile(r"Failed to validate op (\S+) with error (\S+)|Unsupported input/output datatypes requested for the HTP Op "
                    r"'([^']+)' in the node '([^']+)'")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 24):
            h.update(block)
    return h.hexdigest()


def rel(p) -> str:
    p = Path(p).resolve()
    return str(p.relative_to(K)) if p.is_relative_to(K) else str(p)


def write_new(path: Path, doc) -> Path:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {rel(path)}")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    tmp.replace(path)
    return path


def stats(xs):
    return {"median": float(statistics.median(xs)), "min": float(min(xs)), "max": float(max(xs)), "n": len(xs)} if xs else None


def parse_stdout(path: Path) -> dict:
    out = {"inputs": [], "parity": {}, "warmup": [], "rounds": [], "fatal": [], "errors": [], "fixture_errors": []}
    if not path.exists():
        out["missing"] = True
        return out
    for line in path.read_text(errors="replace").splitlines():
        s = line.strip()
        if m := RE["header"].match(s):
            out["header"] = {"pid": int(m[1]), "accel": m[2], "gpu_precision": m[3], "burst": int(m[4]), "fixtures": int(m[6]),
                             "warmup": int(m[7]), "rounds": int(m[8]), "ab_fixture": m[9], "wall_ms": int(m[10])}
        elif m := RE["input"].match(s):
            out["inputs"].append({"index": int(m[1]), "sig_name": m[2], "logical": m[3], "elems": int(m[4]), "rank": int(m[5])})
        elif m := RE["compile_start"].match(s):
            out["compile_start_wall_ms"] = int(m[1])
        elif m := RE["compile_end"].match(s):
            out["compile_end_wall_ms"] = int(m[1])
        elif m := RE["model"].match(s):
            out["model"] = {"path": m[1], "inputs": int(m[2]), "outputs": int(m[3]), "out0_elems": int(m[4]),
                            "compile_ms": float(m[5]), "fully_accelerated": m[6] == "true"}
        elif m := RE["parity"].match(s):
            out["parity"][f"{int(m[1]):03d}"] = {"ms": float(m[2]), "run_ms": float(m[3]), "nonfinite": int(m[4])}
        elif m := RE["parity_summary"].match(s):
            out["parity_summary"] = {"fixtures": int(m[1]), "nonfinite_values": int(m[2]), "first_run_ms": float(m[3]),
                                     "mean_ms": float(m[4])}
        elif m := RE["latency_start"].match(s):
            out["latency_start_wall_ms"] = int(m[1])
        elif m := RE["warmup"].match(s):
            out["warmup"].append([float(m[2]), float(m[3])])
        elif m := RE["round"].match(s):
            out["rounds"].append([float(m[2]), float(m[3])])
        elif m := RE["summary"].match(s):
            out["summary"] = {"n": int(m[1]), "median": float(m[2]), "min": float(m[3]), "max": float(m[4]),
                              "nonfinite_rounds": int(m[5]), "compile_ms": float(m[6]), "fully_accelerated": m[7] == "true"}
        elif m := RE["done"].match(s):
            out["done_wall_ms"] = int(m[1])
        elif s.startswith("FATAL"):
            out["fatal"].append(s)
        elif s.startswith("FIXTURE_ERROR") or s.startswith("WRITE_ERROR"):
            out["fixture_errors"].append(s)
        elif s.startswith("ERROR"):
            out["errors"].append(s)
    return out


def delegation(text: str, where: str) -> dict:
    parts = [{"subgraph": int(m[1]), "selected": int(m[2]), "total": int(m[3]), "partitions": int(m[4])}
             for m in PARTITIONED.finditer(text)]
    repl = [{"replaced": int(m[1]), "of": int(m[2]), "delegate": m[3], "partitions": int(m[4]),
             "subgraph": int(m[5]) if m[5] is not None else None, "subgraph_name": m[6], "line": m[0]}
            for m in REPLACING.finditer(text)]
    rej = []
    for m in REJECT.finditer(text):
        rej.append(m[0])
    lines = text.splitlines()
    return {"source": where, "partitioned": parts, "replacing": repl,
            "partitioned_lines": [ln.strip() for ln in lines if "Partitioned subgraph" in ln][:8],
            "replacing_lines": [ln.strip() for ln in lines if REPLACING.search(ln)][:8],
            "compiler_plugin_lines": [ln.strip() for ln in lines if re.search(r"compiler plugins? were applied", ln)][:4],
            "jit_lines": [ln.strip() for ln in lines if re.search(r"(?i)reserializ|initialized from cached model|caching enabled|"
                                                                   r"Creating new QNN context|Reusing cached QNN context|"
                                                                   r"Context binary \d+ generated", ln)][:12],
            "qnn_rejected_ops": sorted(set(rej))[:40],
            "key_lines": [ln.strip() for ln in lines if KEY_LINE.search(ln)][:200]}


def pid_lines(logcat: Path, pid) -> str:
    if not logcat.exists() or pid is None:
        return ""
    pat = re.compile(rf"^\S+ \S+\s+{pid}\s+\d+ ")
    return "\n".join(ln for ln in logcat.read_text(errors="replace").splitlines() if pat.match(ln))


def score_leg(a) -> int:
    rd = Path(a.round_dir).resolve() if a.round_dir else K / "device" / a.round
    res_dir = Path(a.results).resolve() if a.results else K / "results"
    tag = a.tag
    leg = json.loads((rd / f"{tag}.leg.json").read_text())
    so = parse_stdout(rd / f"{tag}.stdout.txt")
    stdout_text = (rd / f"{tag}.stdout.txt").read_text(errors="replace") if (rd / f"{tag}.stdout.txt").exists() else ""
    logcat = rd / f"{tag}.logcat_all.txt"
    pid = (so.get("header") or {}).get("pid")
    lpid = pid_lines(logcat, pid)
    graph = leg["graph"]
    man_path = K / leg["manifest"]
    man = json.loads(man_path.read_text())
    if man.get("kind", "text") != "text":     # round 12: the audio / vision tower / projector graphs
        return score_generic_leg(a, rd, res_dir, tag, leg, so, stdout_text, logcat, pid, lpid, man_path, man)
    L = int(man["L"])
    count = int(leg["count"])
    mrows = man["rows"][:count]
    order = [x["name"] for x in man["order"]["inputs_subgraph_order"]]
    shapes = {x["name"]: int(np.prod(x["shape"])) for x in man["order"]["inputs_subgraph_order"]}
    ranks = {x["name"]: len(x["shape"]) for x in man["order"]["inputs_subgraph_order"]}
    seen = so["inputs"]
    input_check = {"runner_lines": seen, "manifest_subgraph_order": order,
                   "order_equal": [x["sig_name"] for x in seen] == order and all(x["logical"] == x["sig_name"] for x in seen),
                   "elems_equal": all(shapes.get(x["sig_name"]) == x["elems"] and ranks.get(x["sig_name"]) == x["rank"] for x in seen)
                   and len(seen) == len(order)}
    odoc, orows = SS.oracle_rows()
    dump = rd / f"{tag}.dump"
    macdir = K / leg["mac_dir"]
    lit, per, gate_rows, extra_rows = {}, [], [], []
    missing, bad_size = [], []
    mac = {"rows": 0, "bit_equal": 0, "max_abs_dscore_markers": None, "max_abs_dscore_real": None, "max_abs_dp": None,
           "argmax_agree": 0}
    md_m, md_r, mdp = [], [], []
    for m in mrows:
        f = dump / f"{m['id3']}.f32"
        if not f.exists():
            missing.append(m["id3"])
            continue
        s = np.fromfile(f, dtype="<f4")
        if s.size != L:
            bad_size.append([m["id3"], int(s.size)])
            continue
        o = orows[m["key"]]
        assert o["P"] == m["P"] and o["n"] == m["n"] and o["K"] == m["K"] and o["markers"] == m["markers"], m["key"]
        sel = s[[m["P"] + mm for mm in m["markers"][: m["K"]]]].astype(np.float32)
        p, z = SS.readout(sel, o)
        lit[m["key"]] = (p, z)
        real = m["P"] + m["n"]
        nonfin_real = int((~np.isfinite(s[:real])).sum())
        e = {"id3": m["id3"], "key": m["key"], "role": m["role"], "mode": o["mode"], "P": m["P"], "n": m["n"], "K": m["K"],
             "type": o["type"], "probs": p, "logits": z, "probs_oracle": o["probs"], "logits_oracle": o["logits"],
             "max_abs_dp": (max(abs(x - y) for x, y in zip(p, o["probs"])) if np.isfinite(p).all() else None),
             "near_tie": o["near_tie"], "argmax_equal": bool(np.isfinite(p).all() and int(np.argmax(p)) == int(np.argmax(o["probs"]))),
             "uniform_markers": bool(np.isfinite(sel).all() and float(sel.max() - sel.min()) == 0.0),
             "nonfinite_real": nonfin_real, "nonfinite_all": int((~np.isfinite(s)).sum()),
             "parity_pass_ms": (so["parity"].get(m["id3"]) or {}).get("ms"),
             "parity_pass_run_ms": (so["parity"].get(m["id3"]) or {}).get("run_ms")}
        mf = macdir / f"{m['id3']}.f32"
        if mf.exists():
            ms_ = np.fromfile(mf, dtype="<f4")
            mk = ms_[[m["P"] + mm for mm in m["markers"][: m["K"]]]]
            dm = float(np.abs(sel.astype(np.float64) - mk.astype(np.float64)).max())
            dr = float(np.abs(s[:real].astype(np.float64) - ms_[:real].astype(np.float64)).max())
            pm, _ = SS.readout(mk.astype(np.float32), o)
            e["mac_cpu_max_abs_dscore_markers"], e["mac_cpu_max_abs_dscore_real"] = dm, dr
            e["mac_cpu_max_abs_dp"] = max(abs(x - y) for x, y in zip(p, pm)) if np.isfinite(p).all() else None
            mac["rows"] += 1
            mac["bit_equal"] += int(np.array_equal(s[:real], ms_[:real]))
            if np.isfinite(dm):
                md_m.append(dm)
                md_r.append(dr)
            if e["mac_cpu_max_abs_dp"] is not None:
                mdp.append(e["mac_cpu_max_abs_dp"])
                mac["argmax_agree"] += int(int(np.argmax(p)) == int(np.argmax(pm)))
        per.append(e)
        (gate_rows if m["role"] == "gate" else extra_rows).append(o)
    mac.update(max_abs_dscore_markers=max(md_m, default=None), max_abs_dscore_real=max(md_r, default=None),
               max_abs_dp=max(mdp, default=None), dir=rel(macdir) if macdir.exists() else None)
    ref = {k: (o["probs"], o["logits"]) for k, o in orows.items()}
    st = SS.compare(gate_rows, lit, ref) if gate_rows else None
    st_x = SS.compare(extra_rows, lit, ref) if extra_rows else None
    ctl = None
    if SS.CONTROL in lit and np.isfinite(lit[SS.CONTROL][0]).all():
        d = max(abs(x - y) for x, y in zip(lit[SS.CONTROL][0], ref[SS.CONTROL_REF][0]))
        ctl = {"graph_row": SS.CONTROL, "reference_row": SS.CONTROL_REF, "max_abs_dp": d, "exceeds": d > SS.BAR["red_arm_min_dp"]}
    uni = [e["key"] for e in per if e["role"] == "gate" and e["uniform_markers"]]
    deleg_out = delegation(stdout_text, f"{tag}.stdout.txt")
    deleg_log = delegation(lpid, f"{tag}.logcat_all.txt lines of pid {pid}")
    smp = SS.parse_samples(rd / f"{tag}.samples.txt")
    part = (deleg_out["partitioned"] or deleg_log["partitioned"] or [None])[0]
    disp = [r for r in deleg_out["replacing"] + deleg_log["replacing"] if r["delegate"] == "DispatchDelegate"]
    # a cached run never runs the compiler plugin (no Partitioned line): its proof is the cache line + DispatchDelegate
    from_cache = any("initialized from cached model" in ln for ln in deleg_out["jit_lines"] + deleg_log["jit_lines"])
    summary = None
    if st:
        summary = {"rows": st["rows_compared"], "max_abs_dp": st["max_abs_dp"], "p95_abs_dp": st["p95_abs_dp"],
                   "mean_abs_dp": st["mean_abs_dp"], "max_abs_dlogit": st["max_abs_dlogit"],
                   "argmax_outside_near_tie": f"{st['argmax']['equal_outside_near_tie']}/{st['argmax']['rows_outside_near_tie']}",
                   "near_tie": f"{st['argmax']['near_tie_equal']}/{st['argmax']['near_tie_rows']}",
                   "cutoff_crossings": st["cutoff_crossings"], "nonfinite_rows": len(st["nonfinite_rows"]),
                   "uniform_rows": len(uni), "collapse": bool(uni) and len(uni) == st["rows_compared"],
                   "red_arm_max_abs_dp": (ctl or {}).get("max_abs_dp"),
                   # FACTS §7 (s26_score.compare's bar, rounds 3-9) + the runner's input order; a row whose K scores tie
                   # (one near-tie row in fp16 storage) is reported, not failed; every row uniform = the collapse
                   "bar_pass": bool(st["bar_pass"]) and input_check["order_equal"] and input_check["elems_equal"]}
    doc = {
        "step": f"round {a.round[1:]}: d1-omni decision graph on the S26 through the C-API runner (HTP JIT / GPU) vs the oracle "
                "(scripts/npu_score.py leg)",
        "tag": tag, "round": a.round, "scored_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "leg": leg, "graph": {"tag": graph, "file": graph_file(graph), "L": L},
        "fixtures": {"manifest": rel(man_path), "dir_on_mac": man["dir"],
                     "concat_sha256": man["concat_sha256_bytesorted"], "count_run": count, "gate_rows": len(gate_rows),
                     "extra_rows": len(extra_rows), "dumps_missing": missing, "dumps_bad_size": bad_size},
        "runner": {k: so.get(k) for k in ("header", "model", "parity_summary", "summary", "compile_start_wall_ms",
                                         "compile_end_wall_ms", "latency_start_wall_ms", "done_wall_ms")}
                  | {"fatal": so["fatal"], "errors": so["errors"][:20], "fixture_errors": so["fixture_errors"][:10],
                     "stdout": rel(rd / f"{tag}.stdout.txt"),
                     "stdout_sha256": sha256(rd / f"{tag}.stdout.txt") if (rd / f"{tag}.stdout.txt").exists() else None},
        "input_order_check": input_check,
        "bar": SS.BAR, "summary": summary, "oracle_comparison": st, "extra_rows_comparison": st_x,
        "red_arm": ctl or {"available": False}, "uniform_rows": uni,
        "same_file_mac_cpu": mac,
        "delegation": {
            "npu_partition": part, "dispatch_delegate_lines": [r["line"] for r in disp][:6], "from_cache": from_cache,
            "npu_evidence": bool(disp) and (bool(part) or from_cache),
            "cpu_ops_left": (part["total"] - part["selected"]) if part else None,
            "fully_accelerated_runner": (so.get("model") or {}).get("fully_accelerated"),
            "stdout": deleg_out, "logcat_pid": deleg_log,
            "logcat": rel(logcat) if logcat.exists() else None, "logcat_pid_lines": len(lpid.splitlines()) if lpid else 0},
        "samples": SS.samples_summary(smp, 1300),
        "state_before": SS.parse_state(rd / f"{tag}.state_before.txt"),
        "state_after": SS.parse_state(rd / f"{tag}.state_after.txt"),
        "per_row": per,
    }
    out = write_new(res_dir / f"s26_npu_parity_{tag}_{a.round}.json", doc)
    print(f"{tag}: status {leg.get('status')} rows {len(per)}/{count} "
          + (f"max|dp| {summary['max_abs_dp']:.4g} mean {summary['mean_abs_dp']:.3g} argmax {summary['argmax_outside_near_tie']} "
             f"near-tie {summary['near_tie']} nonfinite {summary['nonfinite_rows']} uniform {summary['uniform_rows']} "
             f"bar {'PASS' if summary['bar_pass'] else 'FAIL'}" if summary else "no rows")
          + f"; partition {part}; dispatch lines {len(disp)}; input order {input_check['order_equal']} -> {rel(out)}")
    if so["rounds"] or so["parity"]:
        calls = [c[0] for c in so["rounds"]]
        runs = [c[1] for c in so["rounds"]]
        gate_ids = [m["id3"] for m in mrows if m["role"] == "gate"]
        pp = [so["parity"][i]["ms"] for i in gate_ids if i in so["parity"]]
        setb = [so["parity"][i]["ms"] for i in man["set_b_fixtures"] if i in so["parity"]]
        tdoc_extra = timing_doc(a, rd, tag, leg, so, man, smp=SS.parse_samples(rd / f"{tag}.samples.txt"),
                                deleg_out=deleg_out, deleg_log=deleg_log, pp=pp, setb=setb)
        tout = write_new(res_dir / f"s26_npu_timing_{tag}_{a.round}.json", tdoc_extra)
        lp = tdoc_extra["latency_phase"]["per_call_ms_write_run_read"]
        print(f"{tag}: latency {lp} parity-pass median {(tdoc_extra['parity_pass']['per_call_ms_gate_rows'] or {}).get('median')} "
              f"compile {tdoc_extra['compile']} -> {rel(tout)}")
    return 0


CUT_REF = K / "out/r7_eager_prefix_T501_aud01cut5.npz"
LONG_REF = K / "out/r7_eager_prefix_T3001_long.npz"
PROJECTOR_MAC = K / "out/d1omni_projector_fp16.tflite"


def adiff(a_, b_) -> dict:
    a64, b64 = np.asarray(a_, np.float64), np.asarray(b_, np.float64)
    assert a64.shape == b64.shape, (a64.shape, b64.shape)
    d = np.abs(a64 - b64)
    rr = float(np.sqrt((b64 ** 2).mean())) if b64.size else 0.0
    fin = np.isfinite(d)
    return {"max_abs": float(d[fin].max()) if fin.any() else None, "mean_abs": float(d[fin].mean()) if fin.any() else None,
            "rel_rms": float(np.sqrt((d[fin] ** 2).mean()) / rr) if rr and fin.any() else None,
            "bit_equal": bool(np.array_equal(np.asarray(a_), np.asarray(b_))), "nonfinite": int((~np.isfinite(a64)).sum())}


def score_generic_leg(a, rd, res_dir, tag, leg, so, stdout_text, logcat, pid, lpid, man_path, man) -> int:
    """Round 12: a leg of an audio / vision tower / projector graph. Output 0 (the runner's --dump full) per fixture vs
    the Mac CPU run of the same file on the same fixture (device/<round>/mac/<graph>/model0/<id3>.f32, from
    scripts/npu_fixtures.py check-specs) and vs the reference (audio: the oracle's prefix, the provider's own prefix of
    the 5 s cut and the 27.2 s clip; tower: the oracle's tower_last_hidden_state; projector: the record prefix vs the
    oracle's); the prefixes for the end to end -> <round dir>/<tag>.prefix.npz (a tower leg's features go through the host
    unshuffle and the Mac CPU projector fp16 file, as round 9's e2e of the Mac chain did). No bar in this file: the
    PASS / FAIL of these graphs is the end to end (`npu_score.py e2e`, FACTS §7)."""
    import d1_vision_host as V

    kind = man["kind"]
    count = int(leg["count"])
    mrows = man["rows"][:count]
    oshape = [int(v) for v in man["out_shape"]]
    n_out = int(man["out_elems"])
    order = [x["name"] for x in man["order"]["inputs_subgraph_order"]]
    shapes = {x["name"]: int(np.prod(x["shape"])) for x in man["order"]["inputs_subgraph_order"]}
    ranks = {x["name"]: len(x["shape"]) for x in man["order"]["inputs_subgraph_order"]}
    seen = so["inputs"]
    input_check = {"runner_lines": seen, "manifest_subgraph_order": order,
                   "order_equal": [x["sig_name"] for x in seen] == order and all(x["logical"] == x["sig_name"] for x in seen),
                   "elems_equal": all(shapes.get(x["sig_name"]) == x["elems"] and ranks.get(x["sig_name"]) == x["rank"] for x in seen)
                   and len(seen) == len(order)}
    dump = rd / f"{tag}.dump"
    macdir = K / leg["mac_dir"]
    per, missing, bad_size, prefixes, by_rec = [], [], [], {}, {}
    proj = None
    for m in mrows:
        f = dump / f"{m['id3']}.f32"
        if not f.exists():
            missing.append(m["id3"])
            continue
        o = np.fromfile(f, dtype="<f4")
        if o.size != n_out:
            bad_size.append([m["id3"], int(o.size)])
            continue
        o = o.reshape(oshape)[0]
        n = int(m["P"] if kind == "audio" else m["patches"] if kind == "vision_tower" else m["prefix_rows"])
        e = {"id3": m["id3"], "key": m["key"], "rows": n, "nonfinite_rows": int((~np.isfinite(o[:n])).sum()),
             "nonfinite_all": int((~np.isfinite(o)).sum()),
             "parity_pass_ms": (so["parity"].get(m["id3"]) or {}).get("ms"),
             "parity_pass_run_ms": (so["parity"].get(m["id3"]) or {}).get("run_ms")}
        mf = macdir / f"{m['id3']}.f32"
        if mf.exists():
            e["vs_mac_cpu_same_file"] = adiff(o[:n], np.fromfile(mf, dtype="<f4").reshape(oshape)[0][:n])
        if kind == "audio":
            if m.get("record"):
                with np.load(K / f"ref/npz/{m['record']}.npz") as z:
                    e["vs_oracle_prefix"] = adiff(o[:n], np.asarray(z["prefix"]))
                prefixes[m["record"]] = o[:n].astype(np.float32)
            elif m["key"] == "aud_01_cut5":
                e["vs_provider_prefix"] = adiff(o[:n], np.load(CUT_REF)["provider"])
            elif m["key"] == "long":
                e["vs_provider_prefix"] = adiff(o[:n], np.load(LONG_REF)["provider_prefix"])
        elif kind == "vision_tower":
            with np.load(K / f"ref/npz/{m['record']}.npz") as z:
                e["vs_oracle_features"] = adiff(o[:n], np.asarray(z["tower_last_hidden_state"][m["crop"], :n]))
            if np.isfinite(o[:n]).all():
                cells = V.pixel_unshuffle(o[:n], tuple(m["grid"]))
                if proj is None:
                    proj = V.LiteRTGraph(PROJECTOR_MAC, "cpu", "fp32", threads=8)
                by_rec.setdefault(m["record"], []).append((m["crop"], proj(soft=V.projector_input(cells))[0][: cells.shape[0]]))
        else:
            by_rec.setdefault(m["record"], []).append((m["crop"], o[:n].copy()))
        per.append(e)
    if proj is not None:
        proj.close()
    crops_of = {}
    for m in man["rows"]:
        if m.get("record") and "crop" in m:
            crops_of[m["record"]] = crops_of.get(m["record"], 0) + 1
    records = []
    for rid, parts in by_rec.items():
        if len(parts) != crops_of.get(rid):
            records.append({"record": rid, "complete": False, "crops": len(parts), "of": crops_of.get(rid)})
            continue
        pre = np.concatenate([p for _, p in sorted(parts, key=lambda t: t[0])]).astype(np.float32)
        prefixes[rid] = pre
        with np.load(K / f"ref/npz/{rid}.npz") as z:
            records.append({"record": rid, "complete": True, "P": int(pre.shape[0]), "vs_oracle_prefix": adiff(pre, np.asarray(z["prefix"]))})
    pnpz = None
    if prefixes:
        pnpz = rd / f"{tag}.prefix.npz"
        if pnpz.exists():
            raise FileExistsError(f"refusing to overwrite {rel(pnpz)}")
        np.savez(pnpz, **prefixes)
    deleg_out = delegation(stdout_text, f"{tag}.stdout.txt")
    deleg_log = delegation(lpid, f"{tag}.logcat_all.txt lines of pid {pid}")
    smp = SS.parse_samples(rd / f"{tag}.samples.txt")
    part = (deleg_out["partitioned"] or deleg_log["partitioned"] or [None])[0]
    disp = [r for r in deleg_out["replacing"] + deleg_log["replacing"] if r["delegate"] == "DispatchDelegate"]
    from_cache = any("initialized from cached model" in ln for ln in deleg_out["jit_lines"] + deleg_log["jit_lines"])
    mx = lambda key, sub: max((e[key][sub] for e in per if key in e and e[key].get(sub) is not None), default=None)
    summary = {"kind": kind, "fixtures": len(per), "of": count, "dumps_missing": missing, "dumps_bad_size": bad_size,
               "nonfinite_rows_total": sum(e["nonfinite_rows"] for e in per),
               "vs_mac_cpu_same_file_max_abs": mx("vs_mac_cpu_same_file", "max_abs"),
               "vs_mac_cpu_same_file_rel_rms_max": mx("vs_mac_cpu_same_file", "rel_rms"),
               "vs_mac_cpu_same_file_bit_equal": f"{sum(bool((e.get('vs_mac_cpu_same_file') or {}).get('bit_equal')) for e in per)}/{len(per)}",
               "vs_oracle_prefix_max_abs": mx("vs_oracle_prefix", "max_abs"), "vs_oracle_prefix_rel_rms_max": mx("vs_oracle_prefix", "rel_rms"),
               "vs_provider_prefix_max_abs": mx("vs_provider_prefix", "max_abs"),
               "vs_oracle_features_max_abs": mx("vs_oracle_features", "max_abs"),
               "records_prefix_vs_oracle_max_abs": max((r["vs_oracle_prefix"]["max_abs"] for r in records if r.get("complete")
                                                        and r["vs_oracle_prefix"]["max_abs"] is not None), default=None),
               "prefix_npz": rel(pnpz) if pnpz else None, "records_in_npz": sorted(prefixes),
               "input_order_ok": input_check["order_equal"] and input_check["elems_equal"],
               "finite_pass": bool(not missing and not bad_size and sum(e["nonfinite_rows"] for e in per) == 0),
               "bar_note": "no bar here: the PASS / FAIL of the audio / vision graphs is the end to end (npu_score.py e2e)"}
    doc = {
        "step": f"round {a.round[1:]}: d1-omni {kind} graph on the S26 through the C-API runner (scripts/npu_score.py leg)",
        "tag": tag, "round": a.round, "scored_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "leg": leg,
        "graph": {"tag": leg["graph"], "file": graph_file(leg["graph"]), "kind": kind, "out_shape": oshape},
        "fixtures": {"manifest": rel(man_path), "dir_on_mac": man["dir"], "concat_sha256": man["concat_sha256_bytesorted"],
                     "count_run": count},
        "runner": {k: so.get(k) for k in ("header", "model", "parity_summary", "summary", "compile_start_wall_ms",
                                         "compile_end_wall_ms", "latency_start_wall_ms", "done_wall_ms")}
                  | {"fatal": so["fatal"], "errors": so["errors"][:20], "fixture_errors": so["fixture_errors"][:10],
                     "stdout": rel(rd / f"{tag}.stdout.txt"),
                     "stdout_sha256": sha256(rd / f"{tag}.stdout.txt") if (rd / f"{tag}.stdout.txt").exists() else None},
        "input_order_check": input_check, "summary": summary, "per_fixture": per, "records": records,
        "delegation": {
            "npu_partition": part, "dispatch_delegate_lines": [r["line"] for r in disp][:6], "from_cache": from_cache,
            "npu_evidence": bool(disp) and (bool(part) or from_cache),
            "cpu_ops_left": (part["total"] - part["selected"]) if part else None,
            "fully_accelerated_runner": (so.get("model") or {}).get("fully_accelerated"),
            "stdout": deleg_out, "logcat_pid": deleg_log,
            "logcat": rel(logcat) if logcat.exists() else None, "logcat_pid_lines": len(lpid.splitlines()) if lpid else 0},
        "samples": SS.samples_summary(smp, 1300),
        "state_before": SS.parse_state(rd / f"{tag}.state_before.txt"),
        "state_after": SS.parse_state(rd / f"{tag}.state_after.txt")}
    out = write_new(res_dir / f"s26_npu_parity_{tag}_{a.round}.json", doc)
    repl = (deleg_out["replacing"] or deleg_log["replacing"] or [None])[0]
    print(f"{tag}: status {leg.get('status')} {kind} fixtures {len(per)}/{count} vs Mac same file max "
          f"{summary['vs_mac_cpu_same_file_max_abs']} nonfinite {summary['nonfinite_rows_total']}; replacing "
          f"{repl and repl['line']}; partition {part}; input order {summary['input_order_ok']} -> {rel(out)}")
    if so["rounds"] or so["parity"]:
        pp = [so["parity"][m["id3"]]["ms"] for m in mrows if m["id3"] in so["parity"]]
        tdoc = timing_doc(a, rd, tag, leg, so, man, smp=smp, deleg_out=deleg_out, deleg_log=deleg_log, pp=pp, setb=[])
        tout = write_new(res_dir / f"s26_npu_timing_{tag}_{a.round}.json", tdoc)
        print(f"{tag}: latency {tdoc['latency_phase']['per_call_ms_write_run_read']} -> {rel(tout)}")
    return 0


def e2e_leg(a) -> int:
    """Round 12: the end to end of an audio / vision leg on the Mac: the phone's prefixes (<round dir>/<tag>.prefix.npz)
    through the fp16 text graph on the Mac CPU -> the host read-out -> vs the oracle (s26_score.py e2e9 = round 9's
    scorer: audio L256, images the smallest of L256 .. L4096 that holds the row; FACTS §7)."""
    from types import SimpleNamespace

    rd = Path(a.round_dir).resolve() if a.round_dir else K / "device" / a.round
    res_dir = Path(a.results).resolve() if a.results else K / "results"
    leg = json.loads((rd / f"{a.tag}.leg.json").read_text())
    man = json.loads((K / leg["manifest"]).read_text())
    kind = {"audio": "audio", "vision_tower": "vision", "projector": "vision"}[man["kind"]]
    pnpz = rd / f"{a.tag}.prefix.npz"
    with np.load(pnpz) as z:
        recs = sorted(z.files)
    out = res_dir / f"s26_npu_parity_{a.tag}_e2e_{a.round}.json"
    rc = SS.e2e9(SimpleNamespace(prefix=[str(pnpz)], kind=kind, tag=a.tag, out=str(out)))
    doc = json.loads(out.read_text())
    doc["step"] = (f"round {a.round[1:]}: end to end of the S26 {kind} prefixes (C-API runner leg {a.tag}) through the Mac "
                   "CPU text graph (scripts/npu_score.py e2e = s26_score.py e2e9)")
    doc["records_in_npz"] = recs
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    tmp.replace(out)
    return rc


def timing_doc(a, rd, tag, leg, so, man, smp, deleg_out, deleg_log, pp, setb) -> dict:
    """The latency json of a leg (round 10's body, shared by the text and the generic legs since round 12)."""
    if True:
        calls = [c[0] for c in so["rounds"]]
        runs = [c[1] for c in so["rounds"]]
        lat_t = so.get("latency_start_wall_ms")
        if not man["set_b_fixtures"]:
            setb = []
        tdoc = {
            "step": "round 10: the C-API runner's latency on the S26 (scripts/npu_score.py leg)",
            "tag": tag, "round": a.round, "scored_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "leg": leg,
            "definition_ms": "the runner's call = write the six inputs (memcpy into locked input buffers) + "
                             "LiteRtRunCompiledModel + read the scores output back; run-only = LiteRtRunCompiledModel alone",
            "compile": {"runner_compile_ms": (so.get("model") or {}).get("compile_ms"),
                        "wall_ms": (so["compile_end_wall_ms"] - so["compile_start_wall_ms"])
                        if so.get("compile_end_wall_ms") and so.get("compile_start_wall_ms") else None,
                        "from_cache": any("initialized from cached model" in ln for ln in deleg_out["jit_lines"] + deleg_log["jit_lines"]),
                        "jit_lines": deleg_out["jit_lines"]},
            "latency_phase": {"ab_fixture": (so.get("header") or {}).get("ab_fixture"),
                              "ab_key": next((m["key"] for m in man["rows"] if m["id3"] == (so.get("header") or {}).get("ab_fixture")), None),
                              "warmup_calls": so["warmup"], "calls": so["rounds"],
                              "per_call_ms_write_run_read": stats(calls), "per_call_ms_run_only": stats(runs),
                              "runner_summary": so.get("summary"),
                              "state_at_start": SS.state_near(lat_t, smp, 1300) if lat_t else None},
            "parity_pass": {"per_call_ms_gate_rows": stats(pp), "first_run_ms": (so.get("parity_summary") or {}).get("first_run_ms"),
                            "set_b_keys": man["set_b"], "set_b_fixtures": man["set_b_fixtures"],
                            "set_b_request_ms": sum(setb) if setb and len(setb) == len(man["set_b_fixtures"]) else None,
                            "set_b_calls_ms": setb},
            "samples": SS.samples_summary(smp, 1300),
            "state_before": SS.parse_state(rd / f"{tag}.state_before.txt"),
            "state_after": SS.parse_state(rd / f"{tag}.state_after.txt"),
            "ready_note": (rd / f"{tag}.ready.txt").read_text().strip() if (rd / f"{tag}.ready.txt").exists() else None,
        }
        # 19:3x: the set's starting state on both sides of a precision pair (thermal / CPU cap / kgsl), at
        # the leg's start (the ready gate's reading and the state file) and at the latency phase's first call
        sb = tdoc["state_before"] or {}
        st0 = tdoc["latency_phase"]["state_at_start"] or {}
        tdoc["set_start_state"] = {
            "ready_note": tdoc["ready_note"],
            "leg_start": {"thermal": sb.get("thermal"), "freq_capped": sb.get("freq_capped"), "kgsl": sb.get("kgsl")},
            "latency_start": {"thermal_status": st0.get("thermal_status"), "cpu_capped": st0.get("cpu_capped"),
                              "kgsl": st0.get("kgsl"), "kgsl_uncapped": st0.get("kgsl_uncapped"),
                              "sample_dt_ms": st0.get("sample_dt_ms")},
            "capped_at_latency_start": (bool(st0.get("cpu_capped")) or not st0.get("kgsl_uncapped")) if st0 else None}
        # round 12 (§2j (d)): the calls split by the nearest 2 s sample (burst = kgsl at 1,300 MHz with
        # thermal_pwrlevel 0 and every CPU policy at its top; round 9's rule); the runner stamps the latency start, the
        # calls run back to back, so call i starts at latency_start + the warm-up calls + calls 0 .. i-1
        if lat_t and calls:
            t, starts = float(lat_t) + sum(w[0] for w in so["warmup"]), []
            for c in calls:
                starts.append(t)
                t += c
            tdoc["latency_phase"]["burst_split"] = SS.classify(list(zip(starts, calls)), smp, 1300)
            tdoc["latency_phase"]["burst_split"]["rule"] = ("nearest 2 s sample within 3 s: burst = kgsl max_clock 1,300 MHz "
                                                            "+ thermal_pwrlevel 0 + every CPU policy at cpuinfo_max_freq")
        if a.round != "r10":
            tdoc["step"] = f"round {a.round[1:]}: the C-API runner's latency on the S26 (scripts/npu_score.py leg)"
            tdoc["definition_ms"] = ("the runner's call = write every input (memcpy into locked input buffers) + "
                                     "LiteRtRunCompiledModel + read output 0 back; run-only = LiteRtRunCompiledModel alone")
    return tdoc


def _f(v, nd=3):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}g}" if abs(v) < 1e-2 or abs(v) >= 1e5 else f"{v:.{nd + 1}g}"
    return str(v)


def table(a) -> int:
    rd = Path(a.round_dir).resolve() if a.round_dir else K / "device" / a.round
    res_dir = Path(a.results).resolve() if a.results else K / "results"
    legs = sorted(p.name[: -len(".leg.json")] for p in rd.glob("*.leg.json"))
    rows = []
    print("| leg | graph | backend | status | delegation (runner stdout / logcat) | parity (gate rows) | 1 call ms median (min–max, n) | compile s | MemAvailable min kB / runner VmHWM kB | thermal seen |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for tag in legs:
        leg = json.loads((rd / f"{tag}.leg.json").read_text())
        pj = res_dir / f"s26_npu_parity_{tag}_{a.round}.json"
        tj = res_dir / f"s26_npu_timing_{tag}_{a.round}.json"
        p = json.loads(pj.read_text()) if pj.exists() else None
        t = json.loads(tj.read_text()) if tj.exists() else None
        s = (p or {}).get("summary")
        d = (p or {}).get("delegation") or {}
        part = d.get("npu_partition")
        repl = [r for r in ((d.get("stdout") or {}).get("replacing") or [])]
        dl = "; ".join(f"{r['replaced']}/{r['of']} {r['delegate']} {r['partitions']} part." for r in repl[:2])
        if part:
            dl = f"HTP {part['selected']}/{part['total']} ops → {part['partitions']} part.; " + dl
        lat = ((t or {}).get("latency_phase") or {}).get("per_call_ms_write_run_read")
        comp = ((t or {}).get("compile") or {}).get("runner_compile_ms")
        smp = (p or {}).get("samples") or {}
        e2e = None
        if s and s.get("kind"):     # round 12: a generic leg = the Mac same-file distance + the end to end (if scored)
            ej = res_dir / f"s26_npu_parity_{tag}_e2e_{a.round}.json"
            e2e = json.loads(ej.read_text())["summary"] if ej.exists() else None
            par = (f"{s['kind']} {s['fixtures']}/{s['of']}: vs Mac same file max {_f(s['vs_mac_cpu_same_file_max_abs'])} "
                   f"(rel_rms {_f(s['vs_mac_cpu_same_file_rel_rms_max'])}), non-finite rows {s['nonfinite_rows_total']}; "
                   + (f"e2e {e2e['rows']} rows max {e2e['max_abs_dp']:.3g} / mean {e2e['mean_abs_dp']:.3g}, argmax "
                      f"{e2e['argmax_outside_near_tie']} → {'PASS' if e2e['bar_pass'] else 'FAIL'}" if e2e else "e2e —"))
        else:
            par = (f"{s['max_abs_dp']:.3g} / mean {s['mean_abs_dp']:.3g}, argmax {s['argmax_outside_near_tie']} + {s['near_tie']}, "
                   f"non-finite {s['nonfinite_rows']}, uniform {s['uniform_rows']} → {'PASS' if s['bar_pass'] else 'FAIL'}") if s else "—"
        ms = f"{lat['median']:.2f} ({lat['min']:.2f}–{lat['max']:.2f}, {lat['n']})" if lat else "—"
        print(f"| {tag} | {leg['graph']} | {leg['accel']}{' ' + leg['precision'] if leg['precision'] != '-' else ''} | "
              f"{leg.get('status')} | {dl or '—'} | {par} | {ms} | {_f(comp / 1000 if comp else None)} | "
              f"{smp.get('mem_available_kb_min', '—')} / {smp.get('app_vmhwm_kb_max', '—')} | {smp.get('thermal_status_seen', '—')} |")
        rows.append({"tag": tag, "leg": leg, "parity_file": rel(pj) if p else None, "timing_file": rel(tj) if t else None,
                     "summary": s, "e2e": e2e, "npu_partition": part, "replacing": repl, "latency": lat, "compile_ms": comp,
                     "burst_split": ((t or {}).get("latency_phase") or {}).get("burst_split"),
                     "set_start_state": (t or {}).get("set_start_state"), "samples": smp})
    # the GPU precision pairs (19:3x): measured one after the other from the same ready rule; a pair whose
    # sides started under different caps (one capped, the other not) is not a pair: both numbers stand, no ratio
    pairs = []
    print("\n| pair (same runner, same hold) | FP32 1 call ms | FP16_WITH_FP32_ACCUM 1 call ms | ready notes | capped at the latency start (FP32 / fp16acc32) | ratio |")
    print("|---|---|---|---|---|---|")
    for p32, p16 in plan_pairs(rd, a.round):
        t = {}
        for x in (p32, p16):
            f = res_dir / f"s26_npu_timing_{x}_{a.round}.json"
            t[x] = json.loads(f.read_text()) if f.exists() else None
        if not (t[p32] and t[p16]):
            pairs.append({"pair": [p32, p16], "available": False})
            print(f"| {p32} / {p16} | {'—' if not t[p32] else 'ran'} | {'—' if not t[p16] else 'ran'} | — | — | — (a side did not run) |")
            continue
        m32 = (t[p32]["latency_phase"]["per_call_ms_write_run_read"] or {}).get("median")
        m16 = (t[p16]["latency_phase"]["per_call_ms_write_run_read"] or {}).get("median")
        s32, s16 = t[p32].get("set_start_state") or {}, t[p16].get("set_start_state") or {}
        c32, c16 = s32.get("capped_at_latency_start"), s16.get("capped_at_latency_start")
        clean = all(str(s.get("ready_note") or "").startswith("clean") for s in (s32, s16))
        paired = c32 is not None and c32 == c16 and clean
        ratio = (m16 / m32) if paired and m32 and m16 else None
        pairs.append({"pair": [p32, p16], "available": True, "fp32_ms": m32, "fp16acc32_ms": m16,
                      "ready_notes": [s32.get("ready_note"), s16.get("ready_note")], "capped_at_latency_start": [c32, c16],
                      "set_start_state": [s32, s16], "paired": paired, "ratio_fp16acc32_over_fp32": ratio})
        print(f"| {p32} / {p16} | {_f(m32)} | {_f(m16)} | {s32.get('ready_note')} / {s16.get('ready_note')} | {c32} / {c16} | "
              f"{_f(ratio) if ratio else 'not a pair: no ratio'} |")
    out = res_dir / f"s26_npu_summary_{a.round}.json"
    write_new(out, {"step": f"round {a.round[1:]}: the NPU chain's legs (scripts/npu_score.py table)", "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "round_dir": rel(rd), "legs": rows, "gpu_precision_pairs": pairs})
    print(f"\n-> {rel(out)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("leg", "table", "e2e"):
        p = sub.add_parser(name)
        p.add_argument("--round", default="r10")
        p.add_argument("--round-dir")
        p.add_argument("--results")
        if name in ("leg", "e2e"):
            p.add_argument("--tag", required=True)
    a = ap.parse_args()
    if a.cmd == "e2e":
        return e2e_leg(a)
    return score_leg(a) if a.cmd == "leg" else table(a)


if __name__ == "__main__":
    sys.exit(main())
