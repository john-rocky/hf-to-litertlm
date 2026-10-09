"""Round 4 step 3-4: Mac timing of the D1Decision graphs (ai-edge-litert 2.2.0 CompiledModel, exporter venv lt094dev).

    cd d1_omni_work; Q=~/code/standup/tools/quiet
    $Q/quiet_hold.py d1c-r4-fp32fp16-L128-cpu -- ~/venvs/lt094dev/bin/python scripts/timing_mac.py \
        --backend cpu --L 128 --forms fp32,fp16          # one window = one backend x one L, fp32 then fp16
    $Q/quiet_hold.py d1c-r4-multisig-fp16 -- ~/venvs/lt094dev/bin/python scripts/timing_mac.py --multisig
    ~/venvs/lt094dev/bin/python scripts/timing_mac.py --aggregate          # no run: the per-form tables

Window file (never overwritten; a retake gets _take<N>): results/timing_mac_r4_<backend>_L<L>[_take<N>].json.
backend: cpu = XNNPACK 8 threads; gpu_fp32 = Metal, GpuOptions(enforce_f32=True); gpu_default = Metal default precision
(fp16 activations; a lever data point: its answers are known to collapse, round 3).
Per form, in this order inside the one window (the same-condition pair):
  1. contention gate: `top -l 1 | head -5` and the 1-minute load average are recorded before every set (the launch's
     record). The launch's rule "load above 8 = another lane's CPU job -> discard" cannot be met on 2026-10-08: with
     no lane job running the load stays 20-40 at 68-73 % idle CPU (Spotlight, 50 sessions' wakeups; the first window,
     results/timing_mac_r4_cpu_L128.json, waited 240 s at 19.9-40.0). The gate that decides is the knowledge rule
     (memory gates-can-lie-in-our-favour #12: no peer process above 120 % CPU) on the second sample of
     `top -l 2 -s 1 -o cpu` (instantaneous): no process outside this process tree above 120 % CPU and CPU idle >= 50 %.
     10 s polls, up to 300 s; still contended -> the set is not measured ("discarded: contention"), exit 3, and the
     window is taken again later.
  2. a fresh child process (one model per process: memory figures carry no other model's residue) compiles the file
     (CompiledModel.from_file = the "compile" seconds; the system Metal shader cache may be warm from earlier runs),
     records memory (proc_pid_rusage phys_footprint / lifetime max, ru_maxrss, ps rss) after the compile and at the
     end, then runs every workload of this L: 5 warm-up calls (times kept apart), then 20 timed calls (a request
     workload: 20 requests of its rows back to back). Per call: [wall clock ms at the start, ms of write + run + read
     back, ms of run only]. Every timed call's scores must be finite; the first timed call of each row is read out
     (host readout) and compared with the oracle (max |dp|) to show the timed graph answers the question.
Workloads (FACTS §7 / launch fact 4; rows = ref/records_ref.json version 2, inputs = the gate's row_inputs):
  L128  a  card_text/refund (47 tokens)                    b  card_text/refund + team + urgency (one request, 3 calls)
        e  img_dogs_01/count (P 84 + 32 = 116: its smallest bucket is L128)
        g  tv4_010/answer (84 tokens, the bucket's median row)
  L256  c  own_fiveq_09's 5 questions (132-148 tokens, one request, 5 calls)
        e_L256  img_dogs_01/count at L256 (the bucket the launch named)
        f  aud_01/topic (P 121 + 63 = 184)              g  aud_reservation_01/topic (P 103 + 54 = 157, the median row)
  L512  g  tv4_009/answer (339)     L1024  g  tv4x_sciq_11/answer (371 = L512's longest row, padded; no natural row)
  L2048 g  own_long_log_10/resolved (1,672)
  L4096 d  long_3400/component (3,467 = the 3.4k-token state)   g  long_3400/tension_10 (3,454, the median row)
--multisig: results/multisig_fp16.json — fresh child processes for (single-signature fp16 L256 file | the 6-signature
fp16 file) x (Metal fp32 precision with constant_tensor_sharing off | on, CPU 8 threads): compile seconds, memory after
the compile and at the end, decide_256 one call (g row, 5 warm-up + 20), and the first 50 L256 gate rows (scores kept)
-> multi vs single bit-equality on the same backend, and |dp| vs the oracle.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
K = HERE.parent
sys.path.insert(0, str(HERE))

WARMUP, REPS = 5, 20
LOAD_MAX, LOAD_WAIT_S, LOAD_POLL_S = 8.0, 300, 10   # LOAD_MAX = the launch's number, recorded against, not gated on
PEER_CPU_MAX, IDLE_MIN = 120.0, 50.0
LOCK = os.path.expanduser("~/code/coreai/_GPU_LOCK")
TMP = K / "out/r4_timing_tmp"
WORKLOADS = {
    128: [("a", "single", ["card_text/refund"]),
          ("b", "request", ["card_text/refund", "card_text/team", "card_text/urgency"]),
          ("e", "single", ["img_dogs_01/count"]),
          ("g", "single", ["tv4_010/answer"])],
    256: [("c", "request", ["own_fiveq_09/signed", "own_fiveq_09/where", "own_fiveq_09/damaged",
                            "own_fiveq_09/request_followed", "own_fiveq_09/photo"]),
          ("e_L256", "single", ["img_dogs_01/count"]),
          ("f", "single", ["aud_01/topic"]),
          ("g", "single", ["aud_reservation_01/topic"])],
    512: [("g", "single", ["tv4_009/answer"])],
    1024: [("g", "single", ["tv4x_sciq_11/answer"])],
    2048: [("g", "single", ["own_long_log_10/resolved"])],
    4096: [("d", "single", ["long_3400/component"]),
           ("g", "single", ["long_3400/tension_10"])],
}
WORKLOAD_NAMES = {"a": "1 question, short state (card_text/refund)",
                  "b": "3 questions in one request (card_text: 3 calls)",
                  "c": "5 questions in one request (own_fiveq_09: 5 calls)",
                  "d": "1 question on a 3.4k-token state (long_3400/component)",
                  "e": "384 px image, 1 question, text side (img_dogs_01/count; vision graph not included)",
                  "e_L256": "the same image row at L256 (the bucket the launch named; its smallest bucket is L128)",
                  "f": "10 s audio, 1 question, text side (aud_01/topic; audio graph not included)",
                  "g": "one call of the bucket (the bucket's median fixture row; L1024 = L512's longest row padded)"}


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def stats(xs):
    a = np.asarray(xs, np.float64)
    return {"median": round(float(np.median(a)), 3), "min": round(float(a.min()), 3), "max": round(float(a.max()), 3),
            "n": int(a.size)}


def file_for(L, form):
    return K / f"out/d1omni_decide_L{L}_{form}.tflite"


def rows_by_key(keys):
    import litert_gate as LG

    oracle = LG.oracle_doc()
    out = {}
    for rec in oracle["records"]:
        for q in rec["questions"]:
            key = f"{rec['id']}/{q['qid']}"
            if key in keys:
                P = int(q.get("prefix") or 0)
                out[key] = {"key": key, "id": rec["id"], "qid": q["qid"], "mode": rec.get("mode"), "P": P,
                            "n": len(q["ids"]), "ids": q["ids"], "markers": q["markers"], "K": int(q["K"]),
                            "type": q["type"], "calibrate": bool(q["calibrate"]), "T": q.get("T"),
                            "temperature_key": q.get("temperature_key"),
                            "oracle_probs": [float(v) for v in q["probs"]]}
    missing = [k for k in keys if k not in out]
    assert not missing, missing
    return out, {"version": oracle.get("version"), "written": oracle.get("written")}


# ---------------------------------------------------------------- load gate

def load_avg():
    out = subprocess.run(["sysctl", "-n", "vm.loadavg"], capture_output=True, text=True).stdout
    return [float(x) for x in out.replace("{", " ").replace("}", " ").split()[:3]]


def top_head():
    return subprocess.run("top -l 1 | head -5", shell=True, capture_output=True, text=True).stdout.splitlines()


def top_cpu():
    out = subprocess.run(["top", "-l", "2", "-s", "1", "-o", "cpu", "-n", "12", "-stats", "pid,ppid,cpu,mem,command"],
                         capture_output=True, text=True).stdout.splitlines()
    idx = [i for i, ln in enumerate(out) if ln.strip().startswith("PID")]
    return out[idx[-1]:idx[-1] + 13] if idx else out[-13:]


def contention():
    """Second sample of `top -l 2 -s 1`: CPU idle % and the processes outside this process tree above PEER_CPU_MAX."""
    out = subprocess.run(["top", "-l", "2", "-s", "1", "-o", "cpu", "-n", "12", "-stats", "pid,ppid,cpu,command"],
                         capture_output=True, text=True).stdout.splitlines()
    cpu_lines = [ln for ln in out if ln.startswith("CPU usage")]
    idle = None
    if cpu_lines:
        for part in cpu_lines[-1].split(","):
            if "idle" in part:
                idle = float(part.strip().split("%")[0])
    heads = [i for i, ln in enumerate(out) if ln.strip().startswith("PID")]
    procs = []
    for ln in (out[heads[-1] + 1:] if heads else []):
        f = ln.split(None, 3)
        if len(f) < 4:
            continue
        try:
            procs.append({"pid": int(f[0]), "ppid": int(f[1]), "cpu": float(f[2]), "command": f[3].strip()})
        except ValueError:
            continue
    mine = {os.getpid(), os.getppid()}
    peers = [p for p in procs if p["cpu"] > PEER_CPU_MAX and p["pid"] not in mine and p["ppid"] not in mine]
    for p in peers:
        try:
            p["full_command"] = subprocess.run(["ps", "-p", str(p["pid"]), "-o", "command="], capture_output=True,
                                               text=True).stdout.strip()[:200]
        except Exception:
            pass
    return {"idle_pct": idle, "peers_above_120": peers, "top": procs[:8],
            "ok": idle is not None and idle >= IDLE_MIN and not peers}


def lock_line():
    try:
        return Path(LOCK).read_text().strip()
    except OSError as e:
        return f"unreadable: {e}"


def load_gate():
    t0 = time.time()
    polls = []
    while True:
        la = load_avg()
        c = contention()
        polls.append({"t": round(time.time() - t0, 1), "load": la, "idle_pct": c["idle_pct"],
                      "peers_above_120": c["peers_above_120"]})
        if c["ok"]:
            break
        if time.time() - t0 > LOAD_WAIT_S:
            return {"ok": False, "rule": "contention", "waited_s": round(time.time() - t0, 1), "polls": polls,
                    "top_head": top_head(), "contention": c, "lock": lock_line(), "at": now()}
        time.sleep(LOAD_POLL_S)
    return {"ok": True, "rule": f"no process outside this tree > {PEER_CPU_MAX:g} % CPU and idle >= {IDLE_MIN:g} % "
                                f"(second top sample); load recorded, launch threshold {LOAD_MAX:g} not met today",
            "waited_s": round(time.time() - t0, 1), "polls": polls[-6:], "top_head": top_head(), "contention": c,
            "lock": lock_line(), "at": now(), "load_at_start": la, "load_above_launch_threshold": la[0] > LOAD_MAX}


# ---------------------------------------------------------------- child

def child_main(a):
    import litert_gate as LG
    import litert_run as R

    doc = {"pid": os.getpid(), "started": now(), "backend": a.backend, "L": a.L, "form": a.form,
           "file": None, "status": "FAIL"}
    out_json = Path(a.out)
    try:
        path = Path(a.file) if a.file else file_for(a.L, a.form)
        path = path if path.is_absolute() else K / path
        doc["file"] = str(path.relative_to(K))
        doc["file_bytes"] = path.stat().st_size
        wl = WORKLOADS[a.L] if not a.workloads else json.loads(a.workloads)
        keys = sorted({k for _, _, ks in wl for k in ks})
        rows, ometa = rows_by_key(set(keys))
        doc["oracle"] = ometa
        inputs = {k: {n: np.ascontiguousarray(v) for n, v in LG.row_inputs(rows[k], a.L).items()} for k in keys}
        backend = "cpu" if a.backend == "cpu" else "gpu"
        precision = "default" if a.backend == "gpu_default" else "fp32"
        doc["memory_before_compile"] = R.memory()
        t0 = time.perf_counter()
        cm, desc = R.open_compiled(path, backend, precision, threads=8, share=a.share)
        doc["compile_seconds"] = round(time.perf_counter() - t0, 3)
        doc["options"] = desc
        doc["memory_after_compile"] = R.memory()
        try:
            doc["is_fully_accelerated"] = bool(cm.is_fully_accelerated())
        except Exception as e:  # informational
            doc["is_fully_accelerated"] = f"unavailable: {type(e).__name__}: {e}"
        sigs = list(cm.get_signature_list())
        doc["signatures"] = sigs
        sig = a.signature or (sigs[0] if len(sigs) == 1 else f"decide_{a.L}")
        doc["signature"] = sig
        run = R.Runner(cm, sig)
        assert run.L == a.L, (run.L, a.L)
        doc["workloads"] = {}
        for name, kind, ks in wl:
            w = {"kind": kind, "rows": [{"key": k, "P": rows[k]["P"], "n": rows[k]["n"],
                                         "positions": rows[k]["P"] + rows[k]["n"]} for k in ks],
                 "what": WORKLOAD_NAMES.get(name, name)}
            warm = []
            for i in range(WARMUP):
                k = ks[i % len(ks)]
                s, wall, tot, rn = run.timed(inputs[k])
                warm.append([round(wall, 3), round(tot, 3), round(rn, 3)])
            calls, finite, first = [], True, {}
            req_tot, req_run = [], []
            for _ in range(REPS):
                tt = tr = 0.0
                for k in ks:
                    s, wall, tot, rn = run.timed(inputs[k])
                    calls.append([round(wall, 3), round(tot, 3), round(rn, 3)])
                    finite &= bool(np.isfinite(s[:rows[k]["P"] + rows[k]["n"]]).all())
                    tt, tr = tt + tot, tr + rn
                    if k not in first:
                        first[k] = s[:rows[k]["P"] + rows[k]["n"]].copy()
                req_tot.append(tt)
                req_run.append(tr)
            check = {}
            for k in ks:
                p, _ = LG.readout(first[k], rows[k])
                check[k] = {"max_abs_dp_vs_oracle": max(abs(x - y) for x, y in zip(p, rows[k]["oracle_probs"])),
                            "argmax_equal": int(np.argmax(p)) == int(np.argmax(rows[k]["oracle_probs"]))}
            w.update(warmup_calls=warm, calls=calls, call_columns=["wall_clock_ms_at_start", "ms_write_run_read",
                                                                   "ms_run_only"],
                     ms_write_run_read=stats([c[1] for c in calls]), ms_run_only=stats([c[2] for c in calls]),
                     finite=finite, readout_check=check)
            if kind == "request":
                w["request_calls"] = len(ks)
                w["request_ms_write_run_read"] = stats(req_tot)
                w["request_ms_run_only"] = stats(req_run)
            doc["workloads"][name] = w
            print(f"{a.backend} L{a.L} {a.form} {name}: call median {w['ms_write_run_read']['median']} ms"
                  + (f", request {w['request_ms_write_run_read']['median']} ms" if kind == "request" else ""),
                  flush=True)
        if a.parity_rows:
            # multisig: the first N L256 gate rows through this signature (scores at the real positions kept)
            g_rows = LG.load_rows(a.L, LG.oracle_doc())[0][:a.parity_rows]
            store = {}
            for r in g_rows:
                x = {n: np.ascontiguousarray(v) for n, v in LG.row_inputs(r, a.L).items()}
                s = run(x)
                store[r["key"]] = s[:r["P"] + r["n"]].copy()
            np.savez(TMP / f"{out_json.stem}.scores.npz", **{k.replace("/", "__"): v for k, v in store.items()})
            doc["parity_rows"] = [r["key"] for r in g_rows]
        doc["memory_end"] = R.memory()
        run.close()
        if hasattr(cm, "close"):
            cm.close()
        doc["status"] = "OK"
    except BaseException as e:  # recorded
        import traceback

        doc["error"] = f"{type(e).__name__}: {e}"
        doc["traceback"] = traceback.format_exc()[-3000:]
    doc["finished"] = now()
    out_json.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    return 0 if doc["status"] == "OK" else 1


def spawn(args, tag):
    """Run one child; -> (child doc, rusage dict)."""
    TMP.mkdir(parents=True, exist_ok=True)
    out = TMP / f"{tag}.json"
    if out.exists():
        out.unlink()
    log = K / f"logs/r4_timing_{tag}.child.log"
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", "--out", str(out)] + args
    t0 = time.time()
    with open(log, "w") as fo:
        child = subprocess.Popen(cmd, stdout=fo, stderr=subprocess.STDOUT, cwd=str(K))
        _, status, ru = os.wait4(child.pid, 0)
    rc = os.waitstatus_to_exitcode(status)
    doc = json.loads(out.read_text()) if out.exists() else {"status": "FAIL", "error": "child wrote no record"}
    doc["child_returncode"] = rc
    doc["child_ru_maxrss_bytes"] = int(ru.ru_maxrss)
    doc["child_seconds_wall"] = round(time.time() - t0, 1)
    doc["child_log"] = str(log.relative_to(K))
    if rc != 0:
        doc["child_log_tail"] = log.read_text(errors="replace").splitlines()[-30:]
    return doc


SPREAD_MAX = 0.20


def spread_check(child):
    """(median - min) / min of every timed series of a set (per call; per request for a request workload)."""
    out = {}
    for name, w in (child.get("workloads") or {}).items():
        for key in ("ms_write_run_read", "request_ms_write_run_read"):
            st = w.get(key)
            if st:
                out[f"{name}.{key}"] = round((st["median"] - st["min"]) / st["min"], 4) if st["min"] else None
    over = {k: v for k, v in out.items() if v is not None and v > SPREAD_MAX}
    return {"median_over_min_minus_1": out, "over": over, "rule": f"retake once when > {SPREAD_MAX:g} (CPU sets)"}


def window_path(backend, L):
    base = K / f"results/timing_mac_r4_{backend}_L{L}.json"
    if not base.exists():
        return base
    n = 2
    while (K / f"results/timing_mac_r4_{backend}_L{L}_take{n}.json").exists():
        n += 1
    return K / f"results/timing_mac_r4_{backend}_L{L}_take{n}.json"


def window_main(a):
    forms = a.forms.split(",")
    out = window_path(a.backend, a.L)
    doc = {"what": f"Mac timing, backend {a.backend}, L{a.L}, forms {forms} in one quiet window",
           "protocol": f"{WARMUP} warm-up calls, then {REPS} timed calls (a request workload: {REPS} requests of its "
                       "rows back to back); per call [wall clock ms at start, ms write + run + read back, ms run "
                       "only]; one fresh process per model",
           "backend": a.backend, "L": a.L, "forms": forms, "window_lock_line_at_start": lock_line(),
           "started": now(), "sets": {},
           "gate_rule": f"before each set: no process outside this process tree above {PEER_CPU_MAX:g} % CPU and CPU "
                        f"idle >= {IDLE_MIN:g} % on the second `top -l 2 -s 1` sample (polls every {LOAD_POLL_S} s, "
                        f"up to {LOAD_WAIT_S} s); the 1-minute load and `top -l 1 | head -5` are recorded (the "
                        f"launch's load <= {LOAD_MAX:g} cannot be met today: 20-40 with 68-73 % idle)"}
    rc = 0
    for form in forms:
        gate = load_gate()
        st = {"load_gate": gate}
        if not gate["ok"]:
            st["status"] = "discarded: contention for the whole wait"
            doc["sets"][form] = st
            rc = 3
            break
        child = spawn(["--backend", a.backend, "--L", str(a.L), "--form", form], f"{a.backend}_L{a.L}_{form}")
        st["load_after"] = {"load": load_avg(), "at": now()}
        spread = spread_check(child)
        if a.backend == "cpu" and child.get("status") == "OK" and spread["over"]:
            # 12:2x: a CPU set whose median is more than 20 % above its min is taken once more in the same
            # window; if the retake is over too, both stay and both are reported
            st["first_attempt"] = {"result": child, "spread": spread}
            st["retake_gate"] = load_gate()
            child = spawn(["--backend", a.backend, "--L", str(a.L), "--form", form],
                          f"{a.backend}_L{a.L}_{form}_retake")
            st["load_after_retake"] = {"load": load_avg(), "at": now()}
            spread = spread_check(child)
            st["retake_spread_still_over"] = spread["over"]
        st["spread"] = spread
        st["result"] = child
        st["status"] = "measured" if child.get("status") == "OK" else f"failed: {child.get('error')}"
        doc["sets"][form] = st
    doc["finished"] = now()
    doc["window_lock_line_at_end"] = lock_line()
    out.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    summary = {f: {"status": s["status"], "load": s["load_gate"].get("load_at_start"),
                   "compile_s": (s.get("result") or {}).get("compile_seconds"),
                   "ms": {n: w["ms_write_run_read"]["median"] for n, w in
                          ((s.get("result") or {}).get("workloads") or {}).items()}}
               for f, s in doc["sets"].items()}
    print(json.dumps({"out": str(out.relative_to(K)), "sets": summary}, indent=1))
    return rc


# ---------------------------------------------------------------- multisig

def multisig_main(a):
    import litert_gate as LG

    out = K / "results/multisig_fp16.json"
    assert not out.exists(), f"refusing to overwrite {out}"
    multi = K / "out/d1omni_decide_multi6_fp16.tflite"
    single = K / "out/d1omni_decide_L256_fp16.tflite"
    wl = json.dumps([("g", "single", ["aud_reservation_01/topic"]), ("a_at_L256", "single", ["card_text/refund"])])
    configs = [("gpu_single", single, "gpu_fp32", False), ("gpu_multi_share_off", multi, "gpu_fp32", False),
               ("gpu_multi_share_on", multi, "gpu_fp32", True), ("cpu_single", single, "cpu", False),
               ("cpu_multi", multi, "cpu", False)]
    doc = {"what": "6-signature fp16 file vs the single-signature fp16 L256 file on the Mac (decide_256)",
           "files": {"multi": {"file": str(multi.relative_to(K)), "bytes": multi.stat().st_size},
                     "single_L256": {"file": str(single.relative_to(K)), "bytes": single.stat().st_size}},
           "bytes_ratio_multi_over_single": multi.stat().st_size / single.stat().st_size,
           "window_lock_line_at_start": lock_line(), "started": now(), "configs": {}}
    rc = 0
    for name, path, backend, share in configs:
        gate = load_gate()
        st = {"load_gate": gate}
        if not gate["ok"]:
            st["status"] = "discarded: contention for the whole wait"
            doc["configs"][name] = st
            rc = 3
            break
        args = ["--backend", backend, "--L", "256", "--form", "fp16", "--file", str(path.relative_to(K)),
                "--signature", "decide_256", "--workloads", wl, "--parity-rows", str(a.parity_rows)]
        if share:
            args.append("--share")
        child = spawn(args, f"multisig_{name}")
        st.update(result=child, status="measured" if child.get("status") == "OK" else f"failed: {child.get('error')}",
                  load_after={"load": load_avg(), "at": now()})
        doc["configs"][name] = st
    # bit-equality and oracle |dp| of the parity rows
    oracle = LG.oracle_doc()
    rows = {r["key"]: r for r in LG.load_rows(256, oracle)[0]}
    ref = LG.oracle_ref(oracle)
    scores = {}
    for name in doc["configs"]:
        p = TMP / f"multisig_{name}.scores.npz"
        if p.exists():
            with np.load(p) as z:
                scores[name] = {k.replace("__", "/"): np.asarray(z[k]) for k in z.files}
    comp = {}
    for be in ("gpu", "cpu"):
        base = scores.get(f"{be}_single")
        for name in [n for n in scores if n.startswith(be) and n != f"{be}_single"]:
            if base is None:
                continue
            keys = [k for k in base if k in scores[name]]
            bit = sum(bool(np.array_equal(base[k], scores[name][k])) for k in keys)
            dmax = max(float(np.abs(base[k].astype(np.float64) - scores[name][k]).max()) for k in keys)
            comp[f"{name}_vs_{be}_single"] = {"rows": len(keys), "bit_equal": bit, "max_abs_dscores": dmax}
    vs_oracle = {}
    for name, sc in scores.items():
        dps = []
        argmax_ok = 0
        for k, s in sc.items():
            p, _ = LG.readout(s, rows[k])
            dps.append(max(abs(x - y) for x, y in zip(p, ref[k][0])))
            argmax_ok += int(np.argmax(p)) == int(np.argmax(ref[k][0]))
        vs_oracle[name] = {"rows": len(sc), "max_abs_dp": max(dps) if dps else None,
                           "mean_of_row_max_abs_dp": float(np.mean(dps)) if dps else None, "argmax_equal": argmax_ok}
    doc.update(parity_rows=a.parity_rows, multi_vs_single=comp, vs_oracle=vs_oracle, finished=now(),
               window_lock_line_at_end=lock_line())
    out.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(json.dumps({"configs": {n: {"status": c["status"], "compile_s": (c.get("result") or {}).get("compile_seconds"),
                                      "phys_after_compile": ((c.get("result") or {}).get("memory_after_compile") or {})
                                      .get("phys_footprint"),
                                      "g_ms": (((c.get("result") or {}).get("workloads") or {}).get("g") or {})
                                      .get("ms_write_run_read")}
                                  for n, c in doc["configs"].items()},
                      "multi_vs_single": comp, "vs_oracle": vs_oracle}, indent=1))
    return rc


def share_probe_main(a):
    """--share-probe: GpuOptions(constant_tensor_sharing=True) on Metal fp32 precision, one fresh child per file: the
    6-signature fp16 file again (the --multisig run aborted), the single-signature fp16 / fp32 L256 files, and the
    single-signature L256 file with the int8 table (fp16fc_i8emb; a lookup that reads an int8 table) ->
    results/multisig_share_probe.json (compile or abort, the fatal line, the child's peak RSS)."""
    out = K / "results/multisig_share_probe.json"
    assert not out.exists(), f"refusing to overwrite {out}"
    wl = json.dumps([("g", "single", ["aud_reservation_01/topic"])])
    files = [("multi6_fp16", "out/d1omni_decide_multi6_fp16.tflite"), ("L256_fp16", "out/d1omni_decide_L256_fp16.tflite"),
             ("L256_fp32", "out/d1omni_decide_L256_fp32.tflite"),
             ("L256_fp16fc_i8emb", "out/d1omni_decide_L256_fp16fc_i8emb.tflite")]
    doc = {"what": "Metal fp32 precision with GpuOptions(constant_tensor_sharing=True): which files compile",
           "window_lock_line_at_start": lock_line(), "started": now(), "probes": {}}
    for name, f in files:
        gate = load_gate()
        child = spawn(["--backend", "gpu_fp32", "--L", "256", "--form", "fp16", "--file", f, "--signature", "decide_256",
                       "--workloads", wl, "--share"], f"share_probe_{name}")
        log = K / child["child_log"]
        fatal = [ln for ln in log.read_text(errors="replace").splitlines() if ln.startswith(("F0000", "E0000"))
                 or "Unsupported" in ln or "Failed" in ln][:10]
        g = (child.get("workloads") or {}).get("g") or {}
        doc["probes"][name] = {"file": f, "status": child.get("status"), "returncode": child.get("child_returncode"),
                               "fatal_lines": fatal, "compile_seconds": child.get("compile_seconds"),
                               "is_fully_accelerated": child.get("is_fully_accelerated"),
                               "memory_after_compile": child.get("memory_after_compile"),
                               "child_ru_maxrss_bytes": child.get("child_ru_maxrss_bytes"),
                               "g_ms_write_run_read": g.get("ms_write_run_read"),
                               "readout_check": g.get("readout_check"), "child_log": child.get("child_log"),
                               "gate": {k: gate.get(k) for k in ("ok", "load_at_start", "waited_s")}}
        print(name, child.get("status"), child.get("child_returncode"), fatal[:2], flush=True)
    doc["finished"] = now()
    out.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    return 0


# ---------------------------------------------------------------- aggregate

def aggregate_main(a):
    """results/timing_mac_r4_<backend>_<form>.json from the window files (the last measured take per L)."""
    files = sorted((K / "results").glob("timing_mac_r4_*_L*.json"))
    best = {}
    for p in files:
        d = json.loads(p.read_text())
        for form, st in d.get("sets", {}).items():
            if st.get("status") != "measured":
                continue
            key = (d["backend"], form, d["L"])
            best[key] = (p, d, st)          # files are sorted: a later take replaces an earlier one
    by_bf = {}
    for (backend, form, L), (p, d, st) in sorted(best.items()):
        r = st["result"]
        e = by_bf.setdefault((backend, form), {"what": f"Mac timing, backend {backend}, form {form}",
                                               "written": now(), "by_L": {}})
        gate = st["load_gate"]
        cont = gate.get("contention") or {}
        e["by_L"][str(L)] = {
            "window_file": str(p.relative_to(K)), "window_lock_line": d.get("window_lock_line_at_start"),
            "set_started": gate.get("at"), "load_at_start": gate.get("load_at_start"),
            "idle_pct_at_start": cont.get("idle_pct"), "gate_rule": gate.get("rule"),
            "top3_at_start": [f"{x['command']} {x['cpu']:g} %" for x in (cont.get("top") or [])[:3]],
            "spread": (st.get("spread") or {}).get("median_over_min_minus_1"),
            "retake": ("first_attempt" in st), "retake_spread_still_over": st.get("retake_spread_still_over"),
            "first_attempt_medians": ({n: w["ms_write_run_read"]["median"] for n, w in
                                       st["first_attempt"]["result"].get("workloads", {}).items()}
                                      if "first_attempt" in st else None),
            "top_head": gate.get("top_head"), "file": r.get("file"), "file_bytes": r.get("file_bytes"),
            "compile_seconds": r.get("compile_seconds"), "is_fully_accelerated": r.get("is_fully_accelerated"),
            "memory_after_compile": r.get("memory_after_compile"), "memory_end": r.get("memory_end"),
            "child_ru_maxrss_bytes": r.get("child_ru_maxrss_bytes"),
            "workloads": {n: {k: w.get(k) for k in ("kind", "what", "rows", "ms_write_run_read", "ms_run_only",
                                                     "request_ms_write_run_read", "request_ms_run_only",
                                                     "finite", "readout_check")}
                          for n, w in r.get("workloads", {}).items()}}
    written = []
    for (backend, form), e in by_bf.items():
        out = K / f"results/timing_mac_r4_{backend}_{form}.json"
        out.write_text(json.dumps(e, indent=1, default=str) + "\n")
        written.append(str(out.relative_to(K)))
    print(json.dumps(written, indent=1))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("cpu", "gpu_fp32", "gpu_default"))
    ap.add_argument("--L", type=int)
    ap.add_argument("--forms", default="fp32,fp16")
    ap.add_argument("--form")
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--file")
    ap.add_argument("--signature")
    ap.add_argument("--workloads")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--parity-rows", type=int, default=0)
    ap.add_argument("--multisig", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--share-probe", action="store_true")
    a = ap.parse_args()
    if a.child:
        return child_main(a)
    if a.share_probe:
        return share_probe_main(a)
    if a.aggregate:
        return aggregate_main(a)
    if a.multisig:
        a.parity_rows = a.parity_rows or 50
        return multisig_main(a)
    return window_main(a)


if __name__ == "__main__":
    sys.exit(main())
