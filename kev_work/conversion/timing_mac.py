"""Mac timing of the shipped row-prefill graphs (V2: float16 FC weights + int8 embedding table): Metal GPU with float32
activations and CPU with 8 threads, through the CompiledModel API of ai-edge-litert.

    python timing_mac.py --job 4b:card --out-4b results/timing_mac_4b.json [--lock <file> --lock-tag <text>]
    python timing_mac.py --job 0.8b:card --job 0.8b:L2048 --out-0.8b ...       (one --out-<model> per model; --job may repeat)

Jobs (the token ids are the reference rows of oracle/oracle_<model>.json; padding and `valid` as in the parity gate):
  card   fiveq: own_fiveq_09's 5 questions = 5 rows (132 / 142 / 132 / 128 / 128 tokens) in the 512-token file; one
         "request" = the 5 calls back to back; 5 warm-up calls, then 20 requests (100 calls): the 5-call total and the
         per-call time (median / min / max). T300: the first 300 tokens of own_long_log_10/first_failure's row, 512-token
         file. T1000: its first 1,000 tokens, 1,024-token file. T300 / T1000: 5 warm-up calls, then 20 calls.
  L2048  the 2,048-token file: T1000 (the same 1,000 tokens) and the whole own_long_log_10/first_failure row (1,805
         tokens); 5 warm-up calls, then 20 calls each.
Every call is timed as write + run + read-back of the full hidden output, and as run only. The graph computes all L
positions, so the time depends on L, not on the token count. Compile seconds are recorded per file and accelerator
(the first compile of that file in this process; the system's Metal shader cache may be warm from earlier runs).
T300 / T1000 are cut from a real row (their correctness is not checked); every call's position-0 hidden state must be
finite.

GPU lock (optional): --lock names a file that other GPU jobs on this machine also respect. The run takes an exclusive
flock on it and writes one line "<tag> pid <pid> since <time>" while it holds it; afterwards the content goes back to
empty and the flock is released. If another process holds the flock or the file has other content, the run waits up to
10 minutes (5 s polls) and then writes "not measured (lock busy)" without running anything. A TERM signal ends the
run through the same cleanup (the lock content goes back to empty).
Outputs are never overwritten."""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

K = Path(__file__).resolve().parents[1]
PAD_ID = 248044
WARMUP, REPS = 5, 20
MODELS = {"0.8b": {"prefix": "kev08b", "oracle": "oracle/oracle_0.8b.json"},
          "4b": {"prefix": "kev4b", "oracle": "oracle/oracle_4b.json"}}
VARIANT = "v2_fp16fc_i8emb"


def stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def top_lines():
    out = subprocess.run(["top", "-l", "1"], capture_output=True, text=True).stdout.splitlines()
    return [ln for ln in out[:10] if ln.startswith(("Load Avg", "CPU usage", "PhysMem"))]


def swap_used():
    return subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.strip()


def memory():
    """ru_maxrss and phys_footprint / lifetime max (proc_pid_rusage v4) of this process, bytes."""
    import ctypes
    import resource
    fields = ("ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups", "ri_pageins", "ri_wired_size",
              "ri_resident_size", "ri_phys_footprint", "ri_proc_start_abstime", "ri_proc_exit_abstime",
              "ri_child_user_time", "ri_child_system_time", "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups",
              "ri_child_pageins", "ri_child_elapsed_abstime", "ri_diskio_bytesread", "ri_diskio_byteswritten",
              "ri_cpu_time_qos_default", "ri_cpu_time_qos_maintenance", "ri_cpu_time_qos_background",
              "ri_cpu_time_qos_utility", "ri_cpu_time_qos_legacy", "ri_cpu_time_qos_user_initiated",
              "ri_cpu_time_qos_user_interactive", "ri_billed_system_time", "ri_serviced_system_time", "ri_logical_writes",
              "ri_lifetime_max_phys_footprint", "ri_instructions", "ri_cycles", "ri_billed_energy", "ri_serviced_energy",
              "ri_interval_max_phys_footprint", "ri_runnable_time")

    class Info(ctypes.Structure):
        _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in fields]

    doc = {"ru_maxrss": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}
    info = Info()
    if ctypes.CDLL("/usr/lib/libproc.dylib").proc_pid_rusage(os.getpid(), 4, ctypes.byref(info)) == 0:
        doc.update(phys_footprint=int(info.ri_phys_footprint),
                   lifetime_max_phys_footprint=int(info.ri_lifetime_max_phys_footprint))
    return doc


def acquire(lock, tag):
    fh = open(lock, "r+")
    t0, seen = time.time(), []
    while True:
        content = Path(lock).read_text().strip()
        if content:
            if not seen or seen[-1] != content:
                seen.append(content)
        else:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if Path(lock).read_text().strip():   # someone wrote between the read and the flock
                    fcntl.flock(fh, fcntl.LOCK_UN)
                else:
                    line = f"{tag} pid {os.getpid()} since {stamp()}"
                    fh.seek(0)
                    fh.truncate()
                    fh.write(line + "\n")
                    fh.flush()
                    return fh, {"acquired_at": stamp(), "waited_s": round(time.time() - t0, 1), "line": line,
                                "seen_other_content": seen[-5:]}
            except BlockingIOError:
                seen.append("<flock held by another process>")
        if time.time() - t0 > 600:
            fh.close()
            return None, {"acquired_at": None, "waited_s": round(time.time() - t0, 1), "seen_other_content": seen[-5:]}
        time.sleep(5)


def release(fh, lock):
    fh.seek(0)
    fh.truncate()
    fh.flush()
    fcntl.flock(fh, fcntl.LOCK_UN)
    fh.close()
    return {"released_at": stamp(), "content_after": Path(lock).read_text(), "bytes_after": Path(lock).stat().st_size}


def inputs(row_ids, L, T=None):
    ids_src = list(row_ids)[: T or len(row_ids)]
    n = len(ids_src)
    assert n <= L, (n, L)
    ids = np.full((1, L), PAD_ID, dtype=np.int32)
    ids[0, :n] = np.asarray(ids_src, dtype=np.int32)
    valid = np.zeros((1, L), dtype=np.float32)
    valid[0, :n] = 1.0
    return ids, valid, n


def stats(xs):
    xs = np.asarray(xs, dtype=np.float64)
    return {"median": round(float(np.median(xs)), 2), "min": round(float(xs.min()), 2), "max": round(float(xs.max()), 2),
            "n": int(xs.size)}


class Graph:
    def __init__(self, path, accel, threads):
        from ai_edge_litert.compiled_model import CompiledModel, CpuOptions, GpuOptions, HardwareAccelerator, Options
        if accel == "gpu_f32":
            opts = Options(hardware_accelerators=HardwareAccelerator.GPU, gpu_options=GpuOptions(enforce_f32=True))
        else:
            opts = Options(hardware_accelerators=HardwareAccelerator.CPU, cpu_options=CpuOptions(num_threads=threads))
        t0 = time.perf_counter()
        self.model = CompiledModel.from_file(str(path), options=opts)
        self.compile_s = time.perf_counter() - t0
        self.fully = bool(self.model.is_fully_accelerated())
        self.key = next(iter(self.model.get_signature_list()))
        out = self.model.get_output_tensor_details(self.key)["hidden"]
        self.L, self.d = int(list(out["shape"])[1]), int(list(out["shape"])[2])
        self.ins = {n: self.model.create_input_buffer_by_name(self.key, n) for n in self.model.get_input_tensor_details(self.key)}
        self.outs = {"hidden": self.model.create_output_buffer_by_name(self.key, "hidden")}

    def call(self, ids, valid):
        t = time.perf_counter()
        self.ins["ids"].write(ids)
        self.ins["valid"].write(valid)
        tr = time.perf_counter()
        self.model.run_by_name(self.key, self.ins, self.outs)
        run_ms = (time.perf_counter() - tr) * 1000
        h = self.outs["hidden"].read(self.L * self.d, np.float32)
        total_ms = (time.perf_counter() - t) * 1000
        return total_ms, run_ms, bool(np.isfinite(np.asarray(h[: self.d])).all())

    def close(self):
        for b in list(self.ins.values()) + list(self.outs.values()):
            try:
                b.destroy()
            except Exception:
                pass
        self.model.close()


def single(graph, ids, valid):
    for _ in range(WARMUP):
        graph.call(ids, valid)
    tot, run, finite = [], [], True
    for _ in range(REPS):
        t_ms, r_ms, ok = graph.call(ids, valid)
        tot.append(t_ms)
        run.append(r_ms)
        finite &= ok
    return {"ms_write_run_read": stats(tot), "ms_run_only": stats(run), "finite": finite}


def request(graph, rows):
    for i in range(WARMUP):
        graph.call(*rows[i % len(rows)][:2])
    req_tot, req_run, per_tot, per_run, finite = [], [], [], [], True
    for _ in range(REPS):
        tt = tr = 0.0
        for ids, valid, _n in rows:
            t_ms, r_ms, ok = graph.call(ids, valid)
            tt, tr = tt + t_ms, tr + r_ms
            per_tot.append(t_ms)
            per_run.append(r_ms)
            finite &= ok
        req_tot.append(tt)
        req_run.append(tr)
    return {"request_5calls_ms_write_run_read": stats(req_tot), "request_5calls_ms_run_only": stats(req_run),
            "per_call_ms_write_run_read": stats(per_tot), "per_call_ms_run_only": stats(per_run), "finite": finite}


def run_job(model, plan, threads):
    """-> (doc parts, results per accelerator) for one model and one job."""
    oracle = json.loads((K / MODELS[model]["oracle"]).read_text())
    fiveq = [q for q in oracle["questions"] if q["id"] == "own_fiveq_09"]
    longq = next(q for q in oracle["questions"] if q["id"] == "own_long_log_10" and q["qid"] == "first_failure")
    assert len(fiveq) == 5 and longq["row_len"] == 1805
    files = {L: K / f"exports/{MODELS[model]['prefix']}_rowprefill_L{L}_{VARIANT}.tflite"
             for L in ((512, 1024) if plan == "card" else (2048,))}
    for p in files.values():
        assert p.exists(), p
    rows = ({"fiveq": {"id": "own_fiveq_09", "row_lens": [q["row_len"] for q in fiveq], "request_input_tokens": 266, "file_L": 512},
             "T300": {"from": "own_long_log_10/first_failure", "tokens": 300, "file_L": 512, "synthetic": True},
             "T1000": {"from": "own_long_log_10/first_failure", "tokens": 1000, "file_L": 1024, "synthetic": True}}
            if plan == "card" else
            {"T1000": {"from": "own_long_log_10/first_failure", "tokens": 1000, "file_L": 2048, "synthetic": True},
             "long_1805": {"from": "own_long_log_10/first_failure", "tokens": 1805, "file_L": 2048, "synthetic": False}})
    results = {}
    for accel in ("gpu_f32", "cpu8"):
        res = {"compile_s": {}, "fully_accelerated": {}, "memory_after_compile": {}, "hidden_size": {}}
        for L, path in files.items():
            g = Graph(path, accel, threads)
            res["compile_s"][f"L{L}"], res["fully_accelerated"][f"L{L}"] = round(g.compile_s, 2), g.fully
            res["memory_after_compile"][f"L{L}"], res["hidden_size"][f"L{L}"] = memory(), g.d
            assert g.L == L, (g.L, L)
            if plan == "card" and L == 512:
                res["fiveq"] = request(g, [inputs(q["row_ids"], 512) for q in fiveq])
                ids, valid, n = inputs(longq["row_ids"], 512, 300)
                res["T300"] = {"tokens": n, **single(g, ids, valid)}
            elif plan == "card":
                ids, valid, n = inputs(longq["row_ids"], 1024, 1000)
                res["T1000"] = {"tokens": n, **single(g, ids, valid)}
            else:
                ids, valid, n = inputs(longq["row_ids"], 2048, 1000)
                res["T1000"] = {"tokens": n, **single(g, ids, valid)}
                ids, valid, n = inputs(longq["row_ids"], 2048)
                res["long_1805"] = {"tokens": n, **single(g, ids, valid)}
            g.close()
            print(model, plan, accel, f"L{L}", json.dumps({k: v for k, v in res.items() if k not in ("memory_after_compile",)}),
                  flush=True)
        res["top_after"] = top_lines()
        results[accel] = res
    files_doc = {str(L): {"file": str(p.relative_to(K)), "bytes": p.stat().st_size} for L, p in files.items()}
    return files_doc, rows, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", action="append", required=True, help="<model>:<card|L2048>, e.g. 4b:card (may repeat)")
    ap.add_argument("--out-4b", default="")
    ap.add_argument("--out-0.8b", dest="out_08b", default="")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--lock", default="", help="lock file shared with other GPU jobs (optional)")
    ap.add_argument("--lock-tag", default="kev timing")
    a = ap.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))   # a TERM still empties the lock (finally below)
    jobs = [tuple(j.split(":")) for j in a.job]
    assert all(m in MODELS and p in ("card", "L2048") for m, p in jobs), jobs
    outs = {"4b": a.out_4b, "0.8b": a.out_08b}
    for m in {m for m, _ in jobs}:
        assert outs[m], f"--out-{m} is required"
        assert not (K / outs[m]).exists(), f"refusing to overwrite {outs[m]}"
    docs = {m: {"what": "Mac timing of the V2 row-prefill graphs (GPU float32 activations, CPU threads)", "model": m,
                "variant": VARIANT, "protocol": f"{WARMUP} warm-up calls, then {REPS} timed calls (fiveq: {REPS} requests "
                "of 5 calls); ms = write + run + read-back of the full hidden output, and run only",
                "threads_cpu": a.threads, "jobs": {}, "top_before_lock": top_lines(), "swap_before_lock": swap_used(),
                "started_at": stamp()} for m in {m for m, _ in jobs}}
    fh, lock = (None, {"path": None, "note": "no lock file given"})
    if a.lock:
        fh, lock = acquire(a.lock, a.lock_tag)
        lock = {"path": a.lock, **lock}
        if fh is None:
            for m, doc in docs.items():
                doc.update(status="not measured (lock busy)", lock=lock)
                (K / outs[m]).write_text(json.dumps(doc, indent=1) + "\n")
            print(json.dumps(lock, indent=1))
            return
    t0 = time.time()
    try:
        for doc in docs.values():
            doc["top_at_lock"], doc["swap_at_lock"] = top_lines(), swap_used()
        for m, plan in jobs:
            files_doc, rows, results = run_job(m, plan, a.threads)
            docs[m]["jobs"][plan] = {"files": files_doc, "rows": rows, "results": results}
        for doc in docs.values():
            doc["status"] = "measured"
    finally:
        if fh is not None:
            lock.update(release(fh, a.lock))
        for m, doc in docs.items():
            doc.update(lock=lock, top_after_release=top_lines(), swap_after=swap_used(), finished_at=stamp(),
                       seconds_wall=round(time.time() - t0, 1))
            doc.setdefault("status", "failed (see the log)")
            (K / outs[m]).write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({m: {k: v for k, v in d.items() if k != "jobs"} for m, d in docs.items()}, indent=1))


if __name__ == "__main__":
    main()
