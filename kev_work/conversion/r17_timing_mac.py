"""Mac timing of two row-graph forms side by side at L64 / L128 / L256: Metal GPU with float32 activations (the Python
GpuOptions has enforce_f32 and the fp16 default only, so the Mac measures fp32) and optionally the CPU with 8 threads.
r14_timing_mac.py reduced to the three buckets: the same rows, the same protocol (timing_mac.py's Graph / single /
request imported unchanged: 5 warm-up calls, then 20 timed calls; fiveq = 20 requests of 5 calls; ms = write + run +
read-back of the full hidden output), the same two locks taken together (the GPU lock file named by KEV_GPU_LOCK,
default W/gpu.lock, which must exist, and W/.b_heavy.lock, in the heavy lock's arrival order).

    python r17_timing_mac.py --forms C0,C7 --out results/timing_mac_r17_C7.json [--accels gpu_f32 cpu8] [--passes 2]

Forms: C0 = R64+sp+ec+dd+vs6 (tag r14B-vs6), C7 = C0 + bk1024@in_proj_z.q_proj (tag r17C7-bkzq), C3 = C0 + bk1024
(tag r17C3-bk1024). Order: per accelerator, per bucket, the first form then the second (pass 0) and the reverse
(pass 1), so drift shows as a pass difference. Never overwrites --out."""
import argparse
import fcntl
import json
import os
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from timing_mac import Graph, K, inputs, memory, request, single, stamp, swap_used, top_lines  # noqa: E402

GPU_LOCK = Path(os.environ.get("KEV_GPU_LOCK", str(K / "gpu.lock")))
HEAVY_LOCK = K / ".b_heavy.lock"
V = "v2_fp16fc_i8emb"
TAGS = {"C0": "r14B-vs6", "C3": "r17C3-bk1024", "C7": "r17C7-bkzq"}
FORMS = {"C0": "R64+sp+ec+dd+vs6", "C3": "R64+sp+ec+dd+vs6+bk1024", "C7": "R64+sp+ec+dd+vs6+bk1024@in_proj_z.q_proj"}


def acquire_both(max_load, wait_s, poll_s=5.0, yield_s=60.0):
    """r14_timing_mac.acquire_both with its own owner line (see there)."""
    import r14_flock as F
    qdir = HEAVY_LOCK.with_suffix(".queue")
    qdir.mkdir(exist_ok=True)
    gfh, hfh = open(GPU_LOCK, "r+"), open(HEAVY_LOCK, "a+")
    t0, seen, head_since, requeues = time.time(), [], None, 0

    def note(x):
        if not seen or seen[-1] != x:
            seen.append(x)
            del seen[:-12]

    def new_ticket():
        t = qdir / f"{time.time_ns():020d}-{os.getpid()}"
        t.touch()
        return t
    ticket = new_ticket()
    try:
        while True:
            if time.time() - t0 > wait_s:
                gfh.close()
                hfh.close()
                return None, None, {"acquired_at": None, "waited_s": round(time.time() - t0, 1), "requeues": requeues,
                                    "seen_while_waiting": seen}
            head = F.oldest_ticket(qdir)
            if head != ticket.name:
                head_since = None
                note(f"queue: behind {head}")
                time.sleep(poll_s)
                continue
            head_since = head_since or time.time()
            load1 = os.getloadavg()[0]
            content = GPU_LOCK.read_text().strip()
            ok = False
            if content:
                note(f"gpu content: {content}")
            elif load1 >= max_load:
                note(f"load1 {load1:.1f}")
            else:
                try:
                    fcntl.flock(gfh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    if GPU_LOCK.read_text().strip():
                        fcntl.flock(gfh, fcntl.LOCK_UN)
                        note("gpu content appeared")
                    else:
                        try:
                            fcntl.flock(hfh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            ok = True
                        except BlockingIOError:
                            fcntl.flock(gfh, fcntl.LOCK_UN)
                            note("heavy lock held (a job without a ticket, or the previous one finishing)")
                except BlockingIOError:
                    note("gpu flock held by another process")
            if ok:
                ticket.unlink(missing_ok=True)
                line = f"kev-litert r17 timing pid {os.getpid()} since {stamp()}"
                gfh.seek(0)
                gfh.truncate()
                gfh.write(line + "\n")
                gfh.flush()
                hfh.seek(0)
                hfh.truncate()
                hfh.write(line + " (Mac timing: no heavy CPU job)\n")
                hfh.flush()
                return gfh, hfh, {"acquired_at": stamp(), "waited_s": round(time.time() - t0, 1), "line": line,
                                  "load1_at_acquire": round(load1, 2), "requeues": requeues, "seen_while_waiting": seen}
            if time.time() - head_since > yield_s:
                ticket.unlink(missing_ok=True)
                ticket = new_ticket()
                requeues += 1
                head_since = None
                note(f"yielded the head after {yield_s:.0f} s (requeue {requeues})")
            time.sleep(poll_s)
    finally:
        if ticket.exists():
            ticket.unlink(missing_ok=True)


def release_both(gfh, hfh):
    for fh in (gfh, hfh):
        fh.seek(0)
        fh.truncate()
        fh.flush()
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()
    return {"released_at": stamp(), "gpu_content_after": GPU_LOCK.read_text(), "heavy_content_after": HEAVY_LOCK.read_text()}


def row_sets(oracle):
    """r14_timing_mac.row_sets's rows for the three row buckets."""
    by = {f"{q['id']}/{q['qid']}": q for q in oracle["questions"]}
    fiveq = [q for q in oracle["questions"] if q["id"] == "own_fiveq_09"]
    assert len(fiveq) == 5
    return {64: ("short64", "single", [by["tv4_010/answer"]["row_ids"]]),
            128: ("p50_80", "single", [by["tv4_007/answer"]["row_ids"]]),
            256: ("fiveq", "request", [q["row_ids"] for q in fiveq])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--max-load", type=float, default=10.0)
    ap.add_argument("--wait-s", type=float, default=5400.0)
    ap.add_argument("--Ls", nargs="+", type=int, default=[64, 128, 256])
    ap.add_argument("--accels", nargs="+", default=["gpu_f32"], choices=["gpu_f32", "cpu8"])
    ap.add_argument("--forms", default="C0,C3", help="two forms, the first is the reference (C0)")
    a = ap.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    out = K / a.out
    assert not out.exists(), f"refusing to overwrite {a.out}"
    oracle = json.loads((K / "oracle/oracle_0.8b.json").read_text())
    sets = row_sets(oracle)
    forms = tuple(a.forms.split(","))
    assert len(forms) == 2 and all(f in TAGS for f in forms), forms
    files = {(f, L): K / f"exports/kev08b_rowprefill_L{L}_{V}_{TAGS[f]}.tflite" for f in forms for L in a.Ls}
    for p in files.values():
        assert p.exists(), p
    doc = {"what": f"Mac timing: form {forms[1]} next to form {forms[0]} at the row buckets",
           "forms": {f: FORMS[f] for f in forms}, "threads_cpu": a.threads, "passes": a.passes, "accels": a.accels,
           "rows": {L: {"name": s[0], "kind": s[1], "tokens": [len(r) for r in s[2]]} for L, s in sets.items() if L in a.Ls},
           "files": {f"{f}_L{L}": {"file": str(p.relative_to(K)), "bytes": p.stat().st_size} for (f, L), p in files.items()},
           "protocol": "timing_mac.single / request (5 warm-up + 20 timed; ms = write + run + read-back of the whole "
                       "hidden); order per pass first, second then second, first",
           "top_before_lock": top_lines(), "swap_before_lock": swap_used(), "started_at": stamp(), "results": {}}
    gfh, hfh, lock = acquire_both(a.max_load, a.wait_s)
    if gfh is None:
        doc.update(status="not measured (locks busy)", lock=lock)
        out.write_text(json.dumps(doc, indent=1) + "\n")
        print(json.dumps(lock, indent=1))
        return
    t0 = time.time()
    try:
        doc["top_at_lock"], doc["swap_at_lock"] = top_lines(), swap_used()
        for accel in a.accels:
            res = {}
            for p in range(a.passes):
                order = forms if p % 2 == 0 else forms[::-1]
                for L in a.Ls:
                    name, kind, rows = sets[L]
                    for f in order:
                        g = Graph(files[(f, L)], "gpu_f32" if accel == "gpu_f32" else "cpu", a.threads)
                        assert g.L == L and g.d == 1024, (g.L, g.d, L)
                        per = {"compile_s": round(g.compile_s, 2), "fully_accelerated": g.fully,
                               "memory_after_compile": memory()}
                        prepared = [inputs(r, L) for r in rows]
                        if kind == "request":
                            per[name] = {"tokens": [n for _, _, n in prepared], **request(g, prepared)}
                        else:
                            ids, valid, n = prepared[0]
                            per[name] = {"tokens": n, **single(g, ids, valid)}
                        g.close()
                        res[f"pass{p}_{f}_L{L}"] = per
                        v = per[name]
                        ms = (v.get("per_call_ms_write_run_read") or v.get("ms_write_run_read"))["median"]
                        print(accel, f"pass{p}", f, f"L{L}", name, ms, "compile", per["compile_s"], flush=True)
            res["top_after"] = top_lines()
            doc["results"][accel] = res
        doc["status"] = "measured"
    finally:
        lock.update(release_both(gfh, hfh))
        doc.update(lock=lock, top_after_release=top_lines(), swap_after=swap_used(), finished_at=stamp(),
                   seconds_wall=round(time.time() - t0, 1), memory_end=memory())
        doc.setdefault("status", "failed (see the log)")
        out.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: v for k, v in doc.items() if k not in ("results",)}, indent=1))


if __name__ == "__main__":
    main()
