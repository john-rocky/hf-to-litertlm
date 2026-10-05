"""Mac timing of the Kev-0.8B row buckets: r14_timing_mac.py with one more form, D = C7 (R64+sp+ec+dd+vs6+
bk1024@in_proj_z.q_proj, tag r17C7-bkzq = the published 64-, 128- and 256-token files), timed next to form C (tag
r14B-vs6) in the same run. Metal GPU with float32 activations and the CPU with 8 threads (ai-edge-litert 2.2.0
CompiledModel).

    python r18_timing_mac.py --forms CD --Ls 64 128 256 --no-pair --out results/timing_mac_r18_C7.json
    python r18_timing_mac.py --forms CD --Ls 64 128 256 --no-pair --accels cpu8 --max-load 2.5 --wait-s 1800 --out results/timing_mac_r18_C7_cpu.json

--accels picks the accelerators (default gpu_f32,cpu8); the second command times the CPU only, taking the locks only
while the 1-minute load average is below 2.5 (and gives up after 1,800 s).

Rows: L64 short64 = tv4_010/answer (64 tokens), L128 p50_80 = tv4_007/answer (80 tokens), L256 fiveq = own_fiveq_09, 5 rows
of 128-142 tokens as one request (and per call); with --Ls 512 1024 2048 also T300 / T1000 (cut from
own_long_log_10/first_failure) and long1805 (the whole row). Protocol: timing_mac.py's Graph / single / request, imported
unchanged (5 warm-up calls, then 20 timed calls; fiveq = 20 requests of 5 calls; ms = write + run + read-back of the full
hidden output). Order: per accelerator, per bucket, the forms in the given order (pass 0) and reversed (pass 1).
Locks: as r14_timing_mac.py (the GPU lock file named by KEV_GPU_LOCK, default W/gpu.lock, which must exist, and
W/.b_heavy.lock in the heavy lock's arrival order, taken together). Never overwrites --out."""
import argparse
import fcntl
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from timing_mac import (Graph, K, inputs, memory, request, single, stamp, swap_used, top_lines)  # noqa: E402
from timing_mac_shared import (Pair, load_requests, options, time_shared)  # noqa: E402

GPU_LOCK = Path(os.environ.get("KEV_GPU_LOCK", str(K / "gpu.lock")))
HEAVY_LOCK = K / ".b_heavy.lock"
V = "v2_fp16fc_i8emb"
TAGS = {"A": {64: "r14A-R64-sp-ec", 128: "r12R64f16safe2", 256: "r12R64f16safe2", 512: "r14A-R64-sp-ec",
              1024: "r14A-R64-sp-ec", 2048: "r14A-R64-sp-ec"},
        "B": {64: "r14B-R64-sp-ec-dd-vs8", 128: "r13R64-sp-ec-dd-vs8", 256: "r14B-R64-sp-ec-dd-vs8",
              512: "r14B-R64-sp-ec-dd-vs8", 1024: "r14B-R64-sp-ec-dd-vs8", 2048: "r14B-R64-sp-ec-dd-vs8"},
        "C": {L: "r14B-vs6" for L in (64, 128, 256, 512, 1024, 2048)},   # form C: the published L512 / L1024 / L2048
        "D": {L: "r17C7-bkzq" for L in (64, 128, 256)}}   # form C7: the published L64 / L128 / L256
PAIR_TAG = {"A": "r14A-R64-sp-ec", "B": "r14B-R64-sp-ec-dd-vs8", "C": "r14B-vs6", "D": "r14B-vs6"}
FORMS = {"A": "R64+sp+ec", "B": "R64+sp+ec+dd+vs8", "C": "R64+sp+ec+dd+vs6", "D": "R64+sp+ec+dd+vs6+bk1024@in_proj_z.q_proj"}


def acquire_both(max_load, wait_s, poll_s=5.0, yield_s=60.0):
    """Both locks together, in the heavy lock's arrival order (scripts/r14_flock.py tickets in K/.b_heavy.queue): wait
    for this ticket to be the oldest, then take the GPU lock (empty content + flock) and the heavy flock only while the
    1-minute load is below max_load; at the head of the queue for more than yield_s without the conditions, give the
    turn back (a new ticket at the end) so the heavy jobs behind do not wait on the GPU / the load. Neither lock is held
    while waiting for the other."""
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
                line = f"kev-litert r18 timing pid {os.getpid()} since {stamp()}"
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


class PairS(Pair):
    """timing_mac_shared.Pair with the constant-tensor-sharing switch (Pair itself always shares)."""

    def __init__(self, path, Ls, Lq, accel, threads, share):
        from ai_edge_litert.compiled_model import CompiledModel
        t0 = time.perf_counter()
        self.model = CompiledModel.from_file(str(path), options=options(accel, threads, share=share))
        self.compile_s = time.perf_counter() - t0
        self.fully = bool(self.model.is_fully_accelerated())
        self.Ls, self.Lq = Ls, Lq
        self.ss, self.sq = f"state_prefill_{Ls}", f"question_step_{Ls}_{Lq}"
        sigs = self.model.get_signature_list()
        self.names = list(sigs[self.ss]["outputs"])
        det = self.model.get_output_tensor_details(self.ss)
        self.numel = {n: int(np.prod(list(det[n]["shape"]))) for n in self.names}
        m = self.model
        self.ins_s = {n: m.create_input_buffer_by_name(self.ss, n) for n in ("ids", "valid")}
        self.outs_s = {n: m.create_output_buffer_by_name(self.ss, n) for n in self.names}
        self.ins_q = {n: m.create_input_buffer_by_name(self.sq, n) for n in sigs[self.sq]["inputs"]}
        self.outs_q = {"hidden": m.create_output_buffer_by_name(self.sq, "hidden")}
        self.direct = {**{n: self.ins_q[n] for n in ("ids", "valid", "state_valid")},
                       **{n: self.outs_s[n] for n in self.names}}


def row_sets(oracle):
    by = {f"{q['id']}/{q['qid']}": q for q in oracle["questions"]}
    longq = by["own_long_log_10/first_failure"]
    fiveq = [q for q in oracle["questions"] if q["id"] == "own_fiveq_09"]
    assert longq["row_len"] == 1805 and len(fiveq) == 5
    return {64: ("short64", "single", [by["tv4_010/answer"]["row_ids"]]),
            128: ("p50_80", "single", [by["tv4_007/answer"]["row_ids"]]),
            256: ("fiveq", "request", [q["row_ids"] for q in fiveq]),
            512: ("T300", "single", [longq["row_ids"][:300]]),
            1024: ("T1000", "single", [longq["row_ids"][:1000]]),
            2048: ("long1805", "single", [longq["row_ids"]])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--max-load", type=float, default=10.0)
    ap.add_argument("--wait-s", type=float, default=5400.0)
    ap.add_argument("--Ls", nargs="+", type=int, default=[64, 128, 256, 512, 1024, 2048])
    ap.add_argument("--no-pair", action="store_true")
    ap.add_argument("--forms", default="AB", help="forms to time; --forms CD = form C next to C7")
    ap.add_argument("--accels", default="gpu_f32,cpu8", help="comma-separated, in this order: gpu_f32, cpu8")
    a = ap.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    out = K / a.out
    assert not out.exists(), f"refusing to overwrite {a.out}"
    oracle = json.loads((K / "oracle/oracle_0.8b.json").read_text())
    sets = row_sets(oracle)
    files = {(f, L): K / f"exports/kev08b_rowprefill_L{L}_{V}_{TAGS[f][L]}.tflite" for f in a.forms for L in a.Ls}
    pairs = {} if a.no_pair else {f: K / f"exports/kev08b_sharedstate_Ls128_Lq64_{V}_{PAIR_TAG[f]}.tflite" for f in a.forms}
    for p in [*files.values(), *pairs.values()]:
        assert p.exists(), p
    doc = {"what": "Mac timing: row buckets on the final-kernel form(s) " + a.forms,
           "forms": {f: FORMS[f] for f in a.forms}, "threads_cpu": a.threads, "passes": a.passes,
           "rows": {L: {"name": s[0], "kind": s[1], "tokens": [len(r) for r in s[2]]} for L, s in sets.items()},
           "files": {f"{f}_L{L}": {"file": str(p.relative_to(K)), "bytes": p.stat().st_size} for (f, L), p in files.items()},
           "pair_files": {f: {"file": str(p.relative_to(K)), "bytes": p.stat().st_size} for f, p in pairs.items()},
           "protocol": "timing_mac.single / request (5 warm-up + 20 timed; ms = write + run + read-back of the whole "
                       "hidden); pair: timing_mac_shared.time_shared (5 warm-up + 20 requests)",
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
        for accel in a.accels.split(","):
            res = {}
            for p in range(a.passes):
                order = a.forms if p % 2 == 0 else a.forms[::-1]
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
        if pairs:
            reqs = load_requests(128, 64)
            doc["pair_requests"] = {rid: {k: v for k, v in r.items() if k in ("n_state", "branches", "row_lens",
                                                                             "input_tokens")} for rid, r in reqs.items()}
            pres = {}
            for p in range(a.passes):
                order = a.forms if p % 2 == 0 else a.forms[::-1]
                for cfg in ("gpu_f32_share", "gpu_f32_noshare", "cpu8"):
                    accel = "cpu8" if cfg == "cpu8" else "gpu_f32"
                    for f in order:
                        pr = PairS(pairs[f], 128, 64, accel, a.threads, share=(cfg == "gpu_f32_share"))
                        per = {"compile_s": round(pr.compile_s, 2), "fully_accelerated": pr.fully,
                               "memory_after_compile": memory(), "requests": {}}
                        for rid, req in reqs.items():
                            per["requests"][rid] = {m: time_shared(pr, req, m) for m in ("direct", "host")}
                        pr.close()
                        pres[f"pass{p}_{cfg}_{f}"] = per
                        print("pair", cfg, f"pass{p}", f, {rid: per["requests"][rid]["direct"]["request_ms"]["median"]
                                                          for rid in reqs}, "compile", per["compile_s"], flush=True)
            doc["pair_results"] = pres
        doc["status"] = "measured"
    finally:
        lock.update(release_both(gfh, hfh))
        doc.update(lock=lock, top_after_release=top_lines(), swap_after=swap_used(), finished_at=stamp(),
                   seconds_wall=round(time.time() - t0, 1), memory_end=memory())
        doc.setdefault("status", "failed (see the log)")
        out.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: v for k, v in doc.items() if k not in ("results", "pair_results")}, indent=1))


if __name__ == "__main__":
    main()
