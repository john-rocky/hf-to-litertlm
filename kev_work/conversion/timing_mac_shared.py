"""Mac timing per request of the Kev-0.8B shared-state pair (V2) next to the row form (V2 row files) under one
lock: Metal GPU with float32 activations and the CPU with 8 threads, CompiledModel API of ai-edge-litert. Its Pair,
Row, load_requests, time_shared, time_rows and options are imported by r14_timing_mac.py and r14_timing_mac_req.py; its
own command line times the earlier release's files.

    python timing_mac_shared.py --pair Ls128_Lq64 --out results/timing_mac_shared.json [--lock <file> --lock-tag <text>]
    python timing_mac_shared.py --memory-probe --pair Ls128_Lq64 --out results/memory_mac_shared.json

Requests (real rows of oracle/oracle_0.8b.json): own_fiveq_09 (5 questions, state 99 tokens, branches 33 / 43 / 33 / 29
/ 29), own_ticket_01 (3 questions, state 109, branches 49 / 37 / 21), tv4_000 (1 question, state 30, branch 64).
Forms, one request = the calls that answer all its questions:
  shared_host    state_prefill once + the 48 state outputs read to numpy and written into question_step's inputs + one
                 question_step per question (write + run + read hidden)
  shared_direct  state_prefill once + one question_step per question taking state_prefill's output buffers directly
  row_L<L>       one row call per question on the L-token row file (L128 only for tv4_000, whose row is 94 tokens)
  row_best       one row call per question on the smaller of the L128 / L256 files that holds that row (both loaded)
Protocol: 5 warm-up requests, then 20 timed requests (median / min / max of the request total); per-call medians of
state_prefill, the state hand-over through the host, question_step and the row calls come from the timed requests.
Every question's readout rows must be finite. Compile seconds per file and accelerator (the file's first compile in this
process). The Pair class here always shares constant tensors (r14_timing_mac.PairS takes the switch).
--memory-probe: for each file (shared pair, row L128, row L512) a fresh child process compiles it on the GPU (float32
activations), runs one request and reports phys_footprint after compile / after the request / lifetime max: whether the
two signatures of the shared file hold the weights twice on the GPU. When single-signature files exist, one more child
compiles both in one process (the two-file layout a host would hold) and runs the request through them (state read back
to the host and written into the second model).
The lock protocol is timing_mac.py's (flock + one owner line, wait up to 10 minutes, else "not measured (lock busy)";
TERM empties the lock). Never overwrites --out."""
import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from timing_mac import K, acquire, memory, release, stamp, swap_used, top_lines  # noqa: E402

PAD_ID = 248044
HIDDEN = 1024
WARMUP, REPS = 5, 20
VARIANT = "v2_fp16fc_i8emb"
REQUESTS = ("own_fiveq_09", "own_ticket_01", "tv4_000")


def padded(ids, L):
    n = len(ids)
    assert n <= L, (n, L)
    a = np.full((1, L), PAD_ID, dtype=np.int32)
    a[0, :n] = np.asarray(ids, dtype=np.int32)
    v = np.zeros((1, L), dtype=np.float32)
    v[0, :n] = 1.0
    return a, v


def stats(xs):
    xs = np.asarray(xs, dtype=np.float64)
    return {"median": round(float(np.median(xs)), 2), "min": round(float(xs.min()), 2), "max": round(float(xs.max()), 2),
            "n": int(xs.size)}


def options(accel, threads, share=False):
    """share = GpuOptions(constant_tensor_sharing=True): the pair's two signatures use one copy of the weights."""
    from ai_edge_litert.compiled_model import CpuOptions, GpuOptions, HardwareAccelerator, Options
    if accel == "gpu_f32":
        return Options(hardware_accelerators=HardwareAccelerator.GPU,
                       gpu_options=GpuOptions(enforce_f32=True, constant_tensor_sharing=share))
    return Options(hardware_accelerators=HardwareAccelerator.CPU, cpu_options=CpuOptions(num_threads=threads))


class Pair:
    def __init__(self, path, Ls, Lq, accel, threads):
        from ai_edge_litert.compiled_model import CompiledModel
        t0 = time.perf_counter()
        self.model = CompiledModel.from_file(str(path), options=options(accel, threads, share=True))
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

    def request(self, req, mode):
        """-> (total ms, {part: ms}, finite)."""
        parts = {"state": 0.0, "roundtrip": 0.0, "questions": []}
        t = time.perf_counter()
        self.ins_s["ids"].write(req["s_ids"])
        self.ins_s["valid"].write(req["s_valid"])
        self.model.run_by_name(self.ss, self.ins_s, self.outs_s)
        t1 = time.perf_counter()
        parts["state"] = (t1 - t) * 1000
        if mode == "host":
            for n in self.names:
                self.ins_q[n].write(np.asarray(self.outs_s[n].read(self.numel[n], np.float32), dtype=np.float32))
        self.ins_q["state_valid"].write(req["s_valid"])
        t2 = time.perf_counter()
        parts["roundtrip"] = (t2 - t1) * 1000
        finite = True
        inputs = self.ins_q if mode == "host" else self.direct
        for q_ids, q_valid, sel in req["questions"]:
            tq = time.perf_counter()
            self.ins_q["ids"].write(q_ids)
            self.ins_q["valid"].write(q_valid)
            self.model.run_by_name(self.sq, inputs, self.outs_q)
            h = np.asarray(self.outs_q["hidden"].read(self.Lq * HIDDEN, np.float32)).reshape(self.Lq, HIDDEN)
            parts["questions"].append((time.perf_counter() - tq) * 1000)
            finite &= bool(np.isfinite(h[sel]).all())
        return (time.perf_counter() - t) * 1000, parts, finite

    def close(self):
        for b in [*self.ins_s.values(), *self.outs_s.values(), *self.ins_q.values(), *self.outs_q.values()]:
            try:
                b.destroy()
            except Exception:
                pass
        self.model.close()


class Row:
    def __init__(self, path, accel, threads):
        from ai_edge_litert.compiled_model import CompiledModel
        t0 = time.perf_counter()
        self.model = CompiledModel.from_file(str(path), options=options(accel, threads))
        self.compile_s = time.perf_counter() - t0
        self.fully = bool(self.model.is_fully_accelerated())
        self.key = next(iter(self.model.get_signature_list()))
        out = self.model.get_output_tensor_details(self.key)["hidden"]
        self.L = int(list(out["shape"])[1])
        self.ins = {n: self.model.create_input_buffer_by_name(self.key, n) for n in ("ids", "valid")}
        self.outs = {"hidden": self.model.create_output_buffer_by_name(self.key, "hidden")}

    def call(self, ids, valid, sel):
        t = time.perf_counter()
        self.ins["ids"].write(ids)
        self.ins["valid"].write(valid)
        self.model.run_by_name(self.key, self.ins, self.outs)
        h = np.asarray(self.outs["hidden"].read(self.L * HIDDEN, np.float32)).reshape(self.L, HIDDEN)
        return (time.perf_counter() - t) * 1000, bool(np.isfinite(h[sel]).all())

    def close(self):
        for b in [*self.ins.values(), *self.outs.values()]:
            try:
                b.destroy()
            except Exception:
                pass
        self.model.close()


def load_requests(Ls, Lq):
    oracle = json.loads((K / "oracle/oracle_0.8b.json").read_text())
    reqs = {r["id"]: r for r in oracle["requests"]}
    out = {}
    for rid in REQUESTS:
        n = reqs[rid]["state_tokens"]
        qs = [q for q in oracle["questions"] if q["id"] == rid]
        s_ids, s_valid = padded(qs[0]["row_ids"][:n], Ls)
        questions, rows = [], []
        for q in qs:
            sel = [q["decide_idx"] - n] + [o - n for o in q["opt_idx"]]
            questions.append((*padded(q["row_ids"][n:], Lq), sel))
            rows.append({"ids": q["row_ids"], "sel": [q["decide_idx"]] + list(q["opt_idx"])})
        out[rid] = {"n_state": n, "branches": [q["row_len"] - n for q in qs], "row_lens": [q["row_len"] for q in qs],
                    "input_tokens": reqs[rid]["usage"]["input_tokens"], "s_ids": s_ids, "s_valid": s_valid,
                    "questions": questions, "rows": rows}
    return out


def time_shared(pair, req, mode):
    for _ in range(WARMUP):
        pair.request(req, mode)
    tot, state, rt, qms, finite = [], [], [], [], True
    for _ in range(REPS):
        t, parts, ok = pair.request(req, mode)
        tot.append(t)
        state.append(parts["state"])
        rt.append(parts["roundtrip"])
        qms += parts["questions"]
        finite &= ok
    return {"request_ms": stats(tot), "state_prefill_ms": stats(state), "roundtrip_ms": stats(rt),
            "question_step_ms": stats(qms), "finite": finite}


def time_rows(graphs, req, pick):
    """graphs {L: Row}; pick(row_len) -> L."""
    def one():
        tot, per, ok = 0.0, [], True
        for r in req["rows"]:
            g = graphs[pick(len(r["ids"]))]
            ids, valid = padded(r["ids"], g.L)
            ms, f = g.call(ids, valid, r["sel"])
            tot += ms
            per.append(ms)
            ok &= f
        return tot, per, ok
    for _ in range(WARMUP):
        one()
    tot, per, finite = [], [], True
    for _ in range(REPS):
        t, p, ok = one()
        tot.append(t)
        per += p
        finite &= ok
    return {"request_ms": stats(tot), "row_call_ms": stats(per), "finite": finite,
            "files_L": sorted({pick(len(r["ids"])) for r in req["rows"]})}


def run_all(pair_tag, Ls, Lq, threads):
    reqs = load_requests(Ls, Lq)
    shared = K / f"exports/kev08b_sharedstate_{pair_tag}_{VARIANT}.tflite"
    rowf = {L: K / f"exports/kev08b_rowprefill_L{L}_{VARIANT}.tflite" for L in (128, 256, 512)}
    results = {}
    for accel in ("gpu_f32", "cpu8"):
        res = {"compile_s": {}, "fully_accelerated": {}, "memory_after_compile": {}, "requests": {}}
        p = Pair(shared, Ls, Lq, accel, threads)
        res["compile_s"]["shared"], res["fully_accelerated"]["shared"] = round(p.compile_s, 2), p.fully
        res["memory_after_compile"]["shared"] = memory()
        for rid, req in reqs.items():
            res["requests"].setdefault(rid, {})
            for mode in ("host", "direct"):
                res["requests"][rid][f"shared_{mode}"] = time_shared(p, req, mode)
            print(accel, rid, {m: res["requests"][rid][f"shared_{m}"]["request_ms"]["median"] for m in ("host", "direct")},
                  flush=True)
        p.close()
        graphs = {}
        for L in (128, 256, 512):
            graphs[L] = Row(rowf[L], accel, threads)
            res["compile_s"][f"row_L{L}"], res["fully_accelerated"][f"row_L{L}"] = round(graphs[L].compile_s, 2), graphs[L].fully
        for rid, req in reqs.items():
            for L in (256, 512):
                res["requests"][rid][f"row_L{L}"] = time_rows(graphs, req, lambda n, L=L: L)
            if max(req["row_lens"]) <= 128:
                res["requests"][rid]["row_L128"] = time_rows(graphs, req, lambda n: 128)
            res["requests"][rid]["row_best"] = time_rows(graphs, req, lambda n: 128 if n <= 128 else 256)
            print(accel, rid, {k: v["request_ms"]["median"] for k, v in res["requests"][rid].items()}, flush=True)
        for g in graphs.values():
            g.close()
        res["top_after"] = top_lines()
        results[accel] = res
    meta = {rid: {k: v for k, v in r.items() if k in ("n_state", "branches", "row_lens", "input_tokens")}
            for rid, r in reqs.items()}
    files = {"shared": {"file": str(shared.relative_to(K)), "bytes": shared.stat().st_size},
             **{f"row_L{L}": {"file": str(p.relative_to(K)), "bytes": p.stat().st_size} for L, p in rowf.items()}}
    return meta, files, results


def split_request(paths, Ls, Lq, req):
    """The two-file layout: state_prefill from one file, question_step from the other, host hand-over."""
    from ai_edge_litert.compiled_model import CompiledModel
    t0 = time.perf_counter()
    ms = CompiledModel.from_file(str(paths[0]), options=options("gpu_f32", 8))
    mq = CompiledModel.from_file(str(paths[1]), options=options("gpu_f32", 8))
    compile_s = time.perf_counter() - t0
    fully = bool(ms.is_fully_accelerated()) and bool(mq.is_fully_accelerated())
    after_compile = memory()
    ss, sq = f"state_prefill_{Ls}", f"question_step_{Ls}_{Lq}"
    names = list(ms.get_signature_list()[ss]["outputs"])
    det = ms.get_output_tensor_details(ss)
    ins_s = {n: ms.create_input_buffer_by_name(ss, n) for n in ("ids", "valid")}
    outs_s = {n: ms.create_output_buffer_by_name(ss, n) for n in names}
    ins_q = {n: mq.create_input_buffer_by_name(sq, n) for n in mq.get_signature_list()[sq]["inputs"]}
    outs_q = {"hidden": mq.create_output_buffer_by_name(sq, "hidden")}
    ins_s["ids"].write(req["s_ids"])
    ins_s["valid"].write(req["s_valid"])
    ms.run_by_name(ss, ins_s, outs_s)
    for n in names:
        ins_q[n].write(np.asarray(outs_s[n].read(int(np.prod(list(det[n]["shape"]))), np.float32), dtype=np.float32))
    ins_q["state_valid"].write(req["s_valid"])
    ok, hs = True, []
    for q_ids, q_valid, sel in req["questions"]:
        ins_q["ids"].write(q_ids)
        ins_q["valid"].write(q_valid)
        mq.run_by_name(sq, ins_q, outs_q)
        h = np.asarray(outs_q["hidden"].read(Lq * HIDDEN, np.float32)).reshape(Lq, HIDDEN)
        ok &= bool(np.isfinite(h[sel]).all())
        hs.append(h[sel])
    after_run = memory()
    for b in [*ins_s.values(), *outs_s.values(), *ins_q.values(), *outs_q.values()]:
        try:
            b.destroy()
        except Exception:
            pass
    ms.close()
    mq.close()
    return compile_s, fully, ok, after_compile, after_run, hs


def memory_child(kind, path, Ls, Lq):
    reqs = load_requests(Ls, Lq)
    req = reqs["own_ticket_01"]
    before = memory()
    if kind == "split":
        paths = [K / p for p in str(path.relative_to(K)).split("+")]
        compile_s, fully, ok, after_compile, after_run, _ = split_request(paths, Ls, Lq, req)
        print(json.dumps({"kind": kind, "file": "+".join(str(p.relative_to(K)) for p in paths),
                          "compile_s": round(compile_s, 2), "fully_accelerated": fully, "finite": ok, "before": before,
                          "after_compile": after_compile, "after_request": after_run}))
        return
    if kind == "shared":
        g = Pair(path, Ls, Lq, "gpu_f32", 8)
        after_compile = memory()
        _, _, ok = g.request(req, "host")
    else:
        g = Row(path, "gpu_f32", 8)
        after_compile = memory()
        ok = True
        rows = [r for r in reqs["own_ticket_01"]["rows"] + reqs["tv4_000"]["rows"] if len(r["ids"]) <= g.L]
        for r in rows:
            ids, valid = padded(r["ids"], g.L)
            ok &= g.call(ids, valid, r["sel"])[1]
    after_run = memory()
    g.close()
    print(json.dumps({"kind": kind, "file": str(Path(path).relative_to(K)), "compile_s": round(g.compile_s, 2),
                      "fully_accelerated": g.fully, "finite": ok, "before": before, "after_compile": after_compile,
                      "after_request": after_run}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="Ls128_Lq64")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--lock", default="")
    ap.add_argument("--lock-tag", default="kev r10 timing")
    ap.add_argument("--memory-probe", action="store_true")
    ap.add_argument("--child", nargs=2, metavar=("KIND", "PATH"))
    a = ap.parse_args()
    Ls, Lq = (int(x[2:]) for x in a.pair.split("_"))
    if a.child:
        return memory_child(a.child[0], K / a.child[1], Ls, Lq)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    out = K / a.out
    assert not out.exists(), f"refusing to overwrite {a.out}"
    doc = {"what": "Mac timing per request of the shared-state pair vs the row form (V2, GPU float32 activations, CPU "
                   "threads)" if not a.memory_probe else "Mac GPU memory of the shared-state pair vs row-form files",
           "pair": a.pair, "variant": VARIANT, "threads_cpu": a.threads,
           "protocol": f"{WARMUP} warm-up requests, then {REPS} timed requests; ms = wall of all calls of the request "
                       "(write + run + read-back of the full hidden output per call)",
           "top_before_lock": top_lines(), "swap_before_lock": swap_used(), "started_at": stamp()}
    fh, lock = None, {"path": None, "note": "no lock file given"}
    if a.lock:
        fh, lock = acquire(a.lock, a.lock_tag)
        lock = {"path": a.lock, **lock}
        if fh is None:
            doc.update(status="not measured (lock busy)", lock=lock)
            out.write_text(json.dumps(doc, indent=1) + "\n")
            return
    t0 = time.time()
    try:
        doc["top_at_lock"], doc["swap_at_lock"] = top_lines(), swap_used()
        if a.memory_probe:
            probes = []
            kinds = [("shared", f"exports/kev08b_sharedstate_{a.pair}_{VARIANT}.tflite"),
                     ("row", f"exports/kev08b_rowprefill_L128_{VARIANT}.tflite"),
                     ("row", f"exports/kev08b_rowprefill_L512_{VARIANT}.tflite")]
            split = [f"exports/kev08b_sharedstate_{a.pair}_{p}only_{VARIANT}.tflite" for p in ("state", "question")]
            if all((K / p).exists() for p in split):
                kinds.append(("split", "+".join(split)))
            for kind, rel in kinds:
                r = subprocess.run([sys.executable, __file__, "--pair", a.pair, "--out", a.out, "--child", kind, rel],
                                   capture_output=True, text=True, cwd=str(K / "scripts"))
                line = next((ln for ln in r.stdout.splitlines() if ln.startswith("{")), None)
                probes.append(json.loads(line) if line else {"kind": kind, "file": rel, "returncode": r.returncode,
                                                             "stderr_tail": r.stderr.splitlines()[-10:]})
                print(json.dumps({k: probes[-1].get(k) for k in ("kind", "file", "compile_s")}), flush=True)
            doc["probes"] = probes
        else:
            doc["requests"], doc["files"], doc["results"] = run_all(a.pair, Ls, Lq, a.threads)
        doc["status"] = "measured"
    finally:
        if fh is not None:
            lock.update(release(fh, a.lock))
        doc.update(lock=lock, top_after_release=top_lines(), swap_after=swap_used(), finished_at=stamp(),
                   seconds_wall=round(time.time() - t0, 1))
        doc.setdefault("status", "failed (see the log)")
        out.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: v for k, v in doc.items() if k not in ("results", "probes")}, indent=1, default=str))


if __name__ == "__main__":
    main()
