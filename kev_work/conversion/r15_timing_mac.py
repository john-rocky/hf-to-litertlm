"""Mac timing of the Kev-4B row graphs (V2, final kernel) next to the earlier release's loop-kernel files, one accelerator
per run: Metal with float32 activations (gpu_f32), Metal at its default precision = float16 activations (gpu_f16) or the
CPU with 8 threads (cpu8); ai-edge-litert CompiledModel.

    python r15_timing_mac.py --accel gpu_f32 --tag r15R64-sp-ec-dd-vs6-in1-fn5
        --graphs L128,L256,L512,L1024,L2048,loopL512,loopL2048 --repeat loopL512,L512
        --out results/timing_mac_r15_4b_gpu_f32.json                                        (one command)

Hold the GPU lock while it runs (the script does not take a lock). Every graph runs in its own child process (compile,
the sets below, close), so the footprint after compile and the process maximum (proc_pid_rusage phys_footprint /
lifetime max) belong to that graph alone. Graphs and their rows (token ids = the Kev-4B reference's rows):
  L64       exports/kev4b_rowprefill_L64_v2_fp16fc_i8emb_<tag>.tflite: short64 (tv4_010/answer, 64 tokens)
  L128      exports/kev4b_rowprefill_L128_v2_fp16fc_i8emb_<tag>.tflite: p50_80 (tv4_007/answer, 80 tokens), short64
            (tv4_010/answer, 64 tokens)
  L256      ..._L256_...: fiveq (own_fiveq_09's 5 rows, 132 / 142 / 132 / 128 / 128 tokens: one request = 5 calls back to
            back), p50_80
  L512      ..._L512_...: T300 (the first 300 tokens of own_long_log_10/first_failure), fiveq
  L1024     ..._L1024_...: T1000 (its first 1,000 tokens)
  L2048     ..._L2048_...: long_1805 (the whole row), T1000
  loopL512  W/staging/Kev-4B-LiteRT/kev-4b_rowprefill_L512_fp16fc_i8emb.tflite (the earlier release's loop-kernel file,
            in a local copy of that release): T300, fiveq
  loopL2048 W/staging/Kev-4B-LiteRT/kev-4b_rowprefill_L2048_fp16fc_i8emb.tflite: long_1805, T1000
Protocol: 5 warm-up calls, then 20 timed calls per single set; fiveq = 5 warm-up calls, then 20 requests of 5 calls; ms =
write + run + read-back of the full hidden output (and run only); median / min / max. Every call's position-0 hidden
state must be finite. --repeat appends the named graphs again at the end (drift check). Never overwrites --out."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from timing_mac import REPS, WARMUP, memory, stamp, stats, swap_used, top_lines  # noqa: E402

K = Path(__file__).resolve().parents[1]
PAD_ID = 248044
V = "v2_fp16fc_i8emb"
PLAN = {"L64": (64, ("short64",)), "L128": (128, ("p50_80", "short64")), "L256": (256, ("fiveq", "p50_80")), "L512": (512, ("T300", "fiveq")),
        "L1024": (1024, ("T1000",)), "L2048": (2048, ("long_1805", "T1000")), "loopL512": (512, ("T300", "fiveq")),
        "loopL2048": (2048, ("long_1805", "T1000"))}


def graph_file(name, tag):
    if name.startswith("loop"):
        return K / f"staging/Kev-4B-LiteRT/kev-4b_rowprefill_L{PLAN[name][0]}_fp16fc_i8emb.tflite"
    return K / f"exports/kev4b_rowprefill_L{PLAN[name][0]}_{V}_{tag}.tflite"


def rows_of(oracle):
    by = {f"{q['id']}/{q['qid']}": q for q in oracle["questions"]}
    longq = by["own_long_log_10/first_failure"]
    assert longq["row_len"] == 1805
    fiveq = [q for q in oracle["questions"] if q["id"] == "own_fiveq_09"]
    assert [q["row_len"] for q in fiveq] == [132, 142, 132, 128, 128]
    return {"p50_80": [by["tv4_007/answer"]["row_ids"]], "short64": [by["tv4_010/answer"]["row_ids"]],
            "fiveq": [q["row_ids"] for q in fiveq], "T300": [longq["row_ids"][:300]], "T1000": [longq["row_ids"][:1000]],
            "long_1805": [longq["row_ids"]]}


def padded(ids, L):
    n = len(ids)
    assert n <= L, (n, L)
    a = np.full((1, L), PAD_ID, dtype=np.int32)
    a[0, :n] = np.asarray(ids, dtype=np.int32)
    v = np.zeros((1, L), dtype=np.float32)
    v[0, :n] = 1.0
    return a, v, n


class Graph:
    def __init__(self, path, accel, threads):
        from ai_edge_litert.compiled_model import CompiledModel, CpuOptions, GpuOptions, HardwareAccelerator, Options
        if accel.startswith("gpu"):
            gopts = GpuOptions(enforce_f32=(accel == "gpu_f32"))
            opts = Options(hardware_accelerators=HardwareAccelerator.GPU, gpu_options=gopts)
            self.options = gopts._as_flat_kwargs()
        else:
            opts = Options(hardware_accelerators=HardwareAccelerator.CPU, cpu_options=CpuOptions(num_threads=threads))
            self.options = {"cpu_threads": threads}
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


def single(g, ids, valid):
    for _ in range(WARMUP):
        g.call(ids, valid)
    tot, run, finite = [], [], True
    for _ in range(REPS):
        t_ms, r_ms, ok = g.call(ids, valid)
        tot.append(t_ms)
        run.append(r_ms)
        finite &= ok
    return {"ms_write_run_read": stats(tot), "ms_run_only": stats(run), "finite": finite}


def request(g, rows):
    for i in range(WARMUP):
        g.call(*rows[i % len(rows)][:2])
    req_tot, req_run, per_tot, per_run, finite = [], [], [], [], True
    for _ in range(REPS):
        tt = tr = 0.0
        for ids, valid, _n in rows:
            t_ms, r_ms, ok = g.call(ids, valid)
            tt, tr = tt + t_ms, tr + r_ms
            per_tot.append(t_ms)
            per_run.append(r_ms)
            finite &= ok
        req_tot.append(tt)
        req_run.append(tr)
    return {"request_ms_write_run_read": stats(req_tot), "request_ms_run_only": stats(req_run),
            "per_call_ms_write_run_read": stats(per_tot), "per_call_ms_run_only": stats(per_run), "finite": finite}


def child(name, accel, tag, threads):
    oracle = json.loads((K / "oracle/oracle_4b.json").read_text())
    rows = rows_of(oracle)
    L, sets = PLAN[name]
    path = graph_file(name, tag)
    before = memory()
    g = Graph(path, accel, threads)
    after_compile = memory()
    assert g.L == L, (g.L, L)
    res = {"graph": name, "file": str(path.relative_to(K)), "bytes": path.stat().st_size, "L": L, "hidden": g.d,
           "accel": accel, "options": g.options, "compile_s": round(g.compile_s, 2), "fully_accelerated": g.fully,
           "memory_before": before, "memory_after_compile": after_compile, "sets": {}}
    for s in sets:
        padded_rows = [padded(ids, L) for ids in rows[s]]
        if s == "fiveq":
            res["sets"][s] = {"tokens": [n for _, _, n in padded_rows], **request(g, padded_rows)}
        else:
            ids, valid, n = padded_rows[0]
            res["sets"][s] = {"tokens": n, **single(g, ids, valid)}
    res["memory_end"] = memory()
    g.close()
    print("CHILD_JSON " + json.dumps(res), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accel", choices=["gpu_f32", "gpu_f16", "cpu8"], required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--graphs", default="L128,L256,L512,L1024,L2048,loopL512")
    ap.add_argument("--repeat", default="", help="graphs to time again at the end (drift check), e.g. loopL512,L512")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="")
    ap.add_argument("--child", default="")
    ap.add_argument("--model", default="4b", choices=["4b"])
    a = ap.parse_args()
    if a.child:
        return child(a.child, a.accel, a.tag, a.threads)
    out = K / a.out
    assert a.out and not out.exists(), f"refusing to overwrite {a.out}"
    order = [g for g in a.graphs.split(",") if g] + [g for g in a.repeat.split(",") if g]
    for g in order:
        assert g in PLAN and graph_file(g, a.tag).exists(), (g, graph_file(g, a.tag))
    doc = {"what": f"Mac timing, Kev-4B V2 final kernel ({a.tag}) + the earlier loop L512, accel {a.accel}",
           "accel": a.accel, "tag": a.tag, "threads_cpu": a.threads, "order": order,
           "protocol": f"{WARMUP} warm-up calls, then {REPS} timed calls (fiveq: {REPS} requests of 5 calls); ms = write "
                       "+ run + read-back of the full hidden output; one child process per graph",
           "top_at_start": top_lines(), "swap_at_start": swap_used(), "started_at": stamp(), "runs": []}
    t0 = time.time()
    try:
        for i, g in enumerate(order):
            r = subprocess.run([sys.executable, __file__, "--child", g, "--accel", a.accel, "--tag", a.tag,
                                "--threads", str(a.threads)], capture_output=True, text=True, cwd=str(K / "scripts"))
            line = next((ln for ln in r.stdout.splitlines() if ln.startswith("CHILD_JSON ")), None)
            rec = json.loads(line[len("CHILD_JSON "):]) if line else {"graph": g, "returncode": r.returncode,
                                                                       "stderr_tail": r.stderr.splitlines()[-15:]}
            rec["position"] = i
            rec["top_after"] = top_lines()
            doc["runs"].append(rec)
            brief = {s: (v.get("ms_write_run_read") or v.get("per_call_ms_write_run_read") or {}).get("median")
                     for s, v in rec.get("sets", {}).items()}
            req = rec.get("sets", {}).get("fiveq", {}).get("request_ms_write_run_read", {}).get("median")
            print(json.dumps({"graph": g, "compile_s": rec.get("compile_s"), "fully": rec.get("fully_accelerated"),
                              "ms": brief, "fiveq_request": req,
                              "fp_after_compile_gb": round((rec.get("memory_after_compile") or {}).get("phys_footprint", 0) / 1e9, 2),
                              "fp_max_gb": round((rec.get("memory_end") or {}).get("lifetime_max_phys_footprint", 0) / 1e9, 2),
                              "rc": rec.get("returncode")}), flush=True)
        doc["status"] = "measured" if all("sets" in r for r in doc["runs"]) else "partial (see runs)"
    finally:
        doc.update(top_at_end=top_lines(), swap_at_end=swap_used(), finished_at=stamp(),
                   seconds_wall=round(time.time() - t0, 1))
        doc.setdefault("status", "failed (see the log)")
        out.write_text(json.dumps(doc, indent=1) + "\n")


if __name__ == "__main__":
    main()
