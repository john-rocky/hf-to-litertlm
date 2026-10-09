"""Round 2 acceptance 8 (speed groundwork): Mac wall time of row graphs through the CompiledModel API.

    $EXPORT scripts/d1_clock_mac.py \
        --tflite exports/tiny_rowprefill_L64_fp32.tflite --tflite exports/tiny_rowprefill_L256_fp32.tflite \
        --accel cpu --accel gpu_f32 --accel gpu_default

Run it only inside a measurement window (a wrapper that holds the GPU lock file named by D1_GPU_LOCK for the whole
run; this script does not take it itself). Per file and accelerator (cpu = XNNPACK with --threads, gpu_f32 = Metal with
GpuOptions(enforce_f32=True), gpu_default = Metal default precision): compile once, then per row of
cache/{tag}/check_rows_L{L}.npz chosen by --rows (default: the full row n = L and the half row n = L/2+1) 5 warm-up calls
and 20 timed calls, each timed as write + run + read-back of the whole hidden output and as run only; median / min / max.
The graph computes all L positions, so the time follows L, not the number of real tokens. The first call's hidden state
must be finite. Load and memory lines (top, vm.swapusage) are recorded before and after. Output (never overwritten):
results/{tag}_clock_mac_<label>.json. Numbers from the tiny model only show that the script runs: never a card number.

Round 6c, the real weights (--real; run it inside `quiet_hold.py d1a-r6c-timing-<L> -- ...`):
    $EXPORT \
        scripts/d1_clock_mac.py --real --tflite exports/real_rowprefill_L256_v2_fp16fc_i8emb.tflite \
        --accel gpu_f32 --accel cpu --out results/timing_mac_real_L256.json
The rows are real rows of fixtures/rows.json (the provider's ids), right-padded with the contract's pad id as the gate
pads them: the card sets of device/timing_rows.json (`one_question` = card_text_001/refund, `three_questions` =
card_text_001's 3 rows back to back = one request on the direct row path, `state_3k4` = own_long_34k_001/
refunded_twice) where their L is the file's L, and `bucket_row` = the first row of rows.json whose smallest bucket is the
file's L (the graph computes all L positions, so one row per bucket times the bucket). Per file x accelerator:
compile once (seconds, is_fully_accelerated, the process's phys_footprint after the compile), then per set --warmup
calls and --reps timed rounds (default 5 + 20; --reps 1 --warmup 1 = a single reference value); a call = write (an
embeds graph: the host's gather of the bfloat16 rows + float32 widening, d1_tables.EmbedTable, is inside the write) +
run + read-back of the whole hidden output, also timed as run only; a request set records the round total and every
call. load1 (os.getloadavg) before and after every set, top / swap / the window lock line before and after the run,
the first call's hidden finite. --cpu-reference-above <L>: on the CPU, files whose L is above it get --reps 1
--warmup 1 (the plan: the CPU is timed fully at L256 and L512 only). --merge <json>...: one table
(bucket x form x accelerator x set) from several outputs -> --out.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

K = Path(__file__).resolve().parents[1]
SIGNATURE = "serving_default"
WARMUP, REPS = 5, 20


def stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def machine_lines() -> dict:
    top = subprocess.run(["top", "-l", "1", "-n", "0"], capture_output=True, text=True).stdout.splitlines()
    swap = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.strip()
    lock = Path(os.environ.get("D1_GPU_LOCK", str(K / "gpu.lock")))
    return {"top": [ln for ln in top if ln.startswith(("Load Avg", "CPU usage", "PhysMem"))], "swap": swap,
            "gpu_lock": lock.read_text().strip() if lock.exists() else None, "at": stamp()}


def stats(xs) -> dict:
    a = np.asarray(xs, dtype=np.float64)
    return {"median": round(float(np.median(a)), 4), "min": round(float(a.min()), 4), "max": round(float(a.max()), 4),
            "n": int(a.size)}


def options(accel: str, threads: int):
    from ai_edge_litert.compiled_model import CpuOptions, GpuOptions, HardwareAccelerator, Options

    if accel == "cpu":
        return Options(hardware_accelerators=HardwareAccelerator.CPU, cpu_options=CpuOptions(num_threads=threads))
    return Options(hardware_accelerators=HardwareAccelerator.GPU,
                   gpu_options=GpuOptions(enforce_f32=accel == "gpu_f32"))


def run_file(path: Path, accel: str, threads: int, row_idx: list[int], embed_table) -> dict:
    from ai_edge_litert.compiled_model import CompiledModel

    t0 = time.perf_counter()
    model = CompiledModel.from_file(str(path), options=options(accel, threads))
    compile_s = time.perf_counter() - t0
    fully = bool(model.is_fully_accelerated())
    in_det = model.get_input_tensor_details(SIGNATURE)
    shape = list(model.get_output_tensor_details(SIGNATURE)["hidden"]["shape"])
    L, d = int(shape[1]), int(shape[2])
    tag = path.stem.split("_")[0]
    rows = np.load(K / f"cache/{tag}/check_rows_L{L}.npz")
    ins = {n: model.create_input_buffer_by_name(SIGNATURE, n) for n in in_det}
    outs = {"hidden": model.create_output_buffer_by_name(SIGNATURE, "hidden")}
    res = {"file": str(path.relative_to(K)), "bytes": path.stat().st_size, "accel": accel, "L": L, "d": d,
           "compile_s": round(compile_s, 3), "is_fully_accelerated": fully, "rows": []}
    for k in row_idx:
        ids, valid, n = rows[f"ids_{k}"].astype(np.int32), rows[f"valid_{k}"].astype(np.float32), int(rows[f"n_{k}"])

        def call():
            t = time.perf_counter()
            if "embeds" in in_det:
                ins["embeds"].write(np.ascontiguousarray(embed_table[ids[0]][None], dtype=np.float32))
            else:
                ins["ids"].write(ids)
            ins["valid"].write(valid)
            tr = time.perf_counter()
            model.run_by_name(SIGNATURE, ins, outs)
            run_ms = (time.perf_counter() - tr) * 1000
            h = np.asarray(outs["hidden"].read(L * d, np.float32))
            return (time.perf_counter() - t) * 1000, run_ms, h

        _, _, h0 = call()
        finite = bool(np.isfinite(h0).all())
        for _ in range(WARMUP - 1):
            call()
        tot, run = [], []
        for _ in range(REPS):
            a, b, _ = call()
            tot.append(a)
            run.append(b)
        res["rows"].append({"row": k, "n": n, "first_call_finite": finite, "ms_write_run_read": stats(tot),
                            "ms_run_only": stats(run)})
    for b in list(ins.values()) + list(outs.values()):
        try:
            b.destroy()
        except Exception:
            pass
    model.close()
    return res


PAD_ID = 124893
BUCKETS = (256, 512, 1024, 2048, 4096)


def phys_footprint() -> dict:
    """This process's phys_footprint and its lifetime maximum (proc_pid_rusage v4), bytes (Kev's timing_mac.memory)."""
    import ctypes
    import os
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
    # the whole rusage_info_v4: the kernel writes all of it (a shorter struct here was overrun: heap corruption, then a
    # bus error at the next allocation-heavy step, round 6c)

    class Info(ctypes.Structure):
        _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in fields]

    doc = {"ru_maxrss": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}
    info = Info()
    if ctypes.CDLL("/usr/lib/libproc.dylib").proc_pid_rusage(os.getpid(), 4, ctypes.byref(info)) == 0:
        doc.update(phys_footprint=int(info.ri_phys_footprint),
                   lifetime_max_phys_footprint=int(info.ri_lifetime_max_phys_footprint))
    return doc


def load1() -> float:
    import os

    return round(os.getloadavg()[0], 2)


def real_sets(L: int) -> list[dict]:
    """The rows timed on an L-token file (module docstring, round 6c)."""
    rows = json.loads((K / "fixtures/rows.json").read_text())["rows"]
    text = [r for r in rows if not (r.get("images") or r.get("image_expansion_pending"))]
    smallest = lambda n: next(b for b in BUCKETS if n <= b)
    sets = []
    for s in json.loads((K / "device/timing_rows.json").read_text())["sets"]:
        if s["L"] == L:
            sets.append({"name": s["name"], "kind": s["kind"], "column": s["column"],
                         "rows": [{"key": r["key"], "ids": r["ids"]} for r in s["rows"]]})
    r = next(r for r in text if smallest(r["row_len"]) == L)
    key = f"{r['id']}/{r['qid']}"
    if not any(s["kind"] == "single" and [x["key"] for x in s["rows"]] == [key] for s in sets):   # L4096: = state_3k4
        sets.append({"name": "bucket_row", "kind": "single", "column": f"one row of the L{L} bucket",
                     "rows": [{"key": key, "ids": r["ids"]}]})
    for s in sets:
        assert all(len(x["ids"]) <= L for x in s["rows"]), s["name"]
    return sets


def real_run(path: Path, accel: str, threads: int, warmup: int, reps: int, embed_table) -> dict:
    from ai_edge_litert.compiled_model import CompiledModel

    t0 = time.perf_counter()
    model = CompiledModel.from_file(str(path), options=options(accel, threads))
    compile_s = time.perf_counter() - t0
    fully = bool(model.is_fully_accelerated())
    in_det = model.get_input_tensor_details(SIGNATURE)
    shape = list(model.get_output_tensor_details(SIGNATURE)["hidden"]["shape"])
    L, d = int(shape[1]), int(shape[2])
    embeds = "embeds" in in_det
    assert not embeds or embed_table is not None, "an embeds graph needs --embed-table"
    ins = {n: model.create_input_buffer_by_name(SIGNATURE, n) for n in in_det}
    outs = {"hidden": model.create_output_buffer_by_name(SIGNATURE, "hidden")}
    res = {"file": str(path.relative_to(K)), "bytes": path.stat().st_size, "accel": accel, "L": L, "d": d,
           "graph_input": "embeds (host gather of bfloat16 rows inside the write)" if embeds else "ids",
           "threads": threads if accel == "cpu" else None, "compile_s": round(compile_s, 3),
           "is_fully_accelerated": fully, "memory_after_compile": phys_footprint(), "warmup": warmup, "reps": reps,
           "sets": []}

    def call(ids_row: list[int]):
        n = len(ids_row)
        t = time.perf_counter()
        ids = np.full((1, L), PAD_ID, dtype=np.int32)
        ids[0, :n] = ids_row
        valid = np.zeros((1, L), dtype=np.float32)
        valid[0, :n] = 1.0
        if embeds:
            ins["embeds"].write(np.ascontiguousarray(embed_table.rows(ids[0])[None], dtype=np.float32))
        else:
            ins["ids"].write(ids)
        ins["valid"].write(valid)
        tr = time.perf_counter()
        model.run_by_name(SIGNATURE, ins, outs)
        run_ms = (time.perf_counter() - tr) * 1000
        h = np.asarray(outs["hidden"].read(L * d, np.float32))
        return (time.perf_counter() - t) * 1000, run_ms, h, n

    for s in real_sets(L):
        rec = {"name": s["name"], "kind": s["kind"], "column": s["column"], "keys": [x["key"] for x in s["rows"]],
               "tokens": [len(x["ids"]) for x in s["rows"]], "load1_before": load1(), "started_at": stamp()}
        first = [call(x["ids"]) for x in s["rows"]]
        rec["first_call_finite"] = all(bool(np.isfinite(h[: n * d]).all()) for _, _, h, n in first)
        for i in range(max(0, warmup - 1)):
            for x in s["rows"]:
                call(x["ids"])
        tot, run, per_tot, per_run = [], [], [], []
        for _ in range(reps):
            tt = rr = 0.0
            for x in s["rows"]:
                a_ms, b_ms, _, _ = call(x["ids"])
                tt, rr = tt + a_ms, rr + b_ms
                per_tot.append(a_ms)
                per_run.append(b_ms)
            tot.append(tt)
            run.append(rr)
        rec.update(ms_write_run_read=stats(tot), ms_run_only=stats(run), load1_after=load1(), finished_at=stamp())
        if len(s["rows"]) > 1:
            rec.update(per_call_ms_write_run_read=stats(per_tot), per_call_ms_run_only=stats(per_run))
        res["sets"].append(rec)
        print(json.dumps({"file": path.name, "accel": accel, "set": s["name"], "median_ms": rec["ms_write_run_read"]["median"],
                          "n": rec["ms_write_run_read"]["n"], "load1": [rec["load1_before"], rec["load1_after"]]}),
              flush=True)
    res["memory_after_sets"] = phys_footprint()
    for b in list(ins.values()) + list(outs.values()):
        try:
            b.destroy()
        except Exception:
            pass
    model.close()
    return res


def real_main(a) -> int:
    out = Path(a.out) if Path(a.out).is_absolute() else K / a.out
    assert not out.exists(), f"refusing to overwrite {out}"
    paths = [Path(p) if Path(p).is_absolute() else K / p for p in a.tflite]
    table = None
    if any("embeds" in p.stem for p in paths):
        from d1_tables import EmbedTable

        table = EmbedTable(K / a.embed_table)
    doc = {"what": "round 6c: Mac wall time of the real d1-3B row graphs, CompiledModel (ai-edge-litert "
                   f"{importlib.metadata.version('ai-edge-litert')}); per set: warm-up calls, then timed rounds; "
                   "ms = write + run + read-back of the whole hidden output",
           "machine_before": machine_lines(), "load1_before": load1(), "results": []}
    if table is not None:
        doc["embed_table"] = {"file": a.embed_table, "shape": [table.vocab, table.hidden]}
    t0 = time.time()
    for p in paths:
        for accel in a.accel:
            L = int(p.stem.split("_L")[1].split("_")[0])
            single = accel == "cpu" and a.cpu_reference_above and L > a.cpu_reference_above
            w, r = (1, 1) if single else (a.warmup, a.reps)
            res = real_run(p, accel, a.threads, w, r, table)
            res["single_reference_value"] = bool(single)
            doc["results"].append(res)
    doc.update(machine_after=machine_lines(), load1_after=load1(), seconds_wall=round(time.time() - t0, 1),
               finished_at=stamp())
    out.write_text(json.dumps(doc, indent=1) + "\n")
    return 0


def merge(a) -> int:
    """--merge: one table from several --real outputs."""
    out = Path(a.out) if Path(a.out).is_absolute() else K / a.out
    assert not out.exists(), f"refusing to overwrite {out}"
    rows, sources = [], []
    for src in a.merge:
        p = Path(src) if Path(src).is_absolute() else K / src
        d = json.loads(p.read_text())
        sources.append({"file": str(p.relative_to(K)), "window": d["machine_before"].get("gpu_lock"),
                        "started": d["machine_before"]["at"], "finished": d.get("finished_at"),
                        "load1_before": d.get("load1_before"), "load1_after": d.get("load1_after")})
        for r in d["results"]:
            form = Path(r["file"]).stem.split(f"_L{r['L']}_")[1]
            for s in r["sets"]:
                rows.append({"L": r["L"], "form": form, "file": r["file"], "graph_input": r["graph_input"],
                             "accel": r["accel"], "set": s["name"], "column": s["column"], "keys": s["keys"],
                             "tokens": s["tokens"], "median_ms": s["ms_write_run_read"]["median"],
                             "min_ms": s["ms_write_run_read"]["min"], "max_ms": s["ms_write_run_read"]["max"],
                             "n": s["ms_write_run_read"]["n"], "run_only_median_ms": s["ms_run_only"]["median"],
                             "per_call_median_ms": (s.get("per_call_ms_write_run_read") or {}).get("median"),
                             "load1": [s["load1_before"], s["load1_after"]], "first_call_finite": s["first_call_finite"],
                             "compile_s": r["compile_s"], "is_fully_accelerated": r["is_fully_accelerated"],
                             "single_reference_value": r.get("single_reference_value", False),
                             "started_at": s["started_at"], "source": str(p.relative_to(K))})
    card = {}
    for name, (L, form, accel, sset) in {"one_question": (256, "v2_fp16fc_i8emb", "gpu_f32", "one_question"),
                                         "three_questions_one_pass": (256, "v2_fp16fc_i8emb", "gpu_f32", "three_questions"),
                                         "state_3k4_one_question": (4096, "v2_fp16fc_i8emb", "gpu_f32", "state_3k4")}.items():
        hit = [r for r in rows if (r["L"], r["form"], r["accel"], r["set"]) == (L, form, accel, sset)]
        card[name] = ({k: hit[0][k] for k in ("L", "form", "accel", "keys", "tokens", "median_ms", "min_ms", "max_ms", "n",
                                              "load1", "source")} if hit else None)
    doc = {"what": "round 6c: Mac timing table of the real d1-3B row graphs (median of the timed rounds; ms = write + "
                   "run + read-back)", "sources": sources, "card_columns": card, "rows": rows}
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print("| L | form | accel | set | tokens | median ms | min / max | n | load1 |")
    print("|---:|---|---|---|---|---:|---|---:|---|")
    for r in sorted(rows, key=lambda r: (r["L"], r["form"], r["accel"], r["set"])):
        print(f"| {r['L']} | {r['form']} | {r['accel']} | {r['set']} | {r['tokens']} | {r['median_ms']} | "
              f"{r['min_ms']} / {r['max_ms']} | {r['n']} | {r['load1'][0]} / {r['load1'][1]} |")
    print(json.dumps({"card_columns": {k: (v and v["median_ms"]) for k, v in card.items()}}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", action="append", default=[])
    ap.add_argument("--accel", action="append", choices=["cpu", "gpu_f32", "gpu_default"], default=[])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--rows", default="0,2", help="row indices of check_rows_L{L}.npz (0 = n L, 2 = n L/2+1)")
    ap.add_argument("--label", default="r2")
    ap.add_argument("--embed-weights", default="cache/tiny/tiny_lfm2_seed0.safetensors")
    ap.add_argument("--real", action="store_true", help="round 6c: real rows (module docstring)")
    ap.add_argument("--out", default="", help="--real / --merge output json")
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--cpu-reference-above", type=int, default=0)
    ap.add_argument("--embed-table", default="cache/real/tables/embed_table.safetensors")
    ap.add_argument("--merge", nargs="+", default=[])
    a = ap.parse_args()
    if a.merge:
        return merge(a)
    assert a.tflite and a.accel, "--tflite and --accel are required"
    if a.real:
        return real_main(a)
    paths = [Path(p) if Path(p).is_absolute() else K / p for p in a.tflite]
    tag = paths[0].stem.split("_")[0]
    out = K / f"results/{tag}_clock_mac_{a.label}.json"
    assert not out.exists(), f"refusing to overwrite {out}"
    table = None
    if any("embeds" in p.stem for p in paths):
        from safetensors.numpy import load_file

        table = load_file(str(K / a.embed_weights))["embed_tokens.weight"].astype(np.float32)
    doc = {"what": "Mac wall time of row graphs, CompiledModel (ai-edge-litert "
                   f"{importlib.metadata.version('ai-edge-litert')}); {WARMUP} warm-up calls then {REPS} timed calls per row",
           "card_use": "never (tiny model: shows that the script runs)" if tag == "tiny" else "see REPRODUCE.md",
           "threads_cpu": a.threads, "machine_before": machine_lines(), "results": []}
    t0 = time.time()
    for p in paths:
        for accel in a.accel:
            r = run_file(p, accel, a.threads, [int(x) for x in a.rows.split(",")], table)
            doc["results"].append(r)
            print(json.dumps({k: r[k] for k in ("file", "accel", "compile_s", "is_fully_accelerated")} |
                             {"median_ms": [x["ms_write_run_read"]["median"] for x in r["rows"]]}), flush=True)
    doc.update(machine_after=machine_lines(), seconds_wall=round(time.time() - t0, 1), finished_at=stamp())
    out.write_text(json.dumps(doc, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
