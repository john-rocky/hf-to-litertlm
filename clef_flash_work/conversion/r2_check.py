#!/usr/bin/env python3
"""Round 2: one row graph file on the Mac (ai-edge-litert 2.2.0 CompiledModel) on its pre-registered gate set
(scripts/r2_sets.py), then (optionally, same process and compile) the ms protocol.

    TMPDIR=<K>/tmp ~/venvs/lt094dev/bin/python scripts/r2_check.py --tflite out/r2/clef_rowprefill_L1024_v2_fp16fc.tflite \
        --acc cpu|gpu_f32|gpu_default --budget-s 1400 [--ms] [--ms-only] [--threads 8]
        -> cache/r2/<form>_L<L>_<acc>/<id>__<arm>.npy (hidden at the real positions, float32 [T, 4096])
           results/r2_check_<form>_L<L>_<acc>_part<k>.json per invocation; when every run of the set has its file:
           results/r2_check_<form>_L<L>_<acc>.json (the parts merged; never overwritten)
           with --ms / --ms-only: results/r2_ms_<form>_L<L>_<acc>.json (never overwritten)
<form> = the file stem after `clef_rowprefill_L<L>_` (v2_fp16fc, v1_dyni8fc, ns_v2_fp16fc, fp32).

acc: cpu = CpuOptions(num_threads=8), no weight cache path (r1's clef_cpu_check.py); gpu_f32 = Metal with
GpuOptions(enforce_f32=True); gpu_default = Metal with GpuOptions() (default precision = fp16 storage); no constant
tensor sharing (the GpuOptions default), one graph compiled per process. The runtime's VERBOSE lines are switched on
(scripts/clef_rt.py, d1_3b_work/scripts/d1_check.py's method) and this process's fd 2 goes to
logs/r2_check_<form>_L<L>_<acc>_part<k>.runtime.log; every `Replacing N out of M node(s) with delegate (D) node,
yielding P partitions` line is parsed (delegated / total / partitions), with is_fully_accelerated(). On Metal the
line is all nodes in one partition: any line with N < M or P > 1 (or no line) ends the run with status
DELEGATION_FAIL and exit 4 after the json is written; a compile or runtime exception = status FAIL, exit 5.
Resumable: a run whose .npy exists is skipped (temp name + rename). --budget-s: no gate run starts when the elapsed time
plus the median call time (first call: compile + 2 x the call estimate) would pass budget - 15 s. Per run: write / run
/ read ms, non-finite values (real and all positions), all-zero rows and constant rows at the real positions (the GPU
fp16 overflow signature, memory fp16-storage-norm-overflow), the hidden max |diff| against the reference's last_hidden
(fixture / red: ref/npz; held-out: Core AI's held-out npz). Memory: phys_footprint after compile, at the end, and its
lifetime maximum (proc_pid_rusage). Files next to the model and in TMPDIR before / after (a weight cache would show).
ms protocol (r1's scripts/clef_ms.py on this compiled model): own_t12's ids cycled to L real tokens (valid all 1; the
call computes every position whatever the content), inputs written once, 5 warm-up calls, 20 timed calls of
run_by_name only; median / min / max / all values, the first call, the read-back finite, the load average before and
after, whether the lock line held a `clef-r2` label for every timed call, the compile seconds and the memory of this
process. --ms runs it after the gate set is complete in this process when the budget still holds 25 calls + 20 s;
--ms-only runs it without gate rows."""
import argparse
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

K = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K / "scripts"))
import clef_host as H  # noqa: E402
import r2_sets as S  # noqa: E402
from clef_export import memory  # noqa: E402
from clef_rt import LINE_KEYS, PARTITIONED, REPLACING, runtime_log_verbose  # noqa: E402

SNAP = K / "hf/hub/models--Cloudflare--clef-flash/snapshots/f0b7d3cd65296abfe459e6c228b618f5d8b26d2f"
SIG = "serving_default"
LOCK = Path(os.environ.get("MEASUREMENT_LOCK_FILE", "/dev/null"))   # the machine-wide measurement lock (its line is recorded)


def lock_line() -> str:
    try:
        return LOCK.read_text(errors="replace").strip()
    except OSError:
        return ""


def listing(d: Path) -> list:
    return sorted(f"{p.name} {p.stat().st_size}" for p in d.iterdir()) if d.is_dir() else []


def parse_name(path: Path):
    stem = path.stem
    assert stem.startswith("clef_rowprefill_L"), stem
    rest = stem[len("clef_rowprefill_L"):]
    L, form = rest.split("_", 1)
    return int(L), form


def delegation(lines: list) -> dict:
    text = "\n".join(lines)
    return {"replacing": [{"delegated": int(m[0]), "total": int(m[1]), "delegate": m[2], "partitions": int(m[3])}
                          for m in REPLACING.findall(text)],
            "partitioned": [{"subgraph": int(m[0]), "selected": int(m[1]), "total": int(m[2]), "partitions": int(m[3])}
                            for m in PARTITIONED.findall(text)]}


def ms_protocol(model, ins, outs, L, D, table, warmup=5, timed=20) -> dict:
    ref = json.loads((K / "ref/records_ref.json").read_text())
    r = next(x for x in ref["rows"] if x["id"] == "own_t12" and x["arm"] == "text")
    ids = (r["ids"] * (L // len(r["ids"]) + 1))[:L]
    embeds, valid, cos, sin = H.row_inputs(ids, L, table)
    doc = {"input": {"record": "own_t12", "record_tokens": r["tokens"], "row_tokens": L, "valid_ones": int(valid.sum())},
           "loadavg_before": os.getloadavg(), "lock_before": lock_line(), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    for n, v in (("embeds", embeds), ("valid", valid), ("cos", cos), ("sin", sin)):
        ins[n].write(v)
    warm = []
    for _ in range(warmup):
        t = time.perf_counter()
        model.run_by_name(SIG, ins, outs)
        warm.append((time.perf_counter() - t) * 1000)
    lock_ok, labels, tms = True, set(), []
    for _ in range(timed):
        ln = lock_line()
        labels.add(ln.split(" timing")[0])
        lock_ok &= "clef-r2" in ln
        t = time.perf_counter()
        model.run_by_name(SIG, ins, outs)
        tms.append((time.perf_counter() - t) * 1000)
    h = np.asarray(outs["hidden"].read(L * D, np.float32), dtype=np.float32)
    doc.update(first_call_ms=warm[0] if warm else None, warmup_ms=warm, timed_ms=tms,
               ms={"median": float(np.median(tms)), "min": float(np.min(tms)), "max": float(np.max(tms)), "n": len(tms)},
               hidden_finite=bool(np.isfinite(h).all()), loadavg_after=os.getloadavg(),
               lock_held_by_this_run_every_timed_call=bool(lock_ok), lock_labels_seen=sorted(labels),
               lock_after=lock_line(), finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    return doc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", required=True, help="K-relative")
    ap.add_argument("--acc", choices=S.ACCS, required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--budget-s", type=float, default=1e9)
    ap.add_argument("--est-call-s", type=float, default=5.0, help="call estimate before the first measured call")
    ap.add_argument("--ms", action="store_true")
    ap.add_argument("--ms-only", action="store_true")
    a = ap.parse_args()
    t_start = time.monotonic()
    path = K / a.tflite
    L, form = parse_name(path)
    tag = f"{form}_L{L}_{a.acc}"
    cache = K / f"cache/r2/{tag}"
    cache.mkdir(parents=True, exist_ok=True)
    merged_path = K / f"results/r2_check_{tag}.json"
    ms_path = K / f"results/r2_ms_{tag}.json"
    if a.ms_only:
        assert not ms_path.exists(), f"refusing to overwrite {ms_path}"
    prefix = "msonly" if a.ms_only else "part"
    k = 1
    while (K / f"results/r2_check_{tag}_{prefix}{k}.json").exists() or (K / f"logs/r2_check_{tag}_{prefix}{k}.runtime.log").exists():
        k += 1
    part_path = K / f"results/r2_check_{tag}_{prefix}{k}.json"
    runs = S.gate_runs(L, a.acc)
    log = K / f"logs/r2_check_{tag}_{prefix}{k}.runtime.log"
    log_f = open(log, "w")
    saved = os.dup(2)
    os.dup2(log_f.fileno(), 2)
    tmp = Path(os.environ.get("TMPDIR", "/nonexistent"))
    doc = {"what": f"round 2: {a.tflite} on {a.acc}", "tflite": a.tflite, "tflite_bytes": path.stat().st_size,
           "tflite_realpath": str(path.resolve()), "L": L, "form": form, "acc": a.acc, "part": k, "ms_only": a.ms_only,
           "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "ai_edge_litert": importlib.metadata.version("ai-edge-litert"),
           "runtime_log": str(log.relative_to(K)), "files_next_to_model_before": listing(path.parent),
           "tmpdir": str(tmp), "tmpdir_before": listing(tmp), "lock_at_start": lock_line(), "loadavg_start": os.getloadavg(),
           "set": {"calls": len(runs), "fixture": sum(r["kind"] == "fixture" for r in runs),
                   "heldout": sum(r["kind"] == "heldout" for r in runs), "red": sum(r["kind"] == "red" for r in runs)}}
    rows, stopped, ms_doc, exc = [], False, None, None
    try:
        doc["logger"] = runtime_log_verbose()
        from ai_edge_litert.compiled_model import CompiledModel, CpuOptions, GpuOptions, HardwareAccelerator, Options
        if a.acc == "cpu":
            cpu = CpuOptions(num_threads=a.threads)
            opts = Options(hardware_accelerators=HardwareAccelerator.CPU, cpu_options=cpu)
            doc["options"] = {"accelerator": "CPU (XNNPACK)", "cpu_options": cpu._as_flat_kwargs()}
        else:
            gopt = GpuOptions(enforce_f32=(a.acc == "gpu_f32"))
            opts = Options(hardware_accelerators=HardwareAccelerator.GPU, gpu_options=gopt)
            doc["options"] = {"accelerator": "GPU (Metal)", "gpu_options": gopt._as_flat_kwargs()}
        doc["memory_before_compile"] = memory()
        t0 = time.perf_counter()
        model = CompiledModel.from_file(str(path), options=opts)
        doc["compile_seconds"] = round(time.perf_counter() - t0, 2)
        doc["memory_after_compile"] = memory()
        try:
            doc["is_fully_accelerated"] = bool(model.is_fully_accelerated())
        except Exception as e:  # informational
            doc["is_fully_accelerated"] = f"unavailable: {type(e).__name__}: {e}"
        sigs = model.get_signature_list()
        assert list(sigs) == [SIG], sigs
        in_det, out_det = model.get_input_tensor_details(SIG), model.get_output_tensor_details(SIG)
        doc["inputs"] = {n: {kk: str(v) for kk, v in d.items()} for n, d in in_det.items()}
        doc["outputs"] = {n: {kk: str(v) for kk, v in d.items()} for n, d in out_det.items()}
        L2, D = (int(x) for x in list(out_det["hidden"]["shape"])[1:])
        assert L2 == L and sorted(in_det) == ["cos", "embeds", "sin", "valid"], (L2, sorted(in_det))
        ins = {n: model.create_input_buffer_by_name(SIG, n) for n in in_det}
        outs = {n: model.create_output_buffer_by_name(SIG, n) for n in out_det}
        table = H.EmbedTable(SNAP)
        call_ms = []
        if not a.ms_only:
            for r in runs:
                name = f"{S.key(r)}.npy"
                if (cache / name).exists():
                    continue
                est = (np.median(call_ms) / 1000 if call_ms else a.est_call_s)
                if time.monotonic() - t_start + est * (1 if call_ms else 2) > a.budget_s - 15:
                    stopped = True
                    break
                row = S.ref_row(r)
                T = row["tokens"]
                embeds, valid, cos, sin = H.row_inputs(row["ids"], L, table)
                t = time.perf_counter()
                ins["embeds"].write(embeds)
                ins["valid"].write(valid)
                ins["cos"].write(cos)
                ins["sin"].write(sin)
                tr = time.perf_counter()
                model.run_by_name(SIG, ins, outs)
                run_ms = (time.perf_counter() - tr) * 1000
                h = np.asarray(outs["hidden"].read(L * D, np.float32), dtype=np.float32).reshape(L, D)
                all_ms = (time.perf_counter() - t) * 1000
                call_ms.append(run_ms)
                hr = h[:T].copy()
                tmpf = cache / (name + ".tmp.npy")
                np.save(tmpf, hr)
                os.replace(tmpf, cache / name)
                z = np.load(S.ref_npz(r))
                rows.append({"id": r["id"], "arm": r["arm"], "kind": r["kind"], "source": r["source"], "tokens": T,
                             "ms_write_run_read": round(all_ms, 1), "ms_run": round(run_ms, 1),
                             "nonfinite_real": int((~np.isfinite(hr)).sum()), "nonfinite_all": int((~np.isfinite(h)).sum()),
                             "zero_rows_real": int((hr == 0).all(axis=1).sum()),
                             "constant_rows_real": int((hr.max(axis=1) == hr.min(axis=1)).sum()),
                             "hidden_max_abs_vs_reference": float(np.abs(hr - z["last_hidden"]).max())})
                print(f"{len(rows)} {S.key(r)} T={T} run {run_ms:.0f} ms max|dh| ref {rows[-1]['hidden_max_abs_vs_reference']:.3e}",
                      flush=True)
        complete = all((cache / f"{S.key(r)}.npy").exists() for r in runs)
        doc["memory_after_gate"] = memory()
        want_ms = a.ms_only or (a.ms and complete and not ms_path.exists())
        if want_ms:
            est = (np.median(call_ms) / 1000 if call_ms else a.est_call_s)
            if a.ms_only or time.monotonic() - t_start + 25 * est + 20 <= a.budget_s:
                ms_doc = ms_protocol(model, ins, outs, L, D, table)
            else:
                doc["ms_deferred"] = f"budget: {a.budget_s - (time.monotonic() - t_start):.0f} s left < 25 calls x {est:.1f} s + 20 s"
        doc["memory_end"] = memory()
        for b in list(ins.values()) + list(outs.values()):
            try:
                b.destroy()
            except Exception:
                pass
        model.close()
        doc["status"] = "OK"
    except BaseException as e:
        doc["status"] = "FAIL"
        doc["error"] = f"{type(e).__name__}: {e}"
        exc = e
        import traceback
        traceback.print_exc()
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        log_f.close()
    lines = log.read_text(errors="replace").splitlines()
    doc["runtime_log_lines"] = len(lines)
    doc["runtime_log_key_lines"] = [ln for ln in lines if LINE_KEYS.search(ln)][:80]
    doc["delegation"] = delegation(lines)
    rep = doc["delegation"]["replacing"]
    if a.acc != "cpu":
        doc["metal_all_nodes_one_partition"] = bool(rep) and all(x["delegated"] == x["total"] and x["partitions"] == 1 for x in rep)
        if doc["status"] == "OK" and not doc["metal_all_nodes_one_partition"]:
            doc["status"] = "DELEGATION_FAIL"
    doc["files_next_to_model_after"] = listing(path.parent)
    doc["tmpdir_after"] = listing(tmp)
    doc["new_files_next_to_model"] = sorted(set(doc["files_next_to_model_after"]) - set(doc["files_next_to_model_before"]))
    doc["new_files_tmpdir"] = sorted(set(doc["tmpdir_after"]) - set(doc["tmpdir_before"]))
    doc.update(rows=rows, runs_this_part=len(rows), stopped_at_budget=stopped,
               seconds_wall=round(time.monotonic() - t_start, 1), finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               lock_at_end=lock_line(), loadavg_end=os.getloadavg())
    if rows:
        ms = np.asarray([r["ms_run"] for r in rows])
        doc["ms_run_this_part"] = {"median": float(np.median(ms)), "min": float(ms.min()), "max": float(ms.max()),
                                   "note": "per-run times of the gate loop (a correctness run, not the ladder row)"}
    part_path.write_text(json.dumps(doc, indent=1) + "\n")
    if ms_doc is not None:
        ms_doc.update(what=f"round 2: ms of {a.tflite} on {a.acc} (one call = one L{L} prefill)", tflite=a.tflite,
                      tflite_bytes=doc["tflite_bytes"], tflite_realpath=doc["tflite_realpath"], acc=a.acc, L=L, form=form,
                      options=doc.get("options"), ai_edge_litert=doc["ai_edge_litert"],
                      load_and_compile_seconds=doc.get("compile_seconds"),
                      memory_after_compile=doc.get("memory_after_compile"), memory_end=doc.get("memory_end"),
                      delegation=doc["delegation"], is_fully_accelerated=doc.get("is_fully_accelerated"),
                      same_process_as=str(part_path.relative_to(K)), gate_rows_before_ms_in_this_process=len(rows))
        if not ms_path.exists():
            ms_path.write_text(json.dumps(ms_doc, indent=1) + "\n")
    brief = {kk: doc.get(kk) for kk in ("status", "error", "part", "compile_seconds", "is_fully_accelerated", "runs_this_part",
                                        "stopped_at_budget", "delegation", "metal_all_nodes_one_partition",
                                        "new_files_next_to_model", "seconds_wall", "ms_deferred")}
    print(json.dumps(brief, indent=1), flush=True)
    if ms_doc is not None:
        print(json.dumps({"ms": ms_doc["ms"], "first_call_ms": ms_doc["first_call_ms"],
                          "lock_ok": ms_doc["lock_held_by_this_run_every_timed_call"]}), flush=True)
    if not a.ms_only and doc["status"] == "OK" and all((cache / f"{S.key(r)}.npy").exists() for r in runs) \
            and not merged_path.exists():
        part_files = sorted((K / "results").glob(f"r2_check_{tag}_part*.json"),
                            key=lambda p: int(p.stem.rsplit("_part", 1)[1]))
        parts = [json.loads(p.read_text()) for p in part_files]
        allrows = [x for p in parts for x in p["rows"]]
        keys_set = [S.key(r) for r in runs]
        assert sorted(f"{x['id']}__{x['arm']}" for x in allrows) == sorted(keys_set), "parts do not cover the set exactly"
        merged = {"what": doc["what"], "tflite": a.tflite, "tflite_bytes": doc["tflite_bytes"],
                  "tflite_realpath": doc["tflite_realpath"], "L": L, "form": form, "acc": a.acc, "parts": len(parts),
                  "set": doc["set"], "runs": len(allrows), "rows": allrows,
                  "all_finite": all(x["nonfinite_real"] == 0 for x in allrows),
                  "zero_rows_real_total": sum(x["zero_rows_real"] for x in allrows),
                  "constant_rows_real_total": sum(x["constant_rows_real"] for x in allrows),
                  "hidden_max_abs_vs_reference": max(x["hidden_max_abs_vs_reference"] for x in allrows),
                  "compile_seconds": [p.get("compile_seconds") for p in parts],
                  "delegation": [p["delegation"] for p in parts],
                  "metal_all_nodes_one_partition": (all(p.get("metal_all_nodes_one_partition") for p in parts)
                                                    if a.acc != "cpu" else None),
                  "is_fully_accelerated": [p.get("is_fully_accelerated") for p in parts],
                  "memory_after_compile": [p.get("memory_after_compile") for p in parts],
                  "memory_end": [p.get("memory_end") for p in parts],
                  "new_files_next_to_model": sorted({f for p in parts for f in p["new_files_next_to_model"]}),
                  "new_files_tmpdir": sorted({f for p in parts for f in p["new_files_tmpdir"]}),
                  "options": doc.get("options"), "ai_edge_litert": doc["ai_edge_litert"],
                  "cache_dir": str(cache.relative_to(K))}
        merged_path.write_text(json.dumps(merged, indent=1) + "\n")
        print(f"MERGED {merged_path.relative_to(K)}", flush=True)
    if doc["status"] == "DELEGATION_FAIL":
        return 4
    if doc["status"] != "OK":
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
