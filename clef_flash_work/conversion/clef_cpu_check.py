#!/usr/bin/env python3
"""Round 1 step 9: the exported row graph on the Mac CPU (ai-edge-litert 2.2.0 CompiledModel, XNNPACK, 8 threads, no
weight cache file) on every reference text run that fits L, plus the red arms; the hidden states go to the author's
head in venv-ref (scripts/clef_score.py --hidden-dir cache/r1/cpu --tag cpu_L1024).

    ~/venvs/lt094dev/bin/python scripts/clef_cpu_check.py --tflite out/r1/clef_rowprefill_L1024_fp32.tflite \
        [--threads 8] [--deadline-s 1440]
        -> cache/r1/cpu/<id>__text.npy (hidden at the real positions, float32 [T, 4096]), one per run
           results/cpu_check_L1024_part<k>.json per invocation (k = 1, 2, ...); when every run has its file:
           results/cpu_check_L1024.json (the parts merged)

Resumable: a run whose .npy exists is skipped (each .npy is written to a temp name and renamed); no new run starts
after --deadline-s seconds. Inputs by signature name: embeds / valid / cos / sin from the NumPy host
(scripts/clef_host.py, the reference's ids); `hidden` read back [1, L, 4096]. Recorded: compile seconds,
is_fully_accelerated, the runtime's delegation lines (VERBOSE log on, this process's fd 2 -> logs/cpu_check_L<L>_part<k>
.runtime.log: XNNPACK's `Replacing N out of M node(s)` = delegated / total nodes and partitions), per run write / run /
read ms and non-finite counts, the hidden max |diff| against the reference's last_hidden and against the torch check's
hidden where it exists, memory (phys_footprint) after compile and at the end, and the files next to the model and in
TMPDIR before / after (a weight cache file would show there; the CpuOptions weight-cache path is left empty)."""
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
from clef_export import memory  # noqa: E402
from clef_rt import LINE_KEYS, PARTITIONED, REPLACING, runtime_log_verbose  # noqa: E402

SNAP = K / "hf/hub/models--Cloudflare--clef-flash/snapshots/f0b7d3cd65296abfe459e6c228b618f5d8b26d2f"
SIG = "serving_default"


def listing(d: Path) -> list:
    return sorted(f"{p.name} {p.stat().st_size}" for p in d.iterdir()) if d.is_dir() else []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--deadline-s", type=float, default=None)
    a = ap.parse_args()
    t_start = time.monotonic()
    path = K / a.tflite
    cache = K / "cache/r1/cpu"
    cache.mkdir(parents=True, exist_ok=True)
    k = 1
    while (K / f"results/cpu_check_L1024_part{k}.json").exists():
        k += 1
    ref = json.loads((K / "ref/records_ref.json").read_text())
    red = json.loads((K / "ref/records_ref_red.json").read_text())
    runs_all = [r for r in ref["rows"] if r["arm"] == "text"] + list(red["rows"])
    log = K / f"logs/cpu_check_part{k}.runtime.log"
    log_f = open(log, "w")
    saved = os.dup(2)
    os.dup2(log_f.fileno(), 2)
    doc = {"what": "round 1 step 9: CompiledModel CPU (XNNPACK) on the fp32 row graph", "tflite": a.tflite,
           "tflite_bytes": path.stat().st_size, "part": k, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "ai_edge_litert": importlib.metadata.version("ai-edge-litert"), "threads": a.threads,
           "runtime_log": str(log.relative_to(K)), "files_next_to_model_before": listing(path.parent),
           "tmpdir": os.environ.get("TMPDIR"), "tmpdir_before": listing(Path(os.environ.get("TMPDIR", "/nonexistent")))}
    rows, stopped = [], False
    try:
        doc["logger"] = runtime_log_verbose()
        from ai_edge_litert.compiled_model import CompiledModel, CpuOptions, HardwareAccelerator, Options
        cpu = CpuOptions(num_threads=a.threads)
        doc["cpu_options"] = cpu._as_flat_kwargs()
        opts = Options(hardware_accelerators=HardwareAccelerator.CPU, cpu_options=cpu)
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
        L, D = (int(x) for x in list(out_det["hidden"]["shape"])[1:])
        assert sorted(in_det) == ["cos", "embeds", "sin", "valid"], sorted(in_det)
        doc["L"] = L
        runs = [r for r in runs_all if r["tokens"] <= L]
        doc["runs_total"] = len(runs)
        doc["runs_skipped_longer_than_L"] = [f"{r['id']}/{r['arm']} {r['tokens']}" for r in runs_all if r["tokens"] > L]
        ins = {n: model.create_input_buffer_by_name(SIG, n) for n in in_det}
        outs = {n: model.create_output_buffer_by_name(SIG, n) for n in out_det}
        table = H.EmbedTable(SNAP)
        for r in runs:
            name = f"{r['id']}__{r['arm']}.npy"
            if (cache / name).exists():
                continue
            if a.deadline_s is not None and time.monotonic() - t_start > a.deadline_s:
                stopped = True
                break
            T = r["tokens"]
            embeds, valid, cos, sin = H.row_inputs(r["ids"], L, table)
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
            hr = h[:T].copy()
            tmp = cache / (name + ".tmp.npy")
            np.save(tmp, hr)
            os.replace(tmp, cache / name)
            z = np.load(K / f"ref/npz/{r['id']}__{r['arm']}.npz")
            row = {"id": r["id"], "arm": r["arm"], "source": r["source"], "tokens": T, "ms_write_run_read": round(all_ms, 1),
                   "ms_run": round(run_ms, 1), "nonfinite_real": int((~np.isfinite(hr)).sum()),
                   "nonfinite_all": int((~np.isfinite(h)).sum()),
                   "hidden_max_abs_vs_reference": float(np.abs(hr - z["last_hidden"]).max())}
            tc = K / f"cache/r1/torch/{name}"
            if tc.exists():
                row["hidden_max_abs_vs_torch_check"] = float(np.abs(hr - np.load(tc)).max())
            rows.append(row)
            print(f"{len(rows)} {r['id']} T={T} run {run_ms:.0f} ms max|dh| ref {row['hidden_max_abs_vs_reference']:.3e}",
                  file=sys.stdout, flush=True)
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
        import traceback
        traceback.print_exc()
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        log_f.close()
    lines = log.read_text(errors="replace").splitlines()
    doc["runtime_log_lines"] = len(lines)
    doc["runtime_log_key_lines"] = [ln for ln in lines if LINE_KEYS.search(ln)][:80]
    doc["delegation"] = {
        "replacing": [{"delegated": int(m[0]), "total": int(m[1]), "delegate": m[2], "partitions": int(m[3])}
                      for m in REPLACING.findall("\n".join(lines))],
        "partitioned": [{"subgraph": int(m[0]), "selected": int(m[1]), "total": int(m[2]), "partitions": int(m[3])}
                        for m in PARTITIONED.findall("\n".join(lines))]}
    doc["files_next_to_model_after"] = listing(path.parent)
    doc["tmpdir_after"] = listing(Path(os.environ.get("TMPDIR", "/nonexistent")))
    doc["new_files_next_to_model"] = sorted(set(doc["files_next_to_model_after"]) - set(doc["files_next_to_model_before"]))
    doc.update(rows=rows, runs_this_part=len(rows), stopped_at_deadline=stopped,
               seconds_wall=round(time.monotonic() - t_start, 1), finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    if rows:
        ms = np.asarray([r["ms_run"] for r in rows])
        doc["ms_run_this_part"] = {"median": float(np.median(ms)), "min": float(ms.min()), "max": float(ms.max()),
                                   "note": "per-run times of the gate loop (a correctness run, not the ladder row)"}
    (K / f"results/cpu_check_L1024_part{k}.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({kk: doc.get(kk) for kk in ("status", "error", "part", "compile_seconds", "is_fully_accelerated",
                                                 "runs_this_part", "stopped_at_deadline", "delegation",
                                                 "new_files_next_to_model", "seconds_wall")}, indent=1), flush=True)
    done = doc["status"] == "OK" and all((cache / f"{r['id']}__{r['arm']}.npy").exists() for r in runs_all if r["tokens"] <= 1024)
    if done and not (K / "results/cpu_check_L1024.json").exists():
        parts = [json.loads((K / f"results/cpu_check_L1024_part{i}.json").read_text()) for i in range(1, k + 1)]
        allrows = [x for p in parts for x in p["rows"]]
        merged = {"what": doc["what"], "tflite": a.tflite, "tflite_bytes": doc["tflite_bytes"], "parts": len(parts),
                  "runs": len(allrows), "rows": allrows,
                  "all_finite": all(x["nonfinite_real"] == 0 for x in allrows),
                  "hidden_max_abs_vs_reference": max(x["hidden_max_abs_vs_reference"] for x in allrows),
                  "hidden_max_abs_vs_torch_check": max((x["hidden_max_abs_vs_torch_check"] for x in allrows
                                                         if "hidden_max_abs_vs_torch_check" in x), default=None),
                  "compile_seconds": [p.get("compile_seconds") for p in parts],
                  "delegation": [p["delegation"] for p in parts], "is_fully_accelerated": [p.get("is_fully_accelerated") for p in parts],
                  "memory_after_compile": [p.get("memory_after_compile") for p in parts],
                  "new_files_next_to_model": sorted({f for p in parts for f in p["new_files_next_to_model"]}),
                  "cpu_options": doc.get("cpu_options"), "ai_edge_litert": doc["ai_edge_litert"]}
        (K / "results/cpu_check_L1024.json").write_text(json.dumps(merged, indent=1) + "\n")
        print("MERGED results/cpu_check_L1024.json", flush=True)
    return 0 if doc["status"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
