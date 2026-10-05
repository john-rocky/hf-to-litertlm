"""Desktop gate: one exported graph on the Mac (ai-edge-litert CompiledModel) against the reference.

    python litert_parity.py --tflite exports/kev08b_rowprefill_L1024_fp32.tflite --L 1024
    python litert_parity.py --tflite exports/kev08b_rowprefill_L1024_v2_fp16fc_i8emb.tflite --L 1024 \
        --variant v2_fp16fc_i8emb --accel cpu | --accel gpu --f32 | --accel gpu      [--model 4b] [--limit N]

The file is over 2 GiB, so CompiledModel.from_file. Inputs are written by signature name (ids int32 [1, L], valid
float32 [1, L]); the output `hidden` [1, L, d] float32 is read back, [decide, *opts] is selected and read out on the
host (head.pt, numpy float32), then compared with the reference: argmax per question, max / mean / p95 |dp| (pooled
over all options), h_sel max |diff| (reference npz), the three full rows (real positions), by source, near-tie rows,
flips, and the red arm (graph(red_arm_000) vs reference(tv4_000) must exceed 0.02). When the torch-graph h_sel npz of
torch_graph_parity.py is present, h_sel is also compared with it. Timing (compile seconds; per row: write + run +
read-back) is informational: the Mac is shared with other work.

With --variant (and always for 4B):
  --accel gpu = Metal (Options(hardware_accelerators=GPU, gpu_options=GpuOptions(enforce_f32=--f32))); the run refuses
    a file whose EMBEDDING_LOOKUP table is not INT8 (--allow-float-table overrides; run that in a child process).
    is_fully_accelerated() is recorded (a GPU row counts only if True).
  non-finite hidden values are counted at real positions and at all positions; a question whose h_sel is non-finite
    is reported, excluded from the |dp| statistics and fails the bar.
  h_sel of every question is saved to cache/r3/hsel_{tag}_L{L}_{variant}.npz (cache/r4b for 4B, or --cache <dir>);
    a GPU run compares itself with the CPU run of the same file when that exists (rows json + npz).
  memory: ru_maxrss and phys_footprint (proc_pid_rusage) after compile and at the end.
--limit N runs the first N questions that fit (outputs get a _smoke{N} suffix).
Outputs (never overwritten): results/litert_{cpu|gpu_f32|gpu_f16}_parity_L{L}[_4b][_{variant}].json and
results/litert_{...}_rows_L{L}[_4b][_{variant}].json."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from r2_common import (HIDDEN, MODEL, ORACLE_NPZ, RESULT_SUFFIX, K, Clock, Head, add_model_arg, cache_dir, compare_question,
                       dump_json, load_oracle, qkey, select, summarize)
from r3_common import embedding_table_dtypes, gpu_vs_cpu, memory

PAD_ID = 248044


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", required=True)
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="first N fitting questions (smoke; outputs get a _smoke suffix)")
    ap.add_argument("--accel", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--f32", action="store_true", help="GPU: GpuOptions(enforce_f32=True) = fp32 activations")
    ap.add_argument("--variant", default="", help="output suffix (e.g. v2_fp16fc_i8emb); enables the GPU, memory and h_sel additions")
    ap.add_argument("--cache", default="", help="dir for the h_sel npz (default cache/r3 for 0.8B, cache/r4b for 4B)")
    ap.add_argument("--allow-float-table", action="store_true",
                    help="GPU: run even if the EMBEDDING_LOOKUP table is not INT8 (V4 probe; run it in a subprocess)")
    add_model_arg(ap)
    a = ap.parse_args()
    assert a.model == MODEL
    L = a.L
    r3 = bool(a.variant) or MODEL != "0.8b"   # additions (memory, non-finite, h_sel cache, GPU vs CPU)
    cache = (K / a.cache) if a.cache else cache_dir("cache/r3")
    path = Path(a.tflite) if Path(a.tflite).is_absolute() else K / a.tflite
    if a.accel == "gpu":
        assert a.variant, "GPU runs need --variant"
        tables = embedding_table_dtypes(path)
        if not a.allow_float_table:
            assert tables == ["INT8"], f"refusing GPU: EMBEDDING_LOOKUP table dtypes {tables} (must be one INT8 table)"
    tag = "cpu" if a.accel == "cpu" else ("gpu_f32" if a.f32 else "gpu_f16")
    sfx = RESULT_SUFFIX + (f"_{a.variant}" if a.variant else "") + (f"_smoke{a.limit}" if a.limit else "")
    out = K / f"results/litert_{tag}_parity_L{L}{sfx}.json"
    out_rows = K / f"results/litert_{tag}_rows_L{L}{sfx}.json"
    hsel_out = cache / f"hsel_{tag}_L{L}{sfx}.npz" if r3 else None
    assert not out.exists() and not out_rows.exists(), f"refusing to overwrite {out.name}"
    clock = Clock()
    started_at = clock.stamp()
    from ai_edge_litert.compiled_model import CompiledModel, CpuOptions, GpuOptions, HardwareAccelerator, Options
    import importlib.metadata
    if a.accel == "gpu":
        gpu_opts = GpuOptions(enforce_f32=a.f32)
        opts = Options(hardware_accelerators=HardwareAccelerator.GPU, gpu_options=gpu_opts)
        accel_desc = {"accelerator": "GPU (Metal)", "gpu_options": gpu_opts._as_flat_kwargs(),
                      "activations": "fp32 (enforce_f32=True)" if a.f32 else "fp16 (default, enforce_f32=False)"}
    else:
        opts = Options(hardware_accelerators=HardwareAccelerator.CPU, cpu_options=CpuOptions(num_threads=a.threads))
        accel_desc = {"accelerator": "CPU", "threads": a.threads}
    mem_before = memory()
    t0 = time.perf_counter()
    model = CompiledModel.from_file(str(path), options=opts)
    compile_s = time.perf_counter() - t0
    mem_compiled = memory()
    print(f"compiled in {compile_s:.1f}s ({tag})", flush=True)
    try:
        fully = bool(model.is_fully_accelerated())
    except Exception as e:   # informational
        fully = f"unavailable: {type(e).__name__}"
    print(f"is_fully_accelerated: {fully}", flush=True)
    sigs = model.get_signature_list()
    assert len(sigs) == 1, sigs
    key = next(iter(sigs))
    in_det, out_det = model.get_input_tensor_details(key), model.get_output_tensor_details(key)
    assert sorted(in_det) == ["ids", "valid"] and list(out_det) == ["hidden"], (in_det, out_det)
    ins = {n: model.create_input_buffer_by_name(key, n) for n in in_det}
    outs = {n: model.create_output_buffer_by_name(key, n) for n in out_det}
    oracle = load_oracle()
    ref = np.load(ORACLE_NPZ)
    tg_path = cache_dir("cache/r2") / f"torch_graph_hsel_L{L}{RESULT_SUFFIX}.npz"
    tg = np.load(tg_path) if tg_path.exists() else None
    head = Head()
    fits = [q for q in oracle["questions"] if q["row_len"] <= L]
    if a.limit:
        fits = fits[: a.limit]
    skipped = [{"key": qkey(q), "row_len": q["row_len"]} for q in oracle["questions"] if q["row_len"] > L]
    rows, per_row, times, run_times, full, red_probs = [], [], [], [], {}, None
    vs_torch, hsel_store, nonfinite_rows = [], {}, []
    nf_real_total = nf_all_total = 0
    for i, q in enumerate(fits):
        n = q["row_len"]
        ids = np.full((1, L), PAD_ID, dtype=np.int32)
        ids[0, :n] = np.asarray(q["row_ids"], dtype=np.int32)
        valid = np.zeros((1, L), dtype=np.float32)
        valid[0, :n] = 1.0
        t = time.perf_counter()
        ins["ids"].write(ids)
        ins["valid"].write(valid)
        tr = time.perf_counter()
        model.run_by_name(key, ins, outs)
        run_times.append((time.perf_counter() - tr) * 1000)
        h = np.asarray(outs["hidden"].read(L * HIDDEN, np.float32), dtype=np.float32).reshape(L, HIDDEN)
        times.append((time.perf_counter() - t) * 1000)
        nf_real = int((~np.isfinite(h[:n])).sum())
        nf_all = int((~np.isfinite(h)).sum())
        nf_real_total += nf_real
        nf_all_total += nf_all
        h_sel = select(h, q)
        if r3:
            hsel_store[qkey(q)] = h_sel.astype(np.float32)
        if not np.isfinite(h_sel).all():
            nonfinite_rows.append({"key": qkey(q), "source": q["source"], "row_len": n,
                                   "nonfinite_real_positions": nf_real, "nonfinite_all_positions": nf_all,
                                   "nonfinite_h_sel": int((~np.isfinite(h_sel)).sum())})
            per_row.append({"key": qkey(q), "source": q["source"], "keys": q["keys"], "probs": None,
                            "probs_oracle": q["probs"], "argmax_key": None, "argmax_key_oracle": q["argmax_key"],
                            "finite_h_sel": False, "nonfinite_real_positions": nf_real,
                            "nonfinite_all_positions": nf_all, "row_len": n, "ms": round(times[-1], 1)})
            if q["id"] == "red_arm_000":
                red_probs = None
            continue
        z_pre, z_post, probs = head(h_sel)
        row = compare_question(q, probs, z_post, h_sel, ref[qkey(q)])
        row.update(finite_real=nf_real == 0, finite_all_positions=nf_all == 0, ms=round(times[-1], 1))
        if r3:
            row.update(nonfinite_real_positions=nf_real, nonfinite_all_positions=nf_all)
        if tg is not None and qkey(q) in tg.files:
            row["h_sel_max_abs_vs_torch_graph"] = float(np.abs(h_sel.astype(np.float64) - tg[qkey(q)]).max())
            vs_torch.append(row["h_sel_max_abs_vs_torch_graph"])
        rows.append(row)
        per_row.append({"key": qkey(q), "source": q["source"], "keys": q["keys"], "probs": row["probs"],
                        "probs_oracle": q["probs"], "z_post": [float(x) for x in z_post], "z_pre": [float(x) for x in z_pre],
                        "argmax_key": row["argmax_key"], "argmax_key_oracle": q["argmax_key"], "max_abs_dp": row["max_abs_dp"],
                        "h_sel_max_abs": row["h_sel_max_abs"], "row_len": n, "ms": row["ms"], "finite_h_sel": True,
                        "nonfinite_real_positions": nf_real, "nonfinite_all_positions": nf_all})
        if qkey(q) + "/full" in ref.files:
            full[qkey(q)] = float(np.abs(h[:n].astype(np.float64) - ref[qkey(q) + "/full"]).max())
        if q["id"] == "red_arm_000":
            red_probs = probs
        if i == 0 or (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(fits)} {qkey(q)} max|dp| {row['max_abs_dp']:.2e} ms {times[-1]:.0f}", flush=True)
    mem_end = memory()
    for b in list(ins.values()) + list(outs.values()):
        try:
            b.destroy()
        except Exception:
            pass
    model.close()
    summary = summarize(rows, red_probs, oracle)
    summary["full_rows_max_abs_hidden_vs_oracle_real_positions"] = full
    summary["full_rows_max_abs_hidden_max"] = max(full.values()) if full else None
    if r3:
        near_keys = {r["key"] for r in summary["near_tie"]["rows"]}
        summary["questions_total"] = len(fits)
        summary["nonfinite_h_sel_questions"] = len(nonfinite_rows)
        summary["near_tie"]["flips"] = sum(1 for r in summary["near_tie"]["rows"] if not r["argmax_equal"])
        summary["near_tie"]["nonfinite"] = sum(1 for r in nonfinite_rows if r["key"] in near_keys)
        summary["flip_count"] = len(summary["flips"])
        if nonfinite_rows:   # a question without a finite answer fails the bar
            summary["bar_pass"] = False
            summary["bar_note"] = f"{len(nonfinite_rows)} questions with non-finite h_sel (excluded from |dp| stats)"
        if red_probs is None and any(q["id"] == "red_arm_000" for q in fits):
            summary["red_arm"] = {"graph": "red_arm_000", "vs_oracle": "tv4_000", "max_abs_dp": None,
                                  "note": "red_arm_000 output non-finite"}
    doc = {
        "step": (f"desktop gate: Mac {accel_desc['accelerator']} parity, {a.variant}, L={L}" if a.variant else
                 f"desktop gate: Mac CPU parity, CompiledModel CPU {a.threads} threads, L={L}")
                + (f" (model {MODEL})" if RESULT_SUFFIX else ""), "L": L, "model": MODEL,
        "started_at": started_at, "seconds_wall": clock.seconds(), "tflite": str(path.relative_to(K)),
        "tflite_bytes": path.stat().st_size,
        "runtime": {"ai_edge_litert": importlib.metadata.version("ai-edge-litert"), "api": "CompiledModel.from_file",
                    **accel_desc, "signature": key,
                    "inputs": {n: {k: str(v) for k, v in d.items()} for n, d in in_det.items()},
                    "outputs": {n: {k: str(v) for k, v in d.items()} for n, d in out_det.items()},
                    "is_fully_accelerated": fully},
        "questions_run": len(fits), "questions_skipped_longer_than_L": skipped,
        "all_real_positions_finite": nf_real_total == 0,
        "all_positions_finite": nf_all_total == 0,
        "h_sel_max_abs_vs_torch_graph": max(vs_torch) if vs_torch else None,
        "timing_condition": "contended (other work shares this Mac); informational, not a card number",
        "compile_seconds": round(compile_s, 2),
        "row_ms_write_run_read": {"median": float(np.median(times)), "min": float(np.min(times)), "max": float(np.max(times)),
                                  "first": float(times[0])},
        "row_ms_run_only": {"median": float(np.median(run_times)), "min": float(np.min(run_times)),
                            "max": float(np.max(run_times))},
        "summary": summary,
        "rows": [{k: v for k, v in r.items() if k not in ("dp", "probs")} for r in rows],
        "rows_file": str(out_rows.relative_to(K)),
    }
    if r3:
        doc["tflite_embedding_table_dtypes"] = embedding_table_dtypes(path)
        doc["allow_float_table"] = bool(a.allow_float_table)
        doc["nonfinite"] = {"real_positions_total": nf_real_total, "all_positions_total": nf_all_total,
                            "questions_with_nonfinite_real": sum(1 for r in per_row if r["nonfinite_real_positions"]),
                            "questions_with_nonfinite_any_position": sum(1 for r in per_row if r["nonfinite_all_positions"]),
                            "questions_with_nonfinite_h_sel": nonfinite_rows}
        doc["memory_bytes"] = {"before_compile": mem_before, "after_compile": mem_compiled, "end": mem_end}
        hsel_out.parent.mkdir(parents=True, exist_ok=True)
        assert not hsel_out.exists(), f"refusing to overwrite {hsel_out}"
        np.savez(hsel_out, **hsel_store)
        doc["hsel_cache"] = str(hsel_out.relative_to(K))
        if a.accel == "gpu":
            cpu_rows = K / f"results/litert_cpu_rows_L{L}{RESULT_SUFFIX}_{a.variant}.json"
            cpu_hsel = cache / f"hsel_cpu_L{L}{RESULT_SUFFIX}_{a.variant}.npz"
            doc["gpu_vs_cpu_same_file"] = gpu_vs_cpu(per_row, cpu_rows, hsel_store, cpu_hsel)
    dump_json(out, doc)
    dump_json(out_rows, {"L": L, "tflite": str(path.relative_to(K)), "accel": tag, "rows": per_row})
    print(json.dumps({k: v for k, v in doc.items() if k not in ("rows", "summary", "runtime", "gpu_vs_cpu_same_file",
                                                                "questions_skipped_longer_than_L")}, indent=1, default=str))
    print(json.dumps({"overall": summary["overall"], "bar_pass": summary["bar_pass"], "red_arm": summary.get("red_arm"),
                      "flips": summary["flips"], "near_tie_flips": summary["near_tie"].get("flips"),
                      "nonfinite_h_sel_questions": summary.get("nonfinite_h_sel_questions"), "full": full}, indent=1))
    if "gpu_vs_cpu_same_file" in doc:
        print(json.dumps({k: v for k, v in doc["gpu_vs_cpu_same_file"].items() if k != "top10_by_dp"}, indent=1))


if __name__ == "__main__":
    main()
