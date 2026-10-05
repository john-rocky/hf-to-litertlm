"""One Kev-4B shared-state file on the Mac (ai-edge-litert 2.2.0 CompiledModel, CPU XNNPACK or Metal GPU) vs the oracle,
vs the row-form graph of the same storage variant and kernel form, and vs the torch pair: r14_litert_shared_parity.py
with --model (hidden size and the state tensor count from the checkpoint config: 4B = 64 state tensors) and --form
(the torch pair's h_sel npz of r15_shared_torch_parity.py).

    python r15_shared_parity.py --model 4b --form <form> --Ls 128 --Lq 64 --tflite <pair file> --variant <variant>
        --accel cpu|gpu [--f32] [--share]
    (the published 4B pair: --form R64+sp+ec+dd+vs6+in1+fn5 --tflite exports/kev4b_sharedstate_Ls128_Lq64_v2_fp16fc_
     i8emb_r15R64-sp-ec-dd-vs6-in1-fn5.tflite --variant v2_fp16fc_i8emb_r15R64-sp-ec-dd-vs6-in1-fn5)

Questions = every oracle question whose request state fits Ls (n <= Ls) and whose branch fits Lq (red_arm_000 included).
Per request: state_prefill_<Ls> runs once (write ids / valid, run), then the questions run in two modes:
  host    the state outputs are read back to numpy and written into question_step's input buffers (once per request,
          timed as the state round trip); per question: write ids / valid / state_valid, run, read hidden
  direct  question_step's input map holds state_prefill's output TensorBuffers themselves (no host copy)
The two modes must give the same probabilities (<= 1e-5). Readout = the oracle's head in numpy float32
(r2_common.Head) on [decide, *opts] taken at question-relative indices (row index - n).
Statistics = the desktop gate's (r2_common.compare_question / summarize; bar strict and near-tie apart as
device_compare.bar_readings) on the host-mode probabilities. Same-question comparisons, each when its rows file is
present: the row-form graph of the same variant and form on both accelerators (results/litert_{cpu,gpu_f32}_rows_
L{128,256}_4b_<variant>.json), the loop kernel's V2 rows at L512 (results/litert_gpu_f32_rows_L512_4b_v2_fp16fc_i8emb
.json), the torch pair (cache/r15/torch_two_phase_hsel_<model>_<form>.npz, h_sel), and for a GPU run the CPU run of the
same file (results/litert_shared_cpu_rows_Ls<Ls>_Lq<Lq>_4b_<v>.json). A GPU run refuses a file whose embedding tables
are not INT8. Timing is contended (other work shares the Mac): informational, never a card number.
Outputs (never overwritten): results/litert_shared_{cpu|gpu_f32|gpu_f16}[_share]_{parity,rows}_Ls<Ls>_Lq<Lq><_4b>_<variant>
.json and cache/r15/hsel_shared_{...}.npz."""
import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np

from r2_common import (CKPT, HIDDEN, MODEL, ORACLE_NPZ, PAD_ID, RESULT_SUFFIX, K, Clock, Head, add_model_arg,
                       compare_question, dump_json, load_oracle, qkey, select, summarize)
from r3_common import embedding_table_dtypes, memory


def bar_readings(rows, nonfinite_keys, near_keys, overall, n_questions):
    from device_compare import bar_readings as br
    return br(rows, nonfinite_keys, near_keys, overall, n_questions)


def padded(ids, L, dtype=np.int32):
    n = len(ids)
    assert n <= L, (n, L)
    a = np.full((1, L), PAD_ID, dtype=dtype)
    a[0, :n] = np.asarray(ids, dtype=dtype)
    v = np.zeros((1, L), dtype=np.float32)
    v[0, :n] = 1.0
    return a, v


def stats(xs):
    xs = np.asarray(xs, dtype=np.float64)
    return {"median": round(float(np.median(xs)), 3), "min": round(float(xs.min()), 3), "max": round(float(xs.max()), 3),
            "n": int(xs.size)} if xs.size else None


def compare_rows(mine, rows_path):
    """{key: probs} vs a rows json of another run -> max |dp|, argmax agreement over the common keys."""
    if not rows_path.exists():
        return {"available": False, "reason": f"missing {rows_path.name}"}
    other = {r["key"]: r for r in json.loads(rows_path.read_text())["rows"]}
    dps, agree, n = [], 0, 0
    for key, p in mine.items():
        o = other.get(key)
        if o is None or o.get("probs") is None or p is None:
            continue
        n += 1
        a, b = np.asarray(p, np.float64), np.asarray(o["probs"], np.float64)
        dps.append(float(np.abs(a - b).max()))
        agree += int(np.argmax(a) == np.argmax(b))
    if not n:
        return {"available": False, "reason": "no common questions"}
    return {"available": True, "file": rows_path.name, "questions": n, "argmax_agree": agree,
            "max_abs_dp": max(dps), "mean_question_max_abs_dp": float(np.mean(dps)),
            "p95_question_max_abs_dp": float(np.percentile(dps, 95))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", required=True)
    ap.add_argument("--Ls", type=int, required=True)
    ap.add_argument("--Lq", type=int, required=True)
    ap.add_argument("--variant", required=True, help="fp32 | v2_fp16fc_i8emb (output suffix and row-form file family)")
    ap.add_argument("--accel", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--f32", action="store_true", help="GPU: GpuOptions(enforce_f32=True)")
    ap.add_argument("--share", action="store_true",
                    help="GPU: GpuOptions(constant_tensor_sharing=True) = the two signatures share the weights on the GPU "
                         "(outputs get a _share tag)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="first N requests (smoke; outputs get a _smoke suffix)")
    ap.add_argument("--form", required=True, help="the kernel form of the file (torch pair npz name)")
    add_model_arg(ap)
    a = ap.parse_args()
    assert a.model == MODEL
    Ls, Lq = a.Ls, a.Lq
    from r15_shared_state import state_shapes
    n_state_tensors = len(state_shapes(json.loads((CKPT / "config.json").read_text()), Ls))
    path = Path(a.tflite) if Path(a.tflite).is_absolute() else K / a.tflite
    tag = "cpu" if a.accel == "cpu" else ("gpu_f32" if a.f32 else "gpu_f16") + ("_share" if a.share else "")
    sfx = f"Ls{Ls}_Lq{Lq}{RESULT_SUFFIX}_{a.variant}" + (f"_smoke{a.limit}" if a.limit else "")
    out = K / f"results/litert_shared_{tag}_parity_{sfx}.json"
    out_rows = K / f"results/litert_shared_{tag}_rows_{sfx}.json"
    hsel_out = K / f"cache/r15/hsel_shared_{tag}_{sfx}.npz"
    for p in (out, out_rows, hsel_out):
        assert not p.exists(), f"refusing to overwrite {p}"
    tables = embedding_table_dtypes(path)
    if a.accel == "gpu":
        assert tables and all(t == "INT8" for t in tables), f"refusing GPU: embedding tables {tables}"
    clock = Clock()
    started = clock.stamp()
    from ai_edge_litert.compiled_model import CompiledModel, CpuOptions, GpuOptions, HardwareAccelerator, Options
    if a.accel == "gpu":
        gpu_opts = GpuOptions(enforce_f32=a.f32, constant_tensor_sharing=a.share)
        opts = Options(hardware_accelerators=HardwareAccelerator.GPU, gpu_options=gpu_opts)
        accel_desc = {"accelerator": "GPU (Metal)", "gpu_options": gpu_opts._as_flat_kwargs(),
                      "activations": "fp32 (enforce_f32=True)" if a.f32 else "fp16 (default)"}
    else:
        opts = Options(hardware_accelerators=HardwareAccelerator.CPU, cpu_options=CpuOptions(num_threads=a.threads))
        accel_desc = {"accelerator": "CPU", "threads": a.threads}
    mem_before = memory()
    t0 = time.perf_counter()
    model = CompiledModel.from_file(str(path), options=opts)
    compile_s = time.perf_counter() - t0
    mem_compiled = memory()
    try:
        fully = bool(model.is_fully_accelerated())
    except Exception as e:  # informational
        fully = f"unavailable: {type(e).__name__}"
    print(f"compiled in {compile_s:.1f}s ({tag}); fully accelerated {fully}", flush=True)
    sig_s, sig_q = f"state_prefill_{Ls}", f"question_step_{Ls}_{Lq}"
    sigs = model.get_signature_list()
    assert set(sigs) == {sig_s, sig_q}, sigs
    state_names = list(sigs[sig_s]["outputs"])
    assert len(state_names) == n_state_tensors and sorted(sigs[sig_q]["inputs"]) == sorted(["ids", "valid", "state_valid"]
                                                                                             + state_names)
    out_det = model.get_output_tensor_details(sig_s)
    numel = {n: int(np.prod(list(out_det[n]["shape"]))) for n in state_names}
    state_bytes = 4 * sum(numel.values())
    ins_s = {n: model.create_input_buffer_by_name(sig_s, n) for n in ("ids", "valid")}
    outs_s = {n: model.create_output_buffer_by_name(sig_s, n) for n in state_names}
    ins_q = {n: model.create_input_buffer_by_name(sig_q, n) for n in sigs[sig_q]["inputs"]}
    outs_q = {"hidden": model.create_output_buffer_by_name(sig_q, "hidden")}
    direct_in = {**{n: ins_q[n] for n in ("ids", "valid", "state_valid")}, **{n: outs_s[n] for n in state_names}}
    oracle = load_oracle()
    reqs = {r["id"]: r for r in oracle["requests"]}
    ref = np.load(ORACLE_NPZ)
    tp_path = K / f"cache/r15/torch_two_phase_hsel_{MODEL}_{a.form.replace('+', '-')}.npz"
    tp = np.load(tp_path) if tp_path.exists() else None
    head = Head()
    by_req = {}
    for q in oracle["questions"]:
        n = reqs[q["id"]]["state_tokens"]
        if n <= Ls and q["row_len"] - n <= Lq:
            by_req.setdefault(q["id"], []).append(q)
    req_items = list(by_req.items())[: a.limit] if a.limit else list(by_req.items())
    rows, per_row, hsel_store, nonfinite = [], [], {}, []
    probs_host, probs_direct, red_probs = {}, {}, None
    t_state, t_roundtrip, t_read, t_write, t_q_host, t_q_direct = [], [], [], [], [], []
    req_host, req_direct = [], []
    mode_diff, vs_torch, direct_error = [], [], None
    for ri, (rid, qs) in enumerate(req_items):
        n = reqs[rid]["state_tokens"]
        s_ids, s_valid = padded(qs[0]["row_ids"][:n], Ls)
        ts = time.perf_counter()
        ins_s["ids"].write(s_ids)
        ins_s["valid"].write(s_valid)
        model.run_by_name(sig_s, ins_s, outs_s)
        t_state.append((time.perf_counter() - ts) * 1000)
        tr = time.perf_counter()
        arrays = {nm: np.asarray(outs_s[nm].read(numel[nm], np.float32), dtype=np.float32) for nm in state_names}
        tr1 = time.perf_counter()
        for nm in state_names:
            ins_q[nm].write(arrays[nm])
        ins_q["state_valid"].write(s_valid)
        tr2 = time.perf_counter()
        t_read.append((tr1 - tr) * 1000)
        t_write.append((tr2 - tr1) * 1000)
        t_roundtrip.append((tr2 - tr) * 1000)
        nonfinite_state = int(sum((~np.isfinite(v)).sum() for v in arrays.values()))
        host_total = t_state[-1] + t_roundtrip[-1]
        direct_total = t_state[-1]
        for q in qs:
            nq = q["row_len"] - n
            q_ids, q_valid = padded(q["row_ids"][n:], Lq)
            rel = dict(q, decide_idx=q["decide_idx"] - n, opt_idx=[o - n for o in q["opt_idx"]])
            hs = {}
            for mode in ("host", "direct"):
                if mode == "direct" and direct_error:
                    continue
                tq = time.perf_counter()
                ins_q["ids"].write(q_ids)
                ins_q["valid"].write(q_valid)
                try:
                    model.run_by_name(sig_q, ins_q if mode == "host" else direct_in, outs_q)
                except Exception as e:  # the direct buffer pass may be refused
                    direct_error = f"{type(e).__name__}: {e}"
                    print(f"direct mode refused: {direct_error}", flush=True)
                    continue
                h = np.asarray(outs_q["hidden"].read(Lq * HIDDEN, np.float32), dtype=np.float32).reshape(Lq, HIDDEN)
                ms = (time.perf_counter() - tq) * 1000
                (t_q_host if mode == "host" else t_q_direct).append(ms)
                if mode == "host":
                    host_total += ms
                else:
                    direct_total += ms
                hs[mode] = h
            h = hs["host"]
            key = qkey(q)
            h_sel = select(h, rel)
            hsel_store[key] = h_sel.astype(np.float32)
            nf_real = int((~np.isfinite(h[:nq])).sum())
            if not np.isfinite(h_sel).all():
                nonfinite.append({"key": key, "row_len": q["row_len"], "nonfinite_real": nf_real,
                                  "nonfinite_state_values": nonfinite_state})
                per_row.append({"key": key, "probs": None, "argmax_key": None, "finite_h_sel": False})
                continue
            z_pre, z_post, probs = head(h_sel)
            row = compare_question(q, probs, z_post, h_sel, ref[key])
            row.update(n_state=n, n_question=nq)
            if "direct" in hs:
                _, _, p_d = head(select(hs["direct"], rel))
                row["max_abs_dp_host_vs_direct"] = float(np.abs(p_d.astype(np.float64) - probs).max())
                row["hidden_bit_equal_host_vs_direct"] = bool((hs["direct"][:nq] == h[:nq]).all())
                mode_diff.append(row["max_abs_dp_host_vs_direct"])
                probs_direct[key] = [float(x) for x in p_d]
            if tp is not None and key in tp.files:
                row["h_sel_max_abs_vs_torch_pair"] = float(np.abs(h_sel.astype(np.float64) - tp[key]).max())
                vs_torch.append(row["h_sel_max_abs_vs_torch_pair"])
            rows.append(row)
            probs_host[key] = row["probs"]
            per_row.append({"key": key, "source": q["source"], "keys": q["keys"], "probs": row["probs"],
                            "probs_direct": probs_direct.get(key), "probs_oracle": q["probs"],
                            "z_post": [float(x) for x in z_post], "argmax_key": row["argmax_key"],
                            "argmax_key_oracle": q["argmax_key"], "max_abs_dp": row["max_abs_dp"],
                            "h_sel_max_abs": row["h_sel_max_abs"], "near_tie_oracle": row["near_tie_oracle"],
                            "n_state": n, "n_question": nq, "finite_h_sel": True})
            if q["id"] == "red_arm_000":
                red_probs = probs
        req_host.append(host_total)
        if not direct_error:
            req_direct.append(direct_total)
        if ri == 0 or (ri + 1) % 50 == 0:
            print(f"{ri + 1}/{len(req_items)} {rid} q {len(qs)} state {t_state[-1]:.1f} ms roundtrip "
                  f"{t_roundtrip[-1]:.1f} ms", flush=True)
    mem_end = memory()
    for b in list(ins_s.values()) + list(outs_s.values()) + list(ins_q.values()) + list(outs_q.values()):
        try:
            b.destroy()
        except Exception:
            pass
    model.close()
    summary = summarize(rows, red_probs, oracle)
    near_keys = {qkey(q) for q in oracle["questions"] if q["near_tie"]}
    n_q = len(rows) + len(nonfinite)
    summary["questions_total"] = n_q
    summary["nonfinite_h_sel_questions"] = len(nonfinite)
    summary["near_tie"]["flips"] = sum(1 for r in summary["near_tie"]["rows"] if not r["argmax_equal"])
    summary.update(bar_readings(rows, [r["key"] for r in nonfinite], near_keys, summary["overall"], n_q))
    if nonfinite:
        summary["bar_pass"] = False
    same_q = {
        f"row_form_{p}_L{L}": compare_rows(probs_host, K / f"results/litert_{p}_rows_L{L}{RESULT_SUFFIX}_{a.variant}.json")
        for p in ("cpu", "gpu_f32") for L in (128, 256)}
    same_q["loop_row_gpu_f32_L512"] = compare_rows(
        probs_host, K / f"results/litert_gpu_f32_rows_L512{RESULT_SUFFIX}_v2_fp16fc_i8emb.json")
    if a.accel == "gpu":
        same_q["shared_cpu_same_file"] = compare_rows(
            probs_host, K / f"results/litert_shared_cpu_rows_Ls{Ls}_Lq{Lq}{RESULT_SUFFIX}_{a.variant}.json")
    doc = {
        "step": f"Mac {accel_desc['accelerator']} shared-state pair, model {MODEL}, {a.variant}, Ls={Ls} Lq={Lq}",
        "model": MODEL, "form": a.form, "state_tensors": n_state_tensors,
        "started_at": started, "seconds_wall": clock.seconds(), "tflite": str(path.relative_to(K)),
        "tflite_bytes": path.stat().st_size, "embedding_table_dtypes": tables,
        "runtime": {"ai_edge_litert": importlib.metadata.version("ai-edge-litert"), "api": "CompiledModel.from_file",
                    **accel_desc, "signatures": {k: {"inputs": len(v["inputs"]), "outputs": len(v["outputs"])}
                                                  for k, v in sigs.items()},
                    "is_fully_accelerated": fully},
        "requests_run": len(req_items), "questions_run": n_q, "state_bytes_per_request": state_bytes,
        "compile_seconds": round(compile_s, 2),
        "modes": {"host": "state outputs read to numpy and written to question_step inputs once per request",
                  "direct": "state_prefill output TensorBuffers passed as question_step inputs",
                  "direct_error": direct_error,
                  "max_abs_dp_host_vs_direct": max(mode_diff) if mode_diff else None,
                  "questions_compared": len(mode_diff),
                  "hidden_bit_equal_questions": sum(1 for r in rows if r.get("hidden_bit_equal_host_vs_direct"))},
        "timing_condition": "contended (other work shares this Mac); informational, not a card number",
        "ms": {"state_prefill_write_run": stats(t_state), "state_roundtrip_read_write": stats(t_roundtrip),
               "state_read": stats(t_read), "state_write": stats(t_write),
               "question_step_host_write_run_read": stats(t_q_host),
               "question_step_direct_write_run_read": stats(t_q_direct),
               "request_host": stats(req_host), "request_direct": stats(req_direct)},
        "h_sel_max_abs_vs_torch_pair": max(vs_torch) if vs_torch else None,
        "same_questions": same_q,
        "summary": summary,
        "nonfinite": nonfinite,
        "memory_bytes": {"before_compile": mem_before, "after_compile": mem_compiled, "end": mem_end},
        "rows": [{k: v for k, v in r.items() if k not in ("dp", "probs")} for r in rows],
        "rows_file": str(out_rows.relative_to(K)), "hsel_cache": str(hsel_out.relative_to(K)),
    }
    dump_json(out, doc)
    dump_json(out_rows, {"Ls": Ls, "Lq": Lq, "tflite": str(path.relative_to(K)), "accel": tag, "rows": per_row})
    hsel_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(hsel_out, **hsel_store)
    o = summary["overall"]
    print(json.dumps({"questions": n_q, "argmax": f"{o['argmax_equal']}/{o['questions']}",
                      "non_near_tie": summary["non_near_tie_argmax"], "near_tie_flips": summary["near_tie_flips"],
                      "max_abs_dp": o["max_abs_dp"], "mean_abs_dp": o["mean_abs_dp_all_options"],
                      "red_arm": (summary.get("red_arm") or {}).get("max_abs_dp"),
                      "bar_strict": summary["bar_strict"], "bar_near_tie_apart": summary["bar_near_tie_apart"],
                      "nonfinite": len(nonfinite), "fully": fully, "compile_s": round(compile_s, 1),
                      "modes": doc["modes"], "vs_torch_pair": doc["h_sel_max_abs_vs_torch_pair"],
                      "same_questions": {k: {kk: v.get(kk) for kk in ("questions", "argmax_agree", "max_abs_dp")}
                                         for k, v in same_q.items()},
                      "ms": {k: (v or {}).get("median") for k, v in doc["ms"].items()},
                      "memory_after_compile": mem_compiled.get("phys_footprint"),
                      "memory_max": mem_end.get("lifetime_max_phys_footprint")}, indent=1))


if __name__ == "__main__":
    main()
