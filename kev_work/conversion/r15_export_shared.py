"""Export the Kev-4B shared-state pair (r15_shared_state.py, an r15_form form installed) as ONE fp32 file with two
signatures that share the weights (state_prefill_<Ls>, question_step_<Ls>_<Lq>), then its V2 storage form; the fp32 file
is deleted after the V2 checks. r14_export_shared.py with --model, the memory record and the 4B contract.

    python r15_export_shared.py --model 4b --Ls 128 --Lq 64 --form R64+sp+ec+dd+vs6+in1+fn5 [--keep-fp32]

1. Guards (required): results/r15_shared_torch_parity<_4b>_<form>.json pass (pair vs row form, 402 questions:
   r15_shared_torch_parity.py) and results/r15_torch_parity<_4b>_<form>.json pass (the form vs the loop kernel:
   r15_torch_parity.py). The two modules being exported are re-checked on own_ticket_01 (3 questions) and tv4_000
   against the oracle (argmax equal, |dp| <= 1e-4).
2. litert_torch.signature("state_prefill_<Ls>", StatePrefill, {"ids", "valid"})
       .signature("question_step_<Ls>_<Lq>", QuestionStep, {"ids", "valid", "state_valid", <state tensors>})
       .convert().export(exports/<prefix>_sharedstate_Ls<Ls>_Lq<Lq>_fp32_<tag>.tflite)   (tag = r15 + form with -)
   static scan -> results/opscan_sharedstate_Ls<Ls>_Lq<Lq><_4b>_<tag>.json; record -> results/export_sharedstate_Ls<Ls>
   _Lq<Lq><_4b>_<tag>.json: signatures vs the contract, op histogram, stop checks (CUSTOM / rank > 4 / GATHER / a
   forbidden op other than the exp clamp's MAXIMUM), seconds, memory (proc_pid_rusage).
3. V2 = quantize_shared_state.build_recipe_v2_keep_rope (fp16 FC by FLOAT_CASTING, int8 table, the RopeSelect FC kept
   fp32) -> exports/<prefix>_sharedstate_Ls<Ls>_Lq<Lq>_v2_fp16fc_i8emb_<tag>.tflite + results/quant_sharedstate_...json
   with quantize_shared_state.py's checks. Never overwrites."""
import argparse
import importlib.metadata
import json
import os
import resource
import time
import traceback

import numpy as np
import torch

from r2_common import (FILE_PREFIX, MODEL, PAD_ID, RESULT_SUFFIX, K, Clock, Head, add_model_arg, dump_json, load_oracle,
                       qkey, select, sha256_file)
from r3_common import memory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ls", type=int, required=True)
    ap.add_argument("--Lq", type=int, required=True)
    ap.add_argument("--form", required=True)
    ap.add_argument("--keep-fp32", action="store_true")
    add_model_arg(ap)
    a = ap.parse_args()
    assert a.model == MODEL
    Ls, Lq = a.Ls, a.Lq
    ftag = a.form.replace("+", "-")
    tag = f"r15{ftag}"
    base = f"Ls{Ls}_Lq{Lq}"
    path = K / f"exports/{FILE_PREFIX}_sharedstate_{base}_fp32_{tag}.tflite"
    v2 = K / f"exports/{FILE_PREFIX}_sharedstate_{base}_v2_fp16fc_i8emb_{tag}.tflite"
    out = K / f"results/export_sharedstate_{base}{RESULT_SUFFIX}_{tag}.json"
    scan_out = K / f"results/opscan_sharedstate_{base}{RESULT_SUFFIX}_{tag}.json"
    q_out = K / f"results/quant_sharedstate_{base}{RESULT_SUFFIX}_v2_fp16fc_i8emb_{tag}.json"
    for p in (path, v2, out, scan_out, q_out):
        assert not p.exists(), f"never overwrite {p}"
    g_pair = json.loads((K / f"results/r15_shared_torch_parity{RESULT_SUFFIX}_{ftag}.json").read_text())
    g_row = json.loads((K / f"results/r15_torch_parity{RESULT_SUFFIX}_{ftag}.json").read_text())
    assert g_pair["summary"]["pass"] and g_pair["form"] == a.form, "pair torch parity did not pass: do not export"
    assert g_row["pass"] and g_row["form"] == a.form, "row-form torch invariance did not pass: do not export"
    clock = Clock()
    started = clock.stamp()
    torch.set_num_threads(4)
    import r15_form as R   # r13_kernel's tokens + fn<k> / in<k> (norm pre-scales)
    from kev_graph import load_text_model
    from r15_shared_state import QuestionStep, StatePrefill, contract, state_inputs
    model, load_info = load_text_model()
    applied = R.apply(model, a.form)
    sp = StatePrefill(model, Ls).eval().requires_grad_(False)
    qs = QuestionStep(model, Ls, Lq).eval().requires_grad_(False)
    oracle = load_oracle()
    reqs = {r["id"]: r for r in oracle["requests"]}
    head = Head()
    checks, sample = [], None
    with torch.no_grad():
        for rid in ("own_ticket_01", "tv4_000"):
            n = reqs[rid]["state_tokens"]
            rows = [q for q in oracle["questions"] if q["id"] == rid]
            s_ids, s_valid = state_inputs(rows[0]["row_ids"][:n], Ls, PAD_ID)
            st = sp(s_ids, s_valid)
            for q in rows:
                q_ids, q_valid = state_inputs(q["row_ids"][n:], Lq, PAD_ID)
                kwargs = {"ids": q_ids, "valid": q_valid, "state_valid": s_valid, **st}
                h = qs(**kwargs)["hidden"][0].numpy()
                rel = dict(q, decide_idx=q["decide_idx"] - n, opt_idx=[o - n for o in q["opt_idx"]])
                _, _, p = head(select(h, rel))
                checks.append({"key": qkey(q), "n_state": n, "n_question": q["row_len"] - n,
                               "max_abs_dp": float(np.abs(p - np.asarray(q["probs"])).max()),
                               "argmax_equal": q["keys"][int(np.argmax(p))] == q["argmax_key"]})
                if sample is None:
                    sample = ({"ids": s_ids, "valid": s_valid}, {k: v.clone() for k, v in kwargs.items()}, qkey(q))
    assert all(c["argmax_equal"] and c["max_abs_dp"] <= 1e-4 for c in checks), checks
    names_q = qs.input_names()
    assert list(sample[1]) == names_q, (list(sample[1])[:5], names_q[:5])
    doc = contract(model.config, Ls, Lq, a.form)
    record = {"Ls": Ls, "Lq": Lq, "model": MODEL, "form": a.form, "tag": tag, "status": "RUNNING", "started_at": started,
              "file": str(path.relative_to(K)), "applied": {k: v for k, v in applied.items() if k != "tokens"},
              "guard": {"pair": {"file": f"results/r15_shared_torch_parity{RESULT_SUFFIX}_{ftag}.json",
                                 "pass": True, "all": {k: v for k, v in g_pair["summary"]["all"].items()
                                                       if k != "vs_stock_loop_row"}},
                        "row": {"file": f"results/r15_torch_parity{RESULT_SUFFIX}_{ftag}.json", "pass": True}},
              "probe_checks": checks, "load_info": load_info, "sample_row": sample[2], "tmpdir": os.environ.get("TMPDIR"),
              "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "litert-torch",
                                                                      "litert-converter", "ai-edge-litert",
                                                                      "ai-edge-quantizer")},
              "contract": doc, "memory_before_convert": memory()}
    t0 = time.perf_counter()
    try:
        import litert_torch
        lrt = (litert_torch.signature(f"state_prefill_{Ls}", sp, sample_kwargs=sample[0])
               .signature(f"question_step_{Ls}_{Lq}", qs, sample_kwargs=sample[1]).convert())
        record["convert_seconds"] = round(time.perf_counter() - t0, 1)
        record["memory_after_convert"] = memory()
        t1 = time.perf_counter()
        lrt.export(str(path))
        record["write_seconds"] = round(time.perf_counter() - t1, 1)
        record["memory_after_write"] = memory()
        del lrt
        from tflite_scan import scan
        s = scan(path)
        dump_json(scan_out, s)
        sigs = {x["key"]: x for x in s["signatures"]}
        want = doc["signatures"]
        sig_ok = sorted(sigs) == sorted(want)
        mismatch = []
        if sig_ok:
            for key, w in want.items():
                for side in ("inputs", "outputs"):
                    got = {e["name"]: (e["shape"], e["dtype"]) for e in sigs[key][side]}
                    exp = {e["name"]: (e["shape"], e["dtype"].upper()) for e in w[side]}
                    if got != exp:
                        mismatch.append({"signature": key, "side": side,
                                         "missing": sorted(set(exp) - set(got)), "extra": sorted(set(got) - set(exp)),
                                         "different": sorted(k for k in set(exp) & set(got) if exp[k] != got[k])})
        expclamp = applied["expclamp"]
        forbidden = {k: v for k, v in s["forbidden_counts"].items() if v and not (expclamp and k == "MAXIMUM")}
        stop = []
        if s["custom_op_count"]:
            stop.append("CUSTOM")
        if s["rank_gt4_tensor_count"]:
            stop.append("RANK>4")
        if s["gather_count"] or s["forbidden_counts"].get("GATHER_ND"):
            stop.append("GATHER")
        if forbidden:
            stop.append("FORBIDDEN")
        if s["int64_tensor_count"]:
            stop.append("INT64")
        record.update(
            status="EXPORTED" if not stop else "STOP:" + "+".join(stop),
            bytes=s["bytes"], sha256=s["sha256"], subgraphs=s["subgraphs"], signatures=s["signatures"],
            signature_matches_contract=bool(sig_ok and not mismatch), signature_mismatch=mismatch,
            operator_count=s["operator_count"], op_histogram=s["op_histogram"],
            tensor_rank_histogram=s["tensor_rank_histogram"], int64_tensor_count=s["int64_tensor_count"],
            rank_gt4_tensor_count=s["rank_gt4_tensor_count"], forbidden_counts=s["forbidden_counts"],
            pad_count=s["pad_count"], batch_matmul_count=s["batch_matmul_count"],
            batch_matmul_constant_left=s["batch_matmul_constant_left"],
            fully_connected_count=s["fully_connected_count"], embedding_lookup_count=s["embedding_lookup_count"],
            gather_count=s["gather_count"], custom_op_count=s["custom_op_count"], stablehlo_ops=s["stablehlo_ops"],
            opscan=str(scan_out.relative_to(K)))
    except BaseException:
        (K / f"logs/export_sharedstate_{base}{RESULT_SUFFIX}_{tag}.traceback.txt").write_text(traceback.format_exc())
        record.update(status="FAIL", seconds=round(time.perf_counter() - t0, 1))
        dump_json(out, record)
        raise
    record["peak_rss_bytes_getrusage"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record["seconds_wall"] = clock.seconds()
    dump_json(out, record)
    print(json.dumps({k: v for k, v in record.items() if k in (
        "status", "bytes", "convert_seconds", "write_seconds", "operator_count", "signature_matches_contract",
        "fully_connected_count", "batch_matmul_constant_left", "subgraphs")}, indent=1, default=str), flush=True)
    print(json.dumps({"lifetime_max_phys_footprint_gb": round(record["memory_after_write"].get(
        "lifetime_max_phys_footprint", 0) / 1e9, 2)}), flush=True)
    assert record["status"] == "EXPORTED" and record["signature_matches_contract"], record["status"]
    del sp, qs, model
    import gc
    gc.collect()
    # ---- V2 (quantize_shared_state.py's recipe and checks) ----
    from quantize_kev import EXPECT, quant_scan
    from quantize_shared_state import ROPE_SCOPE, build_recipe_v2_keep_rope, fc_weights
    from ai_edge_quantizer import quantizer
    recipe, need_cal = build_recipe_v2_keep_rope()
    t2 = time.perf_counter()
    qt = quantizer.Quantizer(str(path), recipe)
    assert not (qt.need_calibration or need_cal)
    qt.quantize().export_model(str(v2))
    q_s = time.perf_counter() - t2
    del qt
    scan_q, scan_src = quant_scan(v2), quant_scan(path)
    exp = EXPECT["v2"]
    emb = scan_q["embedding_lookup"]
    n_fc = scan_src["fully_connected"]
    fcs = fc_weights(v2)
    rope = [(n, d) for n, d in fcs if ROPE_SCOPE in n]
    other = [(n, d) for n, d in fcs if ROPE_SCOPE not in n]
    row_v2 = K / f"exports/{FILE_PREFIX}_rowprefill_L{Ls}_v2_fp16fc_i8emb_{tag}.tflite"
    qchecks = {
        f"fc_count_unchanged_{n_fc}": scan_q["fully_connected"] == n_fc,
        "fc_weight_all_" + exp["fc_weight"] + "_except_rope": bool(other) and all(d == exp["fc_weight"] for _, d in other),
        "rope_fc_FLOAT32": bool(rope) and all(d == "FLOAT32" for _, d in rope),
        "embedding_lookup_tables_" + exp["table"]: bool(emb) and all(e["table_source_dtype"] == exp["table"] for e in emb),
        "batch_matmul_constants_FLOAT32_only": set(scan_q["batch_matmul_constant_operand_dtype"]) <= {"FLOAT32"},
        "custom_0": scan_q["custom_op_count"] == 0,
        "op_histogram_unchanged_except_DEQUANTIZE": {k: v for k, v in scan_q["op_histogram"].items() if k != "DEQUANTIZE"}
        == {k: v for k, v in scan_src["op_histogram"].items() if k != "DEQUANTIZE"},
    }
    qdoc = {"step": f"V2 of the shared-state pair {base} ({tag}), model {MODEL}",
            "input": {"file": str(path.relative_to(K)), "bytes": path.stat().st_size},
            "output": {"file": str(v2.relative_to(K)), "bytes": v2.stat().st_size, "sha256": sha256_file(v2)},
            "row_form_same_variant": ({"file": str(row_v2.relative_to(K)), "bytes": row_v2.stat().st_size}
                                      if row_v2.exists() else None),
            "recipe": recipe, "seconds": round(q_s, 1), "memory_after_quantize": memory(),
            "rope_fully_connected": [{"output": n, "weight_dtype": d} for n, d in rope],
            "scan": {k: scan_q[k] for k in ("operator_count", "op_histogram", "fully_connected", "fc_weight_source_dtype",
                                            "dequantize_count", "embedding_lookup", "batch_matmul_constant_operand_dtype",
                                            "constant_bytes_by_dtype")},
            "checks": qchecks, "checks_pass": all(qchecks.values())}
    dump_json(q_out, qdoc)
    print(json.dumps({k: qdoc[k] for k in ("output", "row_form_same_variant", "seconds", "checks_pass")}, indent=1),
          flush=True)
    assert qdoc["checks_pass"], qchecks
    if not a.keep_fp32:
        path.unlink()
        print(f"deleted {path.name} (rebuild: this script)", flush=True)


if __name__ == "__main__":
    main()
