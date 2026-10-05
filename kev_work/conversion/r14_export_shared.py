"""Export the Kev-0.8B shared-state pair (r14_shared_state.py) with an r13_kernel form as ONE fp32 file with two
signatures (state_prefill_<Ls>, question_step_<Ls>_<Lq>), make its V2 storage form at once and delete the fp32 file.

    python r14_export_shared.py --Ls 128 --Lq 64 --form R64+sp+ec+dd+vs6 --tag r14B-vs6

The published pairs are this script's output for --Ls 128 and --Ls 256 with --Lq 64 --form R64+sp+ec+dd+vs6
--tag r14B-vs6, renamed (conversion/README.md).
1. Guard: results/r14_shared_torch_parity_<form>.json must pass (pair vs row form, same form, 402 questions:
   r14_shared_torch_parity.py). The two modules are re-checked on own_ticket_01 (3 questions) and tv4_000 against the
   oracle (argmax equal, |dp| <= 1e-4).
2. litert_torch.signature("state_prefill_<Ls>", ...).signature("question_step_<Ls>_<Lq>", ...).convert().export(
   exports/kev08b_sharedstate_Ls<Ls>_Lq<Lq>_fp32_<tag>.tflite) -> static scan (results/opscan_sharedstate_Ls<Ls>_Lq<Lq>_
   <tag>.json) -> record results/export_sharedstate_Ls<Ls>_Lq<Lq>_<tag>.json (signatures vs the contract, op histogram
   per file, stop statuses: CUSTOM / rank > 4 / GATHER / forbidden other than the exp clamp's MAXIMUM).
3. V2 = fp16 FLOAT_CASTING FC weights + dynamic int8 embedding table, RopeSelect FC kept fp32
   (quantize_shared_state.build_recipe_v2_keep_rope) -> exports/kev08b_sharedstate_Ls<Ls>_Lq<Lq>_v2_fp16fc_i8emb_<tag>
   .tflite + results/quant_sharedstate_Ls<Ls>_Lq<Lq>_v2_fp16fc_i8emb_<tag>.json with quantize_shared_state.py's checks;
   then the fp32 file is deleted (unless --keep-fp32).
Never overwrites."""
import argparse
import importlib.metadata
import json
import resource
import time
import traceback

import numpy as np
import torch

from r2_common import PAD_ID, K, Clock, Head, dump_json, load_oracle, qkey, select, sha256_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ls", type=int, required=True)
    ap.add_argument("--Lq", type=int, required=True)
    ap.add_argument("--form", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--keep-fp32", action="store_true")
    a = ap.parse_args()
    Ls, Lq, tag = a.Ls, a.Lq, a.tag
    base = f"Ls{Ls}_Lq{Lq}"
    path = K / f"exports/kev08b_sharedstate_{base}_fp32_{tag}.tflite"
    v2 = K / f"exports/kev08b_sharedstate_{base}_v2_fp16fc_i8emb_{tag}.tflite"
    out = K / f"results/export_sharedstate_{base}_{tag}.json"
    scan_out = K / f"results/opscan_sharedstate_{base}_{tag}.json"
    q_out = K / f"results/quant_sharedstate_{base}_v2_fp16fc_i8emb_{tag}.json"
    for p in (path, v2, out, scan_out, q_out):
        assert not p.exists(), f"never overwrite {p}"
    guard_path = K / f"results/r14_shared_torch_parity_{a.form.replace('+', '-')}.json"
    tp = json.loads(guard_path.read_text())
    assert tp["summary"]["pass"], f"{guard_path.name} did not pass: do not export"
    clock = Clock()
    started = clock.stamp()
    torch.set_num_threads(4)
    import r13_kernel as R
    from kev_graph import load_text_model
    from r14_shared_state import QuestionStep, StatePrefill, contract, state_inputs
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
    doc = contract(Ls, Lq, sp.gdn_layers, sp.attn_layers)
    record = {"Ls": Ls, "Lq": Lq, "form": a.form, "tag": tag, "status": "RUNNING", "started_at": started,
              "applied": {k: v for k, v in applied.items() if k != "tokens"}, "file": str(path.relative_to(K)),
              "guard": {"file": str(guard_path.relative_to(K)), "pass": True, "all": tp["summary"]["all"]},
              "probe_checks": checks, "load_info": load_info, "sample_row": sample[2],
              "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "litert-torch",
                                                                      "litert-converter", "ai-edge-litert",
                                                                      "ai-edge-quantizer")},
              "contract": doc}
    t0 = time.perf_counter()
    try:
        import litert_torch
        lrt = (litert_torch.signature(f"state_prefill_{Ls}", sp, sample_kwargs=sample[0])
               .signature(f"question_step_{Ls}_{Lq}", qs, sample_kwargs=sample[1]).convert())
        record["convert_seconds"] = round(time.perf_counter() - t0, 1)
        t1 = time.perf_counter()
        lrt.export(str(path))
        record["write_seconds"] = round(time.perf_counter() - t1, 1)
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
        stop = []
        if s["custom_op_count"]:
            stop.append("CUSTOM")
        if s["rank_gt4_tensor_count"]:
            stop.append("RANK>4")
        if s["gather_count"] or s["forbidden_counts"].get("GATHER_ND"):
            stop.append("GATHER")
        forbidden = {k: v for k, v in s["forbidden_counts"].items() if v and not (applied["expclamp"] and k == "MAXIMUM")}
        if forbidden:
            stop.append("FORBIDDEN")
        if s["stablehlo_ops"]:
            stop.append("STABLEHLO")
        if s["int64_tensor_count"]:
            stop.append("INT64")
        record.update(
            status="EXPORTED" if not stop else "STOP:" + "+".join(stop),
            bytes=s["bytes"], sha256=s["sha256"], subgraphs=s["subgraphs"], signatures=s["signatures"],
            signature_matches_contract=bool(sig_ok and not mismatch), signature_mismatch=mismatch,
            operator_count=s["operator_count"], op_histogram=s["op_histogram"],
            tensor_rank_histogram=s["tensor_rank_histogram"], tensor_dtype_histogram=s["tensor_dtype_histogram"],
            int64_tensor_count=s["int64_tensor_count"], rank_gt4_tensor_count=s["rank_gt4_tensor_count"],
            forbidden_counts=s["forbidden_counts"], forbidden_total=s["forbidden_total"], forbidden_stop=forbidden,
            pad_count=s["pad_count"], pad_summary=s["pad_summary"],
            batch_matmul_count=s["batch_matmul_count"], batch_matmul_constant_left=s["batch_matmul_constant_left"],
            fully_connected_count=s["fully_connected_count"], embedding_lookup_count=s["embedding_lookup_count"],
            gather_count=s["gather_count"], custom_op_count=s["custom_op_count"], stablehlo_ops=s["stablehlo_ops"],
            opscan=str(scan_out.relative_to(K)))
        r10 = K / f"results/export_sharedstate_{base}.json"
        if r10.exists():
            o = json.loads(r10.read_text())
            record["delta_vs_round10_loop_pair"] = {
                "vs": str(r10.relative_to(K)), "operator_count": s["operator_count"] - o["operator_count"],
                "op_histogram": {k: s["op_histogram"].get(k, 0) - o["op_histogram"].get(k, 0)
                                 for k in sorted(set(s["op_histogram"]) | set(o["op_histogram"]))
                                 if s["op_histogram"].get(k, 0) != o["op_histogram"].get(k, 0)}}
    except BaseException:
        (K / f"logs/export_sharedstate_{base}_{tag}.traceback.txt").write_text(traceback.format_exc())
        record.update(status="FAIL", seconds=round(time.perf_counter() - t0, 1))
        dump_json(out, record)
        raise
    record["peak_rss_bytes_getrusage"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record["seconds_wall_export"] = clock.seconds()
    dump_json(out, record)
    print(json.dumps({k: v for k, v in record.items() if k not in ("op_histogram", "contract", "load_info", "signatures",
                                                                    "tensor_rank_histogram", "tensor_dtype_histogram",
                                                                    "pad_summary", "guard")}, indent=1), flush=True)
    assert record["status"] == "EXPORTED" and record["signature_matches_contract"], record["status"]
    # V2 (quantize_shared_state.py's recipe and checks)
    from quantize_kev import EXPECT, quant_scan
    from quantize_shared_state import ROPE_SCOPE, build_recipe_v2_keep_rope, fc_weights
    from ai_edge_quantizer import quantizer
    recipe, need_cal = build_recipe_v2_keep_rope()
    t2 = time.perf_counter()
    qt = quantizer.Quantizer(str(path), recipe)
    assert not (qt.need_calibration or need_cal)
    qt.quantize().export_model(str(v2))
    q_s = time.perf_counter() - t2
    scan_q, scan_src = quant_scan(v2), quant_scan(path)
    exp = EXPECT["v2"]
    emb = scan_q["embedding_lookup"]
    fcs = fc_weights(v2)
    rope = [(n, d) for n, d in fcs if ROPE_SCOPE in n]
    other = [(n, d) for n, d in fcs if ROPE_SCOPE not in n]
    checks_q = {
        f"fc_count_unchanged_{scan_src['fully_connected']}": scan_q["fully_connected"] == scan_src["fully_connected"],
        "fc_weight_all_" + exp["fc_weight"] + "_except_rope": bool(other) and all(d == exp["fc_weight"] for _, d in other),
        "rope_fc_FLOAT32": bool(rope) and all(d == "FLOAT32" for _, d in rope),
        "embedding_lookup_tables_" + exp["table"]: bool(emb) and all(e["table_source_dtype"] == exp["table"] for e in emb),
        "batch_matmul_constants_FLOAT32_only": set(scan_q["batch_matmul_constant_operand_dtype"]) <= {"FLOAT32"},
        "custom_0": scan_q["custom_op_count"] == 0,
        "op_histogram_unchanged_except_DEQUANTIZE": {k: v for k, v in scan_q["op_histogram"].items() if k != "DEQUANTIZE"}
        == {k: v for k, v in scan_src["op_histogram"].items() if k != "DEQUANTIZE"},
    }
    r10v2 = K / f"exports/kev08b_sharedstate_{base}_v2_fp16fc_i8emb.tflite"
    qdoc = {"step": f"V2 of the shared-state pair {base} ({a.form}, {tag})", "input": str(path.relative_to(K)),
            "input_bytes": path.stat().st_size,
            "output": {"file": str(v2.relative_to(K)), "bytes": v2.stat().st_size, "sha256": sha256_file(v2)},
            "round10_loop_pair_v2": {"file": str(r10v2.relative_to(K)), "bytes": r10v2.stat().st_size} if r10v2.exists() else None,
            "recipe": recipe, "seconds": round(q_s, 1),
            "scan": {k: scan_q[k] for k in ("operator_count", "op_histogram", "fully_connected", "fc_weight_source_dtype",
                                            "dequantize_count", "embedding_lookup", "batch_matmul_constant_operand_dtype",
                                            "constant_bytes_by_dtype")},
            "rope_fully_connected": [{"output": n, "weight_dtype": d} for n, d in rope],
            "checks": checks_q, "checks_pass": all(checks_q.values())}
    dump_json(q_out, qdoc)
    print(json.dumps({k: qdoc[k] for k in ("output", "checks", "checks_pass")}, indent=1), flush=True)
    assert qdoc["checks_pass"], checks_q
    if not a.keep_fp32:
        path.unlink()
        print(f"deleted {path.name} (rebuild: this script)")


if __name__ == "__main__":
    main()
