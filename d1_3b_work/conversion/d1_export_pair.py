"""Round 9 acceptance 2: export the shared-state pair (scripts/d1_shared_state.py) as ONE float32 file with two
signatures that share the weights, state_prefill_<Ls> and question_step_<Ls>_<Lq>, then scan it.

    scripts/d1_guarded.sh logs/r9_export_Ls256_Lq128.guard.log $EXPORT scripts/d1_export_pair.py \
        --Ls 256 --Lq 128

1. Guard: results/real_sharedstate_torch_check.json says pass (the two phases vs the row form, acceptance 1).
   --guard-decision "<who decided, when, the message>" (round 9, the ruling on stop condition 1, 2026-10-09 03:2x:
   the planned 1e-4 on the question positions' hidden is below this model's float32 rounding, the design is shown by
   float64, results/real_sharedstate_torch_fp64.json): when the check failed only on that bar, the guard is read again
   from the check's rows (cache/real/pair/torch_check_rows.json, its sha256 as recorded) with the ruling's gate: every
   run and edge case |dp| <= 1e-5, answer-slot max |dh| <= 1e-4, finite, argmax equal; the question positions' max |dh|
   is recorded and must stay <= --guard-alert-hidden (2.5e-4, the float32 noise bound of round 6b; above it the guard
   stops for a review); RowCheck, pad content, guard-off control and coverage as the check's summary.
   The ruling goes into the export json (`guard.override`).
   The two modules being exported are checked again on card_text_001 (3 questions) and own_fiveq_09 (5 questions)
   against the provider's float32 reference (argmax equal, |dp| <= 1e-4; the read-out is host/d1_litert.py's).
2. litert_torch.signature("state_prefill_<Ls>", StatePrefill, {ids, valid})
       .signature("question_step_<Ls>_<Lq>", QuestionStep, {ids, valid, state_valid, the 38 state tensors})
       .convert().export(exports/real_sharedstate_Ls<Ls>_Lq<Lq>_fp32.tflite)
   (Kev r10's call.) Sample values: card_text_001's state and its first question (the graph is static; the values do
   not shape it). Input names = the kwargs keys, output names = the returned dict keys.
3. Static scan (scripts/tflite_scan.py) -> results/real_sharedstate_opscan_Ls<Ls>_Lq<Lq>.json; the summary goes to
   results/real_sharedstate_export_Ls<Ls>_Lq<Lq>.json: the signatures against d1_shared_state.contract (names, shapes,
   dtypes), the op histogram per signature, CUSTOM / rank > 4 / GATHER / GATHER_ND / INT64 / the GPU-rejection list
   (BROADCAST_TO and the rest) = stop statuses, BATCH_MATMUL with a constant left operand (Kev r10 attempt 2: Metal
   refuses it) = stop, the FULLY_CONNECTED count and the RopeSelect FCs, bytes, sha256, seconds, peak RSS (getrusage;
   bytes on macOS).
Never overwrites. On an exception: logs/r9_export_Ls<Ls>_Lq<Lq>.traceback.txt + the json with status FAIL, and the
exception is raised again.

--contract --pairs 256:128,128:64 --min-questions N|128:64=1,256:128=3 --why "..." (round 9 acceptance 6, no export):
host/contract.json gains the `shared_state` section (the v2 pair files with bytes and sha256 re-hashed against their
quant records, the signatures, each pair's min_questions, the hand-over, the GPU options (shipped = no sharing, the
measured cost of sharing), the pick rule, the Mac gates of results/real_sharedstate_*_check.json, the Mac call times of
results/timing_mac_pair.json and the host check); every other section must stay equal (asserted before writing,
d1_contract.py --keep-vision's rule: the file is written as d1_contract.py writes it, json indent 1, ensure_ascii False).

Round 10:
--embeds --Ls <Ls> --Lq <Lq>: the embeds pair (scripts/d1_shared_state.py StatePrefillEmbeds + QuestionStep(embeds=True):
`embeds` float32 [1, L, 2048] in place of `ids`, no table in the graph) -> exports/real_sharedstate_embeds_Ls<Ls>_Lq<Lq>_
fp32.tflite, results/real_sharedstate_embeds_{export,opscan}_Ls<Ls>_Lq<Lq>.json; the guard is the embeds torch check
(results/real_sharedstate_embeds_torch_check.json: pass = the round-9 ruling's gate, the pair among its sizes); the probe
modules get the host's float32 rows of embed_table.safetensors; an EMBEDDING_LOOKUP in the file is a stop.
--contract-r10 [--host-check <json>] [--dry-run] (write_contract_r10): host/contract.json with the text path on the embeds
form (the row graphs L128..L4096, the embeds pairs, the pick by measured call time); vision and every section outside
R10_KEEP unchanged (asserted).
"""
from __future__ import annotations

import argparse
import collections
import importlib.metadata
import json
import mmap
import resource
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d1_shared_state as S  # noqa: E402
from d1_common import K, ROWS  # noqa: E402

PROBE = ("card_text_001", "own_fiveq_09")
ROPE_SCOPE = "RopeSelect"


def per_signature_ops(path: Path) -> dict:
    """{signature key: {op: count}} and the RopeSelect FULLY_CONNECTED sites, from the flatbuffer."""
    from ai_edge_litert import schema_py_generated as schema

    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    model = schema.Model.GetRootAs(mm, 0)
    names = {v: k for k, v in vars(schema.BuiltinOperator).items() if isinstance(v, int)}
    codes = [names.get(max(model.OperatorCodes(i).BuiltinCode(), model.OperatorCodes(i).DeprecatedBuiltinCode()), "?")
             for i in range(model.OperatorCodesLength())]
    sig_of = {}
    for si in range(model.SignatureDefsLength()):
        sd = model.SignatureDefs(si)
        sig_of[sd.SubgraphIndex()] = sd.SignatureKey().decode()
    out, rope = {}, []
    for gi in range(model.SubgraphsLength()):
        g = model.Subgraphs(gi)
        hist = collections.Counter()
        for oi in range(g.OperatorsLength()):
            op = g.Operators(oi)
            name = codes[op.OpcodeIndex()]
            hist[name] += 1
            if name == "FULLY_CONNECTED":
                o = g.Tensors(int(op.OutputsAsNumpy()[0])).Name().decode()
                if ROPE_SCOPE in o:
                    w = g.Tensors(int(op.InputsAsNumpy()[1]))
                    rope.append({"subgraph": sig_of.get(gi, gi), "op_index": oi, "output": o,
                                 "weight_shape": w.ShapeAsNumpy().tolist()})
        out[sig_of.get(gi, str(gi))] = {"operator_count": sum(hist.values()), "op_histogram": dict(sorted(hist.items()))}
    mm.close()
    f.close()
    return {"per_signature": out, "rope_select_fully_connected": rope}


def guard_override(guard: dict, alert_hidden: float, decision: str) -> dict:
    """--guard-decision (module docstring): the check's rows read again with the recorded ruling."""
    from d1_common import sha256_file

    rows_path = K / guard["rows_file"]
    assert sha256_file(rows_path) == guard["rows_file_sha256"], f"{rows_path.name} differs from the check's record"
    d = json.loads(rows_path.read_text())
    runs = d["rows"] + d["edge_cases"]
    s = guard["summary"]
    bar_hsel = S.BAR_HIDDEN
    bad = [{"key": r["key"], "config": r["config"], "hidden_question": r["hidden_question_max_abs_vs_row"],
            "h_sel": r["h_sel_max_abs_vs_row"], "dp": r["max_abs_dp_vs_row"]} for r in runs
           if not (r["h_sel_max_abs_vs_row"] <= bar_hsel and r["max_abs_dp_vs_row"] <= S.BAR_DP and r["finite"]
                   and r["argmax_equal_row"] and r["hidden_question_max_abs_vs_row"] <= alert_hidden)]
    sc = s["state_checks"]
    others = {"coverage": s["questions_covered"] == s["questions_wanted"],
              "rowcheck_bit_equal": s["rowcheck"]["bit_equal"] == s["rowcheck"]["rows"],
              "pad_content_bit_equal": sc["pad_content"]["bit_equal"] == sc["pad_content"]["requests"],
              "guard_off_moves": (sc["guard_off_control"]["min_conv_tail_move"] or 0) > 0}
    return {"gate": {"max_abs_dp_vs_row": S.BAR_DP, "h_sel_max_abs_vs_row": bar_hsel,
                     "hidden_question_alert": alert_hidden, "finite": True, "argmax_equal_row": True},
            "decision": decision, "rows_file": guard["rows_file"], "runs": len(runs), "runs_failing": bad[:20],
            "others": others, "max_hidden_question": max(r["hidden_question_max_abs_vs_row"] for r in runs),
            "max_h_sel": max(r["h_sel_max_abs_vs_row"] for r in runs),
            "max_dp": max(r["max_abs_dp_vs_row"] for r in runs),
            "runs_hidden_question_over_launch_1e-4": sum(r["hidden_question_max_abs_vs_row"] > 1e-4 for r in runs),
            "pass": not bad and all(others.values())}


def probe_checks(lm, sp, qs, Ls, Lq, embed_table=None):
    """The modules being exported against the reference on PROBE; returns (checks, sample kwargs, sample key). With
    embed_table (round 10, the embeds pair): the first input of both modules = the table's float32 rows at the padded
    ids (the host's input), in place of the ids."""
    sys.path.insert(0, str(K / "host"))
    import d1_litert as H

    first = "embeds" if embed_table is not None else "ids"

    def inp(ids_t):
        if embed_table is None:
            return ids_t
        return torch.from_numpy(np.ascontiguousarray(embed_table.rows(ids_t[0].numpy())[None]))

    table = H.ReadoutTable.from_file(S.TABLE)
    ref = {f"{q['id']}/{q['qid']}": q for q in json.loads(S.REFERENCE.read_text())["questions"]}
    rows = json.loads(ROWS.read_text())["rows"]
    checks, sample = [], None
    with torch.inference_mode():
        for rid in PROBE:
            rs = [r for r in rows if r["id"] == rid]
            n = rs[0]["state_len"]
            if n > Ls:      # round 10: own_fiveq_09's state (102 tokens) does not fit the Ls64 pair
                continue
            s_ids, s_valid = S.pad_inputs(rs[0]["ids"][:n], Ls)
            st = sp(inp(s_ids), s_valid)
            for r in rs:
                m = r["row_len"] - n
                if m > Lq:
                    continue
                q_ids, q_valid = S.pad_inputs(r["ids"][n:], Lq)
                kwargs = {first: inp(q_ids), "valid": q_valid, "state_valid": s_valid, **st}
                h = qs(**kwargs)["hidden"][0].numpy()
                p = H.readout(h[m - 1], table, r["readout_ids"])
                q = ref[f"{rid}/{r['qid']}"]
                checks.append({"key": f"{rid}/{r['qid']}", "n_state": n, "n_question": m,
                               "max_abs_dp_vs_reference": float(np.abs(np.asarray(p) - np.asarray(q["probs"])).max()),
                               "argmax_equal": int(np.argmax(p)) == int(q["argmax"])})
                if sample is None:
                    sample = ({first: inp(s_ids), "valid": s_valid}, {k: v.clone() for k, v in kwargs.items()},
                              f"{rid}/{r['qid']}")
    return checks, sample


def parse_min_questions(text: str) -> dict:
    """'3' -> {None: 3}; '128:64=1,256:128=3' -> {(128, 64): 1, (256, 128): 3}."""
    if "=" not in text:
        return {None: int(text)}
    out = {}
    for part in text.split(","):
        shape, n = part.split("=")
        out[tuple(int(x) for x in shape.split(":"))] = int(n)
    return out


def write_contract(a) -> int:
    """--contract (module docstring)."""
    from d1_common import sha256_file

    path = K / "host/contract.json"
    text = path.read_text()
    doc = json.loads(text)
    assert json.dumps(doc, indent=1, ensure_ascii=False) + "\n" == text, "contract.json is not in d1_contract.py's format"
    assert "shared_state" not in doc, "the shared_state section exists already"
    minq = parse_min_questions(a.min_questions)
    timing = json.loads((K / "results/timing_mac_pair.json").read_text())
    files = []
    for spec in a.pairs.split(","):
        Ls, Lq = (int(x) for x in spec.split(":"))
        tag = f"Ls{Ls}_Lq{Lq}"
        stem = f"real_sharedstate_{tag}_v2_fp16fc_i8emb"
        quant = json.loads((K / f"results/{stem}_quant.json").read_text())
        export = json.loads((K / f"results/real_sharedstate_export_{tag}.json").read_text())
        f = K / quant["output"]["file"]
        size, sha = f.stat().st_size, sha256_file(f)
        assert (size, sha) == (quant["output"]["bytes"], quant["output"]["sha256"]), f"{f.name} differs from its record"
        gates = {}
        for acc in ("cpu", "gpu_f32", "gpu_f32_share", "gpu_default_share"):
            p = K / f"results/{stem}_{acc}_check.json"
            if p.exists():
                s = json.loads(p.read_text())["summary"]
                gates[acc] = {"verdict": "PASS" if s["stop_bar_pass"] else "FAIL", "file": str(p.relative_to(K)),
                              **{k: s["line"][k] for k in ("questions_run", "non_near_tie_argmax", "near_tie_argmax",
                                                           "max_abs_dp", "mean_abs_dp_all_options", "handover_equal")},
                              "red_arms": s["line"]["red_arms"], "vs_row_form": s["vs_row_form"]}
        calls = {}
        for r in timing["rows"]:
            if r["file"] == quant["output"]["file"] and r["handover"] == "direct" and r["set"] == "three":
                key = r["accel"] + ("" if r["accel"] == "cpu" else ("_share" if r["share"] else "_noshare"))
                calls[key] = {"state_call_ms": r["state_call_median_ms"], "question_call_ms": r["question_call_median_ms"],
                              "phys_footprint_after_compile": r["phys_footprint_after_compile"]}
        files.append({"Ls": Ls, "Lq": Lq, "file": f.name, "bytes": size, "sha256": sha,
                      "form": "v2_fp16fc_i8emb (fp16 FC + int8 table; the RopeSelect FC and its RoPE table float32)",
                      "built_by": f"scripts/d1_export_pair.py --Ls {Ls} --Lq {Lq} + scripts/d1_storage.py --pair "
                                  f"--variant v2 --tflite exports/real_sharedstate_{tag}_fp32.tflite",
                      "signatures": export["contract"]["signatures"], "state": export["contract"]["state"],
                      "min_questions": minq.get((Ls, Lq), minq.get(None)),
                      "gate_mac": gates, "mac_calls_ms_median_of_20": calls})
    host = K / "results/real_sharedstate_host_check.json"
    section = {
        "status": "round 9: the shared-state pair (text requests of several questions), Mac gates and timing in "
                  "results/real_sharedstate_*_check.json and results/timing_mac_pair.json",
        "what": "one file per (Ls, Lq) with two signatures over one copy of the weights: state_prefill_<Ls> runs the "
                "state once, question_step_<Ls>_<Lq> runs one question's own tokens on top of it; the hidden states are "
                "the row graph's for state + question as one row (positions continue after the state inside the graph)",
        "files": files,
        "rows": "a request's rows are the row contract's (`row`): each question's whole row; the state = the first "
                "state_len tokens (encode(prefix text)), the same in every row; a question's own tokens = the rest of "
                "its row; its answer slot = its last token (index len(own) - 1 of question_step's hidden)",
        "padding": "right, with token_ids.pad, valid 0.0 (state_prefill: ids / valid; question_step: ids / valid, and "
                   "state_valid = the valid of the state_prefill call whose outputs are passed)",
        "handover": {"direct": "state_prefill's output buffers passed as question_step's inputs (no copy)",
                     "host": "state_prefill's outputs read back and written into question_step's inputs",
                     "equal": "byte-identical probabilities and hidden states on the Mac CPU and Metal (gate_mac)"},
        "gpu_options": {"shipped": "GpuOptions(enforce_f32=True), no constant_tensor_sharing: the faster form (gate_mac "
                                   "gpu_f32 PASS); the GPU holds the weights once per signature",
                        "sharing": "GpuOptions(enforce_f32=True, constant_tensor_sharing=True): one copy of the weights "
                                   "(about half the memory) and the same numbers (gate_mac gpu_f32_share PASS), but each "
                                   "call 1.35-2.0x slower on the Mac Metal (mac_calls_ms_median_of_20)",
                        "default_precision": "fp16 storage misses the bar (gate_mac gpu_default_share), as on the row "
                                             "graphs"},
        "pick": {"rule": "requests with pictures -> the embeds row graphs; else the smallest pair (by Ls, then Lq) "
                         "with state <= Ls, every question's own tokens <= Lq and at least its min_questions questions; "
                         "none -> the row graphs",
                 "min_questions": [{"Ls": Ls, "Lq": Lq, "min_questions": n} for (Ls, Lq), n in minq.items()]
                 if None not in minq else minq[None], "why": a.why},
        "host": "host/d1_shared_state.py (D1SharedHost over host/d1_litert.py's D1Host)",
        "host_check": json.loads(host.read_text())["summary"] if host.exists() else None,
    }
    new = dict(doc)
    new["shared_state"] = section
    assert {k: v for k, v in new.items() if k != "shared_state"} == doc
    path.write_text(json.dumps(new, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"written": str(path.relative_to(K)), "sha256": sha256_file(path),
                      "files": [(x["file"], x["bytes"], x["sha256"][:12]) for x in files]}, indent=1))
    return 0


R10_BUCKETS = (128, 256, 512, 1024, 2048, 4096)
R10_PAIRS = ((64, 64), (128, 64), (256, 128))
R10_KEEP = ("buckets", "embeds_graph", "graphs", "shared_state", "status")    # the sections round 10 may change


def r10_call_ms(timing: dict) -> dict:
    """Mac Metal fp32 call medians of the shipped embeds forms in results/timing_mac_r10.json: rows {L: ms per row call}
    (the median over the request sets of the row form), pairs {(Ls, Lq): (state call, question call)} (no sharing,
    direct hand-over)."""
    row, pair = {}, {}
    for r in timing["rows"]:
        if r["accel"] != "gpu_f32" or r.get("input") != "embeds":
            continue
        if r["form"].startswith("row"):
            row.setdefault(int(r["form"].split()[1][1:]), []).append(r["row_call_median_ms"])
        elif "(no sharing)" in r["form"] and r["handover"] == "direct":
            Ls, Lq = (int(x) for x in r["form"].split()[1].replace("Ls", "").split("+Lq"))
            pair.setdefault((Ls, Lq), []).append((r["state_call_median_ms"], r["question_call_median_ms"]))
    med = lambda xs: float(np.median(xs))
    return {"row": {L: round(med(v), 2) for L, v in sorted(row.items())},
            "pair": {k: (round(med([a for a, _ in v]), 2), round(med([b for _, b in v]), 2)) for k, v in sorted(pair.items())}}


def r10_min_questions(call: dict) -> list:
    """For readability: from how many questions each pair beats the row graphs of a bucket (rows all in that bucket)."""
    out = []
    for (Ls, Lq), (ts, tq) in call["pair"].items():
        for L, tr in call["row"].items():
            if L > 256 or L < Lq:
                continue
            n = next((n for n in range(1, 65) if ts + n * tq < n * tr), None)
            out.append({"Ls": Ls, "Lq": Lq, "rows_in": f"L{L}", "pair_faster_from_questions": n,
                        "ms": {"state": ts, "question": tq, "row": tr}})
    return out


def write_contract_r10(a) -> int:
    """--contract-r10 (round 10, design 1 / 5): host/contract.json with the text path on the embeds form. Changes only
    R10_KEEP's sections (asserted, --keep-vision's rule: `vision` and every other section unchanged): embeds_graph.buckets
    = the v2e row graphs L128..L4096 (re-hashed against their quant records, gates from the check jsons; the L256..L2048
    entries must come out equal to round 8's), the ids `graphs` marked not shipped, shared_state = the embeds pairs
    (re-hashed, signatures, state, Mac gates, Mac call medians) with round 9's ids pairs under `files_ids_not_shipped`,
    the pick by measured call time (results/timing_mac_r10.json) and, with --host-check, that check's summary;
    buckets.host_default = host/d1_litert.py BUCKETS; status gains the round-10 sentence."""
    sys.path.insert(0, str(K / "host"))
    import d1_litert as H
    from d1_common import sha256_file
    from d1_contract import ACCELS, gate_line, io_of, shipped_file

    path = K / "host/contract.json"
    text = path.read_text()
    old = json.loads(text)
    assert json.dumps(old, indent=1, ensure_ascii=False) + "\n" == text, "contract.json is not in d1_contract.py's format"
    new = json.loads(text)
    gate_keys = ("verdict", "questions", "max_abs_dp", "mean_abs_dp", "non_near_tie_argmax", "near_tie_argmax",
                 "red_arms", "check")
    buckets = []
    for L in R10_BUCKETS:
        stem = f"real_rowprefill_embeds_L{L}_v2e_fp16fc"
        f = shipped_file(stem, rehash=True)
        assert f is not None, stem
        checks = {acc: K / f"results/{stem}_{acc}_check.json" for acc in ACCELS}
        io = io_of(K / f"results/real_export_embeds_L{L}.json", checks["gpu_f32"])
        gates = {acc: gate_line(c) for acc, c in checks.items() if c.exists()}
        assert gates.get("gpu_f32", {}).get("verdict") == "PASS" and gates.get("cpu", {}).get("verdict") == "PASS", stem
        buckets.append({"L": L, "file": f["file"], "bytes": f["bytes"], "sha256": f["sha256"],
                        "form": "v2e_fp16fc (fp16 FC, no table in the graph)", "inputs": io["inputs"],
                        "outputs": io["outputs"], "record": f["record"], "sha256_rehashed": f["sha256_rehashed"],
                        "gate": {acc: {k: x[k] for k in gate_keys} for acc, x in gates.items()}})
    was = {b["L"]: b for b in old["embeds_graph"]["buckets"]}
    for b in buckets:
        if b["L"] in was:
            assert b == was[b["L"]], f"the L{b['L']} entry differs from round 8's"
    new["embeds_graph"]["buckets"] = buckets
    new["embeds_graph"]["buckets_note"] = (
        "round 8: one embeds row graph per bucket present (the host takes the smallest that holds the row, pick_L); the "
        "top-level fields are the L256 entry of round 6c. Round 10: L128 and L4096 added; these buckets are the "
        "shipped text path (text rows and picture rows, host/d1_litert.py item 10)")
    for g in new["graphs"]:
        g["shipped"] = False
        g["why_not_shipped"] = ("round 10: the text path is unified on the embeds form (embeds_graph.buckets): this ids "
                                "form keeps an int8 table inside the graph (max |dp| 7.1e-3 against 7.6e-6 for v2e on "
                                "Mac Metal fp32), is 264 MB larger per file and no faster; kept as the ladder's evidence "
                                "(host/d1_litert.py D1Host.from_dir(..., ids_graphs=True) still runs it)")
    timing = json.loads((K / "results/timing_mac_r10.json").read_text())
    call = r10_call_ms(timing)
    files = []
    for Ls, Lq in R10_PAIRS:
        tag = f"Ls{Ls}_Lq{Lq}"
        stem = f"real_sharedstate_embeds_{tag}_v2e_fp16fc"
        quant = json.loads((K / f"results/{stem}_quant.json").read_text())
        export = json.loads((K / f"results/real_sharedstate_embeds_export_{tag}.json").read_text())
        fpath = K / quant["output"]["file"]
        size, sha = fpath.stat().st_size, sha256_file(fpath)
        assert (size, sha) == (quant["output"]["bytes"], quant["output"]["sha256"]), f"{fpath.name} differs from its record"
        gates = {}
        for acc in ("cpu", "gpu_f32", "gpu_default", "gpu_f32_share"):
            c = K / f"results/{stem}_{acc}_check.json"
            if c.exists():
                sm = json.loads(c.read_text())["summary"]
                gates[acc] = {"verdict": "PASS" if sm["stop_bar_pass"] else "FAIL", "file": str(c.relative_to(K)),
                              **{k: sm["line"][k] for k in ("questions_run", "non_near_tie_argmax", "near_tie_argmax",
                                                            "max_abs_dp", "mean_abs_dp_all_options", "handover_equal")},
                              "red_arms": sm["line"]["red_arms"], "vs_row_form": sm["vs_row_form"]}
        assert gates["gpu_f32"]["verdict"] == "PASS" and gates["cpu"]["verdict"] == "PASS", stem
        calls = {}
        for r in timing["rows"]:
            if r["file"] == quant["output"]["file"] and r["handover"] == "direct":
                key = r["accel"] + ("" if r["accel"] == "cpu" else ("_share" if r["share"] else "_noshare"))
                calls.setdefault(key, {"state_call_ms": r["state_call_median_ms"],
                                       "question_call_ms": r["question_call_median_ms"],
                                       "phys_footprint_after_compile": r["phys_footprint_after_compile"]})
        files.append({"Ls": Ls, "Lq": Lq, "file": fpath.name, "bytes": size, "sha256": sha,
                      "form": "v2e_fp16fc (fp16 FC; the RopeSelect FC and its RoPE table float32; no table in the "
                              "graph: both signatures take embeds)",
                      "built_by": f"scripts/d1_export_pair.py --embeds --Ls {Ls} --Lq {Lq} + scripts/d1_storage.py --pair "
                                  f"--variant v2e --tflite exports/real_sharedstate_embeds_{tag}_fp32.tflite",
                      "signatures": export["contract"]["signatures"], "state": export["contract"]["state"],
                      "gate_mac": gates, "mac_calls_ms_median_of_20": calls})
    ss = dict(old["shared_state"])
    # a second pass (to add --host-check) keeps the first pass's ids list: `files` already holds the embeds pairs then
    ids_files = old["shared_state"].get("files_ids_not_shipped") or [
        dict(x, shipped=False, why_not_shipped="round 9's ids pair (v2: int8 table inside the graph); the embeds pair "
                                               "of the same shape replaces it in round 10")
        for x in old["shared_state"]["files"]]
    assert all("sharedstate_embeds" not in x["file"] for x in ids_files), [x["file"] for x in ids_files]
    ss.update(
        status="round 10: the embeds pairs (the shipped form: both signatures take embeds = the float32 rows of "
               "embed_table.safetensors at the padded ids) at Ls64+Lq64 / Ls128+Lq64 / Ls256+Lq128; round 9's ids "
               "pairs under files_ids_not_shipped; Mac gates in results/real_sharedstate_embeds_*_check.json, call times "
               "in results/timing_mac_r10.json",
        files=files, files_ids_not_shipped=ids_files,
        input="embeds float32 [1, L, 2048] in both signatures: the float32 rows of embed_table.safetensors (bfloat16) "
              "at the padded ids, the token_ids.pad row on the pads (host/d1_shared_state.py SharedStatePair with "
              "embed_table)",
        pick={"rule": "requests with pictures -> the embeds row graphs; else the form with the smallest expected time "
                      "from the measured call medians below: the rows at the sum of row[L] over the questions (L = the "
                      "smallest bucket that holds the row), each pair that holds the request (state <= Ls, every "
                      "question's own tokens <= Lq) at state + n x question; ties go to the rows "
                      "(host/d1_shared_state.py D1SharedHost(call_ms=...))",
              "call_ms": {"accelerator": "Mac M4 Max Metal fp32 (GpuOptions(enforce_f32=True)), pairs without "
                                         "constant_tensor_sharing, direct hand-over",
                          "row": {str(L): v for L, v in call["row"].items()},
                          "pair": [{"Ls": Ls, "Lq": Lq, "state_call_ms": v[0], "question_call_ms": v[1]}
                                   for (Ls, Lq), v in call["pair"].items()],
                          "source": "results/timing_mac_r10.json (median of 20 requests after 5 warm-ups per set)"},
              "pair_faster_from_questions": r10_min_questions(call),
              "why": a.why or "round 10: with the L128 row graph the row form's cost depends on the rows' bucket, so a "
                              "fixed question count per pair (round 9) no longer picks the faster form; the measured "
                              "call times decide per request"},
        host="host/d1_shared_state.py (D1SharedHost over host/d1_litert.py's D1Host, the embeds row graphs and the "
             "embed table)")
    if a.host_check:
        ss["host_check"] = json.loads(kpath_(a.host_check).read_text())["summary"]
    new["shared_state"] = ss
    new["buckets"]["host_default"] = list(H.BUCKETS)
    r10_status = ("; round 10: the text path on the embeds form (embeds_graph.buckets L128..L4096 and the embeds pairs "
                  "of shared_state), the ids graphs of `graphs` not shipped")
    new["status"] = old["status"] if r10_status in old["status"] else old["status"].rstrip(".") + r10_status
    assert {k: v for k, v in new.items() if k not in R10_KEEP} == {k: v for k, v in old.items() if k not in R10_KEEP}
    assert new["vision"] == old["vision"]
    if a.dry_run:
        print(json.dumps({"dry_run": True, "changed": sorted(k for k in new if new[k] != old.get(k)),
                          "buckets": [(b["L"], b["file"], b["sha256"][:12]) for b in buckets],
                          "pairs": [(x["file"], x["bytes"], x["sha256"][:12]) for x in files],
                          "call_ms": {"row": call["row"], "pair": {f"{k[0]}+{k[1]}": v for k, v in call["pair"].items()}},
                          "pair_faster_from_questions": r10_min_questions(call)}, indent=1))
        return 0
    path.write_text(json.dumps(new, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"written": str(path.relative_to(K)), "sha256": sha256_file(path),
                      "buckets": [(b["L"], b["file"], b["sha256"][:12]) for b in buckets],
                      "pairs": [(x["file"], x["bytes"], x["sha256"][:12]) for x in files],
                      "call_ms": {"row": call["row"], "pair": {f"{k[0]}+{k[1]}": v for k, v in call["pair"].items()}},
                      "pair_faster_from_questions": r10_min_questions(call)}, indent=1))
    return 0


def kpath_(x: str) -> Path:
    return Path(x) if Path(x).is_absolute() else K / x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ls", type=int, default=0)
    ap.add_argument("--Lq", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--contract", action="store_true", help="write host/contract.json's shared_state (docstring)")
    ap.add_argument("--pairs", default="256:128,128:64")
    ap.add_argument("--min-questions", default="2", help="--contract: one number, or per pair as 128:64=1,256:128=3")
    ap.add_argument("--why", default="", help="--contract: the measurement behind --min-questions")
    ap.add_argument("--guard-decision", default="", help="the recorded ruling (module docstring, 1)")
    ap.add_argument("--guard-alert-hidden", type=float, default=2.5e-4, help="the ruling's alert line on the question "
                                                                             "positions' hidden")
    ap.add_argument("--embeds", action="store_true", help="round 10: the embeds pair (module docstring)")
    ap.add_argument("--contract-r10", action="store_true", help="round 10: host/contract.json on the embeds form "
                                                                "(write_contract_r10)")
    ap.add_argument("--host-check", default="", help="--contract-r10: the host check json whose summary goes in")
    ap.add_argument("--dry-run", action="store_true", help="--contract-r10: build and check, print, write nothing")
    a = ap.parse_args()
    if a.contract_r10:
        return write_contract_r10(a)
    if a.contract:
        return write_contract(a)
    assert a.Ls and a.Lq, "--Ls and --Lq are required"
    Ls, Lq = a.Ls, a.Lq
    tag = f"Ls{Ls}_Lq{Lq}"
    kind = "_embeds" if a.embeds else ""
    path = K / f"exports/real_sharedstate{kind}_{tag}_fp32.tflite"
    out = K / f"results/real_sharedstate{kind}_export_{tag}.json"
    scan_out = K / f"results/real_sharedstate{kind}_opscan_{tag}.json"
    for p in (path, out, scan_out):
        assert not p.exists(), f"never overwrite {p}"
    guard_file = S.OUT_EMBEDS if a.embeds else S.OUT
    guard = json.loads(guard_file.read_text())
    assert not guard.get("limit"), "the torch check is a smoke run"
    override = None
    if a.embeds:     # round 10: the embeds check's pass is the ruling's gate itself (d1_shared_state.py --embeds)
        assert guard["input"] == "embeds" and [Ls, Lq] in guard["pairs"], (guard.get("input"), guard.get("pairs"))
        assert guard["summary"]["pass"], "the embeds torch check did not pass: no export"
    if not guard["summary"]["pass"]:
        assert a.guard_decision, "the torch check (acceptance 1) did not pass: no export"
        override = guard_override(guard, a.guard_alert_hidden, a.guard_decision)
        assert override["pass"], override
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    torch.set_num_threads(a.threads)
    t0 = time.time()
    lm, info = S.load_pair_model(S.SNAP)
    et = None
    if a.embeds:
        from d1_tables import EmbedTable

        et = EmbedTable(S.EMBED_TABLE)
    sp = (S.StatePrefillEmbeds if a.embeds else S.StatePrefill)(lm, Ls).eval().requires_grad_(False)
    qs = S.QuestionStep(lm, Ls, Lq, embeds=a.embeds).eval().requires_grad_(False)
    load_s = time.time() - t0
    checks, sample = probe_checks(lm, sp, qs, Ls, Lq, et)
    assert checks and all(c["argmax_equal"] and c["max_abs_dp_vs_reference"] <= 1e-4 for c in checks), checks
    assert list(sample[1]) == qs.input_names(), (list(sample[1])[:5], qs.input_names()[:5])
    doc = S.contract(lm, Ls, Lq, embeds=a.embeds)
    record = {"Ls": Ls, "Lq": Lq, "status": "RUNNING", "started_at": started, "file": str(path.relative_to(K)),
              "input": "embeds" if a.embeds else "ids",
              "guard": {"file": str(guard_file.relative_to(K)), "pass": guard["summary"]["pass"],
                        "all": guard["summary"]["all"], "override": override},
              "probe_checks": checks, "source": info, "sample_row": sample[2], "load_seconds": round(load_s, 1),
              "convert_call": (f"litert_torch.signature('state_prefill_{Ls}', StatePrefill{'Embeds' if a.embeds else ''}, "
                               f"sample_kwargs={{{'embeds' if a.embeds else 'ids'}, valid}})"
                               f".signature('question_step_{Ls}_{Lq}', QuestionStep, sample_kwargs="
                               f"{{{'embeds' if a.embeds else 'ids'}, valid, state_valid, 38 state tensors}})"
                               ".convert().export(path)"),
              "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "litert-torch",
                                                                      "litert-converter", "ai-edge-litert")},
              "threads": a.threads, "contract": doc}
    t1 = time.perf_counter()
    try:
        import litert_torch

        lrt = (litert_torch.signature(f"state_prefill_{Ls}", sp, sample_kwargs=sample[0])
               .signature(f"question_step_{Ls}_{Lq}", qs, sample_kwargs=sample[1]).convert())
        record["convert_seconds"] = round(time.perf_counter() - t1, 1)
        t2 = time.perf_counter()
        lrt.export(str(path))
        record["write_seconds"] = round(time.perf_counter() - t2, 1)
        del lrt
        from tflite_scan import scan

        s = scan(path)
        scan_out.write_text(json.dumps(s, indent=1) + "\n")
        sigs = {x["key"]: x for x in s["signatures"]}
        want = doc["signatures"]
        mismatch = []
        sig_ok = sorted(sigs) == sorted(want)
        if sig_ok:
            for key, w in want.items():
                for side in ("inputs", "outputs"):
                    got = {e["name"]: (e["shape"], e["dtype"]) for e in sigs[key][side]}
                    exp = {e["name"]: (e["shape"], e["dtype"].upper()) for e in w[side]}
                    if got != exp:
                        mismatch.append({"signature": key, "side": side, "missing": sorted(set(exp) - set(got)),
                                         "extra": sorted(set(got) - set(exp)),
                                         "different": sorted(k for k in set(exp) & set(got) if exp[k] != got[k])})
        ps = per_signature_ops(path)
        stop = []
        if s["custom_op_count"]:
            stop.append("CUSTOM")
        if s["rank_gt4_tensor_count"]:
            stop.append("RANK>4")
        if s["gather_count"] or s["forbidden_counts"].get("GATHER_ND"):
            stop.append("GATHER")
        if s["forbidden_total"]:
            stop.append("FORBIDDEN")
        if s["int64_tensor_count"]:
            stop.append("INT64")
        if s["batch_matmul_constant_left"]:
            stop.append("BMM_CONSTANT_LEFT")
        if not ps["rope_select_fully_connected"]:
            stop.append("NO_ROPESELECT_FC")
        if a.embeds and s["embedding_lookup_count"]:      # round 10: the table is the host's
            stop.append("EMBEDDING_LOOKUP")
        record.update(
            status="EXPORTED" if not stop else "STOP:" + "+".join(stop),
            bytes=s["bytes"], sha256=s["sha256"], subgraphs=s["subgraphs"], signatures=s["signatures"],
            signature_matches_contract=bool(sig_ok and not mismatch), signature_mismatch=mismatch,
            operator_count=s["operator_count"], op_histogram=s["op_histogram"], per_signature=ps["per_signature"],
            rope_select_fully_connected=ps["rope_select_fully_connected"],
            tensor_rank_histogram=s["tensor_rank_histogram"], tensor_dtype_histogram=s["tensor_dtype_histogram"],
            int64_tensor_count=s["int64_tensor_count"], int64_tensors=s["int64_tensors"],
            rank_gt4_tensor_count=s["rank_gt4_tensor_count"], forbidden_counts=s["forbidden_counts"],
            forbidden_total=s["forbidden_total"], pad_count=s["pad_count"], pad_summary=s["pad_summary"],
            batch_matmul_count=s["batch_matmul_count"], batch_matmul_constant_left=s["batch_matmul_constant_left"],
            batch_matmul_shape_groups=s["batch_matmul_shape_groups"],
            fully_connected_count=s["fully_connected_count"], embedding_lookup_count=s["embedding_lookup_count"],
            gather_count=s["gather_count"], custom_ops=s["custom_ops"], custom_op_count=s["custom_op_count"],
            stablehlo_ops=s["stablehlo_ops"], opscan=str(scan_out.relative_to(K)))
    except BaseException:
        (K / f"logs/{'r10_export_embeds' if a.embeds else 'r9_export'}_{tag}.traceback.txt").write_text(
            traceback.format_exc())
        record.update(status="FAIL", seconds=round(time.perf_counter() - t1, 1),
                      peak_rss_bytes_getrusage=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        out.write_text(json.dumps(record, indent=1) + "\n")
        raise
    record["peak_rss_bytes_getrusage"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss   # bytes on macOS
    record["seconds_wall"] = round(time.time() - t0, 1)
    record["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k not in ("op_histogram", "batch_matmul_shape_groups", "contract",
                                                                    "source", "signatures", "int64_tensors", "guard",
                                                                    "per_signature")}, indent=1))
    print(json.dumps({k: v["operator_count"] for k, v in record["per_signature"].items()}))
    return 0 if record["status"] == "EXPORTED" and record["signature_matches_contract"] else 1


if __name__ == "__main__":
    sys.exit(main())
