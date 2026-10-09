"""Round 9 acceptance 3 and 5: the shared-state pair file on the Mac through the CompiledModel API, request by request.

    scripts/d1_guarded.sh logs/r9_check_<label>.guard.log $EXPORT scripts/d1_check_pair.py \
        --tflite exports/real_sharedstate_Ls256_Lq128_v2_fp16fc_i8emb.tflite --accel cpu \
        --compare-row cache/real/litert_real_rowprefill_L256_v2_fp16fc_i8emb_cpu.npz
    $EXPORT \
        scripts/d1_check_pair.py --tflite <pair file> --accel gpu --f32 --share --stop-on-bar ...

Parity (default mode). Rows: every reference question (results/reference_real.json) whose request's state
(fixtures/rows.json `state_len`) fits Ls and whose own tokens (row - state) fit Lq, and the 4 red arms of the reference
(their state length from host/d1_litert.py's render of the fixtures/red_arms.json request; the arm rows equal the
reference's ids). Per request: state_prefill once (ids = the state's tokens right-padded with the contract's pad id),
then question_step for each question twice:
  host    state_prefill's outputs read back to numpy and written into question_step's own input buffers (once per
          request; the portable way)
  direct  state_prefill's output TensorBuffers passed as question_step's inputs (no copy through the host).
The answer slot = the question's last real token; read out with host/d1_litert.py `readout` (float32 logits of the
table rows, group max, float64 softmax) and compared with the reference's unrounded probs (host hand-over): argmax
(near ties apart: reference top-2 gap <= 0.02), max / mean / p95 |dp| (mean = the sum over questions and options / the
number of options), the questions over 0.02, the answer-slot hidden against the reference's (results/
reference_real_hidden.npz), non-finite rows, the red arms (each must move some option by > 0.02 against its base
record's reference probs), and the two hand-overs byte for byte (probabilities and the question's whole hidden).
Bar readings as d1_check.py: bar_strict, bar_near_tie_apart.
--compare-row <npz>... (d1_check.py runs of row graphs; keys hsel/<id>/<qid>): answer-slot max |dh| and max |dp|
against the row form's same question (the first npz that holds the key).
--torch-states: the state outputs of the requests d1_shared_state.py --check dumped (cache/real/pair/
torch_state_Ls<Ls>_<id>.npz) against this file's (k / v at the real positions, the conv tails).
GPU: Options(GPU, GpuOptions(enforce_f32=--f32, constant_tensor_sharing=--share)) = Metal. Delegation evidence: the
runtime's VERBOSE lines (d1_check.runtime_log_verbose), this process's fd 2 -> logs/<stem>_<acc>.runtime.log, every
`Replacing` line parsed per subgraph; is_fully_accelerated(); the process's phys_footprint after the compile and at the
end (d1_clock_mac.phys_footprint).
--limit-requests N: the first N requests (no red arms): a run that shows the delegation (the float32 pair on Metal).
--stop-on-bar: exit 1 when the bar (near ties apart) or a red arm fails, or the two hand-overs differ.
Outputs (never overwritten): results/<stem>_<acc>_check.json (summary, red arms, delegation), cache/real/pair/
refrows_<stem>_<acc>.json (per question, gitignored), cache/real/pair/litert_<stem>_<acc>.npz (hsel/<id>/<qid> of the
host hand-over); acc = cpu | gpu_f32[_share] | gpu_default[_share] [+ _lim<N>]. Times are informational (the Mac is
shared): the ms of the ladder come from d1_clock_pair.py.

--host (acceptance 5): host/d1_shared_state.py against host/d1_litert.py on every fixture request with more than one
question and no pictures (fixtures/requests.json, 16 records; --single adds the single-question records a pair holds):
the row host (D1Host over the v2 row graphs of host/contract.json `graphs`, loaded one at a time) and the shared host
(D1SharedHost over --pair files with min_questions 1 = every request a pair holds goes through it, the row host for the
rest) answer the same request; per question the probabilities (max |dp| <= 1e-5) and the argmax must agree,
usage.input_tokens equal; the route of every request (pair Ls/Lq or row form, why) is recorded, with
--pick-min-questions also the route under the contract's rule (route only), and both hosts' probabilities are also
compared with the provider's reference (row path `probs`, tree path `probs_tree`). CPU threads --threads, hand-over
--handover. -> results/real_sharedstate_host_check.json (never overwritten).

Round 10: an embeds pair (scripts/d1_export_pair.py --embeds: `embeds` float32 [1, L, d] in place of `ids`) runs with
the host's table (--embed-table, the float32 rows of the bfloat16 embed_table.safetensors; the pair's kind is read from
its signatures; --torch-states then reads cache/real/pair/embeds/). --host runs the row host on the embeds row graphs
of host/contract.json `embeds_graph.buckets` + the table (the new default; --ids-graphs = round 9's ids row graphs) and
writes --out (default the round-9 name; its per-question rows go to cache/real/pair/<out stem>_rows.json).
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

K = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K / "scripts"))
sys.path.insert(0, str(K / "host"))       # host/ first: `d1_shared_state` here is the HOST module (the torch one in
import d1_litert as H  # noqa: E402       # scripts/ has the same name; this process never imports torch)
import d1_shared_state as HS  # noqa: E402
from d1_check import LINE_KEYS, runtime_log_verbose  # noqa: E402
from d1_clock_mac import phys_footprint  # noqa: E402

ROWS = K / "fixtures/rows.json"
REQUESTS = K / "fixtures/requests.json"
RED_ARMS = K / "fixtures/red_arms.json"
TOKENIZER = K / "hf_small/tokenizer.json"
PAIR_DIR = K / "cache/real/pair"
STATE_DUMP = ("card_text_001", "own_fiveq_09", "tv4_000")   # = scripts/d1_shared_state.py STATE_DUMP
assert Path(HS.__file__).resolve() == (K / "host/d1_shared_state.py").resolve(), HS.__file__
NEAR_TIE, RED_ARM_MIN, BAR_MAX_DP, BAR_MEAN_DP = 0.02, 0.02, 0.02, 0.002
REPLACING = re.compile(r"Replacing (\d+) out of (\d+) node\(s\) with delegate \(([^)]*)\) node, yielding (\d+) "
                       r"partitions(?: for subgraph (\d+) \(([^)]*)\))?")


def kpath(x: str) -> Path:
    return Path(x) if Path(x).is_absolute() else K / x


def sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 24):
            h.update(block)
    return h.hexdigest()


def maxabs(a, b) -> float:
    return float(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).max())


def arm_state_lengths(tok) -> dict:
    """{arm id: (state_len, row ids)} from the host's render of the arm requests."""
    host = H.D1Host(tok)
    out = {}
    for rec in json.loads(RED_ARMS.read_text())["records"]:
        for r in host.rows(rec["request"]):
            out[(rec["id"], r.name)] = (r.state_len, r.ids)
    return out


def parity(a) -> int:
    path = kpath(a.tflite)
    stem = path.stem
    acc = ("cpu" if a.accel == "cpu" else ("gpu_f32" if a.f32 else "gpu_default") + ("_share" if a.share else "")) + \
          (f"_lim{a.limit_requests}" if a.limit_requests else "")
    out = K / f"results/{stem}_{acc}_check.json"
    refrows_out = PAIR_DIR / f"refrows_{stem}_{acc}.json"
    npz_out = PAIR_DIR / f"litert_{stem}_{acc}.npz"
    log = K / f"logs/{stem}_{acc}.runtime.log"
    for p in (out, refrows_out, npz_out, log):
        assert not p.exists(), f"refusing to overwrite {p}"
    PAIR_DIR.mkdir(parents=True, exist_ok=True)
    log_f = open(log, "w")
    saved_fd2 = os.dup(2)
    os.dup2(log_f.fileno(), 2)
    doc = {"what": "round 9: the shared-state pair file through CompiledModel, request by request, vs the provider's "
                   "float32 reference", "tflite": str(path.relative_to(K)), "tflite_bytes": path.stat().st_size,
           "accel": acc, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "ai_edge_litert": importlib.metadata.version("ai-edge-litert"), "runtime_log": str(log.relative_to(K))}
    rows, arms, store = [], [], {}
    try:
        doc["logger"] = runtime_log_verbose()
        tok = H.D1Tokenizer(TOKENIZER)
        table = H.ReadoutTable.from_file(kpath(a.table))
        et = H.EmbedTable(kpath(a.embed_table))      # read only by an embeds pair (round 10)
        t0 = time.perf_counter()
        pair = HS.SharedStatePair(path, "gpu" if a.accel == "gpu" else "cpu", "fp32" if a.f32 else "default", a.threads,
                               handover="both", share=a.share, embed_table=et)
        doc.update(compile_seconds=round(time.perf_counter() - t0, 3), options=pair.options_desc,
                   is_fully_accelerated=pair.fully_accelerated, memory_after_compile=phys_footprint(),
                   signatures={"state": pair.sig_state, "question": pair.sig_question}, Ls=pair.Ls, Lq=pair.Lq,
                   state_tensors=len(pair.names), state_bytes=pair.state_bytes, input=pair.input)
        if pair.input == "embeds":
            doc["embed_table"] = {"file": a.embed_table, "sha256": sha256(kpath(a.embed_table))}
        Ls, Lq = pair.Ls, pair.Lq
        ref = json.loads(kpath(a.reference).read_text())
        hid = np.load(kpath(a.reference_hidden)) if a.reference_hidden else None
        cmp = [np.load(kpath(x)) for x in a.compare_row]
        state_len = {f"{r['id']}/{r['qid']}": r["state_len"] for r in json.loads(ROWS.read_text())["rows"]}
        by_req: dict = {}
        for q in ref["questions"]:
            key = f"{q['id']}/{q['qid']}"
            n = state_len[key]
            if q.get("ids") is not None and n <= Ls and len(q["ids"]) - n <= Lq:
                by_req.setdefault(q["id"], []).append(q)
        items = list(by_req.items())[: a.limit_requests or None]
        t_all = time.perf_counter()
        state_ms, q_ms = [], []

        def run(rid, n, state_ids, qs):
            t = time.perf_counter()
            arrays = pair.run_state(state_ids)
            state_ms.append((time.perf_counter() - t) * 1000)
            res = []
            for q in qs:
                own = q["ids"][n:]
                m = len(own)
                t = time.perf_counter()
                h_host = pair.run_question(own, handover="host")
                h_direct = pair.run_question(own, handover="direct")
                q_ms.append((time.perf_counter() - t) * 1000 / 2)
                res.append((q, m, h_host, h_direct))
            return arrays, res

        def score(q, m, h_host, h_direct, key):
            h = h_host[m - 1]
            rec = {"key": key, "row_len": len(q["ids"]), "n_question": m,
                   "nonfinite_real": int((~np.isfinite(h_host[:m])).sum()),
                   "handover_hidden_bit_equal": bool(np.array_equal(h_host[:m], h_direct[:m]))}
            store[f"hsel/{key}"] = h.copy()
            if hid is not None and key in hid.files and np.isfinite(h).all():
                rec["h_sel_max_abs_diff"] = maxabs(h, hid[key])
            for c in cmp:
                if f"hsel/{key}" in c.files:
                    hc = c[f"hsel/{key}"]
                    rec["h_sel_max_abs_vs_row"] = maxabs(h, hc)
                    if np.isfinite(h).all() and np.isfinite(hc).all():
                        rec["max_abs_dp_vs_row"] = maxabs(H.readout(h, table, q["readout_ids"]),
                                                          H.readout(hc, table, q["readout_ids"]))
                    break
            if not np.isfinite(h).all():
                rec["probs"] = None
                return rec, None
            p = H.readout(h, table, q["readout_ids"])
            p_d = H.readout(h_direct[m - 1], table, q["readout_ids"]) if np.isfinite(h_direct[m - 1]).all() else None
            rec.update(probs=p, probs_direct=p_d, handover_probs_bit_equal=p_d == p)
            return rec, p

        dumps = {}
        for rid, qs in items:
            n = state_len[f"{rid}/{qs[0]['qid']}"]
            state_ids = qs[0]["ids"][:n]
            assert all(q["ids"][:n] == state_ids for q in qs), rid
            arrays, res = run(rid, n, state_ids, qs)
            if a.torch_states and rid in STATE_DUMP:
                tp = (PAIR_DIR / "embeds" if pair.input == "embeds" else PAIR_DIR) / f"torch_state_Ls{Ls}_{rid}.npz"
                if tp.exists():
                    t = np.load(tp)
                    assert int(t["n_state"]) == n and t["state_ids"].tolist() == state_ids, rid
                    kv = max(maxabs(arrays[k][:, :, :n], t[k][:, :, :n]) for k in arrays if not k.startswith("conv_tail"))
                    ct = max(maxabs(arrays[k], t[k]) for k in arrays if k.startswith("conv_tail"))
                    dumps[rid] = {"file": str(tp.relative_to(K)), "n_state": n, "kv_real_max_abs": kv,
                                  "conv_tail_max_abs": ct}
            for q, m, h_host, h_direct in res:
                key = f"{q['id']}/{q['qid']}"
                rec, p = score(q, m, h_host, h_direct, key)
                rec.update(type=q["type"], n_state=n, keys=q["keys"], reference_probs=q["probs"],
                           near_tie=bool(q["near_tie"]), top2_gap=q["top2_gap"])
                if p is not None:
                    dp = np.abs(np.asarray(p) - np.asarray(q["probs"]))
                    order = sorted(range(len(q["probs"])), key=lambda i: -q["probs"][i])
                    rec.update(argmax=int(np.argmax(p)), reference_argmax=order[0],
                               argmax_equal=int(np.argmax(p)) == order[0], max_abs_dp=float(dp.max()),
                               sum_abs_dp=float(dp.sum()), options=len(p),
                               reference_top2=[[q["keys"][i], q["probs"][i]] for i in order[:2]])
                rows.append(rec)
        if not a.limit_requests:
            arm_src = arm_state_lengths(tok)
            for x in ref["red_arms"]["records"]:
                n, ids = arm_src[(x["id"], x["qid"])]
                assert ids == x["ids"], x["id"]
                key = f"{x['id']}/{x['qid']}"
                if n > Ls or len(ids) - n > Lq:
                    arms.append({"key": key, "fits": False, "n_state": n, "n_question": len(ids) - n})
                    continue
                q = dict(x)
                _, res = run(x["id"], n, ids[:n], [q])
                _, m, h_host, h_direct = res[0]
                rec, p = score(q, m, h_host, h_direct, key)
                rec.update(fits=True, n_state=n, base=f"{x['base_id']}/{x['qid']}", kind=x.get("kind"))
                if p is not None:
                    pa, pb = dict(zip(x["keys"], p)), dict(zip(x["keys_base"], x["probs_base"]))
                    rec.update(max_abs_dp_vs_base_reference=max(abs(pa[k] - pb[k]) for k in pa),
                               max_abs_dp_vs_arm_reference=maxabs(p, x["probs"]))
                    rec["red"] = rec["max_abs_dp_vs_base_reference"] > RED_ARM_MIN
                else:
                    rec["red"] = False
                arms.append(rec)
        doc["memory_at_end"] = phys_footprint()
        doc["seconds_rows"] = round(time.perf_counter() - t_all, 1)
        doc["ms_informational"] = {"state_call_median": float(np.median(state_ms)) if state_ms else None,
                                   "question_call_median": float(np.median(q_ms)) if q_ms else None,
                                   "note": "contended Mac; d1_clock_pair.py gives the ladder's ms"}
        doc["torch_state_check"] = dumps
        pair.close()
        doc["status"] = "OK"
    except BaseException as e:
        doc["status"] = "FAIL"
        doc["error"] = f"{type(e).__name__}: {e}"
        import traceback

        traceback.print_exc()
    finally:
        sys.stderr.flush()
        os.dup2(saved_fd2, 2)
        log_f.close()
    lines = log.read_text(errors="replace").splitlines()
    doc["runtime_log_lines"] = len(lines)
    doc["runtime_log_key_lines"] = [ln for ln in lines if LINE_KEYS.search(ln)][:80]
    doc["delegation"] = [{"delegated": int(m[0]), "total": int(m[1]), "delegate": m[2], "partitions": int(m[3]),
                          "subgraph": int(m[4]) if m[4] else None, "subgraph_name": m[5] or None}
                         for m in REPLACING.findall("\n".join(lines))]
    if doc["status"] == "OK":
        doc["summary"] = summarize(rows, arms, a.limit_requests)
        np.savez(npz_out, **store)
        refrows_out.write_text(json.dumps({"tflite": doc["tflite"], "accel": acc, "rows": rows}) + "\n")
        doc.update(rows_file=str(refrows_out.relative_to(K)), rows_file_sha256=sha256(refrows_out),
                   litert_npz=str(npz_out.relative_to(K)), red_arms=arms)
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: doc.get(k) for k in ("status", "error", "accel", "Ls", "Lq", "compile_seconds",
                                               "is_fully_accelerated", "delegation", "torch_state_check")}, indent=1))
    s = doc.get("summary")
    if s:
        print(json.dumps(s["line"], indent=1))
        if a.stop_on_bar and not s["stop_bar_pass"]:
            print("STOP: bar (near ties apart), a red arm or the hand-over equality failed (--stop-on-bar)")
            return 1
    return 0 if doc["status"] == "OK" else 1


def summarize(rows, arms, limited) -> dict:
    q = [r for r in rows if r.get("probs") is not None]
    non = [r for r in q if not r["near_tie"]]
    near = [r for r in q if r["near_tie"]]
    nonfinite = [r["key"] for r in rows if r.get("probs") is None]
    options = sum(r["options"] for r in q)
    all_dp = [float(v) for r in q for v in np.abs(np.asarray(r["probs"]) - np.asarray(r["reference_probs"]))]
    mx = max((r["max_abs_dp"] for r in q), default=None)
    mean = sum(r["sum_abs_dp"] for r in q) / options if options else None
    max_ok, mean_ok = mx is not None and mx <= BAR_MAX_DP, mean is not None and mean <= BAR_MEAN_DP
    non_eq = sum(r["argmax_equal"] for r in non)
    hs = [r["h_sel_max_abs_diff"] for r in rows if "h_sel_max_abs_diff" in r]
    vs_row = [r for r in rows + [x for x in arms if x.get("fits")] if "h_sel_max_abs_vs_row" in r]
    hand = [r for r in rows + [x for x in arms if x.get("fits")] if r.get("probs") is not None]
    fit_arms = [x for x in arms if x.get("fits")]
    s = {"questions_run": len(rows), "requests_run": len({r["key"].split("/")[0] for r in rows}),
         "questions_finite": len(q), "argmax_equal": sum(r["argmax_equal"] for r in q),
         "non_near_tie_argmax": f"{non_eq}/{len(non) + sum(1 for r in rows if r.get('probs') is None and not r['near_tie'])}",
         "near_tie_argmax": f"{sum(r['argmax_equal'] for r in near)}/{len(near)}",
         "near_tie_flips": [{"key": r["key"], "top2_gap": r["top2_gap"], "reference_top2": r["reference_top2"],
                             "probs": r["probs"]} for r in near if not r["argmax_equal"]],
         "non_near_tie_flips": [{"key": r["key"], "top2_gap": r["top2_gap"], "reference_top2": r["reference_top2"],
                                 "probs": r["probs"]} for r in non if not r["argmax_equal"]],
         "max_abs_dp": mx, "mean_abs_dp_all_options": mean,
         "p95_abs_dp_all_options": float(np.percentile(all_dp, 95)) if all_dp else None, "options": options,
         "max_abs_dp_key": max(q, key=lambda r: r["max_abs_dp"])["key"] if q else None,
         "questions_over_max_dp": [{"key": r["key"], "max_abs_dp": r["max_abs_dp"], "near_tie": r["near_tie"]}
                                   for r in sorted(q, key=lambda r: -r["max_abs_dp"]) if r["max_abs_dp"] > BAR_MAX_DP],
         "h_sel_max_abs_diff": max(hs, default=None), "nonfinite": nonfinite,
         "vs_row_form": {"questions": len(vs_row),
                         "h_sel_max_abs": max((r["h_sel_max_abs_vs_row"] for r in vs_row), default=None),
                         "max_abs_dp": max((r.get("max_abs_dp_vs_row", 0.0) for r in vs_row), default=None)},
         "handover": {"questions": len(hand),
                      "probs_bit_equal": sum(bool(r.get("handover_probs_bit_equal")) for r in hand),
                      "hidden_bit_equal": sum(bool(r.get("handover_hidden_bit_equal")) for r in hand)},
         # round 10: the bar reads the arms the pair holds (Ls64+Lq64 holds 3 of the 4: one arm's state is 72 tokens);
         # `all_fit` records whether every arm fits
         "red_arms": {"fit": len(fit_arms), "in_reference": len(arms), "red": sum(bool(x.get("red")) for x in fit_arms),
                      "all_fit": len(fit_arms) == len(arms),
                      "all_red": bool(fit_arms) and all(x.get("red") for x in fit_arms),
                      "not_fit": [{"key": x["key"], "n_state": x["n_state"], "n_question": x["n_question"]}
                                  for x in arms if not x.get("fits")],
                      "max_abs_dp_vs_base": [round(x.get("max_abs_dp_vs_base_reference", 0.0), 6) for x in fit_arms]},
         "bar": {"max_abs_dp": BAR_MAX_DP, "mean_abs_dp": BAR_MEAN_DP, "near_tie_gap": NEAR_TIE, "red_arm_min": RED_ARM_MIN},
         "limited_to_requests": limited or None}
    s["bar_strict"] = bool(not nonfinite and len(q) == len(rows) and s["argmax_equal"] == len(rows) and max_ok and mean_ok)
    s["bar_near_tie_apart"] = bool(not nonfinite and non_eq == len(non) and max_ok and mean_ok)
    s["handover_equal"] = s["handover"]["probs_bit_equal"] == len(hand) and s["handover"]["hidden_bit_equal"] == len(hand)
    s["stop_bar_pass"] = bool(s["bar_near_tie_apart"] and (limited or s["red_arms"]["all_red"]) and s["handover_equal"])
    s["line"] = {k: s[k] for k in ("questions_run", "requests_run", "argmax_equal", "non_near_tie_argmax",
                                   "near_tie_argmax", "max_abs_dp", "mean_abs_dp_all_options", "p95_abs_dp_all_options",
                                   "h_sel_max_abs_diff", "bar_strict", "bar_near_tie_apart", "handover_equal")}
    s["line"].update(red_arms=s["red_arms"]["max_abs_dp_vs_base"], nonfinite=len(nonfinite), vs_row_form=s["vs_row_form"])
    return s


class LazyRowGraphs(dict):
    """{L: row graph} for D1Host that compiles a graph on first use and closes the one before (one at a time); cls =
    H.LiteRTRowGraph (ids graphs) or H.LiteRTEmbedsGraph (embeds graphs, round 10)."""

    def __init__(self, files: dict, accelerator: str, precision: str, threads: int, cls=None):
        super().__init__({L: None for L in files})
        self.files, self.args, self.loaded, self.compiles = files, (accelerator, precision, threads), None, []
        self.cls = cls or H.LiteRTRowGraph

    def __getitem__(self, L):
        if self.loaded is not None and self.loaded[0] != L:
            self.loaded[1].close()
            self.loaded = None
        if self.loaded is None:
            t = time.perf_counter()
            self.loaded = (L, self.cls(self.files[L], *self.args))
            self.compiles.append({"L": L, "seconds": round(time.perf_counter() - t, 2)})
        return self.loaded[1]

    def close(self):
        if self.loaded is not None:
            self.loaded[1].close()
            self.loaded = None


def host_check(a) -> int:
    out = kpath(a.out)
    assert not out.exists(), f"refusing to overwrite {out}"
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    contract = json.loads((K / "host/contract.json").read_text())
    tok = H.D1Tokenizer(TOKENIZER)
    table = H.ReadoutTable.from_file(kpath(a.table))
    pad = int(contract["token_ids"]["pad"])
    if a.ids_graphs:   # round 9's row host: the ids row graphs of contract `graphs`
        files = {int(g["L"]): K / "exports" / g["file"] for g in contract["graphs"]}
        graphs = LazyRowGraphs(files, a.accel, "fp32" if a.f32 else "default", a.threads)
        et = None
        row_host = H.D1Host(tok, {}, table, pad)
        row_host.graphs = graphs     # after __init__: its dict(graphs) copy would drop the lazy __getitem__
    else:              # round 10 (the default): the embeds row graphs of contract `embeds_graph.buckets` + the table
        files = {int(g["L"]): K / "exports" / g["file"] for g in contract["embeds_graph"]["buckets"]}
        graphs = LazyRowGraphs(files, a.accel, "fp32" if a.f32 else "default", a.threads, cls=H.LiteRTEmbedsGraph)
        et = H.EmbedTable(kpath(a.embed_table))
        row_host = H.D1Host(tok, {}, table, pad, embed_table=et)
        row_host.embeds_graphs = graphs
    pairs = [HS.SharedStatePair(kpath(p), a.accel, "fp32" if a.f32 else "default", a.threads, handover=a.handover,
                             share=a.share, pad_id=pad, embed_table=et) for p in a.pair]
    shared = HS.D1SharedHost(row_host, pairs, min_questions=1)    # every request a pair holds goes through it
    shipped, call_ms = None, None
    if a.pick_contract:      # round 10: the contract's rule = the measured call times (routes only, nothing run)
        pk = contract["shared_state"]["pick"]["call_ms"]
        call_ms = {"row": {int(L): float(v) for L, v in pk["row"].items()},
                   "pair": {(int(x["Ls"]), int(x["Lq"])): (float(x["state_call_ms"]), float(x["question_call_ms"]))
                            for x in pk["pair"]}}
        shipped = HS.D1SharedHost(row_host, pairs, call_ms=call_ms)
    elif a.pick_min_questions:   # round 9: the contract's rule = min_questions (routes only, nothing run)
        shipped = HS.D1SharedHost(row_host, pairs, min_questions=parse_min_questions(a.pick_min_questions))
    ref = json.loads(kpath(a.reference).read_text())
    refq = {f"{q['id']}/{q['qid']}": q for q in ref["questions"]}
    all_records = [r for r in json.loads(REQUESTS.read_text())["records"] if not r["request"].get("images")]
    records = [r for r in all_records if len(r["request"]["questions"]) > 1]
    if a.single:      # the single-question records a pair holds (the shipped rule sends those that fit Ls128+Lq64)
        records += [r for r in all_records if len(r["request"]["questions"]) == 1
                    and safe_route(shared, r["request"])["route"] == "pair"]
    out_rows, routes = [], []
    t_all = time.perf_counter()
    for rec in records:
        req = rec["request"]
        route = safe_route(shared, req)
        if shipped is not None:
            route["route_shipped_rule"] = safe_route(shipped, req)
        t = time.perf_counter()
        r_row, err_row = safe_decide(row_host, req)
        t_row = time.perf_counter() - t
        t = time.perf_counter()
        r_pair, err_pair = safe_decide(shared, req)
        t_pair = time.perf_counter() - t
        if err_row or err_pair:   # a request the provider's parser refuses: both hosts must refuse it alike
            routes.append({"id": rec["id"], "questions": len(req["questions"]), **route,
                           "refused_row_host": err_row, "refused_shared_host": err_pair,
                           "input_tokens_equal": err_row == err_pair})
            continue
        routes.append({"id": rec["id"], "questions": len(req["questions"]), **route,
                       "seconds_row_host": round(t_row, 3), "seconds_shared_host": round(t_pair, 3),
                       "input_tokens_equal": r_row["usage"] == r_pair["usage"], "usage": r_pair["usage"]})
        for name, ans_row in r_row["answers"].items():
            ans_pair = r_pair["answers"][name]
            pr, pp = answer_probs(ans_row), answer_probs(ans_pair)
            q = refq.get(f"{rec['id']}/{name}")
            row = {"key": f"{rec['id']}/{name}", "type": ans_row["type"], "route": route["route"],
                   "questions_in_request": len(req["questions"]),
                   "max_abs_dp_shared_vs_row_host": maxabs(pp, pr),
                   "argmax_equal": int(np.argmax(pp)) == int(np.argmax(pr)),
                   "answers_equal": ans_row == ans_pair}
            if q is not None:
                row["max_abs_dp_shared_vs_reference_row"] = maxabs(pp, q["probs"])
                if q.get("probs_tree") is not None:
                    row["max_abs_dp_shared_vs_reference_tree"] = maxabs(pp, q["probs_tree"])
                row["max_abs_dp_row_host_vs_reference_row"] = maxabs(pr, q["probs"])
            else:
                row["reference"] = "not in the reference (the provider's parser refused it)"
            out_rows.append(row)
    graphs.close()
    for p in pairs:
        p.close()
    def stats(rows_, routes_):
        via_pair = [r for r in rows_ if r["route"] == "pair"]
        return {"requests": len(routes_), "questions": len(rows_),
                "routes": {k: sum(1 for x in routes_ if x["route"] == k) for k in ("pair", "row", "refused")},
                "refused_alike": [x["id"] for x in routes_ if x.get("refused_row_host") and x["input_tokens_equal"]],
                "routes_by_pair": {f"Ls{Ls}+Lq{Lq}": sum(1 for x in routes_ if x.get("Ls") == Ls and x.get("Lq") == Lq)
                                   for Ls, Lq in sorted({(p.Ls, p.Lq) for p in pairs})},
                "max_abs_dp_shared_vs_row_host": max((r["max_abs_dp_shared_vs_row_host"] for r in rows_), default=None),
                "max_abs_dp_via_pair": max((r["max_abs_dp_shared_vs_row_host"] for r in via_pair), default=None),
                "argmax_equal": f"{sum(r['argmax_equal'] for r in rows_)}/{len(rows_)}",
                "answers_identical": sum(r["answers_equal"] for r in rows_),
                "input_tokens_equal": f"{sum(x['input_tokens_equal'] for x in routes_)}/{len(routes_)}",
                "max_abs_dp_shared_vs_reference_tree": max((r["max_abs_dp_shared_vs_reference_tree"] for r in rows_
                                                            if "max_abs_dp_shared_vs_reference_tree" in r), default=None),
                "pass": bool(rows_ and max(r["max_abs_dp_shared_vs_row_host"] for r in rows_) <= 1e-5
                             and all(r["argmax_equal"] for r in rows_) and all(x["input_tokens_equal"] for x in routes_))}

    summary = {"multi_question": stats([r for r in out_rows if r["questions_in_request"] > 1],
                                       [x for x in routes if x["questions"] > 1]),
               "bar": {"max_abs_dp_shared_vs_row_host": 1e-5, "argmax": "equal", "input_tokens": "equal"}}
    if a.single:
        summary["single_question"] = stats([r for r in out_rows if r["questions_in_request"] == 1],
                                           [x for x in routes if x["questions"] == 1])
    if shipped is not None:
        summary["shipped_rule"] = {"min_questions": a.pick_min_questions or None,
                                   "call_ms": {"row": {str(L): v for L, v in call_ms["row"].items()},
                                               "pair": {f"Ls{k[0]}+Lq{k[1]}": v for k, v in call_ms["pair"].items()}}
                                   if call_ms else None,
                                   "routes": {k: sum(1 for x in routes if x["route_shipped_rule"]["route"] == k)
                                              for k in ("pair", "row", "refused")},
                                   "routes_by_pair": {f"Ls{Ls}+Lq{Lq}": sum(
                                       1 for x in routes if x["route_shipped_rule"].get("Ls") == Ls
                                       and x["route_shipped_rule"].get("Lq") == Lq)
                                       for Ls, Lq in sorted({(p.Ls, p.Lq) for p in pairs})}}
    summary["pass"] = all(v["pass"] for k, v in summary.items() if k in ("multi_question", "single_question"))
    doc = {"what": "round 9 acceptance 5: host/d1_shared_state.py (pair) vs host/d1_litert.py (row form) on the fixture "
                   "requests of more than one question" + (" and the single-question requests a pair holds" if a.single
                                                            else ""), "started_at": started,
           "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seconds_wall": round(time.perf_counter() - t_all, 1),
           "accel": a.accel, "precision": "fp32" if a.f32 else "default", "threads": a.threads, "handover": a.handover,
           "share": a.share, "min_questions": shared.min_questions,
           "min_questions_note": "1 = every request a pair holds goes through it (the widest test of the pair path); "
                                 "`route_shipped_rule` = where the contract's pick rule (--pick-min-questions) sends it",
           "pairs":[{"file": str(kpath(p).relative_to(K)), "Ls": x.Ls, "Lq": x.Lq}
                                       for p, x in zip(a.pair, pairs)],
           "row_graphs": {str(L): str(f.relative_to(K)) for L, f in files.items()}, "row_graph_compiles": graphs.compiles,
           "row_graph_input": "ids" if a.ids_graphs else "embeds",
           "pair_inputs": sorted({x.input for x in pairs}),
           "host_files_sha256": {p: sha256(K / p) for p in ("host/d1_litert.py", "host/d1_shared_state.py")},
           "summary": summary}
    rows_out = (PAIR_DIR / "host_check_rows.json" if out.name == "real_sharedstate_host_check.json"   # round 9's name
                else PAIR_DIR / f"{out.stem}_rows.json")   # per question and per request (gitignored, like refrows_*)
    assert not rows_out.exists(), f"refusing to overwrite {rows_out}"
    rows_out.write_text(json.dumps({"routes": routes, "rows": out_rows}) + "\n")
    doc.update(rows_file=str(rows_out.relative_to(K)), rows_file_sha256=sha256(rows_out),
               routes_multi_question=[x for x in routes if x["questions"] > 1])
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    print(json.dumps([x for x in routes if x["questions"] > 1], indent=1))
    return 0 if summary["pass"] else 1


def safe_route(host, req) -> dict:
    try:
        return host.route(req)
    except Exception as e:      # the request does not render (the provider's parser refuses it too)
        return {"route": "refused", "why": f"{type(e).__name__}: {e}"}


def safe_decide(host, req) -> tuple:
    try:
        return host.decide(req), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def parse_min_questions(text: str) -> int | dict:
    """'3' -> 3; '128:64=1,256:128=3' -> {(128, 64): 1, (256, 128): 3} (D1SharedHost's min_questions)."""
    if "=" not in text:
        return int(text)
    out = {}
    for part in text.split(","):
        shape, n = part.split("=")
        Ls, Lq = (int(x) for x in shape.split(":"))
        out[(Ls, Lq)] = int(n)
    return out


def answer_probs(ans: dict) -> list[float]:
    """An answer's option probabilities in read-out order (noul: [yes, no])."""
    if ans["type"] == "noul":
        return [ans["noul"], 1.0 - ans["noul"]]
    return list(ans["probabilities"].values())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", default="", help="the pair file (parity mode)")
    ap.add_argument("--accel", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--f32", action="store_true", help="GPU: GpuOptions(enforce_f32=True)")
    ap.add_argument("--share", action="store_true", help="GPU: GpuOptions(constant_tensor_sharing=True)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--reference", default="results/reference_real.json")
    ap.add_argument("--reference-hidden", default="results/reference_real_hidden.npz")
    ap.add_argument("--table", default="cache/real/tables/readout_table.safetensors")
    ap.add_argument("--compare-row", nargs="*", default=[], help="npz of d1_check.py row-graph runs (hsel/<id>/<qid>)")
    ap.add_argument("--torch-states", action="store_true", help="state outputs vs the torch check's dumps")
    ap.add_argument("--limit-requests", type=int, default=0)
    ap.add_argument("--stop-on-bar", action="store_true")
    ap.add_argument("--host", action="store_true", help="acceptance 5 (module docstring)")
    ap.add_argument("--pair", nargs="*", default=[], help="--host: the v2 pair files")
    ap.add_argument("--handover", choices=["direct", "host"], default="direct", help="--host: the shared host's hand-over")
    ap.add_argument("--single", action="store_true", help="--host: also the single-question records a pair holds")
    ap.add_argument("--pick-min-questions", default="", help="--host: the contract's rule, e.g. 128:64=1,256:128=3 "
                                                              "(routes recorded, nothing run)")
    ap.add_argument("--embed-table", default="cache/real/tables/embed_table.safetensors",
                    help="round 10: the host's table for embeds pairs and the embeds row graphs")
    ap.add_argument("--pick-contract", action="store_true", help="--host (round 10): routes under the contract's "
                                                                  "shared_state.pick.call_ms (nothing run)")
    ap.add_argument("--ids-graphs", action="store_true", help="--host: the ids row graphs (round 9) in place of the "
                                                               "embeds row graphs (round 10's default)")
    ap.add_argument("--out", default="results/real_sharedstate_host_check.json", help="--host output (never "
                                                                                     "overwritten)")
    a = ap.parse_args()
    if a.host:
        return host_check(a)
    assert a.tflite, "--tflite is required"
    return parity(a)


if __name__ == "__main__":
    sys.exit(main())
