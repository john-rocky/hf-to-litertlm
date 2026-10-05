"""The Python host (host/kev_litert.py) against the reference and the desktop gate's rows -> results/host_parity[_4b].json.

    python scripts/host_parity.py [--model 4b]          (host environment: tokenizers, numpy, safetensors, ai-edge-litert)

1. tokenizer: host/tokenizer_probes.json (12 strings, expected ids from the author's AutoTokenizer) through the host's
   KevTokenizer (the Kev repository's tokenizer.json): plain ids and user_tokens ids, 12/12 each.
2. contract: kev_litert.encode_rows on every fixture request vs oracle/oracle_<model>.json (the author's to_record ->
   encode -> rows_of): row ids, decide / option indices, keys, type, the score legend, usage.input_tokens. On 4B the
   reference's row ids must also equal the 0.8B reference's (one tokenizer).
3. head: the host's PointerHead (host/kev_<model>_pointer_head.safetensors, numpy float32) on the reference's hidden
   states ([decide, *options] per question) vs the reference's probs and z_post (torch, fp32).
4. answers: kev_litert.to_answers on the reference's own probs == the reference's answers (the author's to_answers),
   exact; on the head-on-reference-hidden probs: decisions equal, numbers within one step of the 4th decimal.
5. author crosscheck: host_author_crosscheck.py --fixtures in the reference environment (edge requests + every
   fixture request and caller string vs the author's live code, with the 0.8B base's AutoTokenizer, which gives the
   4B base's ids too), embedded.
6. graph, CPU: 0.8B = KevLiteRT on the V2 files (L512 / L1024 / L2048), 8 threads: every request whose rows fit 512
   tokens (393 questions) through the host's own length choice, then the rest (9 questions, the 2,048-token graph),
   then a fixed sample of 50 of the 393 forced through the 1,024-token graph; 4B = the same sample of 50 through the
   1,024-token file at 6 threads. Every question's probs, z_post and z_pre must be bit-identical to the desktop gate's
   rows of the same file, runtime and thread count (results/litert_cpu_rows_L*[_4b]_v2_fp16fc_i8emb.json). Also
   reported: the bar numbers against the reference and (0.8B) every response (KevLiteRT.respond) against the
   reference's answers.
Memory: before a graph phase this run waits while reclaimable memory (vm_stat free + inactive + speculative + purgeable)
is low; every 20 rows it closes its graph and waits when it drops below 8 GB. One graph is compiled at a time."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

K = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K / "host"))
import kev_litert as host  # noqa: E402

MODELS = {
    "0.8b": {"tokenizer": "hf/hub/models--jaredpalmer--kev-0.8b/snapshots/788ddbdd65715bb03a56788c822f6c632c9a551d/tokenizer.json",
             "head": "host/kev_0.8b_pointer_head.safetensors", "prefix": "kev08b", "suffix": "", "lengths": (512, 1024, 2048),
             "oracle": "oracle/oracle_0.8b.json", "npz": "oracle/hidden_0.8b.npz", "out": "results/host_parity.json",
             "threads": 8, "wait_gb": 14},
    "4b": {"tokenizer": "hf/hub/models--jaredpalmer--kev-4b/snapshots/591dcb5bd6d05eb0b5131ea6608f93f10243335c/tokenizer.json",
           "head": "host/kev_4b_pointer_head.safetensors", "prefix": "kev4b", "suffix": "_4b", "lengths": (1024,),
           "oracle": "oracle/oracle_4b.json", "npz": "oracle/hidden_4b.npz", "out": "results/host_parity_4b.json",
           "threads": 6, "wait_gb": 30},
}
MODEL = next((sys.argv[i + 1] for i, x in enumerate(sys.argv[:-1]) if x == "--model"), "0.8b")
assert MODEL in MODELS, MODEL
_M = MODELS[MODEL]
TOKENIZER = K / _M["tokenizer"]
HEAD = K / _M["head"]
PROBES = K / "host/tokenizer_probes.json"
GRAPHS = {L: K / f"exports/{_M['prefix']}_rowprefill_L{L}_v2_fp16fc_i8emb.tflite" for L in _M["lengths"]}
ROWS = {L: K / f"results/litert_cpu_rows_L{L}{_M['suffix']}_v2_fp16fc_i8emb.json" for L in _M["lengths"]}
ORACLE_JSON, ORACLE_NPZ, FIXTURES = K / _M["oracle"], K / _M["npz"], K / "fixtures/requests.json"
OUT = K / _M["out"]
THREADS, SAMPLE_1024 = _M["threads"], 50
WAIT_GB, FLOOR_GB = _M["wait_gb"], 8


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 24):
            h.update(block)
    return h.hexdigest()


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def reclaimable_gb():
    """vm_stat free + inactive + speculative + purgeable, GB."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    page = int(re.search(r"page size of (\d+) bytes", out).group(1))
    pages = sum(int(re.search(rf"{name}:\s+(\d+)\.", out).group(1))
                for name in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"))
    return pages * page / 1e9


def swap_used_mb():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used = ([\d.]+)M", out)
    return float(m.group(1)) if m else None


def wait_for_memory(events, where, need=WAIT_GB, max_wait=3600):
    started = time.time()
    first = reclaimable_gb()
    while (now := reclaimable_gb()) < need:
        if time.time() - started > max_wait:
            raise SystemExit(f"{where}: reclaimable {now:.1f} GB < {need} GB for {max_wait} s; stopping")
        log(f"{where}: reclaimable {now:.1f} GB < {need} GB, waiting")
        time.sleep(30)
    events.append({"where": where, "reclaimable_gb_at_check": round(first, 1), "reclaimable_gb_at_go": round(now, 1),
                   "waited_s": round(time.time() - started, 1), "swap_used_mb": swap_used_mb()})


def footprint():
    """ru_maxrss and phys_footprint (proc_pid_rusage) of this process, bytes."""
    import ctypes
    doc = {"ru_maxrss": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}

    class RUsageInfoV4(ctypes.Structure):
        _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in (
            "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups", "ri_pageins", "ri_wired_size",
            "ri_resident_size", "ri_phys_footprint", "ri_proc_start_abstime", "ri_proc_exit_abstime", "ri_child_user_time",
            "ri_child_system_time", "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups", "ri_child_pageins",
            "ri_child_elapsed_abstime", "ri_diskio_bytesread", "ri_diskio_byteswritten", "ri_cpu_time_qos_default",
            "ri_cpu_time_qos_maintenance", "ri_cpu_time_qos_background", "ri_cpu_time_qos_utility",
            "ri_cpu_time_qos_legacy", "ri_cpu_time_qos_user_initiated", "ri_cpu_time_qos_user_interactive",
            "ri_billed_system_time", "ri_serviced_system_time", "ri_logical_writes", "ri_lifetime_max_phys_footprint",
            "ri_instructions", "ri_cycles", "ri_billed_energy", "ri_serviced_energy", "ri_interval_max_phys_footprint",
            "ri_runnable_time")]
    try:
        info = RUsageInfoV4()
        if ctypes.CDLL("/usr/lib/libproc.dylib").proc_pid_rusage(os.getpid(), 4, ctypes.byref(info)) == 0:
            doc.update(phys_footprint=int(info.ri_phys_footprint), lifetime_max_phys_footprint=int(info.ri_lifetime_max_phys_footprint))
    except Exception as e:  # informational
        doc["error"] = repr(e)
    return doc


def qkey(rid, qid):
    return f"{rid}/{qid}"


def numbers(answer):
    """Every number of one answer, by path (for |diff| between two answers of the same question)."""
    out = {}
    for k, v in answer.items():
        if isinstance(v, dict) and k == "probabilities":
            out.update({f"p[{kk}]": vv for kk, vv in v.items()})
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = v
    return out


def answer_diff(a, b):
    """Max |diff| over the numbers of two answers of one question (None if their shapes differ) and whether the
    decision (choice key, legend) is equal."""
    na, nb = numbers(a), numbers(b)
    if set(na) != set(nb) or a.get("type") != b.get("type"):
        return None, False
    decision = a.get("choice") == b.get("choice") and a.get("legend") == b.get("legend")
    return max((abs(na[k] - nb[k]) for k in na), default=0.0), decision


def bar_stats(dps, flips):
    pooled = np.concatenate([np.asarray(d, np.float64) for d in dps]) if dps else np.zeros(0)
    return {"questions": len(dps), "max_abs_dp": float(pooled.max()) if pooled.size else None,
            "mean_abs_dp_all_options": float(pooled.mean()) if pooled.size else None, "flips": flips}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--skip-graph", action="store_true", help="contract, head, answers and crosscheck only (development)")
    ap.add_argument("--skip-crosscheck", action="store_true")
    ap.add_argument("--model", choices=sorted(MODELS), default="0.8b", help="0.8b (default) or 4b (see docstring)")
    a = ap.parse_args()
    assert a.model == MODEL
    out = Path(a.out)
    assert not out.exists(), f"refusing to overwrite {out}"
    t_start = time.time()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    host_sha_at_start = sha256_file(Path(host.__file__))
    fixtures = json.loads(FIXTURES.read_text())
    records = fixtures["records"]
    oracle = json.loads(ORACLE_JSON.read_text())
    oq = {qkey(q["id"], q["qid"]): q for q in oracle["questions"]}
    oreq = {r["id"]: r for r in oracle["requests"]}
    tok = host.KevTokenizer(TOKENIZER)
    head = host.PointerHead(HEAD)
    doc = {"step": "Python host (host/kev_litert.py) parity" + (f" (model {MODEL})" if MODEL != "0.8b" else ""),
           "model": MODEL, "started_at": started_at,
           "platform": platform.platform(), "python": platform.python_version(),
           "packages": {p: importlib.metadata.version(p) for p in ("ai-edge-litert", "numpy", "tokenizers", "safetensors")}}

    # 1. tokenizer probes
    probes = json.loads(PROBES.read_text())
    plain = [tok._tokenizer.encode(p["text"], add_special_tokens=False).ids == p["ids"] for p in probes["probes"]]
    user = [tok.user_tokens(p["text"]) == p["user_ids"] for p in probes["probes"]]
    doc["tokenizer"] = {"file": str(TOKENIZER.relative_to(K)), "sha256": sha256_file(TOKENIZER),
                        "bytes": TOKENIZER.resolve().stat().st_size, "probes_file": str(PROBES.relative_to(K)),
                        "probes": len(plain), "plain_ids_equal": sum(plain), "user_ids_equal": sum(user),
                        "reference": probes["reference"], "base_repo_json_equal_for_the_record": probes["summary"]["base_repo_tokenizer_json_equal"]}
    assert all(plain) and all(user), doc["tokenizer"]
    log(f"tokenizer probes {sum(plain)}/{len(plain)} plain, {sum(user)}/{len(user)} user")

    # 2. contract
    enc = {r["id"]: host.encode_rows(tok, r["request"]) for r in records}
    ids_eq = idx_eq = keys_eq = legend_eq = 0
    first_diffs = []
    for r in records:
        for q in enc[r["id"]]["questions"]:
            ref = oq[qkey(r["id"], q["id"])]
            same_ids = q["row_ids"] == ref["row_ids"]
            same_idx = q["decide_idx"] == ref["decide_idx"] and q["opt_idx"] == ref["opt_idx"]
            same_keys = q["keys"] == ref["keys"] and q["type"] == ref["type"]
            same_legend = q.get("legend") == ref["answer"].get("legend")
            ids_eq, idx_eq, keys_eq, legend_eq = ids_eq + same_ids, idx_eq + same_idx, keys_eq + same_keys, legend_eq + same_legend
            if not (same_ids and same_idx and same_keys and same_legend) and len(first_diffs) < 3:
                first_diffs.append({"key": qkey(r["id"], q["id"]), "ids": same_ids, "idx": same_idx, "keys": same_keys,
                                    "legend": same_legend, "host_len": len(q["row_ids"]), "oracle_len": ref["row_len"]})
    usage_eq = sum(enc[r["id"]]["input_tokens"] == oreq[r["id"]]["usage"]["input_tokens"] for r in records)
    n_q = len(oracle["questions"])
    doc["contract"] = {"questions": n_q, "requests": len(records), "row_ids_equal": ids_eq, "readout_idx_equal": idx_eq,
                       "keys_and_type_equal": keys_eq, "score_legend_equal": legend_eq, "input_tokens_equal": usage_eq,
                       "first_differences": first_diffs,
                       "row_len": {"max": max(len(q["row_ids"]) for e in enc.values() for q in e["questions"]),
                                   "le_512": sum(len(q["row_ids"]) <= 512 for e in enc.values() for q in e["questions"])}}
    log(f"contract ids {ids_eq}/{n_q} idx {idx_eq}/{n_q} keys {keys_eq}/{n_q} legend {legend_eq}/{n_q} usage {usage_eq}/{len(records)}")
    if MODEL != "0.8b":   # one tokenizer: the 4B oracle's rows are the 0.8B oracle's rows
        o08 = {qkey(q["id"], q["qid"]): q for q in json.loads((K / MODELS["0.8b"]["oracle"]).read_text())["questions"]}
        same = sum(o08[k]["row_ids"] == q["row_ids"] and o08[k]["decide_idx"] == q["decide_idx"] and o08[k]["opt_idx"] == q["opt_idx"]
                   for k, q in oq.items())
        doc["contract"]["oracle_rows_equal_to_0.8b_oracle"] = same
        log(f"{MODEL} oracle rows equal to the 0.8B oracle's: {same}/{n_q}")
        assert same == n_q == len(o08), (same, n_q)
    if ids_eq != n_q or idx_eq != n_q:
        out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
        raise SystemExit(f"STOP: row ids {ids_eq}/{n_q}, idx {idx_eq}/{n_q}; first differences in {out}")

    # 3. head on the oracle's hidden states, 4. answers
    npz = np.load(ORACLE_NPZ)
    dp_max = dz_max = 0.0
    argmax_eq = 0
    head_probs = {}
    for k, ref in oq.items():
        z_pre, z_post, p = head(npz[k])
        head_probs[k] = p
        dp_max = max(dp_max, float(np.abs(p.astype(np.float64) - np.asarray(ref["probs"], np.float64)).max()))
        dz_max = max(dz_max, float(np.abs(z_post.astype(np.float64) - np.asarray(ref["z_post"], np.float64)).max()))
        argmax_eq += ref["keys"][int(np.argmax(p))] == ref["argmax_key"]
    doc["head"] = {"file": str(HEAD.relative_to(K)), "sha256": sha256_file(HEAD), "temperature": head.temperature,
                   "on": f"{ORACLE_NPZ.relative_to(K)} h_sel [decide, *opts] per question", "questions": n_q,
                   "max_abs_dp_vs_oracle": dp_max, "max_abs_dz_post_vs_oracle": dz_max, "argmax_equal": argmax_eq,
                   "pass_1e-6": dp_max <= 1e-6}
    log(f"head on oracle hidden: max|dp| {dp_max:.3e}, max|dz| {dz_max:.3e}, argmax {argmax_eq}/{n_q}")
    exact_q = sum(host.to_answers([oq[qkey(r['id'], q['id'])]["probs"]], [q])[q["id"]] == oq[qkey(r["id"], q["id"])]["answer"]
                  for r in records for q in enc[r["id"]]["questions"])
    exact_req = sum(host.to_answers([oq[qkey(r["id"], q["id"])]["probs"] for q in enc[r["id"]]["questions"]],
                                    enc[r["id"]]["questions"]) == oreq[r["id"]]["answers"] for r in records)
    head_diffs, head_exact, head_decision = [], 0, 0
    for r in records:
        for q in enc[r["id"]]["questions"]:
            k = qkey(r["id"], q["id"])
            ans = host.to_answers([head_probs[k].tolist()], [q])[q["id"]]
            d, decision = answer_diff(ans, oq[k]["answer"])
            head_diffs.append(d)
            head_exact += ans == oq[k]["answer"]
            head_decision += decision
    doc["answers"] = {"from_oracle_probs_equal_questions": exact_q, "from_oracle_probs_equal_requests": exact_req,
                      "requests": len(records), "questions": n_q,
                      "from_head_on_oracle_hidden": {"exact": head_exact, "decision_equal": head_decision,
                                                     "max_abs_diff_rounded_numbers": max(head_diffs)}}
    log(f"answers from oracle probs: {exact_q}/{n_q} questions, {exact_req}/{len(records)} requests exact; "
        f"from head on oracle hidden: exact {head_exact}, decision {head_decision}, max diff {max(head_diffs)}")

    # 5. author crosscheck (venv-oracle)
    if not a.skip_crosscheck:
        env = dict(os.environ, HF_HOME=str(K / "hf"), HF_HUB_OFFLINE="1", HF_HUB_DISABLE_XET="1", TOKENIZERS_PARALLELISM="false")
        proc = subprocess.run([str(K / "venv-oracle/bin/python"), str(K / "scripts/host_author_crosscheck.py"), "--fixtures"],
                              capture_output=True, text=True, env=env, check=True)
        cross = json.loads(proc.stdout)
        doc["author_crosscheck"] = {k: v for k, v in cross.items() if k != "edge"}
        doc["author_crosscheck"]["edge_requests_detail"] = [{k: r[k] for k in ("request", "questions", "row_lens")} for r in cross["edge"]]
        log(f"author crosscheck: {cross['requests_equal']}/{cross['requests']} requests, strings {cross['fixture_strings']}, "
            f"invalid refused by both {cross['invalid_refused_by_both']}/{cross['invalid_requests']}")

    if a.skip_graph:
        doc["graph"] = "skipped (--skip-graph)"
        out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
        log(f"wrote {out}")
        return

    # 6. graph
    rows_ref = {L: {r["key"]: r for r in json.loads(p.read_text())["rows"]} for L, p in ROWS.items()}
    kev = host.KevLiteRT({L: GRAPHS[L] for L in GRAPHS}, HEAD, TOKENIZER, accelerator="cpu", threads=THREADS)
    events, phases, responses = [], {}, {}
    short = [r for r in records if all(len(q["row_ids"]) <= 512 for q in enc[r["id"]]["questions"])]
    short_ids = {r["id"] for r in short}
    long_ = [r for r in records if r["id"] not in short_ids]

    def compare(q, rid, L, acc):
        k = qkey(rid, q["id"])
        ref, orc = rows_ref[L][k], oq[k]
        bit = (ref["probs"] == [float(x) for x in q["probs"]] and ref["z_post"] == [float(x) for x in q["z_post"]]
               and ref["z_pre"] == [float(x) for x in q["z_pre"]])
        dp = np.abs(q["probs"].astype(np.float64) - np.asarray(orc["probs"], np.float64))
        am = orc["keys"][int(np.argmax(q["probs"]))]
        acc["n"] += 1
        acc["bit_equal"] += bit
        acc["dps"].append(dp)
        acc["ms"].append(q["ms"])
        if am != orc["argmax_key"]:
            acc["flips"].append({"key": k, "argmax": am, "oracle": orc["argmax_key"], "oracle_top2_gap": orc["top2_gap"],
                                 "near_tie": orc["near_tie"]})
        if not bit and len(acc["not_bit_equal"]) < 5:
            acc["not_bit_equal"].append({"key": k, "probs": [float(x) for x in q["probs"]], "rows_probs": ref["probs"]})

    def run_phase(name, L, reqs=None, sample=None):
        wait_for_memory(events, f"{name} start")
        acc = {"n": 0, "bit_equal": 0, "dps": [], "ms": [], "flips": [], "not_bit_equal": [], "requests": 0}
        t0 = time.time()
        mem_before = footprint()
        done = 0
        if reqs is not None:
            for r in reqs:
                scored = kev.score(r["request"])
                assert {q["length"] for q in scored["questions"]} == {L}, (r["id"], L)
                for q in scored["questions"]:
                    compare(q, r["id"], L, acc)
                responses[r["id"]] = (kev.respond(scored), scored)
                acc["requests"] += 1
                done += len(scored["questions"])
                if done // 20 != (done - len(scored["questions"])) // 20:
                    if reclaimable_gb() < FLOOR_GB:
                        kev.close_graph(L)
                        wait_for_memory(events, f"{name} after {done} rows")
                    log(f"{name}: {done} rows, bit-equal {acc['bit_equal']}/{acc['n']}")
        else:
            for i, (rid, q) in enumerate(sample):
                q = dict(q, length=L)
                kev.readout(q)
                compare(q, rid, L, acc)
                if (i + 1) % 20 == 0:
                    if reclaimable_gb() < FLOOR_GB:
                        kev.close_graph(L)
                        wait_for_memory(events, f"{name} after {i + 1} rows")
                    log(f"{name}: {i + 1} rows, bit-equal {acc['bit_equal']}/{acc['n']}")
        graph = kev._graphs.get(L)
        compile_s = round(graph.compile_seconds, 2) if graph is not None else None
        mem_after = footprint()
        kev.close_graph(L)
        phases[name] = {"L": L, "graph": str(GRAPHS[L].relative_to(K)), "rows_file": str(ROWS[L].relative_to(K)),
                        "questions": acc["n"], "requests": acc["requests"] or None, "bit_equal_probs_z_post_z_pre": acc["bit_equal"],
                        "not_bit_equal_first": acc["not_bit_equal"], "vs_oracle": bar_stats(acc["dps"], acc["flips"]),
                        "compile_seconds_last": compile_s, "row_ms_median": float(np.median(acc["ms"])),
                        "row_ms_max": float(np.max(acc["ms"])), "seconds": round(time.time() - t0, 1),
                        "memory_before": mem_before, "memory_after": mem_after, "swap_used_mb_end": swap_used_mb()}
        log(f"{name} done: bit-equal {acc['bit_equal']}/{acc['n']}, {phases[name]['seconds']} s")
        if acc["bit_equal"] != acc["n"]:
            doc["graph"] = {"phases": phases, "memory_events": events, "stopped": f"{name}: not bit-equal"}
            out.write_text(json.dumps(doc, indent=1, ensure_ascii=False, default=str) + "\n")
            raise SystemExit(f"STOP: {name} probs not bit-equal to {ROWS[L].name} ({acc['bit_equal']}/{acc['n']})")

    if 512 in GRAPHS:
        run_phase("L512", 512, reqs=short)
    if 2048 in GRAPHS:
        run_phase("L2048", 2048, reqs=long_)
    pool = [(r["id"], q) for r in short for q in enc[r["id"]]["questions"]]
    picks = sorted({round(i * (len(pool) - 1) / (SAMPLE_1024 - 1)) for i in range(SAMPLE_1024)})
    run_phase("L1024_sample", 1024, sample=[pool[i] for i in picks])
    kev.close()

    # informational: the example request at the host's default 4 CPU threads against the 8-thread rows (0.8B only)
    threads4 = None
    if 512 in GRAPHS:
        wait_for_memory(events, "threads4 start")
        example_rec = next(r for r in records if r["id"] == "own_ticket_01")
        with host.KevLiteRT({512: GRAPHS[512]}, HEAD, TOKENIZER, accelerator="cpu", threads=4) as kev4:
            scored4 = kev4.score(example_rec["request"])
            resp4 = kev4.respond(scored4)
        refs4 = [rows_ref[512][qkey(example_rec["id"], q["id"])]["probs"] for q in scored4["questions"]]
        threads4 = {"request": example_rec["id"], "threads": 4, "questions": len(scored4["questions"]),
                    "probs_bit_equal_to_8_thread_rows": sum(r == [float(x) for x in q["probs"]] for r, q in zip(refs4, scored4["questions"])),
                    "max_abs_dp_vs_8_thread_rows": max(float(np.abs(q["probs"].astype(np.float64) - np.asarray(r)).max())
                                                       for r, q in zip(refs4, scored4["questions"])),
                    "answers_equal_to_8_thread_response": resp4["answers"] == responses[example_rec["id"]][0]["answers"]}
        log(f"threads 4 example: {threads4}")

    # responses vs the desktop gate's rows (same probs -> same answers) and vs the oracle's answers
    resp_vs_rows = resp_usage = resp_model = 0
    oracle_cmp = {"exact": 0, "decision_equal": 0, "diffs": [], "decision_differs": []}
    for rid, (resp, scored) in responses.items():
        L = scored["questions"][0]["length"]
        from_rows = host.to_answers([rows_ref[L][qkey(rid, q["id"])]["probs"] for q in scored["questions"]], scored["questions"])
        resp_vs_rows += resp["answers"] == from_rows
        resp_usage += resp["usage"]["input_tokens"] == oreq[rid]["usage"]["input_tokens"]
        resp_model += resp["model"] == host.DEFAULT_MODEL
        for q in scored["questions"]:
            ref = oq[qkey(rid, q["id"])]
            d, decision = answer_diff(resp["answers"][q["id"]], ref["answer"])
            oracle_cmp["exact"] += resp["answers"][q["id"]] == ref["answer"]
            oracle_cmp["decision_equal"] += decision
            oracle_cmp["diffs"].append(d)
            if not decision:
                oracle_cmp["decision_differs"].append({"key": qkey(rid, q["id"]), "host": resp["answers"][q["id"]].get("choice"),
                                                       "oracle": ref["answer"].get("choice"), "oracle_top2_gap": ref["top2_gap"]})
    example = responses.get("own_ticket_01", (None,))[0]
    graph_files = {L: {"path": str(p.relative_to(K)), "bytes": p.stat().st_size, "sha256": sha256_file(p)} for L, p in GRAPHS.items()}
    doc["graph"] = {
        "runtime": {"ai_edge_litert": importlib.metadata.version("ai-edge-litert"), "api": "CompiledModel.from_file",
                    "accelerator": "CPU", "threads": THREADS},
        "files": graph_files, "rows_files": {L: {"path": str(p.relative_to(K)), "sha256": sha256_file(p)} for L, p in ROWS.items()},
        "phases": phases,
        "bit_equal_total": sum(p["bit_equal_probs_z_post_z_pre"] for p in phases.values()),
        "questions_total": sum(p["questions"] for p in phases.values()),
        "responses": {"requests": len(responses), "answers_equal_to_rows_probs": resp_vs_rows, "input_tokens_equal_oracle": resp_usage,
                      "model_field": resp_model,
                      "vs_oracle_answers": {"questions": len(oracle_cmp["diffs"]), "exact": oracle_cmp["exact"],
                                            "decision_equal": oracle_cmp["decision_equal"],
                                            "max_abs_diff_rounded_numbers": max(oracle_cmp["diffs"]) if oracle_cmp["diffs"] else None,
                                            "decision_differs": oracle_cmp["decision_differs"]},
                      "example_own_ticket_01": example},
        "threads4_example": threads4,
        "memory_events": events,
        "timing_condition": "contended (other work shares this Mac); informational, not a card number",
    }
    doc["seconds_wall"] = round(time.time() - t_start, 1)
    doc["host_file"] = {"path": "host/kev_litert.py", "sha256": host_sha_at_start,
                        "unchanged_during_run": sha256_file(Path(host.__file__)) == host_sha_at_start}
    doc["inputs"] = {"oracle_json_sha256": sha256_file(ORACLE_JSON), "oracle_npz_sha256": sha256_file(ORACLE_NPZ),
                     "fixtures_sha256": sha256_file(FIXTURES)}
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False, default=str) + "\n")
    log(f"wrote {out} ({doc['seconds_wall']} s); graph bit-equal {doc['graph']['bit_equal_total']}/{doc['graph']['questions_total']}")


if __name__ == "__main__":
    main()
