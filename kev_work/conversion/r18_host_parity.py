"""r16_host_parity.py with one addition: --row-tag "64,128,256=<tag>" takes the gate rows of those buckets from
results/litert_{cpu,gpu_f32}_rows_L<L>_<tag>.json. The published 64-, 128- and 256-token files are form C7 (tag
v2_fp16fc_i8emb_r17C7-bkzq); the 512- to 2,048-token files and the pairs are form C (v2_fp16fc_i8emb_r14B-vs6).

    $HOST scripts/r18_host_parity.py --repo $R --phase row  --accel cpu --threads 4 --row-tag 64,128,256=v2_fp16fc_i8emb_r17C7-bkzq --suffix _C7
    $HOST scripts/r18_host_parity.py --repo $R --phase row  --accel gpu             --row-tag 64,128,256=v2_fp16fc_i8emb_r17C7-bkzq --suffix _C7
    $HOST scripts/r18_host_parity.py --repo $R --phase auto --accel cpu --threads 4 --row-tag 64,128,256=v2_fp16fc_i8emb_r17C7-bkzq --suffix _C7

row   KevLiteRT.from_dir(<repository>, pair_shapes=(), mode="row"): every fixture request is encoded and routed by the
      host (each question to the smallest of L64..L2048 that holds its row); the questions run bucket by bucket through
      KevLiteRT.readout, one compiled graph at a time; probs / z_post / z_pre must equal the gate rows of that bucket and
      accelerator bit for bit. Then a fixed sample of 50 questions forced through L1024, which the host's own choice never
      uses. Also: row ids / readout indices against the oracle, the bar numbers against the oracle, the control, and
      every request's response, stored for the auto phase.
pair  for Ls in (128, 256) and handover in (direct, host): every question that fits the pair runs through it; probs must
      equal the pair gate rows (results/litert_shared_{cpu,gpu_f32,gpu_f32_share}_rows_Ls<Ls>_Lq64_<tag>.json).
auto  KevLiteRT.from_dir(<repository>) with the host's defaults: all 377 requests through score() / respond(); the routing
      table and every question's decision and answer against the row phase's response of the same accelerator; the pair
      questions against the pair gate rows and the row questions against the row gate rows.
Outputs (never overwritten): results/host_parity_v2_<phase>_<accel>[_share-on|off][_t<threads>]<suffix>.json. The host
file's sha256 is recorded at the start and checked at the end."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np

K = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K / "host"))
import kev_litert as H  # noqa: E402

MODELS = {   # --model: the repository, the tag of the gate rows' file names, the oracle
    "0.8b": {"repo": "staging_v2/Kev-0.8B-LiteRT", "tag": "v2_fp16fc_i8emb_r14B-vs6", "oracle": "oracle/oracle_0.8b.json"},
    "4b": {"repo": "staging_v2/Kev-4B-LiteRT", "tag": "4b_v2_fp16fc_i8emb_r15R64-sp-ec-dd-vs6-in1-fn5",
           "oracle": "oracle/oracle_4b.json"},
}
STAGE = K / MODELS["0.8b"]["repo"]     # --repo overrides: the repository root (graphs, head/, tokenizer/)
TAG = MODELS["0.8b"]["tag"]
ORACLE = MODELS["0.8b"]["oracle"]
ROW_TAGS = {}   # --row-tag: bucket -> tag of its gate rows (default TAG)
SAMPLE_1024 = 50


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 24):
            h.update(block)
    return h.hexdigest()


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def footprint():
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
            doc.update(phys_footprint=int(info.ri_phys_footprint),
                       lifetime_max_phys_footprint=int(info.ri_lifetime_max_phys_footprint))
    except Exception as e:  # informational
        doc["error"] = repr(e)
    return doc


def fl(a):
    return [float(x) for x in a]


class Oracle:
    def __init__(self):
        o = json.loads((K / ORACLE).read_text())
        self.q = {f"{q['id']}/{q['qid']}": q for q in o["questions"]}
        self.req = {r["id"]: r for r in o["requests"]}
        self.tv4_000 = next(q for q in o["questions"] if q["id"] == "tv4_000")

    def stats(self, probs):
        """{key: probs} -> the bar numbers against the oracle (control apart) and the flips."""
        dps, flips, red = [], [], None
        nn = nn_ok = near = near_ok = 0
        for key, p in probs.items():
            ref = self.q[key]
            p = np.asarray(p, np.float64)
            if key.startswith("red_arm_000/"):
                red = float(np.abs(p - np.asarray(self.tv4_000["probs"], np.float64)).max())
                continue
            d = np.abs(p - np.asarray(ref["probs"], np.float64))
            dps.append(d)
            am = ref["keys"][int(np.argmax(p))]
            ok = am == ref["argmax_key"]
            if ref["near_tie"]:
                near, near_ok = near + 1, near_ok + ok
            else:
                nn, nn_ok = nn + 1, nn_ok + ok
            if not ok:
                flips.append({"key": key, "argmax": am, "oracle": ref["argmax_key"], "oracle_top2_gap": ref["top2_gap"],
                              "near_tie": ref["near_tie"]})
        pooled = np.concatenate(dps) if dps else np.zeros(0)
        return {"questions": len(dps), "non_near_tie_argmax": f"{nn_ok}/{nn}", "near_tie_argmax": f"{near_ok}/{near}",
                "max_abs_dp": float(pooled.max()) if pooled.size else None,
                "mean_abs_dp_all_options": float(pooled.mean()) if pooled.size else None,
                "control_red_arm_max_abs_dp_vs_tv4_000": red, "flips": flips}


def rtag(L):
    return ROW_TAGS.get(L, TAG)


def load_rows(path):
    return {r["key"]: r for r in json.loads(path.read_text())["rows"]}


def decision(p):
    return int(np.argmax(np.asarray(p)))


def out_path(a):
    name = "host_parity_v2_" + ("4b_" if a.model == "4b" else "") + f"{a.phase}_{a.accel}"
    if a.force_L:
        name += f"_forceL{a.force_L}_n{a.limit}"
    if a.accel == "gpu" and a.phase == "pair":
        name += f"_share-{a.share}"
    if a.accel == "cpu":
        name += f"_t{a.threads}"
    return K / "results" / f"{name}{a.suffix}.json"


def host_kwargs(a):
    return {"accelerator": a.accel, "threads": a.threads, "precision": "fp32"}


def phase_row(a, doc, oracle, records):
    acc = "cpu" if a.accel == "cpu" else "gpu_f32"
    rows_ref = {L: load_rows(K / f"results/litert_{acc}_rows_L{L}_{rtag(L)}.json") for L in H.LENGTHS
                if (K / f"results/litert_{acc}_rows_L{L}_{rtag(L)}.json").exists()}
    doc["gate_rows"] = {L: f"results/litert_{acc}_rows_L{L}_{rtag(L)}.json" for L in rows_ref}
    kev = H.KevLiteRT.from_dir(STAGE, pair_shapes=(), mode="row", **host_kwargs(a))
    if a.force_L:   # a forced bucket and a few rows (the 4B CPU check): no routing, no response, no L1024 sample
        return phase_row_forced(a, doc, oracle, records, kev, rows_ref)
    encs, contract = {}, {"questions": 0, "row_ids_equal": 0, "readout_idx_equal": 0, "input_tokens_equal": 0}
    by_L = {}
    for r in records:
        enc = kev.encode_rows(r["request"])
        kev.route(enc)
        encs[r["id"]] = enc
        contract["input_tokens_equal"] += enc["input_tokens"] == oracle.req[r["id"]]["usage"]["input_tokens"]
        for q in enc["questions"]:
            key = f"{r['id']}/{q['id']}"
            ref = oracle.q[key]
            contract["questions"] += 1
            contract["row_ids_equal"] += q["row_ids"] == ref["row_ids"]
            contract["readout_idx_equal"] += q["decide_idx"] == ref["decide_idx"] and q["opt_idx"] == ref["opt_idx"]
            by_L.setdefault(q["length"], []).append((key, q))
    doc["contract"] = contract
    doc["route_buckets"] = {L: len(v) for L, v in sorted(by_L.items())}
    log(f"contract {contract}; buckets {doc['route_buckets']}")
    phases, probs = {}, {}
    for L in sorted(by_L):
        t0 = time.time()
        acc_ = {"n": 0, "bit_equal": 0, "not_equal": [], "ms": []}
        for key, q in by_L[L]:
            kev.readout(q)
            ref = rows_ref[L][key]
            bit = (ref["probs"] == fl(q["probs"]) and ref["z_post"] == fl(q["z_post"]) and ref["z_pre"] == fl(q["z_pre"]))
            acc_["n"] += 1
            acc_["bit_equal"] += bit
            acc_["ms"].append(q["ms"])
            if not bit and len(acc_["not_equal"]) < 5:
                acc_["not_equal"].append({"key": key, "max_abs_dp_vs_gate": float(np.abs(np.asarray(q["probs"], np.float64)
                                                                                        - np.asarray(ref["probs"])).max())})
            probs[key] = fl(q["probs"])
        g = kev._graphs.get(L)
        phases[f"L{L}"] = {"L": L, "file": kev._paths[L].name, "questions": acc_["n"], "bit_equal_probs_z_post_z_pre": acc_["bit_equal"],
                           "not_equal_first": acc_["not_equal"], "compile_seconds": round(g.compile_seconds, 2) if g else None,
                           "fully_accelerated": g.fully_accelerated if g else None,
                           "row_ms_median": float(np.median(acc_["ms"])), "seconds": round(time.time() - t0, 1),
                           "memory": footprint()}
        kev.close_graph(L)
        log(f"L{L}: {acc_['bit_equal']}/{acc_['n']} bit-equal")
    # the L1024 sample (host_parity.py's picks): the host's own choice never routes a question to L1024
    short = [r for r in records if all(len(q["row_ids"]) <= 512 for q in encs[r["id"]]["questions"])]
    pool = [(f"{r['id']}/{q['id']}", q) for r in short for q in encs[r["id"]]["questions"]]
    picks = sorted({round(i * (len(pool) - 1) / (SAMPLE_1024 - 1)) for i in range(SAMPLE_1024)})
    t0 = time.time()
    n = be = 0
    sample_probs = {}
    for i in picks:
        key, q = pool[i]
        q2 = dict(q, length=1024)
        kev.readout(q2)
        ref = rows_ref[1024][key]
        n += 1
        be += (ref["probs"] == fl(q2["probs"]) and ref["z_post"] == fl(q2["z_post"]) and ref["z_pre"] == fl(q2["z_pre"]))
        sample_probs[key] = fl(q2["probs"])
    g = kev._graphs.get(1024)
    phases["L1024_sample"] = {"L": 1024, "file": kev._paths[1024].name, "questions": n, "bit_equal_probs_z_post_z_pre": be,
                              "compile_seconds": round(g.compile_seconds, 2) if g else None,
                              "fully_accelerated": g.fully_accelerated if g else None, "seconds": round(time.time() - t0, 1),
                              "vs_oracle": oracle.stats(sample_probs)}
    kev.close_graph(1024)
    log(f"L1024 sample: {be}/{n} bit-equal")
    responses = {}
    for r in records:
        enc = encs[r["id"]]
        enc["latency_ms"] = 0.0
        responses[r["id"]] = kev.respond(enc)
    kev.close()
    doc["phases"] = phases
    doc["bit_equal_total"] = sum(p["bit_equal_probs_z_post_z_pre"] for p in phases.values())
    doc["questions_total"] = sum(p["questions"] for p in phases.values())
    doc["vs_oracle_routed"] = oracle.stats(probs)
    doc["probs"] = probs
    doc["responses"] = {rid: {k: v for k, v in resp.items() if k != "latency_ms"} for rid, resp in responses.items()}
    # decisions of the responses against the oracle's answers (argmax of the unrounded probs; the 4-decimal answers
    # differ from the oracle's by the V2 quantization)
    dec_eq, differs = 0, []
    for key, p in probs.items():
        ref = oracle.q[key]
        if ref["keys"][decision(p)] == ref["argmax_key"]:
            dec_eq += 1
        else:
            differs.append({"key": key, "oracle_top2_gap": ref["top2_gap"], "near_tie": ref["near_tie"]})
    doc["decision_vs_oracle"] = {"questions": len(probs), "equal": dec_eq, "differs": differs}
    return doc["bit_equal_total"] == doc["questions_total"]


def phase_row_forced(a, doc, oracle, records, kev, rows_ref):
    """The first --limit questions whose rows fit --force-L, forced through that graph: probs vs the gate rows of that
    bucket and accelerator when they exist (bit for bit), else vs the other accelerator's rows of the same file and the
    oracle (informational)."""
    L = a.force_L
    acc = "cpu" if a.accel == "cpu" else "gpu_f32"
    other = "gpu_f32" if acc == "cpu" else "cpu"
    other_path = K / f"results/litert_{other}_rows_L{L}_{rtag(L)}.json"
    other_rows = load_rows(other_path) if other_path.exists() else {}
    picked = []
    for r in records:
        enc = kev.encode_rows(r["request"])
        for q in enc["questions"]:
            if len(q["row_ids"]) <= L and len(picked) < a.limit:
                picked.append((f"{r['id']}/{q['id']}", dict(q, length=L)))
    probs, bit, dps_other = {}, 0, []
    for key, q in picked:
        kev.readout(q)
        probs[key] = fl(q["probs"])
        if L in rows_ref:
            ref = rows_ref[L][key]
            bit += (ref["probs"] == probs[key] and ref["z_post"] == fl(q["z_post"]) and ref["z_pre"] == fl(q["z_pre"]))
        if key in other_rows:
            dps_other.append(float(np.abs(np.asarray(probs[key]) - np.asarray(other_rows[key]["probs"])).max()))
    g = kev._graphs.get(L)
    doc["forced"] = {"L": L, "questions": len(picked), "gate_rows_same_accelerator": L in rows_ref,
                     "bit_equal_probs_z_post_z_pre": bit if L in rows_ref else None,
                     "vs_other_accelerator_rows": {"file": other_path.name if other_rows else None,
                                                   "max_abs_dp": max(dps_other) if dps_other else None,
                                                   "questions": len(dps_other)},
                     "vs_oracle": oracle.stats(probs), "compile_seconds": round(g.compile_seconds, 2) if g else None,
                     "memory": footprint()}
    doc["probs"] = probs
    kev.close()
    return (bit == len(picked)) if L in rows_ref else True


def phase_pair(a, doc, oracle, records):
    acc = "cpu" if a.accel == "cpu" else ("gpu_f32_share" if a.share == "on" else "gpu_f32")
    row_run = K / ("results/host_parity_v2_" + ("4b_" if a.model == "4b" else "") + "row_" + a.accel
                   + (f"_t{a.threads}" if a.accel == "cpu" else "") + a.suffix + ".json")
    row_probs = json.loads(row_run.read_text())["probs"] if row_run.exists() else None
    doc["row_phase_file"] = str(row_run.relative_to(K)) if row_run.exists() else None
    multi = [r["id"] for r in records if len(r["request"]["questions"]) > 1]
    runs, ok = {}, True
    for Ls in sorted({H.pair_shape(f)[0] for f in STAGE.glob(H.PAIR_GLOB)}):
        ref_path = K / f"results/litert_shared_{acc}_rows_Ls{Ls}_Lq64_{TAG}.json"
        ref = load_rows(ref_path)
        for handover in ("direct", "host"):
            t0 = time.time()
            kev = H.KevLiteRT.from_dir(STAGE, pair_shapes=[(Ls, 64)], mode="auto", pair_ratio=1e-9, handover=handover,
                                       constant_tensor_sharing=(a.share == "on"), **host_kwargs(a))
            probs, z_post, n_req, state_ms, q_ms = {}, {}, 0, [], []
            for r in records:
                enc = kev.encode_rows(r["request"])
                kev.route(enc)
                if enc["route"]["pair"] is None:
                    continue
                kev.readout_pair(enc)
                n_req += 1
                state_ms.append(enc["pair_state_ms"])
                for q in enc["questions"]:
                    if q["path"] == "pair":
                        key = f"{r['id']}/{q['id']}"
                        probs[key], z_post[key] = fl(q["probs"]), fl(q["z_post"])
                        q_ms.append(q["ms"])
            pair = kev._pairs.get((Ls, 64))
            pair_file = kev._pair_paths[(Ls, 64)].name
            compile_s = round(pair.compile_seconds, 2) if pair else None
            fully = pair.fully_accelerated if pair else None
            mem = footprint()
            kev.close()
            field = "probs" if handover == "host" else "probs_direct"
            keys_equal = set(probs) == set(ref)
            bit_p = sum(probs[k] == ref[k][field] for k in probs if k in ref)
            bit_z = sum(z_post[k] == ref[k]["z_post"] for k in z_post if k in ref)
            vs_row = None
            if row_probs is not None:
                d = [float(np.abs(np.asarray(probs[k]) - np.asarray(row_probs[k])).max()) for k in probs]
                vs_row = {"questions": len(d), "max_abs_dp": max(d), "argmax_equal": sum(decision(probs[k]) == decision(row_probs[k])
                                                                                        for k in probs)}
            multi_rows = {rid: {"questions": sum(1 for k in probs if k.startswith(rid + "/")),
                                "bit_equal": sum(1 for k in probs if k.startswith(rid + "/") and probs[k] == ref[k][field])}
                          for rid in multi if any(k.startswith(rid + "/") for k in probs)}
            run = {"Ls": Ls, "Lq": 64, "handover": handover, "file": pair_file,
                   "gate_rows": str(ref_path.relative_to(K)), "gate_field": field, "requests": n_req,
                   "questions": len(probs), "gate_questions": len(ref), "question_set_equal_to_gate": keys_equal,
                   "bit_equal_probs": bit_p, "bit_equal_z_post": bit_z, "multi_question_requests": multi_rows,
                   "vs_row_phase": vs_row, "vs_oracle": oracle.stats(probs), "compile_seconds": compile_s,
                   "fully_accelerated": fully, "state_ms_median": float(np.median(state_ms)) if state_ms else None,
                   "question_ms_median": float(np.median(q_ms)) if q_ms else None, "seconds": round(time.time() - t0, 1),
                   "memory_end_of_run": mem, "probs": probs}
            runs[f"Ls{Ls}_{handover}"] = run
            ok &= keys_equal and bit_p == len(probs) == len(ref)
            log(f"Ls{Ls} {handover}: {len(probs)} questions (gate {len(ref)}), bit-equal probs {bit_p}, z_post {bit_z}; "
                f"vs row {vs_row and vs_row['max_abs_dp']}")
    # direct vs host inside this host
    for Ls in sorted({int(k[2:].split("_")[0]) for k in runs}):
        pd, ph = runs[f"Ls{Ls}_direct"]["probs"], runs[f"Ls{Ls}_host"]["probs"]
        runs[f"Ls{Ls}_direct"]["direct_equal_host_in_this_host"] = sum(pd[k] == ph[k] for k in pd)
    doc["runs"] = runs
    return ok


def phase_auto(a, doc, oracle, records):
    acc_row = "cpu" if a.accel == "cpu" else "gpu_f32"
    acc_pair = "cpu" if a.accel == "cpu" else "gpu_f32_share"
    row_run = K / ("results/host_parity_v2_" + ("4b_" if a.model == "4b" else "") + "row_" + a.accel
                   + (f"_t{a.threads}" if a.accel == "cpu" else "") + a.suffix + ".json")
    row_doc = json.loads(row_run.read_text())
    rows_ref = {L: load_rows(K / f"results/litert_{acc_row}_rows_L{L}_{rtag(L)}.json") for L in H.LENGTHS}
    pair_ref = {Ls: load_rows(K / f"results/litert_shared_{acc_pair}_rows_Ls{Ls}_Lq64_{TAG}.json")
                for Ls in sorted({H.pair_shape(f)[0] for f in STAGE.glob(H.PAIR_GLOB)})}
    kev = H.KevLiteRT.from_dir(STAGE, **host_kwargs(a))
    doc["host_settings"] = {"mode": kev.mode, "pair_ratio": kev.pair_ratio, "handover": kev.handover,
                            "constant_tensor_sharing": kev.constant_tensor_sharing, "lengths": kev.lengths,
                            "pair_shapes": kev.pair_shapes}
    plans = []
    for r in records:
        enc = kev.encode_rows(r["request"])
        rt = kev.route(enc)
        needs = ({("pair",) + tuple(rt["pair"])} if rt["pair"] else set()) | {("row", q["length"]) for q in enc["questions"]
                                                                             if q["path"] == "row"}
        plans.append((r, rt, needs, [q["length"] for q in enc["questions"] if q["path"] == "row"]))
    order = sorted(range(len(plans)), key=lambda i: (sorted(plans[i][2]), i))
    remaining = {}
    for i in order:
        for nd in plans[i][2]:
            remaining[nd] = remaining.get(nd, 0) + 1
    table, probs, answers_equal, dec_eq, dec_n, differs = [], {}, 0, 0, 0, []
    bit_pair = bit_row = n_pair = n_row = 0
    row_lengths = []
    t0 = time.time()
    for i in order:
        r, _, needs, _ = plans[i]
        scored = kev.score(r["request"])
        resp = kev.respond(scored)
        rt = scored["route"]
        kind = "pair" if rt["row_questions"] == 0 else ("row" if rt["pair_questions"] == 0 else "mixed")
        table.append({"request": r["id"], "questions": len(scored["questions"]), "state_tokens": scored["state_tokens"],
                      "route": kind, "pair": rt["pair"], "pair_questions": rt["pair_questions"],
                      "row_questions": rt["row_questions"], "row_positions": rt.get("row_positions"),
                      "pair_positions": rt.get("pair_positions"),
                      "row_lengths": sorted({q["length"] for q in scored["questions"] if q["path"] == "row"})})
        answers_equal += resp["answers"] == row_doc["responses"][r["id"]]["answers"]
        for q in scored["questions"]:
            key = f"{r['id']}/{q['id']}"
            p = fl(q["probs"])
            probs[key] = p
            dec_n += 1
            same = decision(p) == decision(row_doc["probs"][key])
            dec_eq += same
            if not same:
                differs.append({"key": key, "path": q["path"], "oracle_top2_gap": oracle.q[key]["top2_gap"]})
            if q["path"] == "pair":
                n_pair += 1
                ref = pair_ref[q["pair"][0]].get(key)
                bit_pair += ref is not None and ref["probs_direct"] == p
            else:
                n_row += 1
                row_lengths.append(q["length"])
                bit_row += rows_ref[q["length"]][key]["probs"] == p
        for nd in needs:
            remaining[nd] -= 1
            if remaining[nd] == 0:
                if nd[0] == "pair":
                    kev.close_pair(nd[1:])
                else:
                    kev.close_graph(nd[1])
    kev.close()
    from collections import Counter
    doc["routing"] = {
        "requests": len(table),
        "by_route": dict(Counter(t["route"] for t in table)),
        "by_route_and_pair": dict(Counter(f"{t['route']} {('Ls%d_Lq%d' % tuple(t['pair'])) if t['pair'] else '-'}" for t in table)),
        "questions_by_path": {"pair": n_pair, "row": n_row},
        "row_questions_by_length": dict(sorted(Counter(row_lengths).items())),
        "not_row_only": [t for t in table if t["route"] != "row"],
    }
    doc["decisions_vs_row_phase"] = {"questions": dec_n, "equal": dec_eq, "differs": differs,
                                     "row_phase_file": str(row_run.relative_to(K))}
    doc["answers_equal_to_row_phase_requests"] = f"{answers_equal}/{len(table)}"
    doc["max_abs_dp_vs_row_phase"] = max(float(np.abs(np.asarray(probs[k]) - np.asarray(row_doc["probs"][k])).max())
                                         for k in probs)
    doc["gate_bit_equal"] = {"pair_questions": f"{bit_pair}/{n_pair}", "row_questions": f"{bit_row}/{n_row}",
                             "pair_gate_field": f"probs_direct of litert_shared_{acc_pair}_rows_*",
                             "row_gate": f"litert_{acc_row}_rows_*"}
    doc["vs_oracle"] = oracle.stats(probs)
    doc["table"] = table
    doc["seconds_requests"] = round(time.time() - t0, 1)
    return dec_eq == dec_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["row", "pair", "auto"], required=True)
    ap.add_argument("--accel", choices=["cpu", "gpu"], required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--share", choices=["on", "off"], default="on", help="pair phase on the GPU: constant_tensor_sharing")
    ap.add_argument("--repo", default="", help="the repository root with the graph files (default: the model's staging_v2 folder)")
    ap.add_argument("--model", choices=sorted(MODELS), default="0.8b")
    ap.add_argument("--force-L", type=int, default=0, help="row phase: run the first --limit questions that fit this L through it")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--row-tag", default="", help="L,L,...=<tag>: the gate rows of those buckets carry this tag")
    ap.add_argument("--suffix", default="", help="appended to the output name (and to the row phase file the pair / auto "
                                                 "phases read)")
    a = ap.parse_args()
    global STAGE, TAG, ORACLE
    STAGE, TAG, ORACLE = K / MODELS[a.model]["repo"], MODELS[a.model]["tag"], MODELS[a.model]["oracle"]
    if a.row_tag:
        Ls_, t_ = a.row_tag.split("=")
        ROW_TAGS.update({int(x): t_ for x in Ls_.split(",")})
    if a.repo:
        STAGE = Path(a.repo).resolve()
    out = out_path(a)
    assert not out.exists(), f"refusing to overwrite {out}"
    host_file = Path(H.__file__)
    sha_start = sha256_file(host_file)
    records = json.loads((K / "fixtures/requests.json").read_text())["records"]
    oracle = Oracle()
    doc = {"step": f"host parity, phase {a.phase}, {a.accel}", "row_tags": {str(k): v for k, v in ROW_TAGS.items()},
           "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "argv": sys.argv[1:], "platform": platform.platform(), "python": platform.python_version(),
           "packages": {p: importlib.metadata.version(p) for p in ("ai-edge-litert", "numpy", "tokenizers", "safetensors")},
           "staging": str(STAGE), "host_file": {"path": str(host_file.relative_to(K)), "sha256_start": sha_start},
           "accelerator": a.accel, "threads": a.threads if a.accel == "cpu" else None,
           "gpu": {"precision": "fp32 (GpuOptions(enforce_f32=True))", "constant_tensor_sharing": a.share} if a.accel == "gpu" else None}
    t0 = time.time()
    ok = {"row": phase_row, "pair": phase_pair, "auto": phase_auto}[a.phase](a, doc, oracle, records)
    doc["seconds_wall"] = round(time.time() - t0, 1)
    doc["host_file"]["sha256_end"] = sha256_file(host_file)
    doc["host_file"]["unchanged_during_run"] = doc["host_file"]["sha256_end"] == sha_start
    doc["pass"] = bool(ok and doc["host_file"]["unchanged_during_run"])
    doc["memory_end"] = footprint()
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False, default=str) + "\n")
    log(f"wrote {out.relative_to(K)} pass={doc['pass']} ({doc['seconds_wall']} s)")
    raise SystemExit(0 if doc["pass"] else 1)


if __name__ == "__main__":
    main()
