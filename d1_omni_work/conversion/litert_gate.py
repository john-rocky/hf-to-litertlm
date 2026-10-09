"""Round 3 steps 3-4: the D1Decision graph on the Mac through CompiledModel (CPU XNNPACK 8 threads / Metal) against
the oracle (round 2's ref/records_ref.json) or, before the oracle exists, against the eager module; and the eager
reference itself.

    cd d1_omni_work; Q=~/code/standup/tools/quiet
    $Q/quiet_wait.py -- venv-ref/bin/python scripts/litert_gate.py --eager --L 256      # eager reference, checkpoint
    $Q/quiet_wait.py -- ~/venvs/lt094dev/bin/python scripts/litert_gate.py --backend cpu --L 256 --form fp32
    $Q/quiet_hold.py d1c-r3-gpu-fp32-256-fp32 -- ~/venvs/lt094dev/bin/python scripts/litert_gate.py \
        --backend gpu --precision fp32 --L 256 --form fp32                              # one run = one window
    $Q/quiet_wait.py -- ~/venvs/lt094dev/bin/python scripts/litert_gate.py --rescore    # every stored run, no LiteRT run

Rows of bucket L: every question with lower < P + len(ids) <= L (L256: lower 0; L512: lower 256). Row source = the
oracle once it exists (its ids / markers / K / T / calibrate / prefix length; media prefixes from ref/npz/<id>.npz
`prefix`), else results/encoded_rows.json (text rows only; media rows wait for the oracle's prefixes). Inputs = the
host's build_inputs (host/d1_host.py) + the question type's one-hot. A run executes only the rows its store does not
hold yet (same ids, same prefix bytes), so the media rows are added after the oracle appears without re-running the
text rows. Store = out/r3_runs/<tag>.npz (scores at the real positions, float32) + out/r3_runs/<tag>.json (runs,
delegation, per-row meta, pad-content checks); tag = eager_L<L> | cpu_L<L>_<form> | gpu_<precision>_L<L>_<form>.

Scoring (after every run, and --rescore) -> results/litert_<cpu|gpu_fp32|gpu_default>_parity_L<L>_<form>.json:
the host's readout() (float32 torch ops: temperature for text, raw softmax for media, a noul reversed) and, per
reference (`oracle` = the oracle's probs / logits_raw; `eager` = the eager store through the same readout): argmax
agreement (near-tie rows = reference top-2 gap <= 0.02 counted apart), max / p95 / mean |dp| over all options, max
|dlogit| at the markers, options that change side at the cutoffs 0.5 and 0.9, non-finite values, the red arm (graph on
red_arm_000 vs the reference of tv4_000: must exceed 0.02), by mode and by source, per-row probabilities. `primary` =
oracle when it exists, else eager (both sections are kept, with a history of every scoring pass). Plus: pad-content
invariance (rows run again with random ids and prefix at the pad positions: scores at the real positions bit-equal),
GPU vs CPU of the same file (max |dscores| over real positions), delegation (`Replacing N out of M node(s)` with the
runtime's VERBOSE log on, partitions, is_fully_accelerated), compile / run failure text.
Bar (FACTS §7): argmax 100 % outside near-tie rows + max |dp| <= 0.02 + mean |dp| <= 0.002 + no non-finite value.
A GPU run executes in a child process (a delegate abort ends the child; the parent then writes the failure record).
No timing is recorded beyond the wall seconds of each job (this round measures no speed).

Round 4 (additions; L256 / L512 behave as in round 3): buckets L128 (rows with P + n <= 128), L2048 (1024 < P + n <=
2048), L4096 (2048 < P + n <= 4096) and L1024, which has no natural row: every L512 row + the first 30 L256 rows (oracle
order), padded to 1024 -> `vs_smaller_bucket` in the result = the same rows' scores from the L256 / L512 store of the
same backend and form (bit-equal or not, max |dscores|, max |dp|). Stores of the round-4 buckets live in out/r4_runs/,
logs are logs/r4_*. The GPU parent records the child's ru_maxrss (os.wait4) and a CPU run its own peak RSS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
K = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(K / "host"))
import d1_src as S  # noqa: E402  (stdlib only)
import d1_host as H  # noqa: E402  (numpy at import; torch inside readout)

ORACLE = K / "ref/records_ref.json"
ORACLE_NPZ = K / "ref/npz"
RUNS = K / "out/r3_runs"
LOWER = {256: 0, 512: 256}
RUNS_R4 = K / "out/r4_runs"             # round 4 buckets
R4_LOWER = {128: 0, 2048: 1024, 4096: 2048}
L1024_FIRST_L256 = 30                    # L1024 = every L512 row + the first 30 L256 rows (no natural L1024 row)
ALL_L = sorted(set(LOWER) | set(R4_LOWER) | {1024})
QTYPES = {"choice": 0, "score": 1, "noul": 2}
BAR = {"max_abs_dp": 0.02, "mean_abs_dp": 0.002, "near_tie_gap": 0.02, "red_arm_min_dp": 0.02}
CUTOFFS = (0.5, 0.9)
PAD_ROWS = 5
TEMPS = S.config()["temperatures"]


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def write_json(path, doc):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    os.replace(tmp, path)


def sha_ids(ids):
    return hashlib.sha256(np.asarray(ids, np.int32).tobytes()).hexdigest()[:16]


def sha_arr(a):
    return None if a is None else hashlib.sha256(np.ascontiguousarray(a, np.float32).tobytes()).hexdigest()[:16]


# ---------------------------------------------------------------- rows

def oracle_doc():
    return json.loads(ORACLE.read_text()) if ORACLE.exists() else None


def record_sources():
    """record id -> source (round 2 may be rewriting requests.json: one retry, then empty)."""
    for _ in range(2):
        try:
            doc = json.loads((K / "fixtures/requests.json").read_text())
            return {r["id"]: r.get("source") for r in doc["records"]}
        except (OSError, ValueError):
            time.sleep(2)
    return {}


def runs_for(L):
    return RUNS if L in LOWER else RUNS_R4


def round_of(L):
    return 3 if L in LOWER else 4


def load_rows(L, oracle):
    """-> (rows, source name, pending list) for bucket L."""
    if L == 1024:
        assert oracle is not None, "L1024 rows are taken from the oracle"
        r512, _, pend = load_rows(512, oracle)
        r256, _, _ = load_rows(256, oracle)
        return (r512 + r256[:L1024_FIRST_L256],
                f"oracle (L1024: the {len(r512)} L512 rows + the first {L1024_FIRST_L256} L256 rows, padded)", pend)
    lo, rows, pending = LOWER.get(L, R4_LOWER.get(L)), [], []
    if oracle is not None:
        for rec in oracle["records"]:
            for q in rec["questions"]:
                P, n = int(q.get("prefix") or 0), len(q["ids"])
                row = {"key": f"{rec['id']}/{q['qid']}", "id": rec["id"], "qid": q["qid"],
                       "mode": rec.get("mode", "text" if P == 0 else "media"), "P": P, "n": n, "ids": q["ids"],
                       "markers": q["markers"], "K": int(q["K"]), "type": q["type"], "calibrate": bool(q["calibrate"]),
                       "T": q.get("T"), "temperature_key": q.get("temperature_key")}
                if lo < P + n <= L:
                    rows.append(row)
        return rows, "oracle", pending
    enc = json.loads((K / "results/encoded_rows.json").read_text())
    for r in enc["rows"]:
        key, n = f"{r['id']}/{r['qid']}", len(r["ids"])
        if r["mode"] != "text":
            pending.append({"key": key, "mode": r["mode"], "n_ids": n, "reason": "the media prefix comes with the oracle"})
            continue
        if lo < n <= L:
            rows.append({"key": key, "id": r["id"], "qid": r["qid"], "mode": "text", "P": 0, "n": n, "ids": r["ids"],
                         "markers": r["markers"], "K": int(r["K"]), "type": r["type"], "calibrate": True,
                         "T": r["temperature"], "temperature_key": r["temperature_key"]})
    return rows, "encoded_rows_v1", pending


_PREFIX_CACHE = {}


def prefix_of(row):
    if row["P"] == 0:
        return None
    if row["id"] not in _PREFIX_CACHE:
        with np.load(ORACLE_NPZ / f"{row['id']}.npz") as z:
            _PREFIX_CACHE[row["id"]] = np.asarray(z["prefix"], np.float32)
    p = _PREFIX_CACHE[row["id"]]
    assert p.shape == (row["P"], H.D), (row["key"], p.shape, row["P"])
    return p


def row_inputs(row, L):
    x = H.build_inputs(row["ids"], prefix_of(row), L)
    oh = np.zeros((1, 3), np.float32)
    oh[0, QTYPES[row["type"]]] = 1.0
    x["qtype_onehot"] = oh
    return x


def pad_variant(x, row, L, seed):
    """Random ids and prefix at the pad positions (>= P + n), everything else unchanged."""
    g = np.random.default_rng(seed)
    real = row["P"] + row["n"]
    y = {k: v.copy() for k, v in x.items()}
    y["ids"][0, real:] = g.integers(22, 64000, size=L - real, dtype=np.int32)
    y["prefix"][0, real:] = g.standard_normal((L - real, H.D)).astype(np.float32)
    return y


def pick_pad_rows(rows, L, k=PAD_ROWS):
    """Up to k rows with pad positions: the shortest, card_text/refund, the median, the longest that leaves a pad,
    and a media row when the set has one (else the second longest)."""
    cand = sorted([r for r in rows if r["P"] + r["n"] < L], key=lambda r: (r["P"] + r["n"], r["key"]))
    if not cand:
        return []
    pick = [cand[0], cand[len(cand) // 2], cand[-1]]
    pick += [r for r in cand if r["key"] == "card_text/refund"]
    media = [r for r in cand if r["P"] > 0]
    pick += media[:1] if media else cand[-2:-1]
    out, seen = [], set()
    for r in pick + cand:
        if r["key"] not in seen:
            out.append(r)
            seen.add(r["key"])
        if len(out) == k:
            break
    return out


# ---------------------------------------------------------------- store

def tag_of(backend, precision, L, form):
    if backend == "eager":
        return f"eager_L{L}"
    return f"cpu_L{L}_{form}" if backend == "cpu" else f"gpu_{precision}_L{L}_{form}"


def npz_key(key):
    return key.replace("/", "__")


class Store:
    def __init__(self, tag):
        import re

        self.tag = tag
        self.dir = runs_for(int(re.search(r"_L(\d+)", tag).group(1)))
        self.npz, self.meta_path = self.dir / f"{tag}.npz", self.dir / f"{tag}.json"
        self.scores = {}
        if self.npz.exists():
            with np.load(self.npz) as z:
                self.scores = {k: np.asarray(z[k]) for k in z.files}
        self.meta = (json.loads(self.meta_path.read_text()) if self.meta_path.exists()
                     else {"tag": tag, "runs": [], "rows": {}, "pad_content": []})

    def has(self, row):
        m = self.meta["rows"].get(row["key"])
        return (m is not None and npz_key(row["key"]) in self.scores and m["ids_sha"] == sha_ids(row["ids"])
                and m["P"] == row["P"] and m["prefix_sha"] == sha_arr(prefix_of(row)))

    def get(self, row):
        return self.scores.get(npz_key(row["key"])) if self.has(row) else None

    def put(self, row, scores_real, run_index):
        self.scores[npz_key(row["key"])] = np.asarray(scores_real, np.float32)
        self.meta["rows"][row["key"]] = {"ids_sha": sha_ids(row["ids"]), "P": row["P"], "n": row["n"],
                                         "prefix_sha": sha_arr(prefix_of(row)), "run": run_index,
                                         "nonfinite_real": int((~np.isfinite(scores_real)).sum())}

    def save(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.npz.with_name(self.npz.stem + ".tmp.npz")
        np.savez(tmp, **self.scores)
        os.replace(tmp, self.npz)
        write_json(self.meta_path, self.meta)


# ---------------------------------------------------------------- readout and statistics

def readout(scores_real, row):
    q = SimpleNamespace(type=row["type"], options=row["K"])
    p = H.readout(scores_real, row["P"], row["markers"], q, row["calibrate"], TEMPS)
    logits = [float(scores_real[row["P"] + m]) for m in row["markers"][:row["K"]]]
    return [float(v) for v in p], logits


def temperature_check(rows):
    bad = []
    for r in rows:
        if r["calibrate"] and r["T"] is not None:
            T = H.temperature(SimpleNamespace(type=r["type"], options=r["K"]), TEMPS)
            if abs(T - r["T"]) > 1e-9:
                bad.append({"key": r["key"], "host_T": T, "row_T": r["T"]})
    return bad


def top2_gap(p):
    s = sorted(p, reverse=True)
    return s[0] - s[1] if len(s) > 1 else 1.0


def compare(rows, lit, ref, sources):
    """lit / ref: {key: (probs, logits)}; ref rows missing are skipped and counted."""
    per, dps, dls = [], [], []
    by_mode, by_source = {}, {}
    agree = agree_nt = n_nt = n_main = 0
    flips, nt_flips, crossings, nonfinite, missing = [], [], {str(c): [] for c in CUTOFFS}, [], []
    for r in rows:
        if r["key"] not in lit or r["key"] not in ref:
            missing.append(r["key"])
            continue
        (pl, ll), (pr, lr) = lit[r["key"]], ref[r["key"]]
        if not (np.isfinite(pl).all() and np.isfinite(ll).all()):
            nonfinite.append(r["key"])
            continue
        dp = [abs(a - b) for a, b in zip(pl, pr)]
        dl = max(abs(a - b) for a, b in zip(ll, lr)) if lr is not None else None
        al, ar = int(np.argmax(pl)), int(np.argmax(pr))
        near = top2_gap(pr) <= BAR["near_tie_gap"]
        if near:
            n_nt += 1
            agree_nt += al == ar
            if al != ar:
                nt_flips.append(r["key"])
        else:
            n_main += 1
            agree += al == ar
            if al != ar:
                flips.append({"key": r["key"], "lit": pl, "ref": pr})
        for c in CUTOFFS:
            for i, (a, b) in enumerate(zip(pl, pr)):
                if (a >= c) != (b >= c):
                    crossings[str(c)].append({"key": r["key"], "option": i, "ref": b, "lit": a})
        dps.extend(dp)
        if dl is not None:
            dls.append(dl)
        src = sources.get(r["id"], "?")
        for d, name in ((by_mode, r["mode"]), (by_source, src)):
            e = d.setdefault(name, {"rows": 0, "max_abs_dp": 0.0, "max_abs_dlogit": 0.0, "argmax_flips": 0})
            e["rows"] += 1
            e["max_abs_dp"] = max(e["max_abs_dp"], max(dp))
            e["max_abs_dlogit"] = max(e["max_abs_dlogit"], dl or 0.0)
            e["argmax_flips"] += int(al != ar)
        per.append({"key": r["key"], "max_abs_dp": max(dp), "max_abs_dlogit": dl, "near_tie": near,
                    "argmax_equal": al == ar})
    dps_a = np.asarray(dps, np.float64)
    st = {"rows_compared": len(per), "rows_missing_reference_or_run": missing, "options_compared": int(dps_a.size),
          "max_abs_dp": float(dps_a.max()) if dps_a.size else None,
          "p95_abs_dp": float(np.percentile(dps_a, 95)) if dps_a.size else None,
          "mean_abs_dp": float(dps_a.mean()) if dps_a.size else None,
          "max_abs_dlogit": float(max(dls)) if dls else None,
          "argmax": {"rows_outside_near_tie": n_main, "equal_outside_near_tie": agree,
                     "near_tie_rows": n_nt, "near_tie_equal": agree_nt, "flips": flips, "near_tie_flips": nt_flips},
          "cutoff_crossings": {c: len(v) for c, v in crossings.items()}, "cutoff_crossing_rows": crossings,
          "nonfinite_rows": nonfinite, "by_mode": by_mode, "by_source": by_source,
          "top10_by_dp": sorted(per, key=lambda x: -x["max_abs_dp"])[:10]}
    st["bar_pass"] = bool(per and not nonfinite and agree == n_main and st["max_abs_dp"] <= BAR["max_abs_dp"]
                          and st["mean_abs_dp"] <= BAR["mean_abs_dp"])
    return st


def red_arm(lit, ref):
    a, b = lit.get("red_arm_000/answer"), ref.get("tv4_000/answer")
    if a is None or b is None:
        return {"available": False}
    d = max(abs(x - y) for x, y in zip(a[0], b[0]))
    return {"available": True, "graph_row": "red_arm_000/answer", "reference_row": "tv4_000/answer",
            "max_abs_dp": d, "breaks_bar": d > BAR["red_arm_min_dp"], "lit_probs": a[0], "ref_probs": b[0]}


def oracle_ref(oracle):
    out = {}
    for rec in oracle["records"]:
        for q in rec["questions"]:
            out[f"{rec['id']}/{q['qid']}"] = ([float(v) for v in q["probs"]], [float(v) for v in q["logits_raw"]])
    return out


def store_probs(store, rows):
    out = {}
    for r in rows:
        s = store.get(r)
        if s is not None:
            out[r["key"]] = readout(s, r)
    return out


def cross_bucket(tag, rows, store):
    """L1024 (round 4): the same rows in their natural bucket (L256 / L512 store of this backend and form)."""
    out = {"rows": 0, "bit_equal_scores": 0, "max_abs_dscores": 0.0, "max_abs_dp": 0.0, "missing": [], "per_row": []}
    for r in rows:
        Ls = 512 if r["P"] + r["n"] > 256 else 256
        small = Store(tag.replace("_L1024_", f"_L{Ls}_"))
        a, b = store.get(r), small.get(r)
        if a is None or b is None:
            out["missing"].append(r["key"])
            continue
        pa, pb = readout(a, r)[0], readout(b, r)[0]
        bit = bool(np.array_equal(a, b))
        d = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())
        dp = max(abs(x - y) for x, y in zip(pa, pb))
        out["rows"] += 1
        out["bit_equal_scores"] += int(bit)
        out["max_abs_dscores"] = max(out["max_abs_dscores"], d)
        out["max_abs_dp"] = max(out["max_abs_dp"], dp)
        out["per_row"].append({"key": r["key"], "natural_L": Ls, "bit_equal": bit, "max_abs_dscores": d,
                               "max_abs_dp": dp})
    out["all_bit_equal"] = bool(out["rows"] and out["bit_equal_scores"] == out["rows"] and not out["missing"])
    return out


def score(tag, L, form, backend_label):
    """Re-score one LiteRT store against every available reference -> results json (rewritten; history kept)."""
    store = Store(tag)
    oracle = oracle_doc()
    rows, source, pending = load_rows(L, oracle)
    sources = record_sources()
    lit = store_probs(store, rows)
    refs = {}
    eager_store = Store(f"eager_L{L}")
    if eager_store.scores:
        refs["eager"] = store_probs(eager_store, rows)
    if oracle is not None:
        refs["oracle"] = oracle_ref(oracle)
    out_path = K / f"results/litert_{backend_label}_parity_L{L}_{form}.json"
    prev = json.loads(out_path.read_text()) if out_path.exists() else {}
    doc = {"step": f"round {round_of(L)}: Mac {backend_label} parity, form {form}, L={L}", "scored_at": now(), "tag": tag,
           "tflite": f"out/d1omni_decide_L{L}_{form}.tflite", "row_source": source, "rows_in_bucket": len(rows),
           "rows_run": len(lit), "rows_pending": pending, "bar": BAR,
           "temperature_mismatch": temperature_check(rows), "runs": store.meta["runs"],
           "pad_content": store.meta["pad_content"],
           "pad_content_all_bit_equal": (all(p["real_bit_equal"] for p in store.meta["pad_content"])
                                         if store.meta["pad_content"] else None),
           "nonfinite_real_total": sum(m["nonfinite_real"] for m in store.meta["rows"].values()),
           "references": {}}
    for name, ref in refs.items():
        st = compare(rows, lit, ref, sources)
        st["red_arm"] = red_arm(lit, ref)
        doc["references"][name] = st
    oracle_meta = None if oracle is None else {"version": oracle.get("version", 1), "written": oracle.get("written"),
                                               "resize_path": (oracle.get("load") or {}).get("resize_path")}
    doc["oracle_file"] = oracle_meta
    # rows compared against an earlier oracle version stay as a superseded record (11:3x: the image rows
    # are re-run against ref/records_ref.json version 2; the version-1 values are kept, not compared any more)
    sup = dict(prev.get("superseded", {}))
    pv = (prev.get("oracle_file") or {}).get("version")
    if oracle_meta and pv is not None and pv != oracle_meta["version"] and f"oracle_v{pv}" not in sup:
        sup[f"oracle_v{pv}"] = {"oracle_file": prev["oracle_file"], "scored_at": prev.get("scored_at"),
                                "summary": prev.get("summary"),
                                "by_mode": (prev.get("references", {}).get("oracle") or {}).get("by_mode"),
                                "per_row_image": [r for r in prev.get("per_row", []) if r.get("mode") == "image"]}
    if sup:
        doc["superseded"] = sup
    doc["primary"] = "oracle" if "oracle" in refs else ("eager" if "eager" in refs else None)
    if doc["primary"]:
        p = doc["references"][doc["primary"]]
        doc["summary"] = {"reference": doc["primary"], "rows": p["rows_compared"], "max_abs_dp": p["max_abs_dp"],
                          "p95_abs_dp": p["p95_abs_dp"], "mean_abs_dp": p["mean_abs_dp"],
                          "max_abs_dlogit": p["max_abs_dlogit"],
                          "argmax_outside_near_tie": f"{p['argmax']['equal_outside_near_tie']}/{p['argmax']['rows_outside_near_tie']}",
                          "near_tie": f"{p['argmax']['near_tie_equal']}/{p['argmax']['near_tie_rows']}",
                          "cutoff_crossings": p["cutoff_crossings"], "nonfinite_rows": len(p["nonfinite_rows"]),
                          "red_arm_max_abs_dp": p["red_arm"].get("max_abs_dp"), "bar_pass": p["bar_pass"]}
    if backend_label.startswith("gpu"):
        cpu = Store(f"cpu_L{L}_{form}")
        dmax, dmark, n = 0.0, 0.0, 0
        for r in rows:
            a, b = store.get(r), cpu.get(r)
            if a is None or b is None:
                continue
            n += 1
            d = np.abs(a.astype(np.float64) - b.astype(np.float64))
            dmax = max(dmax, float(np.nanmax(d)) if np.isfinite(d).any() else float("inf"))
            dmark = max(dmark, max(float(d[r["P"] + m]) for m in r["markers"][:r["K"]]))
        doc["gpu_vs_cpu_same_file"] = {"rows": n, "max_abs_dscores_real_positions": dmax if n else None,
                                       "max_abs_dscores_markers": dmark if n else None}
    if L == 1024:
        doc["vs_smaller_bucket"] = cross_bucket(tag, rows, store)
    doc["per_row"] = []
    ok = {name: ref for name, ref in refs.items()}
    for r in rows:
        if r["key"] not in lit:
            continue
        e = {"key": r["key"], "mode": r["mode"], "source": sources.get(r["id"]), "P": r["P"], "n": r["n"], "K": r["K"],
             "type": r["type"], "probs": lit[r["key"]][0], "logits": lit[r["key"]][1]}
        for name, ref in ok.items():
            if r["key"] in ref:
                e[f"probs_{name}"] = ref[r["key"]][0]
                e[f"logits_{name}"] = ref[r["key"]][1]
        doc["per_row"].append(e)
    hist = prev.get("scoring_history", [])
    hist.append({"scored_at": doc["scored_at"], "row_source": source, "rows_run": len(lit), "oracle_file": oracle_meta,
                 "references": {n: {k: v[k] for k in ("rows_compared", "max_abs_dp", "p95_abs_dp", "mean_abs_dp",
                                                      "max_abs_dlogit", "cutoff_crossings", "bar_pass")}
                                for n, v in doc["references"].items()}})
    doc["scoring_history"] = hist
    last = store.meta["runs"][-1] if store.meta["runs"] else {}
    if last.get("status", "OK") != "OK":
        doc["status"] = last["status"] if last["status"] not in ("FAIL",) else (
            "COMPILE_FAIL" if "compile_seconds" not in last else "RUN_FAIL")
        doc["failure"] = {k: last.get(k) for k in ("error", "returncode", "signal", "runtime_log_tail",
                                                    "child_stdio_tail") if last.get(k) is not None}
        doc["failure"]["runtime_log_key_lines"] = (last.get("delegation") or {}).get("key_lines", [])[-40:]
    else:
        doc["status"] = "OK" if lit else "NO_ROWS"
    write_json(out_path, doc)
    return doc


# ---------------------------------------------------------------- runs

def run_eager(a):
    import torch

    import d1_graph as G
    import graph_build as B

    torch.set_num_threads(8)
    tag = tag_of("eager", None, a.L, None)
    store = Store(tag)
    rows, source, pending = load_rows(a.L, oracle_doc())
    todo = [r for r in rows if not store.has(r)]
    t0 = time.time()
    model, wrep = B.build(a.L, "checkpoint", S.config())
    for r in todo:
        x = row_inputs(r, a.L)
        with torch.no_grad():
            s = model(**{k: torch.from_numpy(v) for k, v in x.items()})["scores"].numpy().reshape(-1)
        store.put(r, s[:r["P"] + r["n"]].copy(), len(store.meta["runs"]))
    store.meta["runs"].append({"started": now(), "seconds_wall": round(time.time() - t0, 1), "rows_run": len(todo),
                               "row_source": source, "torch": torch.__version__, "python": sys.version.split()[0],
                               "threads": torch.get_num_threads(), "weights": wrep, "L": a.L,
                               "venv": sys.executable, "pending": len(pending)})
    store.save()
    print(json.dumps({"tag": tag, "rows_run": len(todo), "rows_in_bucket": len(rows), "source": source,
                      "seconds": round(time.time() - t0, 1)}))


def run_litert(a):
    """Child body (GPU) or the whole run (CPU)."""
    import litert_run as R

    label = "cpu" if a.backend == "cpu" else f"gpu_{a.precision}"
    tag = tag_of(a.backend, a.precision, a.L, a.form)
    path = K / f"out/d1omni_decide_L{a.L}_{a.form}.tflite"
    store = Store(tag)
    rows, source, pending = load_rows(a.L, oracle_doc())
    todo = [r for r in rows if not store.has(r)]
    run_index = len(store.meta["runs"])
    log = K / f"logs/r{round_of(a.L)}_{tag}_run{run_index}.runtime.log"
    info = {"started": now(), "row_source": source, "rows_in_bucket": len(rows), "rows_to_run": len(todo),
            "tflite": str(path.relative_to(K)), "tflite_bytes": path.stat().st_size, "pid": os.getpid()}
    t0 = time.time()
    status, err = "OK", None
    pad_new = []
    with R.capture_fd2(log):
        try:
            info["logger"] = R.runtime_log_verbose()
            import importlib.metadata as md
            info["ai_edge_litert"] = md.version("ai-edge-litert")
            t1 = time.time()
            cm, desc = R.open_compiled(path, a.backend, a.precision, threads=8)
            info["compile_seconds"] = round(time.time() - t1, 2)
            info["options"] = desc
            try:
                info["is_fully_accelerated"] = bool(cm.is_fully_accelerated())
            except Exception as e:  # informational
                info["is_fully_accelerated"] = f"unavailable: {type(e).__name__}: {e}"
            sig = next(iter(cm.get_signature_list()))
            info["signature"] = sig
            run = R.Runner(cm, sig)
            assert run.L == a.L, (run.L, a.L)
            done = {}
            for i, r in enumerate(todo):
                x = row_inputs(r, a.L)
                s = run(x)
                store.put(r, s[:r["P"] + r["n"]].copy(), run_index)
                done[r["key"]] = (x, s)
                if (i + 1) % 100 == 0:
                    print(f"{i + 1}/{len(todo)}", file=sys.stdout, flush=True)
            have = {p["key"] for p in store.meta["pad_content"]}
            have_media = any(p["P"] > 0 for p in store.meta["pad_content"])
            cands = [r for r in pick_pad_rows(todo, a.L, PAD_ROWS + 2) if r["key"] not in have]
            chosen = cands[:max(PAD_ROWS - len(have), 0)]
            if not have_media and not any(r["P"] > 0 for r in chosen):
                chosen += [r for r in todo if r["P"] > 0 and r["P"] + r["n"] < a.L and r["key"] not in have][:1]
            for j, r in enumerate(chosen):
                x, s = done[r["key"]]
                y = pad_variant(x, r, a.L, seed=1000 + j)
                s2 = run(y)
                real = r["P"] + r["n"]
                pad_new.append({"key": r["key"], "P": r["P"], "n": r["n"], "pad_positions": a.L - real,
                                "real_bit_equal": bool(np.array_equal(s[:real], s2[:real])),
                                "real_max_abs_diff": float(np.abs(s[:real].astype(np.float64) - s2[:real]).max()),
                                "pad_positions_changed": bool(not np.array_equal(s[real:], s2[real:])),
                                "run": run_index})
            run.close()
            if hasattr(cm, "close"):
                cm.close()
        except BaseException as e:  # recorded, then the json is written
            import traceback
            status, err = "FAIL", f"{type(e).__name__}: {e}"
            traceback.print_exc()
    info["seconds_wall"] = round(time.time() - t0, 1)
    import resource
    info["ru_maxrss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    info["status"], info["error"] = status, err
    info["delegation"] = R.delegation_from_log(log)
    store.meta["pad_content"].extend(pad_new)
    store.meta["runs"].append(info)
    store.save()
    doc = score(tag, a.L, a.form, label)
    print(json.dumps({"tag": tag, "status": status, "error": err, "rows_run": len(todo),
                      "replacing": info["delegation"]["replacing"], "is_fully_accelerated": info.get("is_fully_accelerated"),
                      "pad_content": pad_new, "summary": doc.get("summary"),
                      "gpu_vs_cpu": doc.get("gpu_vs_cpu_same_file"), "seconds": info["seconds_wall"]}, indent=1))
    return 0 if status == "OK" else 1


def run_gpu_parent(a):
    """Run the GPU job in a child process; if the child dies without finishing, write the failure record."""
    label = f"gpu_{a.precision}"
    tag = tag_of("gpu", a.precision, a.L, a.form)
    errlog = K / f"logs/r{round_of(a.L)}_{tag}.child_stdio.log"
    cmd = [sys.executable, str(Path(__file__).resolve()), "--backend", "gpu", "--precision", a.precision,
           "--L", str(a.L), "--form", a.form, "--child"]
    t0 = time.time()
    with open(errlog, "a") as fo:
        fo.write(f"--- {now()} {' '.join(cmd)}\n")
        fo.flush()
        child = subprocess.Popen(cmd, stdout=fo, stderr=subprocess.STDOUT, cwd=str(K))
        _, status, ru = os.wait4(child.pid, 0)
        rc = os.waitstatus_to_exitcode(status)
        child.returncode = rc
    child_maxrss = int(ru.ru_maxrss)
    sig = signal.Signals(-rc).name if rc < 0 else None
    out = K / f"results/litert_{label}_parity_L{a.L}_{a.form}.json"
    store = Store(tag)
    recorded = bool(store.meta["runs"]) and store.meta["runs"][-1].get("pid") == child.pid
    if not recorded:   # the child died before saving its run: record the failure from here
        logs = sorted(K.glob(f"logs/r{round_of(a.L)}_{tag}_run*.runtime.log"), key=lambda p: p.stat().st_mtime)
        tail = logs[-1].read_text(errors="replace").splitlines()[-40:] if logs else []
        store.meta["runs"].append({"started": now(), "pid": child.pid, "status": "GPU_FAIL", "returncode": rc,
                                   "signal": sig, "seconds_wall": round(time.time() - t0, 1),
                                   "child_ru_maxrss_bytes": child_maxrss,
                                   "runtime_log_tail": tail, "runtime_log": str(logs[-1].relative_to(K)) if logs else None,
                                   "child_stdio_tail": errlog.read_text(errors="replace").splitlines()[-40:]})
        store.save()
        score(tag, a.L, a.form, label)
    else:   # the child's own record: add its peak RSS as the parent saw it
        store.meta["runs"][-1]["child_ru_maxrss_bytes"] = child_maxrss
        store.save()
        score(tag, a.L, a.form, label)
    print(f"gpu child pid={child.pid} rc={rc} signal={sig} recorded_by_child={recorded} maxrss={child_maxrss} "
          f"seconds={time.time() - t0:.1f} -> {out.relative_to(K)}")
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--backend", choices=("cpu", "gpu"))
    ap.add_argument("--precision", choices=("fp32", "default"), default="fp32")
    ap.add_argument("--L", type=int, choices=ALL_L)
    ap.add_argument("--form", default="fp32")
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--rescore", action="store_true")
    a = ap.parse_args()
    if a.rescore:
        for meta in sorted(RUNS.glob("*.json")) + sorted(RUNS_R4.glob("*.json")):
            tag = meta.stem
            if tag.startswith("eager_"):
                continue
            parts = tag.split("_")
            if parts[0] == "cpu":
                label, L, form = "cpu", int(parts[1][1:]), "_".join(parts[2:])
            else:
                label, L, form = f"gpu_{parts[1]}", int(parts[2][1:]), "_".join(parts[3:])
            d = score(tag, L, form, label)
            print(json.dumps({"tag": tag, "summary": d.get("summary")}))
        return 0
    if a.eager:
        return run_eager(a)
    if a.backend == "gpu" and not a.child:
        return run_gpu_parent(a)
    return run_litert(a)


if __name__ == "__main__":
    sys.exit(main())
