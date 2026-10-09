"""Round 5: host scorer of the S26 gate app's legs (android/d1omni_gate) — or of its stand-in on the Mac
(scripts/fake_runner.py = the app's contract on the Mac CompiledModel CPU), which writes the same files.

    cd d1_omni_work; PY=~/venvs/lt094dev/bin/python   (numpy + torch: the host read-out is host/d1_host.py's float32 torch ops)
    $PY scripts/s26_score.py gate --report device/r5/<tag>.json --sel device/r5/sel_<tag>.f32 --rows device/rows_L256_sub.json \
        [--logcat device/r5/<tag>.logcat.txt] [--samples device/r5/<tag>.samples.txt] [--state-before ... --state-after ...] \
        [--same-file mac_cpu=<report>.json:<sel>.f32 ...] [--round3 metal=results/litert_gpu_fp32_parity_L256_fp16.json ...] \
        [--out results/s26_parity_<tag>.json]
    $PY scripts/s26_score.py timing --report device/r5/<tag>.json --samples device/r5/<tag>.samples.txt \
        [--state-before ... --state-after ...] [--out results/s26_timing_<tag>.json]
    $PY scripts/s26_score.py table [--round r5] [--prefix <tag prefix>] [--results <dir>] [--round-dir <dir>] [--out <json>]
        # every scored leg as markdown tables (numbers read from the result files only) + the legs' statuses

gate: the report's rows (the rows file's first rows, in order) and the sel file (little-endian float32, K per row = the
scores at P + markers[k]) -> the host read-out per row (host/d1_host.py readout: the K scores, / T for a text question
(the oracle's temperature key), softmax, a noul reversed) -> against the oracle (ref/records_ref.json version 2:
probs and logits_raw): argmax outside near-tie rows (oracle top-2 gap <= 0.02) and the near-tie rows apart, max / p95 /
mean |dp| over all options, max |dlogit| at the markers, options that change side at the cutoffs 0.5 and 0.9,
non-finite rows, the control (red_arm_000/answer's probabilities against the oracle of tv4_000/answer, must differ by
more than 0.02), by mode; per row the probabilities and the call's ms. Bar (FACTS §7): argmax 100 % outside near-tie +
max |dp| <= 0.02 + mean |dp| <= 0.002 + no non-finite row. --same-file label=<report>:<sel>: another run of the same
graph file over the same rows (the Mac stand-in, a second phone run) -> per row max |dscore| at the markers, rows
bit-equal, max |dp|, argmax agreement. --round3 label=<results json>: round 3's per-row probabilities and marker scores
of the same graph file (its Mac CPU / Metal runs) for the keys in common. --logcat: the runtime's `Replacing N out of M
node(s) with delegate (X) node, yielding P partitions` lines and the derived full delegation (N = M, 1 partition; the
Kotlin 2.2.0 CompiledModel has no isFullyAccelerated()). --samples / --state-*: the leg's 2 s samples (MemAvailable
minimum, the app's VmHWM maximum, kgsl clock cap and temperature, CPU caps) and the phone state before and after.
timing: per set, every timed call [device ms at its start, ms write + run + read, ms run only] with median / min / max,
a request set's per-request sums, the warm-up calls, the cool-down record, and the burst / capped split of the calls
(the nearest 2 s sample: burst = kgsl max_clock at its top (--full-mhz 1300) with thermal_pwrlevel 0 and every CPU
policy at cpuinfo_max_freq; else capped; no sample within 3 s = unclassified).
Output files are never overwritten.

Round 9 (the app's generic mode; the round 5 subcommands above are unchanged):
    $PY scripts/s26_score.py generic --report device/r9/<tag>.json --out-file device/r9/out_<tag>.f32 --rows <generic rows>
        [--same-file mac_cpu8=<report>.json:<out>.f32] [--prefix-out device/r9/<tag>.prefix.npz] [--logcat ...]
        [--samples ...] [--state-before ... --state-after ...] [--out results/s26_parity_<tag>_r9.json]
      the leg's outputs, sliced per row from out_<tag>.f32, against the oracle npz and the Mac runs of the same file:
      vision_tower = each crop's real-patch features vs tower_last_hidden_state; projector = each crop's prefix rows,
      assembled per record (crops in order) -> the record prefix vs the oracle prefix, the float64 truth and our eager
      (out/r6_runs/{truth,eager}: §0-19's three distances, the rel_rms ratio to the provider fp32's distance to the
      truth), vs the Mac chains of the same files (out/r6_runs/{cpu,gpu}_fp32_fp16), and the Mac CPU projector run here
      on the very same soft input; audio = each clip's P prefix rows vs the oracle prefix and the round 7 Mac stores
      (out/r7_runs/audio_{cpu,gpu_fp32}_fp16.npz, host mel). No bar here (vision / audio PASS / FAIL = the end to end,
      `e2e9`): summary.finite_pass = no non-finite value.
    $PY scripts/s26_score.py soft --report <tower leg>.json --out-file out_<tower leg>.f32 --rows device/g9_vt.json \
        --name <projector rows file name> --out-dir <dir> [--sha-file <dir>/../gen.sha256]
      a tower leg's features -> the host unshuffle (host/d1_vision_host.py) -> the projector's generic rows file
      <name> + its stacked soft input g9_pj_<tower leg>_soft.f32 (the chain's prep=soft:<tower leg>)
    $PY scripts/s26_score.py e2e9 --tag <leg> --prefix device/r9/<leg>.prefix.npz --kind vision|audio [--round-dir ...] \
        [--out results/s26_parity_<leg>_e2e_r9.json]
      the end to end on the Mac: the phone's prefixes through the text decision graph fp16 on the Mac CPU (8 threads;
      vision = the smallest of L256 / 512 / 1024 / 2048 / 4096 that holds the row, round 6's rule; audio = L256, round
      7's) -> the host read-out -> vs the oracle (FACTS §7 bar); the oracle's prefix through the same graph (the text
      graph's own share); the Mac chains' e2e of rounds 6 / 7 (same files) for the phone-vs-Mac difference.
    $PY scripts/s26_score.py table9 [--round r9] [--suffix _r9] [--out results/s26_summary_r9.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.dont_write_bytecode = True
K = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K / "scripts"))
sys.path.insert(0, str(K / "host"))
import d1_src as S  # noqa: E402  (stdlib only)
import d1_host as H  # noqa: E402  (numpy at import; torch inside readout)

ORACLE = K / "ref/records_ref.json"
TEMPS = S.config()["temperatures"]
BAR = {"max_abs_dp": 0.02, "mean_abs_dp": 0.002, "near_tie_gap": 0.02, "red_arm_min_dp": 0.02}
CUTOFFS = (0.5, 0.9)
CONTROL, CONTROL_REF = "red_arm_000/answer", "tv4_000/answer"
REPLACING = re.compile(r"Replacing (\d+) out of (\d+) node\(s\) with delegate \(([^)]*)\) node, yielding (\d+) partitions")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 24):
            h.update(block)
    return h.hexdigest()


def rel(p) -> str:
    p = Path(p).resolve()
    return str(p.relative_to(K)) if p.is_relative_to(K) else str(p)


def oracle_rows():
    doc = json.loads(ORACLE.read_text())
    assert doc.get("version") == 2, doc.get("version")
    out = {}
    for rec in doc["records"]:
        for q in rec["questions"]:
            out[f"{rec['id']}/{q['qid']}"] = {
                "key": f"{rec['id']}/{q['qid']}", "id": rec["id"], "source": rec.get("source"), "mode": rec["mode"],
                "P": int(q.get("prefix") or 0), "n": len(q["ids"]), "ids": q["ids"], "markers": q["markers"], "K": int(q["K"]),
                "type": q["type"], "calibrate": bool(q["calibrate"]), "T": q.get("T"), "near_tie": bool(q["near_tie"]),
                "probs": [float(v) for v in q["probs"]], "logits": [float(v) for v in q["logits_raw"]]}
    return doc, out


def readout(sel, row):
    """The host read-out of one row from its K marker scores (host/d1_host.py readout, float32 torch ops)."""
    q = SimpleNamespace(type=row["type"], options=row["K"])
    p = H.readout(np.asarray(sel, np.float32), 0, list(range(row["K"])), q, row["calibrate"], TEMPS)
    return [float(v) for v in p], [float(v) for v in sel]


def temperature_mismatch(rows):
    bad = []
    for r in rows:
        if r["calibrate"] and r["T"] is not None:
            T = H.temperature(SimpleNamespace(type=r["type"], options=r["K"]), TEMPS)
            if abs(T - r["T"]) > 1e-9:
                bad.append({"key": r["key"], "host_T": T, "oracle_T": r["T"]})
    return bad


def top2_gap(p):
    s = sorted(p, reverse=True)
    return s[0] - s[1] if len(s) > 1 else 1.0


def compare(rows, lit, ref):
    """lit / ref: {key: (probs, logits)} -> the round 3 statistics (scripts/litert_gate.py compare, same definitions)."""
    per, dps, dls, by_mode = [], [], [], {}
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
        dl = max(abs(a - b) for a, b in zip(ll, lr))
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
        dls.append(dl)
        e = by_mode.setdefault(r["mode"], {"rows": 0, "max_abs_dp": 0.0, "max_abs_dlogit": 0.0, "argmax_flips": 0})
        e["rows"] += 1
        e["max_abs_dp"] = max(e["max_abs_dp"], max(dp))
        e["max_abs_dlogit"] = max(e["max_abs_dlogit"], dl)
        e["argmax_flips"] += int(al != ar)
        per.append({"key": r["key"], "max_abs_dp": max(dp), "max_abs_dlogit": dl, "near_tie": near, "argmax_equal": al == ar})
    a = np.asarray(dps, np.float64)
    st = {"rows_compared": len(per), "rows_missing": missing, "options_compared": int(a.size),
          "max_abs_dp": float(a.max()) if a.size else None, "p95_abs_dp": float(np.percentile(a, 95)) if a.size else None,
          "mean_abs_dp": float(a.mean()) if a.size else None, "max_abs_dlogit": float(max(dls)) if dls else None,
          "argmax": {"rows_outside_near_tie": n_main, "equal_outside_near_tie": agree, "near_tie_rows": n_nt,
                     "near_tie_equal": agree_nt, "flips": flips, "near_tie_flips": nt_flips},
          "cutoff_crossings": {c: len(v) for c, v in crossings.items()}, "cutoff_crossing_rows": crossings,
          "nonfinite_rows": nonfinite, "by_mode": by_mode, "top10_by_dp": sorted(per, key=lambda x: -x["max_abs_dp"])[:10]}
    st["bar_pass"] = bool(per and not nonfinite and agree == n_main and st["max_abs_dp"] <= BAR["max_abs_dp"]
                          and st["mean_abs_dp"] <= BAR["mean_abs_dp"])
    return st


def slice_sel(report, rows_doc, sel_path):
    """-> [(key, K float32 values)] in report order, after checking keys, n, P, K and the byte count."""
    recs = report["rows"]
    raw = np.fromfile(sel_path, dtype="<f4")
    need = sum(int(r["K"]) for r in recs)
    assert raw.size == need, f"{Path(sel_path).name}: {raw.size} floats, the report's rows need {need}"
    if "sel_bytes" in report:
        assert raw.size * 4 == int(report["sel_bytes"]), (raw.size * 4, report["sel_bytes"])
    file_rows = rows_doc["rows"]
    assert [r["key"] for r in recs] == [r["key"] for r in file_rows[: len(recs)]], "the report's rows are not the rows file's first rows in order"
    out, off = [], 0
    for r, fr in zip(recs, file_rows):
        assert int(r["K"]) == fr["K"] and int(r["n"]) == len(fr["ids"]) and int(r.get("P", 0)) == fr["P"], r["key"]
        out.append((r["key"], raw[off: off + fr["K"]].copy()))
        off += fr["K"]
    return out


def parse_samples(path):
    """The chain's 2 s samples -> [{dev_ms, kgsl: (clock, max, pwrlevel, temp), cpu: [(policy, cur, max, cpuinfo)],
    mem_available_kb, vmhwm_kb, vmrss_kb, thermal, skin, battery_temp}]"""
    if not path or not Path(path).exists():
        return []
    out, cur = [], None
    for line in Path(path).read_text(errors="replace").splitlines():
        if line.startswith("=== "):
            if cur is not None:
                out.append(cur)
            cur = {"head": line[4:], "cpu": []}
            continue
        if cur is None:
            continue
        f = line.split()
        try:
            if line.startswith("devtime "):
                cur["dev_ms"] = int(f[1]) * 1000
            elif line.startswith("kgsl ") and "kgsl" not in cur and len(f) >= 5:
                cur["kgsl"] = tuple(int(x) for x in f[1:5])
            elif line.startswith("cpu policy") and len(f) >= 5:
                cur["cpu"].append((f[1], int(f[2]), int(f[3]), int(f[4])))
            elif line.startswith("MemAvailable:"):
                cur["mem_available_kb"] = int(f[1])
            elif line.startswith("VmHWM:"):
                cur["vmhwm_kb"] = int(f[1])
            elif line.startswith("VmRSS:"):
                cur["vmrss_kb"] = int(f[1])
            elif "Thermal Status" in line:
                cur["thermal"] = int(f[-1])
            elif "mName=SKIN" in line:
                cur["skin"] = float(re.search(r"mValue=([-0-9.]+)", line).group(1))
            elif line.strip().startswith("temperature:") or line.strip().startswith("temperature"):
                v = re.search(r"(-?\d+)", line)
                if v:
                    cur["battery_temp_c"] = int(v.group(1)) / 10
        except (ValueError, IndexError, AttributeError):
            continue
    if cur is not None:
        out.append(cur)
    return [s for s in out if "dev_ms" in s]


def samples_summary(smp, full_mhz):
    if not smp:
        return None
    kg = [s["kgsl"] for s in smp if "kgsl" in s]
    caps = [f"{s['head'].split()[0]} " + " ".join(f"{p}:{m}/{c}" for p, _, m, c in s["cpu"] if m < c)
            for s in smp if any(m < c for _, _, m, c in s["cpu"])]
    return {"samples": len(smp),
            "mem_available_kb_min": min((s["mem_available_kb"] for s in smp if "mem_available_kb" in s), default=None),
            "app_vmhwm_kb_max": max((s["vmhwm_kb"] for s in smp if "vmhwm_kb" in s), default=None),
            "app_vmrss_kb_max": max((s["vmrss_kb"] for s in smp if "vmrss_kb" in s), default=None),
            "kgsl_clock_mhz_max": max((k[0] for k in kg), default=None),
            "kgsl_max_clock_mhz_min": min((k[1] for k in kg), default=None),
            "kgsl_pwrlevel_max": max((k[2] for k in kg), default=None),
            "kgsl_temp_c_max": (max((k[3] for k in kg), default=0) / 1000) if kg else None,
            "kgsl_capped_samples": sum(1 for k in kg if k[1] < full_mhz or k[2] > 0),
            "thermal_status_seen": sorted({s["thermal"] for s in smp if "thermal" in s}),
            "skin_c_max": max((s["skin"] for s in smp if "skin" in s), default=None),
            "battery_temp_c_max": max((s["battery_temp_c"] for s in smp if "battery_temp_c" in s), default=None),
            "cpu_capped_samples": len(caps), "cpu_caps_first": caps[:6],
            "first": smp[0]["head"], "last": smp[-1]["head"]}


def parse_state(path):
    if not path or not Path(path).exists():
        return None
    out = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        if ": " in line:
            k, v = line.split(": ", 1)
            out[k.strip()] = v.strip()
    return out


def delegation(logcat):
    if not logcat or not Path(logcat).exists():
        return {"lines": [], "note": "no logcat"}
    text = Path(logcat).read_text(errors="replace")
    lines = [m.group(0) for m in REPLACING.finditer(text)]
    parsed = [{"replaced": int(m.group(1)), "of": int(m.group(2)), "delegate": m.group(3), "partitions": int(m.group(4))}
              for m in REPLACING.finditer(text)]
    full = [p for p in parsed if p["replaced"] == p["of"] and p["partitions"] == 1]
    err = [ln for ln in text.splitlines() if re.search(r"(?i)(failed|error|unable|abort|fatal)", ln)][:30]
    return {"lines": lines, "parsed": parsed, "fully_delegated_derived": bool(full),
            "is_fully_accelerated_note": "derived from the Replacing line (N = M, 1 partition): the LiteRT 2.2.0 Kotlin "
                                         "CompiledModel has no isFullyAccelerated()",
            "error_lines": err}


def score_gate(a) -> dict:
    report_path, sel_path, rows_path = Path(a.report).resolve(), Path(a.sel).resolve(), Path(a.rows).resolve()
    report = json.loads(report_path.read_text())
    assert report.get("status") == "DONE", f"report status {report.get('status')}: {report.get('error')}"
    assert report.get("mode", "gate") == "gate", report.get("mode")
    rows_doc = json.loads(rows_path.read_text())
    assert int(rows_doc["L"]) == int(report["L"]) and int(rows_doc["hidden"]) == int(report["hidden"]), (rows_doc["L"], report["L"])
    odoc, orows = oracle_rows()
    sliced = slice_sel(report, rows_doc, sel_path)
    rows = [orows[k] for k, _ in sliced]
    for (k, _), fr in zip(sliced, rows_doc["rows"]):
        o = orows[k]
        assert fr["ids"] == o["ids"] and fr["markers"] == o["markers"] and fr["K"] == o["K"] and fr["P"] == o["P"], f"{k}: rows file != oracle"
    lit = {k: readout(v, orows[k]) for k, v in sliced}
    ref = {k: (o["probs"], o["logits"]) for k, o in orows.items()}
    st = compare(rows, lit, ref)   # the control row too, against its own oracle row (round 3 did the same)
    ctl = None
    if CONTROL in lit:
        d = max(abs(x - y) for x, y in zip(lit[CONTROL][0], ref[CONTROL_REF][0]))
        ctl = {"graph_row": CONTROL, "reference_row": CONTROL_REF, "max_abs_dp": d, "exceeds": d > BAR["red_arm_min_dp"],
               "lit_probs": lit[CONTROL][0], "ref_probs": ref[CONTROL_REF][0]}
    st["red_arm"] = ctl or {"available": False}
    same = {}
    for spec in a.same_file:
        label, pair = spec.split("=", 1)
        rp, sp = (Path(x).resolve() for x in pair.split(":", 1))
        other = json.loads(rp.read_text())
        assert other.get("status") == "DONE" and other.get("graph") == report.get("graph"), (label, other.get("status"), other.get("graph"))
        theirs = dict(slice_sel(other, rows_doc, sp))
        common = [k for k, _ in sliced if k in theirs]
        dsc, dp, agree, bit = [], [], 0, 0
        mine = dict(sliced)
        for k in common:
            x, y = mine[k], theirs[k]
            if not (np.isfinite(x).all() and np.isfinite(y).all()):
                continue
            dsc.append(float(np.abs(x.astype(np.float64) - y.astype(np.float64)).max()))
            bit += int(np.array_equal(x, y))
            p1, p2 = lit[k][0], readout(y, orows[k])[0]
            dp.append(max(abs(u - v) for u, v in zip(p1, p2)))
            agree += int(np.argmax(p1) == np.argmax(p2))
        same[label] = {"report": rel(rp), "sel": rel(sp), "sel_sha256": sha256(sp), "graph": other.get("graph"),
                       "accel": other.get("accel"), "precision": other.get("precision"), "threads": other.get("threads"),
                       "stand_in": other.get("stand_in"), "rows_compared": len(dsc), "argmax_agree": agree,
                       "max_abs_dscore_markers": max(dsc, default=None), "rows_bit_equal": bit, "max_abs_dp": max(dp, default=None)}
    r3 = {}
    for spec in a.round3:
        label, path = spec.split("=", 1)
        d = json.loads((K / path).read_text() if not Path(path).is_absolute() else Path(path).read_text())
        pr = {r["key"]: r for r in d["per_row"]}
        dp, dl, bit, agree, n = [], [], 0, 0, 0
        for k, _ in sliced:
            if k not in pr:
                continue
            n += 1
            p1, l1 = lit[k]
            p2, l2 = pr[k]["probs"], pr[k]["logits"]
            dp.append(max(abs(u - v) for u, v in zip(p1, p2)))
            dl.append(max(abs(u - v) for u, v in zip(l1, l2)))
            bit += int(all(np.float32(u) == np.float32(v) for u, v in zip(l1, l2)))
            agree += int(np.argmax(p1) == np.argmax(p2))
        r3[label] = {"file": path, "tag": d.get("tag"), "tflite": d.get("tflite"), "rows_compared": n, "argmax_agree": agree,
                     "max_abs_dp": max(dp, default=None), "max_abs_dlogit": max(dl, default=None), "rows_logits_bit_equal": bit}
    per = []
    recs = {r["key"]: r for r in report["rows"]}
    for k, v in sliced:
        o = orows[k]
        pl, ll = lit[k]
        e = {"key": k, "mode": o["mode"], "source": o["source"], "P": o["P"], "n": o["n"], "K": o["K"], "type": o["type"],
             "probs": pl, "logits": ll, "probs_oracle": o["probs"], "logits_oracle": o["logits"],
             "max_abs_dp": max(abs(x - y) for x, y in zip(pl, o["probs"])),
             "near_tie": o["near_tie"], "argmax_equal": int(np.argmax(pl)) == int(np.argmax(o["probs"])),
             "write_run_read_ms": recs[k]["write_run_read_ms"], "run_ms": recs[k]["run_ms"], "t_start_ms": recs[k]["t_start_ms"],
             "nonfinite_real": recs[k].get("nonfinite_real")}
        if k == CONTROL:
            e["control_max_abs_dp_vs_tv4_000"] = (ctl or {}).get("max_abs_dp")
        per.append(e)
    smp = parse_samples(a.samples)
    ms = [r["write_run_read_ms"] for r in report["rows"]]
    doc = {
        "step": "round 5: d1-omni decision graph on the S26 gate app (or its Mac stand-in) vs the oracle (scripts/s26_score.py gate)",
        "tag": a.tag or report_path.stem, "scored_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%S%z"),
        "report": rel(report_path), "report_sha256": sha256(report_path), "sel": rel(sel_path), "sel_bytes": sel_path.stat().st_size,
        "sel_sha256": sha256(sel_path), "rows_file": rel(rows_path), "rows_sha256": sha256(rows_path),
        "oracle": {"file": "ref/records_ref.json", "version": odoc.get("version"), "sha256": sha256(ORACLE)},
        "run": {k: report.get(k) for k in ("graph", "graph_bytes", "signature", "accel", "precision", "threads", "L", "hidden",
                                            "limit", "mode", "device", "android", "litert", "compile_ms", "status", "stopped_early",
                                            "memory_before_compile", "memory_after_compile", "memory_at_end", "stand_in", "io_dims")},
        "rows_in_file": len(rows_doc["rows"]), "rows_run": len(sliced), "bar": BAR,
        "temperature_mismatch": temperature_mismatch(rows),
        "summary": {"rows": st["rows_compared"], "max_abs_dp": st["max_abs_dp"], "p95_abs_dp": st["p95_abs_dp"],
                    "mean_abs_dp": st["mean_abs_dp"], "max_abs_dlogit": st["max_abs_dlogit"],
                    "argmax_outside_near_tie": f"{st['argmax']['equal_outside_near_tie']}/{st['argmax']['rows_outside_near_tie']}",
                    "near_tie": f"{st['argmax']['near_tie_equal']}/{st['argmax']['near_tie_rows']}",
                    "cutoff_crossings": st["cutoff_crossings"], "nonfinite_rows": len(st["nonfinite_rows"]),
                    "red_arm_max_abs_dp": (ctl or {}).get("max_abs_dp"), "bar_pass": st["bar_pass"]},
        "oracle_comparison": st,
        "delegation": delegation(a.logcat),
        "timing_ms": {**report.get("summary", {}), "median_all_rows_write_run_read": float(np.median(ms)) if ms else None},
        "samples": samples_summary(smp, a.full_mhz),
        "state_before": parse_state(a.state_before), "state_after": parse_state(a.state_after),
        "same_file": same, "round3_same_file": r3,
        "per_row": per,
    }
    return doc


def stats(xs):
    return {"median": float(statistics.median(xs)), "min": float(min(xs)), "max": float(max(xs)), "n": len(xs)} if xs else None


def classify(calls, smp, full_mhz):
    """calls [(device ms, value)] -> burst / capped (gpu, cpu, both) / unclassified by the nearest sample."""
    res = {"burst": [], "gpu": [], "cpu": [], "both": [], "unclassified": 0}
    for t, v in calls:
        near = min(smp, key=lambda s: abs(s["dev_ms"] - t), default=None)
        if near is None or abs(near["dev_ms"] - t) > 3000 or "kgsl" not in near:
            res["unclassified"] += 1
            continue
        k = near["kgsl"]
        g = k[1] >= full_mhz and k[2] == 0
        c = all(m >= ci for _, _, m, ci in near["cpu"])
        res["burst" if g and c else "both" if not g and not c else "gpu" if not g else "cpu"].append(v)
    return {"burst": stats(res["burst"]), "capped": stats(res["gpu"] + res["cpu"] + res["both"]),
            "capped_by": {x: len(res[x]) for x in ("gpu", "cpu", "both")}, "unclassified": res["unclassified"]}


def state_near(t_ms, smp, full_mhz):
    """Round 9: the nearest 2 s sample to a device time -> kgsl (clock, max clock, thermal_pwrlevel, temp °C), the capped
    CPU policies, and the thermal status of the nearest sample that carries one (the slow read, every ~15 s)."""
    if t_ms is None or not smp:
        return None
    near = min(smp, key=lambda s: abs(s["dev_ms"] - t_ms))
    th = [s for s in smp if "thermal" in s]
    nth = min(th, key=lambda s: abs(s["dev_ms"] - t_ms)) if th else None
    k = near.get("kgsl")
    return {"device_ms": t_ms, "sample_dt_ms": near["dev_ms"] - t_ms,
            "kgsl": {"clock_mhz": k[0], "max_clock_mhz": k[1], "thermal_pwrlevel": k[2], "temp_c": k[3] / 1000} if k else None,
            "kgsl_uncapped": bool(k and k[1] >= full_mhz and k[2] == 0),
            "cpu_capped": [f"{p}:{m}/{c}" for p, _, m, c in near.get("cpu", []) if m < c],
            "thermal_status": nth.get("thermal") if nth else None,
            "thermal_sample_dt_ms": (nth["dev_ms"] - t_ms) if nth else None}


def score_timing(a) -> dict:
    report_path = Path(a.report).resolve()
    report = json.loads(report_path.read_text())
    smp = parse_samples(a.samples)
    sets = {}
    for name, t in (report.get("timing") or {}).items():
        calls = t.get("timed_calls", [])
        tot = [c[1] for c in calls]
        run = [c[2] for c in calls if len(c) > 2]
        n = int(t.get("rows", 1))
        e = {"kind": t.get("kind"), "rows": n, "keys": t.get("keys"), "tokens": t.get("tokens"), "prefix_rows": t.get("prefix_rows"),
             "per_call_ms_write_run_read": stats(tot), "per_call_ms_run_only": stats(run),
             "warmup_ms_write_run_read": [c[1] for c in t.get("warmup_calls", [])], "cool": t.get("cool"),
             "finite_markers": t.get("finite_markers", t.get("finite_outputs")), "stopped_early": t.get("stopped_early", False),
             "split": classify([(c[0], c[1]) for c in calls], smp, a.full_mhz), "timed_calls": calls,
             "calls_format": t.get("calls_format")}
        if n > 1 and calls:
            reqs = [(calls[i][0], sum(c[1] for c in calls[i:i + n])) for i in range(0, len(calls) - n + 1, n)]
            e["request_ms_write_run_read"] = stats([v for _, v in reqs])
            e["request_split"] = classify(reqs, smp, a.full_mhz)
        # round 9: the phone's state when the set began (its first warm-up call) and at its first timed call
        warm = t.get("warmup_calls") or []
        e["state_at_set_start"] = state_near(warm[0][0] if warm else (calls[0][0] if calls else None), smp, a.full_mhz)
        e["state_at_first_timed_call"] = state_near(calls[0][0] if calls else None, smp, a.full_mhz)
        sets[name] = e
    doc = {
        "step": "round 5: d1-omni decision graph timing on the S26 gate app (or its Mac stand-in) (scripts/s26_score.py timing)",
        "tag": a.tag or report_path.stem, "scored_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%S%z"),
        "report": rel(report_path), "report_sha256": sha256(report_path),
        "definition_ms": "app wall time of writing the six inputs + CompiledModel.run() + readFloat(scores); a GPU run() can "
                         "return before the compute ends (OpenCL), so the run-only column is not the compute time on the GPU",
        "run": {k: report.get(k) for k in ("graph", "graph_bytes", "signature", "accel", "precision", "threads", "L", "hidden",
                                            "mode", "device", "android", "litert", "compile_ms", "status", "error", "stopped_early",
                                            "warmup_calls_setting", "reps", "rest_ms", "cool_ms", "gpu_state_before_compile",
                                            "memory_before_compile", "memory_after_compile", "memory_at_end", "stand_in")},
        "full_mhz": a.full_mhz, "samples": samples_summary(smp, a.full_mhz),
        "state_before": parse_state(a.state_before), "state_after": parse_state(a.state_after),
        "delegation": delegation(a.logcat), "sets": sets,
    }
    rn = report_path.with_name(report_path.stem + ".ready.txt")   # the chain's ready gate note for this leg (round 9)
    if rn.exists():
        doc["ready_note"] = rn.read_text().strip()
    return doc


def _fmt(v, nd=3):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}g}" if abs(v) < 1e-2 or abs(v) >= 1e5 else f"{v:.{nd + 1}g}"
    return str(v)


def _state_short(st):
    if not st:
        return "—"
    th = (st.get("thermal") or "").replace("Thermal Status: ", "")
    sk = re.search(r"mValue=([-0-9.]+)", st.get("skin") or "")
    bt = re.search(r"(-?\d+)", st.get("battery_temp") or "")
    kg = re.search(r"max_clock_mhz=(\d+) thermal_pwrlevel=(\d+)", st.get("kgsl") or "")
    mem = re.search(r"(\d+)", st.get("mem") or "")
    return (f"th {th or '?'}, skin {sk.group(1) if sk else '?'} °C (cached), batt {int(bt.group(1)) / 10 if bt else '?'} °C, "
            f"cap {st.get('freq_capped') or '?'}, kgsl max {kg.group(1) if kg else '?'} MHz lvl {kg.group(2) if kg else '?'}, "
            f"MemAvail {int(mem.group(1)) // 1024 if mem else '?'} MiB")


def table(a) -> int:
    """Every scored leg of a round (results/s26_parity_<prefix>*.json, s26_timing_<prefix>*.json) and the legs' statuses
    (device/<round>/*.leg_status, s_skipped.txt) as markdown tables, numbers read from the files only."""
    res = Path(a.results).resolve() if a.results else K / "results"
    rnd = Path(a.round_dir).resolve() if a.round_dir else K / "device" / a.round
    out = {"round": a.round, "gate": [], "timing": [], "legs": {}, "skipped": []}
    for p in sorted(rnd.glob("*.leg_status")):
        out["legs"][p.name[: -len(".leg_status")]] = p.read_text().strip()
    if (rnd / "s_skipped.txt").exists():
        out["skipped"] = (rnd / "s_skipped.txt").read_text().splitlines()
    print("| leg | graph | backend | delegate (logcat) | rows | argmax (near-tie apart) | near-tie | max \\|Δp\\| | p95 | mean | "
          "crossings 0.5 / 0.9 | non-finite | red arm | vs Mac CPU same file (bit-equal rows, max \\|Δscore\\|, max \\|Δp\\|) | "
          "compile s | warm median ms | MemAvailable min MiB | VmHWM max MiB | bar |")
    print("|---|---|---|---|---:|---|---|---:|---:|---:|---|---:|---:|---|---:|---:|---:|---:|---|")
    for p in sorted(res.glob(f"s26_parity_{a.prefix}*.json")):
        d = json.loads(p.read_text())
        s, r, sm = d["summary"], d["run"], d.get("samples") or {}
        dl = d["delegation"].get("parsed") or []
        dtxt = "; ".join(f"{x['replaced']}/{x['of']} {x['delegate']} ({x['partitions']} part.)" for x in dl) or "none in logcat"
        sf = d["same_file"].get("mac_cpu8") or next(iter(d["same_file"].values()), None)
        sftxt = "—" if not sf else f"{sf['rows_bit_equal']}/{sf['rows_compared']}, {_fmt(sf['max_abs_dscore_markers'])}, {_fmt(sf['max_abs_dp'])}"
        back = f"{r['accel']} {r['precision'] if r['accel'] == 'gpu' else str(r.get('threads')) + ' threads'}"
        tm = d["timing_ms"]
        row = {"tag": d["tag"], "graph": r["graph"], "backend": back, "delegate": dtxt, "rows": s["rows"],
               "argmax": s["argmax_outside_near_tie"], "near_tie": s["near_tie"], "max_abs_dp": s["max_abs_dp"],
               "p95_abs_dp": s["p95_abs_dp"], "mean_abs_dp": s["mean_abs_dp"], "crossings": s["cutoff_crossings"],
               "nonfinite": s["nonfinite_rows"], "red_arm": s["red_arm_max_abs_dp"], "same_file": sf,
               "compile_s": (r.get("compile_ms") or 0) / 1000, "warm_median_ms": tm.get("warm_median_write_run_read_ms"),
               "first_call_ms": tm.get("first_call_write_run_read_ms"), "mem_available_min_kb": sm.get("mem_available_kb_min"),
               "vmhwm_max_kb": sm.get("app_vmhwm_kb_max"), "bar_pass": s["bar_pass"], "file": rel(p),
               "state_before": _state_short(d.get("state_before")), "state_after": _state_short(d.get("state_after"))}
        out["gate"].append(row)
        mem = row["mem_available_min_kb"]
        hwm = row["vmhwm_max_kb"]
        print(f"| {row['tag']} | {r['graph']} | {back} | {dtxt} | {row['rows']} | {row['argmax']} | {row['near_tie']} | "
              f"{_fmt(row['max_abs_dp'])} | {_fmt(row['p95_abs_dp'])} | {_fmt(row['mean_abs_dp'])} | "
              f"{row['crossings']['0.5']} / {row['crossings']['0.9']} | {row['nonfinite']} | {_fmt(row['red_arm'])} | {sftxt} | "
              f"{row['compile_s']:.1f} | {_fmt(row['warm_median_ms'], 4)} | {mem // 1024 if mem else '—'} | {hwm // 1024 if hwm else '—'} | "
              f"{'PASS' if row['bar_pass'] else 'FAIL'} |")
    print()
    print("| leg | graph | backend | set | positions (P + n) | per call ms median (min–max, n) | burst median (n) | capped median (n) | "
          "request ms median (min–max, n) | compile s | cool waited ms | kgsl max clock min | thermal seen |")
    print("|---|---|---|---|---|---|---|---|---|---:|---|---:|---|")
    for p in sorted(res.glob(f"s26_timing_{a.prefix}*.json")):
        d = json.loads(p.read_text())
        r, sm = d["run"], d.get("samples") or {}
        back = f"{r['accel']} {r['precision'] if r['accel'] == 'gpu' else str(r.get('threads')) + ' threads'}"
        for name, v in d["sets"].items():
            pc, rq = v["per_call_ms_write_run_read"] or {}, v.get("request_ms_write_run_read") or {}
            b, c = v["split"]["burst"] or {}, v["split"]["capped"] or {}
            pos = [pr + t for pr, t in zip(v.get("prefix_rows") or [0] * len(v["tokens"]), v["tokens"])]
            row = {"tag": d["tag"], "graph": r["graph"], "backend": back, "set": name, "positions": pos, "per_call": pc,
                   "burst": b, "capped": c, "request": rq or None, "compile_s": (r.get("compile_ms") or 0) / 1000,
                   "cool": v.get("cool"), "kgsl_max_clock_min": sm.get("kgsl_max_clock_mhz_min"),
                   "thermal_seen": sm.get("thermal_status_seen"), "file": rel(p),
                   "state_before": _state_short(d.get("state_before")), "state_after": _state_short(d.get("state_after"))}
            out["timing"].append(row)
            cw = (v.get("cool") or {}).get("waited_ms", "—")
            print(f"| {d['tag']} | {r['graph']} | {back} | {name} | {pos} | "
                  f"{_fmt(pc.get('median'), 4)} ({_fmt(pc.get('min'), 4)}–{_fmt(pc.get('max'), 4)}, {pc.get('n')}) | "
                  f"{_fmt(b.get('median'), 4)} ({b.get('n', 0)}) | {_fmt(c.get('median'), 4)} ({c.get('n', 0)}) | "
                  + (f"{_fmt(rq.get('median'), 5)} ({_fmt(rq.get('min'), 5)}–{_fmt(rq.get('max'), 5)}, {rq.get('n')})" if rq else "—")
                  + f" | {row['compile_s']:.1f} | {cw} | {row['kgsl_max_clock_min']} | {row['thermal_seen']} |")
    print()
    print("legs:", json.dumps(out["legs"]))
    for line in out["skipped"]:
        print("skipped:", line)
    if a.out:
        o = Path(a.out).resolve()
        assert not o.exists(), f"refusing to overwrite {o}"
        o.write_text(json.dumps(out, indent=1, default=str) + "\n")
        print("wrote", rel(o))
    return 0


# ---------------------------------------------------------------- round 9: generic legs, soft, e2e, tables

NPZ = K / "ref/npz"
R6_RUNS = K / "out/r6_runs"
R7_RUNS = K / "out/r7_runs"
E2E_BUCKETS_VISION = (256, 512, 1024, 2048, 4096)   # round 6's e2e rule (scripts/vision_check.py e2e): no L128
QT = {"choice": 0, "score": 1, "noul": 2}


def adiff(a, b) -> dict:
    a64, b64 = np.asarray(a, np.float64), np.asarray(b, np.float64)
    assert a64.shape == b64.shape, (a64.shape, b64.shape)
    d = np.abs(a64 - b64)
    ref_max = float(np.abs(b64).max()) if b64.size else 0.0
    ref_rms = float(np.sqrt((b64 ** 2).mean())) if b64.size else 0.0
    return {"max_abs": float(np.nanmax(d)) if d.size else 0.0, "mean_abs": float(np.nanmean(d)) if d.size else 0.0,
            "ref_absmax": ref_max, "rel_rms": float(np.sqrt(np.nanmean(d ** 2)) / ref_rms) if ref_rms else None,
            "bit_equal": bool(np.array_equal(np.asarray(a), np.asarray(b))),
            "nonfinite": int((~np.isfinite(np.asarray(a, np.float64))).sum())}


def write_new(path: Path, doc) -> Path:
    assert not path.exists(), f"refusing to overwrite {path}"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    tmp.replace(path)
    return path


def generic_rows(path) -> dict:
    d = json.loads(Path(path).read_text())
    assert d.get("kind") == "generic", f"{path}: not a generic rows file"
    return d


def generic_outputs(report, rows_doc, out_path):
    """out_<tag>.f32 -> [(rows-file row, {output name: float32 array of its declared shape})] in report order (the
    report's rows = the rows file's first rows, in order; per row every output's whole tensor, declared order)."""
    outs = [(o["name"], [int(v) for v in o["shape"]]) for o in rows_doc["outputs"]]
    per_row = sum(int(np.prod(s)) for _, s in outs)
    raw = np.fromfile(out_path, dtype="<f4")
    recs = report["rows"]
    assert raw.size == per_row * len(recs), f"{Path(out_path).name}: {raw.size} floats, {len(recs)} rows need {per_row * len(recs)}"
    if "out_bytes" in report:
        assert raw.size * 4 == int(report["out_bytes"]), (raw.size * 4, report["out_bytes"])
    file_rows = rows_doc["rows"]
    assert [r["key"] for r in recs] == [r["key"] for r in file_rows[: len(recs)]], "the report's rows are not the rows file's first rows in order"
    res, off = [], 0
    for r, row in zip(recs, file_rows):
        assert int(r["index"]) == int(row.get("index", file_rows.index(row))), (r["key"], r["index"])
        d = {}
        for name, shape in outs:
            n = int(np.prod(shape))
            d[name] = raw[off: off + n].reshape(shape).astype(np.float32)
            off += n
        res.append((row, d))
    return res


def mac_generic(graph_path, rows_doc, rows_dir, threads=8):
    """The Mac CPU run (XNNPACK, `threads`) of a generic graph over every row of the rows file, with the same stacked
    input files (scripts/fake_runner.py GenericGraph = the app's contract) -> {key: {output name: array}}."""
    import fake_runner as FR

    g = FR.GenericGraph(Path(graph_path), threads, None, rows_doc, Path(rows_dir))
    try:
        out = {}
        for i, row in enumerate(rows_doc["rows"]):
            key, _, values = g.prepare(row, i)
            outs = g.call(values)[0]
            out[key] = {s["name"]: o.reshape(s["shape"]) for s, o in zip(g.outputs, outs)}
        return out
    finally:
        g.close()


def _mx(per, *path):
    vals = []
    for e in per:
        v = e
        for p in path:
            v = v.get(p) if isinstance(v, dict) else None
            if v is None:
                break
        if v is not None:
            vals.append(v)
    return max(vals) if vals else None


def score_generic(a) -> dict:
    report_path, out_path, rows_path = (Path(x).resolve() for x in (a.report, a.out_file, a.rows))
    report = json.loads(report_path.read_text())
    assert report.get("status") == "DONE", f"report status {report.get('status')}: {report.get('error')}"
    assert report.get("mode") == "generic", report.get("mode")
    rows_doc = generic_rows(rows_path)
    kind = rows_doc["graph_kind"]
    got = generic_outputs(report, rows_doc, out_path)
    same = {}
    for spec in a.same_file:
        label, pair = spec.split("=", 1)
        rp, op = (Path(x).resolve() for x in pair.split(":", 1))
        other = json.loads(rp.read_text())
        assert other.get("status") == "DONE" and other.get("graph") == report.get("graph"), (label, other.get("status"), other.get("graph"))
        same[label] = {"report": rel(rp), "out": rel(op), "out_sha256": sha256(op), "stand_in": other.get("stand_in"),
                       "threads": other.get("threads"), "rows": {row["key"]: d for row, d in generic_outputs(other, rows_doc, op)}}
    per, recs, prefixes, summary = [], [], {}, {}
    if kind == "vision_tower":
        for row, d in got:
            n, rid, c = int(row["patches"]), row["record"], int(row["crop"])
            f = d["features"][0]
            with np.load(NPZ / f"{rid}.npz") as z:
                ref = np.asarray(z["tower_last_hidden_state"][c, :n])
            e = {"key": row["key"], "record": rid, "crop": c, "grid": row["grid"], "patches": n,
                 "features_vs_oracle": adiff(f[:n], ref), "nonfinite_all_rows": int((~np.isfinite(f)).sum())}
            for label, s in same.items():
                e[f"features_vs_{label}"] = adiff(f[:n], s["rows"][row["key"]]["features"][0][:n])
            per.append(e)
        summary = {"crops": len(per), "features_vs_oracle_max_abs": _mx(per, "features_vs_oracle", "max_abs"),
                   "features_vs_oracle_rel_rms_max": _mx(per, "features_vs_oracle", "rel_rms")}
        for label in same:
            summary[f"features_vs_{label}_max_abs"] = _mx(per, f"features_vs_{label}", "max_abs")
            summary[f"features_vs_{label}_bit_equal_crops"] = sum(e[f"features_vs_{label}"]["bit_equal"] for e in per)
    elif kind == "projector":
        graph = K / "out" / report["graph"]
        mac = mac_generic(graph, rows_doc, rows_path.parent)
        by_rec = {}
        for row, d in got:
            m = int(row["prefix_rows"])
            p = d["prefix"][0]
            e = {"key": row["key"], "record": row["record"], "crop": int(row["crop"]), "prefix_rows": m,
                 "nonfinite_all_rows": int((~np.isfinite(p)).sum()),
                 "prefix_vs_mac_cpu8_same_soft": adiff(p[:m], mac[row["key"]]["prefix"][0][:m])}
            for label, s in same.items():
                e[f"prefix_vs_{label}"] = adiff(p[:m], s["rows"][row["key"]]["prefix"][0][:m])
            per.append(e)
            by_rec.setdefault(row["record"], []).append((int(row["crop"]), p[:m]))
        for rid, parts in by_rec.items():
            pre = np.concatenate([p for _, p in sorted(parts, key=lambda t: t[0])]).astype(np.float32)
            prefixes[rid] = pre
            with np.load(NPZ / f"{rid}.npz") as z:
                ref = np.asarray(z["prefix"])
            tru, eag = np.load(R6_RUNS / "truth" / f"{rid}.npy"), np.load(R6_RUNS / "eager" / f"{rid}.npy")
            d_o, d_t, d_e, d_pt = adiff(pre, ref), adiff(pre, tru), adiff(pre, eag), adiff(ref, tru)
            r = {"record": rid, "P": int(pre.shape[0]), "vs_oracle": d_o, "vs_float64_truth": d_t, "vs_eager": d_e,
                 "provider_fp32_vs_truth": {"max_abs": d_pt["max_abs"], "rel_rms": d_pt["rel_rms"]},
                 "truth_rel_rms_over_provider": d_t["rel_rms"] / d_pt["rel_rms"],
                 "within_2x_provider_rel_rms": bool(d_t["rel_rms"] <= 2.0 * d_pt["rel_rms"]),
                 "truth_max_abs_over_provider": d_t["max_abs"] / d_pt["max_abs"],
                 "worst_token": int(np.nan_to_num(np.abs(pre.astype(np.float64) - tru), nan=np.inf).max(1).argmax())}
            for lab, sub in (("mac_cpu8_chain_r6", "cpu_fp32_fp16"), ("mac_metal_fp32_chain_r6", "gpu_fp32_fp16")):
                f = R6_RUNS / sub / f"{rid}.npy"
                if f.exists():
                    r[f"vs_{lab}"] = adiff(pre, np.load(f))
            recs.append(r)
        summary = {"crops": len(per), "records": len(recs),
                   "prefix_vs_oracle_max_abs": _mx(recs, "vs_oracle", "max_abs"),
                   "prefix_vs_oracle_rel_rms_max": _mx(recs, "vs_oracle", "rel_rms"),
                   "prefix_vs_truth_max_abs": _mx(recs, "vs_float64_truth", "max_abs"),
                   "prefix_vs_eager_max_abs": _mx(recs, "vs_eager", "max_abs"),
                   "truth_rel_rms_over_provider_max": _mx(recs, "truth_rel_rms_over_provider"),
                   "within_2x_provider_rel_rms": f"{sum(r['within_2x_provider_rel_rms'] for r in recs)}/{len(recs)}",
                   "vs_mac_cpu8_chain_r6_max_abs": _mx(recs, "vs_mac_cpu8_chain_r6", "max_abs"),
                   "vs_mac_metal_fp32_chain_r6_max_abs": _mx(recs, "vs_mac_metal_fp32_chain_r6", "max_abs"),
                   "vs_mac_cpu8_chain_r6_bit_equal_records": sum(bool((r.get("vs_mac_cpu8_chain_r6") or {}).get("bit_equal")) for r in recs),
                   "prefix_vs_mac_cpu8_same_soft_max_abs": _mx(per, "prefix_vs_mac_cpu8_same_soft", "max_abs"),
                   "prefix_vs_mac_cpu8_same_soft_bit_equal_crops": sum(e["prefix_vs_mac_cpu8_same_soft"]["bit_equal"] for e in per),
                   "soft_from": rows_doc.get("from_leg")}
    elif kind == "audio":
        stores = {lab: np.load(R7_RUNS / f) for lab, f in (("mac_cpu8_r7", "audio_cpu_fp16.npz"), ("mac_metal_fp32_r7", "audio_gpu_fp32_fp16.npz"))}
        for row, d in got:
            P, rid = int(row["P"]), row["record"]
            p = d["prefix"][0]
            pre = p[:P].astype(np.float32)
            prefixes[rid] = pre
            with np.load(NPZ / f"{rid}.npz") as z:
                ref = np.asarray(z["prefix"])
            e = {"key": row["key"], "record": rid, "P": P, "T_b": row.get("T_b"), "prefix_vs_oracle": adiff(pre, ref),
                 "nonfinite_all_rows": int((~np.isfinite(p)).sum())}
            for lab, z in stores.items():
                if f"{rid}__host" in z.files:
                    e[f"prefix_vs_{lab}"] = adiff(pre, np.asarray(z[f"{rid}__host"]))
            for label, s in same.items():
                e[f"prefix_vs_{label}"] = adiff(pre, s["rows"][row["key"]]["prefix"][0][:P])
            per.append(e)
        summary = {"clips": len(per), "prefix_vs_oracle_max_abs": _mx(per, "prefix_vs_oracle", "max_abs"),
                   "prefix_vs_oracle_rel_rms_max": _mx(per, "prefix_vs_oracle", "rel_rms")}
        for lab in list(stores) + list(same):
            summary[f"prefix_vs_{lab}_max_abs"] = _mx(per, f"prefix_vs_{lab}", "max_abs")
            summary[f"prefix_vs_{lab}_bit_equal_clips"] = sum(bool((e.get(f"prefix_vs_{lab}") or {}).get("bit_equal")) for e in per)
    else:
        raise ValueError(f"unknown graph_kind {kind}")
    nonfinite = sum(sum(int(v) for v in r["nonfinite"].values()) for r in report["rows"])
    summary.update(rows=len(got), rows_in_file=len(rows_doc["rows"]), nonfinite_values=nonfinite,
                   finite_pass=bool(nonfinite == 0 and len(got) == len(rows_doc["rows"])),
                   bar_note="no bar here: the vision / audio PASS / FAIL is the end to end (s26_score.py e2e9)")
    if a.prefix_out and prefixes:
        po = Path(a.prefix_out).resolve()
        assert not po.exists(), f"refusing to overwrite {po}"
        np.savez(po, **prefixes)
        summary["prefix_npz"] = rel(po)
    smp = parse_samples(a.samples)
    ms = [r["write_run_read_ms"] for r in report["rows"]]
    return {
        "step": f"round 9: d1-omni {kind} graph on the S26 gate app's generic mode (or its Mac stand-in) (scripts/s26_score.py generic)",
        "tag": a.tag or report_path.stem, "scored_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "report": rel(report_path), "report_sha256": sha256(report_path), "out_file": rel(out_path),
        "out_bytes": out_path.stat().st_size, "out_sha256": sha256(out_path), "rows_file": rel(rows_path),
        "rows_sha256": sha256(rows_path), "graph_kind": kind,
        "run": {k: report.get(k) for k in ("graph", "graph_bytes", "signature", "accel", "precision", "threads", "limit", "mode",
                                            "device", "android", "litert", "compile_ms", "status", "stopped_early", "io",
                                            "signature_input_count", "signature_output_count", "memory_before_compile",
                                            "memory_after_compile", "memory_at_end", "stand_in")},
        "summary": summary, "per_row": per, "records": recs,
        "same_file": {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in same.items()},
        "delegation": delegation(a.logcat),
        "timing_ms": {**report.get("summary", {}), "median_all_rows_write_run_read": float(np.median(ms)) if ms else None},
        "samples": samples_summary(smp, a.full_mhz),
        "state_before": parse_state(a.state_before), "state_after": parse_state(a.state_after),
    }


def make_soft(a) -> int:
    """A tower leg's features -> the host unshuffle -> the projector's generic rows file + its stacked soft input."""
    sys.path.insert(0, str(K / "host"))
    import d1_vision_host as V

    report_path, out_path, rows_path = (Path(x).resolve() for x in (a.report, a.out_file, a.rows))
    report = json.loads(report_path.read_text())
    assert report.get("status") == "DONE" and report.get("mode") == "generic", (report.get("status"), report.get("mode"))
    rows_doc = generic_rows(rows_path)
    assert rows_doc["graph_kind"] == "vision_tower", rows_doc["graph_kind"]
    got = generic_outputs(report, rows_doc, out_path)
    assert len(got) == len(rows_doc["rows"]), f"the tower leg ran {len(got)} of {len(rows_doc['rows'])} crops"
    out_dir = Path(a.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = report_path.stem
    soft_name = f"g9_pj_{tag}_soft.f32"
    softs, rows, nonfinite = [], [], 0
    for row, d in got:
        h, w = (int(v) for v in row["grid"])
        f = d["features"][0][: h * w]
        nonfinite += int((~np.isfinite(f)).sum())
        cells = V.pixel_unshuffle(f, (h, w))
        softs.append(V.projector_input(cells))
        rows.append({"key": row["key"], "index": len(rows), "record": row["record"], "crop": row["crop"], "grid": [h, w],
                     "prefix_rows": int(cells.shape[0])})
    data = b"".join(s.astype("<f4").tobytes() for s in softs)
    doc = {"kind": "generic", "graph_kind": "projector", "signature": "projector",
           "inputs": [{"name": "soft", "dtype": "float32", "shape": [1, 256, 3072], "file": soft_name}],
           "outputs": [{"name": "prefix", "dtype": "float32", "shape": [1, 256, 1024]}],
           "rows": rows, "sets": [{"name": "p_one_crop", "kind": "single", "rows": ["img_dogs_01/c0"]}],
           "made_by": "scripts/s26_score.py soft (host/d1_vision_host.py pixel_unshuffle + projector_input)",
           "from_leg": {"tag": tag, "report": rel(report_path), "report_sha256": sha256(report_path), "out": rel(out_path),
                        "out_sha256": sha256(out_path), "graph": report.get("graph"), "accel": report.get("accel"),
                        "precision": report.get("precision"), "features_nonfinite": nonfinite}}
    written = []
    for name, payload in ((soft_name, data), (a.name, (json.dumps(doc, indent=1) + "\n").encode())):
        p = out_dir / name
        if p.exists():
            assert p.read_bytes() == payload, f"{p} exists with other content"
        else:
            tmp = p.with_name(p.name + ".tmp")
            tmp.write_bytes(payload)
            tmp.replace(p)
        written.append((name, p))
    if a.sha_file:
        with open(a.sha_file, "a") as f:
            for name, p in written:
                f.write(f"{sha256(p)} {name}\n")
    print(json.dumps({"rows_file": rel(out_dir / a.name), "soft": rel(out_dir / soft_name), "crops": len(rows),
                      "soft_bytes": len(data), "features_nonfinite": nonfinite}, indent=1))
    return 0


def e2e9(a) -> int:
    """The phone's prefixes through the Mac CPU text decision graph (fp16 form) -> probabilities vs the oracle."""
    import litert_run as R

    prefixes = {}
    for path in a.prefix:          # e.g. the T1001 leg (6 clips) and the T2001 leg (card_topic) of one backend
        with np.load(path) as z:
            for k in z.files:
                assert k not in prefixes, f"record {k} in two prefix files"
                prefixes[k] = np.asarray(z[k], np.float32)
    odoc, orows = oracle_rows()
    rows = [o for o in orows.values() if o["id"] in prefixes]
    assert rows, "no oracle row for these prefixes"
    mode = {"vision": "image", "audio": "audio"}[a.kind]
    for o in rows:
        assert o["mode"] == mode and prefixes[o["id"]].shape == (o["P"], 1024), (o["key"], o["mode"], prefixes[o["id"]].shape, o["P"])

    def bucket(o):
        pos = o["P"] + o["n"]
        if a.kind == "audio":
            assert pos <= 256, (o["key"], pos)
            return 256
        return next(b for b in E2E_BUCKETS_VISION if pos <= b)

    by_L = {}
    for o in rows:
        by_L.setdefault(bucket(o), []).append(o)
    lit, lit_o, graphs, bucket_of = {}, {}, {}, {}
    for L, items in sorted(by_L.items()):
        path = K / f"out/d1omni_decide_L{L}_fp16.tflite"
        cm, desc = R.open_compiled(path, "cpu", threads=8)
        run = R.Runner(cm, next(iter(cm.get_signature_list())))
        for o in items:
            with np.load(NPZ / f"{o['id']}.npz") as z:
                oracle_prefix = np.asarray(z["prefix"], np.float32)
            for store, pre in ((lit, prefixes[o["id"]]), (lit_o, oracle_prefix)):
                x = H.build_inputs(o["ids"], pre, L)
                oh = np.zeros((1, 3), np.float32)
                oh[0, QT[o["type"]]] = 1.0
                x["qtype_onehot"] = oh
                sc = run(x)
                store[o["key"]] = readout(np.asarray([sc[o["P"] + m] for m in o["markers"][: o["K"]]], np.float32), o)
            bucket_of[o["key"]] = L
        run.close()
        if hasattr(cm, "close"):
            cm.close()
        graphs[str(L)] = {"file": rel(path), "sha256": sha256(path), "rows": len(items), "options": desc}
    ref = {k: (o["probs"], o["logits"]) for k, o in orows.items()}
    st = compare(rows, lit, ref)
    st_o = compare(rows, lit_o, ref)
    share = max(max(abs(x - y) for x, y in zip(lit[o["key"]][0], lit_o[o["key"]][0])) for o in rows)
    macs = {}
    if a.kind == "vision":
        for lab, f in (("mac_cpu8_chain_r6", "results/vision_parity_cpu_fp32_fp16.json"), ("mac_metal_fp32_chain_r6", "results/vision_parity_gpu_fp32_fp16.json")):
            d = json.loads((K / f).read_text())
            macs[lab] = (f, {r["key"]: r["vision_graph"] for r in d["e2e"]["text_fp16_cpu"]["per_row"]})
    else:
        for lab, f in (("mac_cpu8_r7", "results/audio_parity_cpu_fp16.json"), ("mac_metal_fp32_r7", "results/audio_parity_gpu_fp32_fp16.json")):
            d = json.loads((K / f).read_text())
            macs[lab] = (f, {r["key"]: r["probs_host_mel"] for r in d["per_row"]})
    vs_mac = {}
    for lab, (f, pr) in macs.items():
        keys = [o["key"] for o in rows if o["key"] in pr]
        vs_mac[lab] = {"file": f, "rows_compared": len(keys),
                       "max_abs_dp": max((max(abs(x - y) for x, y in zip(lit[k][0], pr[k])) for k in keys), default=None),
                       "argmax_agree": sum(int(np.argmax(lit[k][0]) == np.argmax(pr[k])) for k in keys)}
    per = []
    for o in rows:
        k = o["key"]
        e = {"key": k, "mode": o["mode"], "L": bucket_of[k], "P": o["P"], "n": o["n"], "K": o["K"], "type": o["type"],
             "probs": lit[k][0], "logits": lit[k][1], "probs_oracle": o["probs"], "probs_oracle_prefix_same_graph": lit_o[k][0],
             "max_abs_dp": max(abs(x - y) for x, y in zip(lit[k][0], o["probs"])), "near_tie": o["near_tie"],
             "argmax_equal": int(np.argmax(lit[k][0])) == int(np.argmax(o["probs"]))}
        for lab, (_, pr) in macs.items():
            if k in pr:
                e[f"probs_{lab}"] = pr[k]
        per.append(e)
    records = sorted({o["id"] for o in rows})
    doc = {"step": f"round 9: end to end of the S26 {a.kind} prefixes through the Mac CPU text graph (scripts/s26_score.py e2e9)",
           "tag": a.tag, "scored_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "prefix_npz": {rel(Path(p).resolve()): sha256(Path(p)) for p in a.prefix}, "records": records, "text_graphs": graphs,
           "oracle": {"file": "ref/records_ref.json", "version": odoc.get("version"), "sha256": sha256(ORACLE)}, "bar": BAR,
           "summary": {"rows": st["rows_compared"], "records": len(records), "max_abs_dp": st["max_abs_dp"],
                       "p95_abs_dp": st["p95_abs_dp"], "mean_abs_dp": st["mean_abs_dp"], "max_abs_dlogit": st["max_abs_dlogit"],
                       "argmax_outside_near_tie": f"{st['argmax']['equal_outside_near_tie']}/{st['argmax']['rows_outside_near_tie']}",
                       "near_tie": f"{st['argmax']['near_tie_equal']}/{st['argmax']['near_tie_rows']}",
                       "cutoff_crossings": st["cutoff_crossings"], "nonfinite_rows": len(st["nonfinite_rows"]),
                       "bar_pass": st["bar_pass"], "oracle_prefix_same_graph_max_abs_dp": st_o["max_abs_dp"],
                       "media_share_max_abs_dp": share, "vs_mac": vs_mac},
           "oracle_comparison": st, "oracle_prefix_same_graph": {k: st_o[k] for k in ("max_abs_dp", "mean_abs_dp", "bar_pass")},
           "per_row": per}
    out = Path(a.out).resolve() if a.out else K / "results" / f"s26_parity_{a.tag}_e2e_r9.json"
    write_new(out, doc)
    print(json.dumps({"out": rel(out), **{k: v for k, v in doc["summary"].items() if k != "vs_mac"},
                      "vs_mac": {k: [v["rows_compared"], v["max_abs_dp"], v["argmax_agree"]] for k, v in vs_mac.items()}}, indent=1, default=str))
    return 0


def table9(a) -> int:
    """Round 9's legs as markdown tables, numbers read from the result files only -> results/s26_summary_r9.json."""
    res = Path(a.results).resolve() if a.results else K / "results"
    rnd = Path(a.round_dir).resolve() if a.round_dir else K / "device" / a.round
    sfx = a.suffix
    out = {"round": a.round, "text_gate": [], "generic": [], "e2e": [], "timing": [], "legs": {}, "skipped": []}
    for p in sorted(rnd.glob("*.leg_status")):
        out["legs"][p.name[: -len(".leg_status")]] = p.read_text().strip()
    if (rnd / "s_skipped.txt").exists():
        out["skipped"] = (rnd / "s_skipped.txt").read_text().splitlines()

    def back(r):
        return f"{r.get('accel')} {r.get('precision') if r.get('accel') == 'gpu' else str(r.get('threads')) + ' threads'}"

    def dl(d):
        x = (d.get("delegation") or {}).get("parsed") or []
        return "; ".join(f"{y['replaced']}/{y['of']} {y['delegate']} ({y['partitions']} part.)" for y in x) or "none in logcat"

    print("| leg | graph | backend | delegate | rows | argmax | near-tie | max \\|Δp\\| | p95 | mean | crossings | non-finite | "
          "red arm | vs Mac CPU same file (bit-equal, max \\|Δscore\\|, max \\|Δp\\|) | compile s | warm median ms | MemAvail min MiB | VmHWM max MiB | bar |")
    print("|---|---|---|---|---:|---|---|---:|---:|---:|---|---:|---:|---|---:|---:|---:|---:|---|")
    files = sorted(p for p in res.glob(f"s26_parity_*{sfx}.json") if not p.name.endswith(f"_e2e{sfx}.json"))
    for p in files:
        d = json.loads(p.read_text())
        if "graph_kind" in d:
            continue
        s, r, sm = d["summary"], d["run"], d.get("samples") or {}
        sf = d["same_file"].get("mac_cpu8") or next(iter(d["same_file"].values()), None)
        sftxt = "—" if not sf else f"{sf['rows_bit_equal']}/{sf['rows_compared']}, {_fmt(sf['max_abs_dscore_markers'])}, {_fmt(sf['max_abs_dp'])}"
        tm = d["timing_ms"]
        row = {"tag": d["tag"], "graph": r["graph"], "backend": back(r), "delegate": dl(d), **s, "same_file": sf,
               "compile_s": (r.get("compile_ms") or 0) / 1000, "warm_median_ms": tm.get("warm_median_write_run_read_ms"),
               "mem_available_min_kb": sm.get("mem_available_kb_min"), "vmhwm_max_kb": sm.get("app_vmhwm_kb_max"), "file": rel(p)}
        out["text_gate"].append(row)
        mem, hwm = row["mem_available_min_kb"], row["vmhwm_max_kb"]
        print(f"| {row['tag']} | {r['graph']} | {row['backend']} | {row['delegate']} | {s['rows']} | {s['argmax_outside_near_tie']} | "
              f"{s['near_tie']} | {_fmt(s['max_abs_dp'])} | {_fmt(s['p95_abs_dp'])} | {_fmt(s['mean_abs_dp'])} | "
              f"{s['cutoff_crossings']['0.5']} / {s['cutoff_crossings']['0.9']} | {s['nonfinite_rows']} | {_fmt(s['red_arm_max_abs_dp'])} | "
              f"{sftxt} | {row['compile_s']:.1f} | {_fmt(row['warm_median_ms'], 4)} | {mem // 1024 if mem else '—'} | "
              f"{hwm // 1024 if hwm else '—'} | {'PASS' if s['bar_pass'] else 'FAIL'} |")
    print()
    print("| leg | kind | graph | backend | delegate | rows | vs oracle (max \\|Δ\\|) | vs Mac (max \\|Δ\\|, bit-equal) | non-finite | "
          "compile s | warm median ms | MemAvail min MiB | VmHWM max MiB |")
    print("|---|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|")
    for p in files:
        d = json.loads(p.read_text())
        if "graph_kind" not in d:
            continue
        s, r, sm, kind = d["summary"], d["run"], d.get("samples") or {}, d["graph_kind"]
        if kind == "vision_tower":
            vo = f"features {_fmt(s.get('features_vs_oracle_max_abs'))}"
            vm = f"features {_fmt(s.get('features_vs_mac_cpu8_max_abs'))}, {s.get('features_vs_mac_cpu8_bit_equal_crops', '—')}/{s['crops']}"
        elif kind == "projector":
            vo = (f"prefix {_fmt(s.get('prefix_vs_oracle_max_abs'))}, truth {_fmt(s.get('prefix_vs_truth_max_abs'))}, "
                  f"rel_rms ratio {_fmt(s.get('truth_rel_rms_over_provider_max'))} ({s.get('within_2x_provider_rel_rms')})")
            vm = (f"same soft {_fmt(s.get('prefix_vs_mac_cpu8_same_soft_max_abs'))}, {s.get('prefix_vs_mac_cpu8_same_soft_bit_equal_crops')}/{s['crops']}; "
                  f"chain CPU {_fmt(s.get('vs_mac_cpu8_chain_r6_max_abs'))} / Metal {_fmt(s.get('vs_mac_metal_fp32_chain_r6_max_abs'))}")
        else:
            vo = f"prefix {_fmt(s.get('prefix_vs_oracle_max_abs'))}"
            vm = (f"r7 CPU {_fmt(s.get('prefix_vs_mac_cpu8_r7_max_abs'))} ({s.get('prefix_vs_mac_cpu8_r7_bit_equal_clips')}/{s['clips']}), "
                  f"r7 Metal {_fmt(s.get('prefix_vs_mac_metal_fp32_r7_max_abs'))}, same-file CPU {_fmt(s.get('prefix_vs_mac_cpu8_max_abs'))}")
        tm = d["timing_ms"]
        row = {"tag": d["tag"], "kind": kind, "graph": r["graph"], "backend": back(r), "delegate": dl(d), "summary": s,
               "compile_s": (r.get("compile_ms") or 0) / 1000, "warm_median_ms": tm.get("warm_median_write_run_read_ms"),
               "mem_available_min_kb": sm.get("mem_available_kb_min"), "vmhwm_max_kb": sm.get("app_vmhwm_kb_max"), "file": rel(p)}
        out["generic"].append(row)
        mem, hwm = row["mem_available_min_kb"], row["vmhwm_max_kb"]
        print(f"| {row['tag']} | {kind} | {r['graph']} | {row['backend']} | {row['delegate']} | {s['rows']} | {vo} | {vm} | "
              f"{s['nonfinite_values']} | {row['compile_s']:.1f} | {_fmt(row['warm_median_ms'], 4)} | {mem // 1024 if mem else '—'} | "
              f"{hwm // 1024 if hwm else '—'} |")
    print()
    print("| leg (e2e) | records | rows | argmax | near-tie | max \\|Δp\\| | p95 | mean | crossings | non-finite | text graph's own share | "
          "media share | vs Mac chains (max \\|Δp\\|) | bar |")
    print("|---|---:|---:|---|---|---:|---:|---:|---|---:|---:|---:|---|---|")
    for p in sorted(res.glob(f"s26_parity_*_e2e{sfx}.json")):
        d = json.loads(p.read_text())
        s = d["summary"]
        vm = ", ".join(f"{k} {_fmt(v['max_abs_dp'])}" for k, v in s["vs_mac"].items())
        out["e2e"].append({"tag": d["tag"], "file": rel(p), **s})
        print(f"| {d['tag']} | {s['records']} | {s['rows']} | {s['argmax_outside_near_tie']} | {s['near_tie']} | {_fmt(s['max_abs_dp'])} | "
              f"{_fmt(s['p95_abs_dp'])} | {_fmt(s['mean_abs_dp'])} | {s['cutoff_crossings']['0.5']} / {s['cutoff_crossings']['0.9']} | "
              f"{s['nonfinite_rows']} | {_fmt(s['oracle_prefix_same_graph_max_abs_dp'])} | {_fmt(s['media_share_max_abs_dp'])} | {vm} | "
              f"{'PASS' if s['bar_pass'] else 'FAIL'} |")
    print()
    print("| leg | graph | backend | set | per call ms median (min–max, n) | burst median (n) | capped median (n) | request ms | "
          "compile s | cool waited ms | kgsl max clock min | thermal seen |")
    print("|---|---|---|---|---|---|---|---|---:|---|---:|---|")
    for p in sorted(res.glob(f"s26_timing_*{sfx}.json")):
        d = json.loads(p.read_text())
        r, sm = d["run"], d.get("samples") or {}
        for name, v in d["sets"].items():
            pc, rq = v["per_call_ms_write_run_read"] or {}, v.get("request_ms_write_run_read") or {}
            b, c = v["split"]["burst"] or {}, v["split"]["capped"] or {}
            row = {"tag": d["tag"], "graph": r["graph"], "backend": back(r), "set": name, "keys": v.get("keys"), "per_call": pc,
                   "burst": b, "capped": c, "capped_by": v["split"].get("capped_by"), "request": rq or None,
                   "compile_s": (r.get("compile_ms") or 0) / 1000, "cool": v.get("cool"),
                   "kgsl_max_clock_min": sm.get("kgsl_max_clock_mhz_min"), "thermal_seen": sm.get("thermal_status_seen"),
                   "file": rel(p)}
            out["timing"].append(row)
            cw = (v.get("cool") or {}).get("waited_ms", "—")
            print(f"| {d['tag']} | {r['graph']} | {row['backend']} | {name} | "
                  f"{_fmt(pc.get('median'), 4)} ({_fmt(pc.get('min'), 4)}–{_fmt(pc.get('max'), 4)}, {pc.get('n')}) | "
                  f"{_fmt(b.get('median'), 4)} ({b.get('n', 0)}) | {_fmt(c.get('median'), 4)} ({c.get('n', 0)}) | "
                  + (f"{_fmt(rq.get('median'), 5)} ({rq.get('n')})" if rq else "—")
                  + f" | {row['compile_s']:.1f} | {cw} | {row['kgsl_max_clock_min']} | {row['thermal_seen']} |")
    print()
    print("legs:", json.dumps(out["legs"]))
    for line in out["skipped"]:
        print("skipped:", line)
    if a.out:
        write_new(Path(a.out).resolve(), out)
        print("wrote", rel(Path(a.out).resolve()))
    return 0


def main9() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("generic", "soft", "e2e9", "table9"))
    ap.add_argument("--report", default="")
    ap.add_argument("--out-file", default="")
    ap.add_argument("--rows", default="")
    ap.add_argument("--same-file", action="append", default=[], help="label=<report.json>:<out.f32>")
    ap.add_argument("--prefix-out", default="")
    ap.add_argument("--logcat", default="")
    ap.add_argument("--samples", default="")
    ap.add_argument("--state-before", default="")
    ap.add_argument("--state-after", default="")
    ap.add_argument("--full-mhz", type=int, default=1300)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--name", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--sha-file", default="")
    ap.add_argument("--prefix", action="append", default=[], help="a leg's <tag>.prefix.npz (repeatable)")
    ap.add_argument("--kind", choices=("vision", "audio"))
    ap.add_argument("--round", default="r9")
    ap.add_argument("--suffix", default="_r9")
    ap.add_argument("--results", default="")
    ap.add_argument("--round-dir", default="")
    a = ap.parse_args()
    if a.what == "soft":
        assert a.report and a.out_file and a.rows and a.name and a.out_dir, "soft needs --report --out-file --rows --name --out-dir"
        return make_soft(a)
    if a.what == "e2e9":
        assert a.tag and a.prefix and a.kind, "e2e9 needs --tag --prefix --kind"
        return e2e9(a)
    if a.what == "table9":
        return table9(a)
    assert a.report and a.out_file and a.rows, "generic needs --report --out-file --rows"
    out = Path(a.out).resolve() if a.out else K / "results" / f"s26_parity_{a.tag or Path(a.report).stem}_r9.json"
    assert not out.exists(), f"refusing to overwrite {out}"
    doc = score_generic(a)
    write_new(out, doc)
    print(json.dumps({"out": rel(out), "tag": doc["tag"], "kind": doc["graph_kind"], **doc["summary"],
                      "delegate": doc["delegation"]["lines"][:3], "compile_ms": doc["run"]["compile_ms"],
                      "warm_median_ms": doc["timing_ms"].get("warm_median_write_run_read_ms")}, indent=1, default=str))
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] in ("generic", "soft", "e2e9", "table9"):
        return main9()
    if len(sys.argv) > 1 and sys.argv[1] == "table":
        tp = argparse.ArgumentParser()
        tp.add_argument("what")
        tp.add_argument("--round", default="r5")
        tp.add_argument("--prefix", default="", help="only results/s26_*_<prefix>*.json (default: every leg)")
        tp.add_argument("--results", default="", help="the results dir (default results/)")
        tp.add_argument("--round-dir", default="", help="the round dir (default device/<round>)")
        tp.add_argument("--out", default="")
        return table(tp.parse_args())
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("gate", "timing"))
    ap.add_argument("--report", required=True)
    ap.add_argument("--sel", default="")
    ap.add_argument("--rows", default="")
    ap.add_argument("--logcat", default="")
    ap.add_argument("--samples", default="")
    ap.add_argument("--state-before", default="")
    ap.add_argument("--state-after", default="")
    ap.add_argument("--same-file", action="append", default=[], help="label=<report.json>:<sel.f32>")
    ap.add_argument("--round3", action="append", default=[], help="label=<results/litert_*_parity_*.json>")
    ap.add_argument("--full-mhz", type=int, default=1300)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    tag = a.tag or Path(a.report).stem
    out = Path(a.out).resolve() if a.out else K / "results" / (f"s26_parity_{tag}.json" if a.what == "gate" else f"s26_timing_{tag}.json")
    assert not out.exists(), f"refusing to overwrite {out}"
    if a.what == "gate":
        assert a.sel and a.rows, "gate needs --sel and --rows"
        doc = score_gate(a)
    else:
        doc = score_timing(a)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    tmp.replace(out)
    if a.what == "gate":
        s = doc["summary"]
        line = {"out": rel(out), **s, "delegate": doc["delegation"]["lines"][:3], "compile_ms": doc["run"]["compile_ms"],
                "warm_median_ms": doc["timing_ms"].get("warm_median_write_run_read_ms"),
                "same_file": {k: {kk: v[kk] for kk in ("rows_compared", "rows_bit_equal", "max_abs_dscore_markers", "max_abs_dp", "argmax_agree")}
                              for k, v in doc["same_file"].items()},
                "round3": doc["round3_same_file"]}
    else:
        line = {"out": rel(out), "status": doc["run"]["status"], "compile_ms": doc["run"]["compile_ms"],
                "sets": {k: {"per_call_median": (v["per_call_ms_write_run_read"] or {}).get("median"),
                             "n": (v["per_call_ms_write_run_read"] or {}).get("n"),
                             "request_median": (v.get("request_ms_write_run_read") or {}).get("median"),
                             "burst": v["split"]["burst"], "capped_n": (v["split"]["capped"] or {}).get("n")} for k, v in doc["sets"].items()}}
    print(json.dumps(line, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
