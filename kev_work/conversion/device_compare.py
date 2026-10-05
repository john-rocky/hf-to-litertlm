"""Phone gate scoring: one run of the debug app on the phone (or of a Mac stand-in that writes the same two files)
against the reference.

    python scripts/device_compare.py --report device/<tag>.json --hsel device/hsel_<tag>.f32 --L 512 \
        [--rows device/rows_L512.json] [--out device/device_parity_<tag>.json] \
        [--expect-rows results/litert_<acc>_rows_L512_v2_fp16fc_i8emb.json \
         --expect-parity results/litert_<acc>_parity_L512_v2_fp16fc_i8emb.json]        (self-test)

Input contract (app): <report>.json = {status, graph, accel, precision, L, compile_ms, rows: [{key, n, k, finite,
write_ms, run_ms, read_ms, write_run_read_ms}], summary}; hsel_<tag>.f32 = little-endian float32, for every report row
in order (1 + k) rows of 1024 = hidden[decide], hidden[opt_1..k] of that question. The rows file (default: the
report's rows_file under device/) must list the same keys in the same order; its decide / opts must equal the
reference's.

Readout and statistics are the desktop gate's, imported (r2_common.py: Head = head.pt in numpy float32,
compare_question, summarize; r3_common.py: gpu_vs_cpu), so a Mac stand-in run reproduces litert_parity.py: argmax per
question, max / mean / p95 |dp| over all options, h_sel max |diff| vs the reference npz, near-tie rows apart, flips, red
arm (graph(red_arm_000) vs reference(tv4_000) must exceed 0.02). A question whose h_sel is not finite is listed,
excluded from the |dp| statistics and fails both bar readings (strict = argmax on all questions; near-tie apart =
argmax on the non-near-tie questions, near-tie flips listed). Same-file comparisons (probs and, where an h_sel store
exists, h_sel): the Mac CPU 8-thread and Mac Metal fp32 rows of the desktop gate
(results/litert_{cpu,gpu_f32}_rows_L{L}_<v>.json) and the Mac stand-in stores device/mac_mock/hsel_mac_{cpu,gpu_f32}_L{L}.npz.
Output: device/device_parity_<tag>.json (never overwritten)."""
import argparse
import json
from pathlib import Path

import numpy as np

from r2_common import BAR, K, ORACLE_NPZ, Head, compare_question, dump_json, load_oracle, qkey, sha256_file, summarize
from r3_common import gpu_vs_cpu

HIDDEN = 1024
VARIANT = "v2_fp16fc_i8emb"


def rel(p):
    p = Path(p).resolve()
    return str(p.relative_to(K)) if p.is_relative_to(K) else str(p)


def slice_hsel(report, rows_doc, hsel_path):
    """-> [(key, h_sel [1+k, 1024] float32)] in report order, after checking keys / k / byte count."""
    recs = report["rows"]
    raw = np.fromfile(hsel_path, dtype="<f4")
    expected = sum((1 + r["k"]) * HIDDEN for r in recs)
    assert raw.size == expected, f"{hsel_path.name}: {raw.size} floats, report rows need {expected}"
    file_keys = [r["key"] for r in rows_doc["rows"]]
    assert [r["key"] for r in recs] == file_keys[: len(recs)], "report rows are not the rows file's first rows in order"
    by_key = {r["key"]: r for r in rows_doc["rows"]}
    out, off = [], 0
    for r in recs:
        row = by_key[r["key"]]
        assert r["k"] == len(row["opts"]) and r.get("n", len(row["ids"])) == len(row["ids"]), r["key"]
        n = (1 + r["k"]) * HIDDEN
        out.append((r["key"], raw[off: off + n].reshape(1 + r["k"], HIDDEN)))
        off += n
    return out


def bar_readings(rows, nonfinite_keys, near_keys, overall, n_questions):
    max_ok = overall.get("max_abs_dp") is not None and overall["max_abs_dp"] <= BAR["max_abs_dp"]
    mean_ok = overall.get("mean_abs_dp_all_options") is not None and overall["mean_abs_dp_all_options"] <= BAR["mean_abs_dp"]
    strict = overall.get("argmax_equal") == n_questions and max_ok and mean_ok and not nonfinite_keys
    non = [r for r in rows if r["key"] not in near_keys]
    non_eq = sum(r["argmax_equal"] for r in non)
    non_total = len(non) + sum(1 for k in nonfinite_keys if k not in near_keys)
    near = [r for r in rows if r["key"] in near_keys]
    return {"bar_strict": bool(strict), "bar_near_tie_apart": bool(non_eq == non_total and max_ok and mean_ok
                                                                    and not nonfinite_keys),
            "non_near_tie_argmax": f"{non_eq}/{non_total}",
            "near_tie_argmax": f"{sum(r['argmax_equal'] for r in near)}/{len(near) + sum(1 for k in nonfinite_keys if k in near_keys)}",
            "near_tie_flips": [r["key"] for r in near if not r["argmax_equal"]]}


def self_test(per_row, summary, expect_rows, expect_parity):
    """Same file, same runtime (Mac stand-in) -> must reproduce litert_parity.py's rows and summary."""
    expect_rows = Path(expect_rows).resolve()
    expect_parity = Path(expect_parity).resolve() if expect_parity else None
    exp = {r["key"]: r for r in json.loads(expect_rows.read_text())["rows"]}
    dp, am, missing, compared = [], 0, [], 0
    for r in per_row:
        e = exp.get(r["key"])
        if e is None:
            missing.append(r["key"])
            continue
        if r["probs"] is None or e.get("probs") is None:
            continue
        compared += 1
        dp.append(float(np.abs(np.asarray(r["probs"], np.float64) - np.asarray(e["probs"], np.float64)).max()))
        am += int(r["argmax_key"] == e["argmax_key"])
    doc = {"expect_rows": rel(expect_rows), "questions_compared": compared, "missing": missing,
           "argmax_agree": am, "max_abs_probs_diff": max(dp) if dp else None}
    if expect_parity:
        es = json.loads(expect_parity.read_text())["summary"]
        keys = ["questions", "argmax_equal", "max_abs_dp", "mean_abs_dp_all_options", "p95_abs_dp_all_options",
                "h_sel_max_abs", "options"]
        doc["expect_parity"] = rel(expect_parity)
        doc["overall_mine"] = {k: summary["overall"].get(k) for k in keys}
        doc["overall_expected"] = {k: es["overall"].get(k) for k in keys}
        doc["overall_max_abs_diff"] = max(abs(float(summary["overall"][k]) - float(es["overall"][k])) for k in keys)
        doc["red_arm_mine"] = (summary.get("red_arm") or {}).get("max_abs_dp")
        doc["red_arm_expected"] = (es.get("red_arm") or {}).get("max_abs_dp")
        doc["near_tie_mine"] = {k: summary["near_tie"].get(k) for k in ("questions", "argmax_equal", "max_abs_dp")}
        doc["near_tie_expected"] = {k: es["near_tie"].get(k) for k in ("questions", "argmax_equal", "max_abs_dp")}
        doc["flips_mine"] = sorted(f["key"] for f in summary["flips"])
        doc["flips_expected"] = sorted(f["key"] for f in es["flips"])
    doc["pass"] = bool(not missing and compared == len(exp) and am == compared and dp and max(dp) <= 1e-6
                       and doc.get("overall_max_abs_diff", 0.0) <= 1e-6
                       and doc.get("flips_mine") == doc.get("flips_expected")
                       and (doc.get("red_arm_mine") is None or abs(doc["red_arm_mine"] - doc["red_arm_expected"]) <= 1e-6))
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--hsel", required=True)
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--rows", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--expect-rows", default="")
    ap.add_argument("--expect-parity", default="")
    a = ap.parse_args()
    report_path, hsel_path = Path(a.report).resolve(), Path(a.hsel).resolve()
    report = json.loads(report_path.read_text())
    assert report.get("status") == "DONE", f"report status {report.get('status')}: {report.get('error')}"
    assert int(report["L"]) == a.L, (report["L"], a.L)
    rows_path = Path(a.rows).resolve() if a.rows else K / "device" / report["rows_file"]
    rows_doc = json.loads(rows_path.read_text())
    assert int(rows_doc["L"]) == a.L
    tag = report_path.stem
    out = Path(a.out).resolve() if a.out else report_path.parent / f"device_parity_{tag}.json"
    oracle = load_oracle()
    qmap = {qkey(q): q for q in oracle["questions"]}
    ref = np.load(ORACLE_NPZ)
    head = Head()
    near_keys = {qkey(q) for q in oracle["questions"] if q["near_tie"]}
    by_key = {r["key"]: r for r in rows_doc["rows"]}
    rows, per_row, nonfinite, red_probs, hsel_store = [], [], [], None, {}
    for key, h_sel in slice_hsel(report, rows_doc, hsel_path):
        q = qmap[key]
        rr = by_key[key]
        assert rr["decide"] == q["decide_idx"] and rr["opts"] == q["opt_idx"] and rr["ids"] == q["row_ids"], key
        hsel_store[key] = h_sel
        if not np.isfinite(h_sel).all():
            nonfinite.append({"key": key, "source": q["source"], "row_len": q["row_len"],
                              "nonfinite_h_sel": int((~np.isfinite(h_sel)).sum())})
            per_row.append({"key": key, "probs": None, "argmax_key": None, "argmax_key_oracle": q["argmax_key"],
                            "finite_h_sel": False})
            continue
        z_pre, z_post, probs = head(h_sel)
        row = compare_question(q, probs, z_post, h_sel, ref[key])
        rows.append(row)
        per_row.append({"key": key, "source": q["source"], "keys": q["keys"], "probs": row["probs"],
                        "probs_oracle": q["probs"], "z_post": [float(x) for x in z_post],
                        "argmax_key": row["argmax_key"], "argmax_key_oracle": q["argmax_key"],
                        "max_abs_dp": row["max_abs_dp"], "h_sel_max_abs": row["h_sel_max_abs"],
                        "near_tie_oracle": row["near_tie_oracle"], "finite_h_sel": True})
        if q["id"] == "red_arm_000":
            red_probs = probs
    summary = summarize(rows, red_probs, oracle)
    n_q = len(rows) + len(nonfinite)
    nonfinite_keys = [r["key"] for r in nonfinite]
    summary["questions_total"] = n_q
    summary["nonfinite_h_sel_questions"] = len(nonfinite)
    summary["near_tie"]["flips"] = sum(1 for r in summary["near_tie"]["rows"] if not r["argmax_equal"])
    summary.update(bar_readings(rows, nonfinite_keys, near_keys, summary["overall"], n_q))
    if nonfinite:
        summary["bar_pass"] = False
        summary["bar_note"] = f"{len(nonfinite)} questions with non-finite h_sel (excluded from |dp| stats)"
    if red_probs is None and any(k.startswith("red_arm_000/") for k in by_key) and \
            any(r["key"].startswith("red_arm_000/") for r in report["rows"]):
        summary["red_arm"] = {"graph": "red_arm_000", "vs_oracle": "tv4_000", "max_abs_dp": None,
                              "note": "red_arm_000 output non-finite"}
    same_file = {}
    for label, rows_json, store in (
            ("mac_cpu8", K / f"results/litert_cpu_rows_L{a.L}_{VARIANT}.json",
             [K / f"device/mac_mock/hsel_mac_cpu_L{a.L}.npz", K / f"results/hsel_cpu_L{a.L}_{VARIANT}.npz"]),
            ("mac_gpu_f32", K / f"results/litert_gpu_f32_rows_L{a.L}_{VARIANT}.json",
             [K / f"device/mac_mock/hsel_mac_gpu_f32_L{a.L}.npz"])):
        npz = next((p for p in store if p.exists() and p.resolve() != hsel_path), store[0])
        if rows_json.exists():
            same_file[label] = gpu_vs_cpu(per_row, rows_json, hsel_store, npz)
            same_file[label].pop("top10_by_dp", None)
    rec = report["rows"]
    ms_total = [r["write_run_read_ms"] for r in rec]
    doc = {
        "step": "phone gate: the S26 debug app (or its Mac stand-in) vs the oracle",
        "report": rel(report_path), "report_sha256": sha256_file(report_path),
        "hsel": rel(hsel_path), "hsel_bytes": hsel_path.stat().st_size, "hsel_sha256": sha256_file(hsel_path),
        "rows_file": rel(rows_path), "rows_sha256": sha256_file(rows_path),
        "head": head.info,
        "run": {k: report.get(k) for k in ("graph", "graph_bytes", "accel", "precision", "threads", "L", "limit", "mode",
                                            "device", "android", "litert", "compile_ms", "status")},
        "timing_ms": {**report.get("summary", {}),
                      "median_all_rows_write_run_read": float(np.median(ms_total)) if ms_total else None},
        "questions_run": n_q, "nonfinite": nonfinite,
        "summary": summary,
        "same_file": same_file,
        "rows": [{k: v for k, v in r.items() if k not in ("dp", "probs")} for r in rows],
        "per_row": per_row,
    }
    if a.expect_rows:
        doc["selftest"] = self_test(per_row, summary, a.expect_rows, a.expect_parity)
    dump_json(out, doc)
    o = summary["overall"]
    print(json.dumps({"out": rel(out), "questions": n_q, "argmax": f"{o.get('argmax_equal')}/{n_q}",
                      "non_near_tie": summary["non_near_tie_argmax"], "near_tie_flips": summary["near_tie_flips"],
                      "max_abs_dp": o.get("max_abs_dp"), "mean_abs_dp": o.get("mean_abs_dp_all_options"),
                      "p95_abs_dp": o.get("p95_abs_dp_all_options"), "h_sel_max_abs": o.get("h_sel_max_abs"),
                      "red_arm": (summary.get("red_arm") or {}).get("max_abs_dp"), "nonfinite": len(nonfinite),
                      "bar_strict": summary["bar_strict"], "bar_near_tie_apart": summary["bar_near_tie_apart"],
                      "same_file": {k: {kk: v.get(kk) for kk in ("questions", "argmax_agree", "max_abs_dp", "h_sel_max_abs")}
                                    for k, v in same_file.items()},
                      "compile_ms": report.get("compile_ms"),
                      "warm_median_ms": report.get("summary", {}).get("warm_median_write_run_read_ms"),
                      "selftest": doc.get("selftest", {}).get("pass") if a.expect_rows else None}, indent=1))


if __name__ == "__main__":
    main()
