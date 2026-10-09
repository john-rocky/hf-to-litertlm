"""Round 2 acceptance 4-6: one exported row graph on the Mac through the CompiledModel API against the torch rows.

    $EXPORT scripts/d1_check.py --tflite exports/tiny_rowprefill_L64_fp32.tflite --accel cpu
    $EXPORT scripts/d1_check.py \
        --tflite exports/tiny_rowprefill_L64_fp32.tflite --accel gpu --f32          (Metal, float32; without --f32 = default)

Rows: cache/{tag}/check_rows_L{L}.npz from d1_graph_check.py (5 rows: n = L, L-1, L/2+1, L/4+3, 1 real tokens; ids,
valid, the torch D1Prefill hidden at every position, the provider's hidden at the real positions); {tag} = the file
name's prefix, L = the graph's signature. Per row: ids (or embeds = the tiny embedding rows of the ids, for a
D1PrefillEmbeds file) and valid are written by signature name, the graph runs, `hidden` [1, L, d] is read back; max
|diff| at the real positions vs torch D1Prefill and vs the provider; non-finite values (real / all positions).
--compare <npz of an earlier run>: max |diff| at the real positions vs that LiteRT run (a storage variant vs the fp32
file on the CPU).
GPU: Options(hardware_accelerators=GPU, gpu_options=GpuOptions(enforce_f32=--f32)) = Metal. Delegation evidence: the
runtime's VERBOSE lines are switched on (`runtime_log_verbose`), this process's fd 2 goes to
logs/{stem}_{accel}.runtime.log, and every line with `Replacing`, `Partitioned subgraph`, `delegat` or an error word is
copied verbatim into the json, with the numbers parsed (delegated / total nodes, partitions). is_fully_accelerated()
is recorded too.
Outputs (never overwritten): results/{stem}_{accel}_check.json, cache/{tag}/litert_{stem}_{accel}.npz (the real-position
hidden of every row). accel = cpu | gpu_f32 | gpu_default. Times are informational (the Mac is shared).
--reference <results/reference_<tag>.json of d1_reference.py> --table <readout_table.safetensors>
[--reference-hidden <results/reference_<tag>_hidden.npz>] [--near-tie-from <results/reference_summary.json>]
[--stop-on-bar] (round 6b, the real weights): after the check rows, every question of the reference whose row fits
the graph (ids from the reference, asserted equal to fixtures/rows.json; right-padded with the contract's pad id) and
every red arm of its `red_arms` (fixtures/red_arms.json) run through this file in the same compiled model; the
answer-slot hidden is read out with the host's own code (host/d1_litert.py `readout`: float32 logits of the table's
rows, group maximum, float64 softmax) and compared with the reference's unrounded `probs` (the provider's row path,
float32 CPU): argmax per question, max / mean / p95 |dp| over all options (mean = the sum over questions and options /
the number of options), the near-tie questions (reference top-2 gap <= 0.02) apart, the answer-slot hidden against the
reference's (max |dh|), non-finite values. Red arms: the file's probabilities of the arm row against the BASE record's
reference probabilities (same option keys) must move by more than 0.02 on some option (a graph that ignored the changed
input would not move). Bar readings (FACTS section 7, the four lanes): bar_strict = argmax on every question + max <=
0.02 + mean <= 0.002 + no non-finite row; bar_near_tie_apart = the same with the near-tie questions' argmax apart (the
bar set before the runs). With --compare, the answer-slot hidden and probabilities are also compared with the earlier run's
(its npz keeps `hsel/<id>/<qid>`). The npz gains `hsel/<id>/<qid>` per row. --stop-on-bar: exit 1 when
bar_near_tie_apart is false or a red arm does not move (the stop conditions for the float32 file on the CPU
and for v2 on Metal float32); without it a bar outside is recorded and the run still exits 0.
Round 6c:
--ref-subset smallest-head [--ref-head 20]: the questions whose smallest bucket (host BUCKETS 256..4096) is this L,
plus the first --ref-head questions of the reference (any length that fits), plus the red arms (the CPU row set of
the long buckets; default `all` = every question that fits L). The set is recorded (`row_set`).
--embed-table <embed_table.safetensors> (scripts/d1_tables.py --full-embed): an embeds graph (D1PrefillEmbeds) gets
the host's float32 rows of the bfloat16 table (d1_tables.EmbedTable) for every id of the row, pads included, for the
check rows and for --reference; without it an embeds graph reads --embed-weights (the tiny float32 table).
The per-question rows of --reference go to cache/{tag}/refrows_{stem}_{accel}.json (gitignored; `rows_file` +
sha256 in the json), so the results json keeps the summary, the red arms and the check rows only.
Round 10:
--compare <npz> of a run at another L (the new L128 bucket against the L256 ids float32 file's CPU run): the check
rows differ in n, so they are compared only where the npz holds a check row of the same shape (else the row records
`compare_row`: none); the questions are compared by `hsel/<id>/<qid>` as before.
--ref-part k/n: the k-th contiguous part of the --reference row set (sizes differ by at most one), the red arms in the
last part only; the outputs carry `_part{k}of{n}` after the accelerator; in parts 1..n-1 --stop-on-bar reads the bar
alone (no arms there). For Metal runs inside short measurement windows (the L4096 bucket: 2.2 s per row).
--merge-parts <the n part jsons>: one json under the whole run's names (results/{stem}_{acc}_check.json, the refrows
file and the npz), its summary computed by the same `parity_summary` over every part's rows and the last part's arms
(the parts must be 1..n of one file, accelerator, reference and row set, and together the row set in order); exit 1
when the merged stop_bar_pass is false.
"""
from __future__ import annotations

import argparse
import ctypes
import importlib.metadata
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

K = Path(__file__).resolve().parents[1]
ROWS = K / "fixtures/rows.json"
SIGNATURE = "serving_default"
NEAR_TIE = 0.02          # reference top-2 gap at or below which a question's argmax is reported apart
RED_ARM_MIN = 0.02       # a red arm must move some option by more than this against its base record
BAR_MAX_DP, BAR_MEAN_DP = 0.02, 0.002
LINE_KEYS = re.compile(r"(?i)replacing|partition|delegat|fail|error|unsupported|not supported|abort|fallback|reject")
REPLACING = re.compile(r"Replacing (\d+) out of (\d+) node\(s\) with delegate \(([^)]*)\) node, yielding (\d+) partitions")
PARTITIONED = re.compile(r"Partitioned subgraph<(\d+)>, selected (\d+) ops, from a total of (\d+) ops\. resulted in (\d+) partitions")


def runtime_log_verbose() -> dict:
    """Let the runtime print its VERBOSE lines (the TFLite `Replacing N out of M node(s) with delegate` line, LiteRT's
    `Partitioned subgraph` line). The CompiledModel wrapper runs the runtime linked into libpywrap_litert_common.dylib
    (libLiteRt.dylib is not loaded on this path, and the common library does not export its logger API), so the local
    symbols are used: their `nm` values plus the image's slide from dyld. Before any use, each address is checked
    against the file (code: the first 16 bytes; data: the 4-byte initial value, which must be INFO = 1); on any mismatch
    nothing is touched. Then both MinimalLogger severity words are set to VERBOSE (0) and the LiteRT default logger's
    minimum severity to VERBOSE (LiteRtSetMinLoggerSeverity(LiteRtGetDefaultLogger(), 0))."""
    import subprocess

    import ai_edge_litert.compiled_model  # noqa: F401  (loads the library)

    lib = (Path(ai_edge_litert.compiled_model.__file__).parent / "libpywrap_litert_common.dylib").resolve()
    want = {"get_logger": "_LiteRtGetDefaultLogger", "set_min": "_LiteRtSetMinLoggerSeverity",
            "get_min": "_LiteRtGetMinLoggerSeverity",
            "tflite_min": "__ZN6tflite16logging_internal13MinimalLogger21minimum_log_severity_E",
            "anon_min": "__ZN12_GLOBAL__N_113MinimalLogger21minimum_log_severity_E"}
    syms = {}
    for line in subprocess.run(["nm", str(lib)], capture_output=True, text=True, check=True).stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] in want.values():
            syms[parts[2]] = int(parts[0], 16)
    sysdl = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    sysdl._dyld_image_count.restype = ctypes.c_uint32
    sysdl._dyld_get_image_name.restype = ctypes.c_char_p
    sysdl._dyld_get_image_vmaddr_slide.restype = ctypes.c_long
    idx = [i for i in range(sysdl._dyld_image_count())
           if Path(sysdl._dyld_get_image_name(i).decode()).resolve() == lib]
    doc = {"library": str(lib), "symbols_found": sorted(k for k, v in want.items() if v in syms), "image_matches": len(idx)}
    if len(idx) != 1 or len(syms) != len(want):
        doc["applied"] = False
        return doc
    slide = sysdl._dyld_get_image_vmaddr_slide(idx[0])
    blob = lib.read_bytes()
    addr = {k: syms[v] + slide for k, v in want.items()}
    code_ok = {k: ctypes.string_at(addr[k], 16) == blob[syms[want[k]]: syms[want[k]] + 16]
               for k in ("get_logger", "set_min", "get_min")}
    data_init = {k: int.from_bytes(blob[syms[want[k]]: syms[want[k]] + 4], "little", signed=True)
                 for k in ("tflite_min", "anon_min")}
    data_now = {k: ctypes.c_int.from_address(addr[k]).value for k in ("tflite_min", "anon_min")}
    doc.update(code_bytes_match=code_ok, data_initial=data_init, data_before=data_now)
    if not all(code_ok.values()) or any(v != 1 for v in data_init.values()) or any(v != 1 for v in data_now.values()):
        doc["applied"] = False
        return doc
    for k in ("tflite_min", "anon_min"):
        ctypes.c_int.from_address(addr[k]).value = 0
    get_logger = ctypes.CFUNCTYPE(ctypes.c_void_p)(addr["get_logger"])
    set_min = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_int)(addr["set_min"])
    get_min = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int))(addr["get_min"])
    logger = get_logger()
    before, after = ctypes.c_int(0), ctypes.c_int(0)
    st_before = get_min(logger, ctypes.byref(before))
    st_set = set_min(logger, 0)
    st_after = get_min(logger, ctypes.byref(after))
    doc.update(applied=True, data_after={k: ctypes.c_int.from_address(addr[k]).value for k in ("tflite_min", "anon_min")},
               litert_logger={"status_get_before": st_before, "before_low_byte": before.value & 0xFF, "status_set": st_set,
                              "status_get_after": st_after, "after_low_byte": after.value & 0xFF})
    return doc


def _kpath(x: str) -> Path:
    return Path(x) if Path(x).is_absolute() else K / x


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 24):
            h.update(block)
    return h.hexdigest()


def row_selection(ref: dict, L: int, subset: str, head_n: int):
    """(questions with ids, the questions that fit L, the row set of --ref-subset) of a reference."""
    sys.path.insert(0, str(K / "host"))
    import d1_litert as H

    qs = [q for q in ref["questions"] if q.get("ids") is not None]
    fit_all = [q for q in qs if len(q["ids"]) <= L]
    if subset == "smallest-head":
        head = {f"{q['id']}/{q['qid']}" for q in qs[:head_n]}
        return qs, fit_all, [q for q in fit_all if H.pick_L(len(q["ids"])) == L or f"{q['id']}/{q['qid']}" in head]
    return qs, fit_all, fit_all


def parse_part(text: str):
    """--ref-part "k/n" -> (k, n) with 1 <= k <= n, or None."""
    if not text:
        return None
    k, n = (int(x) for x in text.split("/"))
    assert 1 <= k <= n, text
    return k, n


def part_bounds(count: int, part) -> tuple[int, int]:
    """[lo, hi) of part k of n over `count` questions (contiguous, sizes differ by at most one)."""
    if not part:
        return 0, count
    k, n = part
    return count * (k - 1) // n, count * k // n


def parity_summary(rows: list, arm_out: list, *, n_questions_reference: int, n_fit_all: int, fit_near_tie: dict,
                   n_arm_recs: int, n_arms_fit: int, row_set: dict, near_tie_check, ms_all: list, seconds_all: float,
                   compare_npz, part=None) -> dict:
    """The --reference summary (module docstring) of question rows and red-arm rows; round 10: shared by a whole run, a
    --ref-part run (its red arms run in the last part only; stop_bar_pass then reads the bar alone in the other parts)
    and --merge-parts (all parts' rows = the whole run's)."""
    qrows = [r for r in rows if r.get("probs") is not None]
    non = [r for r in qrows if not r["near_tie"]]
    near = [r for r in qrows if r["near_tie"]]
    nonfinite = [{"key": r["key"], "nonfinite_values": r.get("nonfinite_hsel_values")} for r in rows
                 if r.get("probs") is None]
    nf_keys = {x["key"] for x in nonfinite}
    options = sum(r["options"] for r in qrows)
    all_dp = [float(v) for r in qrows for v in np.abs(np.asarray(r["probs"]) - np.asarray(r["reference_probs"]))]
    mx = max((r["max_abs_dp"] for r in qrows), default=None)
    mean = sum(r["sum_abs_dp"] for r in qrows) / options if options else None
    max_ok, mean_ok = mx is not None and mx <= BAR_MAX_DP, mean is not None and mean <= BAR_MEAN_DP
    non_eq = sum(r["argmax_equal"] for r in non)
    nf_near = sum(1 for k in nf_keys if fit_near_tie[k])
    nf_non = len(nf_keys) - nf_near
    n_fit = len(fit_near_tie)
    hs = [r["h_sel_max_abs_diff"] for r in rows if "h_sel_max_abs_diff" in r]
    summary = {
        "questions_in_reference": n_questions_reference, "questions_fit_L": n_fit_all,
        "questions_longer_than_L": n_questions_reference - n_fit_all, "questions_run": n_fit, "row_set": row_set,
        "questions_finite": len(qrows), "argmax_equal": sum(r["argmax_equal"] for r in qrows),
        "non_near_tie_argmax": f"{non_eq}/{len(non) + nf_non}",
        "near_tie_argmax": f"{sum(r['argmax_equal'] for r in near)}/{len(near) + nf_near}",
        "near_tie_questions": len(near),
        "near_tie_flips": [{"key": r["key"], "top2_gap": r["top2_gap"], "reference_top2": r["reference_top2"],
                            "probs": r["probs"]} for r in near if not r["argmax_equal"]],
        "non_near_tie_flips": [{"key": r["key"], "top2_gap": r["top2_gap"], "reference_top2": r["reference_top2"],
                                "probs": r["probs"]} for r in non if not r["argmax_equal"]],
        "max_abs_dp": mx, "mean_abs_dp_all_options": mean,
        "p95_abs_dp_all_options": float(np.percentile(all_dp, 95)) if all_dp else None, "options": options,
        "max_abs_dp_key": max(qrows, key=lambda r: r["max_abs_dp"])["key"] if qrows else None,
        "rows_over_max_dp": [{"key": r["key"], "max_abs_dp": r["max_abs_dp"], "reference_top2": r["reference_top2"],
                              "keys": r["keys"], "probs": r["probs"], "reference_probs": r["reference_probs"],
                              "near_tie": r["near_tie"]}
                             for r in sorted(qrows, key=lambda r: -r["max_abs_dp"]) if r["max_abs_dp"] > BAR_MAX_DP],
        "h_sel_max_abs_diff": max(hs, default=None), "h_sel_rows_compared": len(hs),
        "nonfinite": nonfinite, "nonfinite_real_values_total": sum(r["nonfinite_real"] for r in rows),
        "bar": {"max_abs_dp": BAR_MAX_DP, "mean_abs_dp": BAR_MEAN_DP, "near_tie_gap": NEAR_TIE,
                "red_arm_min": RED_ARM_MIN},
        "bar_strict": bool(len(qrows) == n_fit and sum(r["argmax_equal"] for r in qrows) == n_fit and max_ok and mean_ok),
        "bar_near_tie_apart": bool(not nonfinite and non_eq == len(non) and max_ok and mean_ok),
        "red_arms": {"in_reference": n_arm_recs, "fit_L": n_arms_fit, "red": sum(1 for r in arm_out if r["red"]),
                     "all_red": bool(arm_out) and len(arm_out) == n_arm_recs and all(r["red"] for r in arm_out),
                     "min_max_abs_dp_vs_base": min((r.get("max_abs_dp_vs_base_reference", 0.0) for r in arm_out),
                                                   default=None)},
        "near_tie_list_check": near_tie_check,
        "run_ms": {"rows": len(ms_all), "median": float(np.median(ms_all)) if ms_all else None,
                   "p90": float(np.percentile(ms_all, 90)) if ms_all else None, "seconds_all": seconds_all,
                   "note": "contended Mac (other sessions); informational, never a card number"},
    }
    if compare_npz is not None:
        cr = [r for r in rows + arm_out if "h_sel_max_abs_vs_compare" in r]
        summary["vs_compare"] = {
            "npz": compare_npz, "rows_compared": len(cr),
            "rows_nonfinite_in_either": sum(1 for r in rows + arm_out if r.get("compare_nonfinite")),
            "h_sel_max_abs": max((r["h_sel_max_abs_vs_compare"] for r in cr), default=None),
            "max_abs_dp": max((r["max_abs_dp_vs_compare"] for r in cr if "max_abs_dp_vs_compare" in r), default=None),
            "argmax_agree": sum(1 for r in cr if r.get("argmax_equal_vs_compare")),
            "h_sel_bit_equal_rows": sum(1 for r in cr if r["h_sel_max_abs_vs_compare"] == 0.0)}
    arms_due = not part or part[0] == part[1]
    summary["part"] = {"k": part[0], "n": part[1], "red_arms_in_this_part": arms_due} if part else None
    summary["stop_bar_pass"] = bool(summary["bar_near_tie_apart"] and (summary["red_arms"]["all_red"] or not arms_due))
    summary["line"] = {k: summary[k] for k in ("questions_fit_L", "questions_run", "argmax_equal", "non_near_tie_argmax",
                                               "near_tie_argmax", "max_abs_dp", "mean_abs_dp_all_options",
                                               "p95_abs_dp_all_options", "h_sel_max_abs_diff", "bar_strict",
                                               "bar_near_tie_apart")}
    summary["line"].update(red_arms=[round(r.get("max_abs_dp_vs_base_reference", 0.0), 6) for r in arm_out],
                           nonfinite=len(nonfinite), vs_compare=summary.get("vs_compare"))
    return summary


def reference_parity(run_row, L: int, d: int, a, cmp, store: dict) -> dict:
    """--reference (module docstring): the reference's question rows that fit L and its red arms through this file."""
    sys.path.insert(0, str(K / "host"))
    import d1_litert as H

    ref_path, table_path = _kpath(a.reference), _kpath(a.table)
    ref = json.loads(ref_path.read_text())
    table = H.ReadoutTable.from_file(table_path)
    assert table.rows.shape[1] == d, (table.rows.shape, d)
    hid = np.load(_kpath(a.reference_hidden)) if a.reference_hidden else None
    rows_json = {f"{r['id']}/{r['qid']}": r for r in json.loads(ROWS.read_text())["rows"]}
    qs, fit_all, fit_full = row_selection(ref, L, a.ref_subset, a.ref_head)
    part = parse_part(a.ref_part)
    lo, hi = part_bounds(len(fit_full), part)
    fit = fit_full[lo:hi]
    row_set = {"kind": a.ref_subset, "head": a.ref_head if a.ref_subset == "smallest-head" else None,
               "questions_fit_L": len(fit_all), "questions_run": len(fit),
               "smallest_bucket_L": sum(1 for q in fit_all if H.pick_L(len(q["ids"])) == L),
               "keys": [f"{q['id']}/{q['qid']}" for q in fit] if len(fit) < len(fit_all) else "every question that fits L"}
    if part:
        row_set["part"] = {"k": part[0], "n": part[1], "slice": [lo, hi], "questions_in_set": len(fit_full),
                           "red_arms": part[0] == part[1]}
    arm_recs = [x for x in ((ref.get("red_arms") or {}).get("records") or [])]
    arms = [x for x in arm_recs if len(x["ids"]) <= L] if not part or part[0] == part[1] else []
    t_all = time.perf_counter()

    def one(ids):
        n = len(ids)
        ids_l, valid_l = H.pad_row(ids, L, H.PAD_ID)
        t = time.perf_counter()
        h_all = run_row(ids_l, valid_l)
        ms = (time.perf_counter() - t) * 1000
        return h_all[n - 1].copy(), int((~np.isfinite(h_all[:n])).sum()), ms

    def vs_compare(key, h, groups, p):
        if cmp is None or f"hsel/{key}" not in cmp.files:
            return {}
        hc = cmp[f"hsel/{key}"]
        if not (np.isfinite(h).all() and np.isfinite(hc).all()):
            return {"compare_nonfinite": True}
        out = {"h_sel_max_abs_vs_compare": float(np.abs(h.astype(np.float64) - hc.astype(np.float64)).max())}
        if p is not None:
            pc = H.readout(hc, table, groups)
            out.update(max_abs_dp_vs_compare=float(np.abs(np.asarray(p) - np.asarray(pc)).max()),
                       argmax_equal_vs_compare=int(np.argmax(p)) == int(np.argmax(pc)))
        return out

    rows, ms_all = [], []
    for q in fit:
        key = f"{q['id']}/{q['qid']}"
        rj = rows_json.get(key)
        assert rj is not None and rj["ids"] == q["ids"] and rj["readout_ids"] == q["readout_ids"], key
        h, nf_real, ms = one(q["ids"])
        ms_all.append(ms)
        store[f"hsel/{key}"] = h
        rec = {"key": key, "type": q["type"], "path": q.get("path"), "row_len": len(q["ids"]), "keys": q["keys"],
               "nonfinite_real": nf_real, "ms": round(ms, 2)}
        if hid is not None and key in hid.files and np.isfinite(h).all():
            rec["h_sel_max_abs_diff"] = float(np.abs(h.astype(np.float64) - hid[key].astype(np.float64)).max())
        if not np.isfinite(h).all():
            rec["nonfinite_hsel_values"] = int((~np.isfinite(h)).sum())
            rec["probs"] = None
            rec.update(vs_compare(key, h, q["readout_ids"], None))
            rows.append(rec)
            continue
        p = H.readout(h, table, q["readout_ids"])
        pr = q["probs"]
        assert len(pr) == len(p), (key, pr, p)
        dp = np.abs(np.asarray(p, dtype=np.float64) - np.asarray(pr, dtype=np.float64))
        order = sorted(range(len(pr)), key=lambda i: -pr[i])
        rec.update(probs=p, reference_probs=pr, argmax=int(np.argmax(p)), reference_argmax=int(order[0]),
                   argmax_equal=int(np.argmax(p)) == order[0],
                   reference_top2=[[q["keys"][i], pr[i]] for i in order[:2]], top2_gap=q["top2_gap"],
                   near_tie=bool(q["near_tie"]), max_abs_dp=float(dp.max()), sum_abs_dp=float(dp.sum()),
                   options=len(pr), max_abs_h=float(np.abs(h).max()))
        rec.update(vs_compare(key, h, q["readout_ids"], p))
        rows.append(rec)
    arm_out = []
    for x in arms:
        key = f"{x['id']}/{x['qid']}"
        h, nf_real, ms = one(x["ids"])
        store[f"hsel/{key}"] = h
        rec = {"key": key, "base": f"{x['base_id']}/{x['qid']}", "kind": x.get("kind"), "row_len": len(x["ids"]),
               "nonfinite_real": nf_real, "ms": round(ms, 2),
               "reference_max_abs_dp_vs_base": x["max_abs_dp"], "reference_argmax_base": x["argmax_base"],
               "reference_argmax_arm": x["argmax_key"]}
        if hid is not None and key in hid.files and np.isfinite(h).all():
            rec["h_sel_max_abs_diff"] = float(np.abs(h.astype(np.float64) - hid[key].astype(np.float64)).max())
        if not np.isfinite(h).all():
            rec.update(probs=None, red=False)
            rec.update(vs_compare(key, h, x["readout_ids"], None))
            arm_out.append(rec)
            continue
        p = H.readout(h, table, x["readout_ids"])
        pa, pb = dict(zip(x["keys"], p)), dict(zip(x["keys_base"], x["probs_base"]))
        assert set(pa) == set(pb), key
        dp_base = max(abs(pa[k] - pb[k]) for k in pa)
        rec.update(keys=x["keys"], probs=p, probs_base_reference=x["probs_base"], keys_base=x["keys_base"],
                   argmax_key=x["keys"][int(np.argmax(p))], max_abs_dp_vs_base_reference=dp_base,
                   red=dp_base > RED_ARM_MIN,
                   max_abs_dp_vs_arm_reference=float(np.abs(np.asarray(p) - np.asarray(x["probs"])).max()))
        rec.update(vs_compare(key, h, x["readout_ids"], p))
        arm_out.append(rec)
    near_tie_check = None
    if a.near_tie_from:
        summ = json.loads(_kpath(a.near_tie_from).read_text())
        listed = set(summ["near_tie"]["ids"])
        flagged = {f"{q['id']}/{q['qid']}" for q in qs if q["near_tie"]}
        near_tie_check = {"file": a.near_tie_from, "summary_count": len(listed), "reference_flags": len(flagged),
                          "equal": listed == flagged,
                          "in_this_L": sorted(k for k in listed if k in {f"{q['id']}/{q['qid']}" for q in fit})}
    summary = parity_summary(rows, arm_out, n_questions_reference=len(qs), n_fit_all=len(fit_all),
                             fit_near_tie={f"{q['id']}/{q['qid']}": bool(q["near_tie"]) for q in fit},
                             n_arm_recs=len(arm_recs), n_arms_fit=len(arms), row_set=row_set,
                             near_tie_check=near_tie_check, ms_all=ms_all,
                             seconds_all=round(time.perf_counter() - t_all, 1),
                             compare_npz=a.compare if cmp is not None else None, part=part)
    return {"reference": str(ref_path.relative_to(K) if ref_path.is_relative_to(K) else ref_path),
            "reference_sha256": _sha256(ref_path), "table": str(table_path.relative_to(K)),
            "table_sha256": _sha256(table_path), "table_shape": list(table.rows.shape),
            "reference_hidden": a.reference_hidden or None, "pad_id": H.PAD_ID,
            "readout": "host/d1_litert.py readout (float32 logits of the table rows, group max, float64 softmax)",
            "summary": summary, "rows": rows, "red_arms": arm_out}


def merge_parts(paths: list) -> int:
    """--merge-parts (round 10): the n --ref-part runs of one file and accelerator -> results/{stem}_{acc}_check.json,
    cache/{tag}/refrows_{stem}_{acc}.json and cache/{tag}/litert_{stem}_{acc}.npz, the names of a whole run (so the
    chain's D1_RESUME and the contract see the gate). The parts must be 1..n of one n, of one file (bytes), accelerator,
    reference, table and row set, and together the row set in order (re-derived from the reference); the summary is
    parity_summary over every part's question rows and the last part's red arms; the check rows, options and logger
    are part 1's (each part runs the same check rows: their maxima over the parts are recorded too); every part's
    compile time, delegation lines and runtime log are listed under `parts`."""
    sys.path.insert(0, str(K / "host"))
    import d1_litert as H  # noqa: F401  (pick_L of row_selection)

    docs = []
    for p in paths:
        q = _kpath(p)
        docs.append((q, json.loads(q.read_text())))
    parts = sorted(((d["reference_parity"]["summary"]["part"]["k"], d["reference_parity"]["summary"]["part"]["n"], q, d)
                    for q, d in docs), key=lambda x: x[0])
    n = parts[0][1]
    assert [k for k, *_ in parts] == list(range(1, n + 1)) and all(m == n for _, m, *_ in parts), [x[:2] for x in parts]
    first, last = parts[0][3], parts[-1][3]
    for k, _, q, d in parts:
        assert d["status"] == "OK", (q.name, d.get("error"))
        for key in ("tflite", "tflite_bytes"):
            assert d[key] == first[key], (q.name, key)
        assert d["accel"].rsplit("_part", 1)[0] == first["accel"].rsplit("_part", 1)[0], q.name
        for key in ("reference", "reference_sha256", "table", "table_sha256"):
            assert d["reference_parity"][key] == first["reference_parity"][key], (q.name, key)
    acc = first["accel"].rsplit("_part", 1)[0]
    path = K / first["tflite"]
    stem, tag = path.stem, path.stem.split("_")[0]
    out = K / f"results/{stem}_{acc}_check.json"
    npz_out = K / f"cache/{tag}/litert_{stem}_{acc}.npz"
    refrows_out = K / f"cache/{tag}/refrows_{stem}_{acc}.json"
    for p in (out, npz_out, refrows_out):
        assert not p.exists(), f"refusing to overwrite {p}"
    rs0 = first["reference_parity"]["summary"]["row_set"]
    ref = json.loads(_kpath(first["reference_parity"]["reference"]).read_text())
    qs, fit_all, fit_full = row_selection(ref, int(first["L"]), rs0["kind"], rs0["head"] or 20)
    rows, store = [], {}
    for k, _, q, d in parts:
        rp = d["reference_parity"]
        rf = K / rp["rows_file"]
        assert _sha256(rf) == rp["rows_file_sha256"], rf.name
        rows += json.loads(rf.read_text())["rows"]
        z = np.load(K / d["litert_npz"])
        store.update({key: z[key] for key in z.files if key.startswith("hsel/") or k == 1})
    want = [f"{q['id']}/{q['qid']}" for q in fit_full]
    assert [r["key"] for r in rows] == want, (len(rows), len(want))
    arm_out = last["reference_parity"]["red_arms"]
    arm_recs = (ref.get("red_arms") or {}).get("records") or []
    row_set = {k: v for k, v in rs0.items() if k != "part"}
    row_set.update(questions_run=len(fit_full), merged_parts=n,
                   keys=want if len(fit_full) < len(fit_all) else "every question that fits L")
    s_last = last["reference_parity"]["summary"]
    summary = parity_summary(rows, arm_out, n_questions_reference=len(qs), n_fit_all=len(fit_all),
                             fit_near_tie={f"{q['id']}/{q['qid']}": bool(q["near_tie"]) for q in fit_full},
                             n_arm_recs=len(arm_recs), n_arms_fit=s_last["red_arms"]["fit_L"], row_set=row_set,
                             near_tie_check=first["reference_parity"]["summary"]["near_tie_list_check"],
                             ms_all=[r["ms"] for r in rows + arm_out],
                             seconds_all=round(sum(d["reference_parity"]["summary"]["run_ms"]["seconds_all"]
                                                   for *_, d in parts), 1),
                             compare_npz=first.get("compare_npz"), part=None)
    import hashlib

    refrows_text = json.dumps({"tflite": first["tflite"], "accel": acc, "rows": rows,
                               "merged_from": [str(q.relative_to(K)) for *_, q, _ in parts]}) + "\n"
    doc = {k: v for k, v in first.items() if k not in ("reference_parity", "litert_npz", "runtime_log", "accel")}
    doc.update(accel=acc, started_at=first["started_at"], finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               runtime_log=[d["runtime_log"] for *_, d in parts], litert_npz=str(npz_out.relative_to(K)),
               is_fully_accelerated=all(d.get("is_fully_accelerated") is True for *_, d in parts),
               max_abs_real_vs_torch_d1prefill_parts=max(d["max_abs_real_vs_torch_d1prefill"] for *_, d in parts),
               max_abs_real_vs_provider_parts=max(d["max_abs_real_vs_provider"] for *_, d in parts),
               parts=[{"file": str(q.relative_to(K)), "sha256": _sha256(q), "k": k, "started_at": d["started_at"],
                       "compile_seconds": d.get("compile_seconds"), "is_fully_accelerated": d.get("is_fully_accelerated"),
                       "questions_run": d["reference_parity"]["summary"]["questions_run"],
                       "red_arms_run": len(d["reference_parity"]["red_arms"]),
                       "seconds_rows": d["reference_parity"]["summary"]["run_ms"]["seconds_all"],
                       "delegation": d["delegation"]} for k, _, q, d in parts])
    rp = {k: v for k, v in first["reference_parity"].items() if k not in ("summary", "red_arms", "rows_file",
                                                                         "rows_file_sha256")}
    rp.update(summary=summary, red_arms=arm_out, rows_file=str(refrows_out.relative_to(K)),
              rows_file_sha256=hashlib.sha256(refrows_text.encode()).hexdigest())
    doc["reference_parity"] = rp
    doc["what"] = first["what"] + f" (round 10: {n} --ref-part runs merged by --merge-parts)"
    out_text = json.dumps(doc, indent=1) + "\n"     # everything built before the first write
    refrows_out.write_text(refrows_text)
    np.savez(npz_out, **store)
    out.write_text(out_text)
    print(json.dumps({"merged": str(out.relative_to(K)), "parts": n, "rows": len(rows), "red_arms": len(arm_out),
                      "line": summary["line"], "stop_bar_pass": summary["stop_bar_pass"]}, indent=1))
    return 0 if summary["stop_bar_pass"] else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", required=True, help="K-relative or absolute")
    ap.add_argument("--accel", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--f32", action="store_true", help="GPU: GpuOptions(enforce_f32=True)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--rows", default="", help="check rows npz (default cache/{tag}/check_rows_L{L}.npz)")
    ap.add_argument("--compare", default="", help="npz of an earlier d1_check run on the same rows")
    ap.add_argument("--reference", default="", help="results/reference_<tag>.json: every question row that fits L and "
                                                     "the red arms against the reference (round 6b)")
    ap.add_argument("--reference-hidden", default="", help="results/reference_<tag>_hidden.npz (answer-slot hidden)")
    ap.add_argument("--table", default="", help="readout_table.safetensors for --reference")
    ap.add_argument("--near-tie-from", default="", help="results/reference_summary.json: its near-tie list is checked "
                                                        "against the reference's per-question flags")
    ap.add_argument("--stop-on-bar", action="store_true", help="--reference: exit 1 when the bar (near ties apart) or a "
                                                               "red arm fails")
    ap.add_argument("--embed-weights", default="cache/tiny/tiny_lfm2_seed0.safetensors",
                    help="safetensors with embed_tokens.weight (float32) for a D1PrefillEmbeds file")
    ap.add_argument("--embed-table", default="", help="embed_table.safetensors (bfloat16, d1_tables.py --full-embed): "
                                                      "the host's rows for a D1PrefillEmbeds file (round 6c)")
    ap.add_argument("--ref-subset", choices=["all", "smallest-head"], default="all",
                    help="--reference rows: all that fit L, or those whose smallest bucket is L + the first --ref-head")
    ap.add_argument("--ref-head", type=int, default=20)
    ap.add_argument("--ref-part", default="", help="k/n: the k-th contiguous part of the --reference row set, the red "
                                                    "arms in the last part only (round 10: Metal runs of <= 5 min)")
    ap.add_argument("--merge-parts", nargs="+", default=[], help="the n part jsons of one file and accelerator -> the "
                                                                 "whole run's json (round 10; module docstring)")
    a = ap.parse_args()
    if a.merge_parts:
        return merge_parts(a.merge_parts)
    path = Path(a.tflite) if Path(a.tflite).is_absolute() else K / a.tflite
    stem, tag = path.stem, path.stem.split("_")[0]
    acc = "cpu" if a.accel == "cpu" else ("gpu_f32" if a.f32 else "gpu_default")
    if a.ref_part:
        k_, n_ = parse_part(a.ref_part)
        assert a.reference, "--ref-part needs --reference"
        acc += f"_part{k_}of{n_}"
    out = K / f"results/{stem}_{acc}_check.json"
    npz_out = K / f"cache/{tag}/litert_{stem}_{acc}.npz"
    refrows_out = K / f"cache/{tag}/refrows_{stem}_{acc}.json"
    log = K / f"logs/{stem}_{acc}.runtime.log"
    for p in (out, npz_out, log, refrows_out):
        assert not p.exists(), f"refusing to overwrite {p}"
    log_f = open(log, "w")
    saved_fd2 = os.dup(2)
    os.dup2(log_f.fileno(), 2)
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    doc = {"what": "round 2: exported row graph through CompiledModel vs the torch rows", "tflite": str(path.relative_to(K)),
           "tflite_bytes": path.stat().st_size, "accel": acc, "started_at": started,
           "ai_edge_litert": importlib.metadata.version("ai-edge-litert"), "runtime_log": str(log.relative_to(K))}
    try:
        doc["logger"] = runtime_log_verbose()
        from ai_edge_litert.compiled_model import CompiledModel, CpuOptions, GpuOptions, HardwareAccelerator, Options

        if a.accel == "gpu":
            gopt = GpuOptions(enforce_f32=a.f32)
            opts = Options(hardware_accelerators=HardwareAccelerator.GPU, gpu_options=gopt)
            doc["options"] = {"accelerator": "GPU (Metal)", "gpu_options": gopt._as_flat_kwargs()}
        else:
            opts = Options(hardware_accelerators=HardwareAccelerator.CPU, cpu_options=CpuOptions(num_threads=a.threads))
            doc["options"] = {"accelerator": "CPU", "threads": a.threads}
        t0 = time.perf_counter()
        model = CompiledModel.from_file(str(path), options=opts)
        doc["compile_seconds"] = round(time.perf_counter() - t0, 3)
        try:
            doc["is_fully_accelerated"] = bool(model.is_fully_accelerated())
        except Exception as e:   # informational
            doc["is_fully_accelerated"] = f"unavailable: {type(e).__name__}: {e}"
        sigs = model.get_signature_list()
        assert list(sigs) == [SIGNATURE], sigs
        in_det, out_det = model.get_input_tensor_details(SIGNATURE), model.get_output_tensor_details(SIGNATURE)
        doc["inputs"] = {n: {k: str(v) for k, v in d.items()} for n, d in in_det.items()}
        doc["outputs"] = {n: {k: str(v) for k, v in d.items()} for n, d in out_det.items()}
        L, d = int(list(out_det["hidden"]["shape"])[1]), int(list(out_det["hidden"]["shape"])[2])
        embeds_graph = "embeds" in in_det
        rows_path = Path(a.rows) if a.rows else K / f"cache/{tag}/check_rows_L{L}.npz"
        rows = np.load(rows_path if rows_path.is_absolute() else K / rows_path)
        assert int(rows["L"]) == L, (int(rows["L"]), L)
        doc["rows_npz"] = str((rows_path if rows_path.is_absolute() else K / rows_path).relative_to(K))
        table = None
        if embeds_graph and a.embed_table:
            from d1_tables import EmbedTable

            et = EmbedTable(_kpath(a.embed_table))
            assert et.hidden == d, (et.hidden, d)
            doc["embed_table"] = {"file": a.embed_table, "sha256": _sha256(_kpath(a.embed_table)),
                                  "shape": [et.vocab, et.hidden], "dtype": "bfloat16 -> float32 on the host"}

            def embed_rows(ids_row):
                return np.ascontiguousarray(et.rows(ids_row)[None], dtype=np.float32)
        elif embeds_graph:
            from safetensors.numpy import load_file

            table = load_file(str(K / a.embed_weights))["embed_tokens.weight"].astype(np.float32)

            def embed_rows(ids_row):
                return np.ascontiguousarray(table[ids_row][None], dtype=np.float32)
        ins = {n: model.create_input_buffer_by_name(SIGNATURE, n) for n in in_det}
        outs = {"hidden": model.create_output_buffer_by_name(SIGNATURE, "hidden")}
        cmp = np.load(K / a.compare) if a.compare else None
        res, store, run_ms = [], {}, []
        k = 0
        while f"ids_{k}" in rows.files:
            ids, valid, n = rows[f"ids_{k}"].astype(np.int32), rows[f"valid_{k}"].astype(np.float32), int(rows[f"n_{k}"])
            if embeds_graph:
                ins["embeds"].write(embed_rows(ids[0]))
            else:
                ins["ids"].write(ids)
            ins["valid"].write(valid)
            t = time.perf_counter()
            model.run_by_name(SIGNATURE, ins, outs)
            run_ms.append((time.perf_counter() - t) * 1000)
            h = np.asarray(outs["hidden"].read(L * d, np.float32), dtype=np.float32).reshape(L, d)
            real = h[:n]
            row = {"row": k, "n": n, "pads": L - n,
                   "max_abs_real_vs_torch_d1prefill": float(np.abs(real.astype(np.float64) - rows[f"hidden_{k}"][:n]).max()),
                   "max_abs_real_vs_provider": float(np.abs(real.astype(np.float64) - rows[f"provider_{k}"]).max()),
                   "max_abs_hidden_real": float(np.abs(real).max()),
                   "nonfinite_real": int((~np.isfinite(real)).sum()), "nonfinite_all": int((~np.isfinite(h)).sum())}
            if cmp is not None:   # round 10: a compare run of another L has other check rows (n differs)
                if f"real_{k}" in cmp.files and cmp[f"real_{k}"].shape == real.shape:
                    row["max_abs_real_vs_compare"] = float(np.abs(real.astype(np.float64) - cmp[f"real_{k}"]).max())
                else:
                    row["compare_row"] = f"none: {a.compare} holds no check row {k} of n = {n}"
            res.append(row)
            store[f"real_{k}"] = real
            k += 1
        if a.reference:
            assert not embeds_graph or a.embed_table, "--reference on an embeds graph needs --embed-table"

            def run_row(ids_l, valid_l):
                if embeds_graph:
                    ins["embeds"].write(embed_rows(ids_l[0]))
                else:
                    ins["ids"].write(ids_l)
                ins["valid"].write(valid_l)
                model.run_by_name(SIGNATURE, ins, outs)
                return np.asarray(outs["hidden"].read(L * d, np.float32), dtype=np.float32).reshape(L, d)

            rp = reference_parity(run_row, L, d, a, cmp, store)
            # the per-question rows stay out of git (cache/ is ignored); the json keeps the summary and the red arms
            refrows_out.write_text(json.dumps({"tflite": str(path.relative_to(K)), "accel": acc,
                                               "rows": rp.pop("rows")}) + "\n")
            rp["rows_file"] = str(refrows_out.relative_to(K))
            rp["rows_file_sha256"] = _sha256(refrows_out)
            doc["reference_parity"] = rp
        for b in list(ins.values()) + list(outs.values()):
            try:
                b.destroy()
            except Exception:
                pass
        model.close()
        np.savez(npz_out, **store)
        doc.update(L=L, d=d, graph_inputs=sorted(in_det), rows=res, run_ms=[round(x, 3) for x in run_ms],
                   max_abs_real_vs_torch_d1prefill=max(r["max_abs_real_vs_torch_d1prefill"] for r in res),
                   max_abs_real_vs_provider=max(r["max_abs_real_vs_provider"] for r in res),
                   nonfinite_real_total=sum(r["nonfinite_real"] for r in res),
                   nonfinite_all_total=sum(r["nonfinite_all"] for r in res), litert_npz=str(npz_out.relative_to(K)),
                   timing_note="contended Mac (other sessions); informational, never a card number")
        if cmp is not None:
            doc["compare_npz"] = a.compare
            doc["max_abs_real_vs_compare"] = max((r["max_abs_real_vs_compare"] for r in res
                                                  if "max_abs_real_vs_compare" in r), default=None)
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
    doc["delegation"] = {
        "replacing": [{"delegated": int(m[0]), "total": int(m[1]), "delegate": m[2], "partitions": int(m[3])}
                      for m in REPLACING.findall("\n".join(lines))],
        "partitioned": [{"subgraph": int(m[0]), "selected": int(m[1]), "total": int(m[2]), "partitions": int(m[3])}
                        for m in PARTITIONED.findall("\n".join(lines))]}
    out.write_text(json.dumps(doc, indent=1) + "\n")
    brief = {k: doc.get(k) for k in ("status", "error", "accel", "L", "compile_seconds", "is_fully_accelerated",
                                     "max_abs_real_vs_torch_d1prefill", "max_abs_real_vs_provider",
                                     "max_abs_real_vs_compare", "nonfinite_all_total", "delegation")}
    print(json.dumps(brief, indent=1))
    print("\n".join(doc["runtime_log_key_lines"][:20]))
    rp = doc.get("reference_parity")
    if rp is not None:
        print(json.dumps({"reference_parity": rp["summary"]["line"]}, indent=1))
        if a.stop_on_bar and not rp["summary"]["stop_bar_pass"]:
            print("STOP: bar (near ties apart) or a red arm failed (--stop-on-bar)")
            return 1
    return 0 if doc["status"] == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
