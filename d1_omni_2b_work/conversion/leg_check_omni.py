"""One DONE generic leg of a phone run: the pulled out_<tag>.f32 checked against its rows file (bytes, non-finite
values) and, when the Mac's Metal fp32 run of the same rows exists (mac_generic.py: out/mac/m32 over out/rows, and
out/mac_b8/m32 over out/rows_b8), every row's outputs against it by key (max |d|, the smallest cosine; for a media
decide row the Mac row was made from the Mac's own tower / audio outputs, so that difference holds theirs too). The
rows compared are the ones the host reads: a decide row's first T positions, an audio row's first `tokens` =
ceil(frames / 8) positions, a tower row whole, a batch row (one with `slots`, make_rows_b8_omni.py) on each real row
[b, :T] only. The verdict word: NONFINITE, KILLED (a kill line for the leg's own app pid), DIVERGED (an audio encoder
leg at GPU FP32 whose output differs from the Mac's Metal fp32 run of the same input rows by more than 1e-3 anywhere
the host reads), else PASS. The line itself is scored on the whole pass (run_replay.py + score.py). The leg's report,
outputs and logs come from the phones' test app, which is not published.

  $EXPORT conversion/leg_check_omni.py --report <R>/<tag>.json --out-file <R>/out_<tag>.f32 --rows <rows.json> \
      [--run-txt <R>/<tag>.run.txt --logcat <R>/<tag>.logcat_pkg_all_buffers.txt] --out <json>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(WORK, "host"))   # the host modules (replay.py is in this folder)
AUDIO_GRAPH = "d1-omni-2b_audio_"
DIVERGED_BAR = 1e-3   # the audio encoder at GPU FP32 against the Mac's Metal fp32, same input rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--out-file", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--mac-dir", action="append", default=None,
                    help="Mac run dirs as <run dir>=<rows dir> (default: out/mac/m32 and out/mac_b8/m32)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-txt", help="the leg's <tag>.run.txt (run.sh's result line names the app's pid)")
    ap.add_argument("--logcat", help="the leg's <tag>.logcat_pkg_all_buffers.txt")
    a = ap.parse_args()
    from replay import GenericRows

    rep = json.load(open(a.report))
    n = (rep.get("summary") or {}).get("count") or 0
    res = {"report": os.path.relpath(a.report, WORK), "rows_file": os.path.basename(a.rows), "rows_ran": n,
           "graph": rep.get("graph"), "accel": rep.get("accel"), "precision": rep.get("precision"),
           "signature": rep.get("signature"), "compile_ms": rep.get("compile_ms"),
           "memory_before_compile": rep.get("memory_before_compile"), "memory_after_compile": rep.get("memory_after_compile"),
           "memory_at_end": rep.get("memory_at_end"), "summary": rep.get("summary")}
    src = GenericRows(a.rows, a.out_file, rows_limit=n)
    names = [o["name"] for o in src.outputs]
    nonfinite = 0
    for pos in range(n):
        for nm in names:
            nonfinite += int((~np.isfinite(src.output(pos, nm))).sum())
    res["nonfinite_values"] = nonfinite
    # the Mac Metal fp32 run of rows with the same keys (static rows: the same input bytes)
    mac = {}
    pairs = [x.split("=", 1) for x in (a.mac_dir or [f"{WORK}/out/mac/m32={WORK}/out/rows",
                                                     f"{WORK}/out/mac_b8/m32={WORK}/out/rows_b8"])]
    for mac_dir, mac_rows in pairs:
        for rp in glob.glob(os.path.join(mac_dir, "*.json")):
            try:
                mrep = json.load(open(rp))
            except Exception:
                continue
            if not isinstance(mrep, dict) or mrep.get("status") != "DONE":
                continue
            tag = os.path.basename(rp)[:-5]
            mrows = os.path.join(mac_rows, mrep["rows_file"])
            if not os.path.exists(mrows):
                mrows = os.path.join(mac_dir, "gen", mrep["rows_file"])
            try:
                m = GenericRows(mrows, os.path.join(mac_dir, f"out_{tag}.f32"), check_inputs=False)
            except Exception:
                continue
            for pos, r in enumerate(m.rows):
                mac.setdefault(r["key"], (m, pos, r["sha256"]))
    per = []
    for pos in range(n):
        r = src.rows[pos]
        hit = mac.get(r["key"])
        if hit is None:
            continue
        m, mpos, msha = hit
        same_input = msha == r["sha256"]
        for nm in names:
            xo, yo = src.output(pos, nm).astype(np.float64), m.output(mpos, nm).astype(np.float64)
            if r.get("slots"):   # a batch row: the real rows' real positions only
                x = np.concatenate([xo[b, : s_["T"]] for b, s_ in enumerate(r["slots"])])
                y = np.concatenate([yo[b, : s_["T"]] for b, s_ in enumerate(r["slots"])])
            else:
                x = xo.reshape(-1, xo.shape[-1])
                y = yo.reshape(x.shape)
                T = r.get("T") or r.get("tokens") or x.shape[0]   # decide: T; audio: ceil(frames / 8); tower: whole
                x, y = x[:T], y[:T]
            nx, ny = np.linalg.norm(x, axis=-1), np.linalg.norm(y, axis=-1)
            ok = (nx > 0) & (ny > 0)
            cos = float(((x * y).sum(-1)[ok] / (nx[ok] * ny[ok])).min()) if ok.any() else None
            per.append({"key": r["key"], "output": nm, "same_input_as_mac": same_input,
                        "max_abs": float(np.abs(x - y).max()), "cos_min": cos, "ref_max_abs": float(np.abs(y).max())})
    res["vs_mac_metal_fp32"] = {"rows": len(per), "same_input_rows": sum(p["same_input_as_mac"] for p in per),
                                "max_abs_same_input": max([p["max_abs"] for p in per if p["same_input_as_mac"]],
                                                          default=None),
                                "cos_min_same_input": min([p["cos_min"] for p in per if p["same_input_as_mac"]
                                                           and p["cos_min"] is not None], default=None),
                                "per_row": per}
    # Was the leg's own app process killed (the Pixel 11 Pro's MemoryLimiter on the app cgroup's memory.high 8 GiB, then
    # lmkd's killinfo)? Its pid from the run's result line, the kill lines from the leg's logcat of all buffers.
    pid = None
    try:
        m = re.search(r"result=\S+ status=\S+ +pid=(\d+)", open(a.run_txt).read())
        pid = m.group(1) if m else None
    except (OSError, TypeError):
        pass
    limit_lines, kill_lines = [], []
    if pid and a.logcat:
        try:
            for ln in open(a.logcat, errors="replace"):
                if re.search(rf"MemoryLimiter: onLimitExceeded: pid={pid} ", ln):
                    limit_lines.append(ln.strip())
                elif (re.search(rf"killinfo: \[{pid},", ln) or re.search(rf"Killing {pid}:", ln)
                      or re.search(rf"\(pid {pid}\) has died", ln)):
                    kill_lines.append(ln.strip())
        except OSError:
            pass
    res["app_pid"] = pid
    res["memory_limit_lines"] = limit_lines
    res["kill_lines"] = kill_lines
    # PASS = finite, not killed and (an audio leg at GPU FP32) within the bar of the Mac's Metal fp32 on the same input
    # rows; chain.sh needs_pass reads this word, so NONFINITE / KILLED / DIVERGED stop the legs that need it
    mx = res["vs_mac_metal_fp32"]["max_abs_same_input"]
    gated = (str(rep.get("graph") or "").startswith(AUDIO_GRAPH) and rep.get("accel") == "gpu"
             and rep.get("precision") == "fp32")
    res["diverged_rule"] = {"applies": gated, "bar": DIVERGED_BAR, "max_abs_same_input": mx,
                            "mac_rows_same_input": res["vs_mac_metal_fp32"]["same_input_rows"]}
    if nonfinite:
        res["verdict"] = "NONFINITE"
    elif kill_lines:
        res["verdict"] = "KILLED"
    elif gated and (mx is None or mx > DIVERGED_BAR):
        res["verdict"] = "DIVERGED"   # no Mac row to compare with counts too: the stop rule cannot be read
    else:
        res["verdict"] = "PASS"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    v = res["vs_mac_metal_fp32"]
    print(res["verdict"], n, "rows", "vs mac:", v["rows"], "rows,", v["same_input_rows"], "same input, max_abs",
          v["max_abs_same_input"], "cos_min", v["cos_min_same_input"], "compile_ms", res["compile_ms"],
          "pid", pid, "memory.high lines", len(limit_lines), "kill lines", len(kill_lines))


if __name__ == "__main__":
    main()
