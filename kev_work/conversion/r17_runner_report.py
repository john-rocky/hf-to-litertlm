"""Turns one run of the NPU runner (android/measure/kev_npu_runner.cc: its stdout and the pulled hsel.f32) into the gate
app's report shape, so that r17_device_compare.py scores it like a gate app run (same readout, same statistics).

    python r17_runner_report.py --stdout device/r17/<tag>.stdout.txt --hsel-in device/r17/<tag>.hsel.f32
        --manifest npu/fixtures_L128/manifest.json --tag <tag> --accel npu+cpu --precision htp-fp16 --graph <file>
        [--logcat device/r17/<tag>.logcat_all.txt]                                           (one command)

Writes <dir>/<tag>.json (dir = $R17_REL, default device/r17; report: status, graph, accel, precision, L, compile_ms,
rows [{key, n, k, finite, nonfinite_real, nonfinite_all, write_run_read_ms, run_ms}], summary, latency, delegate lines
and their numbers) and <dir>/hsel_<tag>.f32 (a copy of the pulled stream; 1 + k rows of 1024 floats per fixture, fixture
order). The runner's parity pass runs each fixture once after one warm-up call on fixture 000; its ms are the parity-pass
calls (one call per different row); the latency block is the runner's second phase (W warm-up calls, then R timed calls
on one fixture)."""
import argparse
import json
import os
import re
import shutil
import statistics
from pathlib import Path

from r2_common import K, dump_json, sha256_file

PAR = re.compile(r"^parity model\[(\d+)\] fixture (\d{3}) ms=([\d.]+) run_ms=([\d.]+) nonfinite=(\d+) n=(-?\d+) k=(-?\d+) "
                 r"nonfinite_real=(-?\d+) nonfinite_sel=(-?\d+)")
ROUND = re.compile(r"^round (\d+) model\[(\d+)\] write\+run\+readback_ms=([\d.]+) run_ms=([\d.]+)")
WARM = re.compile(r"^warmup (\d+) model\[(\d+)\] ([\d.]+) ms run_ms=([\d.]+)")
MODEL = re.compile(r"^model\[(\d+)\]=(\S+) inputs=(\d+) outputs=(\d+) out0_elems=(\d+) compile_ms=([\d.]+) "
                   r"fully_accelerated=(\w+)")
WALL = re.compile(r"(compile_start|compile_end|latency_start|done) .*?wall_ms=(\d+)")
DELEGATE = re.compile(r"(NPU accelerator|npu_registry|JIT compilation caching|HtpPerformanceMode|Partitioned subgraph|"
                      r"compiler plugins were applied|Replacing \d+ out of \d+ node|DispatchDelegate|LITERT_CL|"
                      r"cached model|Flatbuffer model initialized|Qualcomm|qnn|QNN|unsupported|Unsupported|not supported|"
                      r"fallback|Fallback|ERROR|Error)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stdout", required=True)
    ap.add_argument("--hsel-in", default="")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--accel", required=True)
    ap.add_argument("--precision", required=True)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--logcat", default="")
    a = ap.parse_args()
    out = K / os.environ.get("R17_REL", "device/r17") / f"{a.tag}.json"
    hsel_out = K / os.environ.get("R17_REL", "device/r17") / f"hsel_{a.tag}.f32"
    assert not out.exists() and not hsel_out.exists(), "refusing to overwrite"
    man = json.loads((K / a.manifest).read_text())
    by_id = {m["id3"]: m for m in man["rows"]}
    text = (K / a.stdout).read_text(errors="replace").splitlines()
    rows, rounds, warm, models, walls, fatal, errors = [], [], [], [], {}, [], []
    done = False
    for line in text:
        if m := PAR.match(line):
            if int(m.group(1)) != 0:
                continue
            ent = by_id[m.group(2)]
            n, k = int(m.group(6)), int(m.group(7))
            assert n == ent["row_len"] and k == ent["k"], (line, ent)
            rows.append({"key": ent["key"], "n": n, "k": k, "finite": int(m.group(9)) == 0,
                         "nonfinite_real": int(m.group(8)), "nonfinite_all": int(m.group(5)),
                         "write_run_read_ms": float(m.group(3)), "run_ms": float(m.group(4))})
        elif m := ROUND.match(line):
            rounds.append({"round": int(m.group(1)), "ms": float(m.group(3)), "run_ms": float(m.group(4))})
        elif m := WARM.match(line):
            warm.append({"warmup": int(m.group(1)), "ms": float(m.group(3)), "run_ms": float(m.group(4))})
        elif m := MODEL.match(line):
            models.append({"path": m.group(2), "out0_elems": int(m.group(5)), "compile_ms": float(m.group(6)),
                           "fully_accelerated": m.group(7) == "true"})
        elif line.startswith("FATAL") or "FIXTURE_ERROR" in line or "WRITE_ERROR" in line:
            fatal.append(line)
        elif line.startswith("ERROR"):   # e.g. the QNN validator rejecting an op that then stays on the CPU: not fatal
            errors.append(line)
        if m := WALL.search(line):
            walls.setdefault(m.group(1), int(m.group(2)))
        if line.startswith("done"):
            done = True
    L = man["L"]
    status = "DONE" if done and not fatal and rows else ("PARTIAL" if rows else "FAILED")
    ms = [r["write_run_read_ms"] for r in rows]
    report = {"status": status, "graph": a.graph, "accel": a.accel, "precision": a.precision, "L": L,
              "rows_file": man["rows_file"].removeprefix("device/"), "runner": "kev_npu_runner (android/measure/kev_npu_runner.cc)",
              "compile_ms": models[0]["compile_ms"] if models else None,
              "fully_accelerated": models[0]["fully_accelerated"] if models else None,
              "wall_ms": walls, "fatal": fatal, "error_lines": errors[:50], "rows": rows,
              "summary": {"count": len(rows), "finite_rows": sum(r["finite"] for r in rows),
                          "nonfinite_rows": sum(not r["finite"] for r in rows),
                          "parity_pass_median_write_run_read_ms": statistics.median(ms) if ms else None,
                          "parity_pass_min_ms": min(ms) if ms else None, "parity_pass_max_ms": max(ms) if ms else None},
              "latency": ({"warmup": warm, "n": len(rounds),
                           "median_write_run_read_ms": statistics.median(r["ms"] for r in rounds),
                           "min_ms": min(r["ms"] for r in rounds), "max_ms": max(r["ms"] for r in rounds),
                           "median_run_ms": statistics.median(r["run_ms"] for r in rounds)} if rounds else None),
              "stdout": a.stdout, "stdout_sha256": sha256_file(K / a.stdout)}
    if a.logcat and (K / a.logcat).exists():
        lines = (K / a.logcat).read_text(errors="replace").splitlines()
        report["logcat"] = a.logcat
        report["delegate_lines"] = [l for l in lines if DELEGATE.search(l)][:400]
    # the delegate evidence as numbers (stdout = the runner's stderr too; logcat when present), every run
    ev = text + ((K / a.logcat).read_text(errors="replace").splitlines() if a.logcat and (K / a.logcat).exists() else [])
    part = [re.search(r"Partitioned subgraph<(\d+)>, selected (\d+) ops, from a total of (\d+) ops\. resulted in (\d+) "
                      r"partitions", l) for l in ev]
    repl = [re.search(r"Replacing (\d+) out of (\d+) node\(s\) with delegate \((\w+)\) node, yielding (\d+) partitions", l)
            for l in ev]
    report["delegate_summary"] = {
        "partitioned": sorted({(int(m.group(2)), int(m.group(3)), int(m.group(4))) for m in part if m}),
        "replacing": sorted({(int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4))) for m in repl if m}),
        "npu_registered": any("NPU accelerator registered" in l for l in ev),
        "jit_from_cache": any("initialized from cached model" in l for l in ev),
        "compiler_plugin_applied": any("compiler plugins were applied successfully" in l for l in ev)}
    if a.hsel_in:
        shutil.copyfile(K / a.hsel_in, hsel_out)
        report["hsel"] = str(hsel_out.relative_to(K))
        report["hsel_bytes"] = hsel_out.stat().st_size
        expected = sum((1 + r["k"]) * 1024 * 4 for r in rows)
        report["hsel_bytes_expected"] = expected
        assert hsel_out.stat().st_size == expected, (hsel_out.stat().st_size, expected)
    dump_json(out, report)
    print(json.dumps({k: report[k] for k in ("status", "L", "compile_ms", "fully_accelerated", "summary", "latency")},
                     default=str))


if __name__ == "__main__":
    main()
