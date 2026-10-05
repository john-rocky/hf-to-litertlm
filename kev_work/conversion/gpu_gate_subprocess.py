"""Run one GPU parity gate (scripts/litert_parity.py --accel gpu ...) in a child process, so a delegate abort
(SIGABRT) or any other crash ends the child, not the caller. The child's stdout goes to --log and its stderr to
--log + ".err"; the wrapper records the return code (negative = killed by that signal), the wall seconds and the last
20 stderr lines in results/gpu_subprocess_<name of --expect>. If the child did not write the parity json named by
--expect (it died before the end), the wrapper writes that json itself with status GPU_FAIL and the same evidence, so
the gate has a result file either way.

    python gpu_gate_subprocess.py --expect results/litert_gpu_f32_parity_L1024_v4_fp16fc_fp16emb.json \
        --log logs/litert_gpu_f32_parity_L1024_v4_fp16fc_fp16emb.log -- \
        --tflite exports/kev08b_rowprefill_L1024_v4_fp16fc_fp16emb.tflite --L 1024 --variant v4_fp16fc_fp16emb \
        --accel gpu --f32 --allow-float-table"""
import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

K = Path(__file__).resolve().parents[1]


def tail(path, n=20):
    try:
        return path.read_text(errors="replace").splitlines()[-n:]
    except FileNotFoundError:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect", required=True, help="parity json the child writes on success (K-relative)")
    ap.add_argument("--log", required=True, help="child stdout (K-relative); stderr goes to <log>.err")
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("child", nargs=argparse.REMAINDER, help="-- then the litert_parity.py arguments")
    a = ap.parse_args()
    child = a.child[1:] if a.child[:1] == ["--"] else a.child
    expect, log = K / a.expect, K / a.log
    err = log.with_name(log.name + ".err")
    record_path = K / "results" / f"gpu_subprocess_{expect.name}"
    for p in (record_path, log, err):
        assert not p.exists(), f"refusing to overwrite {p}"
    assert not expect.exists(), f"refusing to overwrite {expect}"
    cmd = [sys.executable, str(K / "scripts/litert_parity.py"), *child]
    started_at, t0 = time.strftime("%Y-%m-%dT%H:%M:%S%z"), time.time()
    with open(log, "w") as fo, open(err, "w") as fe:
        proc = subprocess.Popen(cmd, stdout=fo, stderr=fe, cwd=str(K / "scripts"))
        try:
            rc = proc.wait(timeout=a.timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = proc.wait()
            timed_out = True
    seconds = round(time.time() - t0, 1)
    sig = signal.Signals(-rc).name if rc < 0 else None
    record = {"cmd": cmd, "started_at": started_at, "seconds_wall": seconds, "returncode": rc, "signal": sig,
              "timed_out": timed_out, "log": str(log.relative_to(K)), "stderr_log": str(err.relative_to(K)),
              "stderr_last_20": tail(err), "stdout_last_20": tail(log), "parity_json": str(expect.relative_to(K)),
              "parity_json_written_by_child": expect.exists()}
    if rc != 0 and not expect.exists():
        expect.write_text(json.dumps({
            "status": "GPU_FAIL", "what": "the GPU parity child died before writing its result",
            "returncode": rc, "signal": sig, "timed_out": timed_out, "seconds_wall": seconds, "cmd": cmd,
            "stderr_last_20": record["stderr_last_20"], "stdout_last_20": record["stdout_last_20"],
            "log": record["log"], "stderr_log": record["stderr_log"]}, indent=1) + "\n")
        record["parity_json_written_by_wrapper"] = True
    record_path.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: record[k] for k in ("returncode", "signal", "timed_out", "seconds_wall",
                                             "parity_json_written_by_child")}, indent=1))
    print("\n".join(record["stderr_last_20"]))


if __name__ == "__main__":
    main()
