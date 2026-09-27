#!/usr/bin/env python3
"""Galaxy S26 speed legs for the Fun-ASR-Nano-2512 bundle (type = vibevoice_asr_streaming_work/s26_bench.sh), with the
LiteRT-LM CLI `litert_lm_advanced_main` built from the v0.16.1 tag (the round-3 gate binary, /data/local/tmp/docling_gate).

Per backend (cpu, then gpu; the audio encoder on the CPU in both):
  1. readiness: SKIN < 40 C and scaling_max_freq == cpuinfo_max_freq on every cpufreq policy (the S26 caps the CPU after
     a heavy leg); polled every 20 s for up to 20 min, the last reading goes to bench/s26_<be>_ready.txt. A leg that
     never gets ready still runs and is marked `ready: false`.
  2. one benchmark reading (LM only; a text prompt, not audio):
       --benchmark --benchmark_prefill_tokens=256 --benchmark_decode_tokens=256 --input_prompt_file=prompt256.txt
     In benchmark mode the v0.16.1 runtime tokenizes the prompt and resizes the ids to 256 (session_utils.cc), and
     decodes 256 tokens (tasks.cc). The log must carry `Prefill Turn 1: Processed 256` and `Decode Turn 1: Processed
     256`; without them the leg is recorded as invalid.
  3. three engine-load-only runs: the gate's flags plus --multi_turns=true with one empty stdin line (the CLI builds
     the conversation - which loads the audio executor because --audio_backend is set - then reads the empty line
     and exits). Timed on the host around `adb shell`, the same way s26_gate.py timed each clip, and on the device.
  4. 180 s rest before the next leg.
/proc/uptime is read before and after every command: a smaller value = the phone rebooted = hard stop.
The shared-device hold is taken outside (hold_cli.py + sleeper pid, as round 3); this script refuses to run unless
the hold names that pid, and checks that no other litert process is on the phone before every command.

  python3 s26_bench.py            # writes bench/s26_{cpu,gpu}.log, bench/s26_{cpu,gpu}_ready.txt,
                                  # bench/s26_{cpu,gpu}_load.txt, bench/s26_bench_state.json
"""
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from s26_gate import BIN, D, LIBS, adb, check_hold, device_state, uptime  # noqa: E402

SERIAL = "RFGL80R6A6H"
BUNDLE = "Fun-ASR-Nano-2512.litertlm"
PROMPT_LOCAL = os.path.join(HERE, "..", "minicpm5_work", "ship_lfm25_gpu_variant_20260812", "prompt256.txt")
B = os.path.join(HERE, "bench")
COMMON = (f"--audio_backend=cpu --sampler_backend=cpu --model_path={D}/{BUNDLE} --max_num_tokens=2048")


def ready_reading(s):
    st = device_state(s)
    caps = st["cpufreq_max_vs_cpuinfo"]
    uncapped = bool(caps) and all(a == b for a, b in caps.values())
    cool = st["skin_c"] is not None and st["skin_c"] < 40.0
    return st, uncapped and cool


def wait_ready(s, be):
    t0 = time.time()
    lines = []
    while True:
        st, ok = ready_reading(s)
        line = (f"{st['time']} SKIN={st['skin_c']} thermal_status={st['thermal_status']} "
                + " ".join(f"{k}={v[0]}/{v[1]}" for k, v in sorted(st["cpufreq_max_vs_cpuinfo"].items()))
                + f" ok={int(ok)} waited_s={time.time() - t0:.0f}")
        lines.append(line)
        print("   ready?", line, flush=True)
        if ok or time.time() - t0 >= 1200:
            break
        time.sleep(20)
    with open(os.path.join(B, f"s26_{be}_ready.txt"), "w") as f:
        f.write(lines[-1] + ("\n" if ok else " NOT READY after 20 min (leg ran anyway)\n"))
        if len(lines) > 1:
            f.write(f"# {len(lines)} readings; first: {lines[0]}\n")
    return st, ok, lines


def no_litert(s):
    out = adb(s, ["shell", "ps -A | grep -i litert"], check=False).stdout.strip()
    if out:
        raise SystemExit(f"another litert process is on the phone: {out}")


def run_checked(s, cmd, timeout=1800):
    """adb shell cmd with an uptime check around it; returns (stdout, host wall s, uptime before, after)."""
    no_litert(s)
    up0 = uptime(s)
    t0 = time.time()
    r = adb(s, ["shell", cmd], timeout=timeout, check=False)
    dt = time.time() - t0
    up1 = uptime(s)
    if up0 is None or up1 is None or up1 < up0:
        raise SystemExit(f"STOP: phone gone or rebooted: uptime {up0} -> {up1} (cmd {cmd[:120]})")
    return r.stdout, dt, up0, up1


def main():
    s = SERIAL
    hold = check_hold()
    os.makedirs(B, exist_ok=True)
    model = adb(s, ["shell", "getprop ro.product.model"]).stdout.strip()
    release = adb(s, ["shell", "getprop ro.build.version.release"]).stdout.strip()
    size = adb(s, ["shell", f"toybox stat -c %s {D}/{BUNDLE}"]).stdout.strip()
    local = os.path.join(HERE, "out", "bundle", BUNDLE)
    assert size == str(os.path.getsize(local)), (size, os.path.getsize(local))
    md5_dev = adb(s, ["shell", f"md5sum {D}/{BUNDLE}"]).stdout.split()[0]
    md5_loc = subprocess.run(["md5", "-q", local], capture_output=True, text=True).stdout.strip()
    assert md5_dev == md5_loc, (md5_dev, md5_loc)
    adb(s, ["push", PROMPT_LOCAL, f"{D}/prompt256.txt"])
    psize = adb(s, ["shell", f"toybox stat -c %s {D}/prompt256.txt"]).stdout.strip()
    assert psize == str(os.path.getsize(PROMPT_LOCAL)), (psize, os.path.getsize(PROMPT_LOCAL))
    doc = {"device": model, "os": f"Android {release}" if release else None, "serial": s, "bin": BIN, "bin_label": "litert_lm_advanced_main v0.16.1 tag build (sha256 ea5bf071...)",
           "bundle": BUNDLE, "bundle_bytes": int(size), "bundle_md5_device": md5_dev, "hold": hold,
           "prompt": os.path.relpath(PROMPT_LOCAL, HERE), "legs": {}}
    print(f"device {model} | bundle {size} B md5 {md5_dev} (= local) | prompt256 pushed ({psize} B)", flush=True)
    for k, be in enumerate(("cpu", "gpu")):
        if k:
            print("rest 180 s", flush=True)
            time.sleep(180)
        check_hold()
        st, ok, lines = wait_ready(s, be)
        leg = {"ready": ok, "ready_readings": len(lines), "state_before": st}
        cmd = (f"cd {D} && LD_LIBRARY_PATH={LIBS}:{D} {BIN} --backend={be} {COMMON} --input_prompt_file={D}/prompt256.txt "
               f"--benchmark --benchmark_prefill_tokens=256 --benchmark_decode_tokens=256 > {D}/bench_{be}.log 2>&1 "
               f"< /dev/null; echo EXIT=$?")
        print(f"=== leg {be} {time.strftime('%F %T')}", flush=True)
        out, dt, up0, up1 = run_checked(s, cmd)
        adb(s, ["pull", f"{D}/bench_{be}.log", os.path.join(B, f"s26_{be}.log")])
        log = open(os.path.join(B, f"s26_{be}.log")).read()
        pre = re.search(r"Prefill Turn 1: Processed (\d+) tokens", log)
        dec = re.search(r"Decode Turn 1: Processed (\d+) tokens", log)
        leg.update({"exit": (re.search(r"EXIT=(\d+)", out) or [None, "?"])[1], "host_wall_s": round(dt, 2),
                    "uptime": [up0, up1], "prefill_processed": int(pre.group(1)) if pre else None,
                    "decode_processed": int(dec.group(1)) if dec else None})
        leg["valid"] = leg["prefill_processed"] == 256 and leg["decode_processed"] == 256 and leg["exit"] == "0"
        for ln in log.splitlines():
            if re.search(r"Processed|Speed|Time to first token|Init Total|Replacing \d+ out of|Validation error", ln):
                print("   ", ln.strip()[:160], flush=True)
        print(f"   valid={leg['valid']} exit={leg['exit']} host wall {dt:.1f} s", flush=True)
        loads = []
        for i in range(3):
            lcmd = (f"cd {D} && t0=$(date +%s.%N); printf '\\n' | LD_LIBRARY_PATH={LIBS}:{D} {BIN} --multi_turns=true "
                    f"--backend={be} {COMMON} --max_output_tokens=512 > /dev/null 2> {D}/load_{be}_{i}.log; e=$?; "
                    f"t1=$(date +%s.%N); echo EXIT=$e LOAD_ONLY_DEVICE_S=$(echo \"$t1 - $t0\" | bc)")
            lout, ldt, lu0, lu1 = run_checked(s, lcmd)
            dev = re.search(r"LOAD_ONLY_DEVICE_S=([\d.]+)", lout)
            ex = re.search(r"EXIT=(\d+)", lout)
            loads.append({"host_wall_s": round(ldt, 3), "device_s": float(dev.group(1)) if dev else None,
                          "exit": ex.group(1) if ex else "?", "uptime": [lu0, lu1]})
            print(f"   load-only {be} #{i + 1}: host {ldt:.2f} s, device {loads[-1]['device_s']} s, exit {loads[-1]['exit']}",
                  flush=True)
        adb(s, ["pull", f"{D}/load_{be}_0.log", os.path.join(B, f"s26_{be}_load0_stderr.log")])
        with open(os.path.join(B, f"s26_{be}_load.txt"), "w") as f:
            for i, L in enumerate(loads):
                f.write(f"run {i + 1}: LOAD_ONLY_HOST_S={L['host_wall_s']} LOAD_ONLY_DEVICE_S={L['device_s']} EXIT={L['exit']} "
                        f"uptime {L['uptime'][0]} -> {L['uptime'][1]}\n")
        leg["load_only"] = loads
        leg["state_after"] = device_state(s)
        doc["legs"][be] = leg
        with open(os.path.join(B, "s26_bench_state.json"), "w") as f:
            json.dump(doc, f, indent=1)
    print("S26_BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
