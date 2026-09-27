#!/usr/bin/env python3
"""On-device correctness gate for the Fun-ASR-Nano-2512 bundle on the Galaxy S26 (SM-S942Q) with the official
LiteRT-LM CLI (`litert_lm_advanced_main` android_arm64 built from the v0.16.1 tag, staged with its LiteRt .so in
/data/local/tmp/docling_gate by the docling lane; sha256 ea5bf071... recorded in docling_work/FINDINGS.md).
Type = vibevoice_asr_work/pixel_gate.py. One PROCESS per clip (wall includes engine load):

  cd /data/local/tmp/funasr_gate && LD_LIBRARY_PATH=<libs>:/data/local/tmp/funasr_gate <bin> --backend=<be>
      --audio_backend=cpu --sampler_backend=cpu --model_path=/data/local/tmp/funasr_gate/Fun-ASR-Nano-2512.litertlm
      --max_num_tokens=2048 --max_output_tokens=512 --input_prompt='[audio:/data/local/tmp/funasr_gate/<id>.wav]'
      > out_<tag>.txt 2> err_<tag>.txt < /dev/null; echo EXIT=$?

The prompt is audio only: the v0.16.1 CLI turns `[audio:<path>]` with no surrounding text into a content list with
one audio item (litert_lm_lib.cc BuildContentList), so the bundle's jinja adds the funasr default instruction.
Transcript = the non-log stdout lines, through funasr's post-processing (as mac_gate.py). The first clip runs with
--report_peak_memory_footprint and its stderr is saved in full to s26_gate_<tag>_first_stderr.log.

The shared-device hold is taken OUTSIDE this script (hold_cli.py with a sleeper pid, see ROUND3.md); this script
refuses to run unless the hold file names that pid. /proc/uptime is read before and after every clip: a smaller value
= the phone rebooted = hard stop for the leg (memory: S26 reboot at engine start).

  python3 s26_gate.py --backend cpu --tag s26_r3_cpu_cpu
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mac_gate import loop_suspect, norm_text, postprocess, wer_counts  # noqa: E402

D = "/data/local/tmp/funasr_gate"
LIBS = "/data/local/tmp/docling_gate"
BIN = "/data/local/tmp/docling_gate/litert_lm_advanced_main"
HOLD = os.path.join(HERE, "..", "community_accel_work", "s2_npu_sweep", ".device_hold")
SLEEPER_PID = os.path.join(HERE, "out", "hold_sleeper.pid")
LOG_RE = re.compile(r"^(I\d{4} |W\d{4} |E\d{4} |F\d{4} |VERBOSE:|INFO:|WARNING:|ERROR:|real\t|user\t|sys\t)")


def adb(serial, args, timeout=600, check=True):
    return subprocess.run(["adb", "-s", serial] + args, capture_output=True, text=True, timeout=timeout, check=check)


def wait_device(serial, tries=3, each_s=120):
    """adb dropped: wait for the phone up to `tries` times. Returns True when it is back."""
    for k in range(tries):
        try:
            subprocess.run(["adb", "-s", serial, "wait-for-device"], timeout=each_s, check=True)
            return True
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            print(f"   adb wait {k + 1}/{tries} timed out", flush=True)
    return False


def uptime(serial):
    try:
        r = adb(serial, ["shell", "cat /proc/uptime"], timeout=30, check=False)
        return float(r.stdout.split()[0]) if r.returncode == 0 and r.stdout.strip() else None
    except (subprocess.TimeoutExpired, ValueError, IndexError):
        return None


def device_state(serial):
    s = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "uptime_s": uptime(serial)}
    th = adb(serial, ["shell", "dumpsys thermalservice"], check=False).stdout
    m = re.search(r"Thermal Status: (\d+)", th)
    s["thermal_status"] = int(m.group(1)) if m else None
    m = re.search(r"Temperature\{mValue=([\d.]+), mType=3, mName=SKIN", th)
    s["skin_c"] = float(m.group(1)) if m else None
    fr = adb(serial, ["shell", "for p in /sys/devices/system/cpu/cpufreq/policy*; do echo $p $(cat $p/scaling_max_freq) "
                               "$(cat $p/cpuinfo_max_freq); done"], check=False).stdout
    s["cpufreq_max_vs_cpuinfo"] = {ln.split()[0].rsplit("/", 1)[1]: [int(ln.split()[1]), int(ln.split()[2])]
                                   for ln in fr.splitlines() if len(ln.split()) == 3}
    s["df_data"] = adb(serial, ["shell", "df -h /data | tail -1"], check=False).stdout.strip()
    s["litert_procs"] = adb(serial, ["shell", "ps -A | grep -i litert"], check=False).stdout.strip()
    return s


def check_hold():
    cur = json.load(open(HOLD)) if os.path.exists(HOLD) and open(HOLD).read().strip() else None
    mine = int(open(SLEEPER_PID).read().strip()) if os.path.exists(SLEEPER_PID) else None
    if not cur or mine is None or cur.get("pid") != mine:
        raise SystemExit(f"hold is not ours: hold={cur} sleeper pid={mine}")
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", default="RFGL80R6A6H")
    ap.add_argument("--bin", default=BIN)
    ap.add_argument("--libs", default=LIBS)
    ap.add_argument("--bin-label", default="v0.16.1 tag build (sha256 ea5bf071...)")
    ap.add_argument("--bundle", default="Fun-ASR-Nano-2512.litertlm")
    ap.add_argument("--local-bundle", default=os.path.join(HERE, "out", "bundle", "Fun-ASR-Nano-2512.litertlm"))
    ap.add_argument("--backend", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--audio-backend", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--ids", default="", help="comma-separated fixture ids (default: all 25)")
    ap.add_argument("--max-out", type=int, default=512)
    ap.add_argument("--mac-cpu", default="mac_gate_r10_main_cpu_cpu.json")
    ap.add_argument("--mac-gpu", default="mac_gate_r10_main_gpu_cpu.json")
    ap.add_argument("--skin-max", type=float, default=42.0)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    s = args.serial
    hold = check_hold()

    meta = json.load(open(os.path.join(HERE, "fixtures", "meta.json")))
    ids = [x.strip() for x in args.ids.split(",") if x.strip()] or [m["id"] for m in meta]
    meta_by_id = {m["id"]: m for m in meta}
    oracle = {r["id"]: r["text"] for r in json.load(open(os.path.join(HERE, "oracle_transcripts.json")))["rows"]}
    mac_cpu = {r["id"]: r["text"] for r in json.load(open(os.path.join(HERE, args.mac_cpu)))["rows"]}
    mac_gpu = {r["id"]: r["text"] for r in json.load(open(os.path.join(HERE, args.mac_gpu)))["rows"]}

    model = adb(s, ["shell", "getprop ro.product.model"]).stdout.strip()
    got = adb(s, ["shell", f"toybox stat -c %s {D}/{args.bundle}"]).stdout.strip()
    local_size = os.path.getsize(args.local_bundle)
    assert got == str(local_size), f"device bundle size {got} != local {local_size}"
    before = device_state(s)
    print(f"device {model} ({s}) | bundle {args.bundle} {got} B | bin {args.bin} ({args.bin_label}) | backend "
          f"{args.backend} audio {args.audio_backend} | state {json.dumps(before)}", flush=True)
    assert not before["litert_procs"], f"another litert process is on the phone: {before['litert_procs']}"

    rows, stop = [], None
    for i, fid in enumerate(ids):
        st = device_state(s) if i else before
        while st["skin_c"] is not None and st["skin_c"] >= args.skin_max:
            print(f"   SKIN {st['skin_c']} >= {args.skin_max}: waiting 300 s", flush=True)
            time.sleep(300)
            st = device_state(s)
        if st["litert_procs"]:
            stop = f"litert process on the phone before {fid}: {st['litert_procs']}"
            break
        m = meta_by_id[fid]
        clip = os.path.basename(m["file"])
        peak = " --report_peak_memory_footprint" if i == 0 else ""
        cmd = (f"cd {D} && LD_LIBRARY_PATH={args.libs}:{D} {args.bin} --backend={args.backend} "
               f"--audio_backend={args.audio_backend} --sampler_backend=cpu --model_path={D}/{args.bundle} "
               f"--max_num_tokens=2048 --max_output_tokens={args.max_out}{peak} --input_prompt='[audio:{D}/{clip}]' "
               f"> out_{args.tag}.txt 2> err_{args.tag}.txt < /dev/null; echo EXIT=$?")
        up0 = uptime(s)
        t0 = time.time()
        try:
            r = adb(s, ["shell", cmd], timeout=1800, check=False)
            run_out, run_rc = r.stdout, r.returncode
        except subprocess.TimeoutExpired:
            run_out, run_rc = "", "timeout"
        dt = time.time() - t0
        up1 = uptime(s)
        if up1 is None:
            print(f"   adb lost after {fid} ({dt:.1f} s): waiting for the phone", flush=True)
            back = wait_device(s)
            up1 = uptime(s) if back else None
            if up1 is None or (up0 is not None and up1 < up0):
                stop = f"phone gone or rebooted during {fid}: uptime before {up0}, after {up1}, back={back}"
                rows.append({"id": fid, "error": stop, "wall_s": round(dt, 1)})
                break
        if up0 is not None and up1 is not None and up1 < up0:
            stop = f"phone rebooted during {fid}: uptime {up0} -> {up1}"
            rows.append({"id": fid, "error": stop, "wall_s": round(dt, 1)})
            break
        exit_code = (re.search(r"EXIT=(\d+)", run_out) or [None, "?"])[1]
        out = adb(s, ["shell", f"cat {D}/out_{args.tag}.txt"], check=False).stdout
        err = adb(s, ["shell", f"cat {D}/err_{args.tag}.txt"], check=False).stdout
        text_lines = [ln for ln in out.splitlines() if ln.strip() and not LOG_RE.match(ln)]
        raw = "\n".join(text_lines)
        text = postprocess(raw).strip()
        peak_lines = [ln for ln in err.splitlines() if re.search(r"[Pp]eak", ln)]
        row = {"id": fid, "audio_s": m.get("duration_s"), "text_raw": raw, "text": text, "wall_s": round(dt, 2),
               "exit": exit_code, "adb_rc": run_rc, "stdout_lines": len(out.splitlines()),
               "oracle_equal": text == oracle[fid].strip(), "mac_cpu_equal": text == mac_cpu[fid].strip(),
               "mac_gpu_equal": text == mac_gpu[fid].strip(), "loop_suspect": loop_suspect(text),
               "invalid_decode_lines": err.count("Invalid decode"), "stderr_bytes": len(err),
               "stderr_error_lines": [ln[:300] for ln in err.splitlines() if re.match(r"^(E\d{4} |ERROR:|F\d{4} )", ln)][:8],
               "peak_lines": peak_lines, "uptime_before": up0, "uptime_after": up1, "skin_c_before": st["skin_c"],
               "thermal_status_before": st["thermal_status"]}
        if m.get("text"):
            e, n = wer_counts(norm_text(m["text"]), norm_text(text))
            eo, _ = wer_counts(norm_text(m["text"]), norm_text(oracle[fid]))
            row["wer"], row["oracle_wer"] = [e, n], [eo, n]
        rows.append(row)
        flag = "==" if row["oracle_equal"] else "!="
        print(f"[{fid:12s}] wall {dt:6.2f} s | exit {exit_code} | oracle {flag} | mac_cpu {row['mac_cpu_equal']} | "
              f"mac_gpu {row['mac_gpu_equal']} | inv {row['invalid_decode_lines']} | {text[:150]!r}"
              + (f"\n      peak: {peak_lines}" if peak_lines else ""), flush=True)
        if exit_code != "0":
            print("   stderr tail:", err[-1500:], flush=True)
        if i == 0:
            with open(os.path.join(HERE, f"s26_gate_{args.tag}_first_stderr.log"), "w") as f:
                f.write(err)
            with open(os.path.join(HERE, f"s26_gate_{args.tag}_first_stdout.log"), "w") as f:
                f.write(out)

    ok = [r for r in rows if "error" not in r]
    en = [r for r in ok if "wer" in r]
    audio = sum(r["audio_s"] or 0 for r in ok)
    summ = {"n": len(ok), "requested": len(ids), "stopped": stop,
            "oracle_match": sum(r["oracle_equal"] for r in ok), "mac_cpu_match": sum(r["mac_cpu_equal"] for r in ok),
            "mac_gpu_match": sum(r["mac_gpu_equal"] for r in ok),
            "exit_nonzero": sum(1 for r in ok if r["exit"] != "0"), "empty": sum(1 for r in ok if not r["text"]),
            "loop_suspect": sum(r["loop_suspect"] for r in ok),
            "invalid_decode_lines": sum(r["invalid_decode_lines"] for r in ok),
            "wer_en": [sum(r["wer"][0] for r in en), sum(r["wer"][1] for r in en)] if en else None,
            "oracle_wer_en": [sum(r["oracle_wer"][0] for r in en), sum(r["wer"][1] for r in en)] if en else None,
            "wall_s": round(sum(r["wall_s"] for r in ok), 2), "audio_s": round(audio, 3),
            "wall_rtf_incl_engine_load": round(sum(r["wall_s"] for r in ok) / audio, 4) if audio else None,
            "wall_median_s": sorted(r["wall_s"] for r in ok)[len(ok) // 2] if ok else None,
            "first_clip_peak_lines": ok[0]["peak_lines"] if ok else None}
    after = device_state(s)
    doc = {"tag": args.tag, "device": model, "serial": s, "bin": args.bin, "bin_label": args.bin_label, "libs": args.libs,
           "bundle": args.bundle, "bundle_bytes": int(got), "local_bundle": os.path.relpath(args.local_bundle, HERE),
           "backend": args.backend, "audio_backend": args.audio_backend, "sampler_backend": "cpu",
           "max_output_tokens": args.max_out, "hold": hold, "state_before": before, "state_after": after,
           "mac_cpu_file": args.mac_cpu, "mac_gpu_file": args.mac_gpu, "summary": summ, "rows": rows}
    with open(os.path.join(HERE, f"s26_gate_{args.tag}.json"), "w") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False), flush=True)
    print("S26_GATE_DONE" if not stop else f"S26_GATE_STOPPED: {stop}", flush=True)


if __name__ == "__main__":
    main()
