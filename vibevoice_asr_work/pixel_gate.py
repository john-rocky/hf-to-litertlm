#!/usr/bin/env python3
"""On-device gate for the VibeVoice-ASR-BitNet bundle with the official LiteRT-LM CLI
(litert_lm_advanced_main, android_arm64 built from the v0.16.1 tag — the same binary the
docling lane gated with) on the connected Android phone, per fixture clip:

  ./litert_lm_advanced_main --backend=<be> --audio_backend=<abe> --sampler_backend=cpu
      --model_path=VibeVoice-ASR-BitNet.litertlm --max_num_tokens=2048
      --input_prompt='[audio:<clip>.wav] This is a X.XX seconds audio, please transcribe it.'

The CLI writes the generated text to stdout and absl logs to stderr; both are pulled.  The
device dir /data/local/tmp/vv_gate must already hold the bundle, the binary and clipNN.wav
(see push_pixel.log).  Reports per-clip transcript, wall-clock, corpus WER, peak RSS.

  python pixel_gate.py [--serial S] [--backend cpu|gpu] [--audio-backend cpu|gpu] [--limit N]
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

D = "/data/local/tmp/vv_gate"
# LiteRt accelerator .so kits already staged on each phone (v0.16.x tag libs).
DEFAULT_LIBS = {"RFGL80R6A6H": "/data/local/tmp/docling_gate",   # Galaxy S26 (SM-S942Q)
                "4C131JEKB15210": "/data/local/tmp/g41_gate"}     # Pixel 8a
LOG_RE = re.compile(r"^(I\d{4} |W\d{4} |E\d{4} |VERBOSE:|INFO:|WARNING:|ERROR:|real\t|user\t|sys\t)")


def adb(serial, args, timeout=600, check=True):
    return subprocess.run(["adb", "-s", serial] + args, capture_output=True, text=True,
                          timeout=timeout, check=check)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", default=os.environ.get("SERIAL", "RFGL80R6A6H"))
    ap.add_argument("--libs", default="", help="LD_LIBRARY_PATH dir on the phone (default per serial)")
    ap.add_argument("--bundle", default="VibeVoice-ASR-BitNet.litertlm")
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--audio-backend", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0, help="--num_cpu_threads (0 = CLI default)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    s = args.serial
    libs = args.libs or DEFAULT_LIBS.get(s, D)
    hold = None
    if s == "RFGL80R6A6H":  # shared S26: owned hold (community_accel_work/device_hold.py)
        sys.path.insert(0, os.path.join(C.WORK, "..", "community_accel_work"))
        try:
            import device_hold
            device_hold.acquire("vibevoice_asr_work/pixel_gate.py")
            hold = device_hold
        except ImportError:
            print("(no device_hold module here — running without a shared-device hold)")
    try:
        run(args, s, libs)
    finally:
        if hold is not None:
            hold.release()


def run(args, s, libs):
    model = adb(s, ["shell", "getprop", "ro.product.model"]).stdout.strip()
    got = adb(s, ["shell", f"toybox stat -c %s {D}/{args.bundle}"]).stdout.strip()
    local = os.path.join(C.WORK, "out", "bundle30", args.bundle)
    if os.path.exists(local):
        assert got == str(os.path.getsize(local)), f"device bundle size {got} != local {os.path.getsize(local)}"
    print(f"device {model} ({s}) bundle {args.bundle} {got} bytes  backend={args.backend} audio={args.audio_backend}")

    meta = C.load_meta()
    if args.limit:
        meta = meta[:args.limit]
    rows, errs, words = [], 0, 0
    tag = args.tag or f"{model.replace(' ', '')}_{args.backend}_{args.audio_backend}"
    for i, m in enumerate(meta):
        clip = os.path.basename(m["file"])
        wav = C.load_wav(os.path.join(C.WORK, m["file"]))
        dur = len(wav) / C.SR
        prompt = f"[audio:{D}/{clip}] This is a {dur:.2f} seconds audio, please transcribe it."
        extra = f" --num_cpu_threads={args.threads}" if args.threads else ""
        peak = " --report_peak_memory_footprint" if i == 0 else ""
        cmd = (f"cd {D} && LD_LIBRARY_PATH={libs}:{D} ./litert_lm_advanced_main --backend={args.backend} "
               f"--audio_backend={args.audio_backend} --sampler_backend=cpu --model_path={D}/{args.bundle} "
               f"--max_num_tokens=2048{extra}{peak} --input_prompt='{prompt}' > {D}/out_{tag}.txt 2> {D}/err_{tag}.txt < /dev/null; "
               f"echo EXIT=$?")
        t0 = time.time()
        r = adb(s, ["shell", cmd], timeout=1800, check=False)
        dt = time.time() - t0
        exit_code = (re.search(r"EXIT=(\d+)", r.stdout) or [None, "?"])[1]
        out = adb(s, ["shell", f"cat {D}/out_{tag}.txt"], check=False).stdout
        err = adb(s, ["shell", f"cat {D}/err_{tag}.txt"], check=False).stdout
        text_lines = [ln for ln in out.splitlines() if ln.strip() and not LOG_RE.match(ln)]
        hyp = " ".join(text_lines).strip()
        peak_kb = None
        mm = re.search(r"[Pp]eak.*?(\d+(?:\.\d+)?)\s*(MB|MiB|KB|kB|GB)", err)
        if mm:
            peak_kb = f"{mm.group(1)} {mm.group(2)}"
        e, n = C.wer_counts(C.norm_text(m["text"]), C.norm_text(hyp))
        errs += e
        words += n
        rows.append({"id": m["id"], "dur": round(dur, 2), "hyp": hyp, "ref": m["text"], "errs": e, "words": n,
                     "wall_s": round(dt, 1), "exit": exit_code, "peak": peak_kb})
        print(f"[{m['id']}] {dur:5.1f}s errs={e}/{n} wall={dt:5.1f}s exit={exit_code} peak={peak_kb} | {hyp[:160]}")
        if exit_code != "0":
            print("   stderr tail:", err[-1500:])
        sys.stdout.flush()
        if i == 0:
            open(os.path.join(C.WORK, f"pixel_gate_{tag}_first_stderr.log"), "w").write(err)
    total_audio = sum(r["dur"] for r in rows)
    total_wall = sum(r["wall_s"] for r in rows)
    print(f"corpus WER {errs}/{words} = {100*errs/max(words,1):.2f}%   wall RTF {total_wall/total_audio:.2f} "
          f"(includes per-run engine load)")
    json.dump({"device": model, "serial": s, "bundle": args.bundle, "backend": args.backend,
               "audio_backend": args.audio_backend, "wer_errs": errs, "ref_words": words,
               "wall_rtf": round(total_wall / total_audio, 3), "rows": rows},
              open(os.path.join(C.WORK, f"pixel_gate_{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
