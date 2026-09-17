#!/usr/bin/env python3
"""On-device gate for the streaming bundle with the official LiteRT-LM CLI (litert_lm_advanced_main,
android_arm64, the same binary the BitNet lane gated with) — one PROCESS per clip, one TURN per
26-frame window through `--multi_turns=true` (the CLI reads follow-up prompts from stdin, each
`[audio:<wav>]` line = one user turn on the same Conversation = the vendor's streaming protocol).

  printf '[audio:w00.wav]\n[audio:w01.wav]\n\n' | ./litert_lm_advanced_main --multi_turns=true --backend=<be>
      --audio_backend=<abe> --sampler_backend=cpu --model_path=<bundle> --max_num_tokens=2048

Device dir /data/local/tmp/vvs_gate must hold the bundle, the binary and the window wavs
(`--push-windows` writes + pushes them from the fixtures).  Reports per-clip chunks, corpus WER,
wall-clock per process (includes engine load), peak RSS of the first run.

  python pixel_gate.py [--serial S] [--backend cpu|gpu] [--audio-backend cpu|gpu] [--limit N] [--push-windows]
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
from mac_gate import load_wav, windows_of, write_wav, norm_text, wer_counts, SR  # noqa: E402

BITNET = os.path.join(os.path.dirname(HERE), "vibevoice_asr_work")
D = "/data/local/tmp/vvs_gate"
DEFAULT_LIBS = {"RFGL80R6A6H": "/data/local/tmp/docling_gate",   # Galaxy S26 (SM-S942Q)
                "4C131JEKB15210": "/data/local/tmp/g41_gate"}     # Pixel 8a
PROMPT_MARK = "Please enter the prompt (or press Enter to end): "
LOG_RE = re.compile(r"^(I\d{4} |W\d{4} |E\d{4} |VERBOSE:|INFO:|WARNING:|ERROR:|real\t|user\t|sys\t)")


def adb(serial, args, timeout=600, check=True):
    return subprocess.run(["adb", "-s", serial] + args, capture_output=True, text=True,
                          timeout=timeout, check=check)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", default=os.environ.get("SERIAL", "RFGL80R6A6H"))
    ap.add_argument("--libs", default="")
    ap.add_argument("--bundle", default="VibeVoice-ASR-Streaming-1.5B.litertlm")
    ap.add_argument("--local-bundle", default=os.path.join(HERE, "out", "bundle_wi8", "VibeVoice-ASR-Streaming-1.5B.litertlm"))
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--audio-backend", default="cpu")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--max-out", type=int, default=256)
    ap.add_argument("--tag", default="")
    ap.add_argument("--push-windows", action="store_true")
    args = ap.parse_args()
    s = args.serial
    libs = args.libs or DEFAULT_LIBS.get(s, D)
    model = adb(s, ["shell", "getprop", "ro.product.model"]).stdout.strip()
    got = adb(s, ["shell", f"toybox stat -c %s {D}/{args.bundle}"]).stdout.strip()
    if os.path.exists(args.local_bundle):
        assert got == str(os.path.getsize(args.local_bundle)), f"device bundle size {got} != local {os.path.getsize(args.local_bundle)}"
    print(f"device {model} ({s}) bundle {args.bundle} {got} bytes  backend={args.backend} audio={args.audio_backend}")

    meta = json.load(open(os.path.join(BITNET, "fixtures", "meta.json")))[:args.limit]
    wins = {}
    wdir = os.path.join(HERE, "out", "windows_device")
    os.makedirs(wdir, exist_ok=True)
    for m in meta:
        wav = load_wav(os.path.join(BITNET, m["file"]))
        ws = windows_of(wav)
        names = []
        for j, w in enumerate(ws):
            p = os.path.join(wdir, f"{m['id']}_w{j:02d}.wav")
            if args.push_windows:
                write_wav(p, w)
            names.append(os.path.basename(p))
        wins[m["id"]] = (names, len(wav) / SR)
    if args.push_windows:
        adb(s, ["push"] + [os.path.join(wdir, n) for names, _ in wins.values() for n in names] + [D + "/"], timeout=1800)
        n_dev = adb(s, ["shell", f"ls {D}/*_w*.wav | wc -l"]).stdout.strip()
        print(f"pushed windows: {sum(len(n) for n, _ in wins.values())} local, {n_dev} on device")

    eager = {}
    ep = os.path.join(HERE, "eager_precheck_20clips.json")
    if os.path.exists(ep):
        eager = {r["id"]: r["streaming_chunks"] for r in json.load(open(ep))}
    tag = args.tag or f"{model.replace(' ', '')}_{args.backend}_{args.audio_backend}"
    rows, errs, words, agree, n_chunks = [], 0, 0, 0, 0
    for i, m in enumerate(meta):
        names, dur = wins[m["id"]]
        stdin_lines = "".join(f"[audio:{D}/{n}]\n" for n in names) + "\n"
        extra = f" --num_cpu_threads={args.threads}" if args.threads else ""
        peak = " --report_peak_memory_footprint" if i == 0 else ""
        cmd = (f"cd {D} && printf '%s' '{stdin_lines}' | LD_LIBRARY_PATH={libs}:{D} ./litert_lm_advanced_main --multi_turns=true "
               f"--backend={args.backend} --audio_backend={args.audio_backend} --sampler_backend=cpu --model_path={D}/{args.bundle} "
               f"--max_num_tokens=2048 --max_output_tokens={args.max_out}{extra}{peak} > {D}/out_{tag}.txt 2> {D}/err_{tag}.txt; echo EXIT=$?")
        t0 = time.time()
        r = adb(s, ["shell", cmd], timeout=1800, check=False)
        dt = time.time() - t0
        exit_code = (re.search(r"EXIT=(\d+)", r.stdout) or [None, "?"])[1]
        out = adb(s, ["shell", f"cat {D}/out_{tag}.txt"], check=False).stdout
        err = adb(s, ["shell", f"cat {D}/err_{tag}.txt"], check=False).stdout
        parts = out.split(PROMPT_MARK)
        chunks = []
        for p in parts[1:1 + len(names)]:
            lines = [ln for ln in p.splitlines() if not LOG_RE.match(ln)]
            chunks.append("\n".join(lines).strip("\n"))
        hyp = "".join(chunks)
        hyp_plain = re.sub(r"^\s*(speaker\s*\d+|\[?[Ss]peaker[^:\]]*\]?)\s*:\s*", "", hyp)
        e, n = wer_counts(norm_text(m["text"]), norm_text(hyp_plain))
        errs, words = errs + e, words + n
        ref = eager.get(m["id"], [])
        same = [a.strip() == b.strip() for a, b in zip(chunks, ref)] if len(ref) == len(chunks) else []
        agree += sum(same)
        n_chunks += len(chunks)
        peak_kb = None
        mm = re.search(r"[Pp]eak.*?(\d+(?:\.\d+)?)\s*(MB|MiB|KB|kB|GB)", err)
        if mm:
            peak_kb = f"{mm.group(1)} {mm.group(2)}"
        rows.append({"id": m["id"], "dur": round(dur, 2), "windows": len(names), "chunks": chunks, "eager_chunks": ref,
                     "chunk_agree": same, "hyp": hyp, "ref": m["text"], "errs": e, "words": n,
                     "wall_s": round(dt, 1), "exit": exit_code, "peak": peak_kb, "n_parts": len(parts) - 1})
        print(f"[{m['id']}] {len(names)} windows errs={e}/{n} agree={sum(same)}/{len(chunks)} wall={dt:5.1f}s exit={exit_code} peak={peak_kb} | {hyp[:150]!r}")
        if exit_code != "0" or len(parts) - 1 < len(names) + 1:
            print("   stdout:", out[-600:].replace("\n", "\\n"))
            print("   stderr tail:", err[-1200:])
        sys.stdout.flush()
        if i == 0:
            open(os.path.join(HERE, f"pixel_gate_{tag}_first_stderr.log"), "w").write(err)
    total_audio = sum(r["dur"] for r in rows)
    total_wall = sum(r["wall_s"] for r in rows)
    print(f"corpus WER {errs}/{words} = {100*errs/max(words,1):.2f}%  chunk agreement with eager {agree}/{n_chunks}  "
          f"wall RTF {total_wall/total_audio:.2f} (includes per-process engine load)")
    json.dump({"device": model, "serial": s, "bundle": args.bundle, "backend": args.backend,
               "audio_backend": args.audio_backend, "wer_errs": errs, "ref_words": words, "chunk_agree": agree,
               "chunks": n_chunks, "wall_rtf": round(total_wall / total_audio, 3), "rows": rows},
              open(os.path.join(HERE, f"pixel_gate_{tag}.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
