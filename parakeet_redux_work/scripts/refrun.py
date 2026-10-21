#!/usr/bin/env python3
"""Resumable full-set transcription with reference B or O (CPU fp32, batch 1, mode (i) = the audio alone).

refrun.py --which B|O --manifest <manifest.jsonl> --out <run.jsonl> [--threads 8] [--deadline EPOCH]
          [--decoder generate|loop] [--limit N]
One JSON line per utterance: id, text, tokens (non-blank ids), frames, durs, n_steps, max_run_same_frame,
eos_hit, enc_frames, valid_frames, audio_s, proc_s (features + encoder + decode wall). Utterances already in
<out> are skipped. Stops starting new utterances after --deadline (epoch seconds) and exits 0; writes
<out>.DONE when every manifest id is present. <out>.meta.json = load report + environment.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import redux_ref as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", required=True, choices=["B", "O"])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--deadline", type=float, default=0.0)
    ap.add_argument("--decoder", default="generate", choices=["generate", "loop"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    rows = R.read_jsonl(a.manifest)
    if a.limit:
        rows = rows[:a.limit]
    done = {r["id"] for r in R.read_jsonl(a.out)}
    todo = [r for r in rows if r["id"] not in done]
    if not todo:
        open(a.out + ".DONE", "w").write(time.strftime("%H:%M:%S") + "\n")
        print(f"{a.out}: nothing to do ({len(done)} done)")
        return 0
    if a.deadline and time.time() > a.deadline - 60:
        print("deadline too close, not loading")
        return 0
    R.set_threads(a.threads)
    t0 = time.time()
    model, rep = R.build(a.which)
    proc = R.processor()
    load_s = time.time() - t0
    meta = {"which": a.which, "manifest": a.manifest, "load_report": rep, "env": R.env_record(),
            "decoder": a.decoder, "load_s": round(load_s, 2)}
    with open(a.out + ".meta.json", "a") as f:
        f.write(json.dumps(meta, default=str) + "\n")
    print(f"loaded {a.which} in {load_s:.1f} s; {len(todo)} to do", flush=True)
    n = 0
    per = 0.0
    with open(a.out, "a") as fo:
        for r in todo:
            if a.deadline and time.time() + max(per * 3, 5.0) > a.deadline:
                break
            audio = R.load_wav(r["path"])
            ts = time.time()
            feats, mask = R.features(proc, audio)
            if a.decoder == "generate":
                seq, dur = R.gen(model, feats, mask)
                steps = R.steps_from_generate(seq, dur)
                tokens = [tk for (_, tk, _) in steps if tk != R.BLANK]
                frames = [t for (t, tk, _) in steps if tk != R.BLANK]
                durs = [d for (_, tk, d) in steps if tk != R.BLANK]
                run = mx = 0
                for (_, tk, d) in steps:
                    if tk != R.BLANK and d == 0:
                        run += 1
                        mx = max(mx, run)
                    elif d > 0:
                        run = 0
                eos_hit = 3 in tokens
                n_steps = len(steps)
                enc_frames = valid = None
            else:
                enc, _, valid = R.encode(model, feats, mask)
                res = R.tdt_loop(model, enc, valid)
                tokens, frames, durs = res["tokens"], res["frames"], res["durs"]
                mx, eos_hit, n_steps = res["max_run_same_frame"], res["eos_hit"], res["n_steps"]
                enc_frames = int(enc.shape[0])
            text = R.decode_text(proc, tokens)
            dt = time.time() - ts
            per = dt if per == 0 else 0.8 * per + 0.2 * dt
            fo.write(json.dumps({"id": r["id"], "text": text, "tokens": tokens, "frames": frames, "durs": durs,
                                 "n_steps": n_steps, "max_run_same_frame": mx, "eos_hit": eos_hit,
                                 "enc_frames": enc_frames, "valid_frames": valid,
                                 "audio_s": len(audio) / R.SR, "proc_s": round(dt, 4)}, ensure_ascii=False) + "\n")
            fo.flush()
            n += 1
    left = len(todo) - n
    print(f"{a.out}: +{n} this run, {left} left, {time.time() - t0:.0f} s", flush=True)
    if left == 0:
        open(a.out + ".DONE", "w").write(time.strftime("%H:%M:%S") + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
