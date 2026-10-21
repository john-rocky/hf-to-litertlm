#!/usr/bin/env python3
"""Smoke for the references: build B (strict load) and O, transcribe the first 3 S32 utterances in mode (i) with
generate() and with tdt_loop(), compare the token ids. -> results/smoke_ref.json (+ .DONE when both agree)."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import redux_ref as R  # noqa: E402

OUT = "results/smoke_ref.json"


def main():
    R.set_threads(int(os.environ.get("REF_THREADS", "8")))
    rows = R.read_jsonl("data/sets/librispeech-test-clean/manifest-S32.jsonl")[:3]
    proc = R.processor()
    rep = {"env": R.env_record(), "models": {}}
    ok = True
    for which in ("B", "O"):
        t0 = time.time()
        model, load = R.build(which)
        load["build_s"] = round(time.time() - t0, 2)
        utts = []
        for r in rows:
            audio = R.load_wav(r["path"])
            feats, mask = R.features(proc, audio)
            ts = time.time()
            seq, dur = R.gen(model, feats, mask)
            tg = time.time() - ts
            steps = R.steps_from_generate(seq, dur)
            toks_g = [tk for (_, tk, _) in steps if tk != R.BLANK]
            ts = time.time()
            enc, _, valid = R.encode(model, feats, mask)
            res = R.tdt_loop(model, enc, valid)
            tl = time.time() - ts
            same = toks_g == res["tokens"] and steps == [tuple(s) for s in res["steps"]]
            ok &= same
            utts.append({"id": r["id"], "ref": r["text"], "text_generate": R.decode_text(proc, toks_g),
                         "text_loop": R.decode_text(proc, res["tokens"]), "tokens_equal": toks_g == res["tokens"],
                         "steps_equal": same, "n_tokens": len(toks_g), "mel_frames": int(feats.shape[1]),
                         "mask_frames": int(mask.sum()), "enc_frames": int(enc.shape[0]), "valid_enc": valid,
                         "audio_s": len(audio) / R.SR, "generate_s": round(tg, 3), "loop_s": round(tl, 3)})
            print(which, r["id"], "equal" if same else "DIFF", R.decode_text(proc, toks_g)[:90], flush=True)
        rep["models"][which] = {"load": load, "utts": utts}
        del model
    rep["loop_equals_generate"] = ok
    json.dump(rep, open(OUT, "w"), indent=1, ensure_ascii=False, default=str)
    # both references built and decoded; the loop-vs-generate equality is a flag read by the worker
    open(OUT + ".DONE", "w").write(time.strftime("%H:%M:%S") + "\n")
    print("SMOKE PASS", "loop_equals_generate=" + str(ok))
    return 0


if __name__ == "__main__":
    sys.exit(main())
