#!/usr/bin/env python3
"""Round 3 B: two ways to send the 60.175 s long_en clip that is cut at one runtime window (make_long_split.py:
long_en_p1.wav = 30.24 s, long_en_p2.wav = 29.935 s), litert-lm 0.17.1 python API (~/venvs/lt0171run), greedy,
max_output_tokens 512 per turn, audio-only user messages (template default instruction):

  split_conv  p1 and p2 each in its OWN conversation; the two texts joined with a space
  two_turns   ONE conversation, user turn 1 = p1, user turn 2 = p2 (audio only); the two responses joined. The
              rendered string of each turn is printed and stored (turn 2 = the new text after turn 1's history).

WER against the three references joined (143 words, as mac_gate.py --mode long). Writes mac_gate_<tag>.json.

  python long_gate.py --bundle out/bundle/Fun-ASR-Nano-2512.litertlm --method split_conv --tag r11_long_split_conv
"""
import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
sys.path.insert(0, HERE)
from mac_gate import (EXPECTED_RENDER, LONG_PARTS, loop_suspect, norm_text, postprocess, resp_text,  # noqa: E402
                      wav_seconds, wer_counts)

PARTS = ["long_en_p1", "long_en_p2"]
TURN2_EXPECTED = "<|im_start|>user\n语音转写：<|AUDIO|><|im_end|>\n<|im_start|>assistant\n"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--method", required=True, choices=["split_conv", "two_turns"])
    ap.add_argument("--backend", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-out", type=int, default=512)
    ap.add_argument("--cache-dir", default="", help="default out/runtime_cache/<bundle stem>")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    meta = {m["id"]: m for m in json.load(open(os.path.join(HERE, "fixtures", "meta.json")))}
    ref = " ".join(meta[i]["text"] for i in LONG_PARTS)
    paths = [os.path.join(OUT, "fixtures", f"{p}.wav") for p in PARTS]

    import litert_lm
    from litert_lm import Content, Contents, Message
    from litert_lm import interfaces as I
    from importlib.metadata import version

    def be(name):
        return I.CPU(thread_count=args.threads) if name == "cpu" else I.GPU()

    bundle = os.path.abspath(args.bundle)
    stem = os.path.splitext(os.path.basename(bundle))[0]
    cache_dir = os.path.abspath(args.cache_dir or os.path.join(OUT, "runtime_cache", stem))
    os.makedirs(cache_dir, exist_ok=True)
    print(f"litert-lm {version('litert-lm')} | bundle {bundle} ({os.path.getsize(bundle):,} B) | backend {args.backend} "
          f"| method {args.method} | cache {cache_dir}", flush=True)
    t0 = time.time()
    engine = litert_lm.Engine(bundle, backend=be(args.backend), audio_backend=be("cpu"), cache_dir=cache_dir)
    load_s = time.time() - t0
    print(f"engine loaded in {load_s:.2f} s", flush=True)
    sampler = litert_lm.SamplerConfig(top_k=1, top_p=1.0, temperature=0.0)

    turns = []

    def one(conv, k, path):
        msg = Message.user(Contents.of([Content.AudioFile(os.path.abspath(path))]))
        rendered = conv.render_message_to_string(msg)
        print(f"RENDERED turn/part {k + 1}: {rendered!r}", flush=True)
        t1 = time.time()
        raw = resp_text(conv.send_message(msg))
        wall = time.time() - t1
        text = postprocess(raw)
        tc = conv.token_count
        turns.append({"part": PARTS[k], "file": os.path.relpath(path, HERE), "audio_s": round(wav_seconds(path), 3),
                      "rendered": rendered, "text_raw": raw, "text": text, "wall_s": round(wall, 3),
                      "token_count_after": tc, "loop_suspect": loop_suspect(text)})
        print(f"[{PARTS[k]}] {wav_seconds(path):6.2f} s audio | {wall:6.2f} s | tok {tc} | {text!r}", flush=True)

    if args.method == "split_conv":
        for k, p in enumerate(paths):
            conv = engine.create_conversation(sampler_config=sampler, max_output_tokens=args.max_out)
            try:
                one(conv, k, p)
            finally:
                conv.close()
        render_ok = all(t["rendered"] == EXPECTED_RENDER for t in turns)
    else:
        conv = engine.create_conversation(sampler_config=sampler, max_output_tokens=args.max_out)
        try:
            for k, p in enumerate(paths):
                one(conv, k, p)
        finally:
            conv.close()
        render_ok = turns[0]["rendered"] == EXPECTED_RENDER and turns[1]["rendered"] == TURN2_EXPECTED
    joined = " ".join(t["text"].strip() for t in turns)
    e, n = wer_counts(norm_text(ref), norm_text(joined))
    doc = {"tag": args.tag, "method": args.method, "bundle": os.path.relpath(bundle, HERE),
           "bundle_bytes": os.path.getsize(bundle), "bundle_sha256": sha256(bundle), "backend": args.backend,
           "audio_backend": "cpu", "threads": args.threads, "max_output_tokens": args.max_out,
           "sampler": {"top_k": 1, "top_p": 1.0, "temperature": 0.0}, "litert_lm": version("litert-lm"),
           "engine_load_s": round(load_s, 2), "render_as_expected": render_ok,
           "expected_render_turn1": EXPECTED_RENDER, "expected_render_turn2": TURN2_EXPECTED if args.method == "two_turns" else None,
           "turns": turns, "joined_text": joined, "reference_concat": ref, "wer": [e, n],
           "wall_s": round(sum(t["wall_s"] for t in turns), 3), "audio_s": round(sum(t["audio_s"] for t in turns), 3),
           "empty": sum(1 for t in turns if not t["text"].strip()), "loop_suspect": sum(t["loop_suspect"] for t in turns)}
    doc["rtf"] = round(doc["wall_s"] / doc["audio_s"], 4)
    with open(os.path.join(HERE, f"mac_gate_{args.tag}.json"), "w") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: doc[k] for k in ["method", "wer", "render_as_expected", "empty", "loop_suspect", "rtf",
                                          "engine_load_s"]}, ensure_ascii=False), flush=True)
    print("JOINED:", joined, flush=True)
    print("LONG_GATE_DONE", flush=True)


if __name__ == "__main__":
    main()
