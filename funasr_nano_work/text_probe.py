#!/usr/bin/env python3
"""Round 3 A1: a text-only prompt (no audio item) through the bundle's LM, to separate "the LM fails on the GPU" from
"the audio embeddings break the GPU LM". litert-lm 0.17.1 python API (~/venvs/lt0171run), greedy, one conversation.
Runtime log lines go to stderr; count `Invalid decode and sample result` in the captured log.

  python text_probe.py --bundle out/bundle/Fun-ASR-Nano-2512.litertlm --backend gpu --tag r3a1_text_gpu
"""
import argparse
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
PROMPT = "Write one short English sentence about the sea."


def resp_text(resp):
    if hasattr(resp, "contents"):
        return "".join(getattr(c, "text", "") for c in resp.contents.contents)
    if isinstance(resp, dict):
        c = resp.get("content", "")
        return c if isinstance(c, str) else "".join(i.get("text", "") for i in c if isinstance(i, dict))
    return str(resp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--backend", default="gpu", choices=["cpu", "gpu"])
    ap.add_argument("--audio-backend", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--act-f32", action="store_true", help="engine activation_data_type=FLOAT32")
    ap.add_argument("--act-f16", action="store_true",
                    help="engine activation_data_type=FLOAT16 (control: overrides a bundle's prefer_activation_type)")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--max-out", type=int, default=128)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--cache-dir", default="", help="default out/runtime_cache/<bundle stem>")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

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
    assert not (args.act_f32 and args.act_f16)
    kw = ({"activation_data_type": litert_lm.ActivationDataType.FLOAT32} if args.act_f32 else
          {"activation_data_type": litert_lm.ActivationDataType.FLOAT16} if args.act_f16 else {})
    print(f"litert-lm {version('litert-lm')} | bundle {bundle} ({os.path.getsize(bundle):,} B) | backend {args.backend} "
          f"audio {args.audio_backend} | act_f32 {args.act_f32} | act_f16 {args.act_f16} | cache {cache_dir}", flush=True)
    t0 = time.time()
    engine = litert_lm.Engine(bundle, backend=be(args.backend), audio_backend=be(args.audio_backend),
                              cache_dir=cache_dir, **kw)
    load_s = time.time() - t0
    print(f"engine loaded in {load_s:.2f} s", flush=True)
    sampler = litert_lm.SamplerConfig(top_k=1, top_p=1.0, temperature=0.0)
    msg = Message.user(Contents.of([Content.Text(args.prompt)]))
    conv = engine.create_conversation(sampler_config=sampler, max_output_tokens=args.max_out)
    try:
        rendered = conv.render_message_to_string(msg)
        print("RENDERED:", repr(rendered), flush=True)
        t1 = time.time()
        text = resp_text(conv.send_message(msg))
        wall = time.time() - t1
        tc = conv.token_count
    finally:
        conv.close()
    bangs = len(text) > 0 and set(text) == {"!"}
    print(f"RESPONSE ({wall:.2f} s, token_count {tc}, only '!': {bangs}): {text!r}", flush=True)
    doc = {"tag": args.tag, "bundle": os.path.relpath(bundle, HERE), "bundle_bytes": os.path.getsize(bundle),
           "backend": args.backend, "audio_backend": args.audio_backend, "act_f32": args.act_f32, "act_f16": args.act_f16,
           "litert_lm": version("litert-lm"), "prompt": args.prompt, "rendered": rendered, "text": text,
           "only_bang": bangs, "chars": len(text), "wall_s": round(wall, 3), "token_count": tc,
           "engine_load_s": round(load_s, 2), "max_output_tokens": args.max_out}
    with open(os.path.join(HERE, f"text_probe_{args.tag}.json"), "w") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    print("PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
