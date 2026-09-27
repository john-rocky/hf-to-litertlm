#!/usr/bin/env python3
"""Runtime gate for one of the model's own instructions sent as a text item (the card's prompt table), e.g. the no-ITN
instruction 语音转写，不进行文本规整：. Same loop as mac_gate.py (one engine, one conversation per clip, greedy, max 512
output tokens, funasr post-processing) with a Text item before the audio in the same user message. The first render is
asserted to be funasr's prompt for that instruction; texts are compared with a funasr reference file made with the same
instruction (oracle_transcripts_en_noitn.json: itn=False, the 20 LibriSpeech clips).

  ~/venvs/lt0171run/bin/python prompt_gate.py --text 语音转写，不进行文本规整： --oracle oracle_transcripts_en_noitn.json --tag r13_noitn_cpu
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mac_gate import loop_suspect, norm_text, postprocess, resp_text, sha256, wav_seconds, wer_counts  # noqa: E402

SYS = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
TAIL = "<|AUDIO|><|im_end|>\n<|im_start|>assistant\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=os.path.join(HERE, "out", "bundle", "Fun-ASR-Nano-2512.litertlm"))
    ap.add_argument("--backend", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--text", required=True)
    ap.add_argument("--oracle", required=True, help="funasr reference JSON made with the same instruction (rows: id, text)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    meta = {m["id"]: m for m in json.load(open(os.path.join(HERE, "fixtures", "meta.json")))}
    oracle = {r["id"]: r["text"] for r in json.load(open(os.path.join(HERE, args.oracle)))["rows"]}
    import litert_lm
    from importlib.metadata import version
    from litert_lm import Content, Contents, Message
    from litert_lm import interfaces as I
    be = I.CPU(thread_count=args.threads) if args.backend == "cpu" else I.GPU()
    bundle = os.path.abspath(args.bundle)
    cache = os.path.join(HERE, "out", "runtime_cache", os.path.splitext(os.path.basename(bundle))[0])
    t0 = time.time()
    engine = litert_lm.Engine(bundle, backend=be, audio_backend=I.CPU(thread_count=args.threads), cache_dir=cache)
    load_s = time.time() - t0
    sampler = litert_lm.SamplerConfig(top_k=1, top_p=1.0, temperature=0.0)
    rows, render = [], None
    for k, fid in enumerate(oracle):
        path = os.path.join(HERE, meta[fid]["file"])
        msg = Message.user(Contents.of([Content.Text(args.text), Content.AudioFile(os.path.abspath(path))]))
        conv = engine.create_conversation(sampler_config=sampler, max_output_tokens=512)
        try:
            if k == 0:
                got = conv.render_message_to_string(msg)
                render = {"rendered": got, "expected": SYS + args.text + TAIL, "equal": got == SYS + args.text + TAIL}
                assert render["equal"], render
            t1 = time.time()
            text = postprocess(resp_text(conv.send_message(msg)))
            wall = time.time() - t1
        finally:
            conv.close()
        e, n = wer_counts(norm_text(meta[fid]["text"]), norm_text(text))
        eo, _ = wer_counts(norm_text(meta[fid]["text"]), norm_text(oracle[fid]))
        rows.append({"id": fid, "text": text, "oracle_text": oracle[fid], "oracle_equal": text.strip() == oracle[fid].strip(),
                     "wer": [e, n], "oracle_wer": [eo, n], "wall_s": round(wall, 3), "audio_s": round(wav_seconds(path), 3),
                     "loop_suspect": loop_suspect(text)})
        print(f"[{fid}] {'==' if rows[-1]['oracle_equal'] else '!='} {text[:110]!r}", flush=True)
    summ = {"n": len(rows), "oracle_match": sum(r["oracle_equal"] for r in rows),
            "wer_en": [sum(r["wer"][0] for r in rows), sum(r["wer"][1] for r in rows)],
            "oracle_wer_en": [sum(r["oracle_wer"][0] for r in rows), sum(r["wer"][1] for r in rows)],
            "empty": sum(1 for r in rows if not r["text"].strip()), "loop_suspect": sum(r["loop_suspect"] for r in rows),
            "engine_load_s": round(load_s, 2)}
    doc = {"tag": args.tag, "text_item": args.text, "oracle_file": args.oracle, "bundle": os.path.relpath(bundle, HERE),
           "bundle_sha256": sha256(bundle), "backend": args.backend, "audio_backend": "cpu", "litert_lm": version("litert-lm-api"),
           "render_check": render, "summary": summ, "rows": rows}
    json.dump(doc, open(os.path.join(HERE, f"prompt_gate_{args.tag}.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(summ), flush=True)
    print("PROMPT_GATE_DONE", flush=True)


if __name__ == "__main__":
    main()
