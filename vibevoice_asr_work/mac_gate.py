#!/usr/bin/env python3
"""LiteRT-LM runtime gate (python API, litert-lm-api 0.16.1): run the packed bundle on the
20 fixtures through the real engine (miniaudio decode → skip-mel framing → audio encoder →
generic multimodal prompt → LM) and report transcripts, corpus WER and timings.

  python mac_gate.py --bundle out/bundle/VibeVoice-ASR-BitNet.litertlm [--backend cpu|gpu]
                     [--audio-backend cpu|gpu] [--no-text] [--limit N] [--tag T]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def resp_text(resp):
    """Text of a conversation response (Message with Contents, or its JSON form)."""
    if hasattr(resp, "contents"):
        return "".join(getattr(c, "text", "") for c in resp.contents.contents)
    if isinstance(resp, dict):
        c = resp.get("content", "")
        return c if isinstance(c, str) else "".join(i.get("text", "") for i in c if isinstance(i, dict))
    return str(resp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--audio-backend", default="cpu")
    ap.add_argument("--no-text", action="store_true", help="send audio only (template default instruction)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-out", type=int, default=256)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tag", default="")
    ap.add_argument("--act-f32", action="store_true", help="activation_data_type=F32 (probe fp16 overflow on GPU)")
    args = ap.parse_args()

    import litert_lm
    from litert_lm import Message, Contents, Content

    def be(name):
        from litert_lm import interfaces as I
        return I.CPU(thread_count=args.threads) if name == "cpu" else I.GPU()

    t0 = time.time()
    kw = {}
    if args.act_f32:
        kw["activation_data_type"] = litert_lm.ActivationDataType.FLOAT32
    engine = litert_lm.Engine(args.bundle, backend=be(args.backend), audio_backend=be(args.audio_backend), **kw)
    load_s = time.time() - t0
    print(f"engine loaded in {load_s:.1f}s  backend={args.backend} audio_backend={args.audio_backend}")
    sampler = litert_lm.SamplerConfig(top_k=1, top_p=1.0, temperature=0.0)

    meta = C.load_meta()
    if args.limit:
        meta = meta[:args.limit]
    rows, errs, words = [], 0, 0
    for i, m in enumerate(meta):
        path = os.path.abspath(os.path.join(C.WORK, m["file"]))
        wav = C.load_wav(path)
        dur = len(wav) / C.SR
        items = [Content.AudioFile(path)]
        if not args.no_text:
            items.append(Content.Text(f"This is a {dur:.2f} seconds audio, please transcribe it."))
        msg = Message.user(Contents.of(items))
        conv = engine.create_conversation(sampler_config=sampler, max_output_tokens=args.max_out)
        try:
            if i == 0:
                print("RENDERED PROMPT:", repr(conv.render_message_to_string(msg)))
            t0 = time.time()
            resp = conv.send_message(msg)
            dt = time.time() - t0
            if i == 0:
                print("response type:", type(resp).__name__)
            hyp = resp_text(resp).strip()
        finally:
            conv.close()
        e, n = C.wer_counts(C.norm_text(m["text"]), C.norm_text(hyp))
        errs += e
        words += n
        rows.append({"id": m["id"], "dur": round(dur, 2), "hyp": hyp, "ref": m["text"], "errs": e,
                     "words": n, "sec": round(dt, 2)})
        print(f"[{m['id']}] {dur:5.1f}s errs={e}/{n} {dt:5.1f}s | {hyp}")
        sys.stdout.flush()
    total_audio = sum(r["dur"] for r in rows)
    total_sec = sum(r["sec"] for r in rows)
    print(f"corpus WER {errs}/{words} = {100*errs/max(words,1):.2f}%   RTF {total_sec/total_audio:.3f} "
          f"({total_sec:.1f}s for {total_audio:.1f}s audio)  load {load_s:.1f}s")
    tag = args.tag or f"{args.backend}_{args.audio_backend}{'_notext' if args.no_text else ''}"
    json.dump({"bundle": args.bundle, "backend": args.backend, "audio_backend": args.audio_backend,
               "no_text": args.no_text, "wer_errs": errs, "ref_words": words, "load_s": round(load_s, 1),
               "rtf": round(total_sec / total_audio, 3), "rows": rows},
              open(os.path.join(C.WORK, f"mac_gate_{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
