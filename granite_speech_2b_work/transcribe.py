#!/usr/bin/env python3
"""Transcribe clips with a granite-speech-5.0-2b .litertlm through LiteRT-LM's Python API (litert-lm 0.19.1).

One engine per process; for each clip a new conversation (greedy: top_k 1, top_p 1, temperature 0;
max_output_tokens 256) and one user message [AudioFile(wav), Text(prompt)], which the bundle's template renders as
`USER: <|audio|>{prompt}\\n ASSISTANT:` (checked on the first clip). Writes one JSONL row per clip: id, text, the
conversation's token count (prompt + reply), the decoded token count and the wall time.

The text of a file on the Mac CPU can change between processes at more than one thread (XNNPACK); compare two files
byte for byte at --threads 1.

  python transcribe.py --bundle granite-speech-5.0-2b.litertlm --clips clips/clips.json --out t.jsonl \
      [--backend cpu|gpu] [--threads 4]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def resp_text(resp):
    if hasattr(resp, "contents"):
        return "".join(getattr(c, "text", "") or "" for c in resp.contents.contents)
    if isinstance(resp, dict):
        c = resp.get("content", "")
        return c if isinstance(c, str) else "".join(i.get("text", "") for i in c if isinstance(i, dict))
    return str(resp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--clips", required=True, help="clips.json (fetch_clips.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="cpu", choices=["cpu", "gpu"], help="language model backend")
    ap.add_argument("--audio-backend", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--prompt", default=C.USER_TEXT)
    ap.add_argument("--max-out", type=int, default=C.MAX_NEW)
    ap.add_argument("--cache-dir", default="", help="engine cache dir (default: <out>.cache)")
    args = ap.parse_args()

    import litert_lm
    from litert_lm import Content, Contents, Message
    from litert_lm import interfaces as I
    from importlib.metadata import version

    def be(name):
        return I.CPU(thread_count=args.threads) if name == "cpu" else I.GPU()

    bundle = Path(args.bundle).resolve()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir or (str(out) + ".cache")).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"litert-lm {version('litert-lm')} | {bundle.name} ({bundle.stat().st_size:,} B) | backend {args.backend} "
          f"audio {args.audio_backend} threads {args.threads}", flush=True)
    t0 = time.time()
    engine = litert_lm.Engine(str(bundle), backend=be(args.backend), audio_backend=be(args.audio_backend),
                              cache_dir=str(cache_dir), enable_benchmark=True)
    print(f"engine loaded in {time.time() - t0:.1f} s", flush=True)
    sampler = litert_lm.SamplerConfig(top_k=1, top_p=1.0, temperature=0.0)
    expected = "USER: " + C.AUDIO_TOKEN + args.prompt + "\n ASSISTANT:"
    first = True
    with open(out, "w") as fo:
        for c in C.load_clips(args.clips):
            path = str(Path(c["abspath"]).resolve())
            msg = Message.user(Contents.of([Content.AudioFile(path), Content.Text(args.prompt)]))
            conv = engine.create_conversation(sampler_config=sampler, max_output_tokens=args.max_out)
            err = None
            try:
                if first:
                    rendered = conv.render_message_to_string(msg)
                    print("rendered:", repr(rendered), flush=True)
                    assert rendered == expected, (rendered, expected)
                    first = False
                t1 = time.perf_counter()
                try:
                    text = resp_text(conv.send_message(msg))
                except Exception as e:  # noqa: BLE001
                    text, err = "", repr(e)
                wall = time.perf_counter() - t1
                tc = conv.token_count
                try:
                    dec = int(conv.get_benchmark_info().last_decode_token_count)
                except Exception:  # noqa: BLE001
                    dec = None
            finally:
                conv.close()
            row = {"id": c["id"], "text": text, "kv_token_count": tc, "decode_tokens": dec, "wall_s": round(wall, 3),
                   "backend": args.backend, "threads": args.threads}
            if err:
                row["error"] = err
            fo.write(json.dumps(row, ensure_ascii=False) + "\n")
            fo.flush()
            print(f"[{c['id']}] {wall:6.2f} s | {text}" + (f" | ERROR {err}" if err else ""), flush=True)
    print("TRANSCRIBE_DONE", flush=True)


if __name__ == "__main__":
    main()
