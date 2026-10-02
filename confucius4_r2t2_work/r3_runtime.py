#!/usr/bin/env python3
"""Runtime driver: a generic_model Confucius4 bundle through the RELEASED LiteRT-LM Python API
(litert-lm-api 0.17.1, ~/venvs/lt0171run; pure python, no numpy).

Per clip: a new conversation (greedy: top_k 1, top_p 1, temperature 0), one user message with the clip as an audio
item (Content.AudioFile(<wav path>)), optional text items (= the assistant prefix in the bundle's template).
Records the runtime's own render of the message, the reply text, the language tag / text split like
qwen_asr parse_asr_output (common.split_raw), wall seconds, token_count.

  --prompt vendor   the bundle's jinja (vendor chat_template string: empty system turn + user audio + assistant)
  --prompt litert   chat_template override = litert-torch Qwen3AsrProcessor._PROMPT with the same placeholder
                    (the prompt the (C) bundle baked into its encoder output, for a comparison with it)

  ~/venvs/lt0171run/bin/python r3_runtime.py --bundle out/export/c4r3_30s_C/model.litertlm --clips examples \
      --prompt vendor --out out/r3_rt/c4r3_30s_C_ex5_vendor.jsonl
"""
import argparse
import json
import os
import resource
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402

VENDOR_PROMPT = ("<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>"
                 "<|im_end|>\n<|im_start|>assistant\n")
# litert-torch Qwen3AsrProcessor._PROMPT as a template: no system turn, no newlines after the role names.
LITERT_TEMPLATE = (
    "{%- for m in messages -%}{%- if m.role == 'user' -%}{{- '<|im_start|>user<|audio_start|>' -}}"
    "{%- for c in m.content -%}{%- if c.type == 'audio' -%}{{- '<|audio_pad|>' -}}{%- endif -%}{%- endfor -%}"
    "{{- '<|audio_end|><|im_end|>' -}}{%- endif -%}{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' -}}{%- endif -%}")


def response_text(resp):
    # litert-lm-api 0.15-0.17.1 return a dict of content parts; later builds return a Message whose str() is its text
    if type(resp) is dict:
        return "".join(p.get("text", "") for p in resp.get("content", []) if isinstance(p, dict))
    return str(resp)


def select(spec):
    clips = common.load_clips()
    if spec == "all":
        return clips
    if spec == "examples":
        return [c for c in clips if c["config"] is None]
    if spec.startswith("names:"):
        names = spec[6:].split(",")
        return [c for c in clips if c["name"] in names]
    if spec.startswith("shortest:"):
        return sorted(clips, key=lambda c: c["num_samples"])[:int(spec.split(":")[1])]
    if spec.startswith("first:"):  # first N clips of each FLEURS config in meta order + nothing else
        n = int(spec.split(":")[1])
        out = []
        for cfg in ("en_us", "cmn_hans_cn", "ja_jp"):
            out += [c for c in clips if c["config"] == cfg][:n]
        return out
    raise ValueError(spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--clips", default="examples")
    ap.add_argument("--prompt", choices=["vendor", "litert", "bundle"], default="bundle",
                    help="bundle (= vendor on the first r3 build): the bundle's own jinja; litert: chat_template override")
    ap.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--audio_backend", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--log_verbose", action="store_true", help="litert_lm.set_min_log_severity(VERBOSE) before the Engine")
    ap.add_argument("--max_output_tokens", type=int, default=0)
    ap.add_argument("--cache_dir", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import litert_lm
    from litert_lm import interfaces
    if args.log_verbose:
        litert_lm.set_min_log_severity(litert_lm.LogSeverity.VERBOSE)
    clips = select(args.clips)
    cache = args.cache_dir or os.path.join(HERE, "out", "r3_rt_cache", os.path.basename(os.path.dirname(args.bundle)))
    os.makedirs(cache, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    backend = interfaces.CPU(thread_count=args.threads) if args.backend == "cpu" else interfaces.GPU()
    t0 = time.time()
    audio_backend = interfaces.CPU(thread_count=args.threads) if args.audio_backend == "cpu" else interfaces.GPU()
    eng = litert_lm.Engine(args.bundle, backend=backend, audio_backend=audio_backend, cache_dir=cache)
    load_s = time.time() - t0
    sampler = interfaces.SamplerConfig(top_k=1, top_p=1.0, temperature=0.0)
    tot_audio = tot_wall = 0.0
    n_err = 0
    with open(args.out, "w") as f:
        f.write(json.dumps({"type": "header", "runtime": "litert-lm-api " + getattr(litert_lm, "__version__", "?"),
                            "litert_lm_file": litert_lm.__file__, "bundle": os.path.realpath(args.bundle),
                            "bundle_bytes": os.path.getsize(args.bundle), "backend": args.backend,
                            "audio_backend": args.audio_backend,
                            "threads": args.threads, "prompt": args.prompt, "clips": args.clips,
                            "sampler": "top_k 1, top_p 1.0, temperature 0.0",
                            "max_output_tokens": args.max_output_tokens or None, "load_seconds": round(load_s, 3)},
                           ensure_ascii=False) + "\n")
        for c in clips:
            row = {"clip": c["name"], "config": c["config"], "audio_seconds": round(c["num_samples"] / 16000, 4)}
            try:
                kw = {"sampler_config": sampler}
                if args.prompt == "litert":
                    kw["chat_template"] = LITERT_TEMPLATE
                if args.max_output_tokens:
                    kw["max_output_tokens"] = args.max_output_tokens
                conv = eng.create_conversation(**kw)
                msg = litert_lm.Message.user(litert_lm.Contents.of(litert_lm.Content.AudioFile(c["path"])))
                row["render"] = conv.render_message_to_string(msg)
                t1 = time.time()
                resp = conv.send_message(msg)
                row["seconds"] = round(time.time() - t1, 4)
                raw = response_text(resp)
                lang, text = common.split_raw(raw)
                row.update({"raw": raw, "text": text, "language": lang, "token_count": conv.token_count})
                tot_audio += row["audio_seconds"]
                tot_wall += row["seconds"]
                conv.close()
            except Exception as e:  # noqa: BLE001
                n_err += 1
                row["error"] = f"{type(e).__name__}: {e}"
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(c["name"], row.get("seconds"), row.get("language"), (row.get("text") or row.get("error", ""))[:80],
                  flush=True)
        f.write(json.dumps({"type": "footer", "clips": len(clips), "errors": n_err,
                            "audio_seconds": round(tot_audio, 3), "wall_seconds": round(tot_wall, 3),
                            "rtfx": round(tot_audio / tot_wall, 3) if tot_wall else None,
                            "rtf": round(tot_wall / tot_audio, 4) if tot_audio else None,
                            "peak_rss_bytes_ru_maxrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}) + "\n")


if __name__ == "__main__":
    main()
