#!/usr/bin/env python3
"""Forced language, the two ways the upstream code passes it, on the 9 Mandarin FLEURS clips the eager reference
tags `English` with the official prompt (+ the zh / ja example clips):

  system   transformers apply_transcription_request(audio, language="Chinese"): the language name as the system text
           -> runtime: create_conversation(system_message="Chinese") with the bundle's template
  prefix   qwen-asr force_language: 'language Chinese<asr_text>' after the generation prompt
           -> runtime: a text item after the audio item

  --side eager    (~/venvs/ltmain0918) transformers 5.14.1 fp32 greedy -> out/force_language_eager.jsonl
  --side runtime  (~/venvs/lt0171run) litert-lm-api 0.17.1 CPU greedy -> out/force_language_runtime.jsonl
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402

LANG = {"cmn_hans_cn": "Chinese", "ja_jp": "Japanese", "en_us": "English"}


def clip_list():
    ref = [json.loads(l) for l in open(os.path.join(common.OUT, "ref_eager_official.jsonl")) if '"clip"' in l]
    names = [r["clip"] for r in ref if r["config"] == "cmn_hans_cn" and r["language"] != "Chinese"]
    clips = {c["name"]: c for c in common.load_clips()}
    out = [dict(clips[n], lang="Chinese") for n in names]
    out += [dict(clips["zh"], lang="Chinese"), dict(clips["ja"], lang="Japanese")]
    return out


def eager(out_path):
    import torch
    import transformers
    torch.set_num_threads(8)
    model = transformers.Qwen3ASRForConditionalGeneration.from_pretrained(common.MODEL_DIR, dtype=torch.float32).eval()
    proc = transformers.AutoProcessor.from_pretrained(common.MODEL_DIR)
    base = proc.apply_chat_template([{"role": "user", "content": [{"type": "audio", "audio": "x"}]}],
                                    add_generation_prompt=True, tokenize=False)
    with open(out_path, "w") as f:
        for c in clip_list():
            wav = common.read_wav(c["path"])
            for mode in ("system", "prefix"):
                if mode == "system":
                    inputs = proc.apply_transcription_request(audio=wav, language=c["lang"], return_tensors="pt")
                else:
                    inputs = proc(text=base + f"language {c['lang']}<asr_text>", audio=wav, return_tensors="pt")
                with torch.no_grad():
                    out = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=512)
                n = inputs["input_ids"].shape[1]
                raw = proc.batch_decode(out[:, n:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                row = {"clip": c["name"], "config": c["config"], "mode": mode, "lang": c["lang"], "raw": raw}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(json.dumps(row, ensure_ascii=False)[:200], flush=True)


def runtime(bundle, out_path):
    import litert_lm
    from litert_lm import interfaces
    import runtime_gate
    cache = os.path.join(common.OUT, "rt_cache", os.path.basename(os.path.dirname(bundle)))
    eng = litert_lm.Engine(bundle, backend=interfaces.CPU(thread_count=4), audio_backend=interfaces.CPU(thread_count=4),
                           cache_dir=cache)
    sampler = interfaces.SamplerConfig(top_k=1, top_p=1.0, temperature=0.0)
    with open(out_path, "w") as f:
        for c in clip_list():
            for mode in ("system", "prefix"):
                items = [litert_lm.Content.AudioFile(c["path"])]
                kw = {"sampler_config": sampler}
                if mode == "system":
                    kw["system_message"] = c["lang"]
                else:
                    items.append(litert_lm.Content.Text(f"language {c['lang']}<asr_text>"))
                conv = eng.create_conversation(**kw)
                msg = litert_lm.Message.user(litert_lm.Contents.of(*items))
                render = conv.render_message_to_string(msg)
                raw = runtime_gate.response_text(conv.send_message(msg))
                conv.close()
                row = {"clip": c["name"], "config": c["config"], "mode": mode, "lang": c["lang"], "render": render,
                       "raw": raw}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(json.dumps({k: v for k, v in row.items() if k != "render"}, ensure_ascii=False)[:200], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=["eager", "runtime"], required=True)
    ap.add_argument("--bundle", default=os.path.join(common.OUT, "export", "q17_30s_C_off", "model.litertlm"))
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.side == "eager":
        eager(os.path.join(common.OUT, "force_language_eager.jsonl"))
    else:
        runtime(args.bundle, args.out or os.path.join(common.OUT, "force_language_runtime.jsonl"))


if __name__ == "__main__":
    main()
