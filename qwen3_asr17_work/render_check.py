#!/usr/bin/env python3
"""Render check: the runtime's own render (litert-lm-api 0.17.1 Conversation.render_message_to_string) of the bundle's
jinja vs the strings the checkpoint's chat_template.jinja renders, plus the prompt token count. Adapted from
confucius4_r2t2_work/r3_render_check.py.

  --mode expected  (~/venvs/ltmain0918: transformers) renders the checkpoint's chat_template.jinja for the cases below
                   (apply_chat_template([system(text)?, user(audio)], add_generation_prompt=True) [+ text after the
                   generation prompt]), checks the 'language' case against apply_transcription_request(language=...),
                   tokenizes each with the checkpoint's tokenizer.json (= the bundle's HF tokenizer section) and writes
                   out/render_expected.json. For --prompt litert the expected strings are common.LITERT_TORCH_PROMPT
                   (+ text after it; system messages render nothing).
  --mode runtime   (~/venvs/lt0171run) renders the same cases through the bundle -> out/render_check_<prompt>.json
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
sys.path.insert(0, HERE)
import common  # noqa: E402

CASES = [
    {"case": "audio_only", "system": None, "prefix": None},
    {"case": "language_system", "system": "English", "prefix": None},  # transformers apply_transcription_request(language=)
    {"case": "force_language_prefix", "system": None, "prefix": "language Chinese<asr_text>"},  # qwen-asr force_language
    {"case": "stream_prefix", "system": None, "prefix": "language Chinese<asr_text>開放時間："},
]


def expected():
    import numpy as np
    import transformers
    proc = transformers.AutoProcessor.from_pretrained(common.MODEL_DIR)
    tok = proc.tokenizer
    rows = []
    for c in CASES:
        msgs = ([{"role": "system", "content": [{"type": "text", "text": c["system"]}]}] if c["system"] else []) + \
               [{"role": "user", "content": [{"type": "audio", "audio": "x"}]}]
        s = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False) + (c["prefix"] or "")
        ids = tok(s, add_special_tokens=False)["input_ids"]
        row = {**c, "expected": s, "n_ids_with_one_audio_pad": len(ids), "n_prompt_tokens_390_audio": len(ids) - 1 + 390}
        if c["case"] == "language_system":
            wav = np.zeros(16000, np.float32)
            import torch
            ref = proc.apply_transcription_request(audio=wav, language=c["system"], return_tensors="pt")
            plain = proc(text=s, audio=wav, return_tensors="pt")
            row["equals_apply_transcription_request"] = bool(torch.equal(ref["input_ids"], plain["input_ids"]))
        rows.append(row)
    lit = []
    for c in CASES:
        s = common.LITERT_TORCH_PROMPT + (c["prefix"] or "")
        ids = tok(s, add_special_tokens=False)["input_ids"]
        lit.append({**c, "expected": s, "n_ids_with_one_audio_pad": len(ids),
                    "n_prompt_tokens_390_audio": len(ids) - 1 + 390})
    json.dump({"source": "checkpoint chat_template.jinja via transformers apply_chat_template",
               "official": rows, "litert": lit},
              open(os.path.join(OUT, "render_expected.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(rows, ensure_ascii=False, indent=1))


def runtime(bundle, prompt):
    import litert_lm
    from litert_lm import interfaces
    exp = {r["case"]: r for r in json.load(open(os.path.join(OUT, "render_expected.json")))[prompt]}
    wav = [c for c in common.load_clips() if c["name"] == "zh"][0]["path"]
    cache = os.path.join(OUT, "rt_cache", os.path.basename(os.path.dirname(bundle)))
    os.makedirs(cache, exist_ok=True)
    eng = litert_lm.Engine(bundle, backend=interfaces.CPU(thread_count=4), audio_backend=interfaces.CPU(), cache_dir=cache)
    rows = []
    for c in CASES:
        kw = {"system_message": c["system"]} if c["system"] else {}
        conv = eng.create_conversation(**kw)
        items = [litert_lm.Content.AudioFile(wav)]
        if c["prefix"]:
            items.append(litert_lm.Content.Text(c["prefix"]))
        msg = litert_lm.Message.user(litert_lm.Contents.of(*items))
        got = conv.render_message_to_string(msg)
        row = {"case": c["case"], "runtime_render": got, "expected": exp[c["case"]]["expected"],
               "equal": got == exp[c["case"]]["expected"]}
        if c["case"] == "audio_only":  # prompt token count after one real message (no generation: max 1 token)
            conv.close()
            conv = eng.create_conversation(max_output_tokens=1)
            conv.send_message(msg)
            row["token_count_after_1_output_token"] = conv.token_count
            row["expected_prompt_tokens"] = exp[c["case"]]["n_prompt_tokens_390_audio"]
        rows.append(row)
        conv.close()
    doc = {"bundle": os.path.realpath(bundle), "prompt": prompt, "runtime": litert_lm.__file__, "rows": rows,
           "all_equal": all(r["equal"] for r in rows)}
    json.dump(doc, open(os.path.join(OUT, f"render_check_{prompt}.json"), "w"), ensure_ascii=False, indent=1)
    for r in rows:
        print(r["case"], r["equal"], repr(r["runtime_render"]), r.get("token_count_after_1_output_token", ""),
              r.get("expected_prompt_tokens", ""))
    print("ALL_EQUAL", doc["all_equal"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["expected", "runtime"], required=True)
    ap.add_argument("--bundle", default="")
    ap.add_argument("--prompt", choices=["official", "litert"], default="official", help="the bundle's baked template")
    args = ap.parse_args()
    expected() if args.mode == "expected" else runtime(args.bundle, args.prompt)


if __name__ == "__main__":
    main()
