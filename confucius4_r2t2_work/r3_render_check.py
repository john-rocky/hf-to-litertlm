#!/usr/bin/env python3
"""Render check: the runtime's own render (litert-lm-api 0.17.1 Conversation.render_message_to_string) of the
r3 bundle's jinja vs the strings the vendor code builds, plus a token-count check of the rendered prompt.

  --mode expected  (~/venvs/ltmain0918: transformers + jinja2) renders the vendor chat_template.json exactly as
                   qwen_asr _build_text_prompt does (apply_chat_template([system(context), user(audio)],
                   add_generation_prompt=True) [+ 'language X<asr_text>']) for the cases below, tokenizes each with
                   the bundle's tokenizer (out/hf_layout tokenizer.json, = the bundle's HF_Tokenizer section) and
                   writes out/r3_render_expected.json
  --mode runtime   (~/venvs/lt0171run) renders the same cases through the bundle (and the litert-torch prompt through
                   the chat_template override of r3_runtime.py) and writes out/r3_render_check.json
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
sys.path.insert(0, HERE)

CASES = [
    {"case": "audio_only", "system": None, "prefix": None},
    {"case": "context", "system": "Confucius, Youdao, LiteRT", "prefix": None},
    {"case": "force_language", "system": None, "prefix": "language Chinese<asr_text>"},
    {"case": "stream_prefix", "system": None, "prefix": "language Chinese<asr_text>開放時間："},
]


def expected():
    import transformers
    proc = transformers.AutoProcessor.from_pretrained(os.path.join(OUT, "hf_layout"))
    tok = proc.tokenizer
    tmpl = json.load(open(os.path.join(OUT, "hf_layout", "chat_template.json")))["chat_template"]
    rows = []
    for c in CASES:
        msgs = [{"role": "system", "content": c["system"] or ""},
                {"role": "user", "content": [{"type": "audio", "audio": ""}]}]
        s = proc.apply_chat_template(msgs, chat_template=tmpl, add_generation_prompt=True, tokenize=False)
        if c["prefix"]:
            s = s + c["prefix"]
        ids = tok(s, add_special_tokens=False)["input_ids"]
        rows.append({**c, "expected": s, "n_ids_with_one_audio_pad": len(ids),
                     "n_prompt_tokens_390_audio": len(ids) - 1 + 390})
    vendor_prompt = open(os.path.join(OUT, "vendor_prompt.txt")).read()
    assert rows[0]["expected"] == vendor_prompt, (rows[0]["expected"], vendor_prompt)
    json.dump({"source": "vendor chat_template.json via transformers apply_chat_template", "rows": rows},
              open(os.path.join(OUT, "r3_render_expected.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(rows, ensure_ascii=False, indent=1))


def runtime(bundle, prompt="vendor"):
    import litert_lm
    from litert_lm import interfaces
    import r3_runtime
    exp = {r["case"]: r for r in json.load(open(os.path.join(OUT, "r3_render_expected.json")))["rows"]}
    if prompt == "litert":
        # ship template: the litert-torch prompt; system messages render nothing; text items go after 'assistant\n'
        import common as _c
        exp = {c["case"]: {"expected": _c.LITERT_TORCH_PROMPT + (c["prefix"] or "")} for c in CASES}
    wav = os.path.expanduser("~/code/coreai/_funasr_nano/fixtures/examples/zh.wav")
    if not os.path.exists(wav):
        import common
        wav = [c for c in common.load_clips() if c["name"] == "zh"][0]["path"]
    eng = litert_lm.Engine(bundle, backend=interfaces.CPU(thread_count=4), audio_backend=interfaces.CPU(),
                           cache_dir=os.path.join(OUT, "r3_rt_cache", os.path.basename(os.path.dirname(bundle))))
    rows = []
    for c in CASES:
        kw = {"system_message": c["system"]} if c["system"] else {}
        conv = eng.create_conversation(**kw)
        items = [litert_lm.Content.AudioFile(wav)]
        if c["prefix"]:
            items.append(litert_lm.Content.Text(c["prefix"]))
        got = conv.render_message_to_string(litert_lm.Message.user(litert_lm.Contents.of(*items)))
        rows.append({"case": c["case"], "runtime_render": got, "expected": exp[c["case"]]["expected"],
                     "equal": got == exp[c["case"]]["expected"]})
        conv.close()
    conv = eng.create_conversation(chat_template=r3_runtime.LITERT_TEMPLATE) if prompt == "vendor" else \
        eng.create_conversation()
    got = conv.render_message_to_string(litert_lm.Message.user(litert_lm.Contents.of(
        litert_lm.Content.AudioFile(wav))))
    import common
    rows.append({"case": "litert_torch_prompt_override", "runtime_render": got, "expected": common.LITERT_TORCH_PROMPT,
                 "equal": got == common.LITERT_TORCH_PROMPT})
    conv.close()
    doc = {"bundle": os.path.realpath(bundle), "prompt": prompt, "runtime": litert_lm.__file__, "rows": rows,
           "all_equal": all(r["equal"] for r in rows)}
    name = "r3_render_check.json" if prompt == "vendor" else "r3_render_check_ship.json"
    json.dump(doc, open(os.path.join(OUT, name), "w"), ensure_ascii=False, indent=1)
    for r in rows:
        print(r["case"], r["equal"], repr(r["runtime_render"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["expected", "runtime"], required=True)
    ap.add_argument("--bundle", default=os.path.join(OUT, "export", "c4r3_30s_C", "model.litertlm"))
    ap.add_argument("--prompt", choices=["vendor", "litert"], default="vendor", help="the bundle's baked template")
    args = ap.parse_args()
    expected() if args.mode == "expected" else runtime(args.bundle, args.prompt)


if __name__ == "__main__":
    main()
