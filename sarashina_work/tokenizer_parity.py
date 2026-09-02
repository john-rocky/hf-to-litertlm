#!/usr/bin/env python3
"""Engine-vs-HF tokenizer parity for a sarashina2.2 bundle (HF_Tokenizer_Zlib section).
Two steps because the runtime venv has no transformers:

  ~/venvs/ltconv040dev/bin/python sarashina_work/tokenizer_parity.py hf <hf_model_or_dir> probes.json
  ~/venvs/lt0160run/bin/python    sarashina_work/tokenizer_parity.py engine <bundle> probes.json

Probes: specials as text, Japanese, mixed, emoji/rare kanji (ByteFallback), Latin-1, whitespace.
"""
import json
import sys

PROBES = [
    "<|user|>こんにちは。あなたの名前を教えて</s><|assistant|>",
    "<|system|>あなたは親切なアシスタントです。</s><|user|>日本の首都は？</s><|assistant|>",
    "日本の首都はどこですか？ 簡潔に答えてください。",
    "17 + 25 = 42. The capital of France is Paris.",
    "😀 鬱 𠮷野家 café naïve résumé ° × · Ł",
    "  leading spaces and\ttabs\nnewlines  ",
    "①②③ ㈱ 〒100-0001 ～ 〜 ‐ − ー",
]
mode, target, path = sys.argv[1], sys.argv[2], sys.argv[3]
if mode == "hf":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(target)
    json.dump({p: tok(p, add_special_tokens=False)["input_ids"] for p in PROBES},
              open(path, "w"), ensure_ascii=False, indent=0)
    print("wrote", path)
    sys.exit(0)
from litert_lm import engine as engine_lib
ref = json.load(open(path))
eng = engine_lib.Engine(target, max_num_tokens=512)
fails = 0
for p, hf_ids in ref.items():
    eng_ids = list(eng.tokenize(p))
    same = hf_ids == eng_ids
    fails += not same
    print(("OK  " if same else "DIFF"), repr(p)[:60], "hf", len(hf_ids), "eng", len(eng_ids))
    if not same:
        print("   hf :", hf_ids[:40]); print("   eng:", eng_ids[:40])
print("TOKENIZER_PARITY", "PASS" if fails == 0 else f"FAIL ({fails}/{len(ref)})")
sys.exit(1 if fails else 0)
