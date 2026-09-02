#!/usr/bin/env python3
"""sarashina2.2-instruct (LlamaForCausalLM, SentencePiece vocab 102400) -> .litertlm.

Wraps scripts/export_simple_template.py with the two facts this checkpoint needs:

1. HF tokenizer.json path, NOT the vendor tokenizer.model. litert-torch's
   export_tokenizer copies `tokenizer.model` verbatim whenever the HF tokenizer
   exposes `vocab_file` ending in tokenizer.model. In sarashina's SP model every
   special (`<|user|>`, `<|assistant|>`, `<|system|>`, `</s>`) is a CONTROL piece,
   and a bare sentencepiece Encode never matches CONTROL pieces from text — the
   runtime would tokenize the template's `<|user|>` as `< | user | >` (5 pieces,
   measured in inspect_tokenizer.py). The HF fast tokenizer matches them as
   added tokens (id 9/8/7/2), and its decoder chain has no Strip step
   (Replace + ByteFallback + Fuse), so the Metaspace strip trap does not apply.
   Clearing `vocab_file` makes export_tokenizer fall through to
   save_pretrained(legacy_format=False) -> tokenizer.json.
2. NO_START_TOKEN: tokenizer_config has add_bos_token=false and the official
   chat template emits no <s>; the builder would still write start_token from
   tokenizer.bos_token and the runtime would prepend it. The driver honours
   NO_START_TOKEN=1 (set here unconditionally).

Usage (same positional contract as the driver):
    ~/venvs/ltconv040dev/bin/python sarashina_work/convert_sarashina.py \
        <hf_model_or_dir> <out_dir> templates/sarashina_simple.jinja [recipe]
    env: CACHE (default 4096 here), PREFILL (default ladder 1024..1 here)
"""
import os
import runpy
import sys

os.environ["NO_START_TOKEN"] = "1"
os.environ.setdefault("CACHE", "4096")
os.environ.setdefault("PREFILL", "1024,512,256,128,64,32,16,8,4,2,1")

import transformers  # noqa: E402

_orig = transformers.AutoTokenizer.from_pretrained


def _hf_json_tokenizer(*a, **k):
  tok = _orig(*a, **k)
  vf = getattr(tok, "vocab_file", None)
  if vf:
    tok.vocab_file = None
    print(f"HF_TOKENIZER_JSON: cleared vocab_file ({os.path.basename(vf)}) -> "
          "export_tokenizer will save tokenizer.json instead of copying the SP model")
  return tok


transformers.AutoTokenizer.from_pretrained = _hf_json_tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
driver = os.path.join(ROOT, "scripts", "export_simple_template.py")
sys.argv = [driver] + sys.argv[1:]
runpy.run_path(driver, run_name="__main__")
