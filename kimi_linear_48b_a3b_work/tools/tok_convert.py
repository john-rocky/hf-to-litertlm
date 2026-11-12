#!/usr/bin/env python3
"""tok_convert.py OUT_DIR - the author's tiktoken tokenizer (tok/ next to tools/ = tokenization_kimi.py +
tiktoken.model + tokenizer_config.json at e1df551a) as a Hugging Face tokenizer.json, the form LiteRT-LM's
tokenizer section reads.

Built with Transformers 5.22.2's own TikTokenConverter (convert_slow_tokenizer.py 1914-1991): byte-level BPE whose
vocab is tiktoken's ranks (bytes through GPT-2's bytes_to_unicode) and whose merges are derived from the ranks,
ignore_merges = True (a pre-token that is itself a vocab entry is one token, as tiktoken does), pre-tokenizer =
Split(the author's pat_str, isolated) + ByteLevel(no prefix space, no regex), decoder ByteLevel. The pattern is read
from the author's class (TikTokenTokenizer.pat_str), not retyped. Special tokens = the author's 258 (ids 163584 ..
163841: the 17 of added_tokens_decoder + <|reserved_token_N|>), added in id order and checked to land on their ids.
No normalizer, no BOS / EOS added (the author's encode adds none). Writes OUT_DIR/tokenizer.json and
OUT_DIR/tok_convert.json (versions, counts, sha256).
"""
import hashlib
import importlib.metadata as md
import json
import os
import sys

from transformers import AutoTokenizer
from transformers.convert_slow_tokenizer import TikTokenConverter

K = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
  out = sys.argv[1]
  os.makedirs(out, exist_ok=True)
  src = os.path.join(K, "tok")
  author = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
  specials = sorted(author.special_tokens.items(), key=lambda kv: kv[1])  # [(str, id)] in id order
  base = len(author.model._mergeable_ranks)  # pylint: disable=protected-access
  assert [i for _, i in specials] == list(range(base, base + len(specials))), "special ids not contiguous"
  conv = TikTokenConverter(vocab_file=os.path.join(src, "tiktoken.model"), pattern=type(author).pat_str,
                           extra_special_tokens=[s for s, _ in specials])
  tk = conv.converted()
  for s, i in specials:
    got = tk.token_to_id(s)
    assert got == i, (s, i, got)
  path = os.path.join(out, "tokenizer.json")
  tk.save(path)
  raw = open(path, "rb").read()
  rec = {"source": "moonshotai/Kimi-Linear-48B-A3B-Instruct@e1df551a (tiktoken.model sha256 "
                   + hashlib.sha256(open(os.path.join(src, "tiktoken.model"), "rb").read()).hexdigest() + ")",
         "pattern": type(author).pat_str, "base_tokens": base, "special_tokens": len(specials),
         "vocab_size_with_added": tk.get_vocab_size(with_added_tokens=True),
         "versions": {n: md.version(n) for n in ("transformers", "tokenizers", "tiktoken")},
         "tokenizer_json_sha256": hashlib.sha256(raw).hexdigest(), "tokenizer_json_bytes": len(raw)}
  json.dump(rec, open(os.path.join(out, "tok_convert.json"), "w"), indent=1, ensure_ascii=False)
  print("TOK_CONVERT_DONE", json.dumps({k: rec[k] for k in ("base_tokens", "special_tokens",
                                                             "vocab_size_with_added", "tokenizer_json_sha256")}))
  return 0


if __name__ == "__main__":
  sys.exit(main())
