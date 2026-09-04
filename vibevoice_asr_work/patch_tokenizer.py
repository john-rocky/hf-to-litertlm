#!/usr/bin/env python3
"""Add the three speech-marker tokens to the vendor tokenizer.json.

The BitNet repo ships the Qwen2 (not 2.5) tokenizer: its added_tokens stop at 151645, so
the ids the model was trained with for the speech markers — 151646 <|object_ref_start|>
(speech_start), 151647 <|object_ref_end|> (speech_end), 151648 <|box_start|> (speech_pad)
— have NO string form.  VibeASR.cpp hard-codes the ids.  LiteRT-LM's generic audio path
inserts start_of_audio_token / audio_suffix as TEXT that is tokenized, so the strings must
exist.  Adding them in this order yields exactly 151646/151647/151648 (asserted).
Usage: python patch_tokenizer.py <in tokenizer.json> <out tokenizer.json>
"""
import sys
from tokenizers import Tokenizer, AddedToken
src, dst = sys.argv[1], sys.argv[2]
tk = Tokenizer.from_file(src)
names = ["<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>"]
assert all(tk.token_to_id(n) is None for n in names), "already present"
n = tk.add_special_tokens([AddedToken(x, special=True, normalized=False) for x in names])
ids = {x: tk.token_to_id(x) for x in names}
assert ids == {"<|object_ref_start|>": 151646, "<|object_ref_end|>": 151647, "<|box_start|>": 151648}, ids
# The strings must round-trip as single ids inside ordinary text.
enc = tk.encode("user\n<|object_ref_start|><|box_start|><|object_ref_end|>\nThis is", add_special_tokens=False)
assert [i for i in enc.ids if i >= 151646] == [151646, 151648, 151647], enc.ids
# And the untouched part of the vocabulary must tokenize identically.
for s in ("Hello world", "This is a 5.86 seconds audio, please transcribe it.", "<|im_start|>user\n"):
    assert Tokenizer.from_file(src).encode(s, add_special_tokens=False).ids == tk.encode(s, add_special_tokens=False).ids, s
tk.save(dst)
print("added", n, ids, "->", dst)
