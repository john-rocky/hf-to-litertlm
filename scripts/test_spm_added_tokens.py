"""Validate the added-tokens SP fix on Nanbeige before patching the pipeline:
convert HF tokenizer -> SP, then append the added_tokens (166100-166109) as
USER_DEFINED pieces + pad to the model vocab, and verify <think> lands at 166103.
    ~/clipconv/bin/python scripts/test_spm_added_tokens.py
"""
from transformers import AutoTokenizer
from litert_torch.generative.tools import tokenizer_to_sentencepiece_lib as tok_spm
from sentencepiece import sentencepiece_model_pb2 as spb
import sentencepiece as spm

MID = "Nanbeige/Nanbeige4.1-3B"
MODEL_VOCAB = 166144  # config vocab_size (embedding rows)

tok = AutoTokenizer.from_pretrained(MID, trust_remote_code=True, use_fast=False)
vf = getattr(tok, "vocab_file", None)
if vf and not str(vf).endswith((".model", ".spiece", ".spm")):
    tok.vocab_file = None
spm_bytes = tok_spm.convert(tok)

mp = spb.ModelProto()
mp.ParseFromString(spm_bytes)
base_n = len(mp.pieces)
print(f"base SP pieces: {base_n}")

added = sorted((int(i), t.content) for i, t in tok.added_tokens_decoder.items())
print(f"added_tokens: {added}")

# append: fill any gap with UNUSED, place each added token at its exact id, pad to MODEL_VOCAB
by_id = dict(added)
target = max(MODEL_VOCAB, (max(by_id) + 1) if by_id else 0)
while len(mp.pieces) < target:
    idx = len(mp.pieces)
    p = mp.pieces.add()
    if idx in by_id:
        p.piece = by_id[idx]; p.score = 0.0
        p.type = spb.ModelProto.SentencePiece.USER_DEFINED
    else:
        p.piece = f"<unused_{idx}>"; p.score = 0.0
        p.type = spb.ModelProto.SentencePiece.UNUSED

print(f"patched SP pieces: {len(mp.pieces)}")
new_bytes = mp.SerializeToString()

# verify via protobuf directly (avoid the C++ SP load, which mutex-crashes on this Mac)
print(f"id 166100 -> {mp.pieces[166100].piece!r} (type {mp.pieces[166100].type})")
print(f"id 166103 -> {mp.pieces[166103].piece!r} (type {mp.pieces[166103].type})  (expect '<think>')")
print(f"id 166104 -> {mp.pieces[166104].piece!r}")
print(f"total pieces = {len(mp.pieces)} (model vocab {MODEL_VOCAB})")
assert mp.pieces[166103].piece == "<think>", "MISMATCH"
assert len(mp.pieces) == MODEL_VOCAB, f"count {len(mp.pieces)} != {MODEL_VOCAB}"
print("FIX_VALID")
