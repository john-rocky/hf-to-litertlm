"""Inspect the sarashina2.2 SentencePiece model + HF fast-tokenizer decoder chain.
Decides the tokenizer path (native SP verbatim vs HF tokenizer.json) before export."""
import sys, json
from huggingface_hub import hf_hub_download
from sentencepiece import sentencepiece_model_pb2 as spb
import sentencepiece as spm

repo = sys.argv[1] if len(sys.argv) > 1 else "sbintuitions/sarashina2.2-0.5b-instruct-v0.1"
mp_path = hf_hub_download(repo, "tokenizer.model")
mp = spb.ModelProto(); mp.ParseFromString(open(mp_path, "rb").read())
T = spb.ModelProto.SentencePiece.Type
print("pieces:", len(mp.pieces), "model_type:", mp.trainer_spec.model_type,
      "byte_fallback:", mp.trainer_spec.byte_fallback,
      "add_dummy_prefix:", mp.normalizer_spec.add_dummy_prefix,
      "remove_extra_whitespaces:", mp.normalizer_spec.remove_extra_whitespaces,
      "escape_whitespaces:", mp.normalizer_spec.escape_whitespaces,
      "normalizer:", mp.normalizer_spec.name,
      "unk/bos/eos/pad ids:", mp.trainer_spec.unk_id, mp.trainer_spec.bos_id, mp.trainer_spec.eos_id, mp.trainer_spec.pad_id)
for i in range(20):
    p = mp.pieces[i]; print(f"  id {i:3d} type={T.Name(p.type):12s} piece={p.piece!r}")
n_byte = sum(1 for p in mp.pieces if p.type == T.BYTE)
n_ud = sum(1 for p in mp.pieces if p.type == T.USER_DEFINED)
n_ctrl = sum(1 for p in mp.pieces if p.type == T.CONTROL)
print("counts: BYTE", n_byte, "USER_DEFINED", n_ud, "CONTROL", n_ctrl)
sp = spm.SentencePieceProcessor(model_file=mp_path)
for s in ["<|user|>こんにちは。あなたの名前を教えて</s><|assistant|>",
          "日本の首都はどこですか？ 簡潔に答えてください。",
          "17 + 25 = 42. The capital of France is Paris.",
          "😀 鬱 𠮷野家 café"]:
    ids = sp.encode(s)
    print("RAW-SP", repr(s), "->", ids[:24], "...", "pieces:", [sp.id_to_piece(i) for i in ids[:12]])
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(repo)
print("HF tokenizer class:", type(tok).__name__, "add_bos_token:", getattr(tok, "add_bos_token", None))
for s in ["<|user|>こんにちは。あなたの名前を教えて</s><|assistant|>", "😀 鬱 𠮷野家 café"]:
    ids = tok(s, add_special_tokens=False)["input_ids"]
    print("HF", repr(s), "->", ids[:24], "decode:", repr(tok.decode(ids)))
bt = getattr(tok, "backend_tokenizer", None)
if bt is not None:
    print("normalizer:", json.loads(bt.normalizer.__getstate__().decode() if hasattr(bt.normalizer, "__getstate__") else "{}") if bt.normalizer else None)
    print("pre_tokenizer:", str(bt.pre_tokenizer.__getstate__().decode()) if bt.pre_tokenizer else None)
    print("decoder:", str(bt.decoder.__getstate__().decode()) if bt.decoder else None)
msgs = [{"role": "user", "content": "こんにちは。あなたの名前を教えて"}]
print("rendered:", repr(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)))
print("rendered ids:", tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)[:12])
