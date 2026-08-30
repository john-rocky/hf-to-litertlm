#!/usr/bin/env python3
"""Minimal reproduction for the upstream report: no LiteRT-LM, no bundle. Convert a public HF
byte-level-BPE tokenizer with litert-torch's tokenizer_to_sentencepiece_lib and encode a few
strings with SentencePiece; compare with the HF tokenizer.

    HF_HUB_DISABLE_XET=1 python3 repro_min.py HuggingFaceTB/SmolLM3-3B Qwen/Qwen2.5-0.5B-Instruct
"""
import logging, sys, os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
logging.basicConfig(level=logging.ERROR)
import transformers, sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as spb
from transformers import AutoTokenizer
try:
    import litert_torch
    from litert_torch.generative.tools import tokenizer_to_sentencepiece_lib as lib
    LT = getattr(litert_torch, "__version__", "installed")
except ImportError:  # run against a checkout: LITERT_TORCH_SRC=/path/to/litert-torch
    sys.path.insert(0, os.path.join(os.environ["LITERT_TORCH_SRC"], "litert_torch", "generative", "tools"))
    import tokenizer_to_sentencepiece_lib as lib
    LT = "checkout " + os.environ["LITERT_TORCH_SRC"]
TYPE = spb.ModelProto.SentencePiece.Type
STRINGS = ["<|im_end|>", "<|im_start|>", "<|endoftext|>", "é", "·", "ü", "café", "Zürich", "Łódź", "2025", "It's 5 o'clock"]
print(f"transformers {transformers.__version__}  sentencepiece {spm.__version__}  litert-torch {LT}")
for model in sys.argv[1:] or ["HuggingFaceTB/SmolLM3-3B", "Qwen/Qwen2.5-0.5B-Instruct"]:
    tok = AutoTokenizer.from_pretrained(model)
    print(f"\n=== {model}: unk={tok.unk_token!r} pad={tok.pad_token!r} eos={tok.eos_token!r}")
    tok.vocab_file = None  # otherwise convert() tries to parse vocab.json as a SentencePiece proto
    proto = lib.convert(tok)
    mp = spb.ModelProto(); mp.ParseFromString(proto)
    unk = [(i, p.piece) for i, p in enumerate(mp.pieces) if p.type == TYPE.UNKNOWN]
    print(f"pieces={len(mp.pieces)}  UNKNOWN piece(s)={unk}  unk_id={mp.trainer_spec.unk_id}")
    # the byte-level token whose GPT-2 spelling is the character itself, and the merged token for its UTF-8 bytes
    for spelling in ("é", "Ã©", "·", "Â·"):
        i = tok.convert_tokens_to_ids(spelling)
        if i is not None and i != tok.unk_token_id and i < len(mp.pieces):
            print(f"  HF token {i:6d} spelled {spelling!r:6} decode={tok.decode([i])!r:6} -> SP piece[{i}] = {mp.pieces[i].piece!r} ({TYPE.Name(mp.pieces[i].type)})")
    sp = spm.SentencePieceProcessor(); sp.LoadFromSerializedProto(proto)
    print(f"  {'string':18} {'HF ids':32} SentencePiece ids")
    for s in STRINGS:
        h = tok(s, add_special_tokens=False)["input_ids"]; e = sp.Encode(s)
        print(f"  {s!r:18} {str(h):32} {e}   {'OK' if h == e else 'MISMATCH'}")
