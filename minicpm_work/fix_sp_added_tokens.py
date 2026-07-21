#!/usr/bin/env python3
"""Append HF added tokens to a SentencePiece model as USER_DEFINED pieces.

MiniCPM4/4.1 keep their chat markers (<|im_end|>=73440, <|im_start|>=73441, ...)
in HF added_tokens.json, NOT in tokenizer.model. A .litertlm bundled with the
raw SP model can therefore never emit/stop on <|im_end|>; bundled with the HF
tokenizer.json instead, the runtime drops all spaces on decode (Metaspace).
Fix: extend the SP proto to vocab_size with the added tokens at their exact ids
(UNUSED padding elsewhere) and bundle with `litert-lm-builder sp_tokenizer`.

Usage: fix_sp_added_tokens.py tokenizer.model added_tokens.json out.spiece [vocab_size]
"""
import json
import sys

from sentencepiece import sentencepiece_model_pb2 as spb


def main():
    src, added_path, out = sys.argv[1:4]
    vocab_size = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    mp = spb.ModelProto()
    mp.ParseFromString(open(src, "rb").read())
    base_n = len(mp.pieces)
    by_id = {v: k for k, v in json.load(open(added_path)).items()}
    target = max(vocab_size, max(by_id) + 1)
    while len(mp.pieces) < target:
        idx = len(mp.pieces)
        p = mp.pieces.add()
        if idx in by_id:
            p.piece = by_id[idx]
            p.score = 0.0
            p.type = spb.ModelProto.SentencePiece.USER_DEFINED
        else:
            p.piece = f"<unused_{idx}>"
            p.score = 0.0
            p.type = spb.ModelProto.SentencePiece.UNUSED
    open(out, "wb").write(mp.SerializeToString())
    print(f"{out}: {base_n} -> {len(mp.pieces)} pieces "
          f"(+{sum(1 for i in by_id if i >= base_n)} added tokens)")


if __name__ == "__main__":
    main()
