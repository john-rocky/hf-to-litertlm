#!/usr/bin/env python3
"""Offline check of the SentencePiece piece table that LlgConstraintProvider hands to
llguidance (GetTokens() == piece.piece() verbatim): does any piece contain byte 0x20?"""
import sys, collections, json
from litert_lm_builder import litertlm_core
from litert_lm_builder import litertlm_header_schema_py_generated as schema
from sentencepiece import sentencepiece_model_pb2 as spm_pb
import sentencepiece as spm

path = sys.argv[1]
with open(path, "rb") as f:
    f.seek(litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET)
    hdr_end = int.from_bytes(f.read(8), "little")
    f.seek(litertlm_core.HEADER_BEGIN_BYTE_OFFSET)
    hdr = f.read(hdr_end - litertlm_core.HEADER_BEGIN_BYTE_OFFSET)
    root = schema.LiteRTLMMetaData.GetRootAs(bytearray(hdr), 0)
    sec = None
    for i in range(root.SectionMetadata().ObjectsLength()):
        s = root.SectionMetadata().Objects(i)
        if s.DataType() == schema.AnySectionDataType.SP_Tokenizer:
            sec = (s.BeginOffset(), s.EndOffset()); break
    assert sec, "no SP_Tokenizer section"
    f.seek(sec[0]); raw = f.read(sec[1] - sec[0])
print("SP section bytes:", len(raw), "offsets", sec)
mp = spm_pb.ModelProto(); mp.ParseFromString(raw)
pieces = list(mp.pieces)
T = spm_pb.ModelProto.SentencePiece.Type
types = collections.Counter(T.Name(p.type) for p in pieces)
print("pieces:", len(pieces), dict(types))
with_space = [i for i, p in enumerate(pieces) if b" " in p.piece.encode("utf-8")]
with_meta = [i for i, p in enumerate(pieces) if "▁" in p.piece]
byte_pieces = [i for i, p in enumerate(pieces) if p.type == T.BYTE]
print("pieces containing raw byte 0x20:", len(with_space), [pieces[i].piece for i in with_space[:10]])
print("pieces containing U+2581 (E2 96 81):", len(with_meta))
print("BYTE pieces:", len(byte_pieces), "first:", pieces[byte_pieces[0]].piece if byte_pieces else None,
      "id of <0x20>:", next((i for i in byte_pieces if pieces[i].piece == "<0x20>"), None))
print("normalizer add_dummy_prefix:", mp.normalizer_spec.add_dummy_prefix,
      "escape_whitespaces:", mp.normalizer_spec.escape_whitespaces,
      "byte_fallback:", mp.trainer_spec.byte_fallback)
sp = spm.SentencePieceProcessor(); sp.LoadFromSerializedProto(raw)
for w in ["cat dog", " dog", "dog", "\n", "\t", " ", "  ", "cat\ndog"]:
    ids = sp.EncodeAsIds(w)
    print(json.dumps(w), "->", ids, [pieces[i].piece for i in ids], "decode:", json.dumps(sp.DecodeIds(ids)))
# The pieces the model would want for `cat dog` and what llguidance sees for them
for pid in sp.EncodeAsIds("cat dog"):
    p = pieces[pid]
    print(f"id {pid} piece={p.piece!r} bytes_in_llg_table={p.piece.encode('utf-8').hex()} type={T.Name(p.type)} decoded_alone={sp.DecodeIds([pid])!r}")
# whitespace-only pieces (what a JSON grammar can still take at a whitespace slot)
ws = [(i, p.piece) for i, p in enumerate(pieces) if p.piece and all(ch in " \t\r\n" for ch in p.piece)]
print("pieces made only of raw ASCII whitespace (no U+2581):", len(ws), ws[:12])
ctrl = [(i, p.piece) for i, p in enumerate(pieces) if p.type == T.CONTROL][:8]
print("CONTROL pieces (first 8):", ctrl)
