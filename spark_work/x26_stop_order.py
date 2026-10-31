#!/usr/bin/env python3
"""x26_stop_order.py IN.litertlm OUT.litertlm NAME -- the stop list of a Spark-X2.6 file in the order of the published
file NAME (Spark-X2.6-1.7B_int4-mlp, Spark-X2.6-1.7B_int8, Spark-X2.6-4B_int4, Spark-X2.6-4B_int8); the last step of
spark_work/x26_build.sh for each file.

litert-torch writes the 15 stop entries (the end-of-sentence token as id 1 and as text, and the text after each of 13
characters) in Python set order, which changes with the hash seed of the export process. The published files were
exported without a fixed PYTHONHASHSEED and list the entries in four different orders. The order is not neutral: from
LiteRT-LM 0.17.0 on it decides whether the runtime writes the last punctuation mark of an answer (read in the LiteRT-LM
source of 0.17.0 to 0.18.1; the model cards' Known runtime issues). This step copies IN to OUT and rewrites the
LlmMetadataProto section in place with the same entries in the published order: the section keeps its size, so the
header, every offset and every other byte of the file stay as they are (checked). Needs litert-lm-builder (0.16.1
here).

    python -B spark_work/x26_stop_order.py in.litertlm out/Spark-X2.6-1.7B_int8.litertlm Spark-X2.6-1.7B_int8
"""
import hashlib
import shutil
import sys

from litert_lm_builder import litertlm_core as core
from litert_lm_builder import litertlm_header_schema_py_generated as schema
from litert_lm_builder.runtime.proto import llm_metadata_pb2 as pb

EOS = "<｜end▁of▁sentence｜>"
# 1 = the id entry; a string s = the text entry s + EOS ("" = EOS alone). Read from the published files' headers.
ORDERS = {
    "Spark-X2.6-1.7B_int4-mlp": [".", 1, "", " ", ",", "!", "?", ";", "'", "*", ":", "}", "]", '"', ")"],
    "Spark-X2.6-1.7B_int8": [1, "!", ".", "'", "?", ",", ":", '"', ";", "}", " ", "*", "]", ")", ""],
    "Spark-X2.6-4B_int4": [1, "", ".", ":", '"', ",", "*", "}", "'", "]", " ", "?", "!", ")", ";"],
    "Spark-X2.6-4B_int8": [1, '"', ".", "", "]", "!", "'", ")", "?", ":", ",", "}", " ", "*", ";"],
}


def meta_section(path):
    with open(path, "rb") as f:
        head = f.read(4096)
        end = int.from_bytes(head[core.HEADER_END_LOCATION_BYTE_OFFSET:core.HEADER_END_LOCATION_BYTE_OFFSET + 8], "little")
        if end > len(head):
            f.seek(0)
            head = f.read(end + 64)
        meta = schema.LiteRTLMMetaData.GetRootAs(bytearray(head[core.HEADER_BEGIN_BYTE_OFFSET:end]), 0)
        objs = [meta.SectionMetadata().Objects(i) for i in range(meta.SectionMetadata().ObjectsLength())]
        hits = [(o.BeginOffset(), o.EndOffset()) for o in objs
                if core.any_section_data_type_to_string(o.DataType()) == "LlmMetadataProto"]
        assert len(hits) == 1, hits
        b, e = hits[0]
        f.seek(b)
        return b, e, f.read(e - b)


def key(t):
    return 1 if t.HasField("token_ids") and list(t.token_ids.ids) == [1] else t.token_str


def main(src, dst, name):
    want = [1 if x == 1 else x + EOS for x in ORDERS[name]]
    b, e, raw = meta_section(src)
    m = pb.LlmMetadata()
    m.ParseFromString(raw)
    assert m.SerializeToString() == raw, "the section does not round-trip through the proto"
    have = {key(t): t.SerializeToString() for t in m.stop_tokens}
    assert len(have) == len(m.stop_tokens) and sorted(map(str, have)) == sorted(map(str, want)), "other stop entries"
    n = pb.LlmMetadata()
    n.CopyFrom(m)
    del n.stop_tokens[:]
    for k in want:
        n.stop_tokens.add().ParseFromString(have[k])
    new = n.SerializeToString()
    assert len(new) == len(raw)
    a, c = pb.LlmMetadata(), pb.LlmMetadata()
    a.CopyFrom(m)
    c.ParseFromString(new)
    del a.stop_tokens[:]
    del c.stop_tokens[:]
    assert a.SerializeToString() == c.SerializeToString(), "something besides the stop order changed"
    shutil.copyfile(src, dst)
    with open(dst, "r+b") as f:
        f.seek(b)
        f.write(new)
    with open(src, "rb") as fa, open(dst, "rb") as fb:
        pos = 0
        while True:
            x, y = fa.read(1 << 22), fb.read(1 << 22)
            if not x and not y:
                break
            if x != y:
                diff = [pos + i for i in range(min(len(x), len(y))) if x[i] != y[i]]
                assert len(x) == len(y) and all(b <= p < e for p in diff), "bytes changed outside the metadata section"
            pos += len(x)
    print(f"STOP_ORDER OK {name}: LlmMetadata {hashlib.sha256(raw).hexdigest()[:12]} -> "
          f"{hashlib.sha256(new).hexdigest()[:12]} ({len(new)} B), every other byte unchanged -> {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[3] not in ORDERS:
        sys.exit(__doc__)
    main(*sys.argv[1:4])
