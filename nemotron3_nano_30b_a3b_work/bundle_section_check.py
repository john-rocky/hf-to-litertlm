#!/usr/bin/env python3
"""bundle_section_check.py BUNDLE --tflite W.tflite --tokenizer tokenizer.json — the bundle's sections against their
sources (run with a python that has litert_lm_builder): the TFLiteModel section = the standalone
TFLite byte for byte (sha256 of both, streamed); the HF tokenizer section, unpacked (zlib), = the repo's tokenizer.json
byte for byte; the LlmMetadataProto parsed: stop tokens, max_num_tokens, model type, channels,
template sha256; prefer_activation_type of the TFLite section. Prints one JSON object; exit 0 = both byte checks equal.
"""
import argparse
import hashlib
import io
import json
import sys
import zlib

from litert_lm_builder import litertlm_core
from litert_lm_builder import litertlm_peek


def sha_range(path, start, end, chunk=16 << 20):
    h = hashlib.sha256()
    with open(path, "rb", buffering=0) as f:
        f.seek(start)
        left = end - start
        while left:
            b = f.read(min(chunk, left))
            if not b:
                raise IOError("short read")
            h.update(b)
            left -= len(b)
    return h.hexdigest()


def sha_file(path):
    import os
    return sha_range(path, 0, os.path.getsize(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--tflite", required=True)
    ap.add_argument("--tokenizer", required=True)
    a = ap.parse_args()
    meta = litertlm_peek.read_litertlm_header(a.bundle, io.StringIO())
    out = {"bundle": a.bundle, "sections": []}
    ok = True
    for i in range(meta.SectionMetadata().ObjectsLength()):
        so = meta.SectionMetadata().Objects(i)
        dtype = litertlm_core.any_section_data_type_to_string(so.DataType())
        b, e = so.BeginOffset(), so.EndOffset()
        row = {"type": dtype, "bytes": e - b, "model_type": litertlm_peek.get_model_type(so)}
        for j in range(so.ItemsLength()):
            it = so.Items(j)
            key = it.Key().decode() if it.Key() else None
            if key == "prefer_activation_type":
                from litert_lm_builder import litertlm_header_schema_py_generated as schema
                tab = it.Value()
                if tab is not None and it.ValueType() == schema.VData.StringValue:
                    sv = schema.StringValue()
                    sv.Init(tab.Bytes, tab.Pos)
                    row["prefer_activation_type"] = sv.Value().decode()
                else:
                    row["prefer_activation_type"] = f"value type {it.ValueType()}"
        if dtype == "TFLiteModel":
            row["sha256"] = sha_range(a.bundle, b, e)
            row["standalone_sha256"] = sha_file(a.tflite)
            row["equal"] = row["sha256"] == row["standalone_sha256"]
            ok &= row["equal"]
        elif dtype.startswith("HF_Tokenizer"):
            with open(a.bundle, "rb") as f:
                f.seek(b)
                raw = f.read(e - b)
            try:
                data = zlib.decompress(raw[8:]) if dtype.endswith("Zlib") else raw
            except zlib.error:
                data = zlib.decompress(raw)
            want = open(a.tokenizer, "rb").read()
            row["unpacked_sha256"] = hashlib.sha256(data).hexdigest()
            row["tokenizer_json_sha256"] = hashlib.sha256(want).hexdigest()
            row["equal"] = data == want
            ok &= row["equal"]
        elif dtype == "LlmMetadataProto":
            from litert_lm_builder.runtime.proto import llm_metadata_pb2
            with open(a.bundle, "rb") as f:
                f.seek(b)
                md = llm_metadata_pb2.LlmMetadata()
                md.ParseFromString(f.read(e - b))
            row["stop_tokens"] = [str(s).replace("\n", " ").strip() for s in md.stop_tokens]
            row["max_num_tokens"] = md.max_num_tokens
            row["model_type"] = md.llm_model_type.WhichOneof("model_type")
            row["channels"] = [[c.channel_name, c.start, c.end] for c in md.channels]
            row["template_sha256"] = hashlib.sha256(md.jinja_prompt_template.encode()).hexdigest()
            row["start_token"] = str(md.start_token).strip() or None
        out["sections"].append(row)
    out["pass"] = ok
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
