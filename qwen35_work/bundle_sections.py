#!/usr/bin/env python3
"""bundle_sections.py OUT.json FILE... - the section table of each .litertlm, read out of its header with litert_lm_builder (the header
read of ../manifest/make_manifest.py plus the begin / end offsets and every string item, e.g. the TFLite section's model_type and
prefer_activation_type), each section's sha256 over its bytes, the LlmMetadata fields (start / stop tokens, max_num_tokens, model type,
supports_*, channels, the jinja template's sha256 + length + text), the ExecutorMetadata state buffers (count by type,
maximum_sequence_length, max_history_size), and for the HF_Tokenizer_Zlib section the sha256 / size of the inflated tokenizer.json
(the zlib stream after the 8-byte little-endian length prefix, as the builder writes it). The file sha256 changes with every pack (a
uuid and a timestamp in the header): compare these section hashes instead. Reads only; needs litert-lm-builder (0.18.x)."""
import collections
import hashlib
import json
import os
import sys
import zlib

from litert_lm_builder import litertlm_core
from litert_lm_builder import litertlm_header_schema_py_generated as schema
from litert_lm_builder.runtime.proto import executor_metadata_pb2, llm_metadata_pb2

HDR_BEGIN = litertlm_core.HEADER_BEGIN_BYTE_OFFSET
HDR_END_LOC = litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET


def table(fh):
    fh.seek(0)
    head = fh.read(4096)
    if head[:8] != litertlm_core.HEADER_MAGIC_BYTES:
        raise ValueError("not a litertlm file (bad magic)")
    hdr_end = int.from_bytes(head[HDR_END_LOC:HDR_END_LOC + 8], "little")
    if hdr_end > len(head):
        fh.seek(0)
        head = fh.read(hdr_end + 64)
    meta = schema.LiteRTLMMetaData.GetRootAs(bytearray(head[HDR_BEGIN:hdr_end]), 0)
    rows = []
    for i in range(meta.SectionMetadata().ObjectsLength()):
        so = meta.SectionMetadata().Objects(i)
        row = {"index": i, "type": litertlm_core.any_section_data_type_to_string(so.DataType()),
               "begin": so.BeginOffset(), "end": so.EndOffset(), "bytes": so.EndOffset() - so.BeginOffset()}
        for j in range(so.ItemsLength()):
            it = so.Items(j)
            key = it.Key().decode() if it.Key() else None
            tab = it.Value()
            if key and tab is not None and it.ValueType() == schema.VData.StringValue:
                sv = schema.StringValue()
                sv.Init(tab.Bytes, tab.Pos)
                raw = sv.Value()
                row[key] = raw.decode() if raw else ""
        rows.append(row)
    return rows


def main():
    out, files = sys.argv[1], sys.argv[2:]
    rep = {}
    for p in files:
        small = {}
        with open(p, "rb") as fh:
            rows = table(fh)
            for r in rows:
                fh.seek(r["begin"])
                h, left = hashlib.sha256(), r["bytes"]
                keep = r["type"] in ("LlmMetadataProto", "ExecutorMetadataProto", "HF_Tokenizer_Zlib")
                buf = bytearray() if keep else None
                while left > 0:
                    chunk = fh.read(min(1 << 24, left))
                    if not chunk:
                        break
                    if keep:
                        buf += chunk
                    h.update(chunk)
                    left -= len(chunk)
                r["sha256"] = h.hexdigest()
                if keep:
                    small[r["type"]] = bytes(buf)
        m = {}
        if "LlmMetadataProto" in small:
            meta = llm_metadata_pb2.LlmMetadata()
            meta.ParseFromString(small["LlmMetadataProto"])
            jj = meta.jinja_prompt_template if meta.HasField("jinja_prompt_template") else ""
            m = {"fields_set": [f.name for f, _ in meta.ListFields()],
                 "start_token": str(meta.start_token).strip() if meta.HasField("start_token") else None,
                 "stop_tokens": [str(t).strip() for t in meta.stop_tokens],
                 "max_num_tokens": meta.max_num_tokens,
                 "llm_model_type": meta.llm_model_type.WhichOneof("model_type") if meta.HasField("llm_model_type") else None,
                 "supports_thinking": meta.supports_thinking if meta.HasField("supports_thinking") else None,
                 "supports_function_calling": meta.supports_function_calling if meta.HasField("supports_function_calling") else None,
                 "has_sampler_params": meta.HasField("sampler_params"),
                 "channels": len(meta.channels),
                 "jinja_len": len(jj.encode()), "jinja_sha256": hashlib.sha256(jj.encode()).hexdigest(), "jinja": jj}
        e = {}
        if "ExecutorMetadataProto" in small:
            em = executor_metadata_pb2.ExecutorMetadata()
            em.ParseFromString(small["ExecutorMetadataProto"])
            lm = em.llm_executor_metadata
            types = collections.Counter(b.DESCRIPTOR.fields_by_name["type"].enum_type.values_by_number[b.type].name
                                        for b in lm.state_buffers)
            e = {"state_buffers": len(lm.state_buffers), "by_type": dict(types),
                 "maximum_sequence_length": sorted(set(b.maximum_sequence_length for b in lm.state_buffers)),
                 "max_history_size": lm.max_history_size,
                 "names_sample": [b.prefill_input_name for b in lm.state_buffers][:4]}
        t = {}
        if "HF_Tokenizer_Zlib" in small:
            raw = small["HF_Tokenizer_Zlib"]
            n = int.from_bytes(raw[:8], "little")
            js = zlib.decompress(raw[8:])
            t = {"declared_len": n, "inflated_len": len(js), "inflated_sha256": hashlib.sha256(js).hexdigest()}
        rep[os.path.basename(p)] = {"path": p, "size": os.path.getsize(p), "sections": rows, "llm_metadata": m,
                                    "executor_metadata": e, "tokenizer": t}
        print(os.path.basename(p), os.path.getsize(p))
        for r in rows:
            extra = {k: v for k, v in r.items() if k not in ("index", "type", "begin", "end", "bytes", "sha256")}
            print(f"  [{r['index']}] {r['type']} {r['bytes']:,} B {extra} sha256 {r['sha256']}")
        print("  llm", json.dumps({k: v for k, v in m.items() if k != "jinja"}))
        print("  exec", json.dumps(e))
        print("  tok", json.dumps(t))
    json.dump(rep, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
