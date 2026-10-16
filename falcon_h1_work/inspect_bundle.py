#!/usr/bin/env python3
"""inspect_bundle.py BUNDLE [OUT.json] - header read of a .litertlm (no unpack, no engine): the section list (type, bytes, items),
the ExecutorMetadata state buffers (count by type) and the LlmMetadata summary (start/stop tokens, max_num_tokens, model type,
flags, jinja length). Needs litert-lm-builder 0.18.0."""
import collections, json, sys
from litert_lm_builder import litertlm_core, litertlm_peek
from litert_lm_builder import litertlm_header_schema_py_generated as schema
from litert_lm_builder.runtime.proto import executor_metadata_pb2, llm_metadata_pb2
from google.protobuf import text_format


def sections(path):
    with open(path, "rb") as f:
        prefix = f.read(32)
        end = int.from_bytes(prefix[litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET:litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET + 8], "little")
        f.seek(0)
        head = f.read(end)
    md = schema.LiteRTLMMetaData.GetRootAs(head[litertlm_core.HEADER_BEGIN_BYTE_OFFSET:], 0)
    out = []
    for i in range(md.SectionMetadata().ObjectsLength()):
        s = md.SectionMetadata().Objects(i)
        out.append({"index": i, "type": litertlm_core.any_section_data_type_to_string(s.DataType()), "begin": s.BeginOffset(),
                    "end": s.EndOffset(), "bytes": s.EndOffset() - s.BeginOffset(),
                    "items": [litertlm_peek.kvp_to_dict(s.Items(j)) for j in range(s.ItemsLength())]})
    return out


def read(path, s):
    with open(path, "rb") as f:
        f.seek(s["begin"])
        return f.read(s["bytes"])


path = sys.argv[1]
secs = sections(path)
doc = {"bundle": path, "sections": [{k: s[k] for k in ("index", "type", "bytes", "items")} for s in secs]}
for s in secs:
    print(f"[{s['index']}] {s['type']}: {s['bytes']} B items={s['items']}")
    if s["type"] == "ExecutorMetadataProto":
        m = executor_metadata_pb2.ExecutorMetadata(); m.ParseFromString(read(path, s))
        sb = m.llm_executor_metadata.state_buffers
        kinds = collections.Counter(executor_metadata_pb2.StateBufferType.Name(b.type) if hasattr(executor_metadata_pb2, "StateBufferType") else str(b.type) for b in sb)
        doc["state_buffers"] = len(sb)
        doc["executor_metadata_text"] = text_format.MessageToString(m)
        print(f"    state buffers: {len(sb)}")
        print("    " + doc["executor_metadata_text"][:1500].replace("\n", "\n    "))
    if s["type"] == "LlmMetadataProto":
        m = llm_metadata_pb2.LlmMetadata(); m.ParseFromString(read(path, s))
        summ = {"fields_set": [f.name for f, _ in m.ListFields()], "start_token": text_format.MessageToString(m.start_token).strip(),
                "stop_tokens": [text_format.MessageToString(t).strip() for t in m.stop_tokens], "max_num_tokens": m.max_num_tokens,
                "llm_model_type": text_format.MessageToString(m.llm_model_type).strip()[:300],
                "supports_thinking": m.supports_thinking, "supports_function_calling": m.supports_function_calling,
                "jinja_len": len(m.jinja_prompt_template)}
        doc["llm_metadata"] = summ
        print("    " + json.dumps(summ, ensure_ascii=False)[:1200])
if len(sys.argv) > 2:
    json.dump(doc, open(sys.argv[2], "w"), indent=1, ensure_ascii=False)
