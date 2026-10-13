#!/usr/bin/env python3
"""make_pbtext.py dump BUNDLE OUT.pbtext  |  make_pbtext.py edit RAW.pbtext OUT.pbtext REPORT.json [--jinja FILE] - the metadata
step of convert_g9v3_3b.sh (run with litert-lm 0.18.0: litert-lm-builder 0.18.0 protos).

dump : the bundle's LlmMetadata section (header read, no unpack) as text_format -> OUT.pbtext.
edit : RAW.pbtext (the exporter's LlmMetadata) -> OUT.pbtext with these fields, nothing else touched:
         stop_tokens       = exactly two: id 1 (`</s>`) and id 130073 (`<|im_end|>`) = generation_config eos_token_id; kept
                             as the exporter wrote them when they already are that set (ids, either order), else replaced
         sampler_params    = cleared (the exporter derives TOP_P p 0.95 temp 0.9 from generation_config do_sample; the
                             bundle carries none -> engine default; vendor sampling goes in the card text)
         channels          = exactly one: thought, start '<think>\\n', end '\\n</think>' (LiteRT-LM models/qwen3; the
                             template pre-opens '<think>\\n', the model closes '\\n</think>')
         supports_thinking = true; supports_function_calling = false (no runtime parser for the XML call form)
       start_token, max_num_tokens, llm_model_type, jinja_prompt_template stay the exporter's (checked, not changed: the
       report lists them and whether the jinja equals g9v3.jinja byte for byte).
       --jinja FILE: jinja_prompt_template := FILE's bytes (a history-form swap after export; then jinja_prompt_template is a
       changed field and the check is against FILE).
REPORT.json = the changed top-level fields (RAW vs OUT, parsed) + the kept values."""
import hashlib
import json
import os
import sys

from google.protobuf import text_format
from litert_lm_builder import litertlm_core
from litert_lm_builder import litertlm_header_schema_py_generated as schema
from litert_lm_builder.runtime.proto import llm_metadata_pb2

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "g9v3.jinja")
STOP_IDS = [1, 130073]


def dump(bundle, out):
    with open(bundle, "rb") as f:
        prefix = f.read(32)
        end = int.from_bytes(prefix[litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET:
                                    litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET + 8], "little")
        f.seek(0)
        head = f.read(end)
        md = schema.LiteRTLMMetaData.GetRootAs(head[litertlm_core.HEADER_BEGIN_BYTE_OFFSET:], 0)
        for i in range(md.SectionMetadata().ObjectsLength()):
            s = md.SectionMetadata().Objects(i)
            if litertlm_core.any_section_data_type_to_string(s.DataType()) == "LlmMetadataProto":
                f.seek(s.BeginOffset())
                m = llm_metadata_pb2.LlmMetadata()
                m.ParseFromString(f.read(s.EndOffset() - s.BeginOffset()))
                open(out, "w", encoding="utf-8").write(text_format.MessageToString(m))
                print(f"dumped LlmMetadata ({s.EndOffset() - s.BeginOffset()} B) -> {out}")
                return 0
    print("no LlmMetadataProto section", file=sys.stderr)
    return 1


def stop_ids(m):
    out = []
    for t in m.stop_tokens:
        if t.HasField("token_ids"):
            out.append(list(t.token_ids.ids))
        else:
            out.append(t.token_str)
    return out


def edit(raw, out, report, jinja=None):
    m = llm_metadata_pb2.LlmMetadata()
    text_format.Parse(open(raw, encoding="utf-8").read(), m)
    before = llm_metadata_pb2.LlmMetadata()
    before.CopyFrom(m)
    raw_stops = stop_ids(m)
    if sorted(map(str, raw_stops)) != sorted(map(str, [[i] for i in STOP_IDS])):
        del m.stop_tokens[:]
        for i in STOP_IDS:
            m.stop_tokens.add().token_ids.ids.append(i)
    m.ClearField("sampler_params")
    del m.channels[:]
    ch = m.channels.add()
    ch.channel_name = "thought"
    ch.start = "<think>\n"
    ch.end = "\n</think>"
    m.supports_thinking = True
    m.supports_function_calling = False
    if jinja:
        m.jinja_prompt_template = open(jinja, "rb").read().decode("utf-8")
    open(out, "w", encoding="utf-8").write(text_format.MessageToString(m))
    a = {f.name: v for f, v in before.ListFields()}
    b = {f.name: v for f, v in m.ListFields()}
    changed = sorted(k for k in set(a) | set(b) if (k not in a) or (k not in b) or a[k] != b[k])
    ref = open(jinja or TEMPLATE, "rb").read()
    jj = m.jinja_prompt_template.encode("utf-8")
    rep = {"raw": raw, "out": out, "changed_fields": changed, "raw_stop_tokens": raw_stops, "out_stop_tokens": stop_ids(m),
           "raw_sampler_params": text_format.MessageToString(before.sampler_params) if before.HasField("sampler_params") else None,
           "raw_channels": [text_format.MessageToString(c) for c in before.channels],
           "start_token": text_format.MessageToString(m.start_token).strip(),
           "max_num_tokens": m.max_num_tokens, "llm_model_type": text_format.MessageToString(m.llm_model_type).strip(),
           "jinja_sha256": hashlib.sha256(jj).hexdigest(), "jinja_bytes": len(jj),
           "jinja_equals_lane_template": jj == ref, "lane_template": jinja or TEMPLATE, "lane_sha256": hashlib.sha256(ref).hexdigest()}
    json.dump(rep, open(report, "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: rep[k] for k in ("changed_fields", "raw_stop_tokens", "out_stop_tokens", "start_token",
                                          "max_num_tokens", "llm_model_type", "jinja_equals_lane_template")}, ensure_ascii=False))
    return 0 if rep["jinja_equals_lane_template"] else 1


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        raise SystemExit(dump(sys.argv[2], sys.argv[3]))
    jj = sys.argv[sys.argv.index("--jinja") + 1] if "--jinja" in sys.argv else None
    raise SystemExit(edit(sys.argv[2], sys.argv[3], sys.argv[4], jj))
