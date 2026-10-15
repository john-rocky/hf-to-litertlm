#!/usr/bin/env python3
"""make_pbtext.py dump BUNDLE OUT.pbtext  |  make_pbtext.py edit RAW.pbtext OUT.pbtext REPORT.json - the metadata step of
convert_ministral3_8b.sh (run with litert-lm 0.18.0: litert-lm-builder 0.18.0 protos).

dump : the bundle's LlmMetadata section (header read, no unpack) as text_format -> OUT.pbtext.
edit : RAW.pbtext (the exporter's LlmMetadata) -> OUT.pbtext with these fields, nothing else touched:
         stop_tokens               = exactly one: id 2 (`</s>` = generation_config eos_token_id). The exporter also adds
                                     the template's model suffix "</s>" as a string + 13 punctuation-prefixed copies
                                     (litert_torch export_hf/core/litert_lm_builder.py build_llm_metadata,
                                     _STOP_TOKEN_PREFIXES), in set order (not reproducible between runs); id 2 already
                                     stops every turn end.
         supports_thinking         = false (the template has no thinking branch; the 8B Instruct writes no [THINK])
         supports_function_calling = false (the bundle's template has no tools branch: tools are not rendered, and
                                     generic_model has no Mistral [TOOL_CALLS] parser in LiteRT-LM 0.18). The 0.18 builder
                                     refuses a pack without both fields (litertlm_builder.py validate_metadata).
         sampler_params            = cleared if the exporter wrote any (generation_config has no sampling keys -> expected
                                     none; the bundle then runs the engine default)
       Checked, never changed (the edit refuses otherwise): start_token = `<s>` (the tokenizer's bos_token; the bundle's
       template writes no BOS, the vendor template writes `{{ bos_token }}` = the same one token), no channels,
       max_num_tokens 4096, llm_model_type generic_model, jinja_prompt_template == ministral3_8b_dual.jinja (next to this
       script) byte for byte.
REPORT.json = the changed top-level fields (RAW vs OUT, parsed) + the kept values. Exit 0 = written + every check held."""
import hashlib
import json
import os
import sys

from google.protobuf import text_format
from litert_lm_builder import litertlm_core
from litert_lm_builder import litertlm_header_schema_py_generated as schema
from litert_lm_builder.runtime.proto import llm_metadata_pb2

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ministral3_8b_dual.jinja")
STOP_IDS = [2]


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


def stops(m):
    out = []
    for t in m.stop_tokens:
        if t.HasField("token_ids"):
            out.append(list(t.token_ids.ids))
        else:
            out.append(t.token_str)
    return out


def start(m):
    if not m.HasField("start_token"):
        return None
    t = m.start_token
    return list(t.token_ids.ids) if t.HasField("token_ids") else t.token_str


def edit(raw, out, report):
    m = llm_metadata_pb2.LlmMetadata()
    text_format.Parse(open(raw, encoding="utf-8").read(), m)
    before = llm_metadata_pb2.LlmMetadata()
    before.CopyFrom(m)
    raw_stops = stops(m)
    del m.stop_tokens[:]
    for i in STOP_IDS:
        m.stop_tokens.add().token_ids.ids.append(i)
    m.ClearField("sampler_params")
    m.supports_thinking = False
    m.supports_function_calling = False
    open(out, "w", encoding="utf-8").write(text_format.MessageToString(m))
    a = {f.name: v for f, v in before.ListFields()}
    b = {f.name: v for f, v in m.ListFields()}
    changed = sorted(k for k in set(a) | set(b) if (k not in a) or (k not in b) or a[k] != b[k])
    tpl = open(TEMPLATE, "rb").read()
    jj = m.jinja_prompt_template.encode("utf-8")
    checks = {"start_token_bos": start(m) in ("<s>", [1]), "no_channels": len(m.channels) == 0,
              "max_num_tokens_4096": m.max_num_tokens == 4096,
              "generic_model": m.llm_model_type.WhichOneof("model_type") == "generic_model",
              "jinja_equals_template": jj == tpl, "raw_has_id_2": [2] in raw_stops}
    rep = {"raw": raw, "out": out, "changed_fields": changed, "raw_stop_tokens": raw_stops, "out_stop_tokens": stops(m),
           "start_token": start(m),
           "raw_sampler_params": text_format.MessageToString(before.sampler_params) if before.HasField("sampler_params") else None,
           "raw_supports_thinking": before.supports_thinking if before.HasField("supports_thinking") else None,
           "raw_supports_function_calling": (before.supports_function_calling
                                             if before.HasField("supports_function_calling") else None),
           "raw_fields_set": [f.name for f, _ in before.ListFields()],
           "max_num_tokens": m.max_num_tokens, "llm_model_type": text_format.MessageToString(m.llm_model_type).strip(),
           "jinja_sha256": hashlib.sha256(jj).hexdigest(), "jinja_bytes": len(jj),
           "template": TEMPLATE, "template_sha256": hashlib.sha256(tpl).hexdigest(), "checks": checks}
    json.dump(rep, open(report, "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: rep[k] for k in ("changed_fields", "start_token", "raw_stop_tokens", "out_stop_tokens",
                                          "raw_fields_set", "checks")}, ensure_ascii=False))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        raise SystemExit(dump(sys.argv[2], sys.argv[3]))
    raise SystemExit(edit(sys.argv[2], sys.argv[3], sys.argv[4]))
