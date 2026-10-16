#!/usr/bin/env python3
"""make_pbtext.py BUNDLE TEMPLATE.jinja OUT.pbtext [--no-start-token] - the LlmMetadata of one exported bundle, edited.

Starts from the bundle's LlmMetadata section bytes (read through the header, no unpack) and changes exactly these fields:
  jinja_prompt_template = TEMPLATE.jinja, byte for byte (nemotron_nano_dual.jinja for NVIDIA-Nemotron-Nano-9B-v2)
  start_token           = cleared with --no-start-token (the model's prompt stream has no BOS; the exporter writes start_token
                          from tokenizer.bos_token, `<s>` here)
  supports_thinking     = true (the model reasons by default; `/think` / `/no_think` in the prompt switch it)
Everything else (stop_tokens, max_num_tokens, llm_model_type, sampler_params, ...) is kept. The thought channel is added later by
tools/add_thought_channel.py, so `channels` stays as the bundle has it (none).
Writes OUT.pbtext (text_format, utf-8) and OUT.pbtext.check.json (changed fields, template sha256, parse-back equality, the kept
stop tokens / max_num_tokens / model type). Never overwrites. Run with a python that has litert-lm-builder 0.18.0 (its proto has
supports_thinking). Exit 0 = PASS (the changed fields are exactly the expected ones).
"""
import hashlib
import json
import sys
from pathlib import Path

from google.protobuf import text_format
from litert_lm_builder import litertlm_core
from litert_lm_builder import litertlm_header_schema_py_generated as schema
from litert_lm_builder.runtime.proto import llm_metadata_pb2


def llm_section_bytes(bundle):
    with open(bundle, "rb") as f:
        prefix = f.read(32)
        if prefix[:8] != litertlm_core.HEADER_MAGIC_BYTES:
            raise ValueError("not a .litertlm bundle")
        off = litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET
        end = int.from_bytes(prefix[off:off + 8], "little")
        f.seek(0)
        head = f.read(end)
    meta = schema.LiteRTLMMetaData.GetRootAs(head[litertlm_core.HEADER_BEGIN_BYTE_OFFSET:], 0)
    secs = []
    for i in range(meta.SectionMetadata().ObjectsLength()):
        s = meta.SectionMetadata().Objects(i)
        secs.append((litertlm_core.any_section_data_type_to_string(s.DataType()), s.BeginOffset(), s.EndOffset()))
    llm = [s for s in secs if s[0] == "LlmMetadataProto"]
    if len(llm) != 1:
        raise ValueError(f"expected one LlmMetadataProto section, found {len(llm)}")
    with open(bundle, "rb") as f:
        f.seek(llm[0][1])
        return f.read(llm[0][2] - llm[0][1])


def state(msg, field):
    value = getattr(msg, field.name)
    if field.is_repeated:
        return [v.SerializeToString(deterministic=True) if hasattr(v, "SerializeToString") else v for v in value]
    present = msg.HasField(field.name) if field.has_presence else None
    if field.message_type is not None:
        return (present, value.SerializeToString(deterministic=True) if present else None)
    return (present, value)


def main():
    no_start = "--no-start-token" in sys.argv
    argv = [a for a in sys.argv[1:] if a != "--no-start-token"]
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    bundle, tmpl_path, out = Path(argv[0]), Path(argv[1]), Path(argv[2])
    check = out.with_name(out.name + ".check.json")
    for p in (out, check):
        if p.exists():
            print(f"refusing to overwrite {p}", file=sys.stderr)
            return 2
    raw = llm_section_bytes(bundle)
    orig = llm_metadata_pb2.LlmMetadata()
    orig.ParseFromString(raw)
    tmpl_bytes = tmpl_path.read_bytes()
    new = llm_metadata_pb2.LlmMetadata()
    new.CopyFrom(orig)
    new.jinja_prompt_template = tmpl_bytes.decode("utf-8")
    if no_start:
        new.ClearField("start_token")
    new.supports_thinking = True
    expected = ["jinja_prompt_template", "supports_thinking"] + (["start_token"] if no_start and orig.HasField("start_token") else [])
    text = text_format.MessageToString(new, as_utf8=True)
    back = text_format.Parse(text, llm_metadata_pb2.LlmMetadata())
    changed = [f.name for f in sorted(llm_metadata_pb2.LlmMetadata.DESCRIPTOR.fields, key=lambda f: f.number)
               if state(orig, f) != state(new, f)]
    rec = {"bundle": str(bundle), "orig_llm_metadata_sha256": hashlib.sha256(raw).hexdigest(),
           "orig_start_token": text_format.MessageToString(orig.start_token, as_one_line=True) if orig.HasField("start_token") else "unset",
           "no_start_token": no_start,
           "template": str(tmpl_path), "template_bytes": len(tmpl_bytes),
           "template_sha256": hashlib.sha256(tmpl_bytes).hexdigest(),
           "template_in_message_identical": back.jinja_prompt_template.encode("utf-8") == tmpl_bytes,
           "parse_back_identical": back.SerializeToString(deterministic=True) == new.SerializeToString(deterministic=True),
           "changed_fields": changed, "expected_changed_fields": sorted(expected),
           "changed_fields_equal_expected": sorted(changed) == sorted(expected),
           "kept": {f: (text_format.MessageToString(getattr(orig, f), as_one_line=True) if orig.HasField(f) else "unset")
                    for f in ("llm_model_type", "prompt_templates", "sampler_params")},
           "stop_tokens": [text_format.MessageToString(s, as_one_line=True) for s in orig.stop_tokens],
           "channels": [text_format.MessageToString(c, as_one_line=True) for c in orig.channels],
           "max_num_tokens": orig.max_num_tokens,
           "new_pbtext_bytes": len(text.encode("utf-8")),
           "new_pbtext_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
    ok = rec["template_in_message_identical"] and rec["parse_back_identical"] and rec["changed_fields_equal_expected"]
    rec["status"] = "PASS" if ok else "FAIL"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    check.write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
    print(rec["status"], json.dumps({k: rec[k] for k in ("changed_fields", "orig_start_token", "template_sha256",
                                                         "max_num_tokens", "stop_tokens", "kept")}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
