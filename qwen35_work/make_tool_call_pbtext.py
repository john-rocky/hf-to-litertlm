#!/usr/bin/env python3
"""make_tool_call_pbtext.py ORIG_META_DIR OFFICIAL.pbtext OUT.pbtext - the LlmMetadata of a Qwen3.5 bundle with tool-call parsing.

Starts from the bundle's LlmMetadata section bytes (ORIG_META_DIR/LlmMetadataProto.bin, written by dump_llm_metadata.py)
and changes exactly two fields:
  tool_call_config           = the tool_call_config message of OFFICIAL.pbtext, copied whole: LiteRT-LM's
                               models/qwen3_5/LlmMetadataProto.pbtext at v0.19.0 (commit 771b06e1, sha256 2bdb4bcb...,
                               checked): fence <tool_call> / </tool_call> and the regex syntax for
                               <function=...> / <parameter=...>
  supports_function_calling  = true
Everything else (the chat template, stop tokens, channels, sampler, max_num_tokens, the model type, ...) is kept. The
official file's other fields (its template, pad_token, sampler, min_runtime_version) are not taken.

Refuses (exit 3) when OFFICIAL.pbtext is not the pinned file, when the bundle's llm_model_type is not generic_model
(LiteRT-LM 0.19.0 takes a tool_call_config for generic_model only) or when it already has a tool_call_config.
Writes OUT.pbtext (text_format, utf-8) and OUT.pbtext.check.json (changed fields, the four tool-call strings, parse-back
equality). Never overwrites. Run with litert-lm-builder 0.19.0 (the first builder whose proto has tool_call_config):
  python -I make_tool_call_pbtext.py OUT.meta official.pbtext OUT.pbtext
"""
import hashlib
import json
import sys
from pathlib import Path

from google.protobuf import text_format
from litert_lm_builder.runtime.proto import llm_metadata_pb2

OFFICIAL_SHA256 = '2bdb4bcbd0f61abd4bddeb3c32e76d172b9049281eff9b692aa4b4b965b44229'
EXPECTED = ['supports_function_calling', 'tool_call_config']


def state(msg, field):
    value = getattr(msg, field.name)
    if field.is_repeated:
        return [v.SerializeToString(deterministic=True) if hasattr(v, 'SerializeToString') else v for v in value]
    present = msg.HasField(field.name) if field.has_presence else None
    if field.message_type is not None:
        return (present, value.SerializeToString(deterministic=True) if present else None)
    return (present, value)


def tool_call_strings(msg):
    t = msg.tool_call_config
    return {'fence_start': t.fence_start, 'fence_end': t.fence_end,
            'regex_syntax.tool_call_regex': t.regex_syntax.tool_call_regex,
            'regex_syntax.argument_regex': t.regex_syntax.argument_regex}


def main():
    if len(sys.argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2
    orig_dir, official_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    check = out.with_name(out.name + '.check.json')
    for p in (out, check):
        if p.exists():
            print(f'refusing to overwrite {p}', file=sys.stderr)
            return 2
    official_bytes = official_path.read_bytes()
    official_sha = hashlib.sha256(official_bytes).hexdigest()
    if official_sha != OFFICIAL_SHA256:
        print(f'{official_path}: sha256 {official_sha}, expected {OFFICIAL_SHA256} (models/qwen3_5 at v0.19.0)', file=sys.stderr)
        return 3
    official = text_format.Parse(official_bytes.decode('utf-8'), llm_metadata_pb2.LlmMetadata())
    raw = (orig_dir / 'LlmMetadataProto.bin').read_bytes()
    orig = llm_metadata_pb2.LlmMetadata()
    orig.ParseFromString(raw)
    model_type = orig.llm_model_type.WhichOneof('model_type') if orig.HasField('llm_model_type') else None
    if model_type != 'generic_model':
        print(f'llm_model_type is {model_type}, not generic_model: LiteRT-LM 0.19.0 refuses a tool_call_config there',
              file=sys.stderr)
        return 3
    if orig.HasField('tool_call_config'):
        print('the bundle already has a tool_call_config', file=sys.stderr)
        return 3
    new = llm_metadata_pb2.LlmMetadata()
    new.CopyFrom(orig)
    new.tool_call_config.CopyFrom(official.tool_call_config)
    new.supports_function_calling = True
    text = text_format.MessageToString(new, as_utf8=True)
    back = text_format.Parse(text, llm_metadata_pb2.LlmMetadata())
    changed = [f.name for f in sorted(llm_metadata_pb2.LlmMetadata.DESCRIPTOR.fields, key=lambda f: f.number)
               if state(orig, f) != state(new, f)]
    rec = {'orig_meta_dir': str(orig_dir), 'orig_llm_metadata_sha256': hashlib.sha256(raw).hexdigest(),
           'official': str(official_path), 'official_sha256': official_sha,
           'tool_call_config': tool_call_strings(new),
           'tool_call_config_equals_official': new.tool_call_config.SerializeToString(deterministic=True)
           == official.tool_call_config.SerializeToString(deterministic=True),
           'parse_back_identical': back.SerializeToString(deterministic=True) == new.SerializeToString(deterministic=True),
           'changed_fields': changed, 'changed_fields_equal_expected': sorted(changed) == EXPECTED,
           'supports_function_calling_before': orig.supports_function_calling if orig.HasField('supports_function_calling') else 'unset'}
    rec['PASS'] = rec['tool_call_config_equals_official'] and rec['parse_back_identical'] and rec['changed_fields_equal_expected']
    out.write_text(text)
    check.write_text(json.dumps(rec, indent=1, ensure_ascii=False) + '\n')
    print(f"{out}: changed {changed}; supports_function_calling {rec['supports_function_calling_before']} -> true; "
          f"tool_call_config from {official_path.name} (sha256 {official_sha[:8]}...); PASS {rec['PASS']}")
    return 0 if rec['PASS'] else 1


if __name__ == '__main__':
    sys.exit(main())
