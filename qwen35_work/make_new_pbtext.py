#!/usr/bin/env python3
"""make_new_pbtext.py ORIG_META_DIR TEMPLATE.jinja OUT.pbtext [--template-only] - the LlmMetadata for one Qwen3.5 bundle.

--template-only (convert_qwen35_9b.sh passes it) changes jinja_prompt_template only
(the 0.8B / 2B form: no channels, supports_thinking / supports_function_calling left as the bundle has them); without it the
four fields below (the litert-community/Qwen3.5-4B bundle's metadata form).

Starts from the bundle's LlmMetadata section bytes (ORIG_META_DIR/LlmMetadataProto.bin, written by
dump_llm_metadata.py) and changes exactly four fields:
  jinja_prompt_template      = TEMPLATE.jinja, byte for byte
  channels                   = [{channel_name: "thought", start: "<think>\\n", end: "\\n</think>"}]
                               (LiteRT-LM origin/main models/qwen3_5/LlmMetadataProto.pbtext)
  supports_thinking          = true
  supports_function_calling  = false  (generic_model / fast_vlm have no tool-call parser)
Everything else (stop_tokens, max_num_tokens, llm_model_type, prompt_templates, sampler_params, ...) is kept.
Writes OUT.pbtext (text_format, utf-8) and OUT.pbtext.check.json (changed fields, template sha256, parse-back
equality). Never overwrites. Run: uv run --no-project --with litert-lm==0.18.0 python -I make_new_pbtext.py ...
"""
import hashlib
import json
import sys
from pathlib import Path

from google.protobuf import text_format
from litert_lm_builder.runtime.proto import llm_metadata_pb2

EXPECTED = ['jinja_prompt_template', 'channels', 'supports_thinking', 'supports_function_calling']


def state(msg, field):
    value = getattr(msg, field.name)
    if field.is_repeated:
        return [v.SerializeToString(deterministic=True) if hasattr(v, 'SerializeToString') else v for v in value]
    present = msg.HasField(field.name) if field.has_presence else None
    if field.message_type is not None:
        return (present, value.SerializeToString(deterministic=True) if present else None)
    return (present, value)


def main():
    template_only = '--template-only' in sys.argv
    if template_only:
        sys.argv.remove('--template-only')
    if len(sys.argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2
    orig_dir, tmpl_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    check = out.with_name(out.name + '.check.json')
    for p in (out, check):
        if p.exists():
            print(f'refusing to overwrite {p}', file=sys.stderr)
            return 2
    raw = (orig_dir / 'LlmMetadataProto.bin').read_bytes()
    orig = llm_metadata_pb2.LlmMetadata()
    orig.ParseFromString(raw)
    tmpl_bytes = tmpl_path.read_bytes()
    new = llm_metadata_pb2.LlmMetadata()
    new.CopyFrom(orig)
    new.jinja_prompt_template = tmpl_bytes.decode('utf-8')
    if not template_only:  # --template-only keeps the other three
        del new.channels[:]
        ch = new.channels.add()
        ch.channel_name = 'thought'
        ch.start = '<think>\n'
        ch.end = '\n</think>'
        new.supports_thinking = True
        new.supports_function_calling = False
    text = text_format.MessageToString(new, as_utf8=True)
    back = text_format.Parse(text, llm_metadata_pb2.LlmMetadata())
    changed = [f.name for f in sorted(llm_metadata_pb2.LlmMetadata.DESCRIPTOR.fields, key=lambda f: f.number)
               if state(orig, f) != state(new, f)]
    rec = {'orig_meta_dir': str(orig_dir), 'orig_llm_metadata_sha256': hashlib.sha256(raw).hexdigest(),
           'template': str(tmpl_path), 'template_bytes': len(tmpl_bytes),
           'template_sha256': hashlib.sha256(tmpl_bytes).hexdigest(),
           'template_in_message_identical': back.jinja_prompt_template.encode('utf-8') == tmpl_bytes,
           'parse_back_identical': back.SerializeToString(deterministic=True) == new.SerializeToString(deterministic=True),
           'changed_fields': changed, 'changed_fields_equal_expected': sorted(changed) == sorted(['jinja_prompt_template'] if template_only else EXPECTED),
           'kept': {f: text_format.MessageToString(getattr(orig, f), as_one_line=True) if orig.HasField(f) else 'unset'
                    for f in ('llm_model_type', 'prompt_templates', 'sampler_params', 'start_token', 'min_runtime_version')
                    if f != 'min_runtime_version'},
           'stop_tokens': [text_format.MessageToString(s, as_one_line=True) for s in orig.stop_tokens],
           'max_num_tokens': orig.max_num_tokens,
           'min_runtime_version_set': new.HasField('min_runtime_version'),
           'new_pbtext_bytes': len(text.encode('utf-8')),
           'new_pbtext_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest()}
    ok = rec['template_in_message_identical'] and rec['parse_back_identical'] and rec['changed_fields_equal_expected'] \
        and not rec['min_runtime_version_set']
    rec['status'] = 'PASS' if ok else 'FAIL'
    out.write_text(text, encoding='utf-8')
    check.write_text(json.dumps(rec, indent=1, ensure_ascii=False) + '\n')
    print(rec['status'], json.dumps({k: rec[k] for k in ('changed_fields', 'template_bytes', 'template_sha256',
                                                         'new_pbtext_bytes', 'max_num_tokens', 'stop_tokens')}))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
