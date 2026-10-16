#!/usr/bin/env python3
"""dump_llm_metadata.py BUNDLE OUT_DIR - read a .litertlm header and write its LlmMetadata without unpacking weights.

Writes OUT_DIR/LlmMetadataProto.pbtext (text_format, utf-8), OUT_DIR/LlmMetadataProto.bin (raw section bytes),
OUT_DIR/template.jinja (the jinja_prompt_template field, exact bytes) and OUT_DIR/sections.json (section list:
index, type, begin, end, bytes, per-section metadata). Reads only the header and the LlmMetadata section.
OUT_DIR must not exist. Run with the builder the bundle is packed with:
  uv run --no-project --with litert-lm==0.18.0 python -I dump_llm_metadata.py BUNDLE OUT_DIR
"""
import hashlib
import json
import os
import sys
from pathlib import Path

from google.protobuf import text_format
from litert_lm_builder import litertlm_core
from litert_lm_builder import litertlm_header_schema_py_generated as schema
from litert_lm_builder import litertlm_peek
from litert_lm_builder.runtime.proto import llm_metadata_pb2


def main():
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    bundle, out = Path(sys.argv[1]), Path(sys.argv[2])
    if out.exists():
        print(f'refusing to overwrite {out}', file=sys.stderr)
        return 2
    size = os.path.getsize(bundle)
    with open(bundle, 'rb') as f:
        prefix = f.read(32)
        if prefix[:8] != litertlm_core.HEADER_MAGIC_BYTES:
            raise ValueError('not a .litertlm bundle')
        off = litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET
        header_end = int.from_bytes(prefix[off:off + 8], 'little')
        f.seek(0)
        head = f.read(header_end)
    meta = schema.LiteRTLMMetaData.GetRootAs(head[litertlm_core.HEADER_BEGIN_BYTE_OFFSET:], 0)
    sections = []
    for i in range(meta.SectionMetadata().ObjectsLength()):
        s = meta.SectionMetadata().Objects(i)
        try:
            kind = litertlm_core.any_section_data_type_to_string(s.DataType())
        except ValueError:
            kind = 'Unknown:' + str(s.DataType())
        sections.append({'index': i, 'type': kind, 'begin': s.BeginOffset(), 'end': s.EndOffset(),
                         'bytes': s.EndOffset() - s.BeginOffset(),
                         'metadata': [litertlm_peek.kvp_to_dict(s.Items(j)) for j in range(s.ItemsLength())]})
    llm = [s for s in sections if s['type'] == 'LlmMetadataProto']
    if len(llm) != 1:
        raise ValueError(f'expected one LlmMetadataProto section, found {len(llm)}')
    with open(bundle, 'rb') as f:
        f.seek(llm[0]['begin'])
        raw = f.read(llm[0]['bytes'])
    msg = llm_metadata_pb2.LlmMetadata()
    msg.ParseFromString(raw)
    out.mkdir(parents=True)
    (out / 'LlmMetadataProto.bin').write_bytes(raw)
    (out / 'LlmMetadataProto.pbtext').write_text(text_format.MessageToString(msg, as_utf8=True), encoding='utf-8')
    tmpl = msg.jinja_prompt_template.encode('utf-8')
    (out / 'template.jinja').write_bytes(tmpl)
    info = {'bundle': str(bundle), 'file_bytes': size, 'format_version_prefix_hex': prefix[8:20].hex(),
            'header_end': header_end, 'sections': sections,
            'llm_metadata_bytes': len(raw), 'llm_metadata_sha256': hashlib.sha256(raw).hexdigest(),
            'template_bytes': len(tmpl), 'template_sha256': hashlib.sha256(tmpl).hexdigest(),
            'set_fields': [fd.name for fd, _ in msg.ListFields()]}
    (out / 'sections.json').write_text(json.dumps(info, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps({k: info[k] for k in ('llm_metadata_bytes', 'template_bytes', 'template_sha256', 'set_fields')}))
    for s in sections:
        print(f"  [{s['index']}] {s['type']} {s['bytes']} B {s['metadata']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
