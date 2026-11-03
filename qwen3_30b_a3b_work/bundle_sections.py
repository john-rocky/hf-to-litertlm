#!/usr/bin/env python3
"""bundle_sections.py FILE.litertlm [--json OUT.json] [--no-file-sha]

Prints a .litertlm's sections (index, data type, model_type, begin, end, bytes, sha256 of the section bytes), its
system metadata (uuid, creation_timestamp, ...) and the whole-file sha256. The header is read with
litert_lm_builder's generated flatbuffer schema (litert-lm-builder 0.19.0). Two files whose sections all have the same
sha256 hold the same model, tokenizer and metadata; the container header also carries a uuid and a creation
timestamp, which litert_lm_builder writes anew at every build (make_bundle.py --ids-from copies them from a file).
"""
import argparse
import hashlib
import json
import os
import sys

from litert_lm_builder import litertlm_core
from litert_lm_builder import litertlm_header_schema_py_generated as schema

# litert-lm-builder 0.19.0's generated schema defines VdataCreator, but its object API (KeyValuePairT._UnPack) calls
# VDataCreator (NameError): alias it so the object API can read a header.
if not hasattr(schema, "VDataCreator"):
    schema.VDataCreator = schema.VdataCreator

CHUNK = 64 << 20


def read_header(path):
    """The header of a .litertlm, or of the first bytes of one through the end of its header."""
    with open(path, "rb") as f:
        head = f.read(1 << 16)
        if head[:8] != litertlm_core.HEADER_MAGIC_BYTES:
            raise SystemExit(f"not a litertlm file: {path}")
        loc = litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET
        hdr_end = int.from_bytes(head[loc:loc + 8], "little")
        if hdr_end > len(head):
            f.seek(0)
            head = f.read(hdr_end + 64)
    if len(head) < hdr_end:
        raise SystemExit(f"{path} ends at byte {len(head)}, before the end of its header ({hdr_end})")
    meta = schema.LiteRTLMMetaData.GetRootAs(bytearray(head[litertlm_core.HEADER_BEGIN_BYTE_OFFSET:hdr_end]), 0)
    return schema.LiteRTLMMetaDataT.InitFromObj(meta)


def _str(v):
    return v.decode() if isinstance(v, bytes) else v


def kv(entries):
    return {_str(k.key): (_str(k.value.value) if k.value is not None else None) for k in (entries or [])}


def sha_range(path, begin, end):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(begin)
        left = end - begin
        while left > 0:
            b = f.read(min(CHUNK, left))
            if not b:
                raise IOError(f"short read in {path} at {end - left}")
            h.update(b)
            left -= len(b)
    return h.hexdigest()


def table(path, file_sha=True):
    meta = read_header(path)
    rows = []
    for i, o in enumerate(meta.sectionMetadata.objects):
        it = kv(o.items)
        rows.append({"index": i, "type": litertlm_core.any_section_data_type_to_string(o.dataType),
                     "model_type": it.get("model_type"), "items": it, "begin": o.beginOffset, "end": o.endOffset,
                     "bytes": o.endOffset - o.beginOffset, "sha256": sha_range(path, o.beginOffset, o.endOffset)})
    size = os.path.getsize(path)
    return {"file": os.path.abspath(path), "bytes": size, "system_metadata": kv(meta.systemMetadata.entries),
            "sections": rows, "sha256": sha_range(path, 0, size) if file_sha else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--json")
    ap.add_argument("--no-file-sha", action="store_true", help="skip the whole-file sha256")
    a = ap.parse_args()
    t = table(a.file, not a.no_file_sha)
    print(f"{t['file']}  {t['bytes']:,} B")
    for k, v in t["system_metadata"].items():
        print(f"  system_metadata {k} = {v}")
    for r in t["sections"]:
        print(f"  [{r['index']}] {r['type']:<18} {str(r['model_type'] or ''):<24} {r['bytes']:>16,} B  {r['sha256']}")
    if t["sha256"]:
        print(f"  file sha256 {t['sha256']}")
    if a.json:
        json.dump(t, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
