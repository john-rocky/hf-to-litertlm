#!/usr/bin/env python3
"""Prove a repack changed nothing but metadata.

Editing a bundle's LlmMetadata means unpack -> edit -> pack, which rewrites the
whole container. Before re-uploading, check that it really was metadata-only:
sha256 every section of the published file and of the repacked one and require
everything except LlmMetadataProto to match byte for byte. If the weights are
identical there is no requantization hiding in the repack, the model card's
measured numbers still stand, and the only thing that changes is the file's own
sha256 (which the repo's litertlm_manifest.json must then be regenerated for).

    python scripts/section_identity.py published.litertlm repacked.litertlm

Exit 0 = metadata-only, safe to upload. Exit 1 = a weight or tokenizer section
moved; do not upload.
"""
import hashlib
import sys

from litert_lm_builder import litertlm_core
from litert_lm_builder import litertlm_header_schema_py_generated as schema

HDR_BEGIN = litertlm_core.HEADER_BEGIN_BYTE_OFFSET
HDR_END_LOC = litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET


def sections(path):
    """[(section type, size, sha256)] in file order."""
    def fetch(start, end):
        with open(path, "rb") as f:
            f.seek(start)
            return f.read(end - start + 1)

    head = fetch(0, 4095)
    if head[:8] != litertlm_core.HEADER_MAGIC_BYTES:
        raise ValueError(f"not a litertlm file: {path}")
    hdr_end = int.from_bytes(head[HDR_END_LOC:HDR_END_LOC + 8], "little")
    if hdr_end > len(head):
        head = fetch(0, hdr_end + 64)
    meta = schema.LiteRTLMMetaData.GetRootAs(bytearray(head[HDR_BEGIN:hdr_end]), 0)

    out = []
    for i in range(meta.SectionMetadata().ObjectsLength()):
        so = meta.SectionMetadata().Objects(i)
        dtype = litertlm_core.any_section_data_type_to_string(so.DataType())
        digest = hashlib.sha256()
        begin, end = so.BeginOffset(), so.EndOffset()
        with open(path, "rb") as f:
            f.seek(begin)
            left = end - begin
            while left > 0:
                chunk = f.read(min(1 << 22, left))
                digest.update(chunk)
                left -= len(chunk)
        out.append((dtype, end - begin, digest.hexdigest()))
    return out


def compare(before, after):
    """(metadata_only, rows). rows = [(type, size_before, size_after, same)]."""
    a, b = sections(before), sections(after)
    if len(a) != len(b):
        return False, [("<section count>", len(a), len(b), False)]
    rows, ok = [], True
    for (da, sa, ha), (db, sb, hb) in zip(a, b):
        same = (da == db and sa == sb and ha == hb)
        if da != "LlmMetadataProto" and not same:
            ok = False
        rows.append((da, sa, sb, same))
    return ok, rows


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    ok, rows = compare(sys.argv[1], sys.argv[2])
    for dtype, size_a, size_b, same in rows:
        print(f"  {dtype:22} {size_a:>12} -> {size_b:>12}  {'SAME' if same else 'DIFF'}")
    print("metadata-only:", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
