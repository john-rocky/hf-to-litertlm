#!/usr/bin/env python3
"""set_activation_type_inplace.py: change the activation dtype a .litertlm declares, in place, nothing else.

  python scripts/set_activation_type_inplace.py SRC.litertlm DST.litertlm --activation fp16 [--dry]

DST = a copy of SRC (`cp -c`, an APFS clone, on macOS; `cp --reflink=auto` elsewhere) whose header says VALUE in the
`prefer_activation_type` item of every TFLiteModel section; every section's bytes, the magic and the version stay where
they are. LiteRT-LM reads the item as the GPU executor's activation precision. Unlike scripts/set_activation_type.py,
which unpacks and packs the bundle again (a new uuid and creation time in the header, the sections laid out anew), this
edits only the header's flatbuffer, so the output is the same bytes for the same input. It only changes an existing
item; a section without one stops the run (use scripts/set_activation_type.py for that).

The flatbuffer is packed again with litert_lm_builder's own generated object API (as litert_lm_builder packs it).
Checks, all before DST is written:
  - SRC starts with the magic bytes; the gap between its header end and its first section is zeros;
  - SRC's header round-trips (unpacked and packed again without the edit = the same decoded content);
  - the new header fits before the first section (sections sit on 16 KiB blocks: nothing moves);
  - the decoded new header equals SRC's but for the edited item values (system metadata, section data types,
    offsets and every other item the same).
Written to DST after the copy: bytes 24..32 (the header end), the new flatbuffer from byte 32, zeros up to the first
section. DST is read back and checked again; the first 64 KiB of the first section are compared with SRC's.
Needs litert-lm-builder and flatbuffers (run with 0.19.0 and 25.12.19).
"""
import argparse
import os
import subprocess
import sys

import flatbuffers
from litert_lm_builder import litertlm_core as core
from litert_lm_builder import litertlm_header_schema_py_generated as schema

# The generated module defines the union creator as VdataCreator, but KeyValuePairT._UnPack calls VDataCreator
# (a NameError on the first InitFromObj with litert_lm_builder 0.16.1); the builder itself never unpacks, so only the
# object-API read path meets it.
if not hasattr(schema, "VDataCreator"):
    schema.VDataCreator = schema.VdataCreator

ITEM = b"prefer_activation_type"
HB = core.HEADER_BEGIN_BYTE_OFFSET  # 32
HE = core.HEADER_END_LOCATION_BYTE_OFFSET  # 24


def read_header(path):
    with open(path, "rb") as f:
        head = f.read(HB)
        if head[:8] != core.HEADER_MAGIC_BYTES:
            raise SystemExit(f"{path}: not a .litertlm (magic {head[:8]!r})")
        end = int.from_bytes(head[HE:HE + 8], "little")
        f.seek(HB)
        fb = f.read(end - HB)
    return head, end, fb


def decode(fb):
    return schema.LiteRTLMMetaDataT.InitFromObj(schema.LiteRTLMMetaData.GetRootAs(fb, 0))


def b(x):
    return bytes(x) if isinstance(x, (bytes, bytearray)) else (x.encode() if isinstance(x, str) else x)


def kv_tuple(kv):
    return (b(kv.key), kv.valueType, b(getattr(kv.value, "value", None)))


def summary(t):
    sysm = [kv_tuple(kv) for kv in (t.systemMetadata.entries or [])] if t.systemMetadata else []
    secs = [(o.dataType, o.beginOffset, o.endOffset, [kv_tuple(kv) for kv in (o.items or [])])
            for o in t.sectionMetadata.objects]
    return sysm, secs


def pack(t):
    bld = flatbuffers.Builder(core.BLOCK_SIZE)
    bld.Finish(t.Pack(bld))
    return bytes(bld.Output())


def edit(t, value):
    changed = []
    for i, o in enumerate(t.sectionMetadata.objects):
        if o.dataType != schema.AnySectionDataType.TFLiteModel:
            continue
        hit = False
        for kv in o.items or []:
            if b(kv.key) == ITEM:
                if kv.valueType != schema.VData.StringValue:
                    raise SystemExit(f"section {i}: {ITEM!r} is not a string (type {kv.valueType})")
                changed.append((i, b(kv.value.value), value.encode()))
                kv.value.value = value.encode()
                hit = True
        if not hit:
            raise SystemExit(f"section {i} (TFLiteModel) has no {ITEM!r} item: this tool only edits an existing one")
    if not changed:
        raise SystemExit("no TFLiteModel section")
    return changed


def clone(src, dst):
    if sys.platform == "darwin":
        subprocess.run(["cp", "-c", src, dst], check=True)
    else:
        subprocess.run(["cp", "--reflink=auto", src, dst], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--activation", required=True)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    head, end, fb = read_header(a.src)
    t0 = decode(fb)
    s0 = summary(t0)
    if summary(decode(pack(t0))) != s0:
        raise SystemExit("the source header does not round-trip through the object API")
    first = min(o.beginOffset for o in t0.sectionMetadata.objects)
    with open(a.src, "rb") as f:
        f.seek(end)
        gap = f.read(first - end)
    if gap.strip(b"\0"):
        raise SystemExit(f"the source's gap {end}..{first} is not zeros")
    t1 = decode(fb)
    changed = edit(t1, a.activation)
    want = summary(t1)
    fb1 = pack(t1)
    new_end = HB + len(fb1)
    if new_end > first:
        raise SystemExit(f"the new header ends at {new_end} > the first section at {first}")
    got = summary(decode(fb1))
    if got != want:
        raise SystemExit("the packed header does not decode to the edited content")
    # every difference from the source must be one of the edited items
    d_sys = s0[0] != got[0]
    d_sec = [(i, x, y) for i, (x, y) in enumerate(zip(s0[1], got[1])) if x != y]
    for i, x, y in d_sec:
        if x[:3] != y[:3] or len(x[3]) != len(y[3]):
            raise SystemExit(f"section {i}: type / offsets / item count changed")
        for kx, ky in zip(x[3], y[3]):
            if kx != ky and not (kx[0] == ITEM and ky[0] == ITEM and kx[1] == ky[1]):
                raise SystemExit(f"section {i}: item {kx[0]!r} changed")
    if d_sys or len(s0[1]) != len(got[1]):
        raise SystemExit("system metadata or section count changed")
    print(f"src {a.src}: header end {end}, first section at {first}, {len(s0[1])} sections")
    for i, old, new in changed:
        print(f"  section {i} (TFLiteModel): {ITEM.decode()} {old!r} -> {new!r}")
    print(f"new header end {new_end} ({len(fb1)} B flatbuffer, source {len(fb)} B); differing sections {[i for i, _, _ in d_sec]}")
    if a.dry:
        print("DRY ok")
        return 0
    if os.path.exists(a.dst):
        raise SystemExit(f"{a.dst} exists")
    clone(a.src, a.dst)
    with open(a.dst, "r+b") as f:
        f.seek(HE)
        f.write(new_end.to_bytes(8, "little"))
        f.seek(HB)
        f.write(fb1)
        f.write(b"\0" * (first - new_end))
    h2, e2, fb2 = read_header(a.dst)
    if h2[:HE] != head[:HE] or summary(decode(fb2)) != want or e2 != new_end:
        os.rename(a.dst, a.dst + ".bad")
        raise SystemExit("read-back of DST differs (renamed .bad)")
    with open(a.src, "rb") as f1, open(a.dst, "rb") as f2:
        f1.seek(first)
        f2.seek(first)
        if f1.read(65536) != f2.read(65536):
            os.rename(a.dst, a.dst + ".bad")
            raise SystemExit("the first section's first 64 KiB differ (renamed .bad)")
    print(f"WROTE {a.dst} (size {os.path.getsize(a.dst)} = source {os.path.getsize(a.src)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
