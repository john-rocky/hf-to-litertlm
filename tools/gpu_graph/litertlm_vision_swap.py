#!/usr/bin/env python3
"""litertlm_vision_swap.py OLD.litertlm NEW.litertlm --venc VISION_ENCODER.tflite --vadp VISION_ADAPTER.tflite
                           [--clone] [--ids-from FILE] [--json OUT.json]

Writes a .litertlm that is OLD with only its two vision sections (tf_lite_vision_encoder, tf_lite_vision_adapter)
replaced by the given .tflite files, without unpacking anything else: every other section (LlmMetadata, tokenizer,
embedder, prefill_decode, ...) is copied byte for byte from its range in OLD, with OLD's section items (model_type,
prefer_activation_type, ...) and in OLD's order; the vision sections keep OLD's items and take the new files' bytes.
The header is written by litert_lm_builder 0.19.0's own LitertLmFileBuilder (its block alignment and layout); the
system metadata = OLD's entries with a new uuid and creation_timestamp (the builder always writes those two; a copied
uuid would make two different files look the same to anything keyed on it), so the whole-file sha256 of a rebuild
differs while every section is the same.
--ids-from FILE writes FILE's uuid and creation_timestamp instead. FILE is a .litertlm, or the first bytes of one
through the end of its header (`curl -r 0-65535` fetches the first 64 KiB; the tool stops if the header ends later).
Given the file a rebuild reproduces, the rebuild has that file's whole-file sha256 when its sections and layout are the
same; given OLD, the output depends on the inputs only, but two different files then carry one uuid. The check after
writing also compares the two values with FILE's.
After writing, every section of NEW is hashed and compared: the other sections must equal OLD's, the vision sections
must equal the new files; exit 1 otherwise (SWAP_OK / SWAP_FAIL). --json records both files' section tables (index,
type, model_type, items, begin, end, size, sha256).
--clone (NEW on the same APFS volume as OLD, the vision sections last in OLD): NEW starts as an APFS clone of OLD
(`cp -c`, no data copied) and only the header and the sections from the first vision section on are written; the
sections before it keep OLD's blocks, so NEW takes about the new vision bytes of extra disk instead of a whole file.
Each kept section must land at OLD's own offset (checked while writing: the builder's layout is sequential with block
alignment, so equal header size and equal earlier sections give equal offsets); padding the clone would otherwise
carry over (between the header and the first section, between and after the new sections) is zeroed and the file is
cut to its new length.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import litert_lm_builder.litertlm_builder as lbm
from litert_lm_builder import litertlm_core
from litert_lm_builder import litertlm_header_schema_py_generated as schema

# litert_lm_builder 0.19.0's generated header schema defines VdataCreator but its object API (KeyValuePairT._UnPack)
# calls VDataCreator (NameError); alias it in this process so the object API can read a header
if not hasattr(schema, "VDataCreator"):
    schema.VDataCreator = schema.VdataCreator

VISION = {"tf_lite_vision_encoder": "venc", "tf_lite_vision_adapter": "vadp"}
CHUNK = 64 << 20


def read_header(path):
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


def sval(kvp):
    v = kvp.value.value if kvp.value is not None else None
    return v.decode() if isinstance(v, bytes) else v


def items_of(obj):
    return {(k.key.decode() if isinstance(k.key, bytes) else k.key): sval(k) for k in (obj.items or [])}


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


def sha_file(path):
    return sha_range(path, 0, os.path.getsize(path))


def table(path):
    meta = read_header(path)
    rows = []
    for i, o in enumerate(meta.sectionMetadata.objects):
        it = items_of(o)
        rows.append({"index": i, "type": litertlm_core.any_section_data_type_to_string(o.dataType),
                     "model_type": it.get("model_type"), "items": it, "begin": o.beginOffset, "end": o.endOffset,
                     "size": o.endOffset - o.beginOffset, "sha256": sha_range(path, o.beginOffset, o.endOffset)})
    sysmeta = {(k.key.decode() if isinstance(k.key, bytes) else k.key): sval(k)
               for k in (meta.systemMetadata.entries or [])}
    return rows, sysmeta


def range_writer(path, begin, end):
    def w(stream):
        with open(path, "rb") as f:
            f.seek(begin)
            left = end - begin
            while left > 0:
                b = f.read(min(CHUNK, left))
                if not b:
                    raise IOError(f"short read in {path} at {end - left}")
                stream.write(b)
                left -= len(b)
    return w


def file_writer(path):
    return range_writer(path, 0, os.path.getsize(path))


def skip_writer(begin, size):
    """--clone: the section's bytes are already in place (cloned from OLD); only check the offset and move on"""
    def w(stream):
        if stream.tell() != begin:
            raise RuntimeError(f"clone layout mismatch: the builder put a kept section at {stream.tell()}, OLD has it at {begin}")
        stream.seek(begin + size)
    return w


def zero_range(f, a, b):
    if b <= a:
        return
    f.seek(a)
    left = b - a
    z = bytes(min(CHUNK, left))
    while left > 0:
        n = min(len(z), left)
        f.write(z[:n])
        left -= n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("out")
    ap.add_argument("--venc", required=True)
    ap.add_argument("--vadp", required=True)
    ap.add_argument("--clone", action="store_true")
    ap.add_argument("--ids-from", metavar="FILE")
    ap.add_argument("--json")
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"refusing to overwrite {a.out}")
    new = {"venc": a.venc, "vadp": a.vadp}
    t0 = time.time()
    meta = read_header(a.old)
    b = lbm.LitertLmFileBuilder()
    for kvp in (meta.systemMetadata.entries or []):
        m = lbm.Metadata.from_key_value_pair(kvp)
        if m.key not in ("uuid", "creation_timestamp"):
            b._system_metadata.append(m)
    ids = None
    if a.ids_from:
        ids = {m.key: m for m in map(lbm.Metadata.from_key_value_pair, read_header(a.ids_from).systemMetadata.entries or [])
               if m.key in ("uuid", "creation_timestamp")}
        if len(ids) != 2:
            raise SystemExit(f"--ids-from {a.ids_from}: no uuid and creation_timestamp in its system metadata")
        # the builder writes these two at build time (populate_system_metadata, after the other entries, uuid first);
        # write FILE's in the same place
        copied = [ids["uuid"], ids["creation_timestamp"]]
        lbm.populate_system_metadata = lambda md: [m for m in md if m.key not in ids] + copied
    plan = []
    objs = list(meta.sectionMetadata.objects)
    first_vision = min(i for i, o in enumerate(objs) if items_of(o).get("model_type") in VISION)
    if a.clone and any(items_of(o).get("model_type") not in VISION for o in objs[first_vision:]):
        raise SystemExit("--clone needs the vision sections last in OLD")
    for i, o in enumerate(objs):
        it = items_of(o)
        mt = it.get("model_type")
        md = [lbm.Metadata.from_key_value_pair(k) for k in (o.items or [])]
        if mt in VISION:
            src = new[VISION[mt]]
            writer = file_writer(src)
            plan.append((i, mt, "new", src))
        elif a.clone:
            writer = skip_writer(o.beginOffset, o.endOffset - o.beginOffset)
            plan.append((i, mt, "old (clone, kept in place)", f"{o.beginOffset}-{o.endOffset}"))
        else:
            writer = range_writer(a.old, o.beginOffset, o.endOffset)
            plan.append((i, mt, "old", f"{o.beginOffset}-{o.endOffset}"))
        b._sections.append(lbm._SectionObject(metadata=md, data_type=o.dataType, data_writer=writer, source_path=None))
    swapped = [p for p in plan if p[2] == "new"]
    if sorted(p[1] for p in swapped) != sorted(VISION):
        raise SystemExit(f"expected exactly the two vision sections, got {swapped}")
    for p in plan:
        print("section", *p, flush=True)
    tmp = a.out + ".partial"
    if a.clone:
        import subprocess
        if os.path.exists(tmp):
            os.remove(tmp)
        subprocess.run(["cp", "-c", a.old, tmp], check=True)  # APFS clone: no data copied
        with open(tmp, "r+b") as f:
            b.build(f)
            f.flush()  # the header is read back through another handle below: push the buffered writes to the OS first
            # zero what the clone would carry over: header tail .. first section, and the gaps among / after the new
            # sections; then cut the file at the end of the last section
            nm = read_header(tmp)
            so = list(nm.sectionMetadata.objects)
            loc = litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET
            f.seek(loc)
            hdr_end = int.from_bytes(f.read(8), "little")
            zero_range(f, hdr_end, so[0].beginOffset)
            for x, y in zip(so[first_vision - 1:], so[first_vision:]):
                zero_range(f, x.endOffset, y.beginOffset)
            f.flush()
            f.truncate(so[-1].endOffset)
    else:
        with open(tmp, "wb") as f:
            b.build(f)
    os.replace(tmp, a.out)
    t_write = time.time() - t0
    print(f"WROTE {a.out} {os.path.getsize(a.out)} B in {t_write:.1f} s", flush=True)

    old_rows, old_sys = table(a.old)
    out_rows, out_sys = table(a.out)
    ok = len(old_rows) == len(out_rows)
    checks = []
    for s, o in zip(old_rows, out_rows):
        if s["type"] != o["type"] or s["items"] != o["items"]:
            ok = False
            checks.append([s["index"], "items/type differ", s["items"], o["items"]])
            continue
        if s["model_type"] in VISION:
            want = sha_file(new[VISION[s["model_type"]]])
            good = o["sha256"] == want
            checks.append([s["index"], s["model_type"], "new file", good, o["size"], s["size"]])
        else:
            good = o["sha256"] == s["sha256"] and o["size"] == s["size"]
            checks.append([s["index"], s["model_type"] or s["type"], "= old", good, o["size"], s["size"]])
        ok = ok and good
    for c in checks:
        print("check", *c, flush=True)
    sys_same = {k: v for k, v in old_sys.items() if k not in ("uuid", "creation_timestamp")} == \
               {k: v for k, v in out_sys.items() if k not in ("uuid", "creation_timestamp")}
    ok = ok and sys_same
    print(f"system metadata (without uuid / timestamp) same: {sys_same}", flush=True)
    if ids is not None:
        want = {k: (m.value.decode() if isinstance(m.value, bytes) else m.value) for k, m in ids.items()}
        ids_same = all(out_sys.get(k) == v for k, v in want.items())
        ok = ok and ids_same
        print(f"uuid / creation_timestamp = {a.ids_from}'s: {ids_same}", flush=True)
    rec = {"old": a.old, "old_bytes": os.path.getsize(a.old), "out": a.out, "out_bytes": os.path.getsize(a.out),
           "venc": a.venc, "vadp": a.vadp, "clone": a.clone, "ids_from": a.ids_from,
           "write_s": round(t_write, 1), "total_s": round(time.time() - t0, 1),
           "old_sections": old_rows, "out_sections": out_rows, "old_system": old_sys, "out_system": out_sys,
           "checks": checks, "ok": ok}
    if a.json:
        json.dump(rec, open(a.json, "w"), indent=1)
    print("SWAP_OK" if ok else "SWAP_FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
