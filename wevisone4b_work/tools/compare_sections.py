#!/usr/bin/env python3
"""Print every section of a rebuilt WeVisOne-VL-4B .litertlm next to a reference's.

    PY_EXPORT -I tools/compare_sections.py NEW.litertlm REF

REF is a .litertlm file, or one of these names for the sections hashed from the files themselves:
  published        WeVisOne-VL-4B_int4.litertlm of litert-community/WeVisOne-VL-4B (main 2fcc2c19),
                   3,507,874,304 B, sha256 1c14c17a7e8bd688d888300a68e2f7512b255a275a444a3f22bde395cad4b12f
  before_retrofit  that file before tools/gpu_graph/gpu_graph_retrofit.py rewrote its PREFILL_DECODE (not published),
                   3,507,071,488 B, sha256 8643ea9d4fcdc7024f48377c293ba3217e1bd522de63e66c5e43322ef9ede2c4

For each section, in file order: the section type, its header items (model_type, prefer_activation_type), the bytes
and the sha256 of the bytes as stored in the container, for NEW and for REF, and SAME or DIFF. Every pack writes a new
uuid and creation_timestamp into the file header, so the whole-file sha256 of a rebuild differs from the reference.
With a REF file, the bytes outside the sections (header and padding) are compared as well, after NEW's uuid and
creation_timestamp are replaced by REF's. Exit 0 when every section is SAME (and, with a REF file, no other byte
differs), 1 otherwise. Reads the header with tools/gpu_graph/gpu_graph_retrofit.py's bundle_sections().
"""
import hashlib
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("gpu_graph_retrofit", ROOT / "tools" / "gpu_graph" / "gpu_graph_retrofit.py")
GG = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(GG)

_EMB = {"model_type": "tf_lite_embedder"}
_PD = {"model_type": "tf_lite_prefill_decode", "prefer_activation_type": "fp32"}
_ENC = {"model_type": "tf_lite_vision_encoder"}
_ADA = {"model_type": "tf_lite_vision_adapter"}
_HEAD = [
    ("LlmMetadataProto", {}, 1449, "276d33b5aee13fe1603b39ef329710212c110783ec952a7dd2f58676d5ab65ba"),
    ("ExecutorMetadataProto", {}, 4893, "12ccc482d14139da43da78d885632addea0abef495b2417b48a5a530f13bc7c8"),
    ("HF_Tokenizer_Zlib", {}, 2148495, "f5c11ba97666b975b692374bd1c8807c494311f42ecb799699128902724cd636"),
    ("TFLiteModel", _EMB, 392695648, "9494be473647957366b44be45f7bb5723a35167fb72ac156cdd32c8a0b1b1c81"),
]
_TAIL = [
    ("TFLiteModel", _ENC, 611814736, "26adbb94502caed3a3ca8d02f73269393593465b6305567b18e01818659750e4"),
    ("TFLiteModel", _ADA, 218245632, "88c674a4139bb8c9ed3e91b06baad110f6871f331f8bc1eaf4258a04b95eb793"),
]
RECORDED = {
    "published": {
        "file": (3507874304, "1c14c17a7e8bd688d888300a68e2f7512b255a275a444a3f22bde395cad4b12f"),
        "sections": _HEAD + [("TFLiteModel", _PD, 2282878832,
                              "2235d43a6fe12a75960e003743a1c6c446e6903f33fcf4c5717e88585b652d18")] + _TAIL},
    "before_retrofit": {
        "file": (3507071488, "8643ea9d4fcdc7024f48377c293ba3217e1bd522de63e66c5e43322ef9ede2c4"),
        "sections": _HEAD + [("TFLiteModel", _PD, 2282076016,
                              "9df0e71865add909872f35aeb7fe5f316b2fbe0dfc67f39ffa7415159af809c5")] + _TAIL},
}


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def system_metadata(path):
    """The header's system metadata as {key: string value} (uuid, creation_timestamp, Authors, ...)."""
    from litert_lm_builder import litertlm_core
    from litert_lm_builder import litertlm_header_schema_py_generated as hs
    with open(path, "rb") as f:
        head = f.read(1 << 16)
    end = int.from_bytes(head[litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET:
                              litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET + 8], "little")
    meta = hs.LiteRTLMMetaData.GetRootAs(bytearray(head[litertlm_core.HEADER_BEGIN_BYTE_OFFSET:end]), 0)
    out = {}
    sm = meta.SystemMetadata()
    for j in range(sm.EntriesLength() if sm else 0):
        kv = sm.Entries(j)
        key = kv.Key().decode() if kv.Key() else ""
        if kv.ValueType() == hs.VData.StringValue:
            sv = hs.StringValue()
            sv.Init(kv.Value().Bytes, kv.Value().Pos)
            out[key] = sv.Value().decode() if sv.Value() else ""
        else:
            out[key] = f"<vtype {kv.ValueType()}>"
    return out


def gaps(rows, size):
    """[begin, end) ranges of the file that no section covers (the header and the padding)."""
    out, pos = [], 0
    for r in sorted(rows, key=lambda r: r["begin"]):
        if r["begin"] > pos:
            out.append((pos, r["begin"]))
        pos = max(pos, r["end"])
    if size > pos:
        out.append((pos, size))
    return out


def read_ranges(path, ranges):
    buf = bytearray()
    with open(path, "rb") as f:
        for b, e in ranges:
            f.seek(b)
            buf += f.read(e - b)
    return buf


def outside_sections(new, ref, rows_new, rows_ref):
    """Bytes outside the sections, NEW against REF, with NEW's uuid and creation_timestamp set to REF's."""
    size_new, size_ref = os.path.getsize(new), os.path.getsize(ref)
    layout_new = [(r["begin"], r["end"]) for r in rows_new]
    if size_new != size_ref or layout_new != [(r["begin"], r["end"]) for r in rows_ref]:
        return False, "the section offsets or the file size differ"
    ranges = gaps(rows_new, size_new)
    a, b = read_ranges(new, ranges), read_ranges(ref, ranges)
    sm_new, sm_ref = system_metadata(new), system_metadata(ref)
    notes = []
    for key in ("uuid", "creation_timestamp"):
        x, y = sm_new.get(key, "").encode(), sm_ref.get(key, "").encode()
        if x and y and x != y:
            if len(x) != len(y):
                notes.append(f"{key} lengths differ ({len(x)} / {len(y)})")
            else:
                a = a.replace(x, y)
    n = sum(p != q for p, q in zip(a, b))
    text = f"{sum(e - b for b, e in ranges):,} bytes in {len(ranges)} ranges, {n} differ after the uuid and " \
           f"creation_timestamp swap" + ("; " + "; ".join(notes) if notes else "")
    return n == 0 and not notes, text


def label(items):
    t = items.get("model_type", "")
    t = t[len("tf_lite_"):] if t.startswith("tf_lite_") else t
    act = items.get("prefer_activation_type")
    return t + (f" ({act})" if act else "")


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    new, ref = sys.argv[1], sys.argv[2]
    rows_new = GG.bundle_sections(new)
    if ref in RECORDED:
        rows_ref = [{"type": t, "items": i, "bytes": n, "sha256": h} for t, i, n, h in RECORDED[ref]["sections"]]
        ref_file = RECORDED[ref]["file"]
        print(f"reference: the sections recorded from the {ref} file ({ref_file[0]:,} B, sha256 {ref_file[1]})")
    else:
        rows_ref = GG.bundle_sections(ref)
        size, digest = os.path.getsize(ref), sha256_file(ref)
        named = [k for k, v in RECORDED.items() if v["file"] == (size, digest)]
        print(f"reference: {ref} ({size:,} B, sha256 {digest}" + (f" = the {named[0]} file)" if named else ")"))
    print(f"new:       {new} ({os.path.getsize(new):,} B, sha256 {sha256_file(new)})")
    same_all = len(rows_new) == len(rows_ref)
    for i in range(max(len(rows_new), len(rows_ref))):
        a = rows_new[i] if i < len(rows_new) else None
        b = rows_ref[i] if i < len(rows_ref) else None
        same = a is not None and b is not None and (a["type"], a["items"], a["bytes"], a["sha256"]) == \
            (b["type"], b["items"], b["bytes"], b["sha256"])
        same_all &= same
        r = a or b
        print(f"  {i} {r['type']:<22} {label(r['items']):<22} {'SAME' if same else 'DIFF'}")
        for who, x in (("new", a), ("ref", b)):
            if x is not None:
                print(f"      {who} {x['bytes']:>15,}  {x['sha256']}" +
                      ("" if x["type"] == r["type"] else f"  type {x['type']}") +
                      ("" if x["items"] == r["items"] else f"  items {x['items']}"))
    for who, path in (("new", new),) + ((("ref", ref),) if ref not in RECORDED else ()):
        print(f"  system metadata ({who}): {system_metadata(path)}")
    ok = same_all
    if ref not in RECORDED:
        same_rest, text = outside_sections(new, ref, rows_new, rows_ref)
        print(f"  outside the sections: {text}")
        ok &= same_rest
    print(f"sections: {sum(1 for _ in rows_new)} new / {len(rows_ref)} reference, "
          f"{'all SAME' if same_all else 'not all SAME'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
