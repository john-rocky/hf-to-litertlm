#!/usr/bin/env python3
"""sections.py A.litertlm [B.litertlm] [--json OUT.json] - the section table of a .litertlm file; with B, the two files
compared section by section.

Per section: type, bytes, the sha256 of its raw bytes and its header items (model_type, prefer_activation_type, ...).
The header's system metadata is listed as well. The packer writes a new uuid and creation_timestamp into it on every
pack, so the file sha256 of a rebuild never equals the published one: compare sections. With B the verdict is
SAME_SECTIONS when the section lists match, every section's raw bytes and items are equal, and the system metadata is
equal apart from uuid and creation_timestamp; DIFFERENT otherwise. Exit 0 = SAME_SECTIONS (or a table of one file),
1 = DIFFERENT, 2 = usage. Needs litert-lm-builder (installed with litert-lm 0.18.0).
"""
import argparse
import hashlib
import json
import os
import struct
import sys

from litert_lm_builder import litertlm_core, litertlm_peek
from litert_lm_builder import litertlm_header_schema_py_generated as schema

PACK_KEYS = {"uuid", "creation_timestamp"}


def _items(obj, n, get):
  out = {}
  for j in range(n):
    d = litertlm_peek.kvp_to_dict(get(j))
    out[d["key"]] = {"value": d["value"], "type": d["value_type"]}
  return out


def read(path):
  with open(path, "rb") as f:
    if f.read(8) != litertlm_core.HEADER_MAGIC_BYTES:
      raise SystemExit(f"{path}: not a .litertlm file")
    version = ".".join(str(v) for v in struct.unpack("<III", f.read(12)))
    f.seek(litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET)
    end = struct.unpack("<Q", f.read(8))[0]
    f.seek(litertlm_core.HEADER_BEGIN_BYTE_OFFSET)
    md = schema.LiteRTLMMetaData.GetRootAs(f.read(end - litertlm_core.HEADER_BEGIN_BYTE_OFFSET), 0)
    sm = md.SystemMetadata()
    system = _items(sm, sm.EntriesLength(), sm.Entries) if sm is not None else {}
    sections = []
    table = md.SectionMetadata()
    for i in range(table.ObjectsLength()):
      s = table.Objects(i)
      sections.append({"index": i, "type": litertlm_core.any_section_data_type_to_string(s.DataType()),
                       "begin": s.BeginOffset(), "end": s.EndOffset(),
                       "items": _items(s, s.ItemsLength(), s.Items)})
    for s in sections:
      f.seek(s["begin"])
      left, h = s["end"] - s["begin"], hashlib.sha256()
      while left:
        chunk = f.read(min(left, 1 << 24))
        if not chunk:
          raise SystemExit(f"{path}: section {s['index']} ends past the end of the file")
        h.update(chunk)
        left -= len(chunk)
      s["bytes"] = s["end"] - s["begin"]
      s["sha256"] = h.hexdigest()
  return {"path": path, "bytes": os.path.getsize(path), "format_version": version, "system": system,
          "sections": sections}


def show(doc):
  print(f"{doc['path']}  {doc['bytes']:,} B  format {doc['format_version']}")
  for k, v in doc["system"].items():
    print(f"  system {k} = {v['value']!r}")
  for s in doc["sections"]:
    items = ", ".join(f"{k}={v['value']}" for k, v in s["items"].items())
    print(f"  [{s['index']}] {s['type']:24s} {s['bytes']:>15,} B  {s['sha256']}  {items}")


def compare(a, b):
  rows = []
  same = [s["type"] for s in a["sections"]] == [s["type"] for s in b["sections"]]
  for x, y in zip(a["sections"], b["sections"]):
    r = {"index": x["index"], "type": x["type"], "bytes": [x["bytes"], y["bytes"]],
         "sha256": [x["sha256"], y["sha256"]], "raw_equal": x["sha256"] == y["sha256"] and x["bytes"] == y["bytes"],
         "items_equal": x["items"] == y["items"]}
    same = same and r["raw_equal"] and r["items_equal"]
    rows.append(r)
  keys = sorted(set(a["system"]) | set(b["system"]))
  system = {k: {"a": a["system"].get(k, {}).get("value"), "b": b["system"].get(k, {}).get("value")} for k in keys}
  system_diff = [k for k in keys if a["system"].get(k) != b["system"].get(k)]
  same = same and all(k in PACK_KEYS for k in system_diff)
  return {"sections": rows, "system": system, "system_keys_differing": system_diff,
          "verdict": "SAME_SECTIONS" if same else "DIFFERENT"}


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("a")
  ap.add_argument("b", nargs="?")
  ap.add_argument("--json")
  args = ap.parse_args()
  a = read(args.a)
  show(a)
  doc = {"a": a}
  rc = 0
  if args.b:
    b = read(args.b)
    show(b)
    cmp = compare(a, b)
    for r in cmp["sections"]:
      print(f"  [{r['index']}] {r['type']:24s} raw {'==' if r['raw_equal'] else '!='}  items "
            f"{'==' if r['items_equal'] else '!='}")
    print(f"  system metadata differing: {', '.join(cmp['system_keys_differing']) or 'none'}")
    print(cmp["verdict"])
    doc.update(b=b, compare=cmp)
    rc = 0 if cmp["verdict"] == "SAME_SECTIONS" else 1
  if args.json:
    with open(args.json, "w") as f:
      json.dump(doc, f, indent=1)
  return rc


if __name__ == "__main__":
  sys.exit(main())
