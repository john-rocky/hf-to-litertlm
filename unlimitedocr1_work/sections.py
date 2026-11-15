#!/usr/bin/env python3
"""Print a .litertlm file's size and sha256, then one row per section as the file's own header lists it: index, data
type, model type (TFLite sections), declared prefer_activation_type, bytes and the sha256 of the section's raw bytes.
Read-only (mmap). Needs litert-lm-builder (0.19.1 names the TFLiteWeights sections).

  python sections.py FILE.litertlm [--json OUT.json]
"""
import argparse
import hashlib
import json
import mmap
import os

from litert_lm_builder import litertlm_core
from litert_lm_builder import litertlm_header_schema_py_generated as hschema


def item_kv(item):
  """A section item's (key, value), the value read when it is a string."""
  key = item.Key().decode() if item.Key() else None
  if item.ValueType() == hschema.VData.StringValue:
    sv = hschema.StringValue()
    sv.Init(item.Value().Bytes, item.Value().Pos)
    return key, sv.Value().decode() if sv.Value() else ""
  return key, f"<type {item.ValueType()}>"


def sections(mm):
  at = litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET
  end = int.from_bytes(mm[at:at + 8], "little")
  meta = hschema.LiteRTLMMetaData.GetRootAs(bytes(mm[litertlm_core.HEADER_BEGIN_BYTE_OFFSET:end]), 0)
  smd = meta.SectionMetadata()
  rows = []
  for i in range(smd.ObjectsLength()):
    s = smd.Objects(i)
    items = dict(item_kv(s.Items(j)) for j in range(s.ItemsLength()))
    b, e = s.BeginOffset(), s.EndOffset()
    h = hashlib.sha256()
    for off in range(b, e, 1 << 24):
      h.update(mm[off:min(e, off + (1 << 24))])
    rows.append({"index": i, "data_type": litertlm_core.any_section_data_type_to_string(s.DataType()),
                 "model_type": items.get("model_type"), "prefer_activation_type": items.get("prefer_activation_type"),
                 "begin": b, "end": e, "bytes": e - b, "sha256": h.hexdigest()})
  return rows


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("path")
  ap.add_argument("--json")
  a = ap.parse_args()
  with open(a.path, "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    h = hashlib.sha256()
    for off in range(0, len(mm), 1 << 24):
      h.update(mm[off:off + (1 << 24)])
    rows = sections(mm)
  print(f"{os.path.basename(a.path)}  {os.path.getsize(a.path):,} B  sha256 {h.hexdigest()}")
  for r in rows:
    print(f"  {r['index']}  {r['data_type']:<22} {str(r['model_type'] or ''):<26} "
          f"{str(r['prefer_activation_type'] or ''):<5} {r['bytes']:>15,}  {r['sha256']}")
  if a.json:
    json.dump({"file": a.path, "bytes": os.path.getsize(a.path), "sha256": h.hexdigest(), "sections": rows},
              open(a.json, "w"), indent=1)


if __name__ == "__main__":
  main()
