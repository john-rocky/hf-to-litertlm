#!/usr/bin/env python3
"""sections_items.py A.litertlm [B.litertlm] - read-only: list a bundle's sections (from the LiteRT-LM header) with type,
the section's header items (model_type, prefer_activation_type, ...), size and sha256; with a second bundle, compare the
two section by section in order, header items included, so a file whose GPU activation type is declared does not read as
identical to one that declares none. Exit 0 when every section's bytes and items are equal (or with one bundle), 1
otherwise. Needs litert-lm-builder."""
import hashlib
import mmap
import struct
import sys

from litert_lm_builder import litertlm_header_schema_py_generated as H

TYPE = {v: k for k, v in vars(H.AnySectionDataType).items() if isinstance(v, int)}


def sections(path):
  with open(path, "rb") as f:
    header_end = struct.unpack("<Q", f.read(32)[24:32])[0]
    f.seek(0)
    header = f.read(header_end)
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
  md = H.LiteRTLMMetaData.GetRootAs(header, 32)
  objs = md.SectionMetadata()
  out = []
  for i in range(objs.ObjectsLength()):
    o = objs.Objects(i)
    b, e, t = o.BeginOffset(), o.EndOffset(), o.DataType()
    items = {}
    for j in range(o.ItemsLength()):
      it = o.Items(j)
      key = it.Key().decode() if it.Key() else ""
      if it.ValueType() == H.VData.StringValue:
        v = H.StringValue()
        v.Init(it.Value().Bytes, it.Value().Pos)
        items[key] = v.Value().decode()
      else:
        items[key] = f"<value type {it.ValueType()}>"
    out.append((i, TYPE.get(t, str(t)), items, e - b, hashlib.sha256(mm[b:e]).hexdigest()))
  return out


def fmt_items(items):
  return ", ".join(f"{k}={v}" for k, v in sorted(items.items())) or "-"


def main(argv):
  tables = [(p, sections(p)) for p in argv]
  for name, s in tables:
    print("##", name)
    for i, t, items, n, h in s:
      print(f"  {i} {t:<22} {fmt_items(items):<58} {n:>14,} B  {h}")
  if len(tables) < 2:
    return 0
  (_, sa), (_, sb) = tables[:2]
  same = len(sa) == len(sb)
  print("## compare (in order: type, header items, bytes)")
  for x, y in zip(sa, sb):
    eq = x[1:] == y[1:]
    same &= eq
    why = []
    if x[2] != y[2]:
      why.append(f"items {fmt_items(x[2])} vs {fmt_items(y[2])}")
    if x[3:] != y[3:]:
      why.append(f"data {x[3]:,} B {x[4][:16]} vs {y[3]:,} B {y[4][:16]}")
    print(f"  {x[0]} {x[1]:<22} {'IDENTICAL' if eq else 'DIFFERENT: ' + '; '.join(why)}")
  print("ALL SECTIONS IDENTICAL (bytes and header items)" if same else "SECTIONS DIFFER")
  return 0 if same else 1


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
