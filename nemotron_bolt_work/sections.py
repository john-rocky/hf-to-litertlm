#!/usr/bin/env python3
"""sections.py A.litertlm [B.litertlm] - read-only: list a bundle's sections (from the LiteRT-LM header) with type,
model type, size and sha256; with a second bundle, compare the two section by section in order. Exit 0 when every
section is byte-identical (or with one bundle), 1 otherwise. Needs litert-lm-builder."""
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
    model_type = ""
    for j in range(o.ItemsLength()):
      it = o.Items(j)
      if it.Key() and it.Key().decode() == "model_type":
        v = H.StringValue()
        v.Init(it.Value().Bytes, it.Value().Pos)
        model_type = v.Value().decode()
    out.append((i, TYPE.get(t, str(t)), model_type, e - b, hashlib.sha256(mm[b:e]).hexdigest()))
  return out


def main(argv):
  tables = [(p, sections(p)) for p in argv]
  for name, s in tables:
    print("##", name)
    for i, t, m, n, h in s:
      print(f"  {i} {t:<24} {m:<28} {n:>14,} B  {h}")
  if len(tables) < 2:
    return 0
  (_, sa), (_, sb) = tables[:2]
  same = len(sa) == len(sb)
  print("## compare (in order)")
  for x, y in zip(sa, sb):
    eq = x[1:] == y[1:]
    same &= eq
    print(f"  {x[0]} {x[1]:<24} {x[2]:<28} {'IDENTICAL' if eq else 'DIFFERENT'}" + ("" if eq else f"  {x[3]:,} vs {y[3]:,} B"))
  print("ALL SECTIONS IDENTICAL" if same else "SECTIONS DIFFER")
  return 0 if same else 1


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
