#!/usr/bin/env python3
"""extract_section.py BUNDLE.litertlm OUT.tflite [--model-type prefill_decode]

Copy one TFLiteModel section of a .litertlm to a .tflite file by the offsets in the bundle header and print its size
and sha256. The header is read with litert_lm_builder's schema, the same reader as bundle_sections() in
tools/gpu_graph/gpu_graph_retrofit.py; nothing is unpacked or re-packed, the section bytes are copied as they are (the
bytes `litert-lm unpack` writes for that section). Used to hand the decoder graph of a 3.8 GB bundle to
cache_variant_comp.py without unpacking the bundle.
"""
import argparse
import hashlib
import os
import sys

from litert_lm_builder import litertlm_core
from litert_lm_builder import litertlm_header_schema_py_generated as hs


def sections(path):
  with open(path, "rb") as f:
    head = f.read(1 << 16)
  he = int.from_bytes(head[litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET:litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET + 8],
                      "little")
  meta = hs.LiteRTLMMetaData.GetRootAs(bytearray(head[litertlm_core.HEADER_BEGIN_BYTE_OFFSET:he]), 0)
  out = []
  for i in range(meta.SectionMetadata().ObjectsLength()):
    so = meta.SectionMetadata().Objects(i)
    items = {}
    for j in range(so.ItemsLength()):
      it = so.Items(j)
      k = it.Key().decode() if it.Key() else None
      if it.ValueType() == hs.VData.StringValue:
        sv = hs.StringValue()
        sv.Init(it.Value().Bytes, it.Value().Pos)
        items[k] = sv.Value().decode() if sv.Value() else ""
      else:
        items[k] = f"<vtype {it.ValueType()}>"
    out.append({"type": litertlm_core.any_section_data_type_to_string(so.DataType()), "begin": so.BeginOffset(),
                "end": so.EndOffset(), "items": items})
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("bundle")
  ap.add_argument("out")
  ap.add_argument("--model-type", default="prefill_decode")
  a = ap.parse_args()
  want = a.model_type.lower()
  hits = [s for s in sections(a.bundle) if s["type"] == "TFLiteModel"
          and s["items"].get("model_type", "").lower().replace("tf_lite_", "") == want]
  if len(hits) != 1:
    sys.exit(f"want exactly one TFLiteModel section of model_type {want}, found {len(hits)}")
  s = hits[0]
  h = hashlib.sha256()
  n = s["end"] - s["begin"]
  os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
  with open(a.bundle, "rb") as f, open(a.out + ".part", "wb") as g:
    f.seek(s["begin"])
    left = n
    while left:
      b = f.read(min(left, 1 << 24))
      if not b:
        sys.exit("bundle ended inside the section")
      h.update(b)
      g.write(b)
      left -= len(b)
  os.replace(a.out + ".part", a.out)
  print(f"EXTRACTED {want} items={s['items']} bytes={n} sha256={h.hexdigest()} -> {a.out}")


if __name__ == "__main__":
  main()
