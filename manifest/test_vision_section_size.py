#!/usr/bin/env python3
"""Offline checks of the 2 GiB vision encoder warning in make_manifest.py:
section lists, and a .litertlm header built in memory whose vision encoder
section is declared over 2 GiB (the section bytes are not there; the
generator reads the header and the LlmMetadata section only). No network.

    python manifest/test_vision_section_size.py
"""
import pathlib
import sys
import unittest

import flatbuffers

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import make_manifest as mm  # noqa: E402
from litert_lm_builder import litertlm_core  # noqa: E402
from litert_lm_builder import litertlm_header_schema_py_generated as schema  # noqa: E402
from litert_lm_builder.runtime.proto import llm_metadata_pb2  # noqa: E402

GIB2 = 2 * 1024 ** 3

def kv(key, value):
  sv = schema.StringValueT()
  sv.value = value
  pair = schema.KeyValuePairT()
  pair.key, pair.valueType, pair.value = key, schema.VData.StringValue, sv
  return pair

def fake_bundle(vision_sizes, adapter_size=1024):
  """(bytes, fetch): a header with an LlmMetadata section (its bytes are in
  the file), then one tf_lite_vision_encoder section per size and a
  tf_lite_vision_adapter section, declared only (offsets past the end)."""
  pb = llm_metadata_pb2.LlmMetadata()
  pb.max_num_tokens = 2048
  raw = pb.SerializeToString()
  objs, pos = [], 1 << 16
  def section(dtype, size, model_type=None):
    nonlocal pos
    o = schema.SectionObjectT()
    o.dataType, o.beginOffset, o.endOffset = dtype, pos, pos + size
    o.items = [kv("model_type", model_type)] if model_type else []
    objs.append(o)
    pos = (pos + size + 16383) // 16384 * 16384
  section(schema.AnySectionDataType.LlmMetadataProto, len(raw))
  for size in vision_sizes:
    section(schema.AnySectionDataType.TFLiteModel, size, "tf_lite_vision_encoder")
  section(schema.AnySectionDataType.TFLiteModel, adapter_size, "tf_lite_vision_adapter")
  meta = schema.LiteRTLMMetaDataT()
  meta.systemMetadata = schema.SystemMetadataT()
  meta.systemMetadata.entries = []
  meta.sectionMetadata = schema.SectionMetadataT()
  meta.sectionMetadata.objects = objs
  b = flatbuffers.Builder(1024)
  b.Finish(meta.Pack(b))
  flat = bytes(b.Output())
  begin = litertlm_core.HEADER_BEGIN_BYTE_OFFSET
  end = begin + len(flat)
  data = bytearray(objs[0].beginOffset + len(raw))
  data[:8] = litertlm_core.HEADER_MAGIC_BYTES
  loc = litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET
  data[loc:loc + 8] = end.to_bytes(8, "little")
  data[begin:end] = flat
  data[objs[0].beginOffset:] = raw
  read = []
  def fetch(s, e):
    if e >= len(data):
      raise AssertionError(f"read past the header and LlmMetadata: {s}-{e}")
    read.append((s, e))
    return bytes(data[s:e + 1])
  return fetch, read

class SectionLists(unittest.TestCase):
  def enc(self, size):
    return {"type": "TFLiteModel", "size_bytes": size,
            "model_type": "tf_lite_vision_encoder"}

  def test_exactly_2gib_passes(self):
    self.assertEqual(mm.vision_section_warnings("f", [self.enc(GIB2)]), [])

  def test_one_byte_over_warns(self):
    w = mm.vision_section_warnings("f.litertlm", [self.enc(GIB2 + 1)])
    self.assertEqual(len(w), 1)
    self.assertTrue(w[0].startswith("WARN f.litertlm: "))
    self.assertIn("2,147,483,649 B", w[0])
    self.assertIn("litert_lm_loaded_file_create", w[0])

  def test_other_sections_never_warn(self):
    big = GIB2 * 3
    secs = [{"type": "TFLiteModel", "size_bytes": big, "model_type": t}
            for t in ("tf_lite_vision_adapter", "tf_lite_prefill_decode",
                      "tf_lite_embedder")]
    secs.append({"type": "TFLiteWeights", "size_bytes": big})
    self.assertEqual(mm.vision_section_warnings("f", secs), [])

  def test_one_line_per_section(self):
    secs = [self.enc(GIB2 + 1), self.enc(100), self.enc(7_419_391_744)]
    w = mm.vision_section_warnings("f", secs)
    self.assertEqual(len(w), 2)
    self.assertIn("7,419,391,744 B", w[1])

class FakeHeader(unittest.TestCase):
  def test_declared_size_reaches_the_warning(self):
    fetch, read = fake_bundle([2_553_783_680])
    sections, llm_meta = mm.parse_bundle_header(fetch)
    self.assertEqual(llm_meta.max_num_tokens, 2048)
    enc = [s for s in sections if s.get("model_type") == "tf_lite_vision_encoder"]
    self.assertEqual([s["size_bytes"] for s in enc], [2_553_783_680])
    w = mm.vision_section_warnings("q.litertlm", sections)
    self.assertEqual(len(w), 1)
    self.assertIn("2,553,783,680 B", w[0])
    self.assertTrue(all(e < (1 << 17) for _, e in read))

  def test_boundary_through_the_header(self):
    for size, n in ((GIB2, 0), (GIB2 + 1, 1)):
      fetch, _ = fake_bundle([size], adapter_size=GIB2 + 1)
      sections, _ = mm.parse_bundle_header(fetch)
      self.assertEqual(len(mm.vision_section_warnings("f", sections)), n, size)

if __name__ == "__main__":
  unittest.main(verbosity=2)
