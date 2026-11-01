#!/usr/bin/env python3
"""Offline checks of capabilities.vision / .audio in make_manifest.py: the
model type's image / audio tokens say what the family configuration can
carry, the bundle's sections say what this file carries. A text-only LFM2
file packed with LiteRT-LM's models/lfm2 configuration (image tokens declared
since 2026-09-28) has no vision encoder section and must not be reported as
vision-capable. An LFM2 VL file packed with an empty lfm2 {} type has vision
sections and must be reported as vision-capable: LiteRT-LM's lfm2 processor
then falls back to its default <|image_start|> / <|image_end|>. Each case is
an LlmMetadata built in memory with a section list (the lfm2 cases also
through a .litertlm header built in memory). No network.

    python manifest/test_capabilities.py
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

def tfl(model_type):
  return {"type": "TFLiteModel", "model_type": model_type}

# Section lists for the checks: a text-only file, and what a vision or an audio
# file adds (the LFM3-Omni-2B file has all five of these TFLiteModel sections).
TEXT = [{"type": "LlmMetadataProto"}, {"type": "HF_Tokenizer_Zlib"},
        tfl("tf_lite_prefill_decode"), tfl("tf_lite_embedder")]
VISION = [tfl("tf_lite_vision_encoder"), tfl("tf_lite_vision_adapter")]
AUDIO = [tfl("tf_lite_audio_encoder_hw")]

def lfm2_meta():
  """The model type of LiteRT-LM's models/lfm2 configuration: image tokens."""
  m = llm_metadata_pb2.LlmMetadata()
  m.llm_model_type.lfm2.start_of_image_token.token_str = "<|image_start|>"
  m.llm_model_type.lfm2.end_of_image_token.token_str = "<|image_end|>"
  return m

def empty_lfm2_meta():
  """An empty lfm2 {} type, as the LFM2.5-VL files are packed."""
  m = llm_metadata_pb2.LlmMetadata()
  m.llm_model_type.lfm2.SetInParent()
  return m

def gemma3n_meta():
  """A model type with both image and audio tokens."""
  m = llm_metadata_pb2.LlmMetadata()
  m.llm_model_type.gemma3n.start_of_image_token.token_str = "<start_of_image>"
  m.llm_model_type.gemma3n.start_of_audio_token.token_str = "<start_of_audio>"
  return m

def generic_meta(image=False, audio=False):
  m = llm_metadata_pb2.LlmMetadata()
  m.llm_model_type.generic_model.SetInParent()
  m.llm_model_type.generic_model.image_enabled = image
  m.llm_model_type.generic_model.audio_enabled = audio
  return m

def caps(m, sections):
  c = mm.derive_capabilities(m, sections)
  return c["vision"], c["audio"]

def kv(key, value):
  sv = schema.StringValueT()
  sv.value = value
  pair = schema.KeyValuePairT()
  pair.key, pair.valueType, pair.value = key, schema.VData.StringValue, sv
  return pair

def fake_bundle(meta, model_types):
  """fetch(s, e) over a header with an LlmMetadata section (its bytes are in
  the file) and one TFLiteModel section per model type, declared only."""
  raw = meta.SerializeToString()
  objs, pos = [], 1 << 16
  def section(dtype, size, model_type=None):
    nonlocal pos
    o = schema.SectionObjectT()
    o.dataType, o.beginOffset, o.endOffset = dtype, pos, pos + size
    o.items = [kv("model_type", model_type)] if model_type else []
    objs.append(o)
    pos = (pos + size + 16383) // 16384 * 16384
  section(schema.AnySectionDataType.LlmMetadataProto, len(raw))
  for t in model_types:
    section(schema.AnySectionDataType.TFLiteModel, 1024, t)
  hdr = schema.LiteRTLMMetaDataT()
  hdr.systemMetadata = schema.SystemMetadataT()
  hdr.systemMetadata.entries = []
  hdr.sectionMetadata = schema.SectionMetadataT()
  hdr.sectionMetadata.objects = objs
  b = flatbuffers.Builder(1024)
  b.Finish(hdr.Pack(b))
  flat = bytes(b.Output())
  begin = litertlm_core.HEADER_BEGIN_BYTE_OFFSET
  end = begin + len(flat)
  data = bytearray(objs[0].beginOffset + len(raw))
  data[:8] = litertlm_core.HEADER_MAGIC_BYTES
  loc = litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET
  data[loc:loc + 8] = end.to_bytes(8, "little")
  data[begin:end] = flat
  data[objs[0].beginOffset:] = raw
  def fetch(s, e):
    return bytes(data[s:e + 1])
  return fetch

class Capabilities(unittest.TestCase):
  def test_lfm2_text_only_file_is_not_vision(self):
    self.assertEqual(caps(lfm2_meta(), TEXT), (False, False))

  def test_lfm2_file_with_vision_sections(self):
    self.assertEqual(caps(lfm2_meta(), TEXT + VISION), (True, False))

  def test_empty_lfm2_type_follows_the_sections(self):
    self.assertEqual(caps(empty_lfm2_meta(), TEXT + VISION), (True, False))
    self.assertEqual(caps(empty_lfm2_meta(), TEXT), (False, False))
    self.assertEqual(caps(empty_lfm2_meta(), TEXT + AUDIO), (False, False))

  def test_audio_token_needs_an_audio_section(self):
    self.assertEqual(caps(gemma3n_meta(), TEXT), (False, False))
    self.assertEqual(caps(gemma3n_meta(), TEXT + AUDIO), (False, True))
    self.assertEqual(caps(gemma3n_meta(), TEXT + VISION + AUDIO), (True, True))

  def test_generic_model_flags_are_gated_too(self):
    m = generic_meta(image=True, audio=True)
    self.assertEqual(caps(m, TEXT), (False, False))
    self.assertEqual(caps(m, TEXT + AUDIO + VISION), (True, True))

  def test_sections_never_add_a_capability(self):
    self.assertEqual(caps(generic_meta(), TEXT + VISION + AUDIO), (False, False))
    m = llm_metadata_pb2.LlmMetadata()
    m.llm_model_type.qwen3.SetInParent()
    self.assertEqual(caps(m, TEXT + VISION + AUDIO), (False, False))

  def test_no_sections_is_the_model_type_alone(self):
    self.assertEqual(caps(lfm2_meta(), None), (True, False))
    self.assertEqual(caps(empty_lfm2_meta(), None), (True, False))
    self.assertEqual(caps(gemma3n_meta(), None), (True, True))

  def test_sections_read_from_a_header(self):
    for meta_fn in (lfm2_meta, empty_lfm2_meta):
      for types, want in (([], (False, False)),
                          (["tf_lite_prefill_decode", "tf_lite_embedder"], (False, False)),
                          (["tf_lite_prefill_decode", "tf_lite_vision_encoder",
                            "tf_lite_vision_adapter"], (True, False))):
        sections, meta = mm.parse_bundle_header(fake_bundle(meta_fn(), types))
        self.assertEqual(caps(meta, sections), want, (meta_fn.__name__, types))

if __name__ == "__main__":
  unittest.main(verbosity=2)
