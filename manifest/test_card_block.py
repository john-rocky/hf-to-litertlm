#!/usr/bin/env python3
"""Offline checks of the 0.1.5 additions to make_manifest.py: the sampler hints
looked up in the source model (generation_config.json, then the card's text,
then a code example in the card; nothing on a conflict) and the model card's
`litert_lm:` block (a projection of the manifest, written as YAML that reads
back as written). The Hub is replaced by a dict of answers. No network.

    python manifest/test_card_block.py
"""
import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import make_manifest as mm  # noqa: E402

try:
  import yaml
except ImportError:  # the round-trip checks need it
  yaml = None

BASE = "example/source-model"
SHA = "0" * 40

def hub(answers):
  """A stand-in for make_manifest.hub_get: {file name: body or HTTP status}."""
  def get(url):
    if url.endswith(f"/api/models/{BASE}"):
      return 200, json.dumps({"sha": SHA}).encode()
    name = url.rsplit("/", 1)[-1]
    answer = answers.get(name, 404)
    if isinstance(answer, int):
      return answer, None
    return 200, (answer if isinstance(answer, bytes) else answer.encode())
  return get

CARD = """---
license: apache-2.0
temperature: 0.3
---
# A model
"""

class SamplerLookup(unittest.TestCase):

  def run_lookup(self, answers):
    saved = mm.hub_get
    mm.hub_get = hub(answers)
    try:
      return mm.lookup_sampler(BASE)
    finally:
      mm.hub_get = saved

  def test_generation_config_first(self):
    values, source, _ = self.run_lookup({
        "generation_config.json": json.dumps(
            {"do_sample": True, "temperature": 0.7, "top_k": 20, "top_p": 0.8,
             "repetition_penalty": 1.1}),
        "README.md": CARD + "Use `temperature: 0.1`.\n"})
    self.assertEqual(source, "generation_config")
    self.assertEqual(values, {"top_k": 20, "top_p": 0.8, "temperature": 0.7})

  def test_readme_text(self):
    # LiquidAI/LFM2.5-1.2B-Instruct's card: values in the text, no top_p
    values, source, _ = self.run_lookup({
        "generation_config.json": json.dumps({"bos_token_id": 1, "eos_token_id": 7}),
        "README.md": CARD + "**Generation parameters**: `temperature: 0.1` / "
                     "`top_k: 50` / `repetition_penalty: 1.05`\n"})
    self.assertEqual(source, "readme")
    self.assertEqual(values, {"top_k": 50, "temperature": 0.1})

  def test_code_example(self):
    # taobao-mnn/MNN-Pocket-0.3B's card: values only in a code example
    values, source, _ = self.run_lookup({
        "README.md": CARD + "Run it:\n\n```python\noutputs = model.generate(**inputs, "
                     "max_new_tokens=512, temperature=0.8, top_p=0.8, top_k=20)\n```\n"})
    self.assertEqual(source, "code_example")
    self.assertEqual(values, {"top_k": 20, "top_p": 0.8, "temperature": 0.8})

  def test_text_before_code(self):
    values, source, _ = self.run_lookup({
        "README.md": CARD + "We suggest a temperature of 0.6.\n\n```python\n"
                     "generate(temperature=0.9, top_k=40)\n```\n"})
    self.assertEqual(source, "readme")
    self.assertEqual(values, {"temperature": 0.6})

  def test_two_settings_write_nothing(self):
    # a card with one setting for thinking and another without
    values, source, notes = self.run_lookup({
        "README.md": CARD + "- Thinking: Temperature=0.6, TopP=0.95, TopK=20\n"
                     "- Non-thinking: Temperature=0.7, TopP=0.8, TopK=20\n"})
    self.assertEqual((values, source), ({}, None))
    self.assertIn("more than one value", notes[-1])

  def test_same_value_twice_is_one_value(self):
    values, source, _ = self.run_lookup({
        "README.md": CARD + "temperature: 0.1, top_k: 50\n\nAgain: temperature=0.1\n"})
    self.assertEqual((values, source), ({"top_k": 50, "temperature": 0.1}, "readme"))

  def test_not_sampler_numbers(self):
    values, source, _ = self.run_lookup({
        "README.md": CARD + "```json\n{\"name\": \"get_weather\", \"result\": "
                     "{\"temperature\": 22, \"unit\": \"celsius\"}}\n```\n"
                     "The temperature 25°C. A temperature range 0.6-0.8. Top-p 90%.\n"})
    self.assertEqual((values, source), ({}, None))

  def test_front_matter_is_not_text(self):
    values, source, _ = self.run_lookup({"README.md": CARD + "No values here.\n"})
    self.assertEqual((values, source), ({}, None))

  def test_gated_source_model(self):
    values, source, notes = self.run_lookup({"generation_config.json": 401,
                                             "README.md": 401})
    self.assertEqual((values, source), ({}, None))
    self.assertTrue(any("HTTP 401" in n for n in notes))

  def test_config_values_out_of_range(self):
    self.assertEqual(mm.sampler_from_config({"top_k": 0, "top_p": 1.5, "temperature": 0.0}),
                     {"temperature": 0.0})
    self.assertEqual(mm.sampler_from_config({"top_k": 40.0, "top_p": True}), {"top_k": 40})


def manifest():
  return {
      "manifest_schema": "0.1.5", "repo": "litert-community/Example",
      "generated": "2026-11-08", "generator": "make_manifest.py",
      "model": {
          "display_name": "Example", "base_model": BASE, "context_length": 4096,
          "stateful": False,
          "session_defaults": {"notes": "long text", "top_k": 20, "top_p": 0.8,
                               "temperature": 0.8, "source": "code_example"},
          "capabilities": {"vision": False, "audio": False,
                           "thinking": {"declared": False, "control": "never"},
                           "channels": []},
      },
      "variants": [
          {"file": "Example_int8.litertlm", "sha256": "a" * 64, "size_bytes": 10,
           "quantization": "int8", "backends": ["cpu", "gpu"], "default_backend": "cpu",
           "recommended": [
               {"platform": "android", "device_class": "flagship (Pixel 11 Pro, PowerVR)",
                "backend": "cpu", "reason": "long text"},
               {"platform": "android", "device_class": "flagship (Galaxy S26, Adreno)",
                "soc": "SM8850", "backend": "gpu", "reason": "long text"}],
           "min_runtime_version": "0.19.1", "context_length": 4096,
           "measured": [{"device": "Galaxy S26", "evidence": "private"}],
           "known_issues": ["text"],
           "checked": [{"device": "Galaxy S26", "os": "android", "backend": "gpu",
                        "runtime": "0.19.1", "top_k": 20, "top_p": 0.8,
                        "temperature": 0.8, "runs": 3, "date": "2026-11-08",
                        "evidence": "private"}],
           "sections": [{"type": "LlmMetadataProto"}]},
          {"file": "Example_npu.litertlm", "quantization": "int8",
           "backends": ["npu"], "default_backend": "npu", "context_length": 2048},
      ],
  }


class CardBlock(unittest.TestCase):

  def test_projection(self):
    block = mm.card_block(manifest())["litert_lm"]
    self.assertEqual(block["manifest_schema"], "0.1.5")
    self.assertEqual(block["session_defaults"],
                     {"top_k": 20, "top_p": 0.8, "temperature": 0.8,
                      "source": "code_example"})
    self.assertEqual(block["capabilities"],
                     {"vision": False, "audio": False,
                      "thinking": {"declared": False, "control": "never"}})
    first, second = block["variants"]
    self.assertEqual(list(first), ["file", "backends", "default_backend", "recommended",
                                   "context_length", "min_runtime_version", "checked"])
    self.assertEqual(first["recommended"][1],
                     {"platform": "android", "device_class": "flagship (Galaxy S26, Adreno)",
                      "soc": "SM8850", "backend": "gpu"})
    self.assertNotIn("evidence", first["checked"][0])
    self.assertEqual(second["checked"], [])
    self.assertNotIn("recommended", second)
    text = json.dumps(block)
    for kept_out in ("sha256", "size_bytes", "reason", "notes", "measured",
                     "known_issues", "sections", "private"):
      self.assertNotIn(kept_out, text)

  @unittest.skipIf(yaml is None, "pyyaml not installed")
  def test_yaml_reads_back(self):
    block = mm.card_block(manifest())
    text = mm.card_block_yaml(block)
    self.assertEqual(yaml.safe_load(text), block)
    self.assertIn('min_runtime_version: "0.19.1"', text)
    self.assertIn('date: "2026-11-08"', text)
    self.assertIn("backends: [cpu, gpu]", text)
    self.assertIn('device_class: "flagship (Galaxy S26, Adreno)"', text)
    self.assertIn("    - file: Example_int8.litertlm", text)

  @unittest.skipIf(yaml is None, "pyyaml not installed")
  def test_yaml_quotes_what_yaml_would_change(self):
    obj = {"a": {"k1": "no", "k2": "On", "ver": "0.20", "neg": -1, "x": 1.0,
                 "word": "SM8850", "colon": "a: b", "hash": "a #b", "empty": "",
                 "list": ["yes", "0.1", "gpu"]}}
    text = "\n".join(mm.yaml_lines(obj)) + "\n"
    self.assertEqual(yaml.safe_load(text), obj)

  def test_no_capabilities_no_key(self):
    m = manifest()
    m["model"].pop("capabilities")
    m["model"].pop("session_defaults")
    block = mm.card_block(m)["litert_lm"]
    self.assertNotIn("capabilities", block)
    self.assertNotIn("session_defaults", block)


if __name__ == "__main__":
  unittest.main()
