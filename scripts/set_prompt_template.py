#!/usr/bin/env python3
"""Replace the Jinja prompt template carried in a .litertlm's LlmMetadata (metadata-only repack).

    python set_prompt_template.py in.litertlm out.litertlm template.jinja [--litert-lm PATH]

Weights and tokenizer sections are re-packed byte-identical; only `jinja_prompt_template` changes.
Used to give the Spark-X2.5 bundles the vendor's tool-calling template (tools + tool role) for the
phone-agent demo; the shipped bundle carries the plain-chat subset.
"""
import argparse, os, subprocess, tempfile
from google.protobuf import text_format
from litert_lm_builder.runtime.proto import llm_metadata_pb2


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("template")
  ap.add_argument("--litert-lm", default="litert-lm")
  a = ap.parse_args()
  with tempfile.TemporaryDirectory() as td:
    unpack = os.path.join(td, "unpack")
    subprocess.run([a.litert_lm, "unpack", a.src, "--output-dir", unpack], check=True)
    pb = os.path.join(unpack, "LlmMetadataProto.pbtext")
    meta = llm_metadata_pb2.LlmMetadata()
    text_format.Parse(open(pb).read(), meta)
    old = meta.jinja_prompt_template
    meta.jinja_prompt_template = open(a.template).read()
    open(pb, "w").write(text_format.MessageToString(meta, as_utf8=True))
    print(f"jinja_prompt_template: {len(old)} -> {len(meta.jinja_prompt_template)} chars; channels={len(meta.channels)}")
    if os.path.exists(a.dst):
      os.remove(a.dst)
    subprocess.run([a.litert_lm, "pack", os.path.join(unpack, "model.toml"), "--output", a.dst], check=True)
  print("wrote", a.dst, os.path.getsize(a.dst))


if __name__ == "__main__":
  main()
