#!/usr/bin/env python3
"""Replace the Jinja prompt template carried in a .litertlm's LlmMetadata (metadata-only repack).

    python set_prompt_template.py in.litertlm out.litertlm template.jinja [--litert-lm PATH]

Weights and tokenizer sections are re-packed byte-identical; only `jinja_prompt_template` changes.
Used to give the Spark-X2.5 bundles the vendor's tool-calling template (tools + tool role) for the
phone-agent demo; the shipped bundle carries the plain-chat subset.
Also the fix for a template whose history render depends on the turn's position (the second message
fails on litert-lm 0.18.0): tools/chat_template_prefix/README.md, templates/nemotron3nano_history.jinja.
"""
import argparse, os, subprocess, sys, tempfile
from google.protobuf import text_format
from litert_lm_builder.runtime.proto import llm_metadata_pb2


def run_cli(cmd, ok):
  """Run one litert-lm step. The CLI prints its errors and still exits 0, so the step passed only if it printed `ok`."""
  r = subprocess.run(cmd, capture_output=True, text=True)
  print(r.stdout, end="", flush=True)
  print(r.stderr, end="", file=sys.stderr, flush=True)
  if r.returncode != 0 or ok not in r.stdout + r.stderr:
    raise SystemExit(f"litert-lm {cmd[1]} failed (exit {r.returncode}); output above")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("template")
  ap.add_argument("--litert-lm", default="litert-lm")
  a = ap.parse_args()
  with tempfile.TemporaryDirectory() as td:
    unpack = os.path.join(td, "unpack")
    run_cli([a.litert_lm, "unpack", a.src, "--output-dir", unpack], "Successfully unpacked")
    pb = os.path.join(unpack, "LlmMetadataProto.pbtext")
    meta = llm_metadata_pb2.LlmMetadata()
    text_format.Parse(open(pb).read(), meta)
    old = meta.jinja_prompt_template
    meta.jinja_prompt_template = open(a.template).read()
    open(pb, "w").write(text_format.MessageToString(meta, as_utf8=True))
    print(f"jinja_prompt_template: {len(old)} -> {len(meta.jinja_prompt_template)} chars; channels={len(meta.channels)}")
    if os.path.exists(a.dst):
      os.remove(a.dst)
    run_cli([a.litert_lm, "pack", os.path.join(unpack, "model.toml"), "--output", a.dst], "Successfully packed")
  print("wrote", a.dst, os.path.getsize(a.dst))


if __name__ == "__main__":
  main()
