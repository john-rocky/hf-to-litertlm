#!/usr/bin/env python3
"""Rewrite `prompt_templates.model.prefix` of a .litertlm — metadata only, weights untouched.

Why: litert-torch's structured-template extraction (`parse_chat_template`) derives the
assistant prefix from an assistant HISTORY turn and never renders the generation prompt,
so a thinking model exported through the structured path ships with a scaffold its chat
template never produces at generation time (Qwen3-4B-Thinking-2507:
`<|im_start|>assistant\n<think>\n\n</think>\n\n`; granite-4.2-3b: `…<think></think>`;
a template with the opener only in the generation branch: bare `…assistant\n`).
LiteRT-LM sends `model.prefix` as the generation prompt, so the fix is one metadata
field: set it to the vendor's generation prompt (`tokenizer.apply_chat_template(...,
add_generation_prompt=True)` minus the same render without it).

Guard: the new prefix must end with a declared thought-channel start (LiteRT-LM ≥ 0.16
pre-opens the channel when the rendered prompt ends with it); pass --allow-any-prefix
only for A/B experiments.

Verifies after packing that every tflite section is byte-identical and that the metadata
differs ONLY in prompt_templates.model.prefix.

Requires `pip install litert-lm-builder` and the `litert-lm` CLI (or LITERT_LM_BIN),
plus tools/add_thought_channel.py from this repo (section helpers).

    python3 tools/think_prefix/set_model_prefix.py in.litertlm out.litertlm \
        --prefix '<|im_start|>assistant\n<think>\n'
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import add_thought_channel as atc  # noqa: E402

LT = os.environ.get("LITERT_LM_BIN", "litert-lm")


def verify_sections(src, dst):
  """Every non-metadata section content-identical; metadata may differ only in
  prompt_templates.model.prefix."""
  from litert_lm_builder.runtime.proto import llm_metadata_pb2
  fails = []
  a, b = atc.bundle_sections(src), atc.bundle_sections(dst)
  if [t for t, _, _ in a] != [t for t, _, _ in b]:
    return [f"section list changed: {[t for t,_,_ in a]} -> {[t for t,_,_ in b]}"]
  for (ta, ba, ea), (tb, bb, eb) in zip(a, b):
    if ta != "LlmMetadataProto":
      if atc.section_sha(src, ba, ea) != atc.section_sha(dst, bb, eb):
        import zlib
        if ta == "HF_Tokenizer_Zlib":
          if zlib.decompress(atc.read_section(src, ba, ea)) != \
             zlib.decompress(atc.read_section(dst, bb, eb)):
            fails.append("HF_Tokenizer_Zlib content changed")
        else:
          fails.append(f"{ta} section bytes changed")
      continue
    m_src = llm_metadata_pb2.LlmMetadata()
    m_src.ParseFromString(atc.read_section(src, ba, ea))
    m_dst = llm_metadata_pb2.LlmMetadata()
    m_dst.ParseFromString(atc.read_section(dst, bb, eb))
    m_src.prompt_templates.model.prefix = ""
    m_dst.prompt_templates.model.prefix = ""
    if m_src != m_dst:
      fails.append("LlmMetadataProto differs beyond prompt_templates.model.prefix")
  return fails


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("src")
  ap.add_argument("dst")
  ap.add_argument("--prefix", required=True,
                  help=r"new model prefix; \n escapes are interpreted")
  ap.add_argument("--allow-any-prefix", action="store_true",
                  help="skip the channel-start guard (A/B experiments only)")
  args = ap.parse_args()
  new_prefix = args.prefix.encode().decode("unicode_escape")

  from litert_lm_builder.runtime.proto import llm_metadata_pb2
  from google.protobuf import text_format

  work = tempfile.mkdtemp(prefix="prefixfix_")
  try:
    subprocess.run([LT, "unpack", args.src, "--output-dir", work, "--allow-overwrite"],
                   check=True, stdout=subprocess.DEVNULL)
    before = atc.tflite_hashes(work)
    pb = os.path.join(work, "LlmMetadataProto.pbtext")
    md = llm_metadata_pb2.LlmMetadata()
    text_format.Parse(open(pb).read(), md)
    old = md.prompt_templates.model.prefix
    print(f"model.prefix: {old!r} -> {new_prefix!r}")
    starts = [c.start for c in md.channels]
    if starts and not args.allow_any_prefix and not any(new_prefix.endswith(s) for s in starts):
      print(f"REFUSING: new prefix does not end with a declared channel start {starts!r}",
            file=sys.stderr)
      return 2
    md.prompt_templates.model.prefix = new_prefix
    open(pb, "w").write(text_format.MessageToString(md))
    subprocess.run([LT, "pack", work, "--output", args.dst, "--allow-overwrite"],
                   check=True, stdout=subprocess.DEVNULL)

    verify = tempfile.mkdtemp(prefix="prefixverify_")
    try:
      subprocess.run([LT, "unpack", args.dst, "--output-dir", verify, "--allow-overwrite"],
                     check=True, stdout=subprocess.DEVNULL)
      after = atc.tflite_hashes(verify)
    finally:
      shutil.rmtree(verify, ignore_errors=True)
    if before != after:
      print("WEIGHTS CHANGED — refusing", file=sys.stderr)
      return 1
    fails = verify_sections(args.src, args.dst)
    if fails:
      print("SECTION CONTENT CHANGED — refusing:", *fails, sep="\n  ", file=sys.stderr)
      return 1
    print(f"OK  {args.dst}")
    print(f"    {len(before)} tflite section(s) byte-identical; "
          f"{os.path.getsize(args.src)} -> {os.path.getsize(args.dst)} bytes")
    return 0
  finally:
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
  raise SystemExit(main())
