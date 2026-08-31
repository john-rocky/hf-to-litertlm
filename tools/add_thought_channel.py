#!/usr/bin/env python3
"""Add a thought channel to an existing .litertlm — metadata only, weights untouched.

A reasoning bundle needs `LlmMetadata.channels` to declare its thinking markers
(e.g. `<think>`/`</think>`). Without the block the runtime streams the raw
reasoning inline into the answer and silently ignores any thinking budget
(`--thinking-budget` / ThinkingConfig): the budget machinery keys off the
channel's end-token ids, and with no channel there are none. The exporter does
not emit the block on the structured-template path, so add it after export.

Verifies after packing that every tflite section is byte-identical to the input
and that the metadata differs ONLY in `channels`, so the fix can never be a
quality change.

Requires `pip install litert-lm-builder` and the `litert-lm` CLI (or point
LITERT_LM_BIN at one).

    python3 tools/add_thought_channel.py in.litertlm out.litertlm \
        --start '<think>' --end '</think>'
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

LT = os.environ.get("LITERT_LM_BIN", "litert-lm")


def sha(p):
  h = hashlib.sha256()
  with open(p, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def tflite_hashes(d):
  return {f: sha(os.path.join(d, f)) for f in sorted(os.listdir(d)) if f.endswith(".tflite")}


def bundle_sections(path):
  """Parse the .litertlm header; return [(dtype, begin, end)] in file order."""
  from litert_lm_builder import litertlm_core
  from litert_lm_builder import litertlm_header_schema_py_generated as schema
  with open(path, "rb") as f:
    head = f.read(4096)
    if head[:8] != litertlm_core.HEADER_MAGIC_BYTES:
      raise ValueError(f"{path}: bad magic")
    hdr_end = int.from_bytes(
        head[litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET:
             litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET + 8], "little")
    if hdr_end > len(head):
      f.seek(0)
      head = f.read(hdr_end + 64)
  buf = bytearray(head[litertlm_core.HEADER_BEGIN_BYTE_OFFSET:hdr_end])
  meta = schema.LiteRTLMMetaData.GetRootAs(buf, 0)
  out = []
  for i in range(meta.SectionMetadata().ObjectsLength()):
    so = meta.SectionMetadata().Objects(i)
    out.append((litertlm_core.any_section_data_type_to_string(so.DataType()),
                so.BeginOffset(), so.EndOffset()))
  return out


def section_sha(path, begin, end):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    f.seek(begin)
    left = end - begin
    while left > 0:
      chunk = f.read(min(1 << 22, left))
      if not chunk:
        raise IOError(f"{path}: short read in section [{begin},{end})")
      h.update(chunk)
      left -= len(chunk)
  return h.hexdigest()


def read_section(path, begin, end):
  with open(path, "rb") as f:
    f.seek(begin)
    return f.read(end - begin)


def verify_sections(src, dst):
  """Every section except LlmMetadataProto must survive byte-identical (zlib and
  pbtext-backed sections may re-serialize: compare their CONTENT then). The
  metadata section itself must differ from the source only in `channels`.
  Returns a list of human-readable failures (empty = pass)."""
  import zlib
  from litert_lm_builder.runtime.proto import llm_metadata_pb2
  from litert_lm_builder.runtime.proto import executor_metadata_pb2

  fails = []
  a, b = bundle_sections(src), bundle_sections(dst)
  if [t for t, _, _ in a] != [t for t, _, _ in b]:
    return [f"section list changed: {[t for t, _, _ in a]} -> {[t for t, _, _ in b]}"]
  for (ta, ba, ea), (tb, bb, eb) in zip(a, b):
    if ta != "LlmMetadataProto" and section_sha(src, ba, ea) == section_sha(dst, bb, eb):
      continue
    if ta == "LlmMetadataProto":
      m_src = llm_metadata_pb2.LlmMetadata()
      m_src.ParseFromString(read_section(src, ba, ea))
      m_dst = llm_metadata_pb2.LlmMetadata()
      m_dst.ParseFromString(read_section(dst, bb, eb))
      m_src.ClearField("channels")
      m_dst.ClearField("channels")
      if m_src != m_dst:
        fails.append("LlmMetadataProto differs beyond `channels`")
    elif ta == "HF_Tokenizer_Zlib":
      if zlib.decompress(read_section(src, ba, ea)) != zlib.decompress(read_section(dst, bb, eb)):
        fails.append("HF_Tokenizer_Zlib content changed (inflated bytes differ)")
    elif ta == "ExecutorMetadataProto":
      e_src = executor_metadata_pb2.ExecutorMetadata()
      e_src.ParseFromString(read_section(src, ba, ea))
      e_dst = executor_metadata_pb2.ExecutorMetadata()
      e_dst.ParseFromString(read_section(dst, bb, eb))
      if e_src != e_dst:
        fails.append("ExecutorMetadataProto content changed")
    else:
      fails.append(f"{ta} section bytes changed")
  return fails


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("src")
  ap.add_argument("dst")
  ap.add_argument("--name", default="thought")
  ap.add_argument("--start", default="<think>\n")
  ap.add_argument("--end", default="\n</think>")
  ap.add_argument("--force", action="store_true", help="REPLACE an existing channel block")
  args = ap.parse_args()

  work = tempfile.mkdtemp(prefix="chanfix_")
  try:
    subprocess.run([LT, "unpack", args.src, "--output-dir", work, "--allow-overwrite"],
                   check=True, stdout=subprocess.DEVNULL)
    before = tflite_hashes(work)
    pb = os.path.join(work, "LlmMetadataProto.pbtext")
    s = open(pb).read()
    if "channel_name" in s:
      if not args.force:
        print("ALREADY HAS A CHANNEL — nothing to do (use --force to replace)")
        return 0
      import re as _re
      s = _re.sub(r'channels \{\n(?:  .*\n)*?\}\n', "", s)
      print("  (replacing the existing channel block)")
    esc_start = args.start.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    esc_end = args.end.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    open(pb, "w").write(
        s + f'channels {{\n  channel_name: "{args.name}"\n'
            f'  start: "{esc_start}"\n  end: "{esc_end}"\n}}\n')
    subprocess.run([LT, "pack", work, "--output", args.dst, "--allow-overwrite"],
                   check=True, stdout=subprocess.DEVNULL)

    verify = tempfile.mkdtemp(prefix="chanverify_")
    try:
      subprocess.run([LT, "unpack", args.dst, "--output-dir", verify, "--allow-overwrite"],
                     check=True, stdout=subprocess.DEVNULL)
      after = tflite_hashes(verify)
    finally:
      shutil.rmtree(verify, ignore_errors=True)

    if before != after:
      print("WEIGHTS CHANGED — refusing to call this a metadata-only fix", file=sys.stderr)
      for k in sorted(set(before) | set(after)):
        if before.get(k) != after.get(k):
          print(f"  {k}: {before.get(k)} -> {after.get(k)}", file=sys.stderr)
      return 1
    # Bundle-level: every section except the metadata one must survive with its
    # content intact, and the metadata may differ from the source only in `channels`.
    section_fails = verify_sections(args.src, args.dst)
    if section_fails:
      print("SECTION CONTENT CHANGED — refusing to call this a metadata-only fix",
            file=sys.stderr)
      for msg in section_fails:
        print(f"  {msg}", file=sys.stderr)
      return 1
    print(f"OK  {args.dst}")
    print(f"    channel {args.name!r} = {args.start!r}..{args.end!r}")
    print(f"    {len(before)} tflite section(s) byte-identical; "
          f"{os.path.getsize(args.src)} -> {os.path.getsize(args.dst)} bytes")
    return 0
  finally:
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
  raise SystemExit(main())
