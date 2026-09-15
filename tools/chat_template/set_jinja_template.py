#!/usr/bin/env python3
"""Put a Jinja chat template into an existing .litertlm — metadata only, weights untouched.

Replaces `LlmMetadata.jinja_prompt_template` (the runtime prefers it over the structured
prompt_templates when present) and re-packs. After packing it verifies that every
non-metadata section is byte-identical to the input and that the metadata differs in
nothing but jinja_prompt_template.

    python set_jinja_template.py in.litertlm out.litertlm chat_template.jinja [--litert-lm PATH]
"""
import argparse, hashlib, os, shutil, subprocess, sys, tempfile
from google.protobuf import text_format
from litert_lm_builder import litertlm_core
from litert_lm_builder import litertlm_header_schema_py_generated as schema
from litert_lm_builder.runtime.proto import llm_metadata_pb2


def sections(path):
  with open(path, "rb") as f:
    head = f.read(4096)
    assert head[:8] == litertlm_core.HEADER_MAGIC_BYTES, f"{path}: bad magic"
    hdr_end = int.from_bytes(head[litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET:
                                  litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET + 8], "little")
    if hdr_end > len(head):
      f.seek(0); head = f.read(hdr_end + 64)
  meta = schema.LiteRTLMMetaData.GetRootAs(bytearray(head[litertlm_core.HEADER_BEGIN_BYTE_OFFSET:hdr_end]), 0)
  out = []
  for i in range(meta.SectionMetadata().ObjectsLength()):
    so = meta.SectionMetadata().Objects(i)
    out.append((litertlm_core.any_section_data_type_to_string(so.DataType()), so.BeginOffset(), so.EndOffset()))
  return out


def section_sha(path, a, b):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    f.seek(a); left = b - a
    while left > 0:
      chunk = f.read(min(1 << 22, left)); assert chunk; h.update(chunk); left -= len(chunk)
  return h.hexdigest()


def read_section(path, a, b):
  with open(path, "rb") as f:
    f.seek(a); return f.read(b - a)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("template")
  ap.add_argument("--litert-lm", default=os.environ.get("LITERT_LM_BIN", "litert-lm"))
  a = ap.parse_args()
  new_tmpl = open(a.template).read()
  work = tempfile.mkdtemp(prefix="setjinja_")
  try:
    unpack = os.path.join(work, "unpack")
    subprocess.run([a.litert_lm, "unpack", a.src, "--output-dir", unpack], check=True, stdout=subprocess.DEVNULL)
    pb = os.path.join(unpack, "LlmMetadataProto.pbtext")
    md = llm_metadata_pb2.LlmMetadata(); text_format.Parse(open(pb).read(), md)
    old = md.jinja_prompt_template
    md.jinja_prompt_template = new_tmpl
    open(pb, "w").write(text_format.MessageToString(md, as_utf8=True))
    print(f"jinja_prompt_template: {len(old)} -> {len(new_tmpl)} chars; channels={[(c.channel_name, c.start, c.end) for c in md.channels]}; model.prefix kept={md.prompt_templates.model.prefix!r}")
    if os.path.exists(a.dst):
      os.remove(a.dst)  # `pack` exits 0 without writing when the output exists
    toml = os.path.join(unpack, "model.toml")
    subprocess.run([a.litert_lm, "pack", toml if os.path.exists(toml) else unpack, "--output", a.dst], check=True, stdout=subprocess.DEVNULL)
  finally:
    shutil.rmtree(work, ignore_errors=True)
  # verify
  sa, sb = sections(a.src), sections(a.dst)
  fails = []
  if [t for t, _, _ in sa] != [t for t, _, _ in sb]:
    fails.append(f"section list changed: {[t for t,_,_ in sa]} -> {[t for t,_,_ in sb]}")
  else:
    for (ta, aa, ea), (tb, ab, eb) in zip(sa, sb):
      if ta != "LlmMetadataProto":
        if section_sha(a.src, aa, ea) != section_sha(a.dst, ab, eb):
          fails.append(f"{ta} section bytes changed")
        continue
      m1 = llm_metadata_pb2.LlmMetadata(); m1.ParseFromString(read_section(a.src, aa, ea))
      m2 = llm_metadata_pb2.LlmMetadata(); m2.ParseFromString(read_section(a.dst, ab, eb))
      if m2.jinja_prompt_template != new_tmpl:
        fails.append("dst jinja_prompt_template != template file")
      m1.jinja_prompt_template = ""; m2.jinja_prompt_template = ""
      if m1 != m2:
        fails.append("LlmMetadataProto differs beyond jinja_prompt_template")
  if fails:
    print("REFUSING:", *fails, sep="\n  ", file=sys.stderr); os.remove(a.dst); return 1
  print(f"OK {a.dst}: {len(sa)} sections, all non-metadata sections byte-identical; {os.path.getsize(a.src)} -> {os.path.getsize(a.dst)} bytes")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
