#!/usr/bin/env python3
"""make_bundle.py --tflite MODEL.tflite --out OUT.litertlm --executor-metadata executor_metadata.pb
                  [--tokenizer tokenizer.json] [--act fp32|none] [--max-tokens N] [--pbtext P]

Packs one prefill/decode .tflite (export_hf's, after quantization / the expert graft) into a .litertlm:
  LlmMetadata = LiteRT-LM's own models/qwen3_5/LlmMetadataProto.pbtext at v0.19.1 verbatim (template/, fetched by
                reproduce.sh: jinja inline, stop <|im_end|>, pad <|endoftext|> = the token the exported prefill graph
                masks, generic_model, thought channel, max_num_tokens 4096); --max-tokens only when the graph's cache
                is shorter.
  tokenizer   = the repo's own tokenizer.json bytes (sha256 checked against the Hub's LFS sha256 at the pinned
                revision, TOKENIZER_SHA256 below).
  ExecutorMetadata = export_hf's own `executor_metadata.pb` of the same export (work dir, kept by
                keep_temporary_files): the state buffers it derives from layer_types (global KV, linear attention, conv),
                which the runtime reads (LiteRT-LM runtime/executor/litert/state.cc StateBuffer types); the same section
                export_hf's own bundle carries (core/litert_lm_builder.py package_model).
  TFLiteModel = declares prefer_activation_type (fp32 by default: a hybrid bundle answered nothing with fp16 GPU
                activations in our runs; reproduce.sh passes --act fp16 for the int8 GPU file, which passed its checks
                with fp16 and runs faster).
Writes OUT.build.json (section paths and sha256, the metadata text without the jinja).
From ../qwen3_30b_a3b_work/make_bundle.py; changes: the qwen3_5 pbtext, the tokenizer path and its pinned sha256.
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# tokenizer.json of Qwen/Qwen3.6-35B-A3B at revision 345d467b117d1fd5eda9b879a5988de9ae5f47a7 (the Hub's LFS sha256;
# the same bytes as Qwen/Qwen3.5-35B-A3B's at 59d61f3c)
TOKENIZER_SHA256 = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"


def sha256(path):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for c in iter(lambda: f.read(64 << 20), b""):
      h.update(c)
  return h.hexdigest()


def pinned_tokenizer_sha():
  return TOKENIZER_SHA256


F_PUNCHHOLE = 99  # <sys/fcntl.h>; fpunchhole_t = {uint32 fp_flags, uint32 reserved, off_t fp_offset, off_t fp_length}


def punch_hole(fd, offset, length):
  import fcntl
  import struct
  fcntl.fcntl(fd, F_PUNCHHOLE, struct.pack("<IIqq", 0, 0, offset, length))


def consuming_copy(src_path, orig_copy, chunk=64 << 20, block=4096):
  """A stand-in for the builder's _copy_file_to_stream: other files go to the original; src_path is copied in chunks
  and its blocks before the copied point are deallocated once the chunk is fsync'ed to the bundle."""
  src_real = os.path.realpath(src_path)
  state = {"used": False}

  def copy(f_src, f_dst, buffer_size=1024 * 1024):
    name = getattr(f_src, "name", None)
    if not isinstance(name, str) or os.path.realpath(name) != src_real:
      return orig_copy(f_src, f_dst, buffer_size)
    f_dst.flush()
    begin = f_dst.tell()
    size = os.fstat(f_src.fileno()).st_size
    h = hashlib.sha256()
    wfd = os.open(src_real, os.O_RDWR)
    done = punched = 0
    try:
      while done < size:
        buf = f_src.read(min(chunk, size - done))
        if not buf:
          raise IOError(f"short read at {done} of {size}")
        f_dst.write(buf)
        h.update(buf)
        done += len(buf)
        f_dst.flush()
        os.fsync(f_dst.fileno())
        upto = (done // block) * block
        if upto > punched:
          punch_hole(wfd, punched, upto - punched)
          punched = upto
    finally:
      os.close(wfd)
    state.update(used=True, begin=begin, size=size, sha256=h.hexdigest(), punched=punched)
    return None

  return {"copy": copy, "state": state}


def section_sha256(path, begin, size, chunk=64 << 20):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    f.seek(begin)
    left = size
    while left:
      b = f.read(min(chunk, left))
      if not b:
        raise IOError("short read")
      h.update(b)
      left -= len(b)
  return h.hexdigest()


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--tflite", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--tokenizer", required=True)
  ap.add_argument("--executor-metadata", required=True)
  ap.add_argument("--act", default="fp32")
  ap.add_argument("--max-tokens", type=int)
  ap.add_argument("--pbtext", default=os.path.join(HERE, "template", "LlmMetadataProto.pbtext"))
  ap.add_argument("--consume-tflite", action="store_true",
                  help="free the TFLite's blocks as the builder copies them (F_PUNCHHOLE), then delete it")
  a = ap.parse_args()
  import litert_lm_builder as lb
  from google.protobuf import text_format
  from litert_lm_builder.runtime.proto import llm_metadata_pb2

  tok = os.path.realpath(a.tokenizer)
  tok_sha = sha256(tok)
  want = pinned_tokenizer_sha()
  assert tok_sha == want, (tok_sha, want)

  md = llm_metadata_pb2.LlmMetadata()
  text_format.Parse(open(a.pbtext).read(), md)
  if a.max_tokens:
    md.max_num_tokens = a.max_tokens
  od = os.path.dirname(os.path.abspath(a.out))
  os.makedirs(od, exist_ok=True)
  md_path = a.out + ".llm_metadata.pb"
  with open(md_path, "wb") as f:
    f.write(md.SerializeToString())
  jinja = md.jinja_prompt_template
  shown = llm_metadata_pb2.LlmMetadata()
  shown.CopyFrom(md)
  shown.ClearField("jinja_prompt_template")
  md_text = text_format.MessageToString(shown) + f"jinja_prompt_template: <{len(jinja)} chars>\n"

  b = lb.LitertLmFileBuilder()
  b.add_system_metadata(lb.Metadata(key="Authors", value="", dtype=lb.DType.STRING))
  b.add_llm_metadata(md_path)
  b.add_executor_metadata(a.executor_metadata)
  # the builder wants a path ending in .json; a.tokenizer may be a snapshot link, its bytes are the blob's
  b.add_hf_tokenizer(a.tokenizer if a.tokenizer.endswith(".json") else tok)
  b.add_tflite_model(a.tflite, "prefill_decode", prefer_activation_type=(None if a.act == "none" else a.act))
  consumed = None
  if a.consume_tflite:
    # A 27 GB TFLite and its bundle need twice the disk at once. The builder
    # copies the TFLite into the bundle in order (_copy_file_to_stream); this copy writes the same bytes and, after
    # each 64 MiB is written and fsync'ed, deallocates the source's blocks before that point. Afterwards the bundle's
    # section is read back and hashed against the source's sha256 taken here, before the copy.
    import litert_lm_builder.litertlm_builder as lbb
    tfl_sha = sha256(a.tflite)
    consumed = consuming_copy(a.tflite, lbb._copy_file_to_stream)  # pylint: disable=protected-access
    lbb._copy_file_to_stream = consumed["copy"]  # pylint: disable=protected-access
  with open(a.out, "wb") as f:
    b.build(f)
  if consumed is not None:
    st = consumed["state"]
    if not st.get("used") or st["sha256"] != tfl_sha:
      raise SystemExit(f"CONSUME_FAIL the TFLite copy did not run or its bytes differ: {st}")
    back = section_sha256(a.out, st["begin"], st["size"])
    if back != tfl_sha:
      raise SystemExit(f"CONSUME_FAIL the bundle's section reads back as {back}, the TFLite was {tfl_sha}")
    os.remove(a.tflite)
    print(f"CONSUMED {a.tflite}: section at {st['begin']} + {st['size']} B, read back sha256 = source, "
          f"{st['punched']} B deallocated during the copy; the source file is deleted", flush=True)
  else:
    tfl_sha = sha256(a.tflite)
  rec = {"bundle": os.path.abspath(a.out), "bytes": os.path.getsize(a.out), "tflite": os.path.abspath(a.tflite),
         "tflite_sha256": tfl_sha, "tokenizer": a.tokenizer, "tokenizer_sha256": tok_sha,
         "tflite_consumed": None if consumed is None else {k: v for k, v in consumed["state"].items()},
         "pbtext": a.pbtext, "pbtext_sha256": sha256(a.pbtext),
         "jinja_sha256": hashlib.sha256(jinja.encode()).hexdigest(), "prefer_activation_type": a.act,
         "executor_metadata": os.path.abspath(a.executor_metadata),
         "executor_metadata_sha256": sha256(a.executor_metadata),
         "max_num_tokens": md.max_num_tokens, "llm_metadata_text": md_text, "bundle_sha256": sha256(a.out)}
  json.dump(rec, open(a.out + ".build.json", "w"), indent=1)
  print("BUNDLE_DONE", a.out, rec["bytes"], "B", rec["bundle_sha256"][:16])
  return 0


if __name__ == "__main__":
  sys.exit(main())
