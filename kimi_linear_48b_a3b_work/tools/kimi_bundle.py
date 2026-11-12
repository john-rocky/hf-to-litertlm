#!/usr/bin/env python3
"""kimi_bundle.py --tflite MODEL.tflite --cut CUT_DIR --out OUT.litertlm [--cache N] [--act fp32|none]

Packs one kimi_export.py graph (prefill_<T> + decode signatures, fp32 or quantized) into a .litertlm with
Kimi-Linear's own metadata (there is no official LiteRT-LM config for this model):
  LlmMetadata    stop <|im_end|> = 163586 (generation_config eos), pad [PAD] = 163839 (the token the prefill graph masks,
                 kimi_ext.prefill_valid; the executor pads the prefill window with it), no start token (the author's
                 template and tokenizer add no BOS), generic_model, max_num_tokens = the graph's cache,
                 jinja = template/chat_template.lite.jinja (the vendor template with four rewrites for the
                 runtime's minijinja, README). No sampler params (the author ships none,
                 generation_config.json holds only bos / eos / pad).
  ExecutorMetadata  the state buffers from the cut's layer_types, named as kimi_ext writes them: a KDA layer i =
                 kv_cache_c_i and kv_cache_r_i (TYPE_LINEAR_ATTENTION), an MLA layer i = kv_cache_k_i (the 512-wide
                 latent, TYPE_GLOBAL_KEY_CACHE) and kv_cache_v_i (the 64-wide k_rot, TYPE_GLOBAL_VALUE_CACHE), sequence
                 axis 2, maximum_sequence_length = the cache (export_hf's metadata_builder.py does the same per type).
  tokenizer      tokjson/tokenizer.json (tok_convert.py from the author's tiktoken.model; sha256 checked against
                 tok_convert.json).
  TFLiteModel    prefer_activation_type fp32 by default.
Writes OUT.build.json (section inputs and sha256, the metadata as text without the jinja).
"""
import argparse
import hashlib
import json
import os
import sys

K = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOP_ID, PAD_ID = 163586, 163839
F_PUNCHHOLE = 99  # <sys/fcntl.h>; fpunchhole_t = {uint32 fp_flags, uint32 reserved, off_t fp_offset, off_t fp_length}


def sha256(path):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for c in iter(lambda: f.read(64 << 20), b""):
      h.update(c)
  return h.hexdigest()


# --consume-tflite: a consuming copy, so that the grafted TFLite and its bundle do not need the disk twice (macOS
# F_PUNCHHOLE). The builder copies the TFLite into the bundle in order (_copy_file_to_stream); this copy writes the same
# bytes and, after each 64 MiB is written and fsync'ed, deallocates the source's blocks before that point. Afterwards
# the bundle's section is read back and hashed against the source's sha256 taken before the copy, then the source is
# deleted.
def punch_hole(fd, offset, length):
  import fcntl
  import struct
  fcntl.fcntl(fd, F_PUNCHHOLE, struct.pack("<IIqq", 0, 0, offset, length))


def consuming_copy(src_path, orig_copy, chunk=64 << 20, block=4096):
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
  ap.add_argument("--cut", required=True, help="the checkpoint dir whose config.json the graph was exported from")
  ap.add_argument("--out", required=True)
  ap.add_argument("--cache", type=int, help="default: the export's own record (<tflite>.export.json)")
  ap.add_argument("--act", default="fp32")
  ap.add_argument("--template", default=os.path.join(K, "template", "chat_template.lite.jinja"))
  ap.add_argument("--tokenizer", default=os.path.join(K, "tokjson", "tokenizer.json"))
  ap.add_argument("--consume-tflite", action="store_true",
                  help="free the TFLite's blocks as the builder copies them (F_PUNCHHOLE), then delete it")
  a = ap.parse_args()
  import litert_lm_builder as lb
  from google.protobuf import text_format
  from litert_lm_builder.runtime.proto import executor_metadata_pb2 as em
  from litert_lm_builder.runtime.proto import llm_metadata_pb2

  cache = a.cache
  rec_path = a.tflite + ".export.json"
  if cache is None:
    cache = json.load(open(rec_path))["cache"]
  tok_sha = sha256(a.tokenizer)
  want = json.load(open(os.path.join(K, "tokjson", "tok_convert.json")))["tokenizer_json_sha256"]
  assert tok_sha == want, (tok_sha, want)
  cfg = json.load(open(os.path.join(a.cut, "config.json")))
  n = cfg["num_hidden_layers"]
  kda = set(cfg["linear_attn_config"]["kda_layers"])  # 1-based (configuration_kimi.py: is_kda_layer(i) = i + 1 in ...)

  md = llm_metadata_pb2.LlmMetadata()
  md.stop_tokens.add().token_ids.ids.append(STOP_ID)
  md.pad_token.token_ids.ids.append(PAD_ID)
  md.max_num_tokens = cache
  md.llm_model_type.generic_model.SetInParent()
  md.jinja_prompt_template = open(a.template).read()
  out = os.path.abspath(a.out)
  os.makedirs(os.path.dirname(out), exist_ok=True)
  md_path = out + ".llm_metadata.pb"
  open(md_path, "wb").write(md.SerializeToString())
  shown = llm_metadata_pb2.LlmMetadata()
  shown.CopyFrom(md)
  shown.ClearField("jinja_prompt_template")

  ex = em.ExecutorMetadata()
  lm = ex.llm_executor_metadata
  layout = []
  for i in range(n):
    names = ([(f"kv_cache_c_{i}", em.StateBuffer.TYPE_LINEAR_ATTENTION, None),
              (f"kv_cache_r_{i}", em.StateBuffer.TYPE_LINEAR_ATTENTION, None)] if (i + 1) in kda else
             [(f"kv_cache_k_{i}", em.StateBuffer.TYPE_GLOBAL_KEY_CACHE, 2),
              (f"kv_cache_v_{i}", em.StateBuffer.TYPE_GLOBAL_VALUE_CACHE, 2)])
    for name, typ, axis in names:
      b = lm.state_buffers.add()
      b.prefill_input_name = b.prefill_output_name = b.decode_input_name = b.decode_output_name = name
      b.policy = em.StateBuffer.POLICY_DEFAULT
      b.type = typ
      if axis is not None:
        b.sequence_axis = axis
        b.maximum_sequence_length = cache
      layout.append(name)
  ex_path = out + ".executor_metadata.pb"
  open(ex_path, "wb").write(ex.SerializeToString())

  bld = lb.LitertLmFileBuilder()
  bld.add_system_metadata(lb.Metadata(key="Authors", value="", dtype=lb.DType.STRING))
  bld.add_llm_metadata(md_path)
  bld.add_executor_metadata(ex_path)
  bld.add_hf_tokenizer(a.tokenizer)
  bld.add_tflite_model(a.tflite, "prefill_decode", prefer_activation_type=(None if a.act == "none" else a.act))
  tfl_sha = sha256(a.tflite)
  consumed = None
  if a.consume_tflite:
    import litert_lm_builder.litertlm_builder as lbb
    consumed = consuming_copy(a.tflite, lbb._copy_file_to_stream)  # pylint: disable=protected-access
    lbb._copy_file_to_stream = consumed["copy"]  # pylint: disable=protected-access
  with open(out, "wb") as f:
    bld.build(f)
  if consumed is not None:
    st = consumed["state"]
    if not st.get("used") or st["sha256"] != tfl_sha:
      raise SystemExit(f"CONSUME_FAIL the TFLite copy did not run or its bytes differ: {st}")
    back = section_sha256(out, st["begin"], st["size"])
    if back != tfl_sha:
      raise SystemExit(f"CONSUME_FAIL the bundle's section reads back as {back}, the TFLite was {tfl_sha}")
    os.remove(a.tflite)
    print(f"CONSUMED {a.tflite}: section at {st['begin']} + {st['size']} B, read back sha256 = source, "
          f"{st['punched']} B deallocated during the copy; the source file is deleted", flush=True)
  rec = {"bundle": out, "bytes": os.path.getsize(out), "tflite": os.path.abspath(a.tflite),
         "tflite_sha256": tfl_sha, "tflite_consumed": None if consumed is None else consumed["state"],
         "tokenizer": a.tokenizer, "tokenizer_sha256": tok_sha,
         "template": a.template, "template_sha256": sha256(a.template), "cache": cache, "act": a.act,
         "llm_metadata_text": text_format.MessageToString(shown), "executor_metadata_text": text_format.MessageToString(ex),
         "state_buffers": layout, "builder": lb.__file__, "bundle_sha256": sha256(out)}
  json.dump(rec, open(out + ".build.json", "w"), indent=1)
  print("BUNDLE_DONE", out, rec["bytes"], "B", rec["bundle_sha256"][:16])
  return 0


if __name__ == "__main__":
  sys.exit(main())
