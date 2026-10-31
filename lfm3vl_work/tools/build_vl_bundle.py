#!/usr/bin/env python3
"""Assemble an LFM3-VL-2B .litertlm VLM bundle (litert-lm-builder 0.18.1).

  PY_EXPORT tools/build_vl_bundle.py --name NAME --out-dir OUT --src SRC --template VL.jinja --g 512 \
      --decoder-bundle DEC/float/model.litertlm --embedder-tflite DEC/emb_wi8.tflite --decoder-tflite DEC/int4.tflite \
      --encoder TOWER/vision_encoder_g512_v2.tflite --adapter TOWER/vision_adapter_g512_v2_fp16.tflite --vision-act fp32

Decoder bundle = the float export of tools/vl_decoder.py (lfm3_ext.export_lfm3 on the VL checkpoint: key prefix
model.language_model., text_config, externalize_embedder=True, single_token_embedder=True): its ExecutorMetadata is
taken as it is (litertlm_peek dump), and its LlmMetadata gives max_num_tokens, channels and
supports_thinking / min_runtime_version; --embedder-tflite / --decoder-tflite replace its EMBEDDER and PREFILL_DECODE
graphs with the quantized ones.
Added here: VISION_ENCODER + VISION_ADAPTER (tools/export_vision.py), the source tokenizer.json BYTES (sha256
a39eed58...c7e7), and LlmMetadata = model type fast_vlm {G, G} (or generic_model {image_enabled,
delimiter_regex = image_token_regex = "(<image_soft_token>)", image_tensor_height / width G}), jinja = the VL template,
start token 124894 (the runtime renders bos_token "" and prepends it), stop 124900 (generation_config eos), structured
prompt_templates present and empty (a jinja-only metadata gives a null conversation).
"""
import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile

TOKENIZER_SHA256 = "a39eed58008cbcbc09e5b629b81338a654269c64cd13348aaa6090f4db75c7e7"
BOS, EOS = 124894, 124900
SOFT_RX = "(<image_soft_token>)"


def sha256_file(p):
  h = hashlib.sha256()
  with open(p, "rb") as f:
    for c in iter(lambda: f.read(1 << 24), b""):
      h.update(c)
  return h.hexdigest()


def peek(path, dump_dir=None):
  from litert_lm_builder import litertlm_peek
  out = io.StringIO()
  litertlm_peek.peek_litertlm_file(path, dump_dir, out)
  return out.getvalue()


def read_llm_metadata(pbtext_path):
  from google.protobuf import text_format
  from litert_lm_builder.runtime.proto import llm_metadata_pb2
  md = llm_metadata_pb2.LlmMetadata()
  text_format.Parse(open(pbtext_path).read(), md)
  return md


def metadata(a, src_md, tokenizer):
  from litert_lm_builder.runtime.proto import llm_metadata_pb2, llm_model_type_pb2
  from tokenizers import Tokenizer
  tk = Tokenizer.from_file(tokenizer)
  assert tk.id_to_token(BOS) == "<|startoftext|>" and tk.id_to_token(EOS) == "<|im_end|>"
  for t, want in (("<|vision_start|>", 125009), ("<|vision_end|>", 125010), ("<think>", 124901), ("</think>", 124902)):
    assert tk.token_to_id(t) == want, (t, tk.token_to_id(t))
  assert tk.token_to_id("<image_soft_token>") is None
  md = llm_metadata_pb2.LlmMetadata()
  md.max_num_tokens = src_md.max_num_tokens
  md.start_token.token_ids.ids.append(BOS)
  md.stop_tokens.add().token_ids.ids.append(EOS)
  for affix in (md.prompt_templates.user, md.prompt_templates.model, md.prompt_templates.system):
    affix.SetInParent()
  md.jinja_prompt_template = open(a.template).read()
  for ch in src_md.channels:
    md.channels.add().CopyFrom(ch)
  for f in ("supports_thinking", "supports_function_calling", "min_runtime_version"):
    if src_md.HasField(f):
      setattr(md, f, getattr(src_md, f))
  if a.model_type == "fast_vlm":
    md.llm_model_type.CopyFrom(llm_model_type_pb2.LlmModelType(fast_vlm=llm_model_type_pb2.FastVlm()))
    md.llm_model_type.fast_vlm.image_tensor_height = a.g
    md.llm_model_type.fast_vlm.image_tensor_width = a.g
  else:
    gm = llm_model_type_pb2.GenericModel(image_enabled=True, delimiter_regex=SOFT_RX, image_token_regex=SOFT_RX,
                                         image_tensor_height=a.g, image_tensor_width=a.g)
    md.llm_model_type.CopyFrom(llm_model_type_pb2.LlmModelType(generic_model=gm))
  return md


def build(a):
  import importlib.metadata as imd
  import litert_lm_builder as lb
  from google.protobuf import text_format
  tokenizer = os.path.join(os.path.abspath(a.src), "tokenizer.json")
  assert sha256_file(tokenizer) == TOKENIZER_SHA256, "not the source repository's tokenizer.json"
  out_dir = os.path.abspath(a.out_dir)
  os.makedirs(out_dir, exist_ok=True)
  out_path = os.path.join(out_dir, f"{a.name}.litertlm")
  assert not os.path.exists(out_path), f"refusing to overwrite {out_path}"
  dump = tempfile.mkdtemp(prefix=f"{a.name}_src_", dir=out_dir)
  try:
    peek(a.decoder_bundle, dump)
    files = sorted(os.listdir(dump))
    emb = [f for f in files if f.endswith("_tf_lite_embedder.tflite")]
    dec = [f for f in files if f.endswith("_tf_lite_prefill_decode.tflite")]
    wts = [f for f in files if f.endswith(".weight")]
    assert len(emb) == 1 and len(dec) == 1, files
    assert not wts, f"weight sections in the decoder bundle are not handled: {wts}"
    src_md = read_llm_metadata(os.path.join(dump, "LlmMetadataProto.pbtext"))
    exec_md = os.path.join(dump, "ExecutorMetadataProto.pbtext")
    assert os.path.exists(exec_md), files
    md = metadata(a, src_md, tokenizer)
    md_path = os.path.join(out_dir, f"{a.name}.llm_metadata.pb")
    open(md_path, "wb").write(md.SerializeToString())
    open(os.path.join(out_dir, f"{a.name}.llm_metadata.pbtext"), "w").write(text_format.MessageToString(md))
    src_start = list(src_md.start_token.token_ids.ids) if src_md.HasField("start_token") else None
    src_stop = [list(s.token_ids.ids) or s.token_str for s in src_md.stop_tokens]
    b = lb.LitertLmFileBuilder()
    b.add_system_metadata(lb.Metadata(key="Authors", value="", dtype=lb.DType.STRING))
    b.add_llm_metadata(md_path)
    b.add_hf_tokenizer(tokenizer)
    emb_p = os.path.abspath(a.embedder_tflite) if a.embedder_tflite else os.path.join(dump, emb[0])
    dec_p = os.path.abspath(a.decoder_tflite) if a.decoder_tflite else os.path.join(dump, dec[0])
    sections = [("EMBEDDER", emb_p, None), ("PREFILL_DECODE", dec_p, a.act or None),
                ("VISION_ENCODER", os.path.abspath(a.encoder), a.vision_act or None),
                ("VISION_ADAPTER", os.path.abspath(a.adapter), a.vision_act or None)]
    inputs = {}
    for kind, p, act in sections:
      inputs[kind] = {"path": f"{a.decoder_bundle}:{os.path.basename(p)}" if p.startswith(dump) else p,
                      "bytes": os.path.getsize(p), "sha256": sha256_file(p), "prefer_activation_type": act}
      b.add_tflite_model(p, getattr(lb.TfLiteModelType, kind), prefer_activation_type=act)
    b.add_executor_metadata(exec_md)
    with open(out_path, "wb") as f:
      b.build(f)
    rec = {"status": "DONE", "output": out_path, "bytes": os.path.getsize(out_path),
           "sha256": sha256_file(out_path), "builder": imd.version("litert-lm-builder"), "model_type": a.model_type,
           "g": a.g, "inputs": inputs, "decoder_bundle": a.decoder_bundle,
           "decoder_bundle_sha256": sha256_file(a.decoder_bundle),
           "decoder_bundle_start": src_start, "decoder_bundle_stop": src_stop,
           "executor_metadata": open(exec_md).read(), "metadata_pbtext": text_format.MessageToString(md)}
    open(os.path.join(out_dir, f"{a.name}.build.json"), "w").write(json.dumps(rec, indent=1) + "\n")
    print("BUNDLE_BUILT", rec["output"], rec["bytes"], rec["sha256"], flush=True)
  finally:
    shutil.rmtree(dump, ignore_errors=True)
  return 0


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--name", required=True)
  ap.add_argument("--out-dir", required=True)
  ap.add_argument("--src", required=True, help="the source snapshot folder (its tokenizer.json is bundled as is)")
  ap.add_argument("--template", required=True, help="the VL chat template (tools/vl_template.py)")
  ap.add_argument("--model-type", choices=("fast_vlm", "generic"), default="fast_vlm")
  ap.add_argument("--g", type=int, default=512)
  ap.add_argument("--decoder-bundle", required=True)
  ap.add_argument("--encoder", required=True)
  ap.add_argument("--adapter", required=True)
  ap.add_argument("--embedder-tflite", default="", help="an EMBEDDER tflite replacing the decoder bundle's (e.g. wi8)")
  ap.add_argument("--decoder-tflite", default="", help="a PREFILL_DECODE tflite replacing the decoder bundle's")
  ap.add_argument("--act", default="", help="PREFILL_DECODE prefer_activation_type ('' = not declared)")
  ap.add_argument("--vision-act", default="", help="VISION_* prefer_activation_type ('' = not declared)")
  return build(ap.parse_args())


if __name__ == "__main__":
  sys.exit(main())
