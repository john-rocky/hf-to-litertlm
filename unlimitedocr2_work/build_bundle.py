#!/usr/bin/env python3
"""Pack the Unlimited-OCR-2 .litertlm with litert_lm_builder (litert-lm-builder 0.19.0).

  python build_bundle.py OUT.litertlm --embedder E.tflite --decoder D.tflite --tokenizer tokenizer.json
         --jinja T.jinja [--type generic|fast_vlm] [--image-size 1024] [--vision-encoder VE.tflite
         --vision-adapter VA.tflite] [--start-token-id 0] [--stop-ids 1] [--dec-act fp32] [--cache 4096]
         [--executor EXEC.pbtext]

The tokenizer section is the given tokenizer.json file as is (litert_lm_builder add_hf_tokenizer reads the bytes).
The structured prompt_templates are present with empty prefixes and suffixes (the HF prompt of this model is the plain
text with no role markers; a fast_vlm conversation needs the structured templates present),
and the jinja template is the given file byte for byte. The decoder section declares prefer_activation_type
(default fp32). --executor adds an ExecutorMetadata section from a pbtext (the state_buffers block that
scripts/add_executor_metadata.py's build_pbtext writes) in the same build, so no unpack / re-pack pass is needed.
Prints BUNDLE_DONE <path> <bytes> and the LlmMetadata as text.
"""
import argparse
import os

import litert_lm_builder as lb
from google.protobuf import text_format
from litert_lm_builder.runtime.proto import llm_metadata_pb2


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("out")
  ap.add_argument("--embedder", required=True)
  ap.add_argument("--decoder", required=True)
  ap.add_argument("--tokenizer", required=True)
  ap.add_argument("--jinja", required=True)
  ap.add_argument("--type", default="generic", choices=("generic", "fast_vlm"))
  ap.add_argument("--image-size", type=int, default=1024)
  ap.add_argument("--vision-encoder")
  ap.add_argument("--vision-adapter")
  ap.add_argument("--start-token-id", type=int, default=0)
  ap.add_argument("--no-start-token", action="store_true")
  ap.add_argument("--stop-ids", default="1")
  ap.add_argument("--dec-act", default="fp32")
  ap.add_argument("--cache", type=int, default=4096)
  ap.add_argument("--executor")
  a = ap.parse_args()
  md = llm_metadata_pb2.LlmMetadata()
  md.max_num_tokens = a.cache
  if not a.no_start_token:
    md.start_token.token_ids.ids.append(a.start_token_id)
  for role in ("user", "model", "system"):
    t = getattr(md.prompt_templates, role)
    t.SetInParent()
    t.prefix = ""
    t.suffix = ""
  md.jinja_prompt_template = open(a.jinja).read()
  if a.type == "fast_vlm":
    md.llm_model_type.fast_vlm.image_tensor_height = a.image_size
    md.llm_model_type.fast_vlm.image_tensor_width = a.image_size
  else:
    md.llm_model_type.generic_model.SetInParent()
  for s in a.stop_ids.split(","):
    md.stop_tokens.add().token_ids.ids.append(int(s))
  d = os.path.dirname(os.path.abspath(a.out))
  os.makedirs(d, exist_ok=True)
  md_path = a.out + ".llm_metadata.pb"
  with open(md_path, "wb") as f:
    f.write(md.SerializeToString())
  b = lb.LitertLmFileBuilder()
  b.add_system_metadata(lb.Metadata(key="Authors", value="", dtype=lb.DType.STRING))
  b.add_llm_metadata(md_path)
  b.add_hf_tokenizer(a.tokenizer)
  b.add_tflite_model(a.embedder, "embedder")
  b.add_tflite_model(a.decoder, "prefill_decode", prefer_activation_type=a.dec_act or None)
  if a.vision_encoder:
    b.add_tflite_model(a.vision_encoder, "vision_encoder")
    b.add_tflite_model(a.vision_adapter, "vision_adapter")
  if a.executor:
    b.add_executor_metadata(a.executor)
  with open(a.out, "wb") as f:
    b.build(f)
  os.remove(md_path)
  print("BUNDLE_DONE", a.out, os.path.getsize(a.out), flush=True)
  print(text_format.MessageToString(md)[:3000])


if __name__ == "__main__":
  main()
