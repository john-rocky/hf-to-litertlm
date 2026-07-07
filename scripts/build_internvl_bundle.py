"""Hand-assemble an InternVL3-2B fast_vlm `.litertlm` bundle from the converted
sections, matching the verified FastVLM-0.5B layout:

  EMBEDDER       : token_ids[1,1] -> [1,1,1536]
  PREFILL_DECODE : embeddings + KV -> logits + KV
  VISION_ENCODER : image NHWC[1,448,448,3] in [0,1] -> [1,256,4096]
  VISION_ADAPTER : [1,256,4096] -> [1,256,1536]
  + LlmMetadata  : FastVlm(image 448x448) + ChatML jinja template (image -> <img><image_soft_token></img>)

    ~/clipconv/bin/python scripts/build_internvl_bundle.py
"""
import os, sys

import litert_lm_builder as litertlm_builder
from litert_lm_builder.runtime.proto import llm_metadata_pb2
from litert_lm_builder.runtime.proto import llm_model_type_pb2

DEC = os.environ.get("DEC", "out/internvl-decoder")
VIS = os.environ.get("VIS", "out/internvl-vision-split")
OUT = "out/internvl-bundle"
os.makedirs(OUT, exist_ok=True)

IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "448"))
MAX_TOKENS = int(os.environ.get("CACHE", "2048"))
# How an image renders in the prompt. InternVL wraps it in <img>...</img>; LLaVA-OneVision
# uses a bare placeholder. The runtime splits on the literal <image_soft_token>.
IMG_RENDER = os.environ.get("IMG_RENDER", "<img><image_soft_token></img>")

# ChatML jinja template (from FastVLM-0.5B.litertlm); image item rendered as
# <img><image_soft_token></img> to preserve InternVL's image brackets. The runtime
# fast_vlm processor splits the rendered prompt on the literal <image_soft_token>.
JINJA = (
    "{%- for message in messages -%}"
    "{%- if message.content is string -%}"
    "{%- if message.role == 'user' %}<|im_start|>user\n{{ message.content }}<|im_end|>\n{% endif -%}"
    "{%- if message.role == 'model' %}<|im_start|>assistant\n{{ message.content }}<|im_end|>\n{% endif -%}"
    "{%- if message.role == 'system' %}<|im_start|>system\n{{ message.content }}<|im_end|>\n{% endif -%}"
    "{%- else -%}"
    "{%- if message.role == 'user' %}<|im_start|>user\n"
    "{% elif message.role == 'model' %}<|im_start|>assistant\n"
    "{% elif message.role == 'system' %}<|im_start|>system\n{% endif -%}"
    "{%- for item in message.content %}"
    "{%- if item.type == 'text' %}{{ item.text }}"
    "{%- elif item.type == 'image' -%}{{ '" + IMG_RENDER + "' }}"
    "{%- endif -%}{%- endfor -%}"
    "{%- if message.role == 'user' %}<|im_end|>\n"
    "{% elif message.role == 'model' %}<|im_end|>\n"
    "{% elif message.role == 'system' %}<|im_end|>\n{% endif -%}"
    "{%- endif -%}{%- endfor -%}"
    "{%- if add_generation_prompt %}<|im_start|>assistant\n{% endif -%}"
)


def find_tflite(d, *keywords):
  cands = [f for f in os.listdir(d) if f.endswith(".tflite")]
  for kw in keywords:
    for f in cands:
      if kw in f.lower():
        return os.path.join(d, f)
  raise FileNotFoundError(f"no tflite matching {keywords} in {d}: {cands}")


def main():
  embedder = find_tflite(DEC, "embedder_quantized", "embedder")
  prefill_decode = find_tflite(DEC, "model_quantized", "prefill", "decode")
  # prefer int8-quantized vision sections if present (4x smaller, ViT tolerates int8)
  ve8 = os.path.join(VIS, "vision_encoder_int8.tflite")
  va8 = os.path.join(VIS, "vision_adapter_int8.tflite")
  vision_encoder = ve8 if os.path.exists(ve8) else os.path.join(VIS, "vision_encoder.tflite")
  vision_adapter = va8 if os.path.exists(va8) else os.path.join(VIS, "vision_adapter.tflite")
  # tokenizer: prefer a sentencepiece artifact from the decoder export, else HF json.
  tok_mode = os.environ.get("TOK", "auto")  # auto | hf | sp
  sp = [os.path.join(DEC, f) for f in os.listdir(DEC) if f.endswith((".spiece", ".model", ".spm"))]
  if tok_mode == "hf":
    sp = []
  hf = os.path.join("src_models/internvl3-2b-llm", "tokenizer.json")
  print("embedder:", embedder)
  print("prefill_decode:", prefill_decode)
  print("vision_encoder:", vision_encoder, "vision_adapter:", vision_adapter)
  print("sp tokenizer:", sp, "| hf fallback:", hf)

  md = llm_metadata_pb2.LlmMetadata()
  md.max_num_tokens = MAX_TOKENS
  # STRUCTURED prompt_templates (ChatML) — required by the conversation builder
  # alongside jinja (FastVLM bundle has both; jinja-only => conversation_create fails).
  md.start_token.token_str = "None"  # matches FastVLM (no real BOS for ChatML)
  md.prompt_templates.user.prefix = "<|im_start|>user\n"
  md.prompt_templates.user.suffix = "<|im_end|>\n"
  md.prompt_templates.model.prefix = "<|im_start|>assistant\n"
  md.prompt_templates.model.suffix = "<|im_end|>\n"
  md.prompt_templates.system.prefix = "<|im_start|>system\n"
  md.prompt_templates.system.suffix = "<|im_end|>\n"
  md.jinja_prompt_template = JINJA
  md.llm_model_type.CopyFrom(
      llm_model_type_pb2.LlmModelType(fast_vlm=llm_model_type_pb2.FastVlm()))
  md.llm_model_type.fast_vlm.image_tensor_height = IMAGE_SIZE
  md.llm_model_type.fast_vlm.image_tensor_width = IMAGE_SIZE
  md.stop_tokens.add().token_str = "<|im_end|>"
  md_path = os.path.join(OUT, "llm_metadata.pb")
  with open(md_path, "wb") as f:
    f.write(md.SerializeToString())

  b = litertlm_builder.LitertLmFileBuilder()
  b.add_system_metadata(litertlm_builder.Metadata(
      key="Authors", value="", dtype=litertlm_builder.DType.STRING))
  b.add_llm_metadata(md_path)
  if sp:
    b.add_sentencepiece_tokenizer(sp[0])
    print("added SP tokenizer:", sp[0])
  else:
    b.add_hf_tokenizer(hf)
    print("added HF tokenizer:", hf)
  b.add_tflite_model(embedder, litertlm_builder.TfLiteModelType.EMBEDDER)
  b.add_tflite_model(prefill_decode, litertlm_builder.TfLiteModelType.PREFILL_DECODE)
  b.add_tflite_model(vision_encoder, litertlm_builder.TfLiteModelType.VISION_ENCODER)
  b.add_tflite_model(vision_adapter, litertlm_builder.TfLiteModelType.VISION_ADAPTER)

  out_path = os.path.join(OUT, os.environ.get("OUT_NAME", "InternVL3-2B.litertlm"))
  with open(out_path, "wb") as f:
    b.build(f)
  print("BUNDLE_DONE", out_path, round(os.path.getsize(out_path) / 1e6, 1), "MB")


if __name__ == "__main__":
  main()
