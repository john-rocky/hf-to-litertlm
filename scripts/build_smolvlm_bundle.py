"""Assemble the SmolVLM2 fast_vlm `.litertlm` bundle. SmolVLM uses its own chat
format (User:/Assistant:/<end_of_utterance>, a single leading <|im_start|>, and the
image wrapped by <fake_token_around_image><global-img>...<fake_token_around_image>).

    DEC=... VIS=... ~/clipconv/bin/python scripts/build_smolvlm_bundle.py
"""
import os

import litert_lm_builder as litertlm_builder
from litert_lm_builder.runtime.proto import llm_metadata_pb2
from litert_lm_builder.runtime.proto import llm_model_type_pb2

DEC = os.environ.get("DEC", "out/smolvlm2-decoder")
VIS = os.environ.get("VIS", "out/smolvlm2-vision")
OUT = "out/internvl-bundle"
os.makedirs(OUT, exist_ok=True)
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "512"))
MAX_TOKENS = int(os.environ.get("CACHE", "2048"))

# image -> <fake_token_around_image><global-img><image_soft_token><fake_token_around_image>
IMG = "<fake_token_around_image><global-img><image_soft_token><fake_token_around_image>"
JINJA = (
    "<|im_start|>"
    "{%- for message in messages -%}"
    "{%- if message.content is string -%}"
    "{{ message.role|capitalize }}: {{ message.content }}<end_of_utterance>\n"
    "{%- else -%}"
    "{{ message.role|capitalize }}:"
    "{%- for item in message.content %}"
    "{%- if item.type == 'text' %} {{ item.text }}"
    "{%- elif item.type == 'image' -%}" + IMG +
    "{%- endif -%}{%- endfor -%}<end_of_utterance>\n"
    "{%- endif -%}{%- endfor -%}"
    "{%- if add_generation_prompt %}Assistant:{% endif -%}"
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
  ve8 = os.path.join(VIS, "vision_encoder_int8.tflite")
  va8 = os.path.join(VIS, "vision_adapter_int8.tflite")
  vision_encoder = ve8 if os.path.exists(ve8) else os.path.join(VIS, "vision_encoder.tflite")
  vision_adapter = va8 if os.path.exists(va8) else os.path.join(VIS, "vision_adapter.tflite")
  sp = [os.path.join(DEC, f) for f in os.listdir(DEC) if f.endswith((".spiece", ".model", ".spm"))]
  print("sections:", embedder, prefill_decode, vision_encoder, vision_adapter, "| sp:", sp)

  md = llm_metadata_pb2.LlmMetadata()
  md.max_num_tokens = MAX_TOKENS
  md.start_token.token_str = "<|im_start|>"
  md.prompt_templates.user.prefix = "User:"
  md.prompt_templates.user.suffix = "<end_of_utterance>\n"
  md.prompt_templates.model.prefix = "Assistant:"
  md.prompt_templates.model.suffix = "<end_of_utterance>\n"
  md.prompt_templates.system.prefix = "System:"
  md.prompt_templates.system.suffix = "<end_of_utterance>\n"
  md.jinja_prompt_template = JINJA
  md.llm_model_type.CopyFrom(
      llm_model_type_pb2.LlmModelType(fast_vlm=llm_model_type_pb2.FastVlm()))
  md.llm_model_type.fast_vlm.image_tensor_height = IMAGE_SIZE
  md.llm_model_type.fast_vlm.image_tensor_width = IMAGE_SIZE
  md.stop_tokens.add().token_str = "<end_of_utterance>"
  md_path = os.path.join(OUT, "smolvlm_llm_metadata.pb")
  with open(md_path, "wb") as f:
    f.write(md.SerializeToString())

  b = litertlm_builder.LitertLmFileBuilder()
  b.add_system_metadata(litertlm_builder.Metadata(
      key="Authors", value="", dtype=litertlm_builder.DType.STRING))
  b.add_llm_metadata(md_path)
  if sp:
    b.add_sentencepiece_tokenizer(sp[0]); print("added SP tokenizer:", sp[0])
  else:
    b.add_hf_tokenizer(os.path.join("src_models/smolvlm2-500m-llm", "tokenizer.json"))
    print("added HF tokenizer")
  b.add_tflite_model(embedder, litertlm_builder.TfLiteModelType.EMBEDDER)
  b.add_tflite_model(prefill_decode, litertlm_builder.TfLiteModelType.PREFILL_DECODE)
  b.add_tflite_model(vision_encoder, litertlm_builder.TfLiteModelType.VISION_ENCODER)
  b.add_tflite_model(vision_adapter, litertlm_builder.TfLiteModelType.VISION_ADAPTER)
  out_path = os.path.join(OUT, os.environ.get("OUT_NAME", "SmolVLM2-500M.litertlm"))
  with open(out_path, "wb") as f:
    b.build(f)
  print("BUNDLE_DONE", out_path, round(os.path.getsize(out_path) / 1e6, 1), "MB")


if __name__ == "__main__":
  main()
