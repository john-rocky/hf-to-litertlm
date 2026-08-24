"""Assemble the granite-docling-258M fast_vlm `.litertlm` bundle.

Granite chat format (from the model's own chat_template.jinja):
  <|start_of_role|>user<|end_of_role|>...<|end_of_text|>\n
  <|start_of_role|>assistant<|end_of_role|>...
Image renders as the Idefics3 single-global-image block; the runtime splits on the
literal <image_soft_token>:
  <fake_token_around_image><global-img><image_soft_token><fake_token_around_image>

No start_token on purpose: add_bos_token=False and bos=<|start_of_role|> — a
prepended BOS would double-open the first role turn (the known double-BOS trap).
Tokenizer = HF tokenizer.json (GPT2 byte-BPE, 100k vocab + ~100 DocTags tokens;
SP conversion not attempted — digit/DocTags fidelity is the whole product here).
Stop = <|end_of_text|>. <end_of_utterance> (id 100352) is OUT of the embedding
range (100352 rows) and must never appear in a template.

    DEC=out/docling-decoder VIS=out/docling-vision \
      $PY docling_work/build_granite_docling_bundle.py
"""
import os

import litert_lm_builder as litertlm_builder
from litert_lm_builder.runtime.proto import llm_metadata_pb2
from litert_lm_builder.runtime.proto import llm_model_type_pb2

DEC = os.environ.get("DEC", "out/docling-decoder")
VIS = os.environ.get("VIS", "out/docling-vision")
OUT = os.environ.get("OUT_DIR", "out/docling-bundle")
HF_TOK = os.environ.get("HF_TOK", "src_models/granite-docling-258m/tokenizer.json")
os.makedirs(OUT, exist_ok=True)
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "512"))
MAX_TOKENS = int(os.environ.get("CACHE", "4096"))
VENC = os.environ.get("VENC", "")   # e.g. vision_encoder_int8.tflite
VADP = os.environ.get("VADP", "")

IMG = "<fake_token_around_image><global-img><image_soft_token><fake_token_around_image>"
JINJA = (
    "{%- for message in messages -%}"
    "<|start_of_role|>{{ 'assistant' if message.role == 'model' else message.role }}<|end_of_role|>"
    "{%- if message.content is string -%}"
    "{{ message.content }}"
    "{%- else -%}"
    "{%- for item in message.content -%}"
    "{%- if item.type == 'text' -%}{{ item.text }}"
    "{%- elif item.type == 'image' -%}" + IMG +
    "{%- endif -%}{%- endfor -%}"
    # The \n after <|end_of_text|> is part of granite's trained format and DocTags mode
    # collapses without it ("Powered by TCPDF" blank-page output). `{%- endfor` (leading
    # trim) would eat a literal \n, so keep endfor without the leading dash.
    "{%- endif -%}<|end_of_text|>\n"
    "{% endfor -%}"
    "{%- if add_generation_prompt -%}<|start_of_role|>assistant<|end_of_role|>{%- endif -%}"
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
  if VENC:
    vision_encoder = os.path.join(VIS, VENC)
  else:
    ve8 = os.path.join(VIS, "vision_encoder_int8.tflite")
    vision_encoder = ve8 if os.path.exists(ve8) else os.path.join(VIS, "vision_encoder.tflite")
  if VADP:
    vision_adapter = os.path.join(VIS, VADP)
  else:
    va8 = os.path.join(VIS, "vision_adapter_int8.tflite")
    vision_adapter = va8 if os.path.exists(va8) else os.path.join(VIS, "vision_adapter.tflite")
  print("sections:", embedder, prefill_decode, vision_encoder, vision_adapter, "| tok:", HF_TOK)

  md = llm_metadata_pb2.LlmMetadata()
  md.max_num_tokens = MAX_TOKENS
  # NO md.start_token (see module docstring).
  md.prompt_templates.user.prefix = "<|start_of_role|>user<|end_of_role|>"
  md.prompt_templates.user.suffix = "<|end_of_text|>\n"
  md.prompt_templates.model.prefix = "<|start_of_role|>assistant<|end_of_role|>"
  md.prompt_templates.model.suffix = "<|end_of_text|>\n"
  md.prompt_templates.system.prefix = "<|start_of_role|>system<|end_of_role|>"
  md.prompt_templates.system.suffix = "<|end_of_text|>\n"
  md.jinja_prompt_template = JINJA
  md.llm_model_type.CopyFrom(
      llm_model_type_pb2.LlmModelType(fast_vlm=llm_model_type_pb2.FastVlm()))
  md.llm_model_type.fast_vlm.image_tensor_height = IMAGE_SIZE
  md.llm_model_type.fast_vlm.image_tensor_width = IMAGE_SIZE
  md.stop_tokens.add().token_str = "<|end_of_text|>"
  md_path = os.path.join(OUT, "docling_llm_metadata.pb")
  with open(md_path, "wb") as f:
    f.write(md.SerializeToString())

  b = litertlm_builder.LitertLmFileBuilder()
  b.add_system_metadata(litertlm_builder.Metadata(
      key="Authors", value="", dtype=litertlm_builder.DType.STRING))
  b.add_llm_metadata(md_path)
  b.add_hf_tokenizer(HF_TOK)
  b.add_tflite_model(embedder, litertlm_builder.TfLiteModelType.EMBEDDER)
  # DEC_ACT: precautionary, NOT a fix for anything observed on CPU (the CPU image-turn
  # collapse was the template newline, see JINJA above). Declared because the vision
  # soft tokens run absmax ~165 and Mali-class fp16 accumulation overflows at image
  # positions on exactly this shape of model (north-micro-vision precedent
  # — see cards/north-micro-vision-instruct-litert.md); fp32_fp16 costs nothing where fp16 is safe.
  dec_act = os.environ.get("DEC_ACT", "fp32_fp16")
  b.add_tflite_model(prefill_decode, litertlm_builder.TfLiteModelType.PREFILL_DECODE,
                     prefer_activation_type=dec_act or None)
  b.add_tflite_model(vision_encoder, litertlm_builder.TfLiteModelType.VISION_ENCODER)
  b.add_tflite_model(vision_adapter, litertlm_builder.TfLiteModelType.VISION_ADAPTER)
  out_path = os.path.join(OUT, os.environ.get("OUT_NAME", "granite-docling-258M.litertlm"))
  with open(out_path, "wb") as f:
    b.build(f)
  print("BUNDLE_DONE", out_path, round(os.path.getsize(out_path) / 1e6, 1), "MB")


if __name__ == "__main__":
  main()
