"""Assemble the PaddleOCR-VL-1.6 fast_vlm `.litertlm` bundle.

PaddleOCR-VL chat format (chat_template.jinja):
  <|begin_of_sentence|>User: <|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>PROMPT\n
  Assistant:\n REPLY</s>
The task is selected by the text prompt: "OCR:" | "Table Recognition:" |
"Formula Recognition:" | "Chart Recognition:" | "Text Spotting:".

The bundle tokenizer is the repo SP model with the 1019 added tokens
(<|IMAGE_START|>, <|LOC_0|>..<|LOC_1000|>, <fcel> etc., ids 100295..101313 —
NOT SP pieces upstream) appended as USER_DEFINED pieces at their exact ids and
padded to vocab_size, so table/spotting outputs detokenize correctly on device
(same fix as Nanbeige4.1-3B).

    DEC=out/paddleocr-decoder VIS=out/paddleocr-vision IMAGE_SIZE=560 \
      .venv/bin/python paddleocr_work/build_paddleocr_bundle.py
"""
import json
import os

import litert_lm_builder as litertlm_builder
from litert_lm_builder.runtime.proto import llm_metadata_pb2
from litert_lm_builder.runtime.proto import llm_model_type_pb2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL = os.path.join(ROOT, "src_models/paddleocr-vl-1.6")
DEC = os.environ.get("DEC", os.path.join(ROOT, "out/paddleocr-decoder"))
VIS = os.environ.get("VIS", os.path.join(ROOT, "out/paddleocr-vision"))
OUT = os.environ.get("OUT", os.path.join(ROOT, "out/paddleocr-bundle"))
os.makedirs(OUT, exist_ok=True)
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "560"))
MAX_TOKENS = int(os.environ.get("CACHE", "4096"))
VOCAB = 103424

IMG = "<|IMAGE_START|><image_soft_token><|IMAGE_END|>"
JINJA = (
    "{%- for message in messages -%}"
    "{%- if message.role == 'user' -%}"
    "User: "
    "{%- if message.content is string -%}{{ message.content }}"
    "{%- else -%}"
    "{%- for item in message.content -%}"
    "{%- if item.type == 'image' -%}" + IMG + "{%- endif -%}"
    "{%- endfor -%}"
    "{%- for item in message.content -%}"
    "{%- if item.type == 'text' -%}{{ item.text }}{%- endif -%}"
    "{%- endfor -%}"
    "{%- endif -%}{{ '\\n' }}"
    "{%- elif message.role == 'assistant' -%}"
    "Assistant:\n"
    "{%- if message.content is string -%}{{ message.content }}"
    "{%- else -%}"
    "{%- for item in message.content -%}"
    "{%- if item.type == 'text' -%}{{ item.text }}{%- endif -%}"
    "{%- endfor -%}"
    "{%- endif -%}</s>"
    "{%- elif message.role == 'system' -%}"
    "{%- if message.content is string -%}{{ message.content }}{{ '\\n' }}{%- endif -%}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}Assistant:\n{%- endif -%}"
)


def fixed_spm(dst):
  """repo tokenizer.model + added tokens appended at exact ids, padded to VOCAB."""
  from sentencepiece import sentencepiece_model_pb2 as spb
  mp = spb.ModelProto()
  with open(os.path.join(MODEL, "tokenizer.model"), "rb") as f:
    mp.ParseFromString(f.read())
  base_n = len(mp.pieces)
  by_id = {int(v): k for k, v in
           json.load(open(os.path.join(MODEL, "added_tokens.json"))).items()}
  assert min(by_id) >= base_n, (min(by_id), base_n)
  while len(mp.pieces) < VOCAB:
    idx = len(mp.pieces)
    p = mp.pieces.add()
    if idx in by_id:
      p.piece = by_id[idx]
      p.score = 0.0
      p.type = spb.ModelProto.SentencePiece.USER_DEFINED
    else:
      p.piece = f"<unused_{idx}>"
      p.score = 0.0
      p.type = spb.ModelProto.SentencePiece.UNUSED
  with open(dst, "wb") as f:
    f.write(mp.SerializeToString())
  print(f"FIX_ADDED_TOKENS: SP {base_n} -> {len(mp.pieces)} pieces "
        f"({len(by_id)} added tokens at exact ids)")
  return dst


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
  sp_path = fixed_spm(os.path.join(OUT, "tokenizer.spiece"))
  print("sections:", embedder, prefill_decode, vision_encoder, vision_adapter)

  md = llm_metadata_pb2.LlmMetadata()
  md.max_num_tokens = MAX_TOKENS
  md.start_token.token_str = "<|begin_of_sentence|>"
  md.prompt_templates.user.prefix = "User: "
  md.prompt_templates.user.suffix = "\n"
  md.prompt_templates.model.prefix = "Assistant:\n"
  md.prompt_templates.model.suffix = "</s>"
  md.prompt_templates.system.prefix = ""
  md.prompt_templates.system.suffix = "\n"
  md.jinja_prompt_template = JINJA
  md.llm_model_type.CopyFrom(
      llm_model_type_pb2.LlmModelType(fast_vlm=llm_model_type_pb2.FastVlm()))
  md.llm_model_type.fast_vlm.image_tensor_height = IMAGE_SIZE
  md.llm_model_type.fast_vlm.image_tensor_width = IMAGE_SIZE
  md.stop_tokens.add().token_str = "</s>"
  md_path = os.path.join(OUT, "paddleocr_llm_metadata.pb")
  with open(md_path, "wb") as f:
    f.write(md.SerializeToString())

  b = litertlm_builder.LitertLmFileBuilder()
  b.add_system_metadata(litertlm_builder.Metadata(
      key="Authors", value="", dtype=litertlm_builder.DType.STRING))
  b.add_llm_metadata(md_path)
  b.add_sentencepiece_tokenizer(sp_path)
  print("added SP tokenizer:", sp_path)
  b.add_tflite_model(embedder, litertlm_builder.TfLiteModelType.EMBEDDER)
  b.add_tflite_model(prefill_decode, litertlm_builder.TfLiteModelType.PREFILL_DECODE)
  b.add_tflite_model(vision_encoder, litertlm_builder.TfLiteModelType.VISION_ENCODER)
  b.add_tflite_model(vision_adapter, litertlm_builder.TfLiteModelType.VISION_ADAPTER)
  out_path = os.path.join(OUT, os.environ.get("OUT_NAME", "PaddleOCR-VL-1.6.litertlm"))
  with open(out_path, "wb") as f:
    b.build(f)
  print("BUNDLE_DONE", out_path, round(os.path.getsize(out_path) / 1e6, 1), "MB")


if __name__ == "__main__":
  main()
