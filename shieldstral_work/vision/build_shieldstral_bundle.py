"""Hand-assemble the multimodal Shieldstral `.litertlm` (fast_vlm layout).

  EMBEDDER       : token_ids [1,1]                 -> [1,1,3072]
  PREFILL_DECODE : embeddings + KV                 -> logits + KV
  VISION_ENCODER : NHWC [1,560,560,3] in [0,1]     -> [1,1600,1024]
  VISION_ADAPTER : [1,1600,1024]                   -> [1,420,3072]

The adapter emits 420 rows, not 400: pixtral's token expansion interleaves
`[IMG_BREAK]` at the end of every patch row and one `[IMG_END]` at the end, and
those markers keep their ordinary text embeddings. They are constants, so they are
folded into the adapter and the runtime still injects ONE contiguous block at the
soft-token position (verified corr 1.0 against transformers' own inputs_embeds).

Template: as in the text lane, the fixed system prompt is baked into the user
prefix. The caller sends the `<Instruct>`/`<Query>`/`<Document>` body and places the
image where the document goes — i.e. LAST. That position matters: with the image
ahead of the instructions, benign images scored within ±1 of the decision boundary
(one flipped to "unsafe"); in the document slot the same images score -5.4 to -6.7.

    DEC=... VIS=... python build_shieldstral_bundle.py
"""

import os

import litert_lm_builder as litertlm_builder
from litert_lm_builder.runtime.proto import llm_metadata_pb2
from litert_lm_builder.runtime.proto import llm_model_type_pb2

DEC = os.environ.get("DEC", "shieldstral_work/vision/out_decoder")
VIS = os.environ.get("VIS", "shieldstral_work/vision/out_vision")
SRC = os.environ.get("SRC", "src_models/shieldstral-3b-text")
OUT = os.environ.get("OUT", "shieldstral_work/vision/out_bundle")
os.makedirs(OUT, exist_ok=True)

IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "560"))
MAX_TOKENS = int(os.environ.get("CACHE", "4096"))

SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query "
    'and the Instruction provided. Note that the answer can only be "yes" or "no".')
USER_PREFIX = f"[SYSTEM_PROMPT]{SYSTEM_PROMPT}[/SYSTEM_PROMPT][INST]"
USER_SUFFIX = "[/INST]"

# The runtime splits the rendered prompt on the literal <image_soft_token>.
JINJA = (
    "{%- for message in messages -%}"
    "{%- if message.role == 'user' -%}"
    "{{ '" + USER_PREFIX + "' }}"
    "{%- if message.content is string -%}{{ message.content }}"
    "{%- else -%}"
    "{%- for item in message.content -%}"
    "{%- if item.type == 'text' -%}{{ item.text }}"
    "{%- elif item.type == 'image' -%}{{ '<image_soft_token>' }}"
    "{%- endif -%}{%- endfor -%}"
    "{%- endif -%}"
    "{{ '" + USER_SUFFIX + "' }}"
    "{%- elif message.role == 'model' -%}{{ message.content }}{{ '</s>' }}"
    "{%- endif -%}{%- endfor -%}"
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
  for label, p in (("embedder", embedder), ("prefill_decode", prefill_decode),
                   ("vision_encoder", vision_encoder), ("vision_adapter", vision_adapter)):
    print(f"{label:16} {p}  {os.path.getsize(p)/1e6:.0f} MB")

  md = llm_metadata_pb2.LlmMetadata()
  md.max_num_tokens = MAX_TOKENS
  md.start_token.token_str = "<s>"
  md.prompt_templates.user.prefix = USER_PREFIX
  md.prompt_templates.user.suffix = USER_SUFFIX
  md.prompt_templates.model.suffix = "</s>"
  md.jinja_prompt_template = JINJA
  md.llm_model_type.CopyFrom(
      llm_model_type_pb2.LlmModelType(fast_vlm=llm_model_type_pb2.FastVlm()))
  md.llm_model_type.fast_vlm.image_tensor_height = IMAGE_SIZE
  md.llm_model_type.fast_vlm.image_tensor_width = IMAGE_SIZE
  md.stop_tokens.add().token_ids.ids.append(2)     # </s>, re-derived from THIS tokenizer
  md.stop_tokens.add().token_str = "</s>"
  md_path = os.path.join(OUT, "llm_metadata.pb")
  with open(md_path, "wb") as f:
    f.write(md.SerializeToString())

  b = litertlm_builder.LitertLmFileBuilder()
  b.add_system_metadata(litertlm_builder.Metadata(
      key="Authors", value="", dtype=litertlm_builder.DType.STRING))
  b.add_llm_metadata(md_path)
  sp = [os.path.join(DEC, f) for f in os.listdir(DEC)
        if f.endswith((".spiece", ".model", ".spm"))]
  if sp:
    b.add_sentencepiece_tokenizer(sp[0])
    print("tokenizer: SP", sp[0])
  else:
    hf = os.path.join(SRC, "tokenizer.json")
    b.add_hf_tokenizer(hf)
    print("tokenizer: HF", hf)
  b.add_tflite_model(embedder, litertlm_builder.TfLiteModelType.EMBEDDER)
  b.add_tflite_model(prefill_decode, litertlm_builder.TfLiteModelType.PREFILL_DECODE)
  b.add_tflite_model(vision_encoder, litertlm_builder.TfLiteModelType.VISION_ENCODER)
  b.add_tflite_model(vision_adapter, litertlm_builder.TfLiteModelType.VISION_ADAPTER)

  out_path = os.path.join(OUT, os.environ.get("OUT_NAME", "Shieldstral-1.0-3B-vision_int4.litertlm"))
  with open(out_path, "wb") as f:
    b.build(f)
  print("BUNDLE_DONE", out_path, round(os.path.getsize(out_path) / 1e6, 1), "MB")


if __name__ == "__main__":
  main()
