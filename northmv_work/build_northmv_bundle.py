"""Hand-assemble the North-Micro-Vision-Instruct fast_vlm `.litertlm` bundle
(scripts/build_internvl_bundle.py with the Cohere chat template instead of ChatML):

  EMBEDDER       : token_ids[1,1] -> [1,1,2048]
  PREFILL_DECODE : embeddings + KV -> logits + KV
  VISION_ENCODER : image NHWC[1,512,512,3] in [0,1] -> [1,1024,C]
  VISION_ADAPTER : [1,1024,C] -> [1,256,2048]
  + LlmMetadata  : FastVlm(image 512x512), start token <BOS_TOKEN>(2), Cohere turn tokens,
                   image rendered <|VISION_START|><image_soft_token><|VISION_END|>
                   (HF: <|VISION_START|><|IMAGE_PAD|><|VISION_END|>; the runtime splits the
                   rendered prompt on the literal <image_soft_token> and drops the 256 soft
                   tokens there).

    DEC=out/northmv-decoder VIS=out/northmv-vision TOK=sp CACHE=4096 \
      .venv-vl0930-t515/bin/python northmv_work/build_northmv_bundle.py
"""
import os

import litert_lm_builder as litertlm_builder
from litert_lm_builder.runtime.proto import llm_metadata_pb2
from litert_lm_builder.runtime.proto import llm_model_type_pb2

DEC = os.environ.get("DEC", "out/northmv-decoder")
VIS = os.environ.get("VIS", "out/northmv-vision")
OUT = os.environ.get("OUT_DIR", "out/northmv-bundle")
os.makedirs(OUT, exist_ok=True)

IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "512"))
MAX_TOKENS = int(os.environ.get("CACHE", "4096"))
IMG_RENDER = os.environ.get("IMG_RENDER", "<|VISION_START|><image_soft_token><|VISION_END|>")
BOS_ID = 2  # <BOS_TOKEN>

SOT, EOT = "<|START_OF_TURN_TOKEN|>", "<|END_OF_TURN_TOKEN|>"
USER, MODEL, SYSTEM = "<|USER_TOKEN|>", "<|CHATBOT_TOKEN|>", "<|SYSTEM_TOKEN|>"

# Same shape as the FastVLM/InternVL jinja (string content and list content both handled;
# roles user/model/system as the runtime names them). BOS comes from start_token.
JINJA = (
    "{%- for message in messages -%}"
    "{%- if message.content is string -%}"
    "{%- if message.role == 'user' %}" + SOT + USER + "{{ message.content }}" + EOT + "{% endif -%}"
    "{%- if message.role == 'model' %}" + SOT + MODEL + "{{ message.content }}" + EOT + "{% endif -%}"
    "{%- if message.role == 'system' %}" + SOT + SYSTEM + "{{ message.content }}" + EOT + "{% endif -%}"
    "{%- else -%}"
    "{%- if message.role == 'user' %}" + SOT + USER +
    "{% elif message.role == 'model' %}" + SOT + MODEL +
    "{% elif message.role == 'system' %}" + SOT + SYSTEM + "{% endif -%}"
    "{%- for item in message.content %}"
    "{%- if item.type == 'text' %}{{ item.text }}"
    "{%- elif item.type == 'image' -%}{{ '" + IMG_RENDER + "' }}"
    "{%- endif -%}{%- endfor -%}"
    "{%- if message.role == 'user' %}" + EOT +
    "{% elif message.role == 'model' %}" + EOT +
    "{% elif message.role == 'system' %}" + EOT + "{% endif -%}"
    "{%- endif -%}{%- endfor -%}"
    "{%- if add_generation_prompt %}" + SOT + MODEL + "{% endif -%}"
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
  # VENC / VADP name the vision variants (e.g. vision_encoder_fp16.tflite); default int8.
  vision_encoder = os.path.join(VIS, os.environ.get("VENC", "vision_encoder_int8.tflite"))
  vision_adapter = os.path.join(VIS, os.environ.get("VADP", "vision_adapter_int8.tflite"))
  for p in (vision_encoder, vision_adapter):
    assert os.path.exists(p), p
  tok_mode = os.environ.get("TOK", "auto")  # auto | hf | sp
  sp = [os.path.join(DEC, f) for f in os.listdir(DEC) if f.endswith((".spiece", ".model", ".spm"))]
  if tok_mode == "hf":
    sp = []
  hf = os.environ.get("HF_TOK", "src_models/north-micro-vision-llm/tokenizer.json")
  print("embedder:", embedder)
  print("prefill_decode:", prefill_decode)
  print("vision_encoder:", vision_encoder, "vision_adapter:", vision_adapter)
  print("sp tokenizer:", sp, "| hf fallback:", hf)

  md = llm_metadata_pb2.LlmMetadata()
  md.max_num_tokens = MAX_TOKENS
  md.start_token.token_ids.ids.append(BOS_ID)
  md.prompt_templates.user.prefix = SOT + USER
  md.prompt_templates.user.suffix = EOT
  md.prompt_templates.model.prefix = SOT + MODEL
  md.prompt_templates.model.suffix = EOT
  md.prompt_templates.system.prefix = SOT + SYSTEM
  md.prompt_templates.system.suffix = EOT
  md.jinja_prompt_template = JINJA
  md.llm_model_type.CopyFrom(
      llm_model_type_pb2.LlmModelType(fast_vlm=llm_model_type_pb2.FastVlm()))
  md.llm_model_type.fast_vlm.image_tensor_height = IMAGE_SIZE
  md.llm_model_type.fast_vlm.image_tensor_width = IMAGE_SIZE
  md.stop_tokens.add().token_str = EOT
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
  # DEC_ACT: prefer_activation_type declared on the decoder section. The Cohere decoder
  # develops register-scale activations at the 256 image-token positions; Mali-CL fp16
  # (fp16 accumulation, unlike Apple Metal) overflows there and the model answers as if
  # blind (echoes the question). fp32_fp16 = mixed precision fixes it while keeping most
  # of the fp16 speed; measured on Pixel 8a (northmv_work/FINDINGS.md).
  dec_act = os.environ.get("DEC_ACT", "fp32_fp16")
  b.add_tflite_model(prefill_decode, litertlm_builder.TfLiteModelType.PREFILL_DECODE,
                     prefer_activation_type=dec_act or None)
  b.add_tflite_model(vision_encoder, litertlm_builder.TfLiteModelType.VISION_ENCODER)
  b.add_tflite_model(vision_adapter, litertlm_builder.TfLiteModelType.VISION_ADAPTER)

  out_path = os.path.join(OUT, os.environ.get("OUT_NAME", "North-Micro-Vision-Instruct.litertlm"))
  with open(out_path, "wb") as f:
    b.build(f)
  print("BUNDLE_DONE", out_path, round(os.path.getsize(out_path) / 1e6, 1), "MB")


if __name__ == "__main__":
  main()
