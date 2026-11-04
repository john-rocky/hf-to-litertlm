"""Hand-assemble the Mage-VL-Cowork fast_vlm `.litertlm` bundle (a non-square canvas, 1920 x 1088) from the converted
sections.

Copied from scripts/build_internvl_bundle.py (the Mage-VL ship's builder) and changed for this model:
  * JINJA_FILE: the chat template is read from a file (make_cowork_template.py's output = the vendor template + the
    listed edits) instead of the script's built-in ChatML string;
  * MODEL_PREFIX: the structured model prefix (default `<|im_start|>assistant\n<think>\n` = the vendor template's
    generation prompt with thinking on, so a reply starts inside the think block as the vendor's renders do);
  * CHANNEL=thought: declare the reasoning channel (name `thought`, start `<think>`, end `</think>`,
    is_reasoning_channel); unset = no channel (the shipped file declares none);
  * DEC_ACT: prefer_activation_type on the PREFILL_DECODE section only (unset = none; the ship declares fp32 on every
    TFLite section with scripts/set_activation_type.py after the build);
  * IMAGE_H / IMAGE_W: FastVlm.image_tensor_height / image_tensor_width written separately; IMAGE_SIZE still sets
    both when IMAGE_H / IMAGE_W are unset;
  * VENC / VADP: the exact vision files (default VIS/vision_encoder.tflite and VIS/vision_adapter.tflite = fp32), never
    picked by presence;
  * TOK_JSON is required: the repository's own tokenizer.json (the decoder export writes a tokenizer.json of its own
    into DEC that is not the repository's file byte for byte; the bundle carries the repository's);
  * OUT_DIR; the sha256 of every section input is printed.
Layout:

  EMBEDDER       : token_ids[1,1] -> [1,1,2560]
  PREFILL_DECODE : embeddings + KV -> logits + KV
  VISION_ENCODER : image NHWC[1,1088,1920,3] in [0,1] -> [1,8160,1024]
  VISION_ADAPTER : [1,8160,1024] -> [1,2040,2560]
  + LlmMetadata  : FastVlm(image 1088 x 1920) + the jinja template above + structured ChatML templates

    DEC=<decoder dir> VIS=<vision dir> TOK=hf TOK_JSON=<checkpoint>/tokenizer.json IMAGE_H=1088 IMAGE_W=1920 \\
      CACHE=4096 STOP_ID=151645 JINJA_FILE=cowork_runtime.jinja [CHANNEL=thought] OUT_DIR=<dir> OUT_NAME=x.litertlm \\
      python magevlcowork_work/build_cowork_bundle.py
"""
import hashlib
import os, sys

import litert_lm_builder as litertlm_builder
from litert_lm_builder.runtime.proto import llm_metadata_pb2
from litert_lm_builder.runtime.proto import llm_model_type_pb2

DEC = os.environ.get("DEC", "out/magevlcowork/work/decoder")
VIS = os.environ.get("VIS", "out/magevlcowork/work/vision")
OUT = os.environ.get("OUT_DIR", "out/magevlcowork/work/bundle")
os.makedirs(OUT, exist_ok=True)

IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "448"))
IMAGE_H = int(os.environ.get("IMAGE_H", IMAGE_SIZE))
IMAGE_W = int(os.environ.get("IMAGE_W", IMAGE_SIZE))
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


if os.environ.get("JINJA_FILE"):
  JINJA = open(os.environ["JINJA_FILE"], encoding="utf-8").read()
MODEL_PREFIX = os.environ.get("MODEL_PREFIX", "<|im_start|>assistant\n<think>\n")


def sha256(path):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 24), b""):
      h.update(chunk)
  return h.hexdigest()


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
  # the vision pair is named, never picked by presence
  vision_encoder = os.environ.get("VENC") or os.path.join(VIS, "vision_encoder.tflite")
  vision_adapter = os.environ.get("VADP") or os.path.join(VIS, "vision_adapter.tflite")
  for _p in (vision_encoder, vision_adapter):
    assert os.path.exists(_p), _p
  # tokenizer: prefer a sentencepiece artifact from the decoder export, else HF json.
  tok_mode = os.environ.get("TOK", "auto")  # auto | hf | sp
  sp = [os.path.join(DEC, f) for f in os.listdir(DEC) if f.endswith((".spiece", ".model", ".spm"))]
  if tok_mode == "hf":
    sp = []
  hf = os.environ.get("TOK_JSON")
  if not hf:
    sys.exit("set TOK_JSON to the repository's own tokenizer.json (not the one the decoder export writes into DEC)")
  print("embedder:", embedder)
  print("prefill_decode:", prefill_decode)
  print("vision_encoder:", vision_encoder, "vision_adapter:", vision_adapter)
  print("sp tokenizer:", sp, "| hf fallback:", hf)

  md = llm_metadata_pb2.LlmMetadata()
  md.max_num_tokens = MAX_TOKENS
  # STRUCTURED prompt_templates (ChatML) — required by the conversation builder
  # alongside jinja (FastVLM bundle has both; jinja-only => conversation_create fails).
  # No start_token by default. ChatML families have no BOS, and the runtime
  # prepends whatever string is set here as a real token: the literal "None"
  # this line used to write went out on ten published fast_vlm bundles as the
  # word `None` at the head of every prompt (BOS audit, 2026-08-27). Opt in
  # with START_TOKEN=<literal> only for a family whose template expects one.
  start_token = os.environ.get("START_TOKEN")
  if start_token:
    md.start_token.token_str = start_token
  md.prompt_templates.user.prefix = "<|im_start|>user\n"
  md.prompt_templates.user.suffix = "<|im_end|>\n"
  md.prompt_templates.model.prefix = MODEL_PREFIX
  md.prompt_templates.model.suffix = "<|im_end|>\n"
  md.prompt_templates.system.prefix = "<|im_start|>system\n"
  md.prompt_templates.system.suffix = "<|im_end|>\n"
  md.jinja_prompt_template = JINJA
  md.llm_model_type.CopyFrom(
      llm_model_type_pb2.LlmModelType(fast_vlm=llm_model_type_pb2.FastVlm()))
  md.llm_model_type.fast_vlm.image_tensor_height = IMAGE_H
  md.llm_model_type.fast_vlm.image_tensor_width = IMAGE_W
  # Stop token: STOP_ID env declares it by token id, matching the LLM-bundle
  # convention (token_str stops are string-matched by the runtime and have
  # measured sharp edges — the s1-mini punctuation-prefix lesson). Whether the
  # bundle encodes its ChatML specials correctly is a property of the tokenizer
  # section, not of this declaration: gate the built bundle with
  # scripts/gate_specials.py (REPRODUCE, NuExtract 2026-08-25 -> 2026-09-04).
  stop_id = os.environ.get("STOP_ID")
  if stop_id:
    md.stop_tokens.add().token_ids.ids.append(int(stop_id))
  else:
    md.stop_tokens.add().token_str = "<|im_end|>"
  if os.environ.get("CHANNEL") == "thought":
    md.channels.add(channel_name="thought", start="<think>", end="</think>", is_reasoning_channel=True)
  elif os.environ.get("CHANNEL"):
    raise SystemExit(f"CHANNEL={os.environ['CHANNEL']!r}: only 'thought' is defined")
  md_path = os.path.join(OUT, os.environ.get("OUT_NAME", "bundle") + ".llm_metadata.pb")
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
  b.add_tflite_model(prefill_decode, litertlm_builder.TfLiteModelType.PREFILL_DECODE,
                     prefer_activation_type=os.environ.get("DEC_ACT") or None)
  b.add_tflite_model(vision_encoder, litertlm_builder.TfLiteModelType.VISION_ENCODER)
  b.add_tflite_model(vision_adapter, litertlm_builder.TfLiteModelType.VISION_ADAPTER)

  out_path = os.path.join(OUT, os.environ.get("OUT_NAME", "Mage-VL-Cowork.litertlm"))
  with open(out_path, "wb") as f:
    b.build(f)
  for name, path in (("tokenizer_json", hf), ("embedder", embedder), ("prefill_decode", prefill_decode),
                     ("vision_encoder", vision_encoder), ("vision_adapter", vision_adapter), ("llm_metadata", md_path)):
    print(f"SECTION_INPUT {name} {os.path.getsize(path)} {sha256(path)} {path}")
  print("BUNDLE_DONE", out_path, os.path.getsize(out_path), sha256(out_path))


if __name__ == "__main__":
  main()
