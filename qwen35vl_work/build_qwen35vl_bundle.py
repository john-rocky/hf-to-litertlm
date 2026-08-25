"""Hand-assemble the Qwen3.5 fast_vlm `.litertlm` bundle (build_northmv_bundle.py
with ChatML + the Qwen3.5 think-scaffold template and NO start token):

  EMBEDDER       : token_ids[1,1] -> [1,1,2048]
  PREFILL_DECODE : embeddings + hybrid states -> logits + states
  VISION_ENCODER : image NHWC[1,512,512,3] in [0,1] -> [1,1024,1024]
  VISION_ADAPTER : [1,1024,1024] -> [1,256,2048]
  + LlmMetadata  : FastVlm(image 512x512), NO start token (Qwen3.5 no-BOS rule),
                   ChatML turns with the empty <think> block kept in BOTH the
                   generation prompt and history assistant renders (the engine's
                   incremental-render prefix contract, qwen35_work RESULTS),
                   image rendered <|vision_start|><image_soft_token><|vision_end|>
                   (HF: <|vision_start|><|image_pad|><|vision_end|>; the runtime
                   splits on the literal <image_soft_token> and drops the 256
                   soft tokens there), stop tokens <|endoftext|> AND <|im_end|>.

The decoder section declares prefer_activation_type (DEC_ACT, default fp32):
the gated-delta graphs need fp32 activations on GPU (the text 2B GPU reship
shipped as _int8_fp32act for the same reason).

After building, append ExecutorMetadata (scripts/add_executor_metadata.py — it
auto-selects the tflite with kv_cache_* inputs).

    DEC=out/qwen35vl-decoder VIS=out/qwen35vl-vision TOK=<tokenizer.json> \
      .venv-092/bin/python qwen35vl_work/build_qwen35vl_bundle.py
"""
import json
import os

import litert_lm_builder as litertlm_builder
from litert_lm_builder.runtime.proto import llm_metadata_pb2
from litert_lm_builder.runtime.proto import llm_model_type_pb2

DEC = os.environ.get("DEC", "out/qwen35vl-decoder")
VIS = os.environ.get("VIS", "out/qwen35vl-vision")
OUT = os.environ.get("OUT_DIR", "out/qwen35vl-bundle")
os.makedirs(OUT, exist_ok=True)

IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "512"))
MAX_TOKENS = int(os.environ.get("CACHE", "4096"))
IMG_RENDER = os.environ.get(
    "IMG_RENDER", "<|vision_start|><image_soft_token><|vision_end|>")

THINK = "<think>\n\n</think>\n\n"
U_PRE, U_SUF = "<|im_start|>user\n", "<|im_end|>\n"
M_PRE, M_SUF = "<|im_start|>assistant\n" + THINK, "<|im_end|>\n"
S_PRE, S_SUF = "<|im_start|>system\n", "<|im_end|>\n"

# Same shape as the northmv/FastVLM jinja (string and list content both handled;
# roles user/model/system as the runtime names them). No BOS anywhere.
def _turn(pre, suf, body):
  return pre + body + suf


JINJA = (
    "{%- for message in messages -%}"
    "{%- if message.content is string -%}"
    "{%- if message.role == 'user' %}" + U_PRE + "{{ message.content }}" + U_SUF + "{% endif -%}"
    "{%- if message.role == 'model' or message.role == 'assistant' %}" + M_PRE + "{{ message.content }}" + M_SUF + "{% endif -%}"
    "{%- if message.role == 'system' %}" + S_PRE + "{{ message.content }}" + S_SUF + "{% endif -%}"
    "{%- else -%}"
    "{%- if message.role == 'user' %}" + U_PRE +
    "{% elif message.role == 'model' or message.role == 'assistant' %}" + M_PRE +
    "{% elif message.role == 'system' %}" + S_PRE + "{% endif -%}"
    "{%- for item in message.content %}"
    "{%- if item.type == 'text' %}{{ item.text }}"
    "{%- elif item.type == 'image' -%}{{ '" + IMG_RENDER + "' }}"
    "{%- endif -%}{%- endfor -%}"
    "{%- if message.role == 'user' %}" + U_SUF +
    "{% elif message.role == 'model' or message.role == 'assistant' %}" + M_SUF +
    "{% elif message.role == 'system' %}" + S_SUF + "{% endif -%}"
    "{%- endif -%}{%- endfor -%}"
    "{%- if add_generation_prompt %}" + M_PRE + "{% endif -%}"
)


def find_tflite(d, *keywords):
  cands = [f for f in os.listdir(d) if f.endswith(".tflite")]
  for kw in keywords:
    for f in cands:
      if kw in f.lower():
        return os.path.join(d, f)
  raise FileNotFoundError(f"no tflite matching {keywords} in {d}: {cands}")


def main():
  embedder = find_tflite(DEC, os.environ.get("EMB", "embedder_wi8"))
  prefill_decode = find_tflite(DEC, os.environ.get("DECODE", "prefill_decode_wi8"))
  vision_encoder = os.path.join(VIS, os.environ.get("VENC", "vision_encoder_int8.tflite"))
  vision_adapter = os.path.join(VIS, os.environ.get("VADP", "vision_adapter_int8.tflite"))
  for p in (vision_encoder, vision_adapter):
    assert os.path.exists(p), p
  hf_tok = os.environ.get("TOK")
  assert hf_tok and os.path.exists(hf_tok), f"TOK={hf_tok} (HF tokenizer.json required)"

  # Verify the stop-token ids against the actual tokenizer before baking them.
  from tokenizers import Tokenizer
  tk = Tokenizer.from_file(hf_tok)
  eot_id = tk.token_to_id("<|endoftext|>")
  imend_id = tk.token_to_id("<|im_end|>")
  assert eot_id is not None and imend_id is not None, (eot_id, imend_id)
  for tok in ("<|vision_start|>", "<|vision_end|>"):
    assert tk.token_to_id(tok) is not None, f"missing {tok} in tokenizer"
  print("stop ids:", {"<|endoftext|>": eot_id, "<|im_end|>": imend_id})

  print("embedder:", embedder)
  print("prefill_decode:", prefill_decode)
  print("vision_encoder:", vision_encoder, "vision_adapter:", vision_adapter)

  md = llm_metadata_pb2.LlmMetadata()
  md.max_num_tokens = MAX_TOKENS
  # NO start_token: Qwen3.5 templates begin directly with <|im_start|>.
  md.prompt_templates.user.prefix = U_PRE
  md.prompt_templates.user.suffix = U_SUF
  md.prompt_templates.model.prefix = M_PRE
  md.prompt_templates.model.suffix = M_SUF
  md.prompt_templates.system.prefix = S_PRE
  md.prompt_templates.system.suffix = S_SUF
  md.jinja_prompt_template = JINJA
  md.llm_model_type.CopyFrom(
      llm_model_type_pb2.LlmModelType(fast_vlm=llm_model_type_pb2.FastVlm()))
  md.llm_model_type.fast_vlm.image_tensor_height = IMAGE_SIZE
  md.llm_model_type.fast_vlm.image_tensor_width = IMAGE_SIZE
  md.stop_tokens.add().token_ids.ids.append(eot_id)
  md.stop_tokens.add().token_ids.ids.append(imend_id)
  md_path = os.path.join(OUT, "llm_metadata.pb")
  with open(md_path, "wb") as f:
    f.write(md.SerializeToString())

  b = litertlm_builder.LitertLmFileBuilder()
  b.add_system_metadata(litertlm_builder.Metadata(
      key="Authors", value="", dtype=litertlm_builder.DType.STRING))
  b.add_llm_metadata(md_path)
  b.add_hf_tokenizer(hf_tok)
  b.add_tflite_model(embedder, litertlm_builder.TfLiteModelType.EMBEDDER)
  # DEC_ACT: the gated-delta decoder needs fp32 activations on GPU (the 8/12
  # text GPU reship shipped _int8_fp32act; fp16 paths mis-wire / overflow).
  dec_act = os.environ.get("DEC_ACT", "fp32")
  b.add_tflite_model(prefill_decode, litertlm_builder.TfLiteModelType.PREFILL_DECODE,
                     prefer_activation_type=dec_act or None)
  b.add_tflite_model(vision_encoder, litertlm_builder.TfLiteModelType.VISION_ENCODER)
  b.add_tflite_model(vision_adapter, litertlm_builder.TfLiteModelType.VISION_ADAPTER)

  out_path = os.path.join(OUT, os.environ.get("OUT_NAME", "Qwen3.5-2B-VL.litertlm"))
  with open(out_path, "wb") as f:
    b.build(f)
  print("BUNDLE_DONE", out_path, round(os.path.getsize(out_path) / 1e6, 1), "MB")
  print(json.dumps({"stop": [eot_id, imend_id], "dec_act": dec_act}))


if __name__ == "__main__":
  main()
