"""Hand-assemble the MiniCPM-V-4.7-1B fast_vlm `.litertlm` bundle (copied from
qwen35vl_work/build_qwen35vl_bundle.py and adapted; the source file is untouched):

  EMBEDDER       : token_ids [1,1] -> [1,1,1024]          (embedder_wi8.tflite)
  PREFILL_DECODE : embeddings + 48 hybrid states -> logits (prefill_decode_wi8.tflite,
                   untied lm_head [248094,1024] as an int8 FC, in_proj_a/b float)
  VISION_ENCODER : image NHWC [1,448,448,3] in [0,1] -> [1,256,1152]   (vision_export.py)
  VISION_ADAPTER : [1,256,1152] -> [1,64,1024] = 64 soft tokens per image
  + LlmMetadata  : FastVlm(image 448x448), NO start token, stop <|endoftext|> AND
                   <|im_end|>, structured prompt_templates + a dual-form jinja.

Prompt contract (HF MiniCPM-V-4.7 chat_template.jinja + MiniCPMV4_7Processor):
  - turn = <|im_start|>{role}\n{content}<|im_end|>\n; system only as messages[0].
  - assistant turns (history and generation prompt) carry the empty think block
    <think>\n\n</think>\n\n (HF enable_thinking=False; the engine's incremental
    render needs history == the live stream).
  - string content as is; list content = the items joined by '\n' (HF render_content):
    text item -> item.text, image item -> <image_id>{i}</image_id><image><image_soft_token></image>
    where i counts the images of the whole rendered conversation from 0 (HF
    _prepend_local_ids counts per sample, not per message); other items are dropped.
    HF puts <|image_pad|> x 64 where the runtime splits on <image_soft_token> and
    injects the adapter's 64 embeddings.
  - every literal is emitted through {{- '...' -}} so whitespace control never eats a '\n'.

Modes:
  --check-template OUT.json   (transformers 5.19.0) desk-render this jinja with
      jinja2 and compare it to HF apply_chat_template (string and parts forms) and to the
      processor's expanded input_ids, token by token.
  --build                     (litert-lm-builder 0.15.0) write the bundle.

    python build_mv47_bundle.py --check-template W/template_check.json --src SRC --fixtures W/fixtures
    python build_mv47_bundle.py --build --dec OUT/decoder --vis OUT/vision \
        --venc vision_encoder_fp16.tflite --vadp vision_adapter_fp16.tflite \
        --tok SRC/tokenizer.json --out-dir OUT/bundle --out-name MiniCPM-V-4.7-1B-VL_int8.litertlm
"""
import argparse
import json
import os

IMAGE_SIZE = 448
MAX_TOKENS = 4096
SOFT = "<image_soft_token>"

THINK = "<think>\n\n</think>\n\n"
U_PRE, U_SUF = "<|im_start|>user\n", "<|im_end|>\n"
M_PRE, M_SUF = "<|im_start|>assistant\n" + THINK, "<|im_end|>\n"
S_PRE, S_SUF = "<|im_start|>system\n", "<|im_end|>\n"


def _lit(s):
  """A jinja output statement for a literal, with escaped newlines inside the quotes."""
  assert "'" not in s and "\\" not in s, s
  return "{{- '" + s.replace("\n", "\\n") + "' -}}"


JINJA = (
    "{%- set ns = namespace(img=0) -%}"
    "{%- for message in messages -%}"
    "{%- set role = message['role'] -%}"
    "{%- if role == 'user' or role == 'assistant' or role == 'model' or (role == 'system' and loop.first) -%}"
    "{%- if role == 'user' -%}" + _lit(U_PRE) +
    "{%- elif role == 'system' -%}" + _lit(S_PRE) +
    "{%- else -%}" + _lit(M_PRE) + "{%- endif -%}"
    "{%- if message['content'] is string -%}"
    "{{- message['content'] -}}"
    "{%- elif message['content'] is sequence -%}"
    "{%- set parts = namespace(n=0) -%}"
    "{%- for item in message['content'] -%}"
    "{%- if item['type'] == 'image' -%}"
    "{%- if parts.n > 0 -%}" + _lit("\n") + "{%- endif -%}"
    + _lit("<image_id>") + "{{- ns.img -}}" + _lit("</image_id><image>" + SOFT + "</image>") +
    "{%- set ns.img = ns.img + 1 -%}"
    "{%- set parts.n = parts.n + 1 -%}"
    "{%- elif item['type'] == 'text' -%}"
    "{%- if parts.n > 0 -%}" + _lit("\n") + "{%- endif -%}"
    "{{- item['text'] -}}"
    "{%- set parts.n = parts.n + 1 -%}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- endif -%}"
    + _lit("<|im_end|>\n") +
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}" + _lit(M_PRE) + "{%- endif -%}"
)

# -------------------------------------------------------------------- template check
CASES = [
    # name, messages (image items carry a fixture name), add_generation_prompt
    ("img_text", [{"role": "user", "content": [
        {"type": "image", "fixture": "cats_448.png"},
        {"type": "text", "text": "What is in this image?"}]}], True),
    ("text_after_image_order", [{"role": "user", "content": [
        {"type": "text", "text": "Look at this picture."},
        {"type": "image", "fixture": "kitchen1_448.png"}]}], True),
    ("text_string", [{"role": "user", "content": "What is the capital of France?"}], True),
    ("text_parts", [{"role": "user", "content": [
        {"type": "text", "text": "What is the capital of France?"}]}], True),
    ("system_text_string", [{"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": "Hello!"}], True),
    ("system_text_parts", [{"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                           {"role": "user", "content": [{"type": "text", "text": "Hello!"}]}], True),
    ("two_turn_string", [{"role": "user", "content": "What is 2 + 2?"},
                         {"role": "assistant", "content": "4."},
                         {"role": "user", "content": "And 3 + 3?"}], True),
    ("two_turn_parts", [{"role": "user", "content": [{"type": "text", "text": "What is 2 + 2?"}]},
                        {"role": "assistant", "content": [{"type": "text", "text": "4."}]},
                        {"role": "user", "content": [{"type": "text", "text": "And 3 + 3?"}]}], True),
    ("two_images_two_turns", [{"role": "user", "content": [
        {"type": "image", "fixture": "cats_448.png"},
        {"type": "text", "text": "What is in this image?"}]},
        {"role": "assistant", "content": "Two cats on a pink blanket."},
        {"role": "user", "content": [
            {"type": "image", "fixture": "kitchen1_448.png"},
            {"type": "text", "text": "And in this one?"}]}], True),
]


def check_template(out_path, src, fixtures):
  import jinja2
  import transformers
  from PIL import Image
  from transformers import AutoProcessor

  proc = AutoProcessor.from_pretrained(src)
  tok = proc.tokenizer
  pad_id = tok.convert_tokens_to_ids("<|image_pad|>")
  env = jinja2.Environment(undefined=jinja2.StrictUndefined)
  tmpl = env.from_string(JINJA)

  def bundle_ids(text):
    """Tokenize the bundle render the way the runtime does: split on the soft token,
    tokenize each text piece alone, put 64 image slots at each soft token."""
    out = []
    pieces = text.split(SOFT)
    for k, piece in enumerate(pieces):
      out += tok(piece, add_special_tokens=False).input_ids
      if k < len(pieces) - 1:
        out += [pad_id] * 64
    return out

  rows, all_ok = [], True
  for name, msgs, gen in CASES:
    images = []
    hf_msgs, b_msgs = [], []
    for m in msgs:
      c = m["content"]
      if isinstance(c, str):
        hf_msgs.append({"role": m["role"], "content": c})
        b_msgs.append({"role": m["role"], "content": c})
        continue
      hc, bc = [], []
      for it in c:
        if it["type"] == "image":
          im = Image.open(os.path.join(fixtures, it["fixture"])).convert("RGB")
          assert im.size == (IMAGE_SIZE, IMAGE_SIZE), im.size
          images.append(im)
          hc.append({"type": "image"})
          bc.append({"type": "image"})
        else:
          hc.append(dict(it))
          bc.append(dict(it))
      hf_msgs.append({"role": m["role"], "content": hc})
      b_msgs.append({"role": m["role"], "content": bc})

    hf_template = proc.apply_chat_template(hf_msgs, tokenize=False, add_generation_prompt=gen,
                                           enable_thinking=False)
    if images:
      enc = proc(text=[hf_template], images=images, return_tensors="pt")
    else:
      enc = proc(text=[hf_template], return_tensors="pt")
    hf_ids = enc["input_ids"][0].tolist()
    hf_text = tok.decode(hf_ids, skip_special_tokens=False)
    hf_text_soft = hf_text.replace("<|image_pad|>" * 64, SOFT)
    b_text = tmpl.render(messages=b_msgs, add_generation_prompt=gen)
    b_ids = bundle_ids(b_text)
    first_div = next((i for i, (a, b) in enumerate(zip(hf_ids, b_ids)) if a != b),
                     None if len(hf_ids) == len(b_ids) else min(len(hf_ids), len(b_ids)))
    row = {
        "case": name,
        "n_images": len(images),
        "hf_apply_chat_template": hf_template,
        "hf_expanded_text_soft": hf_text_soft,
        "bundle_render": b_text,
        "text_equal": hf_text_soft == b_text,
        "hf_ids_len": len(hf_ids),
        "bundle_ids_len": len(b_ids),
        "ids_equal": hf_ids == b_ids,
        "first_id_divergence": first_div,
        "image_pad_runs_64": hf_text.count("<|image_pad|>" * 64),
        "image_pad_total": hf_ids.count(pad_id),
    }
    # The engine's prefix contract: render(history, no gen prompt) must be a prefix of
    # render(history + next user turn, gen prompt).
    if len(b_msgs) >= 3:
      prev = tmpl.render(messages=b_msgs[:-1], add_generation_prompt=False)
      row["prefix_contract_ok"] = b_text.startswith(prev)
    rows.append(row)
    ok = row["text_equal"] and row["ids_equal"] and row.get("prefix_contract_ok", True)
    all_ok &= ok
    print(f"[{'ok' if ok else 'NG'}] {name}: text_equal={row['text_equal']} ids_equal={row['ids_equal']} "
          f"len hf/bundle={len(hf_ids)}/{len(b_ids)} first_div={first_div}"
          + (f" prefix_contract_ok={row['prefix_contract_ok']}" if "prefix_contract_ok" in row else ""),
          flush=True)

  res = {
      "transformers": transformers.__version__,
      "jinja2": jinja2.__version__,
      "src": src,
      "processor": type(proc).__name__,
      "image_processor": type(proc.image_processor).__name__,
      "rule": ("HF = processor.apply_chat_template(enable_thinking=False) then the processor's own "
               "expansion (input_ids decoded, <|image_pad|> x 64 -> <image_soft_token>); bundle = this "
               "jinja rendered with jinja2; ids = bundle text split on the soft token, each piece tokenized "
               "alone (add_special_tokens=False) with 64 pad ids per soft token"),
      "jinja": JINJA,
      "all_equal": all_ok,
      "cases": rows,
  }
  with open(out_path, "w") as f:
    json.dump(res, f, indent=1, ensure_ascii=False)
    f.write("\n")
  print("TEMPLATE_CHECK", "ALL_EQUAL" if all_ok else "MISMATCH", out_path)
  return 0 if all_ok else 1


# -------------------------------------------------------------------- build
def build(a):
  import litert_lm_builder as litertlm_builder
  from litert_lm_builder.runtime.proto import llm_metadata_pb2
  from litert_lm_builder.runtime.proto import llm_model_type_pb2
  from tokenizers import Tokenizer

  embedder = os.path.join(a.dec, a.emb)
  prefill_decode = os.path.join(a.dec, a.decode)
  vision_encoder = os.path.join(a.vis, a.venc)
  vision_adapter = os.path.join(a.vis, a.vadp)
  for p in (embedder, prefill_decode, vision_encoder, vision_adapter, a.tok):
    assert os.path.exists(p), p
  os.makedirs(a.out_dir, exist_ok=True)

  # Verify the stop ids and the image layout tokens against the tokenizer before baking them.
  tk = Tokenizer.from_file(a.tok)
  eot_id = tk.token_to_id("<|endoftext|>")
  imend_id = tk.token_to_id("<|im_end|>")
  assert (eot_id, imend_id) == (248044, 248046), (eot_id, imend_id)
  for t in ("<image>", "</image>", "<image_id>", "</image_id>", "<|im_start|>"):
    assert tk.token_to_id(t) is not None, f"missing {t} in tokenizer"
  assert tk.token_to_id(SOFT) is None, "soft token must not be a vocab entry"
  print("stop ids:", {"<|endoftext|>": eot_id, "<|im_end|>": imend_id})
  print("embedder:", embedder, "\nprefill_decode:", prefill_decode)
  print("vision_encoder:", vision_encoder, "\nvision_adapter:", vision_adapter)

  md = llm_metadata_pb2.LlmMetadata()
  md.max_num_tokens = MAX_TOKENS
  # NO start_token: the HF template begins directly with <|im_start|> (r1 nostart ruling).
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
  md_path = os.path.join(a.out_dir, "llm_metadata.pb")
  with open(md_path, "wb") as f:
    f.write(md.SerializeToString())

  b = litertlm_builder.LitertLmFileBuilder()
  b.add_system_metadata(litertlm_builder.Metadata(
      key="Authors", value="", dtype=litertlm_builder.DType.STRING))
  b.add_llm_metadata(md_path)
  b.add_hf_tokenizer(a.tok)
  b.add_tflite_model(embedder, litertlm_builder.TfLiteModelType.EMBEDDER)
  # The gated-delta decoder needs fp32 activations on GPU (family rule; r1 FINAL carries it too).
  b.add_tflite_model(prefill_decode, litertlm_builder.TfLiteModelType.PREFILL_DECODE,
                     prefer_activation_type=a.dec_act or None)
  b.add_tflite_model(vision_encoder, litertlm_builder.TfLiteModelType.VISION_ENCODER)
  b.add_tflite_model(vision_adapter, litertlm_builder.TfLiteModelType.VISION_ADAPTER)

  out_path = os.path.join(a.out_dir, a.out_name)
  with open(out_path, "wb") as f:
    b.build(f)
  print("BUNDLE_DONE", out_path, os.path.getsize(out_path), "B")
  print(json.dumps({"stop": [eot_id, imend_id], "dec_act": a.dec_act, "image": IMAGE_SIZE,
                    "venc": a.venc, "vadp": a.vadp}))
  return 0


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--check-template", default=None, metavar="OUT_JSON")
  ap.add_argument("--src", default=None)
  ap.add_argument("--fixtures", default=None)
  ap.add_argument("--build", action="store_true")
  ap.add_argument("--dec")
  ap.add_argument("--vis")
  ap.add_argument("--emb", default="embedder_wi8.tflite")
  ap.add_argument("--decode", default="prefill_decode_wi8.tflite")
  ap.add_argument("--venc", default="vision_encoder_fp16.tflite")
  ap.add_argument("--vadp", default="vision_adapter_fp16.tflite")
  ap.add_argument("--tok")
  ap.add_argument("--dec-act", default="fp32")
  ap.add_argument("--out-dir")
  ap.add_argument("--out-name")
  ap.add_argument("--print-jinja", action="store_true")
  a = ap.parse_args()
  if a.print_jinja:
    print(JINJA)
    return 0
  if a.check_template:
    return check_template(a.check_template, a.src, a.fixtures)
  if a.build:
    return build(a)
  ap.error("one of --check-template / --build / --print-jinja")


if __name__ == "__main__":
  raise SystemExit(main())
