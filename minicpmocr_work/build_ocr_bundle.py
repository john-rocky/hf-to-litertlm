#!/usr/bin/env python3
"""Assemble the MiniCPM-OCR-1.0 fast_vlm `.litertlm` bundle and desk-check its chat template.

Adapted from minicpmv47_work/build_mv47_bundle.py (hf-to-litertlm):
  EMBEDDER       : token_ids [1,1] -> [1,1,1024]                      (export_qwen35vl_decoder.py)
  PREFILL_DECODE : embeddings + 48 hybrid states -> logits [.,248100]  (tied head: the embedding rows)
  VISION_ENCODER : image NHWC [1,G,G,3] in [0,1] -> [1,(G/14)^2,1152]  (vision_export.py, 4x path)
  VISION_ADAPTER : [1,(G/14)^2,1152] -> [1,(G/28)^2,1024]              G = 448: 256 soft tokens per image
  + LlmMetadata  : FastVlm(image GxG), NO start token (the template begins with <|im_start|>), stop
                   <|endoftext|> (248044) and <|im_end|> (248046), structured prompt_templates, and the jinja of
                   --jinja: default chat_template_ocr.jinja (the ship template, r2) = chat_template_ocr_dual.jinja
                   (byte for byte the template shipped in litert-community/MiniCPM-V-4.7-1B, sha256 05f70447...,
                   1,364 B: direct answer with the empty think block, the second-turn newline moved to the next
                   turn) + page tiles: with the extra-context variable `slice_cols` set, the 2nd and later images
                   of one user message are the processor's <slice>s, `slice_cols` per row (check_template_v2.py).
                   r1's bundles were built with --jinja chat_template_ocr_dual.jinja.

Prompt contract against the checkpoint's chat_template.jinja (enable_thinking=False, its default) and
MiniCPMOcrProcessor: list content = items joined by '\\n'; an image item = <image_id>{i}</image_id><image>
<image_soft_token></image> (i counts the images of the conversation from 0), where HF puts <|image_pad|> x
(h*w/4) for one unsliced image (the 4x mode); the upstream template |trim's the content, this one does not.

    python build_ocr_bundle.py --check-template OUT.json --src SNAPSHOT --images A_s448.png B_s448.png [--g 448]
    python build_ocr_bundle.py --build --dec OUT/decoder --vis OUT/vision --venc ... --vadp ... \
        --tok SNAPSHOT/tokenizer.json --out-dir OUT/bundle --out-name NAME.litertlm [--g 448]
"""
import argparse
import json
import os

MAX_TOKENS = 4096
SOFT = "<image_soft_token>"
HERE = os.path.dirname(os.path.abspath(__file__))
JINJA_PATH = os.path.join(HERE, "chat_template_ocr.jinja")
JINJA = JINJA_PATH  # main() sets it from --jinja

THINK = "<think>\n\n</think>\n\n"
U_PRE, U_SUF = "<|im_start|>user\n", "<|im_end|>\n"
M_PRE, M_SUF = "<|im_start|>assistant\n" + THINK, "<|im_end|>\n"
S_PRE, S_SUF = "<|im_start|>system\n", "<|im_end|>\n"


def cases(a, b):
  return [
      ("img_text", [{"role": "user", "content": [{"type": "image", "fixture": a},
                                                 {"type": "text", "text": "Text Recognition:"}]}], True),
      ("text_after_image_order", [{"role": "user", "content": [{"type": "text", "text": "Table Recognition:"},
                                                               {"type": "image", "fixture": b}]}], True),
      ("text_string", [{"role": "user", "content": "What is the capital of France?"}], True),
      ("text_parts", [{"role": "user", "content": [{"type": "text", "text": "What is the capital of France?"}]}],
       True),
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
      ("two_images_two_turns", [{"role": "user", "content": [{"type": "image", "fixture": a},
                                                             {"type": "text", "text": "Text Recognition:"}]},
                                {"role": "assistant", "content": "The survey collected 1,284 samples."},
                                {"role": "user", "content": [{"type": "image", "fixture": b},
                                                             {"type": "text", "text": "Table Recognition:"}]}], True),
  ]


def check_template(out_path, src, images, g):
  import jinja2
  import transformers
  from PIL import Image
  from transformers import AutoProcessor

  jinja = open(JINJA).read()
  n_soft = (g // 28) ** 2
  proc = AutoProcessor.from_pretrained(src)
  tok = proc.tokenizer
  pad_id = tok.convert_tokens_to_ids("<|image_pad|>")
  tmpl = jinja2.Environment(undefined=jinja2.StrictUndefined).from_string(jinja)

  def bundle_ids(text):
    """The runtime's view: split on the soft token, tokenize each text piece alone, n_soft slots per image."""
    out, pieces = [], text.split(SOFT)
    for k, piece in enumerate(pieces):
      out += tok(piece, add_special_tokens=False).input_ids
      if k < len(pieces) - 1:
        out += [pad_id] * n_soft
    return out

  rows, all_ok = [], True
  for name, msgs, gen in cases(*images[:2]):
    hf_msgs, b_msgs, pil = [], [], []
    for m in msgs:
      if isinstance(m["content"], str):
        hf_msgs.append(dict(m))
        b_msgs.append(dict(m))
        continue
      hc, bc = [], []
      for it in m["content"]:
        if it["type"] == "image":
          im = Image.open(it["fixture"]).convert("RGB")
          assert im.size == (g, g), im.size
          pil.append(im)
          hc.append({"type": "image", "image": im})
          bc.append({"type": "image"})
        else:
          hc.append(dict(it))
          bc.append(dict(it))
      hf_msgs.append({"role": m["role"], "content": hc})
      b_msgs.append({"role": m["role"], "content": bc})
    hf = proc.apply_chat_template(hf_msgs, add_generation_prompt=gen, tokenize=True, return_dict=True,
                                  enable_thinking=False)
    hf_ids = hf["input_ids"]
    hf_ids = hf_ids.tolist() if hasattr(hf_ids, "tolist") else hf_ids
    hf_ids = hf_ids[0] if hf_ids and isinstance(hf_ids[0], list) else hf_ids
    b_text = tmpl.render(messages=b_msgs, add_generation_prompt=gen)
    b_ids = bundle_ids(b_text)
    same = list(hf_ids) == b_ids
    first = next((i for i, (x, y) in enumerate(zip(hf_ids, b_ids)) if x != y), None)
    rows.append({"case": name, "equal": same, "n_hf": len(hf_ids), "n_bundle": len(b_ids), "first_diff": first,
                 "bundle_text": b_text.replace(SOFT, f"<image_soft_token x{n_soft}>")})
    all_ok &= same
    print(f"{name:24s} {'EQUAL' if same else 'DIFF'} hf {len(hf_ids)} bundle {len(b_ids)} first_diff {first}")
  res = {"transformers": transformers.__version__, "jinja2": jinja2.__version__, "g": g, "n_soft": n_soft,
         "jinja_sha256": __import__("hashlib").sha256(jinja.encode()).hexdigest(), "all_equal": all_ok,
         "cases": rows}
  with open(out_path, "w") as f:
    json.dump(res, f, indent=1, ensure_ascii=False)
    f.write("\n")
  print("TEMPLATE_CHECK", "ALL_EQUAL" if all_ok else "MISMATCH", out_path)
  return 0 if all_ok else 1


def build(a):
  import litert_lm_builder as litertlm_builder
  from litert_lm_builder.runtime.proto import llm_metadata_pb2
  from litert_lm_builder.runtime.proto import llm_model_type_pb2
  from tokenizers import Tokenizer

  jinja = open(JINJA).read()
  embedder = os.path.join(a.dec, a.emb)
  prefill_decode = os.path.join(a.dec, a.decode)
  vision_encoder = os.path.join(a.vis, a.venc)
  vision_adapter = os.path.join(a.vis, a.vadp)
  for p in (embedder, prefill_decode, vision_encoder, vision_adapter, a.tok):
    assert os.path.exists(p), p
  os.makedirs(a.out_dir, exist_ok=True)

  tk = Tokenizer.from_file(a.tok)
  eot_id, imend_id = tk.token_to_id("<|endoftext|>"), tk.token_to_id("<|im_end|>")
  assert (eot_id, imend_id) == (248044, 248046), (eot_id, imend_id)
  for t in ("<image>", "</image>", "<image_id>", "</image_id>", "<|im_start|>", "<fcel>", "<nl>"):
    assert tk.token_to_id(t) is not None, f"missing {t} in tokenizer"
  assert tk.token_to_id(SOFT) is None, "soft token must not be a vocab entry"

  md = llm_metadata_pb2.LlmMetadata()
  md.max_num_tokens = a.max_tokens
  md.prompt_templates.user.prefix = U_PRE
  md.prompt_templates.user.suffix = U_SUF
  md.prompt_templates.model.prefix = M_PRE
  md.prompt_templates.model.suffix = M_SUF
  md.prompt_templates.system.prefix = S_PRE
  md.prompt_templates.system.suffix = S_SUF
  md.jinja_prompt_template = jinja
  md.llm_model_type.CopyFrom(llm_model_type_pb2.LlmModelType(fast_vlm=llm_model_type_pb2.FastVlm()))
  md.llm_model_type.fast_vlm.image_tensor_height = a.g
  md.llm_model_type.fast_vlm.image_tensor_width = a.g
  md.stop_tokens.add().token_ids.ids.append(eot_id)
  md.stop_tokens.add().token_ids.ids.append(imend_id)
  md_path = os.path.join(a.out_dir, "llm_metadata.pb")
  with open(md_path, "wb") as f:
    f.write(md.SerializeToString())

  b = litertlm_builder.LitertLmFileBuilder()
  b.add_system_metadata(litertlm_builder.Metadata(key="Authors", value="", dtype=litertlm_builder.DType.STRING))
  b.add_llm_metadata(md_path)
  b.add_hf_tokenizer(a.tok)
  b.add_tflite_model(embedder, litertlm_builder.TfLiteModelType.EMBEDDER)
  b.add_tflite_model(prefill_decode, litertlm_builder.TfLiteModelType.PREFILL_DECODE,
                     prefer_activation_type=a.dec_act or None)
  b.add_tflite_model(vision_encoder, litertlm_builder.TfLiteModelType.VISION_ENCODER)
  b.add_tflite_model(vision_adapter, litertlm_builder.TfLiteModelType.VISION_ADAPTER)
  out_path = os.path.join(a.out_dir, a.out_name)
  with open(out_path, "wb") as f:
    b.build(f)
  print("BUNDLE_DONE", out_path, os.path.getsize(out_path), "B")
  print(json.dumps({"stop": [eot_id, imend_id], "dec_act": a.dec_act, "image": a.g, "max_tokens": a.max_tokens,
                    "venc": a.venc, "vadp": a.vadp, "jinja": os.path.basename(JINJA),
                    "jinja_sha256": __import__("hashlib").sha256(jinja.encode()).hexdigest()}))
  return 0


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--check-template", default=None, metavar="OUT_JSON")
  ap.add_argument("--src", default=None)
  ap.add_argument("--images", nargs="*", default=[])
  ap.add_argument("--g", type=int, default=448)
  ap.add_argument("--build", action="store_true")
  ap.add_argument("--dec")
  ap.add_argument("--vis")
  ap.add_argument("--emb", default="embedder_wi8.tflite")
  ap.add_argument("--decode", default="prefill_decode_wi8.tflite")
  ap.add_argument("--venc", default="vision_encoder_fp16.tflite")
  ap.add_argument("--vadp", default="vision_adapter_fp16.tflite")
  ap.add_argument("--tok")
  ap.add_argument("--dec-act", default="fp32")
  ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
  ap.add_argument("--jinja", default=JINJA_PATH)
  ap.add_argument("--out-dir")
  ap.add_argument("--out-name")
  a = ap.parse_args()
  global JINJA
  JINJA = a.jinja
  if a.check_template:
    return check_template(a.check_template, a.src, a.images, a.g)
  if a.build:
    return build(a)
  ap.error("one of --check-template / --build")


if __name__ == "__main__":
  raise SystemExit(main())
