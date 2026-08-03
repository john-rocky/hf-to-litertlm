"""End-to-end deployed-contract VQA gate for the Mage-VL fast_vlm conversion.

Compares, side by side on describe / spatial / count / OCR / table prompts:

  A) NATIVE: full Mage-VL (trust_remote_code, bf16) with its own processor at
     native dynamic resolution — the model as Microsoft ships it.
  B) DEPLOYED CONTRACT: image aspect-squashed to static IMGxIMG in [0,1] ->
     our static Encoder/Adapter (torch modules, proven == exported tflites by
     convert_magevl_vision.py) -> 196 soft tokens -> standalone Qwen3 decoder
     (magevl_work/prep_magevl_decoder.py output) via inputs_embeds with plain
     sequential positions — exactly what the fast_vlm runtime computes.

Both sides greedy (paired comparison; generation_config ships no sampling params).

    IMG=448 .venv-092/bin/python magevl_work/vqa_gate_magevl.py
"""
import json
import os
import sys

import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

MODEL = os.path.join(ROOT, "src_models/mage-vl")
LLM = os.path.join(ROOT, "src_models/mage-vl-llm")
IMG = int(os.environ.get("IMG", "448"))
MAX_NEW = int(os.environ.get("MAX_NEW", "128"))
PATCH, MERGE = 16, 2
GRID = IMG // PATCH
N_PATCH = GRID * GRID
N_TOK = N_PATCH // (MERGE * MERGE)
IMAGE_PAD_ID = 151655

# Any local photo / document / table screenshots work here (docs can be
# generated with paddleocr_work/make_test_docs.py).
PHOTO = os.environ.get("PHOTO", os.path.join(ROOT, "testimgs/photo.jpg"))
DOC = os.environ.get("DOC", os.path.join(ROOT, "paddleocr_work/testdocs/para.png"))
TABLE = os.environ.get("TABLE", os.path.join(ROOT, "paddleocr_work/testdocs/table.png"))

PROMPTS = [
    (PHOTO, "Describe this image in detail."),
    (PHOTO, "What is in the foreground versus the background?"),
    (TABLE, "How many data rows are in this table, and what is the Total revenue?"),
    (DOC, "Extract all the text from this image."),
]


def build_static_vision():
  """Static Encoder/Adapter with real weights (same rewrite convert_magevl_vision
  exports; torch modules here, exported-tflite parity already proven there)."""
  from vendor.configuration_mage_vl import MageVLVisionConfig
  from vendor.modeling_mage_vl import MageVLVisionPretrainedModel, apply_rotary_pos_emb
  import glob
  from safetensors import safe_open

  with open(os.path.join(MODEL, "config.json")) as f:
    vcfg_dict = json.load(f)["vision_config"]
  vcfg_dict.pop("model_type", None)
  cfg = MageVLVisionConfig(**vcfg_dict)
  cfg._attn_implementation = "eager"
  vt = MageVLVisionPretrainedModel(cfg).eval().float()
  sd = {}
  for shard in sorted(glob.glob(os.path.join(MODEL, "model-*.safetensors"))):
    with safe_open(shard, framework="pt") as f:
      for k in f.keys():
        if k.startswith("model.visual."):
          sd[k[len("model.visual."):]] = f.get_tensor(k).float()
  vt.load_state_dict(sd, strict=False)

  pid = torch.arange(N_PATCH)
  pos_raster = torch.stack([torch.zeros_like(pid), pid // GRID, pid % GRID], dim=-1)
  with torch.no_grad():
    freqs = vt.video_rope.forward_from_positions(pos_raster)
    freqs = torch.cat([freqs, freqs], dim=-1).unsqueeze(0)
  mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
  std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
  n_heads = cfg.num_attention_heads
  head_dim = cfg.hidden_size // n_heads
  scale = vt.encoder.layers[0].self_attn.scale

  @torch.no_grad()
  def run(img01):                                   # [1,IMG,IMG,3] in [0,1]
    x = (img01.permute(0, 3, 1, 2) - mean) / std
    h = vt.embeddings.patch_embedding(x).flatten(2).transpose(1, 2)
    h = vt.layernorm_pre(h)
    for blk in vt.encoder.layers:
      r = blk.layer_norm1(h)
      B, L, _ = r.shape
      q, k, v = (blk.self_attn.qkv(r).reshape(B, L, 3, n_heads, head_dim)
                 .permute(2, 0, 3, 1, 4).unbind(0))
      q, k = apply_rotary_pos_emb(q, k, freqs)
      attn = (torch.matmul(q, k.transpose(2, 3)) * scale).softmax(dim=-1)
      o = torch.matmul(attn, v).transpose(1, 2).reshape(B, L, -1)
      h = h + blk.self_attn.proj(o)
      h = h + blk.mlp(blk.layer_norm2(h))
    f = vt.merger.ln_q(h).reshape(1, GRID, GRID, -1)
    m = torch.cat([f[:, 0::2, 0::2, :], f[:, 0::2, 1::2, :],
                   f[:, 1::2, 0::2, :], f[:, 1::2, 1::2, :]], dim=-1)
    return vt.merger.mlp(m.reshape(1, N_TOK, -1))    # [1,N/4,2560]

  return run


def main():
  from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer, Qwen3ForCausalLM

  print("loading full Mage-VL (bf16, cpu, trust_remote_code)...", flush=True)
  full = AutoModelForImageTextToText.from_pretrained(
      MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True,
      trust_remote_code=True, attn_implementation="eager").eval()
  processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)

  print("loading standalone Qwen3 decoder (bf16)...", flush=True)
  dec = Qwen3ForCausalLM.from_pretrained(
      LLM, dtype=torch.bfloat16, low_cpu_mem_usage=True,
      attn_implementation="eager").eval()
  tok = AutoTokenizer.from_pretrained(LLM)
  static_vision = build_static_vision()

  results = []
  for img_path, question in PROMPTS:
    pil = Image.open(img_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": question}]}]
    chat = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)

    # A) native dynamic-res full model
    inputs = processor(text=[chat], images=[pil], return_tensors="pt")
    with torch.no_grad():
      out = full.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)
    ans_a = processor.tokenizer.decode(
        out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # B) deployed contract: static-448 squash + standalone decoder
    img01 = torch.from_numpy(
        __import__("numpy").asarray(pil.resize((IMG, IMG), Image.BICUBIC))
    ).float().div(255.0).unsqueeze(0)
    soft = static_vision(img01).to(torch.bfloat16)   # [1,N_TOK,2560]
    chat_b = chat.replace("<|image_pad|>", "<|image_pad|>" * N_TOK)
    ids = tok(chat_b, return_tensors="pt", add_special_tokens=False).input_ids
    emb = dec.get_input_embeddings()(ids)
    mask = ids[0] == IMAGE_PAD_ID
    assert int(mask.sum()) == N_TOK, f"pad expansion {int(mask.sum())} != {N_TOK}"
    emb[0, mask] = soft[0]
    with torch.no_grad():
      out_b = dec.generate(inputs_embeds=emb, max_new_tokens=MAX_NEW, do_sample=False)
    ans_b = tok.decode(out_b[0], skip_special_tokens=True)

    print("=" * 100)
    print("IMG:", os.path.basename(img_path), "| Q:", question)
    print("--- A native full model:\n" + ans_a.strip())
    print("--- B deployed static-%d contract:\n" % IMG + ans_b.strip())
    results.append({"image": os.path.basename(img_path), "q": question,
                    "native": ans_a.strip(), "deployed": ans_b.strip()})

  with open(os.path.join(ROOT, "out/magevl-vision/vqa_gate.json"), "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
  print("saved out/magevl-vision/vqa_gate.json")


if __name__ == "__main__":
  main()
