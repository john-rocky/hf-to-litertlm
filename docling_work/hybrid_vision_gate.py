"""Vision-vs-decoder isolation for the granite-docling fast_vlm conversion.

Runs the REAL page through:
  A. eager vision (HF)          -> eager decoder   (reference)
  B. tflite vision fp32         -> eager decoder   (vision conversion damage?)
  C. tflite vision int8         -> eager decoder   (vision quant damage?)
For B/C the 64 soft-token embeddings from the tflite vision path replace the
<image> positions in inputs_embeds, exactly what the runtime does.
Also prints per-path embedding stats vs eager (corr / maxdiff on the real page).

    .venv/bin/python docling_work/hybrid_vision_gate.py docling_work/table_page_512.png
"""
import sys

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL = "src_models/granite-docling-258m"
VIS = "out/docling-vision"
PAGE = sys.argv[1] if len(sys.argv) > 1 else "docling_work/table_page_512.png"
MAXNEW = 900

model = AutoModelForImageTextToText.from_pretrained(
    MODEL, torch_dtype=torch.float32, attn_implementation="eager",
    low_cpu_mem_usage=True).eval()
proc = AutoProcessor.from_pretrained(MODEL)
proc.image_processor.do_image_splitting = False
proc.image_processor.do_resize = False
img = Image.open(PAGE).convert("RGB")
assert img.size == (512, 512), img.size

msgs = [{"role": "user", "content": [{"type": "image"},
                                     {"type": "text", "text": "Convert this page to docling."}]}]
prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
inputs = proc(text=prompt, images=[img], return_tensors="pt")
ids = inputs.input_ids
img_mask = ids == model.config.image_token_id
inner = model.model

# eager vision embeddings [64, 576]
with torch.no_grad():
    eag = inner.connector(inner.vision_model(
        pixel_values=inputs.pixel_values[0]).last_hidden_state)[0]

# tflite vision embeddings from the SAME [0,1] NHWC image the runtime feeds
x01 = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).unsqueeze(0)


def tfl_vision(enc_path, adp_path):
  from ai_edge_litert.interpreter import Interpreter
  def run(p, arr):
    it = Interpreter(model_path=p); it.allocate_tensors()
    d = it.get_input_details()[0]
    it.set_tensor(d["index"], arr.astype(d["dtype"])); it.invoke()
    return it.get_tensor(it.get_output_details()[0]["index"])
  feats = run(enc_path, x01.numpy())
  return torch.from_numpy(run(adp_path, feats))[0]


def gen_with_embeds(vis_emb):
  we = inner.text_model.embed_tokens(ids)
  we[img_mask] = vis_emb.to(we.dtype)
  with torch.no_grad():
    out = model.generate(inputs_embeds=we, attention_mask=inputs.attention_mask,
                         max_new_tokens=MAXNEW, do_sample=False, use_cache=True)
  return proc.tokenizer.decode(out[0], skip_special_tokens=False)


def stats(name, emb):
  a = eag.numpy().reshape(-1).astype("float64")
  b = emb.numpy().reshape(-1).astype("float64")
  print(f"== {name}: corr={np.corrcoef(a, b)[0,1]:.8f} maxdiff={np.max(np.abs(a-b)):.4f} "
        f"eager_absmax={np.abs(a).max():.1f} path_absmax={np.abs(b).max():.1f}")


paths = {
    "A_eager": eag,
    "B_tflite_fp32": tfl_vision(f"{VIS}/vision_encoder.tflite", f"{VIS}/vision_adapter.tflite"),
    "C_tflite_int8": tfl_vision(f"{VIS}/vision_encoder_int8.tflite", f"{VIS}/vision_adapter_int8.tflite"),
}
for name, emb in paths.items():
  if name != "A_eager":
    stats(name, emb)
for name, emb in paths.items():
  text = gen_with_embeds(emb)
  print(f"\n===== {name} -> eager decoder =====")
  print(text[:1200])
