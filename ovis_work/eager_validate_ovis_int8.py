"""Strongest pre-ship check: run the EXACT shipped int8 vision tflites (encoder+adapter)
on a real image, splice the 256 embeddings into a ChatML prompt, and generate with the
Ovis Qwen3 decoder. Confirms the deployed int8 path (not just fp32) stays grounded.

    ~/clipconv/bin/python ovis_work/eager_validate_ovis_int8.py [image] [question]
"""
import sys, numpy as np, torch
import transformers.modeling_utils as _mu
if not hasattr(_mu.PreTrainedModel, "all_tied_weights_keys"): _mu.PreTrainedModel.all_tied_weights_keys = {}
if not hasattr(_mu.PreTrainedModel, "is_parallelizable"): _mu.PreTrainedModel.is_parallelizable = False
from transformers import AutoModelForCausalLM
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

MID = "AIDC-AI/Ovis2.5-2B"
VIS = "out/ovis-vision"
IMG = sys.argv[1] if len(sys.argv) > 1 else "colorization_work/repo/imgs/ansel_adams.jpg"
Q = sys.argv[2] if len(sys.argv) > 2 else "Describe this image in detail."


def tfl_run(p, x):
  it = Interpreter(model_path=p); it.allocate_tensors()
  d = it.get_input_details()[0]
  it.set_tensor(d["index"], x.astype(d["dtype"])); it.invoke()
  return it.get_tensor(it.get_output_details()[0]["index"])


def load():
  return AutoModelForCausalLM.from_pretrained(
      MID, trust_remote_code=True, torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()


print("loading Ovis2.5-2B (decoder+tokenizer only used)...")
try:
  model = load()
except TypeError as e:
  if "tie_weights" not in str(e): raise
  for name, mod in list(sys.modules.items()):
    if "modeling_ovis2_5" in name and hasattr(mod, "Ovis2_5"):
      cls = mod.Ovis2_5; _o = cls.tie_weights; cls.tie_weights = lambda self, *a, **k: _o(self)
  model = load()

tok = model.text_tokenizer
wte = model.llm.get_input_embeddings()

pil = Image.open(IMG).convert("RGB").resize((512, 512), Image.BILINEAR)
img01 = (np.asarray(pil).astype(np.float32) / 255.0)[None]  # [1,512,512,3]
feat = tfl_run(f"{VIS}/vision_encoder_int8.tflite", img01)              # [1,256,4608]
img_emb = torch.from_numpy(tfl_run(f"{VIS}/vision_adapter_int8.tflite", feat)).squeeze(0)  # [256,2048]
print("int8 vision -> img_emb", tuple(img_emb.shape))

with torch.no_grad():
  pre = tok("<|im_start|>user\n", add_special_tokens=False, return_tensors="pt").input_ids
  post = tok("\n" + Q + "<|im_end|>\n<|im_start|>assistant\n",
             add_special_tokens=False, return_tensors="pt").input_ids
  seq = torch.cat([wte(pre)[0], img_emb.to(wte.weight.dtype), wte(post)[0]], dim=0).unsqueeze(0)
  out = model.llm.generate(inputs_embeds=seq, max_new_tokens=200, do_sample=False,
                           pad_token_id=tok.pad_token_id or tok.eos_token_id)
print("\n=== QUESTION ===\n" + Q)
print("\n=== RESPONSE (SHIPPED int8 vision tflites -> Qwen3-1.7B) ===\n" +
      tok.decode(out[0], skip_special_tokens=True))
