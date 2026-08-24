"""HF eager DocTags baselines for granite-docling-258M on the two gate pages.

Two modes per page:
  full : the model's own processor defaults (do_image_splitting=True, tiles at 512)
         -> the quality CEILING (what the released model produces).
  rail : single global image stretched to exactly 512x512, no splitting
         -> the apples-to-apples baseline for the fast_vlm .litertlm (the runtime
            feeds one stretched 512x512 image; any structure already lost HERE is a
            rail/resolution limit, not conversion damage).

    .venv/bin/python docling_work/eager_docling_baseline.py [page.png ...]
Writes docling_work/eager_<page>_<mode>.txt
"""
import json, os, sys, time

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL = "src_models/granite-docling-258m"
PAGES = sys.argv[1:] or ["docling_work/table_page.png",
                         "src_models/granite-docling-258m/assets/new_arxiv.png"]
PROMPT = "Convert this page to docling."
MAXNEW = int(os.environ.get("MAXNEW", "3072"))

model = AutoModelForImageTextToText.from_pretrained(
    MODEL, torch_dtype=torch.float32, attn_implementation="eager",
    low_cpu_mem_usage=True).eval()
proc = AutoProcessor.from_pretrained(MODEL)
msgs = [{"role": "user", "content": [{"type": "image"},
                                     {"type": "text", "text": PROMPT}]}]
prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)

for page in PAGES:
    name = os.path.splitext(os.path.basename(page))[0]
    src = Image.open(page).convert("RGB")
    for mode in ("full", "rail"):
        if mode == "full":
            proc.image_processor.do_image_splitting = True
            proc.image_processor.do_resize = True
            img = src
        else:
            proc.image_processor.do_image_splitting = False
            proc.image_processor.do_resize = False
            img = src.resize((512, 512), Image.BILINEAR)
        inputs = proc(text=prompt, images=[img], return_tensors="pt")
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAXNEW, do_sample=False,
                                 use_cache=True)
        dt = time.time() - t0
        new = out[0, inputs.input_ids.shape[1]:]
        text = proc.tokenizer.decode(new, skip_special_tokens=False)
        dst = f"docling_work/eager_{name}_{mode}.txt"
        with open(dst, "w") as f:
            f.write(text)
        print(json.dumps({"page": name, "mode": mode,
                          "prompt_tokens": int(inputs.input_ids.shape[1]),
                          "new_tokens": int(new.shape[0]), "sec": round(dt, 1),
                          "hit_cap": int(new.shape[0]) >= MAXNEW, "out": dst}))
