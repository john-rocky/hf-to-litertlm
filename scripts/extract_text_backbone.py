"""Extract the text CausalLM backbone from a multimodal *ForConditionalGeneration
checkpoint (e.g. Mistral3/Ministral-3, whose top config is NOT an AutoModelForCausalLM
type but whose text_config IS) into a standalone checkpoint that export_hf converts via
the normal text path. Fails LOUDLY (no silent weight drop) if the remap is incomplete.

    ~/clipconv/bin/python scripts/extract_text_backbone.py <hf_model> <out_text_dir>
"""
import sys
import torch
from transformers import (AutoModelForImageTextToText, AutoModelForCausalLM,
                          AutoTokenizer, AutoConfig)

mid, outdir = sys.argv[1], sys.argv[2]
cfg = AutoConfig.from_pretrained(mid, trust_remote_code=True)
tcfg = getattr(cfg, "text_config", None)
assert tcfg is not None, f"{mid} has no text_config"
print(f"text backbone model_type: {getattr(tcfg,'model_type',None)}")

print("loading full multimodal model (fp32, cpu)...")
full = AutoModelForImageTextToText.from_pretrained(
    mid, dtype=torch.float32, trust_remote_code=True)
fsd = full.state_dict()

causal = AutoModelForCausalLM.from_config(tcfg)
tgt = causal.state_dict()

VIS = ("vision", "visual", "multi_modal", "connector", "vision_tower",
       "patch_merger", "mm_projector", "image_", "img_")
src = {k: v for k, v in fsd.items() if not any(s in k.lower() for s in VIS)}

new, unmatched = {}, []
for tk, tv in tgt.items():
    # match by exact key or shared suffix (robust to any wrapper prefix), shape-checked
    cands = [k for k in src if (k == tk or k.endswith("." + tk)) and src[k].shape == tv.shape]
    if not cands:
        tail = tk.split("model.", 1)[-1]
        cands = [k for k in src if k.endswith(tail) and src[k].shape == tv.shape]
    cands = sorted(set(cands), key=len)
    if cands:
        new[tk] = src[cands[0]]
    else:
        unmatched.append(tk)

print(f"matched {len(new)}/{len(tgt)} target tensors; unmatched={len(unmatched)}")
if unmatched:
    print("UNMATCHED (first 15):", unmatched[:15])
    print("SAMPLE src keys:", list(src.keys())[:12])
    raise SystemExit("ABORT: text-backbone remap incomplete — inspect keys above")

causal.load_state_dict(new, strict=True)
causal.config._name_or_path = mid
causal.save_pretrained(outdir, safe_serialization=True)
AutoTokenizer.from_pretrained(mid, trust_remote_code=True).save_pretrained(outdir)
print("EXTRACT_DONE", outdir)
