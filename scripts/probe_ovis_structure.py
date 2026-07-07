"""Inspect Ovis2.5-2B structure to learn its vision API before attempting an export
probe. Prints the module tree + the image->visual-embeddings path.
    ~/clipconv/bin/python scripts/probe_ovis_structure.py
"""
import torch
import transformers.modeling_utils as _mu
# Ovis remote code predates transformers 5.12 refactors — provide the attrs it expects.
if not hasattr(_mu.PreTrainedModel, "all_tied_weights_keys"):
    _mu.PreTrainedModel.all_tied_weights_keys = {}
if not hasattr(_mu.PreTrainedModel, "is_parallelizable"):
    _mu.PreTrainedModel.is_parallelizable = False
from transformers import AutoModelForCausalLM

MID = "AIDC-AI/Ovis2.5-2B"
print("loading Ovis2.5-2B (bf16, cpu, trust_remote_code)...")


def _patch_ovis_tie_weights():
    import sys
    for name, mod in list(sys.modules.items()):
        if "modeling_ovis2_5" in name and hasattr(mod, "Ovis2_5"):
            cls = mod.Ovis2_5
            if not getattr(cls.tie_weights, "_patched", False):
                _orig = cls.tie_weights
                def _tie(self, *a, **k):  # drop transformers-5.12 missing_keys/recompute_mapping
                    return _orig(self)
                _tie._patched = True
                cls.tie_weights = _tie
            return True
    return False


def _load():
    return AutoModelForCausalLM.from_pretrained(
        MID, trust_remote_code=True, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True).eval()

try:
    model = _load()
except TypeError as e:
    if "tie_weights" not in str(e):
        raise
    print("  patching Ovis2_5.tie_weights (drop new kwargs) and retrying...")
    _patch_ovis_tie_weights()
    model = _load()

print("\n=== top children ===")
for n, m in model.named_children():
    print(f"  {n}: {type(m).__name__}")

print("\n=== attrs of interest ===")
for name in ("visual_tokenizer", "vte", "vit", "llm", "get_visual_tokenizer",
             "get_vte", "get_llm", "backbone", "config"):
    print(f"  {name}: {hasattr(model, name)}")

vt = getattr(model, "visual_tokenizer", None)
if vt is not None:
    print("\n=== visual_tokenizer children ===")
    for n, m in vt.named_children():
        print(f"  {n}: {type(m).__name__}")
    print("  visual_tokenizer methods:",
          [m for m in dir(vt) if not m.startswith("_") and callable(getattr(vt, m, None))][:30])

print("\n=== image/visual/vision methods on model ===")
print([m for m in dir(model)
       if any(k in m.lower() for k in ("image", "visual", "vision", "pixel", "merge"))
       and not m.startswith("__")])

print("\n=== visual_vocab_size / config vision ===")
print("visual_vocab_size:", getattr(model.config, "visual_vocab_size", None))
vit = getattr(model.config, "vit_config", None)
print("vit_config type:", getattr(vit, "model_type", None) if vit else None,
      "image_size:", getattr(vit, "image_size", None) if vit else None)
print("DONE_INSPECT")
