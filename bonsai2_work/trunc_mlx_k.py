"""MLX logits with only the first K decoder layers active (later layers replaced by identity), K in argv."""
import os, sys, numpy as np
pack, out, K = sys.argv[1], sys.argv[2], int(sys.argv[3])
sys.path.insert(0, os.path.join(pack, "runtime"))
import mlx.core as mx
from vision_artifact import load_vl_model
from transformers import AutoTokenizer
model, _, _ = load_vl_model(pack, load_processor=False); lm = model.language_model
class Ident:
    def __init__(self, layer): object.__setattr__(self, "_l", layer)
    def __getattr__(self, n): return getattr(object.__getattribute__(self, "_l"), n)
    def __call__(self, x, *a, **k): return x
lm.model.layers = [l if i < K else Ident(l) for i, l in enumerate(lm.model.layers[:4])]
tok = AutoTokenizer.from_pretrained(pack)
ids = tok.apply_chat_template([{"role": "user", "content": "What is the capital of France? Answer in one word."}], add_generation_prompt=True, enable_thinking=False, tokenize=True, return_dict=False)
ids = list(ids["input_ids"]) if hasattr(ids, "keys") else list(ids)
o = lm(mx.array([ids]), cache=lm.make_cache()); lg = (o.logits if hasattr(o, "logits") else o)[0].astype(mx.float32); mx.eval(lg)
np.savez(out, ids=np.array(ids), logits=np.array(lg)); print("K", K, "saved", out)
