"""MLX reference logits for a model truncated to the first NL layers (NL=4 keeps fa_idx=3 valid)."""
import os, sys, numpy as np
pack, out, NL = sys.argv[1], sys.argv[2], int(sys.argv[3])
sys.path.insert(0, os.path.join(pack, "runtime"))
import mlx.core as mx
from vision_artifact import load_vl_model
from transformers import AutoTokenizer
model, _, _ = load_vl_model(pack, load_processor=False); lm = model.language_model
lm.model.layers = lm.model.layers[:NL]
tok = AutoTokenizer.from_pretrained(pack)
ids = tok.apply_chat_template([{"role": "user", "content": "What is the capital of France? Answer in one word."}], add_generation_prompt=True, enable_thinking=False, tokenize=True, return_dict=False)
ids = list(ids["input_ids"]) if hasattr(ids, "keys") else list(ids)
o = lm(mx.array([ids]), cache=lm.make_cache()); lg = (o.logits if hasattr(o, "logits") else o)[0].astype(mx.float32); mx.eval(lg)
lg = np.array(lg); np.savez(out, ids=np.array(ids), logits=lg)
print("trunc", NL, "logits", lg.shape, "last top5", np.argsort(-lg[-1])[:5].tolist(), "saved", out)
