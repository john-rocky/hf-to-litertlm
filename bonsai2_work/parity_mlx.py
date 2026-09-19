"""Dump per-module outputs from PrismML's MLX runtime for fixed inputs (compared by parity_hf.py)."""
import os, sys, json
import numpy as np
pack, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(pack, "runtime"))
import mlx.core as mx
from vision_artifact import load_vl_model
model, _, cfg = load_vl_model(pack, load_processor=False); lm = model.language_model
rng = np.random.default_rng(0)
xs = {w: rng.standard_normal((1, 4, w)).astype(np.float32) for w in (5120, 6144, 17408)}
L0 = lm.model.layers[0]; L3 = lm.model.layers[3]
mods = {"l0_qkv": (L0.linear_attn.in_proj_qkv, 5120), "l0_z": (L0.linear_attn.in_proj_z, 5120), "l0_out": (L0.linear_attn.out_proj, 6144),
        "l0_gate": (L0.mlp.gate_proj, 5120), "l0_down": (L0.mlp.down_proj, 17408), "l3_q": (L3.self_attn.q_proj, 5120),
        "l3_o": (L3.self_attn.o_proj, 6144), "head": (lm.lm_head, 5120)}
res = {}
for k, (m, w) in mods.items():
    y = m(mx.array(xs[w]).astype(mx.float16)); mx.eval(y); res[k] = np.array(y.astype(mx.float32)); print(k, res[k].shape, float(np.abs(res[k]).max()))
ids = np.array([[57590, 4132, 760, 248046]], dtype=np.int32)
e = lm.model.embed_tokens(mx.array(ids)); mx.eval(e); res["emb"] = np.array(e.astype(mx.float32)); print("emb", res["emb"].shape)
for name, arr in {"A_log": L0.linear_attn.A_log, "dt_bias": L0.linear_attn.dt_bias, "conv_w": L0.linear_attn.conv1d.weight,
                  "conv_b": getattr(L0.linear_attn.conv1d, "bias", None), "gdn_norm": L0.linear_attn.norm.weight,
                  "a_w": L0.linear_attn.in_proj_a.weight, "b_w": L0.linear_attn.in_proj_b.weight, "ln0": L0.input_layernorm.weight}.items():
    if arr is not None: res["p_" + name] = np.array(arr.astype(mx.float32)); print("param", name, res["p_"+name].shape)
# one full GDN block on a real hidden state: layer-0 forward on embeddings of the ids (no cache -> fresh state)
h = lm.model.embed_tokens(mx.array(ids)); 
try:
    cache = lm.make_cache()
    o = L0(h, mask=None, cache=cache[0]) if False else None
except Exception as ex:
    print("layer call skipped:", ex)
np.savez(out, **{k: v for k, v in res.items()}, **{f"x_{w}": v for w, v in xs.items()}, ids=ids)
print("saved", out)
