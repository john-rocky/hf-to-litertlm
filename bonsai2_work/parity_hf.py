"""Compare the dequantized HF checkpoint + Hadamard modules against parity_mlx.npz (PrismML runtime outputs)."""
import json, os, sys
import numpy as np, torch
from safetensors import safe_open
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hadamard_export as H
ckpt, npz = sys.argv[1], sys.argv[2]
d = np.load(npz)
idx = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
def T(name):
    with safe_open(os.path.join(ckpt, idx[name]), framework="pt") as f: return f.get_tensor(name).float()
from safetensors.torch import load_file
signs = {int(k.split("_")[1]): v for k, v in load_file(os.path.join(ckpt, "hadamard_signs.safetensors")).items()}
hn = torch.from_numpy(H.sylvester_hadamard(1024) / 32.0).float()
def lin(name, w):
    l = torch.nn.Linear(T(name).shape[1], T(name).shape[0], bias=False); l.weight = torch.nn.Parameter(T(name)); return H.HadamardLinear(l, H.HadamardRotate(signs[w], hn))
mods = {"l0_qkv": ("model.language_model.layers.0.linear_attn.in_proj_qkv.weight", 5120), "l0_z": ("model.language_model.layers.0.linear_attn.in_proj_z.weight", 5120),
        "l0_out": ("model.language_model.layers.0.linear_attn.out_proj.weight", 6144), "l0_gate": ("model.language_model.layers.0.mlp.gate_proj.weight", 5120),
        "l0_down": ("model.language_model.layers.0.mlp.down_proj.weight", 17408), "l3_q": ("model.language_model.layers.3.self_attn.q_proj.weight", 5120),
        "l3_o": ("model.language_model.layers.3.self_attn.o_proj.weight", 6144), "head": ("lm_head.weight", 5120)}
def rep(k, ours, ref):
    ours, ref = np.asarray(ours, np.float32), np.asarray(ref, np.float32)
    err = np.abs(ours - ref).max(); scale = np.abs(ref).max(); corr = np.corrcoef(ours.ravel(), ref.ravel())[0, 1]
    print(f"{k:10s} max|diff| {err:9.4f}  ref max {scale:8.3f}  rel {err/scale:8.2e}  corr {corr:.6f}")
with torch.no_grad():
    for k, (name, w) in mods.items():
        x = torch.from_numpy(d[f"x_{w}"]).half().float()  # MLX fed fp16-cast inputs
        rep(k, lin(name, w)(x).numpy(), d[k])
    emb = torch.nn.Embedding(248320, 5120); emb.weight = torch.nn.Parameter(T("model.language_model.embed_tokens.weight"))
    he = H.HadamardEmbedding(emb, H.HadamardRotate(signs[5120], hn, inverse=True))
    rep("emb", he(torch.from_numpy(d["ids"]).long()).numpy(), d["emb"])
    # a variant check: what if the inverse should NOT apply the sign (or applies it before)?
    e_raw = emb(torch.from_numpy(d["ids"]).long())
    rep("emb_nosign", (H.HadamardRotate(torch.ones(5120), hn, inverse=True)(e_raw)).numpy(), d["emb"])
    rep("emb_signfirst", (H.HadamardRotate(signs[5120], hn, inverse=False)(e_raw)).numpy(), d["emb"])
    for pn, hf in {"A_log": "model.language_model.layers.0.linear_attn.A_log", "dt_bias": "model.language_model.layers.0.linear_attn.dt_bias",
                   "gdn_norm": "model.language_model.layers.0.linear_attn.norm.weight", "a_w": "model.language_model.layers.0.linear_attn.in_proj_a.weight",
                   "b_w": "model.language_model.layers.0.linear_attn.in_proj_b.weight", "ln0": "model.language_model.layers.0.input_layernorm.weight"}.items():
        rep("p_" + pn, T(hf).numpy(), d["p_" + pn])
    cw = T("model.language_model.layers.0.linear_attn.conv1d.weight")  # [out, 1, k]
    rep("p_conv_w", cw.transpose(1, 2).numpy(), d["p_conv_w"])  # mlx [out, k, 1]
