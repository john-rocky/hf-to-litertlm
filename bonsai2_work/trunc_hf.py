"""HF logits for the truncated checkpoint (fp32 and bf16) vs trunc_mlx.npz."""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hadamard_export as H
from transformers import AutoModelForCausalLM
ckpt, npz = sys.argv[1], sys.argv[2]
d = np.load(npz); ids = torch.tensor([d["ids"].tolist()]); ref = d["logits"]
for dt in (torch.float32, torch.bfloat16):
    m = AutoModelForCausalLM.from_pretrained(ckpt, dtype=dt); m.eval(); H.install_hadamard(m, ckpt)
    with torch.no_grad(): lg = m(ids).logits[0].float().numpy()
    for pos in (-1, 0, 5):
        a, b = lg[pos], ref[pos]; corr = np.corrcoef(a, b)[0, 1]
        print(f"{str(dt):15s} pos {pos:3d}: corr {corr:.6f}  max|diff| {np.abs(a-b).max():.3f}  ref max {np.abs(b).max():.2f}  top5 hf {np.argsort(-a)[:5].tolist()} mlx {np.argsort(-b)[:5].tolist()}")
    del m
