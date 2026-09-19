import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hadamard_export as H
from transformers import AutoModelForCausalLM
ckpt, npz = sys.argv[1], sys.argv[2]
d = np.load(npz); ids = torch.tensor([d["ids"].tolist()]); ref = d["logits"]
m = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.float32); m.eval(); H.install_hadamard(m, ckpt)
with torch.no_grad(): lg = m(ids, use_cache=False).logits[0].float().numpy()
for pos in (0, 1, 2, 5, -1):
    a, b = lg[pos], ref[pos]; print(f"{os.path.basename(ckpt):14s} pos {pos:3d}: corr {np.corrcoef(a,b)[0,1]:.6f} max|diff| {np.abs(a-b).max():.3f} ref max {np.abs(b).max():.2f}")
