"""model.pt (official, funasr oracle) vs model.safetensors (-vllm, port source): every tensor bit-equal? Run with the funasr_oracle venv."""
import torch
from safetensors import safe_open
ck = torch.load("out/hf_official/model.pt", map_location="cpu", weights_only=True)
sd = ck.get("state_dict", ck)
print("model.pt top-level keys:", list(ck.keys())[:5] if isinstance(ck, dict) and "state_dict" in ck else "state_dict at top", "n", len(sd))
with safe_open("out/hf_vllm/model.safetensors", framework="pt") as f:
    ks = set(f.keys())
    assert ks == set(sd.keys()), (len(ks ^ set(sd.keys())))
    neq = [k for k in ks if not (sd[k].dtype == f.get_tensor(k).dtype and torch.equal(sd[k], f.get_tensor(k)))]
print("tensors", len(ks), "not bit-equal", len(neq), neq[:5])
