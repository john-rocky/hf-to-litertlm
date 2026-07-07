"""CPU-execution parity: converted DeepSeek-V2 tflite (LiteRT CPU) vs HF reference.

Validates that the litert_torch converter + the deepseek_v2 real-RoPE model_ext +
the LiteRT CPU runtime preserve the model's numerics, by comparing per-position
logits of the converted graph (run on CPU via ai_edge_litert) against the STOCK
(complex-RoPE) HF model on identical tokens and identical weights.

    ~/clipconv/bin/python scripts/parity_cpu_dsv2.py <hf_dir> <export_dir>

Reference = stock HF (complex RoPE), the ground truth the converter must reproduce.
Target    = model.tflite (float) and model_quantized.tflite (int8) from <export_dir>.
"""
import sys, types, os, glob
class _D:
    def __getattr__(self, n): return lambda *a, **k: None
    def __call__(self, *a, **k): return None
_pp = types.ModuleType("scipy.sparse.linalg._propack"); _pp.__file__="<s>"; _pp.__spec__=None
for _nm in ("_spropack","_dpropack","_cpropack","_zpropack"): setattr(_pp,_nm,_D())
sys.modules["scipy.sparse.linalg._propack"]=_pp
_opt=types.ModuleType("scipy.optimize"); _opt.__file__="<s>"; _opt.__spec__=None
_opt.linear_sum_assignment=lambda *a,**k: None
sys.modules["scipy.optimize"]=_opt

import numpy as np
import torch
import transformers
from ai_edge_litert.interpreter import Interpreter

HF_DIR = sys.argv[1]
EXPORT_DIR = sys.argv[2]
L = 8                      # sequence length to test
NEG = -1e30               # additive mask "blocked" value

# ---- HF reference: STOCK complex-RoPE model (no patch) ----
torch.manual_seed(0)
ref_model = transformers.AutoModelForCausalLM.from_pretrained(HF_DIR, torch_dtype=torch.float32).eval()
vocab = ref_model.config.vocab_size
rng = np.random.default_rng(0)
ids = rng.integers(0, vocab, size=(L,)).astype(np.int32)
with torch.no_grad():
    ref_logits = ref_model(torch.tensor(ids)[None, :]).logits[0].float().numpy()  # [L, vocab]
print(f"tokens={ids.tolist()}  vocab={vocab}  ref_logits {ref_logits.shape}")
print(f"reference rotary class: {type(ref_model.model.rotary_emb).__name__} (stock = complex RoPE)")


def run_tflite_decode(tfl_path):
    itp = Interpreter(model_path=tfl_path); itp.allocate_tensors()
    runner = itp.get_signature_runner("decode")
    det = runner.get_input_details()
    # zero-init caches
    state = {}
    for n, d in det.items():
        if n.startswith("kv_cache"):
            state[n] = np.zeros([int(x) for x in d["shape"]], dtype=np.float32)
    cache_len = int(det["mask"]["shape"][-1])
    out_logits = np.zeros((L, vocab), dtype=np.float32)
    for i in range(L):
        mask = np.full((1, 1, 1, cache_len), NEG, dtype=np.float32)
        mask[..., : i + 1] = 0.0                       # causal: allow positions 0..i
        feed = dict(state)
        feed["tokens"] = np.array([[ids[i]]], dtype=np.int32)
        feed["input_pos"] = np.array([i], dtype=np.int32)
        feed["mask"] = mask
        out = runner(**feed)
        out_logits[i] = np.asarray(out["logits"]).reshape(-1)[:vocab]
        for n in state:                                # feed updated caches forward
            state[n] = np.asarray(out[n])
    return out_logits


def compare(tag, tfl_logits):
    d = np.abs(tfl_logits - ref_logits)
    maxabs = d.max(); mse = (d ** 2).mean()
    rel = maxabs / (np.abs(ref_logits).max() + 1e-9)
    top1 = (tfl_logits.argmax(-1) == ref_logits.argmax(-1)).mean()
    # top-5 set agreement averaged over positions
    t5 = np.argsort(-tfl_logits, -1)[:, :5]; r5 = np.argsort(-ref_logits, -1)[:, :5]
    top5 = np.mean([len(set(t5[i]) & set(r5[i])) / 5.0 for i in range(L)])
    # spearman-ish: correlation of logit vectors per position
    corr = np.mean([np.corrcoef(tfl_logits[i], ref_logits[i])[0, 1] for i in range(L)])
    print(f"\n[{tag}]  max|diff|={maxabs:.4e} rel={rel:.4e} mse={mse:.4e}")
    print(f"[{tag}]  top-1 argmax agree={top1*100:.1f}%  top-5 overlap={top5*100:.1f}%  logit corr={corr:.6f}")
    return top1, corr


for tfl in sorted(glob.glob(os.path.join(EXPORT_DIR, "*.tflite"))):
    name = os.path.basename(tfl)
    kind = "FLOAT" if name == "model.tflite" else "INT8 " if "quant" in name else name
    compare(f"{kind} {name}", run_tflite_decode(tfl))
