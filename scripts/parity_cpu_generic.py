"""Generic CPU-execution parity: converted tflite (LiteRT CPU) vs HF reference.

Works for any decoder arch (full-attention, MLA, SSM/linear-attn hybrid) by
auto-detecting the `decode` signature's I/O: anything that is both an input and an
output is "state" (kv-cache / conv-state / recurrent-state / ssm-state) and is
zero-initialised then fed forward step by step; `tokens`/`input_pos`/`mask` are
driven from the test sequence. Per-position logits are compared to the stock HF
model on identical tokens + identical weights.

    ~/clipconv/bin/python scripts/parity_cpu_generic.py <hf_dir> <export_dir> [L]

For C13/C16 (SSM/linear-attn) this runs at the TFLITE-INTERPRETER level, which works
even though the LiteRT-LM *engine* rejects them at GetKVCacheRootNames — isolating
the remaining gap to litert-lm serving, not the converted graph.
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

import numpy as np, torch, transformers
from ai_edge_litert.interpreter import Interpreter

HF_DIR = sys.argv[1]; EXPORT_DIR = sys.argv[2]
L = int(sys.argv[3]) if len(sys.argv) > 3 else 8
NEG = -1e30
DRIVE = {"tokens", "input_pos", "mask", "positions", "input_position", "input_positions"}

torch.manual_seed(0)
ref_model = transformers.AutoModelForCausalLM.from_pretrained(
    HF_DIR, dtype=torch.float32, trust_remote_code=True).eval()
vocab = ref_model.config.vocab_size
ids = np.random.default_rng(0).integers(0, vocab, size=(L,)).astype(np.int32)
with torch.no_grad():
    ref_logits = ref_model(torch.tensor(ids)[None, :]).logits[0].float().numpy()
print(f"HF={HF_DIR}  arch={ref_model.config.architectures}  vocab={vocab}  L={L}")


def run(tfl_path):
    itp = Interpreter(model_path=tfl_path); itp.allocate_tensors()
    if "decode" not in itp.get_signature_list():
        print("  (no 'decode' signature)"); return None
    r = itp.get_signature_runner("decode")
    indet = r.get_input_details(); outdet = r.get_output_details()
    state_names = [n for n in indet if n in outdet and n not in DRIVE]   # in AND out = state
    has_mask = "mask" in indet
    cache_len = int(indet["mask"]["shape"][-1]) if has_mask else None
    state = {n: np.zeros([int(x) for x in indet[n]["shape"]], dtype=np.float32) for n in state_names}
    logits = np.zeros((L, vocab), dtype=np.float32)
    for i in range(L):
        feed = dict(state)
        feed["tokens"] = np.array([[ids[i]]], dtype=np.int32)
        if "input_pos" in indet: feed["input_pos"] = np.array([i], dtype=np.int32)
        if has_mask:
            m = np.full((1, 1, 1, cache_len), NEG, dtype=np.float32); m[..., : i + 1] = 0.0
            feed["mask"] = m
        out = r(**feed)
        logits[i] = np.asarray(out["logits"]).reshape(-1)[:vocab]
        for n in state: state[n] = np.asarray(out[n])
    return logits, state_names


def compare(tag, res):
    if res is None: return
    tfl, snames = res
    d = np.abs(tfl - ref_logits); maxabs = d.max()
    rel = maxabs / (np.abs(ref_logits).max() + 1e-9)
    top1 = (tfl.argmax(-1) == ref_logits.argmax(-1)).mean()
    t5 = np.argsort(-tfl, -1)[:, :5]; r5 = np.argsort(-ref_logits, -1)[:, :5]
    top5 = np.mean([len(set(t5[i]) & set(r5[i])) / 5.0 for i in range(L)])
    corr = np.mean([np.corrcoef(tfl[i], ref_logits[i])[0, 1] for i in range(L)])
    print(f"\n[{tag}] state inputs: {snames}")
    print(f"[{tag}] max|diff|={maxabs:.4e} rel={rel:.4e}  top-1={top1*100:.1f}%  top-5={top5*100:.1f}%  corr={corr:.6f}")


for tfl in sorted(glob.glob(os.path.join(EXPORT_DIR, "*.tflite"))):
    name = os.path.basename(tfl)
    kind = "FLOAT" if name == "model.tflite" else "INT8 " if "quant" in name else name
    compare(f"{kind} {name}", run(tfl))
