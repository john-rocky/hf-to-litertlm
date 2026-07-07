"""Measure how much accuracy is lost by approximating a SiLU-gated MoE's expert
activation with the fused kernel's exact erf-GELU. Isolates the ACTIVATION swap only
(fp32, no quantization) on a real SiLU MoE (granite-3.0-1b-a400m, already cached).

Conditions:
  (a) baseline SiLU (native)
  (b) plain erf-GELU swap                          gelu(gate)*up
  (c) best-fit scaled erf-GELU  c*gelu(a*gate)*up  (reauthoring-optimal; realizable by
      folding a into W_gate and c into W_up — the best the GELU-only kernel can do)

Metrics vs baseline: perplexity on fixed text, next-token top-1 agreement, mean KL.

    ~/clipconv/bin/python scripts/measure_silu_gelu_degradation.py [hf_model]
"""
import sys, math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "ibm-granite/granite-3.0-1b-a400m-instruct"
TEXT = (
    "The mitochondria is the powerhouse of the cell. In modern machine learning, "
    "mixture-of-experts models route each token to a small subset of feed-forward "
    "experts, which lets the network grow its parameter count without a proportional "
    "increase in compute per token. Natalie bought 3 boxes of pencils. Each box has "
    "12 pencils. She gave 8 pencils to her brother. How many pencils does she have "
    "left? To solve this, first multiply 3 by 12 to get 36, then subtract 8 to get 28. "
    "The quick brown fox jumps over the lazy dog while the sun sets slowly behind the "
    "distant mountains, painting the sky in shades of orange and deep crimson."
)

print(f"loading {MODEL} (fp32, cpu)...")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()

# locate the MoE expert-activation modules (gated: act(gate)*up)
moe_mods = [m for m in model.modules() if hasattr(m, "activation") and type(m).__name__.endswith("MoE")]
print(f"found {len(moe_mods)} MoE modules with a swappable .activation")

ids = tok(TEXT, return_tensors="pt").input_ids

@torch.no_grad()
def run():
    out = model(ids)
    return out.logits[0]  # [seq, vocab]

@torch.no_grad()
def perplexity(logits):
    # CE of predicting token t+1 from position t
    lp = F.log_softmax(logits[:-1].float(), dim=-1)
    tgt = ids[0, 1:]
    nll = -lp[torch.arange(tgt.shape[0]), tgt]
    return math.exp(nll.mean().item())

class _ActWrap(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self._fn = fn
    def forward(self, x):
        return self._fn(x)

def set_act(fn):
    mod = fn if isinstance(fn, torch.nn.Module) else _ActWrap(fn)
    for m in moe_mods:
        m.activation = mod

# --- (a) baseline SiLU ---
base_act = moe_mods[0].activation
base_logits = run()
base_ppl = perplexity(base_logits)
base_arg = base_logits.argmax(-1)

def compare(name, logits):
    ppl = perplexity(logits)
    agree = (logits.argmax(-1) == base_arg).float().mean().item()
    kl = F.kl_div(F.log_softmax(logits.float(), -1), F.log_softmax(base_logits.float(), -1),
                  log_target=True, reduction="batchmean").item()
    print(f"  {name:26s} ppl={ppl:8.3f} (Δ{ppl-base_ppl:+7.3f}, {100*(ppl-base_ppl)/base_ppl:+6.1f}%)  "
          f"top1-agree={100*agree:5.1f}%  meanKL={kl:.4f}")

# --- (b) plain erf-GELU ---
set_act(lambda x: F.gelu(x))          # default = exact erf
gelu_logits = run()

# --- (c) best-fit scaled erf-GELU: fit c*gelu(a*x) ~= silu(x) over actual gate activations ---
set_act(base_act)
sample = torch.randn(200000) * 3.0    # gate pre-acts ~ heavy-ish; wide range
sil = F.silu(sample)
best = None
for a in [x/100 for x in range(80, 261, 2)]:
    g = F.gelu(a * sample)
    c = (sil * g).sum() / (g * g).sum()
    mse = ((sil - c * g) ** 2).mean().item()
    if best is None or mse < best[0]:
        best = (mse, a, c.item())
_, a_opt, c_opt = best
set_act(lambda x, a=a_opt, c=c_opt: c * F.gelu(a * x))
scaled_logits = run()

print(f"\n=== SiLU -> GELU degradation on {MODEL} (activation-only, fp32) ===")
print(f"baseline SiLU               ppl={base_ppl:8.3f}")
compare("plain erf-GELU", gelu_logits)
compare(f"best-fit c*gelu(a*x) a={a_opt:.2f} c={c_opt:.3f}", scaled_logits)

# short greedy generations to eyeball coherence
set_act(base_act)
prompt = "Question: What is the capital of France? Answer:"
pids = tok(prompt, return_tensors="pt").input_ids
@torch.no_grad()
def gen():
    return tok.decode(model.generate(pids, max_new_tokens=24, do_sample=False)[0][pids.shape[1]:], skip_special_tokens=True)
print("\n--- greedy gen sanity ---")
print("SiLU :", gen().replace(chr(10), " "))
set_act(lambda x: F.gelu(x))
print("GELU :", gen().replace(chr(10), " "))
set_act(lambda x, a=a_opt, c=c_opt: c * F.gelu(a * x))
print("scaled:", gen().replace(chr(10), " "))
