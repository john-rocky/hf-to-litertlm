"""Per-layer int4 sensitivity scan (lever #2) — data-free rescue targeting.

Fake-quantizes weights with the exact ai_edge_quantizer BMIX4 form (blockwise-32,
symmetric, NARROW range [-7,7], min-max scale = absmax/7 per block along the
contracting dim) and measures logit KL vs the fp reference on a small prompt set.

Two directions per layer:
  solo  : only layer i int4        -> KL_i        (damage caused by layer i)
  rescue: all int4 EXCEPT layer i  -> KL_all - KL_-i (marginal gain of rescuing i)
Plus per-projection-type scans (all layers at once) and lm_head.

    ~/clipconv/bin/python scripts/int4_sensitivity_scan.py <hf_id> <out_json> [device]
"""
import json
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_ID = sys.argv[1]
OUT_JSON = sys.argv[2]
DEVICE = sys.argv[3] if len(sys.argv) > 3 else "cpu"

PAIRS = [
    ("What is 17 + 25?", "17 + 25 = 42."),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ("Say 'thank you' in French.", "Merci."),
    ("How many days are in a week?", "There are 7 days in a week."),
    ("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
     "In May she sold 48 / 2 = 24 clips. Altogether she sold 48 + 24 = 72 clips. #### 72"),
    ("Explain photosynthesis in one sentence.",
     "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen."),
    ("What is the opposite of hot?", "The opposite of hot is cold."),
    ("List three primary colors.", "The three primary colors are red, blue, and yellow."),
]

PROJS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def fake_quant_bmix4(w: torch.Tensor, block: int = 32) -> torch.Tensor:
    """Blockwise-32 symmetric narrow-range int4 min-max, along the in (last) dim."""
    out_f, in_f = w.shape
    assert in_f % block == 0, (w.shape, block)
    W = w.float().reshape(out_f, in_f // block, block)
    scale = W.abs().amax(dim=-1, keepdim=True) / 7.0
    scale = torch.clamp(scale, min=1e-9)
    q = torch.clamp(torch.round(W / scale), -7, 7)
    return (q * scale).reshape(out_f, in_f).to(w.dtype)


def main():
    tok = AutoTokenizer.from_pretrained(HF_ID)
    model = AutoModelForCausalLM.from_pretrained(HF_ID, dtype=torch.float32).to(DEVICE).eval()
    layers = model.model.layers
    n_layers = len(layers)
    print(f"{HF_ID}: {n_layers} layers, device={DEVICE}")

    # tokenized prompt+answer, and the position from which we score
    batches = []
    for q, a in PAIRS:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True
        )
        full = tok(prompt + a, return_tensors="pt").input_ids.to(DEVICE)
        p_len = tok(prompt, return_tensors="pt").input_ids.shape[1]
        batches.append((full, p_len - 1))

    @torch.no_grad()
    def logits_all():
        outs = []
        for ids, start in batches:
            lg = model(ids).logits[0, start:-1].float()  # predict answer tokens
            outs.append(lg)
        return outs

    @torch.no_grad()
    def kl_vs(ref):
        got = logits_all()
        kls = []
        for r, g in zip(ref, got):
            kls.append(F.kl_div(
                F.log_softmax(g, dim=-1), F.softmax(r, dim=-1), reduction="batchmean"
            ).item())
        return sum(kls) / len(kls)

    def named_linears(layer_idx=None, proj=None, lm_head=False):
        mods = []
        if lm_head:
            mods.append(("lm_head", model.lm_head))
            return mods
        for i, lyr in enumerate(layers):
            if layer_idx is not None and i != layer_idx:
                continue
            for p in PROJS:
                if proj is not None and p != proj:
                    continue
                holder = lyr.self_attn if "proj" in p and p in ("q_proj", "k_proj", "v_proj", "o_proj") else lyr.mlp
                mods.append((f"L{i}.{p}", getattr(holder, p)))
        return mods

    def quantize(mods):
        saved = [(m, m.weight.data) for _, m in mods]
        for _, m in mods:
            m.weight.data = fake_quant_bmix4(m.weight.data)
        return saved

    def restore(saved):
        for m, w in saved:
            m.weight.data = w

    ref = logits_all()
    results = {"model": HF_ID, "n_layers": n_layers, "solo": {}, "proj": {}, "rescue": {}}

    # all-int4 reference damage (incl. lm_head, mirroring FULLY_CONNECTED int4)
    all_mods = named_linears() + named_linears(lm_head=True)
    saved = quantize(all_mods)
    kl_all = kl_vs(ref)
    restore(saved)
    results["kl_all_int4"] = kl_all
    print(f"KL all-int4 = {kl_all:.5f}")

    # solo scans
    for i in range(n_layers):
        saved = quantize(named_linears(layer_idx=i))
        kl = kl_vs(ref)
        restore(saved)
        results["solo"][str(i)] = kl
        print(f"solo L{i:02d}: KL={kl:.6f}")
    saved = quantize(named_linears(lm_head=True))
    results["solo"]["lm_head"] = kl_vs(ref)
    restore(saved)
    print(f"solo lm_head: KL={results['solo']['lm_head']:.6f}")

    # proj-type scans
    for p in PROJS:
        saved = quantize(named_linears(proj=p))
        kl = kl_vs(ref)
        restore(saved)
        results["proj"][p] = kl
        print(f"proj {p}: KL={kl:.6f}")

    # rescue scans (leave-one-out from all-int4)
    for i in list(range(n_layers)) + ["lm_head"]:
        if i == "lm_head":
            keep = named_linears(lm_head=True)
        else:
            keep = named_linears(layer_idx=i)
        keep_ids = {id(m) for _, m in keep}
        mods = [(n, m) for n, m in all_mods if id(m) not in keep_ids]
        saved = quantize(mods)
        kl = kl_vs(ref)
        restore(saved)
        results["rescue"][str(i)] = kl_all - kl
        print(f"rescue L{i}: ΔKL={kl_all - kl:.6f} (residual {kl:.6f})")

    json.dump(results, open(OUT_JSON, "w"), indent=2)
    print("WROTE", OUT_JSON)


if __name__ == "__main__":
    main()
