#!/usr/bin/env python3
"""dense_check.py SRC_DIR DENSE_DIR OUT.json - prove the dense rewrite computes what the original computes.

Original = SRC_DIR loaded by AutoModelForCausalLM (GraniteMoeHybridForCausalLM, 0 experts); dense = DENSE_DIR loaded the
same way (GraniteForCausalLM). Both fp32 on CPU, one after the other in this process (the first is freed before the
second loads). Same input ids for both (the source tokenizer, vendor chat template, add_generation_prompt=True).
Per prompt (short / mid / long): full-sequence logits -> max |delta|, mean |delta|, argmax agreement over every
position; then greedy 50 new tokens (do_sample False) -> ids and decoded strings must be identical.
Guards (transformers 5.x can leave non-persistent buffers zeroed after loading): rotary inv_freq of each loaded model
is non-zero and equals the
value computed from the config (rope_theta, head_dim).
transformers 5.14.1 cannot run generate() on the original as it is:
GraniteMoeHybridModel.forward builds BOTH mask kinds (L1149-1152: "full_attention" and "linear_attention") and
masking_utils.create_recurrent_attention_mask calls past_key_values.has_previous_state(), which raises ValueError
("can only be called on LinearAttention layers, and the current Cache seem to only contain Attention layers") when the
config has no mamba layer and a 2D attention_mask + cache are given (= every generate()). The linear_attention mask is
read only by layers whose layers_block_type is "linear_attention" (L1163: causal_mask_mapping[layers_block_type[i]]);
this config has none, so the wrapper below returns None for it when the config has no such layer = computation-neutral
by construction. The full-sequence logits (no attention_mask, no cache) never reach that branch.
Bar: max |delta| <= 1e-3 fp32 on every prompt, greedy 50 identical. Exit 0 = PASS, 1 = FAIL, 2 = usage.
DTYPE=fp64 (env): the same comparison in float64, logits only (no generate) = the control for fp32 rounding:
with bit-identical weights the only difference between the two graphs is the MLP
input matmul shape (one [16384, 2560] matmul + chunk vs two [8192, 2560] matmuls), i.e. the fp32 summation order.
"""
import gc
import json
import os
import sys
import time

import torch
import transformers.models.granitemoehybrid.modeling_granitemoehybrid as _gmh
from transformers import AutoModelForCausalLM, AutoTokenizer

_orig_rec_mask = _gmh.create_recurrent_attention_mask
PATCH_CALLS = {"neutralized": 0, "passed_through": 0}


def _rec_mask_no_linear_layers(**kw):
    if "linear_attention" not in list(kw["config"].layers_block_type):
        PATCH_CALLS["neutralized"] += 1
        return None
    PATCH_CALLS["passed_through"] += 1
    return _orig_rec_mask(**kw)


_gmh.create_recurrent_attention_mask = _rec_mask_no_linear_layers

BAR = 1e-3
LONG_SRC = None  # set in main(): SRC_DIR/README.md, the model card of the pinned commit


def _r(x, n):
    """x to n decimal places, as a float."""
    return float(f"{x:.{n}f}")


def prompts(tok):
    readme = open(LONG_SRC, encoding="utf-8").read().splitlines()
    # README lines 15-41 (Model Summary .. Capabilities) and 44-88 (Generation example): prose + code, ~1.3k tokens.
    long_text = "\n".join(readme[14:41] + readme[43:88])
    msgs = {
        "short": [{"role": "user", "content": "What is the capital of France?"}],
        "mid": [{"role": "system", "content": "You are a careful math tutor. Explain each step briefly."},
                {"role": "user", "content": "Natalia sold clips to 48 of her friends in April, and then she sold "
                 "half as many clips in May. How many clips did Natalia sell altogether in April and May? Show the "
                 "arithmetic, then give the final answer on its own line."}],
        "long": [{"role": "user", "content": "Read the following model description and answer: which twelve "
                  "languages does the model support, and what is its release date?\n\n" + long_text}],
    }
    out = {}
    for name, m in msgs.items():
        text = tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
        out[name] = {"messages": m, "rendered": text, "ids": ids}
    return out


def rope_guard(model):
    cfg = model.config
    rot = model.model.rotary_emb
    base = cfg.rope_parameters["rope_theta"]
    dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    ref = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
    got = rot.inv_freq.detach().float().cpu()
    return {"rope_theta": base, "head_dim": dim, "inv_freq_absmax": float(got.abs().max()),
            "inv_freq_max_abs_err_vs_config": float((got - ref).abs().max()),
            "ok": bool(got.abs().max() > 0 and torch.allclose(got, ref, rtol=0, atol=1e-12))}


FP64 = os.environ.get("DTYPE") == "fp64"
DT = torch.float64 if FP64 else torch.float32


def run(path, ps):
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(path, dtype=DT)
    model.eval()
    info = {"class": type(model).__name__, "load_s": _r(time.time() - t0, 1), "rope": rope_guard(model),
            "dtype": str(next(model.parameters()).dtype),
            "n_params": sum(p.numel() for p in model.parameters())}
    res = {}
    with torch.inference_mode():
        for name, p in ps.items():
            t1 = time.time()
            logits = model(input_ids=p["ids"]).logits[0].to(DT).clone()
            if FP64:
                new = []
            else:
                gen = model.generate(input_ids=p["ids"], attention_mask=torch.ones_like(p["ids"]), max_new_tokens=50,
                                     do_sample=False, temperature=None, top_p=None, top_k=None)
                new = gen[0, p["ids"].shape[1]:].tolist()
            res[name] = {"logits": logits, "new_ids": new, "s": _r(time.time() - t1, 1)}
            print(f"{info['class']} {name}: {p['ids'].shape[1]} tokens, {res[name]['s']} s", flush=True)
    del model
    gc.collect()
    return info, res


def main():
    if len(sys.argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2
    src, dense, out_path = sys.argv[1:4]
    global LONG_SRC
    LONG_SRC = os.path.join(src, "README.md")
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(src)
    ps = prompts(tok)
    info_a, res_a = run(src, ps)
    info_b, res_b = run(dense, ps)
    rows, ok = {}, info_a["rope"]["ok"] and info_b["rope"]["ok"]
    for name, p in ps.items():
        la, lb = res_a[name]["logits"], res_b[name]["logits"]
        d = (la - lb).abs()
        am_a, am_b = la.argmax(-1), lb.argmax(-1)
        agree = int((am_a == am_b).sum())
        top_a = torch.topk(la[-1], 5)
        txt_a = tok.decode(res_a[name]["new_ids"])
        txt_b = tok.decode(res_b[name]["new_ids"])
        row = {"n_tokens": int(p["ids"].shape[1]),
               "first_ids": p["ids"][0, :8].tolist(),
               "max_abs_delta": float(d.max()), "mean_abs_delta": float(d.mean()),
               "max_abs_logit": float(la.abs().max()),
               "argmax_agree": f"{agree}/{la.shape[0]}",
               "last_top5_ids": top_a.indices.tolist(), "last_top5_logits": [_r(float(x), 5) for x in top_a.values],
               "greedy50_ids_identical": res_a[name]["new_ids"] == res_b[name]["new_ids"],
               "greedy50_text_identical": txt_a == txt_b,
               "greedy50_original": txt_a, "greedy50_dense": txt_b,
               "pass": float(d.max()) <= BAR and agree == la.shape[0] and res_a[name]["new_ids"] == res_b[name]["new_ids"]}
        if FP64:
            row.pop("greedy50_original"); row.pop("greedy50_dense")
            row["greedy50_ids_identical"] = row["greedy50_text_identical"] = "not run (DTYPE=fp64, logits only)"
        ok = ok and row["pass"]
        rows[name] = row
        print(f"{name}: max|d| {row['max_abs_delta']:.3e} argmax {row['argmax_agree']} greedy same "
              f"{row['greedy50_ids_identical']} -> {txt_a[:80]!r}", flush=True)
    doc = {"bar": f"max |delta| <= {BAR} (fp32, CPU) on every prompt, argmax agree on every position, "
                  f"greedy 50 identical", "dtype": str(DT).replace("torch.", ""), "device": "cpu",
           "torch": torch.__version__, "transformers": __import__("transformers").__version__,
           "original": {"path": os.path.abspath(src), **info_a}, "dense": {"path": os.path.abspath(dense), **info_b},
           "prompts": {k: {"messages": v["messages"] if k != "long" else "README L15-41 + L44-88 (see tool)",
                           "rendered_tail": v["rendered"][-120:]} for k, v in ps.items()},
           "recurrent_mask_wrapper_calls": PATCH_CALLS,
           "rows": rows, "verdict": "PASS" if ok else "FAIL"}
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=1)
    print(doc["verdict"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
