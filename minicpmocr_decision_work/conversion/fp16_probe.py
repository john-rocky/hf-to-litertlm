#!/usr/bin/env python3
"""Where float16 activations break, read in PyTorch on the author's fixture inputs (no graph, no runtime): every norm's input range, the sums of squares its lowering computes, and the largest activation per block, against float16's largest value, 65,504. Needs the fixtures of fixtures_author.py (--pass 2 for the larger records).

    PYTHONHASHSEED=0 $EXPORT conversion/fp16_probe.py <work> [--records a,b,c] [--pictures a,b,c]      -> results/fp16_probe.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("work", type=Path, help="the work directory")
ap.add_argument("--records", default="ex_0,ex_1,ex_2,1962,1897,2198,1924,1918")
ap.add_argument("--pictures", default="1849_0,2172_1,2210_0,1927_0,1875_1,1957_1")
ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()
WORK = a.work.resolve()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(WORK / "host"))
import torch  # noqa: E402
from transformers.models.qwen3_5 import modeling_qwen3_5 as M  # noqa: E402

import decide_graph as G  # noqa: E402
import tower_graph as T  # noqa: E402

torch.set_num_threads(a.threads)
torch.set_grad_enabled(False)
LIMIT = 65504.0


def merge(store, name, x, what):
    v = float(x.abs().max())
    s = store.setdefault(name, {"kind": what, "max_abs": 0.0})
    s["max_abs"] = max(s["max_abs"], v)


def norm_stats(store, name, x):
    """x [.., d]: mean(x^2), the sum of squares, the largest square, whichever the lowering computes."""
    x = x.float()
    sq = x * x
    s = store.setdefault(name, {"kind": "norm", "max_abs": 0.0, "max_square": 0.0, "max_mean_square": 0.0, "max_sum_square": 0.0, "min_mean_square_real": float("inf")})
    s["max_abs"] = max(s["max_abs"], float(x.abs().max()))
    s["max_square"] = max(s["max_square"], float(sq.max()))
    s["max_mean_square"] = max(s["max_mean_square"], float(sq.mean(-1).max()))
    s["max_sum_square"] = max(s["max_sum_square"], float(sq.sum(-1).max()))
    s["min_mean_square_real"] = min(s["min_mean_square_real"], float(sq.mean(-1).min()))


def summary(store, side, names_key, names):
    ordered = sorted(store.items(), key=lambda kv: -kv[1].get("max_sum_square", kv[1]["max_abs"]))
    return {names_key: names, "sites": dict(ordered),
            "norm_inputs_over_255": [k for k, v in ordered if v["kind"] == "norm" and v["max_abs"] > 255],
            "mean_square_over_limit": [k for k, v in ordered if v["kind"] == "norm" and v["max_mean_square"] > LIMIT],
            "sum_square_over_limit": [k for k, v in ordered if v["kind"] == "norm" and v["max_sum_square"] > LIMIT],
            "max_block_out": max((v["max_abs"] for v in store.values() if v["kind"] == "block_out"), default=None),
            "max_linear_out": max((v["max_abs"] for v in store.values() if v["kind"] == "linear_out"), default=None)}


out = {"limit_fp16": LIMIT}
model, _info = G.load_text_model()
G.apply_form(model)
store = {}
hooks = []
for name, m in model.named_modules():
    if isinstance(m, M.Qwen3_5RMSNormGated):
        hooks.append(m.register_forward_pre_hook(lambda mod, ar, kw, name=name: norm_stats(store, name, ar[0] if ar else kw["hidden_states"]), with_kwargs=True))
    elif isinstance(m, M.Qwen3_5RMSNorm):
        hooks.append(m.register_forward_pre_hook(lambda mod, ar, name=name: norm_stats(store, name, ar[0])))
    elif isinstance(m, M.Qwen3_5DecoderLayer):
        hooks.append(m.register_forward_hook(lambda mod, ar, o, name=name: merge(store, name, o[0] if isinstance(o, tuple) else o, "block_out")))
    elif isinstance(m, torch.nn.Linear):
        hooks.append(m.register_forward_hook(lambda mod, ar, o, name=name: merge(store, name, o, "linear_out")))
G.set_attention(model, G.ATTN)
records = a.records.split(",")
for tag in records:
    z = np.load(WORK / f"out/fix/rec_{tag}.npz")
    L = z["ids"].shape[0]
    row = G.DecideRow(model, L, 0)
    h = row(embeds=torch.from_numpy(z["inputs_embeds"])[None], valid=torch.ones(1, L), cos=torch.from_numpy(z["cos"])[None], sin=torch.from_numpy(z["sin"])[None])["hidden"][0]
    print("record", tag, "T", L, "hidden max", float(h.abs().max()), flush=True)
    del row
for hk in hooks:
    hk.remove()
del model
out["decide"] = summary(store, "decide", "records", records)

sd = T.load_visual_state(WORK / "snapshot")
sd.pop("pos_embed.weight")
store = {}
pictures = a.pictures.split(",")
for tag in pictures:
    z = np.load(WORK / f"out/fix/pic_{tag}.npz")
    n = z["pixel_values"].shape[0]
    mod = T.TowerRow(sd, n).eval()
    hooks = []
    for name, m in mod.named_modules():
        if isinstance(m, torch.nn.LayerNorm):
            hooks.append(m.register_forward_pre_hook(lambda md, ar, name=name: norm_stats(store, name, ar[0] - ar[0].mean(-1, keepdim=True))))
        elif isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(lambda md, ar, o, name=name: merge(store, name, o, "linear_out")))
        elif isinstance(m, T.Block):
            hooks.append(m.register_forward_hook(lambda md, ar, o, name=name: merge(store, name, o, "block_out")))
    feed = T.pad_inputs(z["pixel_values"], z["pos"], z["cos"], z["sin"], n)
    r = mod(**{k: torch.from_numpy(v) for k, v in feed.items()})["rows"][0]
    print("picture", tag, "n", n, "rows max", float(r.abs().max()), flush=True)
    for hk in hooks:
        hk.remove()
out["tower"] = summary(store, "tower", "pictures", pictures)
(WORK / "results/fp16_probe.json").write_text(json.dumps(out, indent=1) + "\n")
norm = out["decide"]["sites"].get("norm", {})
print(f"FP16_PROBE the decision graph's final norm: input up to {norm.get('max_abs')}, sum of squares up to {norm.get('max_sum_square')} against {LIMIT}; decision blocks up to {out['decide']['max_block_out']}, tower blocks up to {out['tower']['max_block_out']}")
