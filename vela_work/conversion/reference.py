"""Step 2: the publisher's stack as the reference (oracle/).

    python3   # transformers 4.57.6 + sentence-transformers 5.1.2 scripts/reference.py [--threads 8] [--limit N]

Stack = the one the card / AMD_RUNTIME.md qualify: sentence-transformers 5.1.2 + transformers 4.57.6 (the repo's own
`transformers_version`), torch fp32 on the CPU, SDPA forced to the MATH backend (AMD_RUNTIME.md: "the existing embedding
training and evaluation runs used the MATH backend"). SentenceTransformer(hf/) = Transformer -> Pooling(mean) ->
Normalize, exactly the repo's modules.json. Every fixture text is encoded ALONE (batch 1: no padding in the
reference); a batched probe (batch 32, right-padded by sentence-transformers) is recorded for the batch-shape delta.
Writes:
  oracle/embeddings.npz      key -> float32[768] (unit vectors), the reference of every gate
  oracle/oracle.json         per-row ids check (ST tokenizer == tokenizers), pair cosines, task metrics (PAWS-X AP per
                             language, STS-B v2 Spearman when the fixture holds it), the batched-vs-single delta, env
  oracle/rope_L512.npz       cos / sin [512, 64] of both layer types from the 4.57.6 rotary modules (yarn, factor 4)
  oracle/fp16_headroom.json  per-site maxima over all fixture texts: LayerNorm inputs (|x|, (x-mean)^2, centered and raw
                             sum of squares), residual stream, GeGLU inputs / product, attention output — the sites that
                             decide the fp16 rewrite (phocinae_work: mmBERT layers 12-21 + final overflow)
"""
import argparse
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

W = Path(__file__).resolve().parents[1]
HF = W / "hf"
FP16_MAX = 65504.0


def spearman(a, b):
    def rank(x):
        x = np.asarray(x, dtype=np.float64)
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), dtype=np.float64)
        # average ranks for ties
        i = 0
        while i < len(x):
            j = i
            while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
                j += 1
            r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra @ rb) / (np.linalg.norm(ra) * np.linalg.norm(rb)))


def average_precision(scores, labels):
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="mergesort")
    y = np.asarray(labels)[order]
    hits = np.cumsum(y)
    prec = hits / (np.arange(len(y)) + 1)
    return float((prec * y).sum() / max(1, y.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from sentence_transformers import SentenceTransformer
    from tokenizers import Tokenizer
    t0 = time.time()
    model = SentenceTransformer(str(HF), device="cpu", model_kwargs={"torch_dtype": torch.float32, "attn_implementation": "sdpa"},
                                config_kwargs={"reference_compile": False})
    model.eval()
    body = model[0].auto_model
    assert type(body).__name__ == "ModernBertModel", type(body).__name__
    assert model[1].get_pooling_mode_str() == "mean", model[1].get_pooling_mode_str()
    assert type(model[2]).__name__ == "Normalize"
    print("modules:", [type(m).__name__ for m in model], "max_seq_length", model.max_seq_length, "prompts", model.prompts, flush=True)
    cfg = body.config
    # 4.57.6 has no config.layer_types: layer i is global (full) when i % global_attn_every_n_layers == 0, else local (sliding)
    layer_types = ["full_attention" if i % cfg.global_attn_every_n_layers == 0 else "sliding_attention" for i in range(cfg.num_hidden_layers)]
    for i, layer in enumerate(body.layers):
        assert (layer.attn.local_attention == (-1, -1)) == (layer_types[i] == "full_attention"), (i, layer.attn.local_attention)
    print("layer_types:", {t: layer_types.count(t) for t in set(layer_types)}, "rope_scaling:", cfg.rope_scaling, flush=True)

    # --- rope tables from the 4.57.6 modules (one per layer type)
    rope = {}
    for i, layer in enumerate(body.layers):
        lt = layer_types[i]
        if lt in rope:
            continue
        re_ = layer.attn.rotary_emb
        pos = torch.arange(512)[None]
        dummy = torch.zeros(1, 512, 3, cfg.num_attention_heads, cfg.hidden_size // cfg.num_attention_heads)
        with torch.no_grad():
            cos, sin = re_(dummy, position_ids=pos)
        rope[lt] = dict(cos=cos[0].numpy().astype(np.float32), sin=sin[0].numpy().astype(np.float32),
                        rope_type=getattr(re_, "rope_type", "?"), attention_scaling=float(getattr(re_, "attention_scaling", 1.0)),
                        inv_freq=re_.inv_freq.numpy().astype(np.float64), layer=i)
        print(f"rope {lt}: layer {i} type {rope[lt]['rope_type']} scaling {rope[lt]['attention_scaling']} inv_freq[0:3] {rope[lt]['inv_freq'][:3]} [-3:] {rope[lt]['inv_freq'][-3:]}", flush=True)
    np.savez(W / "oracle" / "rope_L512.npz", **{f"{lt}_{k}": v for lt, d in rope.items() for k, v in d.items() if k in ("cos", "sin", "inv_freq")})

    # --- fp16 headroom taps (forward hooks; the reference runs unpadded, so every position is valid)
    taps = {}

    def rec(key, value):
        v = float(value)
        if v > taps.get(key, float("-inf")):
            taps[key] = v

    def ln_hook(name):
        def hook(mod, inputs):
            x = inputs[0][0]
            c = x - x.mean(-1, keepdim=True)
            rec(f"ln:{name}:abs", x.abs().max()); rec(f"ln:{name}:sq", (c * c).max()); rec(f"ln:{name}:sumsq", (c * c).sum(-1).max()); rec(f"ln:{name}:sumsq_raw", (x * x).sum(-1).max())
        return hook

    def wi_hook(name):
        def hook(mod, inputs, out):
            inp, gate = out[0].chunk(2, dim=-1)
            rec(f"wi:{name}", torch.maximum(inp.abs().max(), gate.abs().max())); rec(f"geglu:{name}", (torch.nn.functional.gelu(inp) * gate).abs().max())
        return hook

    def out_hook(name):
        def hook(mod, inputs, out):
            rec(f"{name}", out[0].abs().max() if not isinstance(out, tuple) else out[0][0].abs().max())
        return hook

    handles = [body.embeddings.norm.register_forward_pre_hook(ln_hook("emb")), body.final_norm.register_forward_pre_hook(ln_hook("final"))]
    for i, layer in enumerate(body.layers):
        if not isinstance(layer.attn_norm, torch.nn.Identity):
            handles.append(layer.attn_norm.register_forward_pre_hook(ln_hook(f"attn{i}")))
        handles.append(layer.mlp_norm.register_forward_pre_hook(ln_hook(f"mlp{i}")))
        handles.append(layer.mlp.Wi.register_forward_hook(wi_hook(f"enc{i}")))
        handles.append(layer.attn.Wo.register_forward_pre_hook(lambda m, inp, n=f"attnout:enc{i}": rec(n, inp[0].abs().max())))
        handles.append(layer.register_forward_hook(out_hook(f"residual:enc{i}")))

    fx = json.loads((W / "fixtures" / "texts.json").read_text())
    rows = fx["rows"][: a.limit] if a.limit else fx["rows"]
    tok = Tokenizer.from_file(str(HF / "tokenizer.json"))
    emb, recs, ids_equal = {}, [], 0
    t1 = time.time()
    with sdpa_kernel(SDPBackend.MATH):
        for n, r in enumerate(rows):
            st_ids = model.tokenize([r["text"]])["input_ids"][0].tolist()
            same = st_ids == r["ids"] == tok.encode(r["text"]).ids
            ids_equal += same
            v = model.encode([r["text"]], batch_size=1, normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
            emb[r["key"]] = v
            recs.append(dict(key=r["key"], row_len=r["row_len"], st_len=len(st_ids), ids_equal=same, finite=bool(np.isfinite(v).all()), norm=float(np.linalg.norm(v))))
            if (n + 1) % 500 == 0:
                print(f"{n + 1}/{len(rows)} {time.time() - t1:.0f}s", flush=True)
        single_s = time.time() - t1
        # batched probe: the first 64 pair sentences as sentence-transformers batches them (batch 32, padded)
        probe = [r for r in rows if r["group"] == "pair"][:64]
        bt = model.encode([r["text"] for r in probe], batch_size=32, normalize_embeddings=True, convert_to_numpy=True)
        batch_delta = dict(rows=len(probe), max_abs=float(max(np.abs(bt[i] - emb[r["key"]]).max() for i, r in enumerate(probe))),
                           min_cos=float(min(bt[i] @ emb[r["key"]] for i, r in enumerate(probe))))
    for h in handles:
        h.remove()
    np.savez(W / "oracle" / "embeddings.npz", **emb)

    # --- pair cosines and task metrics
    pairs = [p for p in fx["pairs"] if p["a"] in emb and p["b"] in emb]
    pair_cos = {p["key"]: float(emb[p["a"]] @ emb[p["b"]]) for p in pairs}
    metrics = {}
    for lang in ("en", "es", "ja", "zh"):
        ps = [p for p in pairs if p["group"] == "pair" and p["lang"] == lang]
        if ps:
            metrics[f"pawsx_{lang}_ap"] = dict(pairs=len(ps), positives=sum(p["label"] for p in ps), ap=average_precision([pair_cos[p["key"]] for p in ps], [p["label"] for p in ps]),
                                                 mean_cos_pos=float(np.mean([pair_cos[p["key"]] for p in ps if p["label"] == 1])), mean_cos_neg=float(np.mean([pair_cos[p["key"]] for p in ps if p["label"] == 0])))
    sts = [p for p in pairs if p["group"] == "stsb"]
    if sts:
        metrics["stsb_v2_test_cosine_spearman"] = dict(pairs=len(sts), spearman=spearman([pair_cos[p["key"]] for p in sts], [p["score"] for p in sts]),
                                                       card_value=0.811864, card_source="hf/evaluation/known13.json STSBenchmark.v2 test cosine_spearman candidate")
    for g in ("long128", "long256", "long512", "edge512"):
        ps = [p for p in pairs if p["group"] == g]
        if ps:
            metrics[f"{g}_pair_cos"] = dict(pairs=len(ps), mean=float(np.mean([pair_cos[p["key"]] for p in ps])), min=float(min(pair_cos[p["key"]] for p in ps)))
    over = {k: v for k, v in taps.items() if v > FP16_MAX}
    near = {k: v for k, v in taps.items() if FP16_MAX >= v > FP16_MAX / 4}
    headroom = dict(rows=len(rows), fp16_max=FP16_MAX, over_fp16=dict(sorted(over.items(), key=lambda kv: -kv[1])), within_4x_of_fp16=near,
                    worst40=dict(sorted(taps.items(), key=lambda kv: -kv[1])[:40]), all_sites=dict(sorted(taps.items())))
    (W / "oracle" / "fp16_headroom.json").write_text(json.dumps(headroom, indent=1))
    env = dict(python=platform.python_version(), macos=platform.mac_ver()[0], machine=platform.machine(), threads=a.threads,
               **{p: importlib.metadata.version(p) for p in ("torch", "transformers", "sentence-transformers", "tokenizers", "numpy", "safetensors")},
               sdpa_backend="MATH (torch.nn.attention.sdpa_kernel)", attn_implementation=body.config._attn_implementation,
               weights_sha256=(HF / "SHA256SUMS").read_text().split("model.safetensors")[0][-65:-1] if False else None)
    doc = dict(model="vllm-sr/Vela-1.0-Encoder-307M-Embedding", revision=(HF / "REVISION").read_text().strip(),
               pipeline=[type(m).__name__ for m in model], pooling=model[1].get_pooling_mode_str(), normalize=True, dim=int(emb[rows[0]["key"]].shape[0]),
               max_seq_length=model.max_seq_length, prompts=model.prompts, default_prompt_name=model.default_prompt_name,
               config=dict(num_hidden_layers=cfg.num_hidden_layers, hidden_size=cfg.hidden_size, intermediate_size=cfg.intermediate_size, num_attention_heads=cfg.num_attention_heads,
                           vocab_size=cfg.vocab_size, local_attention=cfg.local_attention, global_attn_every_n_layers=cfg.global_attn_every_n_layers,
                           global_rope_theta=cfg.global_rope_theta, local_rope_theta=cfg.local_rope_theta, rope_scaling=cfg.rope_scaling,
                           layer_types={t: layer_types.count(t) for t in set(layer_types)}, pad_token_id=cfg.pad_token_id, bos=cfg.bos_token_id, eos=cfg.eos_token_id,
                           norm_eps=cfg.norm_eps, norm_bias=cfg.norm_bias),
               rope={lt: dict(layer=d["layer"], rope_type=d["rope_type"], attention_scaling=d["attention_scaling"], inv_freq_first=d["inv_freq"][:4].tolist(), inv_freq_last=d["inv_freq"][-4:].tolist()) for lt, d in rope.items()},
               rows=len(recs), ids_equal=ids_equal, finite=sum(r["finite"] for r in recs), seconds_single=round(single_s, 1), ms_per_text=round(1000 * single_s / len(recs), 2),
               batch_probe=batch_delta, pair_cos=pair_cos, metrics=metrics, fp16_headroom=dict(over_fp16=headroom["over_fp16"], within_4x=near),
               env=env, seconds=round(time.time() - t0, 1), per_row=recs)
    (W / "oracle" / "oracle.json").write_text(json.dumps(doc, indent=1))
    print(json.dumps({k: v for k, v in doc.items() if k not in ("pair_cos", "per_row")}, indent=1, default=str))


if __name__ == "__main__":
    main()
