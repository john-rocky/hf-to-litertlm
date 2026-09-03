"""Qwen3.5 MTP drafter alpha-oracle (P0 gate for the speculative-decoding lane).

Measures per-token acceptance of the checkpoint's own 15-tensor `mtp.*` drafter
against the base model's greedy decode, entirely on desktop, no exporter code.

Semantics (verified against sources, 2026-08-31):
  - vLLM qwen3_5_mtp.py: h = fc(cat(pre_fc_norm_embedding(emb(tok)),
    pre_fc_norm_hidden(hidden))) -> Qwen3_5DecoderLayer(full_attention)
    -> mtp.norm -> tied lm_head. `hidden` = base model POST-final-norm hidden
    (Qwen3_5TextModel.forward applies self.norm before returning).
  - Chained steps feed the drafter's own post-mtp.norm output back as `hidden`
    (vllm llm_base_proposer.py: hidden_states = ret_hidden_states).
  - vLLM qwen3_5 uses constant_draft_positions=False (positions increment per
    chained step, drafter keeps its own KV cache, teacher-forced in sync).
    The LiteRT-LM runtime instead holds input_pos FIXED across the G chained
    steps (llm_litert_mtp_drafter.cc; same as vLLM's Gemma4Proposer with
    constant_draft_positions=True). Both are measured here.

Attention-history modes measured:
  inc   - vLLM-faithful ceiling: query rope position p-1+i at chained step i,
          attends teacher KV[0..p-1] + previously drafted K/V of this round.
  fixed - LiteRT-contract: query rope position p-1 at every chained step,
          attends teacher KV[0..p-2] + own in-graph K/V only. Teacher cache =
          what a verify/prefill graph that computes drafter K/V teacher-forced
          would hold (position-addressed, overwrite-rollback compatible).
  local - no history at all: attends only its own position (softmax over one
          key). Cheapest export: no drafter cache buffers in the bundle.

Head variants: fp32 (reference), int8 (per-output-channel symmetric),
int4b32 (block-32 minmax symmetric proxy for the shipped int4 recipe).

Usage:
  python mtp_work/qwen35_mtp_alpha_oracle.py [--model Qwen/Qwen3.5-0.8B]
      [--max-new 256] [--g 3] [--json OUT] [--smoke] [--device mps]
"""

import argparse
import copy
import glob
import json
import os
import time

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5RMSNorm,
    apply_rotary_pos_emb,
    repeat_kv,
)

CHAT_PROMPTS = [
    [{"role": "user", "content": "Why is the sky blue?"}],
    [{"role": "user", "content": "What is the capital of France?"}],
    [{"role": "user", "content": "Explain photosynthesis in one sentence."}],
    [{"role": "user", "content": "Write a short story about a robot learning to paint."}],
    [{"role": "user", "content": "List the first ten prime numbers and briefly explain what makes a number prime."}],
    [{"role": "user", "content": "Explain how to make a cup of tea, step by step."}],
    [
        {"role": "user", "content": "Can you recommend a weekend trip near Tokyo?"},
        {"role": "assistant", "content": "Sure! Hakone is a great choice: hot springs, a lake cruise, and views of Mt. Fuji, all about 90 minutes from Tokyo by train."},
        {"role": "user", "content": "Sounds good. What should I pack for two days there in autumn?"},
    ],
    [
        {"role": "user", "content": "I'm learning Python. What should I study first?"},
        {"role": "assistant", "content": "Start with variables, basic types, control flow, and functions. Then move on to lists and dictionaries, which you will use constantly."},
        {"role": "user", "content": "Show me a simple example that uses a dictionary."},
    ],
]

GSM_PROMPTS = [
    [{"role": "user", "content": "A bakery sells muffins for $3 each. On Monday it sold 24 muffins and on Tuesday it sold 18 more muffins than on Monday. How much money did the bakery make in total over the two days?"}],
    [{"role": "user", "content": "Tom has 5 boxes of pencils. Each box holds 12 pencils. He gives 8 pencils to his friend and then buys 2 more boxes. How many pencils does he have now?"}],
    [{"role": "user", "content": "A train travels at 60 miles per hour for 2 hours, then at 40 miles per hour for 3 hours. How many miles does it travel in total?"}],
    [{"role": "user", "content": "Sara is saving for a bicycle that costs $180. She saves $15 per week and already has $45. How many weeks will it take her to afford the bicycle?"}],
    [{"role": "user", "content": "A farmer has 3 fields. Each field grows 240 carrots. He sells three quarters of all his carrots at the market. How many carrots does he have left?"}],
    [{"role": "user", "content": "James reads 16 pages of a book every night. The book has 384 pages and he has already read 96 pages. How many more nights will it take him to finish the book?"}],
    [{"role": "user", "content": "A school buys 12 packs of markers. Each pack contains 8 markers and costs $6. If the school pays with $100, how much change does it get?"}],
    [{"role": "user", "content": "Lily runs 3 kilometers on every weekday and 5 kilometers on Saturdays. She rests on Sundays. How many kilometers does she run in 4 full weeks?"}],
]

MODES = ("inc", "fixed", "local")


def find_shards(model_id):
    pat = os.path.expanduser(
        f"~/.cache/huggingface/hub/models--{model_id.replace('/', '--')}/snapshots/*/*.safetensors")
    shards = sorted(glob.glob(pat))
    assert shards, f"no safetensors found for {model_id} in HF cache"
    return shards


class Drafter(torch.nn.Module):
    def __init__(self, text_cfg, shards, device):
        super().__init__()
        dcfg = copy.deepcopy(text_cfg)
        dcfg.layer_types = ["full_attention"]
        dcfg.num_hidden_layers = 1
        try:
            dcfg._attn_implementation = "eager"
        except Exception:
            pass
        h = text_cfg.hidden_size
        eps = text_cfg.rms_norm_eps
        self.layer = Qwen3_5DecoderLayer(dcfg, 0)
        self.ne = Qwen3_5RMSNorm(h, eps=eps)
        self.nh = Qwen3_5RMSNorm(h, eps=eps)
        self.nf = Qwen3_5RMSNorm(h, eps=eps)
        self.fc = torch.nn.Linear(2 * h, h, bias=False)

        tensors = {}
        for sh in shards:
            with safe_open(sh, framework="pt") as f:
                for k in f.keys():
                    if k.startswith("mtp."):
                        tensors[k] = f.get_tensor(k).to(torch.float32)
        assert tensors, "no mtp.* tensors found in checkpoint"
        print(f"loaded {len(tensors)} mtp.* tensors")
        self.fc.weight.data.copy_(tensors.pop("mtp.fc.weight"))
        self.ne.weight.data.copy_(tensors.pop("mtp.pre_fc_norm_embedding.weight"))
        self.nh.weight.data.copy_(tensors.pop("mtp.pre_fc_norm_hidden.weight"))
        self.nf.weight.data.copy_(tensors.pop("mtp.norm.weight"))
        layer_sd = {k[len("mtp.layers.0."):]: v for k, v in tensors.items()}
        missing, unexpected = self.layer.load_state_dict(layer_sd, strict=True), None
        self.float().to(device).eval()


@torch.no_grad()
def trace_greedy(model, ids, max_new, eos_ids, device):
    tm = model.model.language_model
    seq = list(ids)
    toks = torch.tensor([ids], device=device)
    cache = None
    hiddens = []
    for step in range(max_new):
        out = tm(input_ids=toks, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        h = out.last_hidden_state
        hiddens.append(h[0].float())
        nxt = int(model.lm_head(h[:, -1]).argmax(-1))
        if nxt in eos_ids:
            break
        seq.append(nxt)
        toks = torch.tensor([[nxt]], device=device)
    H = torch.cat(hiddens, 0)  # [>=len(seq)-1, hidden], H[t] = post-norm hidden at position t
    assert H.shape[0] >= len(seq) - 1
    return seq, H


@torch.no_grad()
def teacher_and_step1(drafter, tm, seq, H, device):
    """Teacher-forced drafter pass over the whole trace.

    Slot q (q = 0..T-2) holds input x_q = fc(cat(ne(emb(seq[q+1])), nh(H[q])))
    at rope position q. Returns per-slot step-1 outputs for causal (hist) and
    self-only (local) masks, plus the post-rope teacher K/V.
    """
    T = len(seq)
    ids_next = torch.tensor(seq[1:], device=device)
    E = tm.embed_tokens(ids_next).float()
    X = drafter.fc(torch.cat([drafter.ne(E), drafter.nh(H[: T - 1])], -1))[None]
    S = X.shape[1]
    pos = torch.arange(S, device=device)[None]
    cos, sin = tm.rotary_emb(X, pos)

    causal = torch.triu(torch.full((1, 1, S, S), float("-inf"), device=device), diagonal=1)
    out = drafter.layer(X, position_embeddings=(cos, sin), attention_mask=causal)
    H1_hist = drafter.nf(out)[0]

    local = torch.full((1, 1, S, S), float("-inf"), device=device)
    idx = torch.arange(S, device=device)
    local[0, 0, idx, idx] = 0.0
    out = drafter.layer(X, position_embeddings=(cos, sin), attention_mask=local)
    H1_local = drafter.nf(out)[0]

    att = drafter.layer.self_attn
    ln = drafter.layer.input_layernorm(X)
    hs = (1, S, -1, att.head_dim)
    k = att.k_norm(att.k_proj(ln).view(hs)).transpose(1, 2)
    v = att.v_proj(ln).view(hs).transpose(1, 2)
    _, k = apply_rotary_pos_emb(k, k, cos, sin)
    return X[0], H1_hist, H1_local, k, v


@torch.no_grad()
def manual_step(drafter, tm, x, qpos, K, V, device):
    """One drafter step for x [1,1,h] with explicit allowed K/V (own k/v appended).

    Returns (post-mtp.norm hidden [h], own post-rope k, own v)."""
    layer = drafter.layer
    att = layer.self_attn
    ln = layer.input_layernorm(x)
    qg = att.q_proj(ln).view(1, 1, -1, att.head_dim * 2)
    q, gate = torch.chunk(qg, 2, dim=-1)
    gate = gate.reshape(1, 1, -1)
    q = att.q_norm(q).transpose(1, 2)
    k = att.k_norm(att.k_proj(ln).view(1, 1, -1, att.head_dim)).transpose(1, 2)
    v = att.v_proj(ln).view(1, 1, -1, att.head_dim).transpose(1, 2)
    pos = torch.tensor([[qpos]], device=device)
    cos, sin = tm.rotary_emb(x, pos)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    Kc = torch.cat([K, k], 2) if K is not None else k
    Vc = torch.cat([V, v], 2) if V is not None else v
    Ke = repeat_kv(Kc, att.num_key_value_groups)
    Ve = repeat_kv(Vc, att.num_key_value_groups)
    w = torch.softmax(((q @ Ke.transpose(2, 3)) * att.scaling).float(), -1)
    a = (w @ Ve).transpose(1, 2).reshape(1, 1, -1)
    a = a * torch.sigmoid(gate)
    h = x + att.o_proj(a)
    h = h + layer.mlp(layer.post_attention_layernorm(h))
    return drafter.nf(h)[0, 0], k, v


def build_heads(W_bf16, device, names):
    """Returns name -> (W_used, ids_map). ids_map=None means full vocab;
    otherwise W_used rows correspond to vocab ids ids_map[row] (top-K by BPE id
    order — a frequency proxy — plus the special-token block at the top)."""
    W = W_bf16.float()
    V = W.shape[0]
    heads = {}
    for name in names:
        if name == "fp32":
            heads[name] = (W, None)
        elif name == "int8":
            s = W.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
            heads[name] = (torch.round(W / s).clamp(-127, 127) * s, None)
        elif name == "int4b32":
            h = W.shape[1]
            blocks = W.view(V, h // 32, 32)
            s = blocks.abs().amax(dim=2, keepdim=True).clamp_min(1e-8) / 7.0
            heads[name] = ((torch.round(blocks / s).clamp(-7, 7) * s).view(V, h),
                           None)
        elif name.startswith("top") and name.endswith("k"):
            K = int(name[3:-1]) * 1024
            ids = torch.cat([torch.arange(K, device=device),
                             torch.arange(247000, V, device=device)])
            heads[name] = (W[ids].contiguous(), ids)
        else:
            raise ValueError(f"unknown head variant {name}")
    return heads


def new_agg(G):
    return {"positions": 0, "hist": [0] * (G + 1),
            "a": [[0, 0] for _ in range(G)]}


def add_chain(agg, c, G):
    agg["positions"] += 1
    agg["hist"][c] += 1
    for i in range(G):
        if i <= c:
            agg["a"][i][1] += 1
            agg["a"][i][0] += 1 if c > i else 0
        else:
            break


def summarize(agg, G):
    n = max(agg["positions"], 1)
    alphas = [num / den if den else 0.0 for num, den in agg["a"]]
    mean_acc = sum((c + 1) * k for c, k in enumerate(agg["hist"])) / n
    # E[accepted tokens per verify round] for smaller G by chain truncation
    e_by_g = {}
    for g in range(1, G + 1):
        e, p = 1.0, 1.0
        for i in range(g):
            p *= alphas[i]
            e += p
        e_by_g[g] = e
    return {"positions": agg["positions"], "alpha": alphas, "hist": agg["hist"],
            "tokens_per_round_G": e_by_g, "mean_acc_chain": mean_acc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--g", type=int, default=3)
    ap.add_argument("--json", default=None)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--heads", default="fp32,int8,int4b32")
    ap.add_argument("--modes", default="inc,fixed,local")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = args.device
    G = args.g
    modes = tuple(args.modes.split(","))
    torch.set_grad_enabled(False)

    tok = AutoTokenizer.from_pretrained(args.model)
    cfg = AutoConfig.from_pretrained(args.model)
    text_cfg = cfg.text_config
    print(f"loading {args.model} (bf16, {device}) ...", flush=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16).to(device).eval()
    tm = model.model.language_model
    drafter = Drafter(text_cfg, find_shards(args.model), device)
    heads = build_heads(tm.embed_tokens.weight.detach(), device,
                        args.heads.split(","))
    print(f"drafter + heads ready ({list(heads)})", flush=True)

    eos_ids = {tok.eos_token_id}
    if text_cfg.eos_token_id is not None:
        eos_ids.add(text_cfg.eos_token_id)

    styles = {"chat": CHAT_PROMPTS, "gsm": GSM_PROMPTS}
    if args.smoke:
        styles = {k: v[:2] for k, v in styles.items()}

    agg = {}  # (style, mode, head) -> agg dict
    for style in styles:
        for mode in modes:
            for hn in heads:
                agg[(style, mode, hn)] = new_agg(G)
    traces_meta = []
    checked_manual = False

    for style, prompts in styles.items():
        for pi, msgs in enumerate(prompts):
            t0 = time.time()
            enc = tok.apply_chat_template(msgs, add_generation_prompt=True)
            if hasattr(enc, "ids"):
                ids = list(enc.ids)
            elif enc and hasattr(enc[0], "ids"):
                ids = list(enc[0].ids)
            else:
                ids = list(enc)
            max_new = 64 if args.smoke else args.max_new
            seq, H = trace_greedy(model, ids, max_new, eos_ids, device)
            n_gen = len(seq) - len(ids)
            snippet = tok.decode(seq[len(ids):len(ids) + 24])
            if n_gen < G + 2:
                print(f"[{style}/{pi}] gen={n_gen} too short, skipped")
                continue
            X, H1h, H1l, K, V = teacher_and_step1(drafter, tm, seq, H, device)

            if not checked_manual:
                for p in (len(ids) + 1, len(ids) + 5, len(seq) - G - 2):
                    href, _, _ = manual_step(
                        drafter, tm, X[p - 1][None, None], p - 1,
                        K[:, :, : p - 1], V[:, :, : p - 1], device)
                    diff = (href - H1h[p - 1]).abs().max().item()
                    assert diff < 2e-3, f"manual-vs-layer mismatch {diff} at p={p}"
                print("manual attention path == layer.forward (checked)")
                checked_manual = True

            # step-1 drafts for all slots, per (h1-kind, head), vectorized
            d1 = {}
            for hn, (W, idmap) in heads.items():
                a = (H1h @ W.T).argmax(-1)
                b = (H1l @ W.T).argmax(-1)
                if idmap is not None:
                    a, b = idmap[a], idmap[b]
                d1[("hist", hn)] = a
                d1[("local", hn)] = b

            p_lo, p_hi = len(ids), len(seq) - 1 - G
            for p in range(p_lo, p_hi + 1):
                refs = seq[p + 1: p + 1 + G]
                for mode in modes:
                    h1kind = "local" if mode == "local" else "hist"
                    H1 = H1l if mode == "local" else H1h
                    for hn, (W, idmap) in heads.items():
                        d = int(d1[(h1kind, hn)][p - 1])
                        c = 0
                        h_prev = H1[p - 1]
                        own_k, own_v = [], []
                        while c < G and d == refs[c]:
                            c += 1
                            if c == G:
                                break
                            # chained step c+1 (0-indexed step i=c)
                            e = tm.embed_tokens(
                                torch.tensor([d], device=device)).float()
                            x = drafter.fc(torch.cat(
                                [drafter.ne(e), drafter.nh(h_prev[None])],
                                -1))[None]
                            if mode == "local":
                                Ku = Vu = None
                                qpos = p - 1
                            elif mode == "fixed":
                                Ku, Vu = K[:, :, : p - 1], V[:, :, : p - 1]
                                qpos = p - 1
                            else:  # inc
                                Ku = torch.cat([K[:, :, :p]] + own_k, 2)
                                Vu = torch.cat([V[:, :, :p]] + own_v, 2)
                                qpos = p - 1 + c
                            h_prev, k1, v1 = manual_step(
                                drafter, tm, x, qpos, Ku, Vu, device)
                            if mode == "inc":
                                own_k.append(k1)
                                own_v.append(v1)
                            j = int((h_prev @ W.T).argmax(-1))
                            d = int(idmap[j]) if idmap is not None else j
                        add_chain(agg[(style, mode, hn)], c, G)

            dt = time.time() - t0
            pos_n = max(p_hi - p_lo + 1, 0)
            a1_fx = agg[(style, "fixed", "fp32")] if "fp32" in heads else None
            print(f"[{style}/{pi} {str(msgs[-1]['content'])[:40]:40s}] "
                  f"gen={n_gen:3d} pos={pos_n:3d} {dt:5.1f}s "
                  f"| {snippet[:48]!r}", flush=True)
            traces_meta.append({"style": style, "idx": pi, "gen": n_gen,
                                "prompt_len": len(ids), "positions": pos_n,
                                "snippet": snippet})

    print("\n=== AGGREGATE (per style x mode x head) ===")
    results = {}
    order_h = list(heads)
    for style in list(styles) + ["ALL"]:
        for mode in modes:
            for hn in order_h:
                if style == "ALL":
                    m = new_agg(G)
                    for s2 in styles:
                        src = agg[(s2, mode, hn)]
                        m["positions"] += src["positions"]
                        for i in range(G + 1):
                            m["hist"][i] += src["hist"][i]
                        for i in range(G):
                            m["a"][i][0] += src["a"][i][0]
                            m["a"][i][1] += src["a"][i][1]
                else:
                    m = agg[(style, mode, hn)]
                s = summarize(m, G)
                results[f"{style}/{mode}/{hn}"] = s
                al = " ".join(f"a{i+1}={a:.3f}" for i, a in enumerate(s["alpha"]))
                eg = " ".join(f"E[G={g}]={e:.2f}"
                              for g, e in s["tokens_per_round_G"].items())
                print(f"{style:5s} {mode:5s} {hn:7s} pos={s['positions']:5d} "
                      f"{al}  {eg}  hist={s['hist']}")

    key = "ALL/fixed/int8" if "int8" in heads else "ALL/fixed/fp32"
    a1 = results[key]["alpha"][0]
    verdict = "GO" if a1 >= 0.7 else ("DROP" if a1 < 0.5 else "GRAY ZONE")
    print(f"\nGATE ({key}): alpha1={a1:.3f} -> {verdict} "
          f"(>=0.7 GO / <0.5 DROP; kill-gate is alpha<~0.5 at G>=2)")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"model": args.model, "G": G, "max_new": args.max_new,
                       "heads": list(heads), "modes": list(modes),
                       "traces": traces_meta, "aggregates": results}, f,
                      indent=2)
        print("wrote", args.json)


if __name__ == "__main__":
    main()
