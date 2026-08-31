#!/usr/bin/env python3
"""Convert lightonai/mLateOn (multilingual ModernBERT ColBERT) to LiteRT.

    python convert_mlateon_colbert.py <model_dir_or_id> out/mlateon_colbert

Encoder lane, NOT export_hf. ModernBERT backbone (the bekko/granite-r2 mask
lane) with a pylate late-interaction head: the graph emits ONE 128-d vector PER
TOKEN; MaxSim scoring lives host-side.

Signatures (batch 1, right-padded static lengths):
  encode_{32,128,256,512}: (input_ids i32 [1,S], attention_mask i32 [1,S])
      -> token_embeddings f32 [1,S,128], L2-normalized per token

Model shape (config.json, read 2026-08-31): mmBERT-base ModernBertModel, 22L,
hidden 768, heads 12, GeLU, vocab 256002 (Gemma-style SP + [Q]/[D]), rope theta
160000 both layer types, layer pattern full/sliding/sliding (global every 3rd),
local_attention 128 -> config.sliding_window property = 64 (HALF window,
|q-k| <= 64 inclusive). `position_embedding_type: "sans_pos"` is an ignored
extra key in transformers 5.14.1 — the vendor path (sentence_transformers ->
stock ModernBertModel) applies standard RoPE from rope_parameters, so we do too.

Head (modules.json + pylate.models.Dense, read from pylate main 2026-08-31):
three Dense modules, ALL Identity activation, residual as a PARALLEL bias-free
Linear when dims differ (out = linear(x) + residual(x)):
  1_Dense 768->1536 residual   2_Dense 1536->768 residual   3_Dense 768->128
The whole stack is linear, so it folds EXACTLY into one 128x768 matrix:
  W = W3 @ (W2l + W2r) @ (W1l + W1r)   (folded in float64, gated vs the
module-stack reference below). Then per-token L2 normalize (pylate encode
default normalize_embeddings=True).

Host contract (pylate colbert.py, read 2026-08-31 — NOT the LFM2.5-ColBERT one):
  * pad id = MASK token 4 (pylate: mask if the tokenizer has one, else EOS;
    this tokenizer has one). config.json's pad_token_id 0 is the trap — the
    vendor's own onnx_config.json says 4. With pads masked out (below) the pad
    id is NOT load-bearing here — gated — but hosts should still use 4.
  * queries: `do_query_expansion: false` -> NO expansion, natural length.
    tokenize -> [cls=2, ..., eos=1], insert [Q]=256000 at position 1 (mask
    gets a 1). ALL valid positions are scored. skiplist_words is EMPTY.
  * documents: same with [D]=256001. Keep valid positions only.
  * score = MaxSim over kept vectors (unit length, dot == cosine).

Traps handled here:
  * Sliding-window empty rows NaN (the bekko/granite-r2 wall): a pad query row
    at q >= n_valid + 64 has no valid key inside its +-64 window -> all-masked
    row -> SDPA NaN. There is no in-graph pool to poison here, but the NaN
    sits in the [1,S,128] output and int8 kernels can HIDE it (fp32 NaN, int8
    finite-looking) — so fp32 finiteness over ALL positions is gated, and the
    hand bias allows the diagonal: allowed(q,k) =
      full:    valid[k] | (q==k)
      sliding: (valid[k] | (q==k)) & (|q-k| <= 64)
    The diagonal term is a no-op for valid queries (they attend themselves
    already); it only keeps pad rows finite, and the host drops those.
  * Because pads are masked as keys AND there is no pooling, padded-vs-
    unpadded must agree essentially bitwise on valid rows (unlike
    LFM2.5-ColBERT, where pads feed query expansion and the pad id is
    load-bearing — opposite contracts, both gated in their own lanes).
  * tf5.x meta-load zero check on both rope inv_freq buffers.
  * The card publishes exact MaxSim scores for its usage example
    ([9.6029, 9.5838, 9.5877, 9.4578]); the full host-contract
    reimplementation is gated against them BEFORE tracing.

Quantization: wi8fc (FC int8 DRQ + EMBEDDING_LOOKUP int8 channelwise) + fp16.
The 256002x768 table is ~64% of the 307M params.
"""
import argparse
import collections
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_MODEL = "lightonai/mLateOn"
PAD_ID = 4       # <mask> — pylate's choice; config.json's 0 is wrong for this path
CLS_ID = 2
EOS_ID = 1
Q_ID = 256000    # "[Q] "
D_ID = 256001    # "[D] "
DIM = 128

CARD_QUERY = "Which planet is the Red Planet?"
CARD_DOCS = [
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Mars, connu pour son apparence rougeâtre, est souvent appelé la planète rouge.",
    "Mars, bekannt für sein rötliches Erscheinungsbild, wird oft als der Rote Planet bezeichnet.",
    "Venus is often called Earth's twin because of its similar size and proximity.",
]
CARD_SCORES = [9.6029, 9.5838, 9.5877, 9.4578]  # README, MultiVectorEncoder


class ColbertEncoder(nn.Module):
    """ModernBertModel body -> folded 768->128 projection -> per-token L2 norm."""

    def __init__(self, body, w128x768):
        super().__init__()
        self.body = body
        self.half_window = int(body.config.sliding_window)  # 64 = local_attention//2
        self.proj = nn.Linear(w128x768.shape[1], w128x768.shape[0], bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(w128x768)

    def forward(self, input_ids, attention_mask):
        S = input_ids.shape[1]
        valid = attention_mask.to(torch.bool)  # [1,S]
        zero = torch.zeros((), dtype=torch.float32)
        ninf = torch.full((), float("-inf"), dtype=torch.float32)

        idx = torch.arange(S)
        eye = idx[:, None] == idx[None, :]                       # [S,S] const
        key_ok = valid[:, None, :] | eye[None]                   # [1,S,S]
        full_bias = torch.where(key_ok[:, None], zero, ninf)     # [1,1,S,S]

        window = (idx[:, None] - idx[None, :]).abs() <= self.half_window
        slide_bias = torch.where((key_ok & window[None])[:, None], zero, ninf)

        h = self.body(
            input_ids=input_ids,
            attention_mask={
                "full_attention": full_bias,
                "sliding_attention": slide_bias,
            },
        ).last_hidden_state                                      # [1,S,768]
        return F.normalize(self.proj(h), p=2, dim=-1)            # [1,S,128]


def load_model(model_id):
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        model_id, dtype=torch.float32, attn_implementation="sdpa"
    ).eval()
    assert type(model).__name__ == "ModernBertModel", type(model).__name__
    assert int(model.config.sliding_window) == 64, model.config.sliding_window
    assert model.config.layer_types[0] == "full_attention"
    for name, buf in model.named_buffers():
        if "inv_freq" in name:
            assert float(buf.abs().min()) > 0, f"{name} zeroed by meta-load"
    return model


def load_folded_head(model_id):
    """Fold the three all-linear pylate Dense modules into one 128x768 matrix."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    def dense(d):
        if os.path.isdir(model_id):
            p = os.path.join(model_id, d, "model.safetensors")
            cfgp = os.path.join(model_id, d, "config.json")
        else:
            p = hf_hub_download(model_id, f"{d}/model.safetensors")
            cfgp = hf_hub_download(model_id, f"{d}/config.json")
        cfg = json.load(open(cfgp))
        assert cfg["bias"] is False, cfg
        assert cfg["activation_function"].endswith("Identity"), cfg
        w = load_file(p)
        return cfg, {k: v.to(torch.float64) for k, v in w.items()}

    mats = []
    for d, want_res in (("1_Dense", True), ("2_Dense", True), ("3_Dense", False)):
        cfg, w = dense(d)
        assert cfg["use_residual"] is want_res, (d, cfg)
        m = w["linear.weight"]
        if want_res:
            m = m + w["residual.weight"]  # out = linear(x)+residual(x) = (Wl+Wr)x
        mats.append(m)
        print(f"{d}: {tuple(w['linear.weight'].shape)} residual={want_res}")
    W = (mats[2] @ mats[1] @ mats[0]).to(torch.float32)  # [128,768]
    assert tuple(W.shape) == (DIM, 768), W.shape

    # keep the unfolded stack for the fold gate
    def stack(x64):
        h = mats[0] @ x64
        h = mats[1] @ h
        return (mats[2] @ h).to(torch.float32)

    return W, stack


# ------------------------------------------------------------- host contract
def query_tokens(tok, text):
    e = tok(text)["input_ids"]                    # [cls, ..., eos]
    ids = [e[0], Q_ID] + e[1:]
    return np.array(ids, np.int32)


def doc_tokens(tok, text, max_len=512):
    e = tok(text, truncation=True, max_length=max_len - 1)["input_ids"]
    ids = [e[0], D_ID] + e[1:]
    return np.array(ids[:max_len], np.int32)


def pad_to(ids, S):
    n = min(len(ids), S)
    a = np.full((1, S), PAD_ID, np.int32)
    m = np.zeros((1, S), np.int32)
    a[0, :n] = ids[:n]
    m[0, :n] = 1
    return torch.from_numpy(a), torch.from_numpy(m), n


def maxsim(q, d):
    return float((q @ d.T).max(axis=1).sum())


def doc_sample(S, frac=0.7, seed=1):
    g = torch.Generator().manual_seed(seed)
    n = max(4, int(S * frac))
    ids = torch.full((1, S), PAD_ID, dtype=torch.int32)
    mask = torch.zeros(1, S, dtype=torch.int32)
    ids[0, 0] = CLS_ID
    ids[0, 1] = D_ID
    ids[0, 2:n - 1] = torch.randint(10, 255000, (n - 3,), generator=g,
                                    dtype=torch.int32)
    ids[0, n - 1] = EOS_ID
    mask[0, :n] = 1
    return ids, mask, n


def torch_ref(mod, ids, mask):
    with torch.inference_mode():
        return mod(ids, mask).numpy()


def eager_checks(enc, model, W, stack, model_id):
    print("\n-- eager gates --")
    # (a) the fold is exact
    g = torch.Generator().manual_seed(0)
    x = torch.randn(768, 16, generator=g, dtype=torch.float64)
    d = float((torch.from_numpy(W.numpy() @ x.numpy().astype(np.float32))
               - stack(x)).abs().max())
    print(f"folded head vs 3-module stack: max|diff| {d:.3e}")
    assert d < 1e-4, "Dense fold is not exact — check residual semantics"

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    assert tok.pad_token_id == PAD_ID == tok.mask_token_id
    assert tok.convert_tokens_to_ids("[Q] ") == Q_ID
    assert tok.convert_tokens_to_ids("[D] ") == D_ID

    # (b) hand dict masks vs the vendor stock path (2D mask), padded+unpadded
    text = "Mars, known for its reddish appearance, is often referred to as the Red Planet."
    ids_np = doc_tokens(tok, text)
    n = len(ids_np)
    ids_u = torch.from_numpy(ids_np[None]).to(torch.int32)
    ones = torch.ones(1, n, dtype=torch.int32)
    with torch.inference_mode():
        h = model(input_ids=ids_u, attention_mask=ones).last_hidden_state
        vendor = F.normalize(enc.proj(h), p=2, dim=-1).numpy()[0]
    ours_u = torch_ref(enc, ids_u, ones)[0]
    du = float(np.abs(ours_u - vendor).max())
    ids_p, mask_p, _ = pad_to(ids_np, 128)
    with torch.inference_mode():
        hp = model(input_ids=ids_p, attention_mask=mask_p).last_hidden_state
        vendor_p = F.normalize(enc.proj(hp), p=2, dim=-1).numpy()[0, :n]
    ours_p = torch_ref(enc, ids_p, mask_p)[0, :n]
    dp = float(np.abs(ours_p - vendor_p).max())
    print(f"masks vs vendor 2D path: unpadded max|diff| {du:.3e}, padded {dp:.3e}")
    assert du < 1e-5 and dp < 1e-5, "hand-built masks disagree with the vendor path"

    # (c) NaN wall: short text in a long signature — every position must be
    #     finite in fp32 (this is the gate int8 would fake its way past), and
    #     valid rows must match the unpadded run.
    ids_p, mask_p, _ = pad_to(ids_np, 512)
    v = torch_ref(enc, ids_p, mask_p)[0]
    assert np.isfinite(v).all(), "NaN in fp32 output — the diagonal guard failed"
    dv = float(np.abs(v[:n] - ours_u).max())
    print(f"512-sig all-position finite: OK; valid rows vs unpadded max|diff| {dv:.3e}")
    assert dv < 1e-5

    # (d) pad-content invariance on valid rows (pads are masked keys here)
    ids2 = ids_p.clone()
    g2 = torch.Generator().manual_seed(7)
    ids2[0, n:] = torch.randint(10, 255000, (512 - n,), generator=g2,
                                dtype=torch.int32)
    v2 = torch_ref(enc, ids2, mask_p)[0]
    leak = float(np.abs(v[:n] - v2[:n]).max())
    print(f"pad-content invariance (valid rows): max|diff| {leak:.3e} (must be 0)")
    assert leak == 0.0

    # (e) the card's own example, exact published MaxSim scores
    q = torch_ref(enc, *pad_to(query_tokens(tok, CARD_QUERY), 32)[:2])[0]
    qn = len(query_tokens(tok, CARD_QUERY))
    scores = []
    for doc in CARD_DOCS:
        dt = doc_tokens(tok, doc)
        dv_ = torch_ref(enc, *pad_to(dt, 128)[:2])[0][:len(dt)]
        scores.append(maxsim(q[:qn], dv_))
    err = max(abs(a - b) for a, b in zip(scores, CARD_SCORES))
    print(f"card MaxSim: {[round(s, 4) for s in scores]} (published {CARD_SCORES}) "
          f"max|diff| {err:.4f}")
    assert min(scores[:3]) > scores[3], "Venus outranks a Mars doc"
    assert err < 0.05, "host-contract reimplementation drifts from the card"


def op_report(path, tag):
    from ai_edge_litert.interpreter import Interpreter

    it = Interpreter(model_path=path)
    hist = collections.Counter(d["op_name"] for d in it._get_ops_details())
    print(f"{tag} ops:", dict(sorted(hist.items(), key=lambda kv: -kv[1])))
    i64 = [d["name"] for d in it.get_tensor_details()
           if "int64" in str(d["dtype"]).lower()]
    print(f"{tag} int64 tensors: {len(i64)}")
    assert "GATHER_ND" not in hist, "GATHER_ND entered the graph (mobile-GPU wall)"
    return dict(hist)


def run_sig(path, name, ids, mask):
    from ai_edge_litert.interpreter import Interpreter

    it = Interpreter(model_path=path, num_threads=os.cpu_count())
    r = it.get_signature_runner(name)(input_ids=ids.numpy(),
                                      attention_mask=mask.numpy())
    return list(r.values())[0]


def check_variant(path, tag, enc, seq_lens, results):
    for S in sorted(seq_lens):
        ids, mask, n = doc_sample(S, frac=0.25 if S == 32 else 0.7)
        ref = torch_ref(enc, ids, mask)
        got = run_sig(path, f"encode_{S}", ids, mask)
        assert np.isfinite(got).all(), f"encode_{S} {tag}: non-finite output"
        cos = float((ref[0, :n] * got[0, :n]).sum(-1).min())
        print(f"SMOKE encode_{S} {tag} vs torch: max|diff| "
              f"{np.abs(ref[0, :n] - got[0, :n]).max():.3e} min per-token cos {cos:.6f}")
        results["sigs"].setdefault(f"encode_{S}", {})[f"{tag}_min_cos"] = cos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    ap.add_argument("out_dir", nargs="?", default="out/mlateon_colbert")
    ap.add_argument("--seqs", default="512,256,128,32")
    ap.add_argument("--skip-fp16", action="store_true")
    args = ap.parse_args()

    seq_lens = [int(s) for s in args.seqs.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)
    fp32 = os.path.join(args.out_dir, "colbert_fp32.tflite")
    wi8 = os.path.join(args.out_dir, "colbert_wi8fc.tflite")
    fp16 = os.path.join(args.out_dir, "colbert_fp16.tflite")

    torch.manual_seed(0)
    print(f"loading {args.model} ...")
    model = load_model(args.model)
    W, stack = load_folded_head(args.model)
    enc = ColbertEncoder(model, W).eval()
    n_params = sum(p.numel() for p in enc.parameters())
    print(f"params {n_params/1e6:.1f}M  vocab {model.config.vocab_size}  dim {DIM}")

    eager_checks(enc, model, W, stack, args.model)

    print(f"\nconverting (litert_torch multi-signature) seqs={seq_lens} ...")
    import litert_torch

    conv = None
    for S in seq_lens:
        ids, mask, _ = doc_sample(S, frac=0.25 if S == 32 else 0.7)
        kw = {"input_ids": ids, "attention_mask": mask}
        sig = (f"encode_{S}", enc)
        conv = (litert_torch.signature(*sig, sample_kwargs=kw) if conv is None
                else conv.signature(*sig, sample_kwargs=kw))
    edge = conv.convert()
    edge.export(fp32)
    print(f"fp32: {os.path.getsize(fp32)/1e6:.1f} MB")

    results = {"model": args.model, "params_m": n_params / 1e6, "dim": DIM,
               "seqs": seq_lens, "pad_id": PAD_ID,
               "query_prefix_id": Q_ID, "doc_prefix_id": D_ID,
               "scoring": "maxsim", "normalized": "per-token L2",
               "ops": op_report(fp32, "fp32"), "sigs": {}}
    assert "EMBEDDING_LOOKUP" in results["ops"], (
        "embedding lowered to GATHER, not EMBEDDING_LOOKUP — the int8 recipe "
        "would miss the 256002x768 table")
    check_variant(fp32, "fp32", enc, seq_lens, results)

    print("quantizing wi8fc ...")
    from ai_edge_quantizer import quantizer, recipe_manager, qtyping

    G, OP = qtyping.QuantGranularity, qtyping.TFLOperationName
    rm = recipe_manager.RecipeManager()
    rm.add_dynamic_config(regex=".*", operation_name=OP.FULLY_CONNECTED, num_bits=8)
    rm.add_dynamic_config(regex=".*", operation_name=OP.EMBEDDING_LOOKUP,
                          num_bits=8, granularity=G.CHANNELWISE)
    qt = quantizer.Quantizer(fp32, rm.get_quantization_recipe())
    assert not qt.need_calibration
    qt.quantize().export_model(wi8)
    print(f"wi8fc: {os.path.getsize(wi8)/1e6:.1f} MB")
    check_variant(wi8, "wi8fc", enc, seq_lens, results)

    if not args.skip_fp16:
        print("quantizing fp16 ...")
        rm16 = recipe_manager.RecipeManager()
        rm16.add_quantization_config(
            regex=".*", operation_name=OP.ALL_SUPPORTED,
            algorithm_key="float_casting",
            op_config=qtyping.OpQuantizationConfig(
                weight_tensor_config=qtyping.TensorQuantizationConfig(
                    num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
                compute_precision=qtyping.ComputePrecision.FLOAT))
        quantizer.Quantizer(fp32, rm16.get_quantization_recipe()) \
            .quantize().export_model(fp16)
        print(f"fp16: {os.path.getsize(fp16)/1e6:.1f} MB")
        check_variant(fp16, "fp16", enc, seq_lens, results)

    results["sizes_mb"] = {t: round(os.path.getsize(p) / 1e6, 1)
                           for t, p in (("fp32", fp32), ("wi8fc", wi8), ("fp16", fp16))
                           if os.path.exists(p)}
    with open(os.path.join(args.out_dir, "convert_report.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("DONE:", fp32, wi8)


if __name__ == "__main__":
    main()
