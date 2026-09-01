#!/usr/bin/env python3
"""Convert mixedbread-ai/mxbai-edge-colbert-v0-32m (ModernBERT ColBERT) to LiteRT.

    python convert_mxbai_colbert.py <model_dir_or_id> out_mxbai

Encoder lane, NOT export_hf. The smallest ColBERT we ship: ettin-32m backbone
(ModernBertModel 10L/384h/6heads, intermediate 576, vocab 50370 = base 50368 +
[Q]/[D]) with a pylate Dense x3 head. The graph emits ONE 64-d vector PER
TOKEN; MaxSim scoring and skiplist filtering live host-side.

Signatures (batch 1, right-padded static lengths):
  encode_{48,128,256,512}: (input_ids i32 [1,S], attention_mask i32 [1,S])
      -> token_embeddings f32 [1,S,64], L2-normalized per token

Model shape (config.json, read 2026-09-01): local_attention 128 ->
config.sliding_window = 64 (HALF window, inclusive), global attention every
3rd layer starting at 0, BOTH rope thetas 160000.0 (explicit in config —
NOT ModernBERT's default local 10000). `position_embedding_type: "sans_pos"`
is an ignored extra key, same as mLateOn/ettin.

Head (modules.json + pylate.models.Dense, wheel source read 2026-09-01):
three pylate Dense modules, ALL Identity activation, use_residual false,
bias false: 384->768 -> 768->768 -> 768->64. Entirely linear => folds
EXACTLY into one 64x384 matrix W = W3 @ W2 @ W1 (float64 fold, gated).
Then per-token L2 normalize (pylate normalize_embeddings default).

Host contract (pylate colbert.py wheel + repo onnx_config.json, read
2026-09-01 — the mLateOn-shaped contract but with three differences):
  * pad id = MASK token 50284 (pylate: mask if present). config.json's
    pad_token_id 50283 is the trap; the vendor's own onnx_config.json says
    50284. Pads are masked keys -> not load-bearing, but hosts use 50284.
  * do_lower_case TRUE (sentence_bert_config.json): texts are .lower()ed
    BEFORE tokenizing. mLateOn/ettin do not do this — per-model contract.
  * skiplist is NOT empty (unlike mLateOn): 32 punctuation chars, converted
    to bare token ids. Applied to DOCUMENTS only, host-side, AFTER encoding:
    doc kept-mask = attention_mask & ~in_skiplist(ids). Specials
    (CLS/[D]/SEP) are kept. Queries keep every valid position
    (do_query_expansion false -> natural length, no expansion).
  * queries: tokenizer truncates to query_length-1 = 47 (incl CLS+SEP),
    then [Q]=50368 inserted at position 1. documents: truncate to
    document_length-1 = 511, then [D]=50369 at position 1.
  * score = MaxSim over kept vectors (unit length, dot == cosine).

Anchors (README usage, MultiVectorEncoder, printed values):
  query "Which planet is known as the Red Planet?" -> (12, 64)
  documents[0] (Venus) -> (18, 64)
  MaxSim scores [11.2081, 11.5308, 11.4104, 11.4756] (Mars > Saturn >
  Jupiter > Venus — margins are small at 32M, which is exactly why the
  quantization A/B below matters more here than on any other encoder ship).

Traps handled (same walls as mLateOn/bekko):
  * Sliding-window empty-row NaN -> diagonal-allow hand bias, fp32
    all-position finiteness gated.
  * Masks are a hand-built dict {"full_attention", "sliding_attention"}
    [1,1,S,S], gated vs the vendor 2D-mask path padded AND unpadded.
  * Pad-content invariance gated (pads are masked keys, no pooling).
  * tf5.x meta-load zero check on rope inv_freq buffers.

Quantization: wi8fc (FC int8 DRQ + EMBEDDING_LOOKUP int8 channelwise) + fp16.
At 32M params / 64-d vectors the int8 headroom is the smallest of any encoder
ship — if wi8fc shows task-level damage, fp16 (~66 MB) is the honest ship.
"""
import argparse
import collections
import json
import os
import string
import sys


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_MODEL = "mixedbread-ai/mxbai-edge-colbert-v0-32m"
PAD_ID = 50284   # [MASK] — pylate's choice; config.json's 50283 is the trap
CLS_ID = 50281
SEP_ID = 50282
Q_ID = 50368     # "[Q] "
D_ID = 50369     # "[D] "
DIM = 64
QUERY_LEN = 48   # config_sentence_transformers.json query_length
DOC_LEN = 512    # document_length

CARD_QUERY = "Which planet is known as the Red Planet?"
CARD_DOCS = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
]
CARD_SCORES = [11.2081, 11.5308, 11.4104, 11.4756]  # README, MultiVectorEncoder
CARD_Q_VECS = 12   # printed query_embeddings.shape
CARD_D0_VECS = 18  # printed document_embeddings[0].shape (Venus doc)


class ColbertEncoder(nn.Module):
    """ModernBertModel body -> folded 384->64 projection -> per-token L2 norm."""

    def __init__(self, body, w64x384):
        super().__init__()
        self.body = body
        self.half_window = int(body.config.sliding_window)  # 64 = local_attention//2
        self.proj = nn.Linear(w64x384.shape[1], w64x384.shape[0], bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(w64x384)

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
        ).last_hidden_state                                      # [1,S,384]
        return F.normalize(self.proj(h), p=2, dim=-1)            # [1,S,64]


def load_model(model_id):
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        model_id, dtype=torch.float32, attn_implementation="sdpa"
    ).eval()
    assert type(model).__name__ == "ModernBertModel", type(model).__name__
    cfg = model.config
    assert cfg.num_hidden_layers == 10 and cfg.hidden_size == 384
    assert cfg.vocab_size == 50370
    assert int(cfg.sliding_window) == 64, cfg.sliding_window
    assert cfg.layer_types[0] == "full_attention"
    assert cfg.layer_types[1] == "sliding_attention"
    for lt, p in cfg.rope_parameters.items():
        assert p["rope_theta"] == 160000.0, (lt, p)  # local is NOT the 10k default
    for name, buf in model.named_buffers():
        if "inv_freq" in name:
            assert float(buf.abs().min()) > 0, f"{name} zeroed by meta-load"
    return model


def load_folded_head(model_id):
    """Fold the three all-linear pylate Dense modules into one 64x384 matrix."""
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
        assert cfg["use_residual"] is False, cfg
        assert cfg["activation_function"].endswith("Identity"), cfg
        w = load_file(p)
        return {k: v.to(torch.float64) for k, v in w.items()}["linear.weight"]

    mats = [dense(d) for d in ("1_Dense", "2_Dense", "3_Dense")]
    assert tuple(mats[0].shape) == (768, 384)
    assert tuple(mats[1].shape) == (768, 768)
    assert tuple(mats[2].shape) == (64, 768)
    W = (mats[2] @ mats[1] @ mats[0]).to(torch.float32)  # [64,384]

    def stack(x64):
        return (mats[2] @ (mats[1] @ (mats[0] @ x64))).to(torch.float32)

    return W, stack


# ------------------------------------------------------------- host contract
def query_tokens(tok, text):
    e = tok(text.lower(), truncation=True, max_length=QUERY_LEN - 1)["input_ids"]
    ids = [e[0], Q_ID] + e[1:]
    return np.array(ids, np.int32)


def doc_tokens(tok, text, max_len=DOC_LEN):
    e = tok(text.lower(), truncation=True, max_length=max_len - 1)["input_ids"]
    ids = [e[0], D_ID] + e[1:]
    return np.array(ids[:max_len], np.int32)


def skiplist_ids(tok):
    return {tok.convert_tokens_to_ids(w) for w in string.punctuation}


def doc_keep(ids_np, skip):
    return np.array([i not in skip for i in ids_np], bool)


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
    ids[0, 2:n - 1] = torch.randint(1000, 50000, (n - 3,), generator=g,
                                    dtype=torch.int32)
    ids[0, n - 1] = SEP_ID
    mask[0, :n] = 1
    return ids, mask, n


def torch_ref(mod, ids, mask):
    with torch.inference_mode():
        return mod(ids, mask).numpy()


def eager_checks(enc, model, W, stack, model_id):
    print("\n-- eager gates --")
    # (a) the fold is exact
    g = torch.Generator().manual_seed(0)
    x = torch.randn(384, 16, generator=g, dtype=torch.float64)
    d = float((torch.from_numpy(W.numpy() @ x.numpy().astype(np.float32))
               - stack(x)).abs().max())
    print(f"folded head vs 3-module stack: max|diff| {d:.3e}")
    assert d < 1e-4, "Dense fold is not exact"

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    assert tok.mask_token_id == PAD_ID
    assert tok.convert_tokens_to_ids("[Q] ") == Q_ID
    assert tok.convert_tokens_to_ids("[D] ") == D_ID
    skip = skiplist_ids(tok)
    assert len(skip) == 32 and None not in skip

    # (b) hand dict masks vs the vendor stock path (2D mask), padded+unpadded
    ids_np = doc_tokens(tok, CARD_DOCS[1])
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
    assert du < 1e-5 and dp < 1e-5

    # (c) NaN wall: short doc in the 512 signature, all positions finite
    ids_p, mask_p, _ = pad_to(ids_np, 512)
    v = torch_ref(enc, ids_p, mask_p)[0]
    assert np.isfinite(v).all(), "NaN in fp32 output — diagonal guard failed"
    dv = float(np.abs(v[:n] - ours_u).max())
    print(f"512-sig all-position finite: OK; valid rows vs unpadded max|diff| {dv:.3e}")
    assert dv < 1e-5

    # (d) pad-content invariance on valid rows
    ids2 = ids_p.clone()
    g2 = torch.Generator().manual_seed(7)
    ids2[0, n:] = torch.randint(1000, 50000, (512 - n,), generator=g2,
                                dtype=torch.int32)
    v2 = torch_ref(enc, ids2, mask_p)[0]
    leak = float(np.abs(v[:n] - v2[:n]).max())
    print(f"pad-content invariance (valid rows): max|diff| {leak:.3e} (must be 0)")
    assert leak == 0.0

    # (e) the card's own example: shapes AND scores
    q_ids = query_tokens(tok, CARD_QUERY)
    assert len(q_ids) == CARD_Q_VECS, (
        f"query kept-vector count {len(q_ids)} != printed {CARD_Q_VECS}")
    q = torch_ref(enc, *pad_to(q_ids, QUERY_LEN)[:2])[0][:len(q_ids)]
    scores, kept_counts = [], []
    for doc in CARD_DOCS:
        dt = doc_tokens(tok, doc)
        keep = doc_keep(dt, skip)
        kept_counts.append(int(keep.sum()))
        dv_ = torch_ref(enc, *pad_to(dt, 128)[:2])[0][:len(dt)][keep]
        scores.append(maxsim(q, dv_))
    assert kept_counts[0] == CARD_D0_VECS, (
        f"doc0 kept-vector count {kept_counts[0]} != printed {CARD_D0_VECS}")
    err = max(abs(a - b) for a, b in zip(scores, CARD_SCORES))
    print(f"card MaxSim: {[round(s, 4) for s in scores]} (published {CARD_SCORES}) "
          f"max|diff| {err:.4f}  kept {kept_counts}")
    order = np.argsort(scores)[::-1].tolist()
    assert order == [1, 3, 2, 0], f"rank order broke: {order}"
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
        ids, mask, n = doc_sample(S, frac=0.25 if S == 48 else 0.7)
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
    ap.add_argument("out_dir", nargs="?", default="out_mxbai")
    ap.add_argument("--seqs", default="512,256,128,48")
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
        ids, mask, _ = doc_sample(S, frac=0.25 if S == 48 else 0.7)
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
               "query_len": QUERY_LEN, "doc_len": DOC_LEN,
               "do_lower_case": True,
               "scoring": "maxsim over kept vectors (doc: skiplist+pad filtered)",
               "normalized": "per-token L2",
               "ops": op_report(fp32, "fp32"), "sigs": {}}
    assert "EMBEDDING_LOOKUP" in results["ops"], (
        "embedding lowered to GATHER, not EMBEDDING_LOOKUP")
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
                           for t, p in (("fp32", fp32), ("wi8", wi8), ("fp16", fp16))
                           if os.path.exists(p)}
    with open(os.path.join(args.out_dir, "convert_report.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("DONE:", fp32, wi8)


if __name__ == "__main__":
    main()
