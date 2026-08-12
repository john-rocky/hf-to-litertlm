#!/usr/bin/env python3
"""Convert ibm-granite/granite-embedding-*-multilingual-r2 (ModernBERT) to LiteRT.

    python convert_granite_embedding_r2.py <model_dir_or_id> out/granite_embed_311m

Encoder lane, NOT export_hf: no KV cache, so the HF eager model is traced
directly with litert_torch multi-signature convert (same lane as
convert_lfm25_encoder.py / convert_nemotron3_embed.py).

Signatures (batch 1, right-padded static lengths):
  embed_{64,128,256,512}: (input_ids i32 [1,S], attention_mask i32 [1,S])
      -> embedding f32 [1,768]  (CLS token, then L2 normalize)

Pooling is **CLS, not mean** (`1_Pooling/config.json` has
pooling_mode_cls_token: true) and there are no prompt prefixes
(`config_sentence_transformers.json` prompts are empty strings) — both differ
from the Nemotron embedder, so do not carry that recipe over blind.

ModernBERT specifics handled here:
  * Alternating attention: 22 layers, every 3rd is `full_attention`, the rest
    are `sliding_attention` with a 64-wide half-window, and each type has its
    OWN rope frequency set (global theta 150000 / local 160000). The model
    reads `config.layer_types`; nothing to do beyond leaving it alone.
  * `ModernBertModel.forward` accepts `attention_mask` as a **dict** of
    pre-built masks and uses it verbatim, so we build both masks ourselves and
    never touch transformers' `sdpa_mask`. That machinery is what emitted
    GATHER_ND (a mobile-GPU-hostile op) on the Nemotron export, and under
    torch.export its "skip the mask when nothing is padded" branch specializes
    at trace time — which would bake a graph that ignores attention_mask.
    Both hazards disappear when the masks are explicit.
    Semantics verified bitwise against transformers at S=128 and S=512:
        full_attention    = pad mask, broadcast over queries
        sliding_attention = pad mask AND |q - k| <= 64
  * Model is explicitly non-causal (`ModernBertAttention.is_causal = False`).
  * tf5.x meta-load can zero init-computed buffers -> assert every inv_freq
    buffer (there are two sets here, one per layer type) is > 0 after load.

Quantization: wi8fc — dynamic-range int8 on FULLY_CONNECTED + int8 channelwise
on the embedding table. The 262152x768 vocab table is ~65% of all parameters
here, so reaching it matters more than usual.
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

DEFAULT_MODEL = "ibm-granite/granite-embedding-311m-multilingual-r2"
PAD_ID = 0
NEG = float("-inf")


class Embedder(nn.Module):
    """ModernBertModel -> CLS token -> L2 normalize, with hand-built masks."""

    def __init__(self, body, window):
        super().__init__()
        self.body = body
        self.window = window

    def forward(self, input_ids, attention_mask):
        S = input_ids.shape[1]
        valid = attention_mask.to(torch.bool)[:, None, None, :]      # [1,1,1,S]
        zero = torch.zeros((), dtype=torch.float32)
        neg = torch.full((), NEG, dtype=torch.float32)
        full = torch.where(valid, zero, neg)                          # [1,1,1,S]
        q = torch.arange(S, device=input_ids.device)[:, None]
        k = torch.arange(S, device=input_ids.device)[None, :]
        band = ((q - k).abs() <= self.window)[None, None]             # [1,1,S,S]
        # Self-attention is always allowed, which matters only for PAD query
        # rows: once the padded region is further than the 64-wide window from
        # every valid token (q >= n_valid + 64, i.e. 64 rows at S=512), the
        # sliding row `valid & band` is entirely False and softmax runs over
        # all -inf. Eager PyTorch SDPA absorbs that, the lowered SOFTMAX does
        # not — it produced NaN in the fp32/fp16 graphs at embed_512 while int8
        # silently hid it. Adding the diagonal cannot change any observable
        # output: for a valid query k=q is already allowed, and pad rows never
        # reach a valid row because pad KEYS stay masked in every layer.
        # Verified bitwise identical on the CLS output, with the NaN gone.
        eye = (q == k)[None, None]
        sliding = torch.where((valid & band) | eye, zero, neg)        # [1,1,S,S]
        h = self.body(
            input_ids=input_ids,
            attention_mask={"full_attention": full, "sliding_attention": sliding},
        ).last_hidden_state
        return F.normalize(h[:, 0], p=2, dim=-1)                      # CLS pooling


def load_model(model_id):
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        model_id, dtype=torch.float32, attn_implementation="sdpa"
    ).eval()
    cfg = model.config
    assert type(model).__name__ == "ModernBertModel", type(model).__name__
    n_inv = 0
    for name, buf in model.named_buffers():
        if "inv_freq" in name:
            n_inv += 1
            assert float(buf.min()) > 0, f"{name} zeroed by meta-load — bad export"
    assert n_inv >= 2, f"expected per-layer-type rope buffers, found {n_inv}"
    print(f"layer_types: {collections.Counter(cfg.layer_types)}  "
          f"sliding_window={cfg.sliding_window}  cls_token_id={cfg.cls_token_id}")
    return model, cfg.sliding_window


def sample(S, pad_frac=0.25, seed=None, cls_id=2):
    """Right-padded batch-1 sample. ALWAYS has pads (see module docstring)."""
    g = torch.Generator().manual_seed(0 if seed is None else seed)
    n_pad = max(1, int(S * pad_frac))
    n_valid = S - n_pad
    ids = torch.randint(10, 260000, (1, S), generator=g, dtype=torch.int32)
    ids[0, 0] = cls_id
    mask = torch.ones(1, S, dtype=torch.int32)
    ids[0, n_valid:] = PAD_ID
    mask[0, n_valid:] = 0
    return ids, mask, n_valid


def torch_ref(mod, ids, mask):
    with torch.inference_mode():
        return mod(ids, mask).numpy()


def assert_pad_invariance(mod, cls_id, S=128):
    ids, mask, n_valid = sample(S, cls_id=cls_id)
    ids2 = ids.clone()
    g = torch.Generator().manual_seed(7)
    ids2[0, n_valid:] = torch.randint(10, 260000, (S - n_valid,), generator=g,
                                      dtype=torch.int32)
    a, b = torch_ref(mod, ids, mask), torch_ref(mod, ids2, mask)
    leak = float(np.abs(a - b).max())
    print(f"EAGER pad-content invariance: max|diff| {leak:.3e} (must be 0)")
    assert leak == 0.0, "pad content reaches the CLS token — masks are wrong"
    c = torch_ref(mod, ids[:, :n_valid], mask[:, :n_valid])
    print(f"EAGER padded-vs-unpadded: max|diff| {np.abs(a-c).max():.3e} "
          f"cos {float(a[0]@c[0]):.8f} (float noise)")


def op_report(path, tag):
    from ai_edge_litert.interpreter import Interpreter

    it = Interpreter(model_path=path)
    hist = collections.Counter(d["op_name"] for d in it._get_ops_details())
    print(f"{tag} ops:", dict(sorted(hist.items(), key=lambda kv: -kv[1])))
    i64 = [d["name"] for d in it.get_tensor_details()
           if "int64" in str(d["dtype"]).lower()]
    print(f"{tag} int64 tensors: {len(i64)}", i64[:5])
    for hostile in ("GATHER_ND", "LOGICAL_AND"):
        if hostile in hist:
            print(f"  note: {hostile} x{hist[hostile]} present")


def run_sig(path, name, ids, mask):
    from ai_edge_litert.interpreter import Interpreter

    it = Interpreter(model_path=path, num_threads=os.cpu_count())
    r = it.get_signature_runner(name)(input_ids=ids.numpy(),
                                      attention_mask=mask.numpy())
    return list(r.values())[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    ap.add_argument("out_dir", nargs="?", default="out/granite_embed_311m")
    ap.add_argument("--seqs", default="512,256,128,64")
    ap.add_argument("--skip-fp16", action="store_true")
    args = ap.parse_args()

    seq_lens = [int(s) for s in args.seqs.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)
    fp32 = os.path.join(args.out_dir, "embed_fp32.tflite")
    wi8 = os.path.join(args.out_dir, "embed_wi8fc.tflite")
    fp16 = os.path.join(args.out_dir, "embed_fp16.tflite")
    # ai_edge_quantizer refuses to overwrite, which on a re-run leaves STALE
    # quantized files sitting next to a freshly exported fp32 — they then get
    # "verified" and shipped. Clear them up front so a re-run is idempotent.
    for stale in (wi8, fp16):
        if os.path.exists(stale):
            print(f"removing stale {os.path.basename(stale)}")
            os.remove(stale)

    torch.manual_seed(0)
    print(f"loading {args.model} ...")
    model, window = load_model(args.model)
    cls_id = model.config.cls_token_id
    emb = Embedder(model, window).eval()
    n_params = sum(p.numel() for p in model.parameters())
    vocab_params = model.config.vocab_size * model.config.hidden_size
    print(f"params {n_params/1e6:.1f}M (vocab table "
          f"{vocab_params/1e6:.1f}M = {100*vocab_params/n_params:.0f}%)")

    assert_pad_invariance(emb, cls_id)

    print(f"converting (litert_torch multi-signature) seqs={seq_lens} ...")
    import litert_torch

    conv = None
    for S in seq_lens:
        ids, mask, _ = sample(S, cls_id=cls_id)
        kw = {"input_ids": ids, "attention_mask": mask}
        sig = (f"embed_{S}", emb)
        conv = (litert_torch.signature(*sig, sample_kwargs=kw) if conv is None
                else conv.signature(*sig, sample_kwargs=kw))
    edge = conv.convert()
    edge.export(fp32)
    print(f"fp32: {os.path.getsize(fp32)/1e6:.1f} MB")
    op_report(fp32, "fp32")

    results = {"model": args.model, "params_m": n_params / 1e6, "sigs": {}}
    for S in sorted(seq_lens):
        ids, mask, _ = sample(S, cls_id=cls_id)
        ref = torch_ref(emb, ids, mask)
        got = run_sig(fp32, f"embed_{S}", ids, mask)
        # A fully-masked sliding row makes the lowered SOFTMAX emit NaN where
        # eager PyTorch absorbs it — and int8 hides it behind a healthy-looking
        # cosine. Assert finiteness rather than trusting a cosine that would
        # just read "nan".
        assert np.isfinite(got).all(), f"embed_{S} fp32 output is not finite"
        cos = float(ref[0] @ got[0])
        print(f"SMOKE embed_{S} fp32 vs torch: max|diff| "
              f"{np.abs(ref-got).max():.3e} cos {cos:.8f}")
        results["sigs"][f"embed_{S}"] = {"fp32_cos": cos}

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
    for S in sorted(seq_lens):
        ids, mask, _ = sample(S, cls_id=cls_id)
        ref = torch_ref(emb, ids, mask)
        got = run_sig(wi8, f"embed_{S}", ids, mask)
        assert np.isfinite(got).all(), f"embed_{S} wi8fc output is not finite"
        cos = float(ref[0] @ got[0])
        print(f"SMOKE embed_{S} wi8fc vs torch: cos {cos:.6f}")
        results["sigs"][f"embed_{S}"]["wi8fc_cos"] = cos

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
        for S in sorted(seq_lens):
            ids, mask, _ = sample(S, cls_id=cls_id)
            ref = torch_ref(emb, ids, mask)
            got = run_sig(fp16, f"embed_{S}", ids, mask)
            assert np.isfinite(got).all(), f"embed_{S} fp16 output is not finite"
            cos = float(ref[0] @ got[0])
            print(f"SMOKE embed_{S} fp16 vs torch: cos {cos:.8f}")
            results["sigs"][f"embed_{S}"]["fp16_cos"] = cos

    with open(os.path.join(args.out_dir, "convert_report.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("DONE:", fp32, wi8)


if __name__ == "__main__":
    main()
