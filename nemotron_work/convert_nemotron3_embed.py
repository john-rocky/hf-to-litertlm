#!/usr/bin/env python3
"""Convert nvidia/Nemotron-3-Embed-1B-BF16 (bidirectional Ministral3 embedder) to LiteRT.

    python convert_nemotron3_embed.py <model_dir_or_id> out/nemotron3_embed_1b

Encoder lane, NOT export_hf: the model has no KV cache in the embedding path, so
we trace the HF eager model directly with litert_torch multi-signature convert
(same lane as scripts/convert_lfm25_encoder.py).

Signatures (batch 1, right-padded static lengths):
  embed_{64,128,256,512}: (input_ids i32 [1,S], attention_mask i32 [1,S])
      -> embedding f32 [1,2048]  (mean pool over valid positions, then L2 norm)

The pooled+normalized vector is IN the graph: sentence-transformers applies
Pooling(mean, include_prompt=True) then Normalize on top of the bare
`Ministral3Model`, and neither lives in the HF module. Callers must still add
the `query: ` / `passage: ` prefix themselves (it is plain text, tokenized
normally, and included in the mean).

Traps handled here (all verified against the installed sources, tf 5.14.1):
  * `architectures: ["Ministral3Model"]` + `is_causal: false` — stock
    transformers honours it: create_causal_mask() falls back to
    create_bidirectional_mask() when config.is_causal is False, and the flag
    reaches sdpa_attention_forward as an explicit is_causal=False kwarg
    (verified by spying on torch SDPA, and by a causality experiment: changing
    the LAST token moves position 0 by 5.96). `Ministral3Attention.is_causal`
    is hardcoded True and is NOT what decides this — do not "fix" it.
  * BUT create_bidirectional_mask SKIPS the mask entirely when nothing is padded
    (`allow_is_bidirectional_skip` -> sdpa gets attn_mask=None). Under
    torch.export that branch is specialized at trace time, so a sample with an
    all-ones mask would bake a graph that ignores attention_mask and lets pad
    tokens into the attention AND the mean. Every sample here is right-padded,
    and `assert_pad_invariance` gates it: changing pad-region ids must not move
    the embedding at all.
  * pooling/normalize are config-only hints (`pooling: "avg"`, modules.json
    2_Normalize) that the HF class ignores — they are applied here explicitly.
  * tf5.x meta-load can zero init-computed buffers (rope inv_freq) -> assert.
  * llama_4_scaling attention scale is 1 + 0.1*log(1+floor(pos/16384)) = exactly
    1.0 for every position below 16384, so it constant-folds away for these
    sequence lengths. It is NOT dropped — it would matter past 16k.
  * input_ids stay int32 end-to-end (F.embedding accepts int32; no CAST/int64).

Quantization: wi8fc — dynamic-range int8 on FULLY_CONNECTED + int8 channelwise
on the embedding table. fp16 variant is emitted for desktop (the LFM encoder
ship found fp16 jetsams on iOS: XNNPACK dequantizes and packs per signature).
"""
import argparse
import collections
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_MODEL = "nvidia/Nemotron-3-Embed-1B-BF16"
PAD_ID = 11  # <pad>, from config.json / tokenizer_config.json


class Embedder(nn.Module):
    """Ministral3Model body -> mean pool over valid positions -> L2 normalize.

    The attention bias is built here instead of letting transformers build it.
    Feeding a 2D mask routes through `sdpa_mask`, whose index-based construction
    lowers to GATHER_ND -> LOGICAL_AND -> SELECT_V2 in the flatbuffer. Read out
    of an actual export, those constants are: index grid [[0,0],[0,1],...,[0,S-1]]
    (an identity gather at batch 1), an all-True q-side mask, and
    `where(mask, 0.0, -inf)`. So the entire thing is just
        bias[0,0,q,k] = 0.0 if attention_mask[0,k] else -inf,   independent of q
    which we can emit directly. transformers early-exits mask creation for any
    4D mask (`len(attention_mask.shape) == 4` in `_preprocess_mask_arguments`),
    so the hand-built bias is used verbatim.

    Two wins: GATHER_ND is gone (it is a known hard mobile-GPU blocker and the
    only such op that survived here), and the [1,1,S,S] mask is never
    materialized — the [1,1,1,S] bias broadcasts over the query axis inside the
    score add. Verified bitwise identical to the transformers-built path in
    eager, and bitwise identical end-to-end after conversion.
    """

    def __init__(self, body):
        super().__init__()
        self.body = body

    def forward(self, input_ids, attention_mask):
        valid = attention_mask.to(torch.bool)[:, None, None, :]  # [1,1,1,S]
        bias = torch.where(
            valid,
            torch.zeros((), dtype=torch.float32),
            torch.full((), float("-inf"), dtype=torch.float32),
        )
        h = self.body(
            input_ids=input_ids, attention_mask=bias, use_cache=False
        ).last_hidden_state
        m = attention_mask.to(h.dtype)[:, :, None]
        pooled = (h * m).sum(dim=1) / m.sum(dim=1)
        return F.normalize(pooled, p=2, dim=-1)


def load_model(model_id):
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        model_id, dtype=torch.float32, attn_implementation="sdpa"
    ).eval()
    cfg = model.config
    assert type(model).__name__ == "Ministral3Model", type(model).__name__
    assert getattr(cfg, "is_causal", True) is False, (
        "config.is_causal is not False — this would export a CAUSAL graph"
    )
    inv = model.rotary_emb.inv_freq
    assert float(inv.min()) > 0, "rope inv_freq zeroed by meta-load — bad export"
    return model


def sample(S, pad_frac=0.25, seed=None):
    """Right-padded batch-1 sample. ALWAYS has pads (see module docstring)."""
    g = torch.Generator().manual_seed(0 if seed is None else seed)
    n_pad = max(1, int(S * pad_frac))
    n_valid = S - n_pad
    ids = torch.randint(10, 120000, (1, S), generator=g, dtype=torch.int32)
    mask = torch.ones(1, S, dtype=torch.int32)
    ids[0, n_valid:] = PAD_ID
    mask[0, n_valid:] = 0
    return ids, mask, n_valid


def torch_ref(mod, ids, mask):
    with torch.inference_mode():
        return mod(ids, mask).numpy()


def assert_pad_invariance(mod, S=128):
    """Gate: pad-region token ids must not influence the embedding at all.

    This is what catches a graph (or an eager path) that skipped the mask.
    """
    ids, mask, n_valid = sample(S)
    ids2 = ids.clone()
    g = torch.Generator().manual_seed(7)
    ids2[0, n_valid:] = torch.randint(
        10, 120000, (S - n_valid,), generator=g, dtype=torch.int32
    )
    a = torch_ref(mod, ids, mask)
    b = torch_ref(mod, ids2, mask)
    leak = float(np.abs(a - b).max())
    print(f"EAGER pad-content invariance: max|diff| {leak:.3e} (must be 0)")
    assert leak == 0.0, "pad content leaks into the embedding — mask was skipped"

    # And the padded forward must agree with the truly unpadded one.
    c = torch_ref(mod, ids[:, :n_valid], mask[:, :n_valid])
    d = float(np.abs(a - c).max())
    cos = float((a[0] @ c[0]) / (np.linalg.norm(a[0]) * np.linalg.norm(c[0])))
    print(f"EAGER padded-vs-unpadded: max|diff| {d:.3e} cos {cos:.8f} (float noise)")


def op_report(path, tag):
    from ai_edge_litert.interpreter import Interpreter

    it = Interpreter(model_path=path)
    hist = collections.Counter(d["op_name"] for d in it._get_ops_details())
    print(f"{tag} ops:", dict(sorted(hist.items(), key=lambda kv: -kv[1])))
    i64 = [
        d["name"] for d in it.get_tensor_details() if "int64" in str(d["dtype"]).lower()
    ]
    print(f"{tag} int64 tensors: {len(i64)}", i64[:5])
    return hist


def run_sig(path, name, ids, mask):
    from ai_edge_litert.interpreter import Interpreter

    it = Interpreter(model_path=path, num_threads=os.cpu_count())
    r = it.get_signature_runner(name)(
        input_ids=ids.numpy(), attention_mask=mask.numpy()
    )
    return list(r.values())[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    ap.add_argument("out_dir", nargs="?", default="out/nemotron3_embed_1b")
    ap.add_argument("--seqs", default="512,256,128,64")
    ap.add_argument("--skip-fp16", action="store_true")
    args = ap.parse_args()

    seq_lens = [int(s) for s in args.seqs.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)
    fp32 = os.path.join(args.out_dir, "embed_fp32.tflite")
    wi8 = os.path.join(args.out_dir, "embed_wi8fc.tflite")
    fp16 = os.path.join(args.out_dir, "embed_fp16.tflite")

    torch.manual_seed(0)
    print(f"loading {args.model} ...")
    model = load_model(args.model)
    emb = Embedder(model).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params {n_params/1e6:.1f}M  hidden {model.config.hidden_size} "
          f"layers {model.config.num_hidden_layers} vocab {model.config.vocab_size}")

    assert_pad_invariance(emb)

    print(f"converting (litert_torch multi-signature) seqs={seq_lens} ...")
    import litert_torch

    conv = None
    for S in seq_lens:
        ids, mask, _ = sample(S)
        kw = {"input_ids": ids, "attention_mask": mask}
        sig = (f"embed_{S}", emb)
        conv = (
            litert_torch.signature(*sig, sample_kwargs=kw)
            if conv is None
            else conv.signature(*sig, sample_kwargs=kw)
        )
    edge = conv.convert()
    edge.export(fp32)
    print(f"fp32: {os.path.getsize(fp32)/1e6:.1f} MB")
    op_report(fp32, "fp32")

    # fp32 tflite vs torch, on every signature
    results = {"model": args.model, "params_m": n_params / 1e6, "sigs": {}}
    for S in sorted(seq_lens):
        ids, mask, _ = sample(S)
        ref = torch_ref(emb, ids, mask)
        got = run_sig(fp32, f"embed_{S}", ids, mask)
        cos = float(ref[0] @ got[0])
        print(f"SMOKE embed_{S} fp32 vs torch: max|diff| "
              f"{np.abs(ref-got).max():.3e} cos {cos:.8f}")
        results["sigs"][f"embed_{S}"] = {"fp32_cos": cos}

    print("quantizing wi8fc ...")
    from ai_edge_quantizer import quantizer, recipe_manager, qtyping

    G = qtyping.QuantGranularity
    OP = qtyping.TFLOperationName
    rm = recipe_manager.RecipeManager()
    rm.add_dynamic_config(regex=".*", operation_name=OP.FULLY_CONNECTED, num_bits=8)
    rm.add_dynamic_config(
        regex=".*", operation_name=OP.EMBEDDING_LOOKUP, num_bits=8,
        granularity=G.CHANNELWISE,
    )
    qt = quantizer.Quantizer(fp32, rm.get_quantization_recipe())
    assert not qt.need_calibration
    qt.quantize().export_model(wi8)
    print(f"wi8fc: {os.path.getsize(wi8)/1e6:.1f} MB")

    for S in sorted(seq_lens):
        ids, mask, _ = sample(S)
        ref = torch_ref(emb, ids, mask)
        got = run_sig(wi8, f"embed_{S}", ids, mask)
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
                    num_bits=16, dtype=qtyping.TensorDataType.FLOAT
                ),
                compute_precision=qtyping.ComputePrecision.FLOAT,
            ),
        )
        qt16 = quantizer.Quantizer(fp32, rm16.get_quantization_recipe())
        qt16.quantize().export_model(fp16)
        print(f"fp16: {os.path.getsize(fp16)/1e6:.1f} MB")
        for S in sorted(seq_lens):
            ids, mask, _ = sample(S)
            ref = torch_ref(emb, ids, mask)
            got = run_sig(fp16, f"embed_{S}", ids, mask)
            cos = float(ref[0] @ got[0])
            print(f"SMOKE embed_{S} fp16 vs torch: cos {cos:.8f}")
            results["sigs"][f"embed_{S}"]["fp16_cos"] = cos

    with open(os.path.join(args.out_dir, "convert_report.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("DONE:", fp32, wi8)


if __name__ == "__main__":
    main()
