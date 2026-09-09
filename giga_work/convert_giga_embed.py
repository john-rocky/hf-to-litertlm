#!/usr/bin/env python3
"""Convert ai-sage/Giga-Embeddings-instruct-480M-0826 (bidirectional Qwen3 embedder) to LiteRT.

    python convert_giga_embed.py [model_dir_or_id] out/giga_embed_480m

Encoder lane, NOT export_hf (same lane as nemotron_work/convert_nemotron3_embed.py): there is no
KV cache in the embedding path, so the HF eager model is traced directly with litert_torch
multi-signature convert.

Signatures (batch 1, right-padded static lengths):
  embed_{64,128,256,512}: (input_ids i32 [1,S], attention_mask i32 [1,S])
      -> embedding f32 [1,1024]  (mean over valid positions, then L2 norm)

Model facts (config.json / tokenizer.json / 1_Pooling/config.json, read 2026-09-09):
  * `architectures: ["Qwen3BidirectionalModel"]`, remote code `modeling_gigarembed.py`: a thin
    `Qwen3Model` subclass that sets every layer's `self_attn.is_causal = False` and builds its
    mask with transformers' `create_bidirectional_mask`. 28 layers, hidden 1024, 16/8 heads,
    head_dim 64, every layer `full_attention`, `sliding_window: null`, vocab 128256, 484M params.
  * `tokenizer.json` post-processor is TemplateProcessing `<s> A </s>`: the host tokenizer adds
    bos (id 1) AND eos (id 2); both carry attention_mask 1 and both are inside the mean.
    `pad_token` is also `</s>` (id 2) — pad and eos share an id; the MASK separates them.
  * Pooling = mean (`include_prompt: true`), then Normalize. Neither lives in the HF module,
    both are folded into the graph here.
  * Prompts are plain text: queries get `Instruct: {task}\\nQuery: ` (the `query` prompt in
    config_sentence_transformers.json), documents nothing. Tokenized normally, in the mean.

Traps handled (verified against the installed sources, tf 5.14.1, and the remote code):
  * The vendor forward passes `use_cache=False` to every decoder layer itself and forwards
    `**kwargs` next to it — passing `use_cache` into the model call is a duplicate keyword.
  * `create_bidirectional_mask` returns None when nothing is padded; under torch.export that
    branch specializes and the graph would ignore attention_mask (Nemotron lesson). Every
    trace sample is right-padded and pad-content invariance is gated. The model is handed a
    4-D additive bias `[1,1,1,S]` built from attention_mask (the 4-D early exit returns it
    verbatim), which also keeps `sdpa_mask`'s GATHER_ND out of the flatbuffer; it is gated
    against the vendor's own 2-D-mask path.
  * Bidirectionality is gated by experiment (changing a late valid token must move position 0),
    not by reading a flag.
  * tf5.x meta-load can zero init-computed buffers (rope inv_freq) -> asserted.

Quantization: wi8fc = FC int8 dynamic-range + EMBEDDING_LOOKUP int8 channelwise. fp16 variant
is emitted for desktop.
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

DEFAULT_MODEL = "ai-sage/Giga-Embeddings-instruct-480M-0826"
BOS_ID = 1      # <s>   (tokenizer post-processor prepends it)
EOS_ID = 2      # </s>  (post-processor appends it)
PAD_ID = 2      # </s>  is also the pad token (config.json pad_token_id)
VOCAB = 128000  # base BPE vocab; ids >= 128000 are added/special tokens
HIDDEN = 1024


def make_bias(attention_mask):
    """[1,S] int mask -> [1,1,1,S] additive bias (0 keep / -inf pad), broadcast over queries."""
    valid = attention_mask.to(torch.bool)[:, None, None, :]
    return torch.where(valid, torch.zeros((), dtype=torch.float32),
                       torch.full((), float("-inf"), dtype=torch.float32))


class Embedder(nn.Module):
    """Qwen3BidirectionalModel body -> mean pool over valid positions -> L2 normalize."""

    def __init__(self, body):
        super().__init__()
        self.body = body

    def forward(self, input_ids, attention_mask):
        h = self.body(input_ids=input_ids, attention_mask=make_bias(attention_mask)).last_hidden_state
        m = attention_mask.to(h.dtype)[:, :, None]
        pooled = (h * m).sum(dim=1) / m.sum(dim=1)
        return F.normalize(pooled, p=2, dim=-1)


def load_model(model_id):
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        model_id, trust_remote_code=True, dtype=torch.float32, attn_implementation="sdpa"
    ).eval()
    cfg = model.config
    assert type(model).__name__ == "Qwen3BidirectionalModel", type(model).__name__
    assert getattr(cfg, "is_causal", True) is False, "config.is_causal is not False"
    assert all(not layer.self_attn.is_causal for layer in model.layers), "a layer is still causal"
    assert all(t == "full_attention" for t in cfg.layer_types), cfg.layer_types
    assert cfg.sliding_window is None and not cfg.use_sliding_window
    assert cfg.hidden_size == HIDDEN and cfg.pad_token_id == PAD_ID, (cfg.hidden_size, cfg.pad_token_id)
    inv = model.rotary_emb.inv_freq
    assert float(inv.min()) > 0, "rope inv_freq zeroed by meta-load — bad export"
    return model


def sample(S, pad_frac=0.25, seed=None):
    """Right-padded batch-1 sample shaped like a real input: <s> ... </s> then pads.
    ALWAYS has pads (see module docstring)."""
    g = torch.Generator().manual_seed(0 if seed is None else seed)
    n_pad = max(1, int(S * pad_frac))
    n_valid = S - n_pad
    ids = torch.randint(10, VOCAB, (1, S), generator=g, dtype=torch.int32)
    ids[0, 0] = BOS_ID
    ids[0, n_valid - 1] = EOS_ID
    ids[0, n_valid:] = PAD_ID
    mask = torch.ones(1, S, dtype=torch.int32)
    mask[0, n_valid:] = 0
    return ids, mask, n_valid


def torch_ref(mod, ids, mask):
    with torch.inference_mode():
        return mod(ids, mask).numpy()


def assert_bidirectional(model, S=64):
    """Gate: a causal model cannot move position 0 by changing a later token."""
    ids, mask, n_valid = sample(S)
    ids2 = ids.clone()
    ids2[0, n_valid - 2] = (int(ids2[0, n_valid - 2]) + 12345) % VOCAB
    with torch.inference_mode():
        h1 = model(input_ids=ids, attention_mask=make_bias(mask)).last_hidden_state
        h2 = model(input_ids=ids2, attention_mask=make_bias(mask)).last_hidden_state
    moved = float((h1[0, 0] - h2[0, 0]).abs().max())
    print(f"EAGER bidirectional: changing token {n_valid-2} moves position 0 by {moved:.3f} (must be > 0)")
    assert moved > 1e-3, "position 0 did not move — the graph is CAUSAL"


def assert_bias_matches_vendor_mask(model, emb, S=128):
    """Gate: the hand-built 4-D bias must equal the vendor's own 2-D-mask path."""
    ids, mask, n_valid = sample(S)
    with torch.inference_mode():
        h2d = model(input_ids=ids, attention_mask=mask.to(torch.int64)).last_hidden_state
        h4d = model(input_ids=ids, attention_mask=make_bias(mask)).last_hidden_state
        m = mask.to(h2d.dtype)[:, :, None]
        p2d = F.normalize((h2d * m).sum(1) / m.sum(1), p=2, dim=-1).numpy()
    p4d = torch_ref(emb, ids, mask)
    d_hidden = float((h2d[0, :n_valid] - h4d[0, :n_valid]).abs().max())
    d_pooled = float(np.abs(p2d - p4d).max())
    print(f"EAGER 4-D bias vs vendor 2-D mask path: hidden max|diff| {d_hidden:.3e} "
          f"pooled max|diff| {d_pooled:.3e}")
    assert d_hidden < 1e-4 and d_pooled < 1e-5


def assert_pad_invariance(mod, S=128):
    """Gate: pad-region token ids must not influence the embedding at all."""
    ids, mask, n_valid = sample(S)
    ids2 = ids.clone()
    g = torch.Generator().manual_seed(7)
    ids2[0, n_valid:] = torch.randint(10, VOCAB, (S - n_valid,), generator=g, dtype=torch.int32)
    a = torch_ref(mod, ids, mask)
    b = torch_ref(mod, ids2, mask)
    leak = float(np.abs(a - b).max())
    print(f"EAGER pad-content invariance: max|diff| {leak:.3e} (must be 0)")
    assert leak == 0.0, "pad content leaks into the embedding — mask was skipped"
    c = torch_ref(mod, ids[:, :n_valid], mask[:, :n_valid])
    d = float(np.abs(a - c).max())
    cos = float((a[0] @ c[0]) / (np.linalg.norm(a[0]) * np.linalg.norm(c[0])))
    print(f"EAGER padded-vs-unpadded: max|diff| {d:.3e} cos {cos:.8f} (float noise)")


def op_report(path, tag):
    from ai_edge_litert.interpreter import Interpreter

    it = Interpreter(model_path=path)
    hist = collections.Counter(d["op_name"] for d in it._get_ops_details())
    print(f"{tag} ops:", dict(sorted(hist.items(), key=lambda kv: -kv[1])))
    i64 = [d["name"] for d in it.get_tensor_details() if "int64" in str(d["dtype"]).lower()]
    print(f"{tag} int64 tensors: {len(i64)}", i64[:5])
    return hist


def run_sig(path, name, ids, mask):
    from ai_edge_litert.interpreter import Interpreter

    it = Interpreter(model_path=path, num_threads=os.cpu_count())
    r = it.get_signature_runner(name)(input_ids=ids.numpy(), attention_mask=mask.numpy())
    return list(r.values())[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    ap.add_argument("out_dir", nargs="?", default="out/giga_embed_480m")
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

    assert_bidirectional(model)
    assert_bias_matches_vendor_mask(model, emb)
    assert_pad_invariance(emb)

    print(f"converting (litert_torch multi-signature) seqs={seq_lens} ...")
    import litert_torch

    conv = None
    for S in seq_lens:
        ids, mask, _ = sample(S)
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
        ids, mask, _ = sample(S)
        ref = torch_ref(emb, ids, mask)
        got = run_sig(fp32, f"embed_{S}", ids, mask)
        assert np.isfinite(got).all()
        cos = float(ref[0] @ got[0])
        print(f"SMOKE embed_{S} fp32 vs torch: max|diff| {np.abs(ref-got).max():.3e} cos {cos:.8f}")
        results["sigs"][f"embed_{S}"] = {"fp32_cos": cos}

    print("quantizing wi8fc ...")
    from ai_edge_quantizer import quantizer, recipe_manager, qtyping

    G = qtyping.QuantGranularity
    OP = qtyping.TFLOperationName
    rm = recipe_manager.RecipeManager()
    rm.add_dynamic_config(regex=".*", operation_name=OP.FULLY_CONNECTED, num_bits=8)
    rm.add_dynamic_config(regex=".*", operation_name=OP.EMBEDDING_LOOKUP, num_bits=8,
                          granularity=G.CHANNELWISE)
    qt = quantizer.Quantizer(fp32, rm.get_quantization_recipe())
    assert not qt.need_calibration
    qt.quantize().export_model(wi8)
    print(f"wi8fc: {os.path.getsize(wi8)/1e6:.1f} MB")

    for S in sorted(seq_lens):
        ids, mask, _ = sample(S)
        ref = torch_ref(emb, ids, mask)
        got = run_sig(wi8, f"embed_{S}", ids, mask)
        assert np.isfinite(got).all()
        cos = float(ref[0] @ got[0])
        print(f"SMOKE embed_{S} wi8fc vs torch: cos {cos:.6f}")
        results["sigs"][f"embed_{S}"]["wi8fc_cos"] = cos

    if not args.skip_fp16:
        print("quantizing fp16 ...")
        rm16 = recipe_manager.RecipeManager()
        rm16.add_quantization_config(
            regex=".*", operation_name=OP.ALL_SUPPORTED, algorithm_key="float_casting",
            op_config=qtyping.OpQuantizationConfig(
                weight_tensor_config=qtyping.TensorQuantizationConfig(
                    num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
                compute_precision=qtyping.ComputePrecision.FLOAT))
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
