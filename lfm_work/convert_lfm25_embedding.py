#!/usr/bin/env python3
"""Convert LiquidAI/LFM2.5-Embedding-350M (Lfm2BidirectionalModel) to LiteRT.

    .venv-092/bin/python scripts/convert_lfm25_embedding.py <model_dir_or_id> out/lfm25_embed_350m

Encoder lane, NOT export_hf: a bi-encoder with no KV cache, so the HF eager
model is traced directly with litert_torch multi-signature convert (same lane
as convert_lfm25_encoder.py / convert_granite_embedding_r2.py).

Signatures (batch 1, right-padded static lengths):
  embed_{64,128,256,512}: (input_ids i32 [1,S], attention_mask i32 [1,S])
      -> embedding f32 [1,1024]  (CLS = position 0, then L2 normalize)

Host-side contract (read off the repo's own files, not the prose):
  * CLS pooling — `1_Pooling/config.json` pooling_mode_cls_token: true. The
    tokenizer puts `<|startoftext|>` (id 1) at position 0, so CLS == BOS.
  * Asymmetric prompts `"query: "` / `"document: "`, WITH the trailing space:
    `config_sentence_transformers.json` and the card prose agree once the JSON
    is actually read (both give `Ġ`-prefixed content tokens, e.g. 3747 not
    3493). Omitting the prefix is documented to degrade retrieval.
  * `modules.json` has no `2_Normalize`, but every documented usage passes
    normalize_embeddings=True, so the L2 normalize is folded into the graph.
  * Max sequence length 512 (`sentence_bert_config.json`).

Traps handled here (verified against the installed sources, tf 5.14.1):
  * The repo's remote code targets transformers 4.56.2. It installs
    `Lfm2ShortConv.forward = _shortconv_forward`, which forwards **kwargs
    straight into `_noncausal_shortconv_forward(hidden_states,
    past_key_values, cache_position, attention_mask)`. tf 5.14.1's
    Lfm2DecoderLayer also passes `seq_idx=...`, so any forward dies with
        TypeError: _noncausal_shortconv_forward() got an unexpected keyword
        argument 'seq_idx'
    Fixed by rebinding `forward` to a wrapper that filters kwargs down to
    slow_forward's own signature. The cached remote code is never edited — the
    filter keeps working when the vendor updates it.
  * Pad semantics: take the FA2 path (zero the pads), not the sdpa default.
    `apply_mask_to_padding_states` no-ops here (`attention_mask.shape[0] > 1`
    is False at batch 1, and in this transformers version the conv layers get
    the 2D mask via create_recurrent_attention_mask), so the symmetric
    ShortConv reads pad-token embeddings and the SAME TEXT returns a different
    vector at embed_128 vs embed_512 (measured max|d| 0.004955). Zeroing the
    pads restores exact length invariance — 0.0 at every length — and is the
    vendor's own FA2 behaviour, which the card reports as equivalent to the
    default within 0.002 nDCG across 11 languages. Rebound unconditionally in
    the remote module.
  * Masks are built here and passed as a **dict**, which tf 5.14.1's
    Lfm2Model.forward accepts verbatim ({"full_attention": ..., "conv": ...}).
    That skips transformers' own mask builders, whose "skip the mask when
    nothing is padded" branch would specialize at trace time and bake a graph
    that ignores attention_mask.
  * tf5.x meta-load can zero init-computed buffers (rope inv_freq) for
    remote-code models -> assert inv_freq > 0 after load.
  * input_ids stay int32 end-to-end (F.embedding accepts int32; no CAST/int64
    in the graph — int64 was the LFM GPU wall on the decoder side).

Quantization: wi8fc — dynamic-range int8 on FULLY_CONNECTED + int8 channelwise
on the embedding table, depthwise ShortConv convs stay float (ALL_SUPPORTED
int8 is a proven conv-killer on LFM2 hybrids).
"""
import argparse
import collections
import inspect
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_MODEL = "LiquidAI/LFM2.5-Embedding-350M"
PAD_ID = 0
BOS_ID = 1
NEG = -1e9  # the vendor's own mask fill; -inf would NaN a fully-masked row
QUERY_PREFIX = "query: "
DOC_PREFIX = "document: "


class Embedder(nn.Module):
    """Lfm2BidirectionalModel -> CLS (position 0) -> L2 normalize.

    Masks are hand-built and handed to the body as a dict so no transformers
    mask builder runs during trace (see module docstring).
    """

    def __init__(self, body):
        super().__init__()
        self.body = body

    def forward(self, input_ids, attention_mask):
        m = attention_mask.to(torch.float32)                    # [1,S]
        # Pad-only additive mask, broadcast over query rows. Bidirectional:
        # every query may read every valid key. No row is ever fully masked
        # (pad queries still see the valid keys), so softmax stays finite.
        full = ((1.0 - m) * NEG)[:, None, None, :]              # [1,1,1,S]
        h = self.body(
            input_ids=input_ids,
            attention_mask={"full_attention": full, "conv": m},
            use_cache=False,
        ).last_hidden_state
        return F.normalize(h[:, 0], p=2, dim=-1)                # CLS pooling


def load_model(model_id):
    from transformers import AutoModel
    from transformers.models.lfm2.modeling_lfm2 import Lfm2ShortConv

    model = AutoModel.from_pretrained(
        model_id, trust_remote_code=True, dtype=torch.float32,
        attn_implementation="sdpa",
    ).eval()
    assert type(model).__name__ == "Lfm2BidirectionalModel", type(model).__name__

    # tf5.x meta-load trap: init-computed buffers can materialize as zeros.
    inv = model.rotary_emb.inv_freq
    assert float(inv.min()) > 0, "rope inv_freq zeroed by meta-load — bad export"

    # Trap 1 — drop kwargs the vendor's slow_forward does not declare (seq_idx).
    keep = set(inspect.signature(Lfm2ShortConv.slow_forward).parameters)

    def _forward_filtered(self, *args, **kwargs):
        return self.slow_forward(*args, **{k: v for k, v in kwargs.items()
                                           if k in keep})

    Lfm2ShortConv.forward = _forward_filtered

    # Trap 2 — force the FA2-style pad zeroing at batch 1 (see docstring).
    remote_mod = sys.modules[type(model).__module__]

    def _apply_mask_always(hidden_states, attention_mask):
        if attention_mask is not None and attention_mask.ndim == 2:
            hidden_states = hidden_states * attention_mask[:, :, None].to(
                hidden_states.dtype)
        return hidden_states

    remote_mod.apply_mask_to_padding_states = _apply_mask_always

    cfg = model.config
    print(f"layer_types: {collections.Counter(cfg.layer_types)}  "
          f"hidden {cfg.hidden_size}  vocab {cfg.vocab_size}  "
          f"conv_L_cache {cfg.conv_L_cache}")
    return model


def sample(S, pad_frac=0.25, seed=0, vocab=65536):
    """Right-padded batch-1 sample. ALWAYS has pads (that is what we gate)."""
    g = torch.Generator().manual_seed(seed)
    n_valid = S - max(1, int(S * pad_frac))
    ids = torch.randint(10, vocab - 1, (1, S), generator=g, dtype=torch.int32)
    ids[0, 0] = BOS_ID
    mask = torch.ones(1, S, dtype=torch.int32)
    ids[0, n_valid:] = PAD_ID
    mask[0, n_valid:] = 0
    return ids, mask, n_valid


def torch_ref(mod, ids, mask):
    with torch.inference_mode():
        return mod(ids, mask).numpy()


def eager_checks(emb, model):
    """Three eager gates that must pass before anything is exported."""
    # (a) our hand-built masks reproduce the vendor's own mask path. Run it on
    #     UNPADDED input so the pad-semantics decision cannot flatter the check
    #     — this isolates the mask construction alone.
    ids, _, _ = sample(96, pad_frac=0.0)
    with torch.inference_mode():
        vendor = model(input_ids=ids,
                       attention_mask=torch.ones_like(ids, dtype=torch.float32),
                       use_cache=False).last_hidden_state
        vendor = F.normalize(vendor[:, 0], p=2, dim=-1).numpy()
    ours = torch_ref(emb, ids, torch.ones_like(ids))
    d = float(np.abs(ours - vendor).max())
    print(f"EAGER vs vendor mask path (unpadded): max|diff| {d:.3e}")
    assert d < 1e-6, "hand-built masks disagree with the vendor path"

    # (b) pad CONTENT must not reach the CLS vector.
    S = 128
    ids, mask, n_valid = sample(S)
    ids2 = ids.clone()
    g = torch.Generator().manual_seed(7)
    ids2[0, n_valid:] = torch.randint(10, 65535, (S - n_valid,), generator=g,
                                      dtype=torch.int32)
    a, b = torch_ref(emb, ids, mask), torch_ref(emb, ids2, mask)
    leak = float(np.abs(a - b).max())
    print(f"EAGER pad-content invariance: max|diff| {leak:.3e} (must be 0)")
    assert leak == 0.0, "pad content reaches CLS — pad zeroing not applied"

    # (c) length invariance: the same text padded to any length must give the
    #     same vector, or embed_128 and embed_512 disagree on identical input.
    c = torch_ref(emb, ids[:, :n_valid], mask[:, :n_valid])
    d = float(np.abs(a - c).max())
    print(f"EAGER padded-vs-unpadded: max|diff| {d:.3e} (must be 0) "
          f"cos {float(a[0] @ c[0]):.8f}")
    assert d == 0.0, "padded forward differs from unpadded — pad semantics wrong"


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
            print(f"  note: {hostile} x{hist[hostile]} present (mobile-GPU hostile)")
    return dict(hist)


def run_sig(path, name, ids, mask):
    from ai_edge_litert.interpreter import Interpreter

    it = Interpreter(model_path=path, num_threads=os.cpu_count())
    r = it.get_signature_runner(name)(input_ids=ids.numpy(),
                                      attention_mask=mask.numpy())
    return list(r.values())[0]


def check_variant(path, tag, emb, seq_lens, results):
    for S in sorted(seq_lens):
        ids, mask, _ = sample(S)
        ref = torch_ref(emb, ids, mask)
        got = run_sig(path, f"embed_{S}", ids, mask)
        # Assert finiteness rather than trusting a cosine that would read "nan":
        # int8 has been observed to hide a NaN behind a healthy-looking cosine.
        assert np.isfinite(got).all(), f"embed_{S} {tag} output is not finite"
        cos = float(ref[0] @ got[0])
        print(f"SMOKE embed_{S} {tag} vs torch: max|diff| "
              f"{np.abs(ref-got).max():.3e} cos {cos:.8f}")
        results["sigs"].setdefault(f"embed_{S}", {})[f"{tag}_cos"] = cos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    ap.add_argument("out_dir", nargs="?", default="out/lfm25_embed_350m")
    ap.add_argument("--seqs", default="512,256,128,64")
    ap.add_argument("--skip-fp16", action="store_true")
    args = ap.parse_args()

    seq_lens = [int(s) for s in args.seqs.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)
    fp32 = os.path.join(args.out_dir, "embed_fp32.tflite")
    wi8 = os.path.join(args.out_dir, "embed_wi8fc.tflite")
    fp16 = os.path.join(args.out_dir, "embed_fp16.tflite")
    # ai_edge_quantizer refuses to overwrite, which on a re-run leaves STALE
    # quantized files next to a freshly exported fp32 — they then get
    # "verified" and shipped. Clear them up front so a re-run is idempotent.
    for stale in (wi8, fp16):
        if os.path.exists(stale):
            print(f"removing stale {os.path.basename(stale)}")
            os.remove(stale)

    torch.manual_seed(0)
    print(f"loading {args.model} ...")
    model = load_model(args.model)
    emb = Embedder(model).eval()
    n_params = sum(p.numel() for p in model.parameters())
    vocab_params = model.config.vocab_size * model.config.hidden_size
    print(f"params {n_params/1e6:.1f}M (vocab table {vocab_params/1e6:.1f}M "
          f"= {100*vocab_params/n_params:.0f}%)")

    eager_checks(emb, model)

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

    results = {"model": args.model, "params_m": n_params / 1e6,
               "dim": model.config.hidden_size, "seqs": seq_lens,
               "pooling": "cls", "normalized": True,
               "prompts": {"query": QUERY_PREFIX, "document": DOC_PREFIX},
               "ops": op_report(fp32, "fp32"), "sigs": {}}
    check_variant(fp32, "fp32", emb, seq_lens, results)

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
    check_variant(wi8, "wi8fc", emb, seq_lens, results)

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
        check_variant(fp16, "fp16", emb, seq_lens, results)

    results["sizes_mb"] = {
        t: round(os.path.getsize(p) / 1e6, 1)
        for t, p in (("fp32", fp32), ("wi8fc", wi8), ("fp16", fp16))
        if os.path.exists(p)}
    with open(os.path.join(args.out_dir, "convert_report.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("DONE:", fp32, wi8)


if __name__ == "__main__":
    main()
