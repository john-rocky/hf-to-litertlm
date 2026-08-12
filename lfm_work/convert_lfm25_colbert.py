#!/usr/bin/env python3
"""Convert LiquidAI/LFM2.5-ColBERT-350M (late-interaction retriever) to LiteRT.

    .venv-092/bin/python scripts/convert_lfm25_colbert.py <model_dir_or_id> out/lfm25_colbert_350m

Encoder lane, NOT export_hf. Same `Lfm2BidirectionalModel` backbone as
LFM2.5-Embedding-350M, but a completely different contract: this is a
multi-vector model. The graph emits ONE 128-d vector PER TOKEN; scoring
(MaxSim), query expansion and the punctuation skiplist all live host-side.

Signatures (batch 1, right-padded static lengths):
  encode_{32,128,256,512}: (input_ids i32 [1,S], attention_mask i32 [1,S])
      -> token_embeddings f32 [1,S,128], L2-normalized per token

There is deliberately no separate query/document signature: the graph is
identical for both. What differs is entirely host-side (which prefix token you
insert, how you pad, and which positions you keep) — see the host contract
below. Queries use `encode_32` (pylate's `query_length`).

## Host contract — read this before using the output (all from pylate 1.6.0)

Queries (`ColBERT.tokenize(is_query=True)`):
  1. tokenize to `query_length - 1` = **31** with `padding="max_length"`,
     padded with the **EOS token (id 7)** — pylate sets `pad_token_id` to the
     mask token if the tokenizer has one and to EOS otherwise; this tokenizer
     has no mask token, and the model card's own snippet does exactly this
     (`model.tokenizer.pad_token = model.tokenizer.eos_token`).
  2. insert `[Q] ` (id 64400) at position **1**, i.e. after BOS -> length 32,
     and insert a 1 at position 1 of the attention mask.
  3. `attend_to_expansion_tokens: false`, so the expansion positions keep
     attention_mask **0** — they are not attended to as keys.
  4. **Keep all 32 vectors for scoring.** The expansion positions carry real
     query signal; that is what query expansion means. No skiplist on queries.

Documents (`is_query=False`):
  1. tokenize to `document_length - 1` = **511**, insert `[D] ` (id 64401) at
     position 1 -> up to 512.
  2. keep positions where `skiplist_mask AND attention_mask` — punctuation
     token ids (32 of them, from `config_sentence_transformers.json`) are
     dropped, as are pads.

Scoring is MaxSim: for each kept query vector take the max dot product over the
kept document vectors, then sum over query vectors. Vectors are already unit
length, so a dot product is a cosine.

## Traps handled here

  * **Do NOT zero the padding states.** This is the exact opposite of
    LFM2.5-Embedding-350M, and the two repos ship DIFFERENT remote code: the
    ColBERT copy gates `apply_mask_to_padding_states` on
    `_attn_implementation == "flash_attention_2"` and leaves hidden states
    untouched on eager/sdpa, with the vendor's own note that transformers >=5.x
    routes the raw 2D pad mask here and "would zero padding/query-expansion
    states and shift per-token embeddings (hurts ColBERT MaxSim)". The
    Embedding-350M copy has no such gate. Same file name, different file —
    diff them, never assume. `config.json` also sets
    `disable_flash_attention: true`, so the FA2 path is refused at load.
  * Because pads are not zeroed, **the pad token id is load-bearing**: padding
    with 0 instead of 7 changes the output. Gated below.
  * The remote code targets transformers 4.56.2; on 5.14.1 the decoder layer
    passes `seq_idx` into a `slow_forward` that never declared it. Rebind
    `Lfm2ShortConv.forward` to a kwarg-filtering wrapper — never edit the
    cached remote code.
  * Masks are built here and passed as a dict, which `Lfm2Model.forward`
    accepts verbatim, so no transformers mask builder specializes at trace
    time.
  * tf5.x meta-load can zero init-computed buffers -> assert inv_freq > 0.

Quantization: wi8fc — int8 dynamic-range on FULLY_CONNECTED + int8 channelwise
on the embedding table, ShortConv depthwise convs stay float.
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

DEFAULT_MODEL = "LiquidAI/LFM2.5-ColBERT-350M"
PAD_ID = 7          # EOS — pylate's query-expansion / padding token here
BOS_ID = 1
Q_PREFIX_ID = 64400  # "[Q] "
D_PREFIX_ID = 64401  # "[D] "
NEG = -1e9
QUERY_LEN = 32
DOC_LEN = 512


class ColbertEncoder(nn.Module):
    """Lfm2BidirectionalModel -> pylate Dense(1024->128) -> per-token L2 norm."""

    def __init__(self, body, dense_weight):
        super().__init__()
        self.body = body
        self.proj = nn.Linear(dense_weight.shape[1], dense_weight.shape[0], bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(dense_weight)

    def forward(self, input_ids, attention_mask):
        m = attention_mask.to(torch.float32)                     # [1,S]
        # Pad-only additive mask, broadcast over query rows. Bidirectional, and
        # no row is ever fully masked, so softmax stays finite.
        full = ((1.0 - m) * NEG)[:, None, None, :]               # [1,1,1,S]
        h = self.body(
            input_ids=input_ids,
            attention_mask={"full_attention": full, "conv": m},
            use_cache=False,
        ).last_hidden_state                                      # [1,S,1024]
        return F.normalize(self.proj(h), p=2, dim=-1)            # [1,S,128]


def load_model(model_id):
    from transformers import AutoModel
    from transformers.models.lfm2.modeling_lfm2 import Lfm2ShortConv

    model = AutoModel.from_pretrained(
        model_id, trust_remote_code=True, dtype=torch.float32,
        attn_implementation="sdpa",
    ).eval()
    assert type(model).__name__ == "Lfm2BidirectionalModel", type(model).__name__
    assert float(model.rotary_emb.inv_freq.min()) > 0, "rope inv_freq zeroed by meta-load"

    keep = set(inspect.signature(Lfm2ShortConv.slow_forward).parameters)

    def _forward_filtered(self, *args, **kwargs):
        return self.slow_forward(*args, **{k: v for k, v in kwargs.items()
                                           if k in keep})

    Lfm2ShortConv.forward = _forward_filtered

    # Confirm — do not assume — that this repo's remote code is the variant that
    # LEAVES padding states alone on sdpa. If a future checkout drops the gate,
    # query-expansion vectors would silently be zeroed and MaxSim would degrade.
    src = inspect.getsource(sys.modules[type(model).__module__])
    gated = 'flash_attention_2"' in src and "x = hidden_states" in src
    print(f"remote code gates pad-zeroing on FA2 only: {gated}")
    assert gated, ("this checkout's modeling_lfm2_bidirectional.py zeroes padding "
                   "states on sdpa — that shifts query-expansion vectors and "
                   "hurts MaxSim; do not export it")
    return model


def load_dense(model_dir):
    """pylate Dense: Linear(1024->128), bias=False, Identity activation."""
    from safetensors.torch import load_file

    cfg_path = os.path.join(model_dir, "1_Dense", "config.json")
    cfg = json.load(open(cfg_path))
    assert cfg["bias"] is False and cfg["use_residual"] is False, cfg
    assert cfg["activation_function"].endswith("Identity"), cfg
    w = load_file(os.path.join(model_dir, "1_Dense", "model.safetensors"))["linear.weight"]
    assert tuple(w.shape) == (cfg["out_features"], cfg["in_features"]), w.shape
    print(f"Dense: {tuple(w.shape)} {w.dtype} -> float32, bias=False")
    return w.to(torch.float32)


def query_sample(n_real=6, seed=0):
    """A pylate-shaped query: BOS, [Q], n_real tokens, then EOS expansion."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.full((1, QUERY_LEN), PAD_ID, dtype=torch.int32)
    mask = torch.zeros(1, QUERY_LEN, dtype=torch.int32)
    ids[0, 0] = BOS_ID
    ids[0, 1] = Q_PREFIX_ID
    ids[0, 2:2 + n_real] = torch.randint(10, 60000, (n_real,), generator=g,
                                         dtype=torch.int32)
    mask[0, :2 + n_real] = 1
    return ids, mask, 2 + n_real


def doc_sample(S, frac=0.7, seed=1):
    """A right-padded document: BOS, [D], tokens, then EOS padding."""
    g = torch.Generator().manual_seed(seed)
    n = max(3, int(S * frac))
    ids = torch.full((1, S), PAD_ID, dtype=torch.int32)
    mask = torch.zeros(1, S, dtype=torch.int32)
    ids[0, 0] = BOS_ID
    ids[0, 1] = D_PREFIX_ID
    ids[0, 2:n] = torch.randint(10, 60000, (n - 2,), generator=g, dtype=torch.int32)
    mask[0, :n] = 1
    return ids, mask, n


def torch_ref(mod, ids, mask):
    with torch.inference_mode():
        return mod(ids, mask).numpy()


def eager_checks(enc, model):
    print("\n-- eager gates --")
    # (a) hand-built dict masks reproduce the vendor's own mask path, unpadded.
    ids, mask, _ = doc_sample(96, frac=1.0)
    ones = torch.ones_like(mask)
    with torch.inference_mode():
        h = model(input_ids=ids, attention_mask=ones.float(),
                  use_cache=False).last_hidden_state
        vendor = F.normalize(enc.proj(h), p=2, dim=-1).numpy()
    d = float(np.abs(torch_ref(enc, ids, ones) - vendor).max())
    print(f"masks vs vendor path (unpadded): max|diff| {d:.3e}")
    assert d < 1e-6, "hand-built masks disagree with the vendor path"

    # (b) query expansion must SURVIVE. If a future remote-code change zeroed
    #     padding states, these vectors would collapse to a constant direction
    #     and MaxSim would quietly lose the expansion signal.
    ids, mask, n_real = query_sample()
    v = torch_ref(enc, ids, mask)[0]
    exp = v[n_real:]
    spread = float(np.abs(exp - exp[0]).max())
    print(f"query expansion: {len(exp)} vectors, |v| {np.linalg.norm(exp, axis=1).min():.4f}"
          f"..{np.linalg.norm(exp, axis=1).max():.4f}, spread across positions {spread:.4f}")
    assert spread > 1e-3, "expansion vectors are identical — padding states got zeroed"

    # (c) the pad id is load-bearing (pads are NOT zeroed by design). Changing it
    #     must move the output — that is why the host has to pad with EOS.
    ids2 = ids.clone()
    ids2[0, n_real:] = 0
    d_pad = float(np.abs(v - torch_ref(enc, ids2, mask)[0]).max())
    print(f"pad id 7 vs 0: max|diff| {d_pad:.4f} (must be > 0 — pad id is part of the contract)")
    assert d_pad > 1e-4, "pad id does not matter — unexpected, re-check the gate"

    # (d) document padded-vs-unpadded drift, at the kept positions. Not zero for
    #     this model (pads are not zeroed), so report it rather than assert it:
    #     it is the cost of a static shape and it bounds how much a longer
    #     signature can move a document's vectors.
    for S in (128, 512):
        ids, mask, n = doc_sample(S, frac=0.25)
        a = torch_ref(enc, ids, mask)[0, :n]
        b = torch_ref(enc, ids[:, :n], mask[:, :n])[0]
        cos = float((a * b).sum(1).min())
        print(f"doc padded({S}) vs unpadded({n}): max|diff| {np.abs(a-b).max():.4f}  "
              f"min per-token cos {cos:.6f}")


def op_report(path, tag):
    from ai_edge_litert.interpreter import Interpreter

    it = Interpreter(model_path=path)
    hist = collections.Counter(d["op_name"] for d in it._get_ops_details())
    print(f"{tag} ops:", dict(sorted(hist.items(), key=lambda kv: -kv[1])))
    i64 = [d["name"] for d in it.get_tensor_details()
           if "int64" in str(d["dtype"]).lower()]
    print(f"{tag} int64 tensors: {len(i64)}")
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


def check_variant(path, tag, enc, seq_lens, results):
    for S in sorted(seq_lens):
        ids, mask, n = (query_sample() if S == QUERY_LEN else doc_sample(S))
        ref = torch_ref(enc, ids, mask)
        got = run_sig(path, f"encode_{S}", ids, mask)
        assert np.isfinite(got).all(), f"encode_{S} {tag} output is not finite"
        # Per-token cosine over the positions the host actually keeps.
        keep = slice(None) if S == QUERY_LEN else slice(0, n)
        cos = float((ref[0, keep] * got[0, keep]).sum(-1).min())
        print(f"SMOKE encode_{S} {tag} vs torch: max|diff| "
              f"{np.abs(ref-got).max():.3e} min per-token cos {cos:.6f}")
        results["sigs"].setdefault(f"encode_{S}", {})[f"{tag}_min_cos"] = cos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    ap.add_argument("out_dir", nargs="?", default="out/lfm25_colbert_350m")
    ap.add_argument("--seqs", default="512,256,128,32")
    ap.add_argument("--skip-fp16", action="store_true")
    args = ap.parse_args()

    seq_lens = [int(s) for s in args.seqs.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)
    fp32 = os.path.join(args.out_dir, "colbert_fp32.tflite")
    wi8 = os.path.join(args.out_dir, "colbert_wi8fc.tflite")
    fp16 = os.path.join(args.out_dir, "colbert_fp16.tflite")
    for stale in (wi8, fp16):
        if os.path.exists(stale):
            print(f"removing stale {os.path.basename(stale)}")
            os.remove(stale)

    torch.manual_seed(0)
    print(f"loading {args.model} ...")
    model = load_model(args.model)
    dense = load_dense(args.model)
    enc = ColbertEncoder(model, dense).eval()
    n_params = sum(p.numel() for p in enc.parameters())
    print(f"params {n_params/1e6:.1f}M  vocab {model.config.vocab_size}  dim 128")

    eager_checks(enc, model)

    print(f"\nconverting (litert_torch multi-signature) seqs={seq_lens} ...")
    import litert_torch

    conv = None
    for S in seq_lens:
        ids, mask, _ = (query_sample() if S == QUERY_LEN else doc_sample(S))
        kw = {"input_ids": ids, "attention_mask": mask}
        sig = (f"encode_{S}", enc)
        conv = (litert_torch.signature(*sig, sample_kwargs=kw) if conv is None
                else conv.signature(*sig, sample_kwargs=kw))
    edge = conv.convert()
    edge.export(fp32)
    print(f"fp32: {os.path.getsize(fp32)/1e6:.1f} MB")

    results = {"model": args.model, "params_m": n_params / 1e6, "dim": 128,
               "seqs": seq_lens, "pad_id": PAD_ID, "query_len": QUERY_LEN,
               "query_prefix_id": Q_PREFIX_ID, "doc_prefix_id": D_PREFIX_ID,
               "scoring": "maxsim", "normalized": "per-token L2",
               "ops": op_report(fp32, "fp32"), "sigs": {}}
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
