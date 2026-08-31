#!/usr/bin/env python3
"""Convert microsoft/harrier-oss-v1-0.6b (causal Qwen3 last-token embedder) to LiteRT.

    python convert_harrier_embed.py <model_dir_or_id> out/harrier_embed_0p6b

Encoder lane, NOT export_hf: no KV cache in the embedding path, so we trace the
HF eager model directly with litert_torch multi-signature convert (same lane as
convert_nemotron3_embed.py / convert_bekko_embed.py).

Signatures (batch 1, right-padded static lengths):
  embed_{64,128,256,512}: (input_ids i32 [1,S], attention_mask i32 [1,S])
      -> embedding f32 [1,1024]  (LAST-VALID-token hidden state, L2-normalized)

Model shape (config.json, read 2026-08-31): Qwen3Model (bare decoder body, tied
embeddings, no LM head), 28 layers, hidden 1024, GQA 16:8, head_dim 128, SiLU,
vocab 151936 (embedding table = 26% of the 596M params), full attention on all
layers, rope theta 1e6, max_position 32768. Attention stays CAUSAL — this is the
e5-mistral recipe, not a bidirectional conversion; do NOT flip anything.

Contract (README + config_sentence_transformers.json, read 2026-08-31):
  * pooling = LAST non-padding token (1_Pooling: pooling_mode_lasttoken), then
    L2 normalize (modules.json 2_Normalize). Both are IN the graph here.
  * the tokenizer APPENDS `<|endoftext|>` (151643) itself via a
    TemplateProcessing post-processor (measured: `tok("summit define")` →
    `[1242, 1763, 6979, 151643]`), so the pooled last-valid token is that
    appended `<|endoftext|>` — the e5-mistral shape, delivered by the
    tokenizer rather than by user code. Any host that tokenizes with this
    repo's tokenizer.json gets it for free; a host that hand-rolls BPE
    without the post-processor silently pools the last TEXT token instead
    (a different, degraded contract). Note it is the same id as PAD.
  * queries MUST carry a one-sentence instruction prefix
    ("Instruct: {task}\\nQuery: "); documents are raw (README FAQ: skipping the
    instruction degrades — measured in verify_harrier_embed.py --prefix-ab).

Traps handled here:
  * Qwen3Model.forward accepts `attention_mask` as a dict
    {"full_attention": 4D bias} and skips its own mask construction (same
    walrus as ModernBERT / the bekko lane). We hand-build the causal+padding
    bias [1,1,S,S]: allowed(q,k) = (k <= q) & valid[k]. Row 0..q always
    contains position 0 (valid), so no empty rows / no NaN guard needed.
  * Last-token pool must REBIND to the mask, not to position S-1: with right
    padding the last valid position is sum(mask)-1. Static-graph form:
    onehot = (arange(S) == sum(mask)-1); pooled = sum(h * onehot). Relies on
    the contiguous right-padding contract (1s then 0s) that the signature
    interface documents. A graph that pools position S-1 unconditionally reads
    pad garbage — the pad-invariance + mask-liveness gates catch it.
  * tf5.x meta-load zero check on rotary inv_freq.
  * input_ids stay int32 end-to-end.

Quantization: wi8fc (FC int8 DRQ + EMBEDDING_LOOKUP int8 channelwise) + fp16.
No embt8 variant: the table is 26% of params here, not the bekko 80% shape.
"""
import argparse
import collections
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_MODEL = "microsoft/harrier-oss-v1-0.6b"
PAD_ID = 151643  # <|endoftext|>, config.json pad_token_id


class Embedder(nn.Module):
    """Qwen3Model body -> last-valid-token hidden state -> L2 normalize."""

    def __init__(self, body):
        super().__init__()
        self.body = body

    def forward(self, input_ids, attention_mask):
        S = input_ids.shape[1]
        valid = attention_mask.to(torch.bool)  # [1,S]
        zero = torch.zeros((), dtype=torch.float32)
        ninf = torch.full((), float("-inf"), dtype=torch.float32)

        idx = torch.arange(S)
        causal = idx[None, :] <= idx[:, None]  # [S,S] constant, k <= q
        allowed = valid[:, None, :] & causal[None]  # [1,S,S]
        bias = torch.where(allowed[:, None], zero, ninf)  # [1,1,S,S]

        h = self.body(
            input_ids=input_ids,
            attention_mask={"full_attention": bias},
            use_cache=False,
        ).last_hidden_state

        last = attention_mask.sum(dim=1, keepdim=True) - 1  # [1,1]
        onehot = (idx[None, :] == last).to(h.dtype)  # [1,S]
        pooled = (h * onehot[:, :, None]).sum(dim=1)  # [1,H]
        return F.normalize(pooled, p=2, dim=-1)


def load_model(model_id):
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        model_id, dtype=torch.float32, attn_implementation="sdpa"
    ).eval()
    assert type(model).__name__ == "Qwen3Model", type(model).__name__
    assert model.config.sliding_window is None
    inv = model.rotary_emb.inv_freq
    assert float(inv.abs().min()) > 0, "rope inv_freq zeroed by meta-load — bad export"
    return model


def sample(S, pad_frac=0.25, seed=None):
    """Right-padded batch-1 sample. ALWAYS has pads (mask must be traced live)."""
    g = torch.Generator().manual_seed(0 if seed is None else seed)
    n_pad = max(1, int(S * pad_frac))
    n_valid = S - n_pad
    ids = torch.randint(10, 151000, (1, S), generator=g, dtype=torch.int32)
    mask = torch.ones(1, S, dtype=torch.int32)
    ids[0, n_valid:] = PAD_ID
    mask[0, n_valid:] = 0
    return ids, mask, n_valid


def torch_ref(mod, ids, mask):
    with torch.inference_mode():
        return mod(ids, mask).numpy()


def assert_eager_gates(mod, model_id, S=128):
    ids, mask, n_valid = sample(S)

    # causality: changing the LAST valid token must not move EARLIER hidden
    # states (position 0's pooled proxy: pool at position 0 via a mask of 1)
    with torch.inference_mode():
        valid = mask.to(torch.bool)
        idx = torch.arange(S)
        causal = idx[None, :] <= idx[:, None]
        bias = torch.where((valid[:, None, :] & causal[None])[:, None],
                           torch.zeros(()), torch.full((), float("-inf")))
        h1 = mod.body(input_ids=ids, attention_mask={"full_attention": bias},
                      use_cache=False).last_hidden_state
        ids2 = ids.clone()
        ids2[0, n_valid - 1] = 42
        h2 = mod.body(input_ids=ids2, attention_mask={"full_attention": bias},
                      use_cache=False).last_hidden_state
    front = float((h1[0, 0] - h2[0, 0]).abs().max())
    back = float((h1[0, n_valid - 1] - h2[0, n_valid - 1]).abs().max())
    print(f"EAGER causality: last-token edit moves pos0 by {front:.3e} (must be 0), "
          f"itself by {back:.3e} (must be >0)")
    assert front == 0.0 and back > 0.0, "attention is not causal — wrong mask"

    # pad-content invariance
    a = torch_ref(mod, ids, mask)
    ids3 = ids.clone()
    g = torch.Generator().manual_seed(7)
    ids3[0, n_valid:] = torch.randint(10, 151000, (S - n_valid,), generator=g,
                                      dtype=torch.int32)
    b = torch_ref(mod, ids3, mask)
    leak = float(np.abs(a - b).max())
    print(f"EAGER pad-content invariance: max|diff| {leak:.3e} (must be 0)")
    assert leak == 0.0

    # padded vs unpadded
    c = torch_ref(mod, ids[:, :n_valid], mask[:, :n_valid])
    cos = float(a[0] @ c[0])
    print(f"EAGER padded-vs-unpadded: max|diff| {np.abs(a - c).max():.3e} "
          f"cos {cos:.8f}")
    assert cos > 0.9999

    # wrapper vs the README's own reference implementation (stock 2D-mask path)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    texts = [
        "Instruct: Given a web search query, retrieve relevant passages that "
        "answer the query\nQuery: how much protein should a female eat",
        "Definition of summit for English Language Learners.",
        "富士山は日本で一番高い山です。",
    ]
    worst = 1.0
    for t in texts:
        tids = tok(t)["input_ids"]
        n = len(tids)
        a2 = torch.full((1, S), PAD_ID, dtype=torch.int32)
        m2 = torch.zeros(1, S, dtype=torch.int32)
        a2[0, :n] = torch.tensor(tids, dtype=torch.int32)
        m2[0, :n] = 1
        ours = torch_ref(mod, a2, m2)
        with torch.inference_mode():
            h = mod.body(input_ids=torch.tensor([tids]),
                         attention_mask=torch.ones(1, n, dtype=torch.long),
                         use_cache=False).last_hidden_state
            ref = F.normalize(h[:, -1], p=2, dim=-1).numpy()
        cos = float(ours[0] @ ref[0])
        worst = min(worst, cos)
        print(f"EAGER wrapper-vs-stock ({n} tok): cos {cos:.8f} "
              f"max|diff| {np.abs(ours - ref).max():.3e}")
    assert worst > 0.99999, "wrapper diverges from the stock last-token path"


def op_report(path, tag):
    from ai_edge_litert.interpreter import Interpreter

    it = Interpreter(model_path=path)
    hist = collections.Counter(d["op_name"] for d in it._get_ops_details())
    print(f"{tag} ops:", dict(sorted(hist.items(), key=lambda kv: -kv[1])))
    i64 = [
        d["name"] for d in it.get_tensor_details() if "int64" in str(d["dtype"]).lower()
    ]
    print(f"{tag} int64 tensors: {len(i64)}", i64[:5])
    assert "GATHER_ND" not in hist, "GATHER_ND entered the graph (mobile-GPU wall)"
    return hist


def run_sig(path, name, ids, mask):
    from ai_edge_litert.interpreter import Interpreter

    it = Interpreter(model_path=path, num_threads=os.cpu_count())
    r = it.get_signature_runner(name)(
        input_ids=ids.numpy(), attention_mask=mask.numpy()
    )
    return list(r.values())[0]


def smoke(path, tag, emb, seq_lens, results, key):
    for S in sorted(seq_lens):
        ids, mask, _ = sample(S)
        ref = torch_ref(emb, ids, mask)
        got = run_sig(path, f"embed_{S}", ids, mask)
        assert np.isfinite(got).all(), f"{tag} embed_{S}: non-finite output"
        cos = float(ref[0] @ got[0] / (np.linalg.norm(got[0]) or 1.0))
        print(f"SMOKE embed_{S} {tag} vs torch: max|diff| "
              f"{np.abs(ref - got).max():.3e} cos {cos:.8f}")
        results["sigs"].setdefault(f"embed_{S}", {})[key] = cos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    ap.add_argument("out_dir", nargs="?", default="out/harrier_embed_0p6b")
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

    assert_eager_gates(emb, args.model)

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
    hist = op_report(fp32, "fp32")
    assert "EMBEDDING_LOOKUP" in hist, (
        "embedding lowered to GATHER, not EMBEDDING_LOOKUP — the int8 recipe "
        "would miss the 151936x1024 table"
    )

    results = {"model": args.model, "params_m": n_params / 1e6, "sigs": {}}
    smoke(fp32, "fp32", emb, seq_lens, results, "fp32_cos")

    from ai_edge_quantizer import quantizer, recipe_manager, qtyping

    G = qtyping.QuantGranularity
    OP = qtyping.TFLOperationName

    print("quantizing wi8fc (FC int8 DRQ + embedding int8 channelwise) ...")
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
    smoke(wi8, "wi8fc", emb, seq_lens, results, "wi8fc_cos")

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
        smoke(fp16, "fp16", emb, seq_lens, results, "fp16_cos")

    with open(os.path.join(args.out_dir, "convert_report.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("DONE:", fp32, wi8)


if __name__ == "__main__":
    main()
