#!/usr/bin/env python3
"""Convert hotchpotch/bekko-embedding-v1-a25m (mmBERT/ModernBERT embedder) to LiteRT.

    python convert_bekko_embed.py <model_dir_or_id> out/bekko_embed_a25m

Encoder lane, NOT export_hf: no KV cache in the embedding path, so we trace the
HF eager model directly with litert_torch multi-signature convert (same lane as
scripts/convert_nemotron3_embed.py / convert_lfm25_encoder.py).

Signatures (batch 1, right-padded static lengths):
  embed_{64,128,256,512}: (input_ids i32 [1,S], attention_mask i32 [1,S])
      -> embedding f32 [1,384]  (mean pool over valid positions, then L2 norm)

Pooling + normalize are IN the graph (nemotron precedent: the pooled vector is
the product; keeping it in-graph removes the caller's chance to get masking or
normalization subtly wrong, and shrinks the per-call output copy). modules.json
has no Normalize stage, but the vendor's own quickstart/GGUF contract is
L2-normalized output (`normalize_embeddings=True`, `--embd-normalize 2`); an
L2-normalized 384-d vector still truncates cleanly to 256/128/64 (Matryoshka),
because pre-scaling by a scalar does not change the renormalized truncation.
NO prefixes: bekko is trained without query:/passage: prefixes (README FAQ).

Model shape (config.json, read 2026-08-31): ModernBertModel, 13 layers,
hidden 384, 6 heads MHA (fused Wqkv), GeGLU MLP, vocab 256000 (Gemma-style,
80% of the 123M params are the embedding table), max_position 8192,
layer_types: full_attention every 3rd layer, else sliding_attention with
local_attention=128 -> config.sliding_window property = 64 (HALF-window;
masking predicate is `abs(q_idx - kv_idx) <= 64`, inclusive — read out of
transformers 5.14.1 masking_utils.sliding_window_bidirectional_overlay).

Traps handled here (all verified against the installed sources, tf 5.14.1):
  * ModernBertModel.forward accepts `attention_mask` as a DICT
    {layer_type: 4D bias} and then skips its own mask construction entirely
    (`isinstance(attention_mask_mapping := attention_mask, dict)`). We build
    both biases by hand and pass the dict — so transformers' sdpa_mask index
    machinery never enters the graph, and the all-ones-mask skip
    (`allow_is_bidirectional_skip`) can never specialize a mask-ignoring graph
    at trace time (the nemotron trap, arrived at via a supported entry point).
  * Sliding-window empty rows -> NaN: with right padding, a pad query at
    position q >= n_valid + 64 has only pad keys inside its +-64 window; an
    all-masked row makes SDPA emit NaN, and mask*NaN = NaN poisons the mean
    pool. HF's own sdpa_mask "fixes" fully-masked rows by unmasking them
    wholesale; we instead always allow the diagonal:
        allowed(q,k) = (valid[k] | (q==k)) & (|q-k| <= 64)
    For valid queries the extra term is a no-op (q==k is already a valid key);
    pad queries get finite garbage that the pooling mask then drops. Same
    class as [[sliding-window-empty-row-nan]].
  * Full-attention bias is the broadcast [1,1,1,S] key-validity bias (pad
    query rows attend all valid keys — never empty, no fix needed).
  * tf5.x meta-load can zero init-computed non-persistent buffers; here that is
    `{layer_type}_inv_freq` on ModernBertRotaryEmbedding -> assert all > 0.
  * input_ids stay int32 end-to-end (nn.Embedding accepts int32).

Quantization A/B (the vendor's own artifacts are the prior: their DEFAULT
ONNX/OpenVINO ships quantize ONLY the embedding table to row-wise int8 and keep
transformer compute fp32; their transformer-int8 files are marked experimental/
not recommended, and their GGUF card says low-bit variants lose fidelity):
  * embt8 — EMBEDDING_LOOKUP int8 channelwise only (vendor-default analog)
  * wi8fc — FC int8 DRQ + EMBEDDING_LOOKUP int8 channelwise (mobile pick,
            must survive the JSTS/retrieval gates in verify_bekko_embed.py)
  * fp16  — whole-model float16
"""
import argparse
import collections
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_MODEL = "hotchpotch/bekko-embedding-v1-a25m"
PAD_ID = 0  # <pad>, config.json + tokenizer


class Embedder(nn.Module):
    """ModernBertModel body -> mean pool over valid positions -> L2 normalize.

    Both attention biases are built here and handed to the body as the
    layer-type dict it accepts verbatim (see module docstring). fp32, additive:
    0.0 where attendable, -inf where not.
    """

    def __init__(self, body):
        super().__init__()
        self.body = body
        self.half_window = int(body.config.sliding_window)  # 64 = local_attention // 2

    def forward(self, input_ids, attention_mask):
        S = input_ids.shape[1]
        valid = attention_mask.to(torch.bool)  # [1,S]
        zero = torch.zeros((), dtype=torch.float32)
        ninf = torch.full((), float("-inf"), dtype=torch.float32)

        # full attention: [1,1,1,S] key-validity bias, broadcasts over queries
        full_bias = torch.where(valid[:, None, None, :], zero, ninf)

        # sliding attention: [1,1,S,S]; diagonal always allowed (NaN guard)
        idx = torch.arange(S)
        dist = (idx[:, None] - idx[None, :]).abs()
        window = dist <= self.half_window  # [S,S] constant
        eye = idx[:, None] == idx[None, :]  # [S,S] constant
        allowed = (valid[:, None, :] | eye[None]) & window[None]  # [1,S,S]
        slide_bias = torch.where(allowed[:, None], zero, ninf)  # [1,1,S,S]

        h = self.body(
            input_ids=input_ids,
            attention_mask={
                "full_attention": full_bias,
                "sliding_attention": slide_bias,
            },
        ).last_hidden_state
        m = attention_mask.to(h.dtype)[:, :, None]
        pooled = (h * m).sum(dim=1) / m.sum(dim=1)
        return F.normalize(pooled, p=2, dim=-1)


def load_model(model_id):
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        model_id, dtype=torch.float32, attn_implementation="sdpa"
    ).eval()
    assert type(model).__name__ == "ModernBertModel", type(model).__name__
    assert int(model.config.sliding_window) == 64, model.config.sliding_window
    for name, buf in model.named_buffers():
        if "inv_freq" in name:
            assert float(buf.abs().min()) > 0, (
                f"{name} zeroed by meta-load — bad export"
            )
    return model


def sample(S, pad_frac=0.25, seed=None):
    """Right-padded batch-1 sample. ALWAYS has pads (see module docstring)."""
    g = torch.Generator().manual_seed(0 if seed is None else seed)
    n_pad = max(1, int(S * pad_frac))
    n_valid = S - n_pad
    ids = torch.randint(10, 255000, (1, S), generator=g, dtype=torch.int32)
    mask = torch.ones(1, S, dtype=torch.int32)
    ids[0, n_valid:] = PAD_ID
    mask[0, n_valid:] = 0
    return ids, mask, n_valid


def torch_ref(mod, ids, mask):
    with torch.inference_mode():
        return mod(ids, mask).numpy()


def assert_pad_invariance(mod, S=512, pad_frac=0.9):
    """Gate: pad content must not leak, and heavy padding must not NaN.

    pad_frac=0.9 leaves n_valid=51 in S=512, so every pad query past position
    ~115 has an empty +-64 window — exactly the row shape that NaNs without
    the diagonal guard.
    """
    ids, mask, n_valid = sample(S, pad_frac=pad_frac)
    a = torch_ref(mod, ids, mask)
    assert np.isfinite(a).all(), "NaN/inf under heavy right padding — empty sliding rows"

    ids2 = ids.clone()
    g = torch.Generator().manual_seed(7)
    ids2[0, n_valid:] = torch.randint(
        10, 255000, (S - n_valid,), generator=g, dtype=torch.int32
    )
    b = torch_ref(mod, ids2, mask)
    leak = float(np.abs(a - b).max())
    print(f"EAGER pad-content invariance (S={S}, n_valid={n_valid}): "
          f"max|diff| {leak:.3e} (must be 0)")
    assert leak == 0.0, "pad content leaks into the embedding — mask was skipped"

    # And the padded forward must agree with the truly unpadded one.
    c = torch_ref(mod, ids[:, :n_valid], mask[:, :n_valid])
    d = float(np.abs(a - c).max())
    cos = float(a[0] @ c[0])
    print(f"EAGER padded-vs-unpadded: max|diff| {d:.3e} cos {cos:.8f} (float noise)")
    assert cos > 0.9999, "padded and unpadded disagree — mask semantics wrong"


def assert_matches_stock(mod, model_id):
    """Gate: hand-built masks == transformers' own create_bidirectional_* path.

    Runs the SAME body through the stock entry point (2D mask, transformers
    builds the layer-type masks itself) on real tokenized text, padded and
    unpadded, and requires float-noise agreement with the wrapper.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    texts = [
        "A Japanese dish made with vinegared rice, often shaped with seafood.",
        "天ぷらは魚や野菜に衣をつけて揚げた料理です。",
        "El monte Fuji es la montaña más alta de Japón.",
    ]
    worst = 1.0
    for t in texts:
        ids = tok(t, add_special_tokens=True)["input_ids"]
        n = len(ids)
        S = 128
        a = torch.full((1, S), PAD_ID, dtype=torch.int32)
        m = torch.zeros(1, S, dtype=torch.int32)
        a[0, :n] = torch.tensor(ids, dtype=torch.int32)
        m[0, :n] = 1
        ours = torch_ref(mod, a, m)
        with torch.inference_mode():
            h = mod.body(
                input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, n, dtype=torch.long)
            ).last_hidden_state
            ref = F.normalize(h.mean(dim=1), p=2, dim=-1).numpy()
        cos = float(ours[0] @ ref[0])
        worst = min(worst, cos)
        print(f"EAGER wrapper-vs-stock ({n} tok): cos {cos:.8f} "
              f"max|diff| {np.abs(ours - ref).max():.3e}")
    assert worst > 0.99999, "wrapper mask semantics diverge from stock transformers"


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
    ap.add_argument("out_dir", nargs="?", default="out/bekko_embed_a25m")
    ap.add_argument("--seqs", default="512,256,128,64")
    ap.add_argument("--skip-fp16", action="store_true")
    args = ap.parse_args()

    seq_lens = [int(s) for s in args.seqs.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)
    fp32 = os.path.join(args.out_dir, "embed_fp32.tflite")
    wi8 = os.path.join(args.out_dir, "embed_wi8fc.tflite")
    emb8 = os.path.join(args.out_dir, "embed_embt8.tflite")
    fp16 = os.path.join(args.out_dir, "embed_fp16.tflite")

    torch.manual_seed(0)
    print(f"loading {args.model} ...")
    model = load_model(args.model)
    emb = Embedder(model).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params {n_params/1e6:.1f}M  hidden {model.config.hidden_size} "
          f"layers {model.config.num_hidden_layers} vocab {model.config.vocab_size}")

    assert_pad_invariance(emb)
    assert_matches_stock(emb, args.model)

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
        "would miss the 256000x384 table (80% of all parameters)"
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

    print("quantizing embt8 (embedding table int8 only — vendor-default analog) ...")
    rme = recipe_manager.RecipeManager()
    rme.add_dynamic_config(
        regex=".*", operation_name=OP.EMBEDDING_LOOKUP, num_bits=8,
        granularity=G.CHANNELWISE,
    )
    qte = quantizer.Quantizer(fp32, rme.get_quantization_recipe())
    assert not qte.need_calibration
    qte.quantize().export_model(emb8)
    print(f"embt8: {os.path.getsize(emb8)/1e6:.1f} MB")
    smoke(emb8, "embt8", emb, seq_lens, results, "embt8_cos")

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
    print("DONE:", fp32, wi8, emb8)


if __name__ == "__main__":
    main()
