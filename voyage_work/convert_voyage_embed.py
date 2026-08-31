#!/usr/bin/env python3
"""Convert voyageai/voyage-4-nano (bidirectional Qwen3 mean-pool embedder) to LiteRT.

    python convert_voyage_embed.py <model_dir_or_id> out/voyage_embed_nano

Encoder lane, NOT export_hf: no KV cache in the embedding path, so we trace the
HF eager model directly with litert_torch multi-signature convert (same lane as
convert_harrier_embed.py / convert_bekko_embed.py).

Signatures (batch 1, right-padded static lengths):
  embed_{64,128,256,512}: (input_ids i32 [1,S], attention_mask i32 [1,S])
      -> embedding f32 [1,2048]  (mean-pooled over valid tokens, L2-normalized)

Model shape (config.json + modeling_qwen3_bidirectional.py, read 2026-08-31):
`architectures` says Qwen3ForCausalLM but that is a LIE — auto_map.AutoModel
points at remote-code Qwen3BidirectionalModel: a bare Qwen3Model body
(12 layers, hidden 1024, GQA 16:8, head_dim 128, SiLU, vocab 151936, full
attention, rope theta 1e6) with `use_bidirectional_attention: true`,
layer.self_attn.is_causal=False, plus a per-token `linear` 1024 -> 2048
(config.num_labels, bias-free) applied to last_hidden_state BEFORE pooling.

Contract (modules.json + 1_Pooling/config.json + config_sentence_transformers.json):
  * pooling = MEAN over valid tokens of the 2048-dim projected states, then
    L2 normalize. Both are IN the graph here. include_prompt=true (the prompt
    tokens count toward the mean).
  * prompts on BOTH sides (unlike harrier's query-only instruction):
      query:    "Represent the query for retrieving supporting documents: "
      document: "Represent the document for retrieval: "
  * tokenizer post_processor is plain ByteLevel — NO auto-appended eos/bos
    (harrier's TemplateProcessing append does not exist here). add_bos False.
  * MRL: hosts may truncate the 2048 output to 1024/512/256 and re-normalize.
  * selling point is the SHARED EMBEDDING SPACE with the voyage-4 cloud series,
    so gate E (torch-encoded index x quantized queries) is the production gate.

Traps handled here:
  * None-mask export specialization: the remote forward special-cases
    attention_mask=None to all-ones inside a vmap'd create_causal_mask
    (rank-5 intermediates). We never trace that path — we hand-build the
    bidirectional+padding bias [1,1,S,S] and feed Qwen3Model's dict-mask
    walrus directly (same as the harrier/bekko lane):
    allowed(q,k) = valid[k] for every q. The reference mask reduces to the
    same thing: (causal | valid[k]) & valid[k] = valid[k]. No empty rows
    (valid tokens always exist), so no NaN guard needed.
  * `self.linear` 1024->2048 must be applied BEFORE the pool (per-token, as
    the remote code does). Mean commutes with a bias-free linear so pooling
    first would be mathematically equal, but we keep the reference order.
  * mean pool must divide by sum(mask), not S — pad-invariance + mask-liveness
    gates catch a graph that pools over pad rows.
  * tf5.x meta-load zero check on rotary inv_freq.
  * input_ids stay int32 end-to-end.

Quantization: wi8fc (FC int8 DRQ + EMBEDDING_LOOKUP int8 channelwise) + fp16.
Embedding table is 44% of the 309M params (155.6M) — wi8fc already covers it;
no separate embt8 variant (that was for bekko's 80% shape). The card's QAT
claim ("int8 output precision supported") is about OUTPUT vectors, not weights
— weight-int8 tolerance is measured, not assumed.
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

DEFAULT_MODEL = "voyageai/voyage-4-nano"
PAD_ID = 151643  # <|endoftext|>, tokenizer_config pad_token
DIM = 2048

QUERY_PREFIX = "Represent the query for retrieving supporting documents: "
DOC_PREFIX = "Represent the document for retrieval: "


class Embedder(nn.Module):
    """Qwen3Model body -> linear 1024->2048 per token -> masked mean -> L2 norm."""

    def __init__(self, wrapper):
        super().__init__()
        self.body = wrapper.model
        self.linear = wrapper.linear

    def forward(self, input_ids, attention_mask):
        S = input_ids.shape[1]
        valid = attention_mask.to(torch.bool)  # [1,S]
        zero = torch.zeros((), dtype=torch.float32)
        ninf = torch.full((), float("-inf"), dtype=torch.float32)

        allowed = valid[:, None, :].expand(1, S, S)  # [1,S,S], key-only gate
        bias = torch.where(allowed[:, None], zero, ninf)  # [1,1,S,S]

        h = self.body(
            input_ids=input_ids,
            attention_mask={"full_attention": bias},
            use_cache=False,
        ).last_hidden_state
        h = self.linear(h)  # [1,S,2048] BEFORE the pool (reference order)

        m = attention_mask.to(h.dtype)[:, :, None]  # [1,S,1]
        pooled = (h * m).sum(dim=1) / m.sum(dim=1)  # [1,2048]
        return F.normalize(pooled, p=2, dim=-1)


def load_model(model_id):
    # NOT AutoModel(trust_remote_code=True): the repo's remote code targets
    # transformers 4.51 and sets no `config_class`, which crashes 5.14's
    # AutoModel.register. We import the reviewed vendored copy instead and
    # patch the missing attribute.
    import importlib.util

    from transformers import Qwen3Config

    vendored = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "modeling_qwen3_bidirectional.py")
    assert os.path.exists(vendored), f"missing vendored remote code: {vendored}"
    spec = importlib.util.spec_from_file_location(
        "modeling_qwen3_bidirectional", vendored)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = module.Qwen3BidirectionalModel
    cls.config_class = Qwen3Config

    # Second 4.51-era incompatibility: the remote forward calls
    # create_causal_mask(input_embeds=..., cache_position=...); 5.x renamed the
    # former to inputs_embeds and dropped the latter. Shim it so the REFERENCE
    # path (used by the eager wrapper-vs-remote gate) runs unmodified.
    from transformers.masking_utils import create_causal_mask as _ccm_5x

    def _ccm_compat(config, input_embeds, attention_mask, cache_position=None,
                    past_key_values=None, position_ids=None,
                    or_mask_function=None):
        return _ccm_5x(config=config, inputs_embeds=input_embeds,
                       attention_mask=attention_mask,
                       past_key_values=past_key_values,
                       position_ids=position_ids,
                       or_mask_function=or_mask_function)

    module.create_causal_mask = _ccm_compat

    model = cls.from_pretrained(
        model_id, dtype=torch.float32, attn_implementation="sdpa",
    ).eval()
    assert type(model).__name__ == "Qwen3BidirectionalModel", type(model).__name__
    assert model.config.use_bidirectional_attention is True
    assert model.config.sliding_window is None
    assert all(not l.self_attn.is_causal for l in model.model.layers)
    assert model.linear.bias is None and list(model.linear.weight.shape) == [2048, 1024]
    inv = model.model.rotary_emb.inv_freq
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


def ref_pool(last_hidden, mask2d):
    """README's own mean_pool + normalize, on the wrapper's projected states."""
    m = mask2d[:, :, None].to(last_hidden.dtype)
    pooled = (last_hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)
    return F.normalize(pooled, p=2, dim=-1).numpy()


def assert_eager_gates(wrapper, mod, model_id, S=128):
    ids, mask, n_valid = sample(S)

    # BIdirectionality: editing the LAST valid token MUST move position 0's
    # hidden state (a causal graph would leave it untouched — the harrier
    # gate, inverted). Run the body with our bias, look at raw states.
    with torch.inference_mode():
        valid = mask.to(torch.bool)
        allowed = valid[:, None, :].expand(1, S, S)
        bias = torch.where(allowed[:, None], torch.zeros(()),
                           torch.full((), float("-inf")))
        h1 = mod.body(input_ids=ids, attention_mask={"full_attention": bias},
                      use_cache=False).last_hidden_state
        ids2 = ids.clone()
        ids2[0, n_valid - 1] = 42
        h2 = mod.body(input_ids=ids2, attention_mask={"full_attention": bias},
                      use_cache=False).last_hidden_state
    front = float((h1[0, 0] - h2[0, 0]).abs().max())
    print(f"EAGER bidirectionality: last-token edit moves pos0 by {front:.3e} "
          f"(must be >0 — causal would be 0)")
    assert front > 0.0, "attention is causal — is_causal flip or mask is wrong"

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

    # wrapper vs the remote code's own forward (2D-mask vmap path) — both
    # unpadded and right-padded, on real prompted texts
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    texts = [
        QUERY_PREFIX + "Which planet is known as the Red Planet?",
        DOC_PREFIX + "Mars, known for its reddish appearance, is often "
        "referred to as the Red Planet.",
        DOC_PREFIX + "富士山は日本で一番高い山です。",
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
            out_u = wrapper(input_ids=torch.tensor([tids]),
                            attention_mask=torch.ones(1, n, dtype=torch.long))
            ref_u = ref_pool(out_u.last_hidden_state,
                             torch.ones(1, n, dtype=torch.long))
            out_p = wrapper(input_ids=a2.to(torch.long),
                            attention_mask=m2.to(torch.long))
            ref_p = ref_pool(out_p.last_hidden_state, m2)
        cos_u = float(ours[0] @ ref_u[0])
        cos_p = float(ours[0] @ ref_p[0])
        worst = min(worst, cos_u, cos_p)
        print(f"EAGER wrapper-vs-remote ({n} tok): cos unpadded {cos_u:.8f} "
              f"padded {cos_p:.8f}")
    assert worst > 0.99999, "wrapper diverges from the remote-code path"


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
    ap.add_argument("out_dir", nargs="?", default="out/voyage_embed_nano")
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
    wrapper = load_model(args.model)
    emb = Embedder(wrapper).eval()
    n_params = sum(p.numel() for p in wrapper.parameters())
    cfg = wrapper.config
    print(f"params {n_params/1e6:.1f}M  hidden {cfg.hidden_size} "
          f"layers {cfg.num_hidden_layers} vocab {cfg.vocab_size} out {DIM}")

    assert_eager_gates(wrapper, emb, args.model)

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
