#!/usr/bin/env python3
"""Convert cross-encoder/ettin-reranker-400m-v1 (ModernBERT cross-encoder) to LiteRT.

    python convert_ettin_reranker.py <model_dir_or_id> out_ettin

Encoder lane, NOT export_hf. ModernBERT backbone (the bekko/mlateon dict-mask
lane) with the sentence-transformers v5 module-stack CrossEncoder head: the
graph emits ONE raw relevance score per (query, passage) pair; candidate
ranking lives host-side (one invoke per candidate).

Signatures (batch 1, right-padded static lengths):
  score_{128,256,512}: (input_ids i32 [1,S], attention_mask i32 [1,S])
      -> score f32 [1,1], RAW logit (see activation note below)

Model shape (config.json, read 2026-09-01): jhu-clsp/ettin-encoder-400m
ModernBertModel, 28L, hidden 1024, heads 16, GeLU, vocab 50368, rope theta
160000 both layer types, layer pattern full/sliding/sliding (global every
3rd), local_attention 128 -> config.sliding_window property = 64 (HALF
window, |q-k| <= 64 inclusive). `position_embedding_type: "sans_pos"` is an
ignored extra key in transformers 5.14.1, same as mLateOn — the vendor path
(sentence_transformers -> stock ModernBertModel) applies standard RoPE.

Head (modules.json + ST 5.4.1 wheel source, read 2026-09-01). This is NOT the
classic AutoModelForSequenceClassification CrossEncoder — it is the module
stack:
  1_Pooling  cls (config classifier_pooling "mean" is a base-model leftover;
             the ST module config says cls and the ST path is the vendor path)
  2_Dense    1024->1024, bias=False, activation GELU (exact/erf)  <- NOT
             linear, so unlike mLateOn the head CANNOT fold to one matrix;
             it is implemented in-graph as-is (tiny anyway)
  3_LayerNorm nn.LayerNorm(1024), weight+bias, eps 1e-5
  4_Dense    1024->1, bias=True, Identity -> "scores"

Host contract (ST 5.4.1 CrossEncoder.predict + base Transformer.preprocess,
read from the wheel 2026-09-01):
  * pairs tokenize as tokenizer([(q, d)], padding, truncation="longest_first")
    -> [CLS] q [SEP] d [SEP] (tokenizer.json TemplateProcessing pair; no
    token_type_ids, model_input_names is ids+mask only). pad 50283, cls 50281,
    sep 50282.
  * SCORING-API TRAP (the Shieldstral analog): predict() would default to
    sigmoid for num_labels=1, but config_sentence_transformers.json pins
    activation_fn = torch.nn.Identity. The published card scores
    ([3.6875, 11.6875, 4.75, 9.375]) are RAW logits. Hosts must NOT sigmoid
    if they want card-comparable numbers (sigmoid is monotonic, ranking is
    unaffected — but the scale contract is raw).
  * Card scores were printed from a bfloat16 run (model_kwargs dtype
    "bfloat16" in the usage snippet; every value is a multiple of 0.0625).
    fp32 reproduces them only to bf16 tolerance — the anchor gate here is
    rank order (passage 1 > 3 > 2 > 0) + max|diff| < 0.5, with exact fp32
    values recorded in the report for the verify stage to compare against.

Traps handled here (same walls as the mLateOn/bekko lane):
  * Sliding-window empty rows NaN: a pad query row at q >= n_valid + 64 has
    an empty +-64 window -> all-masked row -> SDPA NaN -> next layer's
    softmax gives pad KEYS weight 0 but 0 * NaN = NaN, which contaminates
    the CLS row and the pooled score. The hand bias allows the diagonal:
      full:    valid[k] | (q==k)
      sliding: (valid[k] | (q==k)) & (|q-k| <= 64)
    Gate: padded score == unpadded score (fp32), all-finite.
  * Masks are a hand-built dict {"full_attention", "sliding_attention"}
    [1,1,S,S] into stock ModernBERT's dict-mask walrus, gated against the
    vendor 2D-mask path padded AND unpadded.
  * Pad-content invariance: pads are masked keys and CLS pooling reads row 0
    only -> changing pad ids must change nothing, exactly.
  * tf5.x meta-load zero check on both rope inv_freq buffers.

Quantization: wi8fc (FC int8 DRQ + EMBEDDING_LOOKUP int8 channelwise) + fp16.
The 50368x1024 table is only ~13% of the 395M params — FC DRQ carries the
size win here, unlike bekko (80% table).
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

DEFAULT_MODEL = "cross-encoder/ettin-reranker-400m-v1"
PAD_ID = 50283
CLS_ID = 50281
SEP_ID = 50282

CARD_QUERY = "Which planet is known as the Red Planet?"
CARD_PASSAGES = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
]
CARD_SCORES = [3.6875, 11.6875, 4.75, 9.375]  # README predict(), bf16 run


class RerankScorer(nn.Module):
    """ModernBertModel body -> CLS pool -> Dense/GELU -> LayerNorm -> Dense -> raw score."""

    def __init__(self, body, w2, ln_w, ln_b, w4, b4):
        super().__init__()
        self.body = body
        self.half_window = int(body.config.sliding_window)  # 64 = local_attention//2
        h = body.config.hidden_size
        self.dense2 = nn.Linear(h, h, bias=False)
        self.norm3 = nn.LayerNorm(h)
        self.dense4 = nn.Linear(h, 1, bias=True)
        with torch.no_grad():
            self.dense2.weight.copy_(w2)
            self.norm3.weight.copy_(ln_w)
            self.norm3.bias.copy_(ln_b)
            self.dense4.weight.copy_(w4)
            self.dense4.bias.copy_(b4)
        self.act2 = nn.GELU()  # exact/erf — 2_Dense config says torch.nn GELU

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
        ).last_hidden_state                                      # [1,S,1024]
        cls = h[:, 0]                                            # [1,1024]
        x = self.norm3(self.act2(self.dense2(cls)))
        return self.dense4(x)                                    # [1,1] raw score


def load_model(model_id):
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        model_id, dtype=torch.float32, attn_implementation="sdpa"
    ).eval()
    assert type(model).__name__ == "ModernBertModel", type(model).__name__
    assert model.config.num_hidden_layers == 28
    assert int(model.config.sliding_window) == 64, model.config.sliding_window
    assert model.config.layer_types[0] == "full_attention"
    for name, buf in model.named_buffers():
        if "inv_freq" in name:
            assert float(buf.abs().min()) > 0, f"{name} zeroed by meta-load"
    return model


def load_head(model_id):
    """Load the three ST head modules; assert the configs we read from the repo."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    def sub(d):
        if os.path.isdir(model_id):
            p = os.path.join(model_id, d, "model.safetensors")
            cfgp = os.path.join(model_id, d, "config.json")
        else:
            p = hf_hub_download(model_id, f"{d}/model.safetensors")
            cfgp = hf_hub_download(model_id, f"{d}/config.json")
        return json.load(open(cfgp)), load_file(p)

    pool_cfg, _ = (json.load(open(os.path.join(model_id, "1_Pooling", "config.json"))), None) \
        if os.path.isdir(model_id) else \
        (json.load(open(hf_hub_download(model_id, "1_Pooling/config.json"))), None)
    assert pool_cfg["pooling_mode"] == "cls", pool_cfg

    c2, w2 = sub("2_Dense")
    assert (c2["in_features"], c2["out_features"], c2["bias"]) == (1024, 1024, False), c2
    assert c2["activation_function"].endswith("GELU"), c2

    c3, w3 = sub("3_LayerNorm")
    assert c3 == {"dimension": 1024}, c3

    c4, w4 = sub("4_Dense")
    assert (c4["in_features"], c4["out_features"], c4["bias"]) == (1024, 1, True), c4
    assert c4["activation_function"].endswith("Identity"), c4

    return (w2["linear.weight"], w3["norm.weight"], w3["norm.bias"],
            w4["linear.weight"], w4["linear.bias"])


# ------------------------------------------------------------- host contract
def pair_tokens(tok, query, doc, max_len=512):
    """[CLS] q [SEP] d [SEP], truncation longest_first — the vendor call."""
    e = tok(query, doc, truncation="longest_first", max_length=max_len)["input_ids"]
    assert e[0] == CLS_ID and e[-1] == SEP_ID
    return np.array(e, np.int32)


def pad_to(ids, S):
    n = min(len(ids), S)
    a = np.full((1, S), PAD_ID, np.int32)
    m = np.zeros((1, S), np.int32)
    a[0, :n] = ids[:n]
    m[0, :n] = 1
    return torch.from_numpy(a), torch.from_numpy(m), n


def pair_sample(S, frac=0.7, seed=1):
    """Random pair-shaped ids: [CLS] ... [SEP] ... [SEP] + pads."""
    g = torch.Generator().manual_seed(seed)
    n = max(6, int(S * frac))
    ids = torch.full((1, S), PAD_ID, dtype=torch.int32)
    mask = torch.zeros(1, S, dtype=torch.int32)
    body = torch.randint(1000, 50000, (n - 3,), generator=g, dtype=torch.int32)
    ids[0, 0] = CLS_ID
    q_len = max(2, (n - 3) // 4)
    ids[0, 1:1 + q_len] = body[:q_len]
    ids[0, 1 + q_len] = SEP_ID
    ids[0, 2 + q_len:n - 1] = body[q_len:]
    ids[0, n - 1] = SEP_ID
    mask[0, :n] = 1
    return ids, mask, n


def torch_ref(mod, ids, mask):
    with torch.inference_mode():
        return mod(ids, mask).numpy()


def eager_checks(scorer, model, model_id):
    print("\n-- eager gates --")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    assert tok.pad_token_id == PAD_ID
    assert tok.cls_token_id == CLS_ID and tok.sep_token_id == SEP_ID

    def head(cls_vec):
        with torch.inference_mode():
            x = scorer.norm3(scorer.act2(scorer.dense2(cls_vec)))
            return scorer.dense4(x)

    # (a) hand dict masks vs the vendor stock path (2D mask), padded+unpadded
    ids_np = pair_tokens(tok, CARD_QUERY, CARD_PASSAGES[1])
    n = len(ids_np)
    ids_u = torch.from_numpy(ids_np[None]).to(torch.int32)
    ones = torch.ones(1, n, dtype=torch.int32)
    with torch.inference_mode():
        h = model(input_ids=ids_u, attention_mask=ones).last_hidden_state
        vendor_u = float(head(h[:, 0])[0, 0])
    ours_u = float(torch_ref(scorer, ids_u, ones)[0, 0])
    ids_p, mask_p, _ = pad_to(ids_np, 128)
    with torch.inference_mode():
        hp = model(input_ids=ids_p, attention_mask=mask_p).last_hidden_state
        vendor_p = float(head(hp[:, 0])[0, 0])
    ours_p = float(torch_ref(scorer, ids_p, mask_p)[0, 0])
    print(f"masks vs vendor 2D path: unpadded |diff| {abs(ours_u - vendor_u):.3e}, "
          f"padded {abs(ours_p - vendor_p):.3e}")
    assert abs(ours_u - vendor_u) < 1e-4 and abs(ours_p - vendor_p) < 1e-4

    # (b) NaN wall: short pair in the 512 signature. If a pad row went NaN it
    #     contaminates CLS through 0*NaN — so finite AND == unpadded is the gate.
    ids_p, mask_p, _ = pad_to(ids_np, 512)
    s512 = float(torch_ref(scorer, ids_p, mask_p)[0, 0])
    assert np.isfinite(s512), "NaN reached the score — diagonal guard failed"
    print(f"512-sig vs unpadded: |diff| {abs(s512 - ours_u):.3e}")
    assert abs(s512 - ours_u) < 1e-4

    # (c) pad-content invariance
    ids2 = ids_p.clone()
    g2 = torch.Generator().manual_seed(7)
    ids2[0, n:] = torch.randint(1000, 50000, (512 - n,), generator=g2,
                                dtype=torch.int32)
    s2 = float(torch_ref(scorer, ids2, mask_p)[0, 0])
    print(f"pad-content invariance: |diff| {abs(s512 - s2):.3e} (must be 0)")
    assert s512 == s2

    # (d) the card's own example. Published values are from a bf16 run — gate
    #     rank order strictly, values to bf16-drift tolerance, record fp32.
    scores = []
    for doc in CARD_PASSAGES:
        t = pair_tokens(tok, CARD_QUERY, doc)
        scores.append(float(torch_ref(scorer, *pad_to(t, 128)[:2])[0, 0]))
    err = max(abs(a - b) for a, b in zip(scores, CARD_SCORES))
    print(f"card scores fp32: {[round(s, 4) for s in scores]} "
          f"(published bf16 {CARD_SCORES}) max|diff| {err:.4f}")
    order = np.argsort(scores)[::-1].tolist()
    assert order == [1, 3, 2, 0], f"rank order broke: {order}"
    assert err < 0.5, "fp32 drifts from the bf16 card values beyond bf16 tolerance"
    return scores


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


def check_variant(path, tag, scorer, seq_lens, results):
    for S in sorted(seq_lens):
        ids, mask, n = pair_sample(S, frac=0.7)
        ref = float(torch_ref(scorer, ids, mask)[0, 0])
        got = float(run_sig(path, f"score_{S}", ids, mask)[0, 0])
        assert np.isfinite(got), f"score_{S} {tag}: non-finite score"
        print(f"SMOKE score_{S} {tag} vs torch: {got:.4f} vs {ref:.4f} "
              f"|diff| {abs(got - ref):.3e}")
        results["sigs"].setdefault(f"score_{S}", {})[f"{tag}_absdiff"] = abs(got - ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    ap.add_argument("out_dir", nargs="?", default="out_ettin")
    ap.add_argument("--seqs", default="512,256,128")
    ap.add_argument("--skip-fp16", action="store_true")
    args = ap.parse_args()

    seq_lens = [int(s) for s in args.seqs.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)
    fp32 = os.path.join(args.out_dir, "reranker_fp32.tflite")
    wi8 = os.path.join(args.out_dir, "reranker_wi8fc.tflite")
    fp16 = os.path.join(args.out_dir, "reranker_fp16.tflite")

    torch.manual_seed(0)
    print(f"loading {args.model} ...")
    model = load_model(args.model)
    scorer = RerankScorer(model, *load_head(args.model)).eval()
    n_params = sum(p.numel() for p in scorer.parameters())
    print(f"params {n_params/1e6:.1f}M  vocab {model.config.vocab_size}")

    card_fp32 = eager_checks(scorer, model, args.model)

    print(f"\nconverting (litert_torch multi-signature) seqs={seq_lens} ...")
    import litert_torch

    conv = None
    for S in seq_lens:
        ids, mask, _ = pair_sample(S, frac=0.7)
        kw = {"input_ids": ids, "attention_mask": mask}
        sig = (f"score_{S}", scorer)
        conv = (litert_torch.signature(*sig, sample_kwargs=kw) if conv is None
                else conv.signature(*sig, sample_kwargs=kw))
    edge = conv.convert()
    edge.export(fp32)
    print(f"fp32: {os.path.getsize(fp32)/1e6:.1f} MB")

    results = {"model": args.model, "params_m": n_params / 1e6,
               "seqs": seq_lens, "pad_id": PAD_ID,
               "pair_template": "[CLS] q [SEP] d [SEP]",
               "score": "raw logit (activation_fn Identity — do NOT sigmoid)",
               "card_scores_fp32": card_fp32,
               "ops": op_report(fp32, "fp32"), "sigs": {}}
    assert "EMBEDDING_LOOKUP" in results["ops"], (
        "embedding lowered to GATHER, not EMBEDDING_LOOKUP — the int8 recipe "
        "would miss the 50368x1024 table")
    check_variant(fp32, "fp32", scorer, seq_lens, results)

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
    check_variant(wi8, "wi8fc", scorer, seq_lens, results)

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
        check_variant(fp16, "fp16", scorer, seq_lens, results)

    results["sizes_mb"] = {t: round(os.path.getsize(p) / 1e6, 1)
                           for t, p in (("fp32", fp32), ("wi8fc", wi8), ("fp16", fp16))
                           if os.path.exists(p)}
    with open(os.path.join(args.out_dir, "convert_report.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("DONE:", fp32, wi8)


if __name__ == "__main__":
    main()
