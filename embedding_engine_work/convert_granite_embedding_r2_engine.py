#!/usr/bin/env python3
"""granite-embedding-311m-multilingual-r2 -> the LiteRT-LM 0.17.0 EmbeddingEngine bundle contract.

    .venv-092/bin/python scripts/convert_granite_embedding_r2_engine.py <model_dir_or_id> out/granite_embed_311m_engine

The shipped .tflite (scripts/convert_granite_embedding_r2.py) is one graph:
(input_ids, attention_mask) -> CLS-pooled, L2-normalized [1,768]. The runtime's
EmbeddingEngine (litert-lm 0.17.0, runtime/executor/embedding_litert_compiled_model_executor.cc
+ runtime/testdata/test_embedding.litertlm) wants that split in two tflites:

  TF_LITE_EMBEDDER      one signature, ONE input `token` int32[1] (4 bytes), output f32 [1,1,D]:
                        the token-embedding table lookup. The runtime calls it per token and
                        writes the rows into the encoder's `embeddings` buffer.
  TF_LITE_TEXT_ENCODER  signatures named `encoder_<S>`, inputs `embeddings` f32 [1,S,D] and
                        `input_mask` f32 [1,S] (the runtime fills 1.0 for real tokens and 0.0
                        beyond), output `encodings` f32 [1,dim] — pooling stays IN-GRAPH.

Everything else (hand-built alternating full/sliding masks, the diagonal on pad rows,
CLS pooling, L2 normalize) is carried over unchanged from convert_granite_embedding_r2.py;
the encoder feeds the rows through `ModernBertEmbeddings.norm` exactly as the fused graph
did, because the embedder emits the RAW table rows (`tok_embeddings`), not normed ones.

Smoke: for every S, torch(fused Embedder) vs [lookup tflite -> encoder tflite] composed.
Quant: wi8fc = FULLY_CONNECTED int8 DRQ on the encoder + EMBEDDING_LOOKUP int8 channelwise
on the embedder (the 262152x768 table is ~65% of the parameters).
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# Works in both trees: private (everything under scripts/) and the public mirror (this file
# under embedding_engine_work/, the fused exports under their lane dirs, _stub under scripts/).
for _d in (_HERE, *[os.path.join(_HERE, "..", d) for d in
                    ("scripts", "granite_embed_work", "lfm_work", "lfm25_embed_work", "nemotron_work", "nemotron_embed_work")]):
    if os.path.isdir(_d) and _d not in sys.path:
        sys.path.insert(0, _d)
import _stub  # noqa: F401  (macOS scipy/_propack guard, import FIRST)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from convert_granite_embedding_r2 import (  # the fused reference + helpers
    DEFAULT_MODEL, NEG, PAD_ID, Embedder, load_model, sample, torch_ref,
)
from embedding_engine_common import (  # noqa: E402  shared by every EmbeddingEngine export
    Lookup, op_report, quantize, run_encoder, run_lookup, trace_rows,
)


class Encoder(nn.Module):
    """embeddings f32[1,S,D] + input_mask f32[1,S] -> CLS-pooled, L2-normalized f32[1,D]."""

    def __init__(self, body, window):
        super().__init__()
        self.body = body
        self.window = window

    def forward(self, embeddings, input_mask):
        S = embeddings.shape[1]
        valid = (input_mask > 0.5)[:, None, None, :]                    # [1,1,1,S]
        zero = torch.zeros((), dtype=torch.float32)
        neg = torch.full((), NEG, dtype=torch.float32)
        full = torch.where(valid, zero, neg)
        q = torch.arange(S, device=embeddings.device)[:, None]
        k = torch.arange(S, device=embeddings.device)[None, :]
        band = ((q - k).abs() <= self.window)[None, None]
        eye = (q == k)[None, None]
        sliding = torch.where((valid & band) | eye, zero, neg)
        h = self.body(
            inputs_embeds=embeddings,   # ModernBertEmbeddings applies norm+drop to inputs_embeds
            attention_mask={"full_attention": full, "sliding_attention": sliding},
        ).last_hidden_state
        return F.normalize(h[:, 0], p=2, dim=-1)


def compose_ref(lookup, encoder, ids, mask):
    with torch.inference_mode():
        rows = torch.cat([lookup(ids[0, i:i + 1]) for i in range(ids.shape[1])], dim=1)
        return encoder(rows, mask.to(torch.float32)).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    ap.add_argument("out_dir", nargs="?", default="out/granite_embed_311m_engine")
    ap.add_argument("--seqs", default="512,256,128,64")
    ap.add_argument("--skip-fp16", action="store_true")
    args = ap.parse_args()
    seq_lens = [int(s) for s in args.seqs.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)
    P = lambda n: os.path.join(args.out_dir, n)  # noqa: E731

    torch.manual_seed(0)
    model, window = load_model(args.model)
    cls_id = model.config.cls_token_id
    fused = Embedder(model, window).eval()
    lookup = Lookup(model.embeddings.tok_embeddings).eval()
    encoder = Encoder(model, window).eval()

    # Eager: fused graph == lookup∘encoder (bitwise up to float noise)
    for S in (64, 128):
        ids, mask, _ = sample(S, cls_id=cls_id)
        a = torch_ref(fused, ids, mask)
        b = compose_ref(lookup, encoder, ids, mask)
        print(f"EAGER split-vs-fused S={S}: max|diff| {np.abs(a-b).max():.3e} cos {float(a[0]@b[0]):.8f}")
        assert np.abs(a - b).max() < 1e-4

    import litert_torch

    # --- embedder (one signature, one int32[1] input)
    tok = torch.tensor([cls_id], dtype=torch.int32)
    litert_torch.signature("embedder", lookup, sample_kwargs={"token": tok}).convert().export(P("embedder_fp32.tflite"))
    op_report(P("embedder_fp32.tflite"), "embedder fp32")

    # --- encoder (encoder_<S> signatures; inputs `embeddings` + `input_mask`)
    conv = None
    for S in seq_lens:
        ids, mask, _ = sample(S, cls_id=cls_id)
        rows = trace_rows(lookup, ids)
        kw = {"embeddings": rows, "input_mask": mask.to(torch.float32)}
        sig = (f"encoder_{S}", encoder)
        conv = (litert_torch.signature(*sig, sample_kwargs=kw) if conv is None
                else conv.signature(*sig, sample_kwargs=kw))
    conv.convert().export(P("encoder_fp32.tflite"))
    op_report(P("encoder_fp32.tflite"), "encoder fp32")

    results = {"model": args.model, "sigs": {}}

    def smoke(tag, emb_path, enc_path):
        for S in sorted(seq_lens):
            ids, mask, _ = sample(S, cls_id=cls_id)
            ref = torch_ref(fused, ids, mask)
            got = run_encoder(enc_path, S, run_lookup(emb_path, ids), mask)
            assert np.isfinite(got).all(), f"{tag} encoder_{S} output is not finite"
            cos = float(ref[0] @ got[0])
            print(f"SMOKE {tag} encoder_{S} (lookup->encoder) vs torch fused: max|diff| "
                  f"{np.abs(ref-got).max():.3e} cos {cos:.8f}")
            results["sigs"].setdefault(f"encoder_{S}", {})[f"{tag}_cos"] = cos

    smoke("fp32", P("embedder_fp32.tflite"), P("encoder_fp32.tflite"))

    from ai_edge_quantizer import qtyping
    G, OP = qtyping.QuantGranularity, qtyping.TFLOperationName
    print("quantizing wi8fc ...")
    quantize(P("encoder_fp32.tflite"), P("encoder_wi8fc.tflite"),
             lambda rm: rm.add_dynamic_config(regex=".*", operation_name=OP.FULLY_CONNECTED, num_bits=8))
    quantize(P("embedder_fp32.tflite"), P("embedder_wi8.tflite"),
             lambda rm: rm.add_dynamic_config(regex=".*", operation_name=OP.EMBEDDING_LOOKUP,
                                              num_bits=8, granularity=G.CHANNELWISE))
    smoke("wi8fc", P("embedder_wi8.tflite"), P("encoder_wi8fc.tflite"))

    if not args.skip_fp16:
        print("quantizing fp16 ...")

        def fp16_recipe(rm):
            rm.add_quantization_config(
                regex=".*", operation_name=OP.ALL_SUPPORTED, algorithm_key="float_casting",
                op_config=qtyping.OpQuantizationConfig(
                    weight_tensor_config=qtyping.TensorQuantizationConfig(
                        num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
                    compute_precision=qtyping.ComputePrecision.FLOAT))
        quantize(P("encoder_fp32.tflite"), P("encoder_fp16.tflite"), fp16_recipe)
        # The embedder table stays fp32 in-graph: the converter marks EMBEDDING_LOOKUP
        # illegal on f16 tables ([[qwen3-tts]] gotcha), so an fp16 embedder would only
        # materialize the fp32 table at prepare time. Pair encoder_fp16 with embedder_wi8.
        smoke("fp16", P("embedder_wi8.tflite"), P("encoder_fp16.tflite"))

    with open(P("convert_report.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("DONE:", args.out_dir)


if __name__ == "__main__":
    main()
