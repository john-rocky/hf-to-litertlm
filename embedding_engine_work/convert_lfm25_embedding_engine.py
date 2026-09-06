#!/usr/bin/env python3
"""LiquidAI/LFM2.5-Embedding-350M -> the LiteRT-LM 0.17.0 EmbeddingEngine bundle contract.

    .venv-092/bin/python scripts/convert_lfm25_embedding_engine.py <model_dir_or_id> out/lfm25_embed_350m_engine

Same split as convert_granite_embedding_r2_engine.py (see that docstring for the contract):
  TF_LITE_EMBEDDER      `token` int32[1] -> raw `embed_tokens` row f32[1,1,1024]
  TF_LITE_TEXT_ENCODER  `encoder_<S>`: `embeddings` f32[1,S,1024] + `input_mask` f32[1,S] -> CLS-pooled,
                        L2-normalized f32[1,1024]
Everything model-specific (the remote-code kwarg filter, forced FA2-style pad zeroing, the
dict mask {"full_attention", "conv"}, rank-4 repeat_kv) is inherited from convert_lfm25_embedding.py.

Split-specific guard: the runtime fills only the first `len(tokens)` rows of the `embeddings`
buffer, so pad rows hold whatever the buffer held before (zeros on a fresh buffer, a previous
text's rows afterwards). The fused graph zeroed pad rows by multiplying with the mask inside the
conv layers; a multiply would turn a non-finite stale row into NaN, so the encoder first REPLACES
pad rows with zeros via torch.where — no arithmetic touches stale content.
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
import _stub  # noqa: F401
import _rank4_repeat_kv  # noqa: F401  (installed by load_model)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from convert_lfm25_embedding import (  # noqa: E402
    BOS_ID, DEFAULT_MODEL, NEG, Embedder, load_model, sample, torch_ref,
)
from embedding_engine_common import (  # noqa: E402  shared by every EmbeddingEngine export
    Lookup, op_report, quantize, run_encoder, run_lookup, trace_rows,
)


class Encoder(nn.Module):
    def __init__(self, body):
        super().__init__()
        self.body = body

    def forward(self, embeddings, input_mask):
        m = (input_mask > 0.5).to(torch.float32)                        # [1,S]
        emb = torch.where((m > 0.5)[:, :, None], embeddings, torch.zeros((), dtype=embeddings.dtype))
        full = ((1.0 - m) * NEG)[:, None, None, :]                       # [1,1,1,S]
        h = self.body(
            inputs_embeds=emb,
            attention_mask={"full_attention": full, "conv": m},
            use_cache=False,
        ).last_hidden_state
        return F.normalize(h[:, 0], p=2, dim=-1)


def compose_ref(lookup, encoder, ids, mask):
    with torch.inference_mode():
        rows = torch.cat([lookup(ids[0, i:i + 1]) for i in range(ids.shape[1])], dim=1)
        return encoder(rows, mask.to(torch.float32)).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    ap.add_argument("out_dir", nargs="?", default="out/lfm25_embed_350m_engine")
    ap.add_argument("--seqs", default="512,256,128,64")
    ap.add_argument("--skip-fp16", action="store_true")
    args = ap.parse_args()
    seq_lens = [int(s) for s in args.seqs.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)
    P = lambda n: os.path.join(args.out_dir, n)  # noqa: E731

    torch.manual_seed(0)
    model = load_model(args.model)
    fused = Embedder(model).eval()
    lookup = Lookup(model.embed_tokens).eval()
    encoder = Encoder(model).eval()

    for S in (64, 128):
        ids, mask, _ = sample(S)
        a = torch_ref(fused, ids, mask)
        b = compose_ref(lookup, encoder, ids, mask)
        print(f"EAGER split-vs-fused S={S}: max|diff| {np.abs(a-b).max():.3e} cos {float(a[0]@b[0]):.8f}")
        assert np.abs(a - b).max() < 1e-4
        # stale/non-finite pad rows must not reach the output
        ids2, mask2, n_valid = sample(S)
        with torch.inference_mode():
            rows = torch.cat([lookup(ids2[0, i:i + 1]) for i in range(S)], dim=1).clone()
            rows[0, n_valid:] = float("nan")
            c = encoder(rows, mask2.to(torch.float32)).numpy()
        ref = torch_ref(fused, ids2, mask2)
        print(f"EAGER NaN-in-pad-rows S={S}: finite {np.isfinite(c).all()} max|diff| vs fused {np.abs(ref-c).max():.3e}")
        assert np.isfinite(c).all() and np.abs(ref - c).max() < 1e-4

    import litert_torch
    tok = torch.tensor([BOS_ID], dtype=torch.int32)
    litert_torch.signature("embedder", lookup, sample_kwargs={"token": tok}).convert().export(P("embedder_fp32.tflite"))
    op_report(P("embedder_fp32.tflite"), "embedder fp32")

    conv = None
    for S in seq_lens:
        ids, mask, _ = sample(S)
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
            ids, mask, _ = sample(S)
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
        smoke("fp16", P("embedder_wi8.tflite"), P("encoder_fp16.tflite"))

    with open(P("convert_report.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("DONE:", args.out_dir)


if __name__ == "__main__":
    main()
