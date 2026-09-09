#!/usr/bin/env python3
"""ai-sage/Giga-Embeddings-instruct-480M-0826 -> the LiteRT-LM 0.17.0 EmbeddingEngine bundle contract.

    python embedding_engine_work/convert_giga_embed_engine.py [model_dir_or_id] out/giga_embed_480m_engine

Same split as convert_granite_embedding_r2_engine.py / convert_nemotron3_embed_engine.py
(contract in embedding_engine_common.py):
  TF_LITE_EMBEDDER      `token` int32[1] -> raw `embed_tokens` row f32[1,1,1024]
  TF_LITE_TEXT_ENCODER  `encoder_<S>`: `embeddings` f32[1,S,1024] + `input_mask` f32[1,S]
                        -> mean over valid positions (specials + prompt included, as
                        sentence-transformers does with include_prompt=True), L2-normalized f32[1,1024]
The bidirectional additive bias is built from `input_mask` exactly as convert_giga_embed.py builds
it from attention_mask. Stale pad rows in the runtime's `embeddings` buffer are replaced with zeros
via torch.where before the body, so a non-finite stale row can never reach the mean.

Special tokens: the model's tokenizer post-processor is `<s> A </s>` and the runtime does NOT run
it, so the bundle's EmbeddingMetadata declares bos = [1] and eos = [2] (see
embedding_engine_work/pack_giga_embed.sh) and the engine's insert_special_tokens (default true)
inserts both. Turning the option off drops both specials from the mean and returns a different
vector — the card must say so.
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# Works in both trees: private (everything under scripts/) and the public mirror (this file
# under embedding_engine_work/, the fused export under giga_work/, _stub under scripts/).
for _d in (_HERE, *[os.path.join(_HERE, "..", d) for d in ("scripts", "giga_work")]):
    if os.path.isdir(_d) and _d not in sys.path:
        sys.path.insert(0, _d)
import _stub  # noqa: F401

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from convert_giga_embed import (  # noqa: E402
    DEFAULT_MODEL, Embedder, load_model, sample, torch_ref,
)
from embedding_engine_common import (  # noqa: E402  shared by every EmbeddingEngine export
    Lookup, op_report, quantize, run_encoder, run_lookup, trace_rows,
)


class Encoder(nn.Module):
    def __init__(self, body):
        super().__init__()
        self.body = body

    def forward(self, embeddings, input_mask):
        valid1 = input_mask > 0.5                                          # [1,S]
        emb = torch.where(valid1[:, :, None], embeddings, torch.zeros((), dtype=embeddings.dtype))
        valid = valid1[:, None, None, :]                                   # [1,1,1,S]
        bias = torch.where(valid, torch.zeros((), dtype=torch.float32),
                           torch.full((), float("-inf"), dtype=torch.float32))
        h = self.body(inputs_embeds=emb, attention_mask=bias).last_hidden_state
        m = valid1.to(h.dtype)[:, :, None]
        pooled = (h * m).sum(dim=1) / m.sum(dim=1)
        return F.normalize(pooled, p=2, dim=-1)


def compose_ref(lookup, encoder, ids, mask):
    with torch.inference_mode():
        rows = torch.cat([lookup(ids[0, i:i + 1]) for i in range(ids.shape[1])], dim=1)
        return encoder(rows, mask.to(torch.float32)).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    ap.add_argument("out_dir", nargs="?", default="out/giga_embed_480m_engine")
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
        ids, mask, n_valid = sample(S)
        a = torch_ref(fused, ids, mask)
        b = compose_ref(lookup, encoder, ids, mask)
        print(f"EAGER split-vs-fused S={S}: max|diff| {np.abs(a-b).max():.3e} cos {float(a[0]@b[0]):.8f}")
        assert np.abs(a - b).max() < 1e-4
        with torch.inference_mode():
            rows = torch.cat([lookup(ids[0, i:i + 1]) for i in range(S)], dim=1).clone()
            rows[0, n_valid:] = float("nan")
            c = encoder(rows, mask.to(torch.float32)).numpy()
        print(f"EAGER NaN-in-pad-rows S={S}: finite {np.isfinite(c).all()} max|diff| vs fused {np.abs(a-c).max():.3e}")
        assert np.isfinite(c).all() and np.abs(a - c).max() < 1e-4

    import litert_torch
    tok = torch.tensor([1], dtype=torch.int32)
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
