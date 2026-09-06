"""Shared pieces of the EmbeddingEngine (litert-lm >= 0.17.0) bundle exports.

The runtime's bundle contract (runtime/executor/embedding_litert_compiled_model_executor.cc +
runtime/testdata/test_embedding.litertlm, verified by running on 0.17.0):

  TF_LITE_EMBEDDER      one signature, ONE int32[1] input (`token`), output f32 rank >= 3
                        ([1,1,D]) — the runtime calls it per token and fills the encoder's
                        `embeddings` buffer with the rows.
  TF_LITE_TEXT_ENCODER  signatures named `encoder*`, inputs whose names contain `embeddings`
                        ([1,S,D] or [S,D]) and optionally `input_mask` (f32 [1,S], runtime
                        writes 1.0/0.0) — output index 0 is the final vector (pooling
                        IN-GRAPH); the smallest signature >= token count is used.
  tokenizer             the runtime does NOT run the HF tokenizer's post-processor: any
                        bos/eos the model needs go into EmbeddingMetadata.bos_token/eos_token
                        and are inserted by the engine's insert_special_tokens (default true).
"""
import collections
import os

import numpy as np
import torch
import torch.nn as nn


class Lookup(nn.Module):
    """token int32[1] -> raw embedding row f32[1,1,D] (pre-norm, as the fused graph reads it)."""

    def __init__(self, table):
        super().__init__()
        self.table = table

    def forward(self, token):
        return self.table(token)[None]


def run_lookup(path, ids):
    """Drive an embedder .tflite the way the runtime does: one token per invoke -> [1,S,D]."""
    from ai_edge_litert.interpreter import Interpreter

    it = Interpreter(model_path=path)
    sig = list(it.get_signature_list())[0]
    runner = it.get_signature_runner(sig)
    in_name = list(it.get_signature_list()[sig]["inputs"])[0]
    rows = [list(runner(**{in_name: ids[0, i:i + 1].numpy()}).values())[0] for i in range(ids.shape[1])]
    return np.concatenate(rows, axis=1)


def run_encoder(path, S, rows, mask):
    from ai_edge_litert.interpreter import Interpreter

    it = Interpreter(model_path=path, num_threads=os.cpu_count())
    r = it.get_signature_runner(f"encoder_{S}")(embeddings=rows.astype(np.float32),
                                                 input_mask=mask.numpy().astype(np.float32))
    return list(r.values())[0]


def op_report(path, tag):
    from ai_edge_litert.interpreter import Interpreter

    it = Interpreter(model_path=path)
    hist = collections.Counter(d["op_name"] for d in it._get_ops_details())
    print(f"{tag} ops:", dict(sorted(hist.items(), key=lambda kv: -kv[1])))
    sigs = it.get_signature_list()
    print(f"{tag} signatures:", {k: (v["inputs"], v["outputs"]) for k, v in sigs.items()})
    for d in it.get_input_details():
        print(f"   in  {d['name']} {d['dtype'].__name__} {list(d['shape'])}")
    for d in it.get_output_details():
        print(f"   out {d['name']} {d['dtype'].__name__} {list(d['shape'])}")


def quantize(src, dst, recipe_fn):
    from ai_edge_quantizer import quantizer, recipe_manager

    rm = recipe_manager.RecipeManager()
    recipe_fn(rm)
    qt = quantizer.Quantizer(src, rm.get_quantization_recipe())
    assert not qt.need_calibration
    if os.path.exists(dst):
        os.remove(dst)
    qt.quantize().export_model(dst)
    print(f"{os.path.basename(dst)}: {os.path.getsize(dst)/1e6:.1f} MB")


def trace_rows(lookup, ids):
    """Sample `embeddings` for a trace: ordinary tensors, not inference tensors (torch.export
    refuses those: "Inference tensors cannot be saved for backward")."""
    with torch.no_grad():
        return torch.cat([lookup(ids[0, i:i + 1]) for i in range(ids.shape[1])], dim=1).detach().clone()
