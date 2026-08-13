#!/usr/bin/env python3
"""Convert LiquidAI/LFM2.5-Encoder-{350M,230M} (bidirectional LFM2 encoders) to LiteRT.

    python convert_lfm25_encoder.py LiquidAI/LFM2.5-Encoder-230M out_lfm25_encoder_230m

Unlike the LFM2.5 decoder conversions this is NOT an export_hf run: the model is
a BERT-style bidirectional encoder (masked LM) with no KV cache, so we trace the
HF eager model directly with litert_torch multi-signature convert.

Signatures (batch 1, right-padded static lengths):
  encode_{64,128,256,512}: (input_ids i32 [1,S], attention_mask i32 [1,S])
      -> last_hidden_state f32 [1,S,1024], zeroed at padded positions.
  mlm_128: same inputs -> masked-LM logits f32 [1,128,65536] (tied head; the
      vocab matrix dedupes against the embedding table in the flatbuffer).

Tested with litert-torch 0.9.2 / transformers 5.14.1 / torch 2.12.

Traps handled here (all verified against the installed transformers sources):
  * transformers' apply_mask_to_padding_states is a NO-OP at batch==1
    (`attention_mask.shape[0] > 1` guard). Static tflite signatures are
    exactly batch-1 right-padded input, so without a fix the symmetric
    ShortConv (padding=k//2) reads pad-token embeddings past the sequence end
    and the last valid tokens diverge from the unpadded reference. We rebind
    the symbol in the remote-code module (patching
    transformers.models.lfm2.modeling_lfm2 does nothing — the remote module
    imported the function by value) to an unconditional multiply; zeroed pad
    inputs make the padded forward token-exact vs the unpadded one (conv1d
    zero-pads at the true boundary anyway). The gate below is pad-CONTENT
    invariance (must be bitwise 0); padded-vs-unpadded abs diff carries
    shape-dependent reduction-order float noise (~1e-4) and is informational.
  * transformers 5.x meta-load can zero init-computed buffers (rope inv_freq)
    for remote-code models -> assert inv_freq > 0 after load.
  * input_ids stay int32 end-to-end (F.embedding accepts int32; no CAST/int64
    chains in the graph).

Quantization: wi8fc — dynamic-range int8 on FULLY_CONNECTED + int8 channelwise
on the embedding table, depthwise ShortConv convs stay float (ALL_SUPPORTED
int8 is a proven conv-killer on LFM2 hybrids, see REPRODUCE.md). A near-lossless
fp16 variant is also written. On-device note: the int8 file is the mobile
artifact (verified bit-exact vs desktop on an iPhone 17 Pro); the fp16 file's
XNNPACK init unpacks weights to fp32 per signature subgraph and exceeds phone
memory limits — treat fp16 as desktop-only.
"""
import os
import sys
import types

# macOS/prebuilt-scipy guard: scipy >= 1.18 pulls _propack eagerly through the
# sparse.csgraph import chain (the converter's layout pass uses
# csgraph.maximum_flow, which itself is unaffected). Stub the propack module so
# the import survives; harmless where the real module works.
class _D:  # noqa: E301
    def __getattr__(self, n):
        return lambda *a, **k: None

    def __call__(self, *a, **k):
        return None


_pp = types.ModuleType("scipy.sparse.linalg._propack")
_pp.__file__ = "<stub>"
_pp.__spec__ = None
_pp.__getattr__ = lambda n: _D()  # noqa: E731
sys.modules.setdefault("scipy.sparse.linalg._propack", _pp)

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _rank4_repeat_kv  # rank-4 GQA repeat_kv respelling (GPU delegation)

MODEL = sys.argv[1] if len(sys.argv) > 1 else "LiquidAI/LFM2.5-Encoder-230M"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "out_lfm25_encoder_230m"
SEQ_LENS = (512, 256, 128, 64)
MLM_LENS = (128,)
os.makedirs(OUT_DIR, exist_ok=True)
FP32 = os.path.join(OUT_DIR, "encoder_fp32.tflite")
FP16 = os.path.join(OUT_DIR, "encoder_fp16.tflite")
WI8 = os.path.join(OUT_DIR, "encoder_wi8fc.tflite")


def load_model():
    from transformers import AutoModelForMaskedLM

    _rank4_repeat_kv.install()
    mlm = AutoModelForMaskedLM.from_pretrained(
        MODEL, trust_remote_code=True, dtype=torch.float32,
        attn_implementation="sdpa",
    ).eval()

    inv = mlm.lfm2.rotary_emb.inv_freq
    assert float(inv.min()) > 0, "rope inv_freq zeroed by meta-load — bad export"

    remote_mod = sys.modules[type(mlm).__module__]

    def _apply_mask_always(hidden_states, attention_mask):
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask[:, :, None].to(
                hidden_states.dtype
            )
        return hidden_states

    remote_mod.apply_mask_to_padding_states = _apply_mask_always
    _rank4_repeat_kv.rebind_all()
    return mlm


class Encoder(nn.Module):
    def __init__(self, body):
        super().__init__()
        self.body = body

    def forward(self, input_ids, attention_mask):
        m = attention_mask.to(torch.float32)
        h = self.body(
            input_ids=input_ids, attention_mask=m, use_cache=False
        ).last_hidden_state
        return h * m[:, :, None]  # deterministic zeros at padded positions


class MaskedLM(nn.Module):
    def __init__(self, mlm):
        super().__init__()
        self.mlm = mlm

    def forward(self, input_ids, attention_mask):
        return self.mlm(
            input_ids=input_ids, attention_mask=attention_mask.to(torch.float32)
        ).logits


def sample(S):
    ids = torch.randint(10, 60000, (1, S), dtype=torch.int32)
    mask = torch.ones(1, S, dtype=torch.int32)
    n_pad = S // 4
    ids[0, S - n_pad:] = 0
    mask[0, S - n_pad:] = 0
    return ids, mask


def torch_ref(enc, ids, mask):
    with torch.no_grad():
        return enc(ids, mask).numpy()


def tflite_sig(path, name):
    from ai_edge_litert.interpreter import Interpreter

    it = Interpreter(model_path=path, num_threads=os.cpu_count())
    return it.get_signature_runner(name)


def main():
    torch.manual_seed(0)
    print(f"loading {MODEL} ...")
    mlm = load_model()
    enc = Encoder(mlm.lfm2).eval()
    lm = MaskedLM(mlm).eval()
    print(f"params {sum(p.numel() for p in mlm.parameters())/1e6:.1f}M")

    # masking-completeness gate: changing pad-region ids must not change valid
    # positions at all (embeddings are zeroed before conv; attention scores get
    # -1e9 -> exp underflows to exact 0).
    S, n_valid = 64, 48
    ids, mask = sample(S)
    ids2 = ids.clone()
    ids2[0, n_valid:] = torch.randint(10, 60000, (S - n_valid,), dtype=torch.int32)
    ref_pad = torch_ref(enc, ids, mask)[0, :n_valid]
    ref_pad2 = torch_ref(enc, ids2, mask)[0, :n_valid]
    leak = float(np.abs(ref_pad - ref_pad2).max())
    ref_exact = torch_ref(enc, ids[:, :n_valid], mask[:, :n_valid])[0]
    d = float(np.abs(ref_pad - ref_exact).max())
    print(f"EAGER pad-content invariance: max|diff| {leak:.3e} (must be 0)")
    print(f"EAGER padded-vs-unpadded: max|diff| {d:.3e} (float noise)")
    assert leak == 0.0, "pad content leaks into valid positions — mask patch not applied?"

    print("converting (litert_torch multi-signature) ...")
    import litert_torch

    conv = None
    for S in SEQ_LENS:
        ids, mask = sample(S)
        kw = {"input_ids": ids, "attention_mask": mask}
        sig = (f"encode_{S}", enc)
        conv = (
            litert_torch.signature(*sig, sample_kwargs=kw)
            if conv is None
            else conv.signature(*sig, sample_kwargs=kw)
        )
    for S in MLM_LENS:
        ids, mask = sample(S)
        conv = conv.signature(
            f"mlm_{S}", lm, sample_kwargs={"input_ids": ids, "attention_mask": mask}
        )
    conv.convert().export(FP32)
    print(f"fp32: {os.path.getsize(FP32)/1e6:.1f} MB")

    ids, mask = sample(128)
    ref = torch_ref(enc, ids, mask)
    got = list(tflite_sig(FP32, "encode_128")(
        input_ids=ids.numpy(), attention_mask=mask.numpy()).values())[0]
    print(f"SMOKE encode_128 fp32 vs torch: max|diff| {np.abs(ref-got).max():.3e} "
          f"corr {np.corrcoef(ref.ravel(), got.ravel())[0,1]:.8f}")

    # ---- wi8fc (FC int8 DRQ + embedding int8 cw, convs float) + fp16 --------
    from ai_edge_quantizer import quantizer, recipe_manager, qtyping
    from ai_edge_quantizer.algorithm_manager import AlgorithmName

    print("quantizing wi8fc ...")
    rm = recipe_manager.RecipeManager()
    rm.add_dynamic_config(regex=".*", operation_name=qtyping.TFLOperationName.FULLY_CONNECTED, num_bits=8)
    rm.add_dynamic_config(
        regex=".*", operation_name=qtyping.TFLOperationName.EMBEDDING_LOOKUP,
        num_bits=8, granularity=qtyping.QuantGranularity.CHANNELWISE,
    )
    qt = quantizer.Quantizer(FP32, rm.get_quantization_recipe())
    assert not qt.need_calibration
    qt.quantize().export_model(WI8)
    print(f"wi8fc: {os.path.getsize(WI8)/1e6:.1f} MB")

    print("quantizing fp16 ...")
    rm = recipe_manager.RecipeManager()
    rm.add_quantization_config(
        regex=".*", operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=qtyping.TensorQuantizationConfig(
                num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
            compute_precision=qtyping.ComputePrecision.FLOAT,
        ),
        algorithm_key=AlgorithmName.FLOAT_CASTING,
    )
    qt = quantizer.Quantizer(FP32, rm.get_quantization_recipe())
    qt.quantize().export_model(FP16)
    print(f"fp16: {os.path.getsize(FP16)/1e6:.1f} MB")

    for kind, path in (("wi8fc", WI8), ("fp16", FP16)):
        got = list(tflite_sig(path, "encode_128")(
            input_ids=ids.numpy(), attention_mask=mask.numpy()).values())[0]
        print(f"SMOKE encode_128 {kind} vs torch: corr "
              f"{np.corrcoef(ref.ravel(), got.ravel())[0,1]:.6f}")
    print("DONE:", FP32, WI8, FP16)
    print("full parity report: python verify_lfm25_encoder.py", MODEL, OUT_DIR)


if __name__ == "__main__":
    main()
