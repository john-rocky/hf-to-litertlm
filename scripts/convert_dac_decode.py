"""Descript Audio Codec (DAC) 16kHz DECODER -> LiteRT, clean path.

Decode path only: RVQ codes [1, n_q, T] -> quantizer.from_codes -> decoder -> audio [1, 1, S].
DAC (Descript lineage) converts clean — nn.Conv1d with STATIC padding (no int64 buffers, no
reflect-extra-pad), unlike encodec/mimi SEANet (C25). Ops: EMBEDDING_LOOKUP (RVQ), Snake1d
(SIN), Conv1d (CONV_2D), ConvTranspose1d (TRANSPOSE_CONV). Decode-side is fixed-length so it
is fully static. The only on-device risk is TRANSPOSE_CONV (C31, Mali ML Drift v4).

    ~/clipconv/bin/python scripts/convert_dac_decode.py [T_frames]
"""
import _stub  # noqa: F401
import os, sys
from collections import Counter

import numpy as np
import torch
import litert_torch
from ai_edge_litert.interpreter import Interpreter
from transformers.models.dac.modeling_dac import DacModel

BANNED = {"GATHER", "GATHER_ND", "TOPK_V2", "GELU", "ERF", "WHERE",
          "SELECT", "SELECT_V2", "BROADCAST_TO", "POW", "NON_MAX_SUPPRESSION_V5"}


def fp16_recipe():
    from ai_edge_quantizer import recipe_manager
    from ai_edge_quantizer.recipe import AlgorithmName, qtyping
    rm = recipe_manager.RecipeManager()
    op_config = qtyping.OpQuantizationConfig(
        weight_tensor_config=qtyping.TensorQuantizationConfig(
            num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
        compute_precision=qtyping.ComputePrecision.FLOAT,
    )
    rm.add_quantization_config(regex=".*", operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
                               op_config=op_config, algorithm_key=AlgorithmName.FLOAT_CASTING)
    return rm.get_quantization_recipe()


def op_report(path):
    itp = Interpreter(model_path=path); itp.allocate_tensors()
    ops = Counter(o["op_name"] for o in itp._get_ops_details())
    maxnd = max(len(t["shape"]) for t in itp.get_tensor_details())
    banned = sorted(k for k in ops if k.upper() in BANNED)
    flex = sorted(k for k in ops if "flex" in k.lower() or k.lower() == "custom")
    return itp, dict(ops), banned, flex, maxnd


def main():
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    out_dir = "out/dac"; os.makedirs(out_dir, exist_ok=True)
    model = DacModel.from_pretrained("descript/dac_16khz").eval()
    cfg = model.config
    n_q = cfg.n_codebooks
    print(f"config: n_codebooks={n_q} codebook_size={cfg.codebook_size} "
          f"upsampling_ratios={cfg.upsampling_ratios} hidden_size={cfg.hidden_size}")

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__(); self.quantizer = m.quantizer; self.decoder = m.decoder

        def forward(self, codes):                     # [1, n_q, T] int
            qr = self.quantizer.from_codes(codes)[0]  # [1, dim, T]
            return self.decoder(qr)                   # [1, 1, S]

    wrap = Wrap(model).eval()
    codes = torch.randint(0, cfg.codebook_size, (1, n_q, T), dtype=torch.long)
    with torch.no_grad():
        ref = wrap(codes).numpy()
    hop = int(np.prod(cfg.upsampling_ratios))
    print(f"T={T} frames -> audio {ref.shape} (hop={hop}, ~{ref.shape[-1]/cfg.sampling_rate:.2f}s)")

    fp32 = f"{out_dir}/dac_16khz_decode.tflite"
    litert_torch.convert(wrap, (codes,)).export(fp32)
    print(f"FP32 exported: {round(os.path.getsize(fp32)/1e6, 2)} MB")
    itp, ops, banned, flex, maxnd = op_report(fp32)
    print(f"  ops={ops}")
    print(f"  >>> BANNED={banned or 'NONE'} FLEX/CUSTOM={flex or 'NONE'} max_ndim={maxnd}")

    # parity
    ins, outs = itp.get_input_details(), itp.get_output_details()
    itp.set_tensor(ins[0]["index"], codes.numpy().astype(ins[0]["dtype"]))
    itp.invoke()
    got = itp.get_tensor(outs[0]["index"])
    corr = float(np.corrcoef(ref.flatten(), got.flatten())[0, 1])
    print(f"  >>> parity FP32 vs eager: corr={corr:.6f} max|diff|={np.abs(ref-got).max():.2e}")

    # FP16
    from ai_edge_quantizer import quantizer
    fp16 = f"{out_dir}/dac_16khz_decode_fp16.tflite"
    if os.path.exists(fp16):
        os.remove(fp16)
    qt = quantizer.Quantizer(float_model=fp32)
    qt.load_quantization_recipe(fp16_recipe())
    qt.quantize().export_model(fp16)
    print(f"FP16 exported: {round(os.path.getsize(fp16)/1e6, 2)} MB")
    itp16, _, b16, f16, nd16 = op_report(fp16)
    print(f"  >>> FP16 BANNED={b16 or 'NONE'} FLEX/CUSTOM={f16 or 'NONE'} max_ndim={nd16}")
    itp16.set_tensor(itp16.get_input_details()[0]["index"], codes.numpy().astype(ins[0]["dtype"]))
    itp16.invoke()
    g16 = itp16.get_tensor(itp16.get_output_details()[0]["index"])
    print(f"  >>> parity FP16 vs eager: corr={float(np.corrcoef(ref.flatten(), g16.flatten())[0,1]):.6f}")


if __name__ == "__main__":
    main()
