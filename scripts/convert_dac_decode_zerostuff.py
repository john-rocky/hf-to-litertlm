"""DAC 16kHz decoder -> LiteRT, GPU-clean via ZeroStuffConvT1d.

The real DAC decoder's ConvTranspose1d (upsampling_ratios [8,5,4,2], kernel=2*ratio) does NOT
convert: the odd stride-5 transposed conv fails to legalize (`mhlo.convolution` lhs_dilation=5),
and even the even strides emit TRANSPOSE_CONV which Mali ML Drift rejects (C31). Fix = replace
every ConvTranspose1d with the C20 zero-stuff equivalent (1D generalization, kernel=2*stride,
DAC padding): nearest-upsample x stride x constant mask -> conv1d(flipped weight, pad=K-1) -> crop.
Numerically exact; ops = RESIZE_NEAREST/MUL/CONV/SLICE (all GPU-clean), no TRANSPOSE_CONV.

    ~/clipconv/bin/python scripts/convert_dac_decode_zerostuff.py [T_frames]
"""
import _stub  # noqa: F401
import os, sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import litert_torch
from ai_edge_litert.interpreter import Interpreter
from transformers.models.dac.modeling_dac import DacModel

BANNED = {"GATHER", "GATHER_ND", "TOPK_V2", "GELU", "ERF", "WHERE", "SELECT", "SELECT_V2",
          "BROADCAST_TO", "POW", "TRANSPOSE_CONV"}


class ZeroStuffConvT1d(nn.Module):
    """ConvTranspose1d(K, S, P, output_padding=OP) as zero-stuff + conv1d. L = fixed input len."""
    def __init__(self, ct, L):
        super().__init__()
        self.s = ct.stride[0]; self.k = ct.kernel_size[0]
        self.p = ct.padding[0]; self.op = ct.output_padding[0]; self.L = L
        self.register_buffer("w", ct.weight.flip(2).transpose(0, 1).contiguous())  # [Cout,Cin,K]
        self.register_buffer("b", ct.bias.detach().clone() if ct.bias is not None
                             else torch.zeros(ct.out_channels))
        mk = np.zeros((L * self.s,), np.float32); mk[::self.s] = 1.0
        self.register_buffer("mask", torch.from_numpy(mk)[None, None])      # [1,1,L*S]

    def forward(self, x):                                  # x: [B, Cin, L]
        s, k, p, op = self.s, self.k, self.p, self.op
        # nearest-upsample in 2D (singleton height) -> RESIZE_NEAREST (GPU-clean),
        # not the 1D path which lowers to GATHER_ND.
        xn = F.interpolate(x.unsqueeze(2), size=(1, self.L * s), mode="nearest").squeeze(2)
        xn = xn * self.mask                                                  # zero-stuff
        y = F.conv1d(xn, self.w, bias=self.b, padding=k - 1)                 # [B,Cout, L*S+K-1]
        out_len = (self.L - 1) * s + k - 2 * p + op
        return y[:, :, p:p + out_len]


def swap_convtranspose(decoder, lengths):
    for name, mod in list(decoder.named_modules()):
        if isinstance(mod, nn.ConvTranspose1d):
            parent = decoder
            *path, last = name.split(".")
            for q in path:
                parent = getattr(parent, q)
            setattr(parent, last, ZeroStuffConvT1d(mod, lengths[name]))


def fp16_recipe():
    from ai_edge_quantizer import recipe_manager
    from ai_edge_quantizer.recipe import AlgorithmName, qtyping
    rm = recipe_manager.RecipeManager()
    oc = qtyping.OpQuantizationConfig(
        weight_tensor_config=qtyping.TensorQuantizationConfig(
            num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
        compute_precision=qtyping.ComputePrecision.FLOAT)
    rm.add_quantization_config(regex=".*", operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
                               op_config=oc, algorithm_key=AlgorithmName.FLOAT_CASTING)
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
    cfg = model.config; n_q = cfg.n_codebooks

    class Wrap(nn.Module):
        def __init__(self, m):
            super().__init__(); self.quantizer = m.quantizer; self.decoder = m.decoder

        def forward(self, codes):
            qr = self.quantizer.from_codes(codes)[0]
            return self.decoder(qr)

    wrap = Wrap(model).eval()
    codes = torch.randint(0, cfg.codebook_size, (1, n_q, T), dtype=torch.long)
    with torch.no_grad():
        ref = wrap(codes).numpy()
    print(f"config ratios={cfg.upsampling_ratios}; out audio {ref.shape}")

    # dry-run hooks: capture each ConvTranspose1d input length
    lengths, hooks = {}, []
    for nm, mod in wrap.decoder.named_modules():
        if isinstance(mod, nn.ConvTranspose1d):
            def mk(n):
                def h(m, inp, out): lengths[n] = inp[0].shape[-1]
                return h
            hooks.append(mod.register_forward_hook(mk(nm)))
    with torch.no_grad():
        wrap(codes)
    for h in hooks:
        h.remove()
    print(f"ConvTranspose1d layers (name: in_len, K, S, P): "
          f"{[(n, lengths[n]) for n in lengths]}")

    swap_convtranspose(wrap.decoder, lengths)
    with torch.no_grad():
        ref2 = wrap(codes).numpy()
    corr_eager = float(np.corrcoef(ref.flatten(), ref2.flatten())[0, 1])
    print(f">>> EAGER parity (zero-stuff vs original): corr={corr_eager:.7f} "
          f"max|diff|={np.abs(ref - ref2).max():.2e}")
    if corr_eager < 0.999:
        print("!! eager parity failed — aborting before convert"); return

    fp32 = f"{out_dir}/dac_16khz_decode_zs.tflite"
    litert_torch.convert(wrap, (codes,)).export(fp32)
    print(f"FP32 exported: {round(os.path.getsize(fp32)/1e6, 2)} MB")
    itp, ops, banned, flex, maxnd = op_report(fp32)
    print(f"  ops={ops}")
    print(f"  >>> BANNED={banned or 'NONE'} FLEX/CUSTOM={flex or 'NONE'} max_ndim={maxnd}")
    ins, outs = itp.get_input_details(), itp.get_output_details()
    itp.set_tensor(ins[0]["index"], codes.numpy().astype(ins[0]["dtype"])); itp.invoke()
    got = itp.get_tensor(outs[0]["index"])
    print(f"  >>> parity tflite vs eager: corr={float(np.corrcoef(ref2.flatten(), got.flatten())[0,1]):.6f}")

    from ai_edge_quantizer import quantizer
    fp16 = f"{out_dir}/dac_16khz_decode_zs_fp16.tflite"
    if os.path.exists(fp16):
        os.remove(fp16)
    qt = quantizer.Quantizer(float_model=fp32); qt.load_quantization_recipe(fp16_recipe())
    qt.quantize().export_model(fp16)
    print(f"FP16 exported: {round(os.path.getsize(fp16)/1e6, 2)} MB")
    _, _, b16, f16, nd16 = op_report(fp16)
    print(f"  >>> FP16 BANNED={b16 or 'NONE'} FLEX/CUSTOM={f16 or 'NONE'} max_ndim={nd16}")


if __name__ == "__main__":
    main()
