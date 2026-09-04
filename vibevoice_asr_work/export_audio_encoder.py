#!/usr/bin/env python3
"""Export the VibeVoice-ASR-BitNet audio front-end (acoustic σ-VAE encoder + semantic encoder
+ multi-modal projector, with the vendor's -25 dBFS RMS normalisation folded in) as ONE
single-signature .tflite that fits LiteRT-LM's generic audio contract:

  input  `audio`     f32 [1, T, 3200]   raw 24 kHz PCM framed by the runtime (skip-mel path,
                                        frame == hop == 3200 samples == one latent frame)
  output `features`  f32 [1, T, 1536]   projected audio embeddings (audio_shrink_factor 1)

The runtime zero-pads the window; frames that are exactly all-zero are treated as padding
for the RMS statistic (real speech never contains an exactly-zero 133 ms frame).

  python export_audio_encoder.py --seconds 30 [--out out/audio_encoder] [--quant fp32,wi8fc]

Writes audio_encoder_{seconds}s_{fp32,wi8fc,fp16}.tflite + a parity/timing json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


class AsrAudioEncoder(nn.Module):
    def __init__(self, model, target_db_fs=-25.0, eps=1e-6):
        super().__init__()
        self.acoustic = model.model.acoustic_tokenizer_encoder
        self.semantic = model.model.semantic_tokenizer_encoder
        self.proj = model.model.multi_modal_projector
        self.gain = float(10 ** (target_db_fs / 20))
        self.eps = eps

    def forward(self, audio):  # [1, T, 3200]
        # Every statistic stays rank-3 ([1,1,1]): the GPU delegate rejects rank-0 tensors
        # ("DIV/MUL ... has bad input dims size: 0" — measured on Mac Metal, 22 CPU fallbacks
        # in an otherwise fully-delegated 1856-op graph).
        valid = (audio.abs().amax(dim=-1, keepdim=True) > 0).to(audio.dtype)  # [1, T, 1]
        n_frames = valid.sum(dim=(1, 2), keepdim=True).clamp(min=1.0)  # [1,1,1]
        # fp16-safe: a sum of squares over 720k raw samples overflows fp16 (65504) on the mobile
        # GPU delegate (S26 OpenCL: all 1857 nodes delegated, infinite-float capping -> rms=65504
        # -> silence -> empty transcript). Mean per frame (<= 1) then mean over <= 225 frames.
        frame_ms = (audio * audio).mean(dim=-1, keepdim=True)  # [1, T, 1]
        ms = (frame_ms * valid).sum(dim=(1, 2), keepdim=True) / n_frames  # [1,1,1]
        rms = torch.sqrt(ms)
        y = audio * (self.gain / (rms + self.eps))
        mx = y.abs().amax(dim=(1, 2), keepdim=True)  # [1,1,1]
        scale = torch.where(mx > 1.0, 1.0 / (mx + self.eps), torch.ones_like(mx))
        y = y * scale
        x = y.reshape(1, 1, -1)
        a = self.acoustic(x).latents
        s = self.semantic(x).latents
        # Dict return -> litert_torch names the signature output "features" (the runtime's
        # AudioStaticEncoder requires that exact name; positional returns become output_0).
        return {"features": self.proj(a, s)}


def op_report(path, tag):
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=path)
    ops = {}
    for d in it._get_ops_details():
        ops[d["op_name"]] = ops.get(d["op_name"], 0) + 1
    print(f"[{tag}] ops:", dict(sorted(ops.items(), key=lambda kv: -kv[1])))
    return ops


def run_tflite(path, audio, threads=8):
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=path, num_threads=threads)
    sig = it.get_signature_runner()
    print("signature io:", list(sig.get_input_details()), list(sig.get_output_details()))
    t0 = time.time()
    out = sig(audio=audio)
    dt = time.time() - t0
    t0 = time.time()
    out = sig(audio=audio)
    dt2 = time.time() - t0
    return out["features"], dt, dt2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(C.WORK, "hf_native"))
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--out", default=os.path.join(C.WORK, "out", "audio_encoder"))
    ap.add_argument("--quant", default="fp32,wi8fc")
    ap.add_argument("--clip", default="fixtures/clip02.wav")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    T = int(round(args.seconds * C.SR / C.HOP))
    tag = f"{int(args.seconds)}s"

    from transformers import VibeVoiceAsrForConditionalGeneration
    model = VibeVoiceAsrForConditionalGeneration.from_pretrained(args.model, dtype=torch.float32).eval()
    model.config.acoustic_tokenizer_encoder_config.vae_std = 0.0
    enc = AsrAudioEncoder(model).eval()
    n_params = sum(p.numel() for p in enc.parameters())
    print(f"encoder params {n_params/1e6:.1f}M, window T={T} frames ({T*C.HOP/C.SR:.1f}s)")

    # Reference clip padded into the window (what the runtime does).
    wav = C.load_wav(os.path.join(C.WORK, args.clip))
    x, n_valid = C.pad_to_hop(wav)
    assert n_valid <= T, (n_valid, T)
    frames = np.zeros((1, T, C.HOP), np.float32)
    frames[0, :n_valid] = x.reshape(n_valid, C.HOP)
    audio = torch.from_numpy(frames)

    # Eager reference (a) module on framed input, (b) HF path on the un-padded normalised clip.
    with torch.no_grad():
        ref_mod = enc(audio)["features"][0, :n_valid].numpy()
        xn = C.normalize_audio(wav)
        xn, _ = C.pad_to_hop(xn)
        ref_hf = model.get_audio_features(input_values=torch.from_numpy(xn)[None, None, :]).pooler_output[0].numpy()
    d = np.abs(ref_mod - ref_hf).max()
    print(f"module vs HF get_audio_features on {n_valid} valid frames: max|diff| {d:.3e} (rms {np.sqrt((ref_hf**2).mean()):.3f})")
    assert d < 2e-2, "in-graph normalisation deviates from the vendor normaliser"

    import litert_torch
    t0 = time.time()
    conv = litert_torch.signature("encode", enc, sample_kwargs={"audio": audio})
    edge = conv.convert()
    fp32 = os.path.join(args.out, f"audio_encoder_{tag}_fp32.tflite")
    edge.export(fp32)
    print(f"fp32 export {os.path.getsize(fp32)/1e6:.1f} MB in {time.time()-t0:.0f}s")
    del edge, conv
    ops = op_report(fp32, "fp32")
    results = {"seconds": args.seconds, "T": T, "clip": args.clip, "n_valid": n_valid,
               "params_m": n_params / 1e6, "ops": ops, "variants": {}}

    def gate(path, name):
        feats, dt_first, dt = run_tflite(path, frames)
        got = feats[0, :n_valid]
        err = np.abs(got - ref_hf).max()
        rel = err / (np.abs(ref_hf).max() + 1e-9)
        cos = float((got * ref_hf).sum() / (np.linalg.norm(got) * np.linalg.norm(ref_hf) + 1e-9))
        pad = np.abs(feats[0, n_valid:]).max() if n_valid < T else 0.0
        r = {"size_mb": round(os.path.getsize(path) / 1e6, 1), "max_abs_err": float(err), "rel_err": float(rel),
             "cos": cos, "first_invoke_s": round(dt_first, 2), "invoke_s": round(dt, 2), "pad_region_max": float(pad)}
        print(f"[{name}] {r}")
        results["variants"][name] = r

    variants = [v.strip() for v in args.quant.split(",") if v.strip()]
    if "fp32" in variants:
        gate(fp32, "fp32")
    if "wi8fc" in variants:
        from ai_edge_quantizer import quantizer, recipe_manager, qtyping
        OP = qtyping.TFLOperationName
        wi8 = os.path.join(args.out, f"audio_encoder_{tag}_wi8fc.tflite")
        if os.path.exists(wi8):
            os.remove(wi8)
        rm = recipe_manager.RecipeManager()
        rm.add_dynamic_config(regex=".*", operation_name=OP.FULLY_CONNECTED, num_bits=8)
        qt = quantizer.Quantizer(fp32, rm.get_quantization_recipe())
        assert not qt.need_calibration
        qt.quantize().export_model(wi8)
        op_report(wi8, "wi8fc")
        gate(wi8, "wi8fc")
    if "wi8all" in variants:
        from ai_edge_quantizer import quantizer, recipe_manager, qtyping
        OP = qtyping.TFLOperationName
        wi8 = os.path.join(args.out, f"audio_encoder_{tag}_wi8all.tflite")
        if os.path.exists(wi8):
            os.remove(wi8)
        rm = recipe_manager.RecipeManager()
        for op in (OP.FULLY_CONNECTED, OP.CONV_2D, OP.DEPTHWISE_CONV_2D):
            rm.add_dynamic_config(regex=".*", operation_name=op, num_bits=8)
        qt = quantizer.Quantizer(fp32, rm.get_quantization_recipe())
        assert not qt.need_calibration
        qt.quantize().export_model(wi8)
        op_report(wi8, "wi8all")
        gate(wi8, "wi8all")
    if "fp16" in variants:
        from ai_edge_quantizer import quantizer, recipe_manager, qtyping
        fp16 = os.path.join(args.out, f"audio_encoder_{tag}_fp16.tflite")
        if os.path.exists(fp16):
            os.remove(fp16)
        rm16 = recipe_manager.RecipeManager()
        rm16.add_quantization_config(
            regex=".*", operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
            op_config=qtyping.OpQuantizationConfig(
                weight_tensor_config=qtyping.TensorQuantizationConfig(num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
                compute_precision=qtyping.ComputePrecision.FLOAT, explicit_dequantize=True))
        quantizer.Quantizer(fp32, rm16.get_quantization_recipe()).quantize().export_model(fp16)
        gate(fp16, "fp16")
    json.dump(results, open(os.path.join(args.out, f"audio_encoder_{tag}_report.json"), "w"), indent=1)
    print("ENCODER_DONE")


if __name__ == "__main__":
    main()
