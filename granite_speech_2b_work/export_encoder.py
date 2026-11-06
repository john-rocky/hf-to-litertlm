#!/usr/bin/env python3
"""Export the audio path of granite-speech-5.0-2b (port.py) as one single-signature TFLite file, then its fp16-weight
form (the published one).

  signature `encode`: input `audio` f32 [1, 3600, 160] (36 s of 16 kHz PCM, frame = hop = 160 samples)
                      -> `features` f32 [1, 360, 2048] + `mask` uint8 [1, 360]

  OUT/granite_speech_audio_3600f_fp32.tflite   litert-torch export of port.GraniteSpeechAudioGraph (weights BF16 -> fp32)
  OUT/granite_speech_audio_3600f_fp16.tflite   ai-edge-quantizer FLOAT_CASTING of every supported weight to fp16,
                                               explicit DEQUANTIZE, compute in fp32
  OUT/export_encoder.json                      per file: bytes, sha256, operator counts, signature, and with
                                               --check-wav the comparison with the eager port on that clip

  python export_encoder.py --src <snapshot dir> --out <dir> [--check-wav <snapshot>/multilingual_sample.wav]
"""
import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def export_fp32(src, out):
    import torch
    import litert_torch
    import port as P  # imported as the top-level module `port`: the module path is part of every tensor name
    g, n = P.load_graph(str(src), win_frames=C.WIN_FRAMES, feature_scale=1.0)
    t0 = time.time()
    edge = litert_torch.signature("encode", g, sample_kwargs={"audio": torch.zeros(1, C.WIN_FRAMES, C.HOP)}).convert()
    edge.export(str(out))
    return round(time.time() - t0, 1), n, g


def quantize_fp16(src, out):
    from ai_edge_quantizer import algorithm_manager, qtyping, quantizer, recipe_manager
    OP = qtyping.TFLOperationName
    rm = recipe_manager.RecipeManager()
    rm.add_quantization_config(
        regex=".*", operation_name=OP.ALL_SUPPORTED,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=qtyping.TensorQuantizationConfig(num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
            compute_precision=qtyping.ComputePrecision.FLOAT, explicit_dequantize=True),
        algorithm_key=algorithm_manager.AlgorithmName.FLOAT_CASTING)
    qt = quantizer.Quantizer(str(src), rm.get_quantization_recipe())
    assert not qt.need_calibration
    t0 = time.time()
    if out.exists():
        out.unlink()
    qt.quantize().export_model(str(out))
    return round(time.time() - t0, 1)


def interp(path, threads=None):
    from ai_edge_litert.interpreter import Interpreter
    return Interpreter(model_path=str(path), num_threads=threads) if threads else Interpreter(model_path=str(path))


def op_hist(path):
    c = collections.Counter(d["op_name"] for d in interp(path)._get_ops_details())
    return dict(sorted(c.items(), key=lambda kv: -kv[1]))


def sig_io(run):
    f = lambda d: {k: {"dtype": np.dtype(v["dtype"]).name, "shape": [int(s) for s in v["shape"]]} for k, v in d.items()}
    return {"inputs": f(run.get_input_details()), "outputs": f(run.get_output_details())}


WANT_IO = {"inputs": {"audio": {"dtype": "float32", "shape": [1, C.WIN_FRAMES, C.HOP]}},
           "outputs": {"features": {"dtype": "float32", "shape": [1, C.OUT_TOKENS, 2048]},
                       "mask": {"dtype": "uint8", "shape": [1, C.OUT_TOKENS]}}}


def check_clip(path, graph, wav, threads):
    """TFLite vs the eager port on one clip framed as the runtime frames it: valid audio tokens (last non-zero entry of
    `mask` + 1, as LiteRT-LM counts them) against the processor's count, and the features of those tokens."""
    import torch
    x = C.wav_float(wav)
    assert len(x) <= C.WIN_SAMPLES, (wav, len(x))
    buf = np.zeros(C.WIN_SAMPLES, np.float32)
    buf[:len(x)] = x
    win = buf.reshape(1, C.WIN_FRAMES, C.HOP)
    run = interp(path, threads).get_signature_runner("encode")
    t0 = time.perf_counter()
    o = run(audio=win)
    ms = (time.perf_counter() - t0) * 1e3
    with torch.no_grad():
        ref = graph(torch.from_numpy(win))
    nz = np.nonzero(o["mask"][0])[0]
    valid = int(nz[-1]) + 1 if len(nz) else 0
    want = C.num_audio_tokens(len(x))
    a = o["features"][0, :valid].astype(np.float64)
    b = ref["features"][0, :valid].numpy().astype(np.float64)
    cos = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)))
    return {"wav": Path(wav).name, "samples": len(x), "valid_tokens": valid, "processor_tokens": want,
            "mask_equal_eager": bool(np.array_equal(o["mask"][0], ref["mask"][0].numpy())),
            "cos_vs_eager": cos, "max_abs_vs_eager": float(np.abs(a - b).max()), "invoke_ms": round(ms, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="snapshot of ibm-granite/granite-speech-5.0-2b (config.json, model.safetensors)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--check-wav", default="", help="a clip of up to 36 s to compare each file with the eager port on")
    args = ap.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fp32 = out / f"granite_speech_audio_{C.WIN_FRAMES}f_fp32.tflite"
    fp16 = out / f"granite_speech_audio_{C.WIN_FRAMES}f_fp16.tflite"
    rep = {"src": str(args.src)}
    dt, n, _ = export_fp32(args.src, fp32)
    rep["export_s"], rep["graph_tensors"] = dt, n
    print(f"exported {fp32.name} {fp32.stat().st_size:,} B in {dt} s ({n} checkpoint tensors)", flush=True)
    rep["quantize_fp16_s"] = quantize_fp16(fp32, fp16)
    import importlib.metadata as md
    rep["versions"] = {p: md.version(p) for p in ("litert-torch", "ai-edge-quantizer", "ai-edge-litert",
                                                  "litert-converter", "torch", "torchaudio")}
    graph = None
    if args.check_wav:  # a fresh eager module (not the one the export traced)
        import port as P
        graph, _ = P.load_graph(str(args.src), win_frames=C.WIN_FRAMES, feature_scale=1.0)
    rep["files"] = {}
    for p in (fp32, fp16):
        ops = op_hist(p)
        f = {"bytes": p.stat().st_size, "sha256": C.sha256(p), "n_ops": sum(ops.values()), "ops": ops}
        io = sig_io(interp(p).get_signature_runner("encode"))
        assert io == WANT_IO, io
        f["signature_encode"] = io
        assert not [k for k in ops if "FLEX" in k.upper() or "CUSTOM" in k.upper()], ops
        if args.check_wav:
            f["check"] = check_clip(p, graph, args.check_wav, args.threads)
        rep["files"][p.name] = f
        print(f"{p.name} {f['bytes']:,} B sha256 {f['sha256']} ops {f['n_ops']}"
              + (f" | {json.dumps(f['check'])}" if args.check_wav else ""), flush=True)
    (out / "export_encoder.json").write_text(json.dumps(rep, indent=1))
    print("EXPORT_ENCODER_DONE", flush=True)


if __name__ == "__main__":
    main()
