"""End-to-end FP16-tflite codec round-trip (what the Android app runs) + export the demo audio.
  test audio[1,16000] -> encoder.tflite(GPU) -> z -> numpy RVQ encode -> codes ->
  numpy RVQ decode -> z_q -> decoder.tflite(GPU) -> audio_out.  Compare vs torch m(audio).

    ~/clipconv/bin/python scripts/dac_prep_demo.py
"""
import _stub  # noqa: F401
import os
import numpy as np
import torch
from ai_edge_litert.interpreter import Interpreter
from transformers.models.dac.modeling_dac import DacModel

OUT = "out/dac"


def l2norm(x, axis):
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-12)


def load_rvq(m):
    qs = m.quantizer.quantizers
    return ([q.in_proj.weight.detach().numpy()[:, :, 0] for q in qs],
            [q.in_proj.bias.detach().numpy() for q in qs],
            [q.codebook.weight.detach().numpy() for q in qs],
            [q.out_proj.weight.detach().numpy()[:, :, 0] for q in qs],
            [q.out_proj.bias.detach().numpy() for q in qs])


def rvq_encode(z, W_in, b_in, CB, W_out, b_out):
    residual = z[0].astype(np.float64); z_q = np.zeros_like(residual)
    codes = np.zeros((len(CB), z.shape[-1]), np.int64)
    for i in range(len(CB)):
        proj = W_in[i] @ residual + b_in[i][:, None]
        code = (l2norm(proj.T, 1) @ l2norm(CB[i], 1).T).argmax(1)
        qrep = W_out[i] @ CB[i][code].T + b_out[i][:, None]
        z_q += qrep; residual = residual - qrep; codes[i] = code
    return codes


def rvq_decode(codes, CB, W_out, b_out):
    z_q = np.zeros((W_out[0].shape[0], codes.shape[1]), np.float64)
    for i in range(len(CB)):
        z_q += W_out[i] @ CB[i][codes[i]].T + b_out[i][:, None]
    return z_q[None].astype(np.float32)


def run(tfl, x):
    itp = Interpreter(model_path=tfl); itp.allocate_tensors()
    di, do = itp.get_input_details()[0], itp.get_output_details()[0]
    itp.set_tensor(di["index"], x.astype(di["dtype"])); itp.invoke()
    return itp.get_tensor(do["index"])


def main():
    m = DacModel.from_pretrained("descript/dac_16khz").eval()
    rvq = load_rvq(m)

    # 1s 16kHz structured test tone: C-major arpeggio + light vibrato (audible, self-contained)
    sr = 16000; t = np.arange(sr) / sr
    audio = np.zeros(sr, np.float32)
    for k, f in enumerate([261.63, 329.63, 392.0, 523.25]):
        seg = (t >= k * 0.25) & (t < (k + 1) * 0.25)
        audio[seg] = 0.3 * np.sin(2 * np.pi * f * t[seg]) * np.hanning(seg.sum())
    x = audio[None, None]

    z = run(f"{OUT}/dac_16khz_encoder_fp16.tflite", x)              # [1,1024,50]
    codes = rvq_encode(z, *rvq)
    zq = rvq_decode(codes, rvq[2], rvq[3], rvq[4])
    out = run(f"{OUT}/dac_16khz_deconly_zs_fp16.tflite", zq)        # [1,1,~16000]
    with torch.no_grad():
        ref = m(torch.from_numpy(x)).audio_values.numpy()
    n = min(out.shape[-1], ref.shape[-1])
    corr = np.corrcoef(out.flatten()[:n], ref.flatten()[:n])[0, 1]
    print(f"codes {codes.shape} (12x50 int, vs raw {sr} float = "
          f"{sr*16/(codes.size*10):.0f}:1 @10bit); fp16-tflite round-trip vs torch corr={corr:.4f}")

    audio.tofile(f"{OUT}/test_audio.bin")
    print(f"saved {OUT}/test_audio.bin ({sr} float32)")
    np.asarray(out.flatten()[:sr], np.float32).tofile(f"{OUT}/test_audio_recon.bin")
    print("saved recon for reference")


if __name__ == "__main__":
    main()
