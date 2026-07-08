"""Split the codec decoder for mixed-precision CPU inference.

The codec is the dominant device stage and FLOP-bound (upsampling convs). Global
XNNPACK FORCE_FP16 is 2.2× but breaks quality (SnakeBeta / transformer large-
magnitude activations → ASR unintelligible). This splits the graph at the clean
`pre_transformer` boundary so the fp16-sensitive front can stay fp32 while the
conv-heavy back runs in fp16:

  Part A (fp32): codes → quantizer.decode → pre_conv → pre_transformer → hidden
  Part B (fp16): hidden → upsample → decoder → wav

Exports both graphs (T=64), then benchmarks A-fp32 + B-fp16 vs full fp32 for
time and waveform quality (corr + ASR).

  .venv/bin/python qwen3tts_work/export_codec_split.py
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "qwen3tts_work")
from qtok12.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model
from qtok12.configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Config

SRC = "/Users/majimadaisuke/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-Base/snapshots/5d83992436eae1d760afd27aff78a71d676296fc/speech_tokenizer"
OUT = "out/qwen3tts-codec"
T = 64
os.makedirs(OUT, exist_ok=True)

cfg = Qwen3TTSTokenizerV2Config.from_pretrained(SRC)
m = Qwen3TTSTokenizerV2Model.from_pretrained(SRC, config=cfg, torch_dtype=torch.float32)
m.eval()
dec = m.decoder
dec.pre_transformer.config.use_cache = False
rot = dec.pre_transformer.rotary_emb
dim = dec.pre_transformer.config.head_dim
theta = dec.pre_transformer.config.rope_theta
with torch.no_grad():
    rot.inv_freq.copy_(1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)))
rot.attention_scaling = 1.0


class PartA(torch.nn.Module):
    """codes -> hidden (RVQ + pre_conv + pre_transformer)."""

    def __init__(self, dec):
        super().__init__()
        self.dec = dec

    def forward(self, codes):
        h = self.dec.quantizer.decode(codes)
        h = self.dec.pre_conv(h).transpose(1, 2)
        h = self.dec.pre_transformer(inputs_embeds=h).last_hidden_state
        return h.permute(0, 2, 1)  # [B, C, T]


class PartB(torch.nn.Module):
    """hidden -> wav (upsample + SEANet decoder)."""

    def __init__(self, dec):
        super().__init__()
        self.dec = dec

    def forward(self, hidden):
        for blocks in self.dec.upsample:
            for block in blocks:
                hidden = block(hidden)
        wav = hidden
        for block in self.dec.decoder:
            wav = block(wav)
        return wav.clamp(min=-1, max=1)


partA = PartA(dec).eval()
partB = PartB(dec).eval()

# reference codes + fp32 wav
d = np.load("qwen3tts_work/ref/codec_equiv_ref.npz")
codes_np = d["codes"].astype(np.int32)
wav_ref = d["wav"][0, 0]
Tr = codes_np.shape[-1]
codes_pad = np.zeros((1, 16, T), np.int32)
codes_pad[0, :, :Tr] = codes_np[0]

with torch.no_grad():
    hidden = partA(torch.tensor(codes_pad))
    wav_torch = partB(hidden).numpy()[0, 0]
print(f"torch split-vs-ref: corr {np.corrcoef(wav_torch[:Tr*1920], wav_ref[:Tr*1920])[0,1]:.6f}")
print(f"hidden shape {tuple(hidden.shape)}")

import litert_torch  # noqa: E402

edA = litert_torch.convert(partA, (torch.zeros(1, 16, T, dtype=torch.int32),))
pathA = f"{OUT}/codec_partA_T{T}.tflite"
edA.export(pathA)
edB = litert_torch.convert(partB, (torch.zeros_like(hidden),))
pathB = f"{OUT}/codec_partB_T{T}.tflite"
edB.export(pathB)
print(f"A -> {pathA} ({os.path.getsize(pathA)/1e6:.0f} MB)")
print(f"B -> {pathB} ({os.path.getsize(pathB)/1e6:.0f} MB)")

# ---- bench: A-fp32 + B-fp16 vs full fp32 ----
from ai_edge_litert.compiled_model import CompiledModel
from ai_edge_litert.options import CpuOptions, Options


def load(path, flags):
    opt = Options(cpu_options=CpuOptions(num_threads=4, xnnpack_flags=flags))
    return CompiledModel.from_file(path, options=opt)


def run(model, inp):
    ib = model.create_input_buffers(0)
    ob = model.create_output_buffers(0)
    ib[0].write(inp.reshape(-1))
    model.run_by_index(0, ib, ob)
    return ib, ob, model


hshape = tuple(hidden.shape)
hn = int(np.prod(hshape))
mA = load(pathA, None)
for flagsB, label in [(None, "B-fp32"), (4, "B-fp16")]:
    mB = load(pathB, flagsB)
    ibA = mA.create_input_buffers(0); obA = mA.create_output_buffers(0)
    ibA[0].write(codes_pad.reshape(-1))
    ibB = mB.create_input_buffers(0); obB = mB.create_output_buffers(0)
    # warm + correctness
    mA.run_by_index(0, ibA, obA)
    h = np.array(obA[0].read(hn, np.float32))
    ibB[0].write(h)
    mB.run_by_index(0, ibB, obB)
    wav = np.array(obB[0].read(T * 1920, np.float32))[:Tr * 1920]
    c = float(np.corrcoef(wav, wav_ref)[0, 1])
    # time
    for _ in range(2):
        mA.run_by_index(0, ibA, obA); ibB[0].write(np.array(obA[0].read(hn, np.float32))); mB.run_by_index(0, ibB, obB)
    t0 = time.perf_counter(); n = 6
    for _ in range(n):
        mA.run_by_index(0, ibA, obA)
        ibB[0].write(np.array(obA[0].read(hn, np.float32)))
        mB.run_by_index(0, ibB, obB)
    ms = (time.perf_counter() - t0) / n * 1000
    import soundfile as sf
    sf.write(f"qwen3tts_work/ref/codec_split_{label}.wav", wav, 24000)
    print(f"A-fp32 + {label}: {ms:.0f} ms/chunk  wav corr {c:.6f}  max|d| {np.abs(wav-wav_ref).max():.4f}")
print("CODEC_SPLIT_DONE")
