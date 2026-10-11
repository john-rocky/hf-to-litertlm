# EMA Lightning (Turkish TTS, 8.6M) → LiteRT

Conversion of [canberkkkkkk/ema-lightning](https://huggingface.co/canberkkkkkk/ema-lightning) (Apache-2.0: a 5.6M
flow-matching DiT distilled to four steps + a 3.0M HiFi-GAN decoder, 48 kHz, one Turkish voice) into four static-shape
LiteRT graphs with the integer bookkeeping on the host. Published at
[litert-community/EMA-Lightning-LiteRT](https://huggingface.co/litert-community/EMA-Lightning-LiteRT) with a torch-free
host loop (`ship_docs/ema_host.py`) and a card of measured numbers (`ship_docs/README.md`).

## The cut

| graph | in → out | runs on |
|---|---|---|
| `text` | letter ids + mask → text features h [L,224], frames per letter | CPU (0.7 ms) |
| `align` | h, letter / frame positions, the aligner window → one condition per frame | CPU (0.2 ms) |
| `step` | latents x, condition, the step's modulation vectors → velocity (×4, Euler on the host) | HTP 1.56 ms / CPU 2.5 / GPU 3.2 |
| `decoder` | latents [64,P] + seven length masks → 48 kHz audio | GPU fp16 (P120 = 4.8 s: 31 ms) |

Every boolean or integer step of the package (word indices, the frame timeline, the aligner's window, the decoder's
per-layer length masks, the Euler mixing) runs on the host and enters the graphs as float inputs, so the graphs are
float-only and compile on CPU, GPU and the HTP. Buckets: letters 32 / 64 / 128 / 256, frames 48 / 96 / 192 / 384 / 512,
decoder windows 48 / 120 frames (the package's own window plan: 1 s first, then 4 s, 8 frames of context). Each kind
ships as one multi-signature `.tflite` (every bucket a signature, weights shared): 27.3 MB for the whole family.

## What had to change for the converter and the GPU

- `ConvTranspose1d(k 10, s 5, p 2)` (decoder stage 3) legalizes to neither SAME nor VALID (5n + 1 outputs) in
  litert-torch 0.9.4 → export the VALID transposed convolution and crop p on each side (`DecoderGraph.up`).
- `frame_q.expand(B, T, D)` in the aligner exports as BROADCAST_TO, which the GPU delegate refuses ("Failed to compile
  model") → the query is one [1,1,D] constant and the RoPE multiply broadcasts it (`aligner_cond`).
- The ConvNeXt layer scale `gamma * pw2(...)` exports as MUL(runtime rank-2, constant rank-3 [1,1,C]) — silently wrong
  on Adreno / Metal (LiteRT #10445 / #10231: constants bypass the delegate's 4D broadcast reshape; the text graph came
  back with corr 0.37 on the S26 GPU) → fold gamma into pw2's weight rows and bias (`text_stage`).
- An unrolled four-step graph makes the converter keep one transposed copy of every FC weight per step (29 MB for a
  17 MB DiT) → one step per graph, the Euler loop on the host, the per-step modulation vectors in a 30 KB table.
- `expm1` → `exp − 1`; `masked_fill` → `x·m + (−1e4)·(1 − m)`.

## What the numbers said

- Parity bar: frames identical to the pip package (CPU, seed 0), latents corr ≥ 0.9999, audio SNR ≥ 30 dB on six
  sentences. fp32 graphs: 56–83 dB. Dynamic int8 anywhere: 6–24 dB (FAIL — the four-step distilled DiT amplifies
  weight noise). fp16 weights: decoder lossless-grade (56 dB), text PASS, DiT FAIL on one sentence (26–28 dB).
- GPU fp16 activations (S26): decoder PASS (41–46 dB), aligner PASS (48 dB), one step 31 dB (edge), four steps or the
  text graph FAIL → the acoustic graphs run at fp32 (CPU) and the decoder on the GPU at fp16.
- HTP (LiteRT 2.2.0 + the Qualcomm plugin, JIT): the step compiles in 1 s and runs 1.56 ms (41 dB) — the fastest step;
  the decoder compiles for 518 s (P48, 128 ms per call) or not at all within 600 s (P120).
- The small graphs run faster on the CPU than on the GPU (per-launch overhead); the decoder is the GPU's win.

## Reproduce

```bash
python3.12 -m venv .venv && .venv/bin/pip install ema-lightning==1.1.2 litert-torch==0.9.4 ai-edge-quantizer soundfile scipy
mkdir -p precheck/weights && cd precheck/weights && \
  curl -sLO https://huggingface.co/canberkkkkkk/ema-lightning/resolve/main/ema.pt && \
  curl -sLO https://huggingface.co/canberkkkkkk/ema-lightning/resolve/main/decoder.pt && cd ../..
python oracle_ref.py 4                                   # the package on CPU, six sentences -> fixtures/*.npz + .wav
python export_graphs.py --out out/fp32 --text 32,64,128,256 --align 32x48,64x96,128x192,256x384,256x512 \
       --step 48,96,192,384,512 --decoder 48,120 --multisig
python verify_tflite.py --split --multi --forms text=fp32,sound=fp32,decoder=fp32   # fp32: min SNR 66 dB
python quantize.py --mode fp16 --only decoder --match multi                         # the decoder's fp16 weights
python verify_tflite.py --split --multi --forms text=fp32,sound=fp32,decoder=fp16   # the ship set: min SNR 56 dB
python assemble_ship.py --out out/ship/EMA-Lightning-LiteRT --forms text=fp32,sound=fp32,decoder=fp16 --split --multi
python out/ship/EMA-Lightning-LiteRT/ema_host.py "Merhaba, bu ses telefonunuzun üzerinde üretiliyor." --out merhaba.wav
```

`hostloop_e2e.py --fixtures` times the host loop on the Mac; `bench_mac.py` times each graph. The sha256 of the
checkpoints read: `ema.pt` 95aec03d…, `decoder.pt` 9595819b…. The Hub's own `assets/sample-*.wav` have the package's
frame timeline but a different noise stream; the reference is the package on CPU.
