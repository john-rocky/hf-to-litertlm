---
license: apache-2.0
base_model: canberkkkkkk/ema-lightning
language:
- tr
pipeline_tag: text-to-speech
library_name: litert
tags:
- text-to-speech
- tts
- turkish
- litert
- tflite
- on-device
- flow-matching
- hifigan
---

# EMA Lightning — LiteRT (on-device Turkish TTS, 48 kHz)

[EMA Lightning](https://huggingface.co/canberkkkkkk/ema-lightning) is an 8.6M-parameter Turkish text-to-speech model
(a 5.6M flow-matching DiT distilled to four steps + a 3.0M HiFi-GAN decoder, one voice, 48 kHz). This repository holds
the same weights as three static-shape LiteRT graphs plus the host code between them, so the model runs on a phone
through LiteRT `CompiledModel` (CPU / GPU) without PyTorch. Converted with litert-torch 0.9.4; every graph is checked
against the PyTorch package on six fixed sentences (table below).

**Turkish only, one voice.** The graphs read spoken-form, lowercased Turkish in the model's 49-symbol alphabet
(`a–z ç ö ü ğ ı ş`, space and `! " % & ' ( ) , - . / : ; ?`). Numbers, dates, amounts and abbreviations are turned into
words by [normalizer-tr](https://github.com/erdemtuna/normalizer-tr) (Rust, Apache-2.0; Python wheels on PyPI, a
WebAssembly build exists) — that step is not inside the graphs. `ema_litert.py` runs it on a Mac / Linux / Windows host;
an Android or iOS app either builds the crate natively or takes text that is already written out in words.

## Files

Each graph file holds every bucket as a signature (one copy of the weights); the host picks the signature by the
piece's length. Shapes are fixed per signature.

| File | Size | Signatures | In → Out |
|---|---|---|---|
| `ema_text_multi_fp32.tflite` | 4.9 MB | `ema_text_L{32,64,128,256}` | ids i32 [1,L], mask f32 [1,L] → h [1,L,224], dur [1,L] (letters → text features + frames per letter) |
| `ema_align_multi_fp32.tflite` | 0.7 MB | `ema_align_L{L}_T{T}` for (32,48) (64,96) (128,192) (256,384) (256,512) | h, cg [1,L], fg [1,T], allow [1,1,T,L] → cond [1,T,224] (the windowed Gaussian aligner, once per piece) |
| `ema_step_multi_fp32.tflite` | 15.3 MB | `ema_step_T{48,96,192,384,512}` | x [1,T,64], cond, bc [1,6,224], sg [1,2,224], fbias [1,1,1,T] → v [1,T,64] (one DiT velocity pass; four calls per piece) |
| `ema_decoder_multi_fp16.tflite` | 6.4 MB | `ema_decoder_P{48,120}` | z [1,64,P] + 7 masks [1,1,len_i] → audio f32 [1, P·1920] (HiFi-GAN, 48 kHz; windows of 1 s then 4 s with 8 frames of context) |
| `ema_step_table.npz` | 30 KB | — | the four steps' modulation vectors (t = 0, ¼, ½, ¾) |
| `ema_host.py` | — | — | the host loop, no PyTorch: `pip install ai-edge-litert normalizer-tr numpy` |
| `config.json` | — | — | vocabulary, buckets, step times, decoder stage lengths, file list with sha256 |

Weights: text fp32 (its fp16-weight file runs on the CPU but does not compile on the GPU delegate), aligner fp32, DiT
steps fp32 (fp16 weights on the DiT failed the parity bar on one of six sentences), decoder fp16 (lossless-grade on the
CPU, native on the GPU). 27.3 MB in all. `config.json` lists each file's sha256 and signatures.

```bash
pip install ai-edge-litert normalizer-tr numpy        # torch is optional: with it, seed 0 is the package's own take
python ema_host.py "Merhaba, bu ses telefonunuzun üzerinde üretiliyor." --out merhaba.wav --seed 0
```

## How the host loop works

1. Normalize (normalizer-tr, "fallback" policy) → lowercase the Turkish way (`İ → i`, `I → ı`) → strip accents except
   Turkish letters → replace anything outside the alphabet with a space; cut into pieces of at most ~10 s at sentence,
   clause, then word boundaries (the package's chunker).
2. `text`: letters → `h` and per-letter frame counts. The host groups the counts per word, rounds them to whole frames
   (1–250 per word, ≤ 3000 per piece) and gives every frame a word and a position inside it (`fw`, `fp`), every letter
   a position inside its word (`cg`, `fg` in letters).
3. `align`: one condition vector per frame (each frame reads only its own and the adjacent words — the `allow` mask).
4. `step` × 4 with the seeded Gaussian noise: `x1 = x + (1 − t)·v`, then `x = (1 − t')·noise' + t'·x1` (the package's
   DMD2-distilled sampler). The seed sets the take; seed 0 on CPU reproduces the PyTorch package's output.
5. `decoder` on windows of 25 frames (the first second, so audio starts early) then 100 frames, with 8 frames of
   context on each side cut away after decoding; the pieces join sample-exactly.

```python
from ema_host import EmaLiteRT
tts = EmaLiteRT(threads=4)
audio, stats = tts.say("Merhaba, bu ses telefonunuzun üzerinde üretiliyor.", seed=0)  # float32 at 48 kHz
```
`ema_host.py` is ~250 lines of NumPy; an Android or iOS app ports the same steps (two `CompiledModel`s: the acoustic
graphs on the CPU, the decoder on the GPU) and needs a build of normalizer-tr or pre-normalized text.

## Performance (measured)

Galaxy S26 (Snapdragon, 12 GB), Android 16, LiteRT 2.2.0 `CompiledModel`, 2026-10-11. One call = write the inputs +
run + read the outputs, median of 20 calls after 3 warm-ups on a rested phone (thermal status 0, no CPU cap). Per-graph
times are for the shipped fp32 graphs; "GPU fp16" = the GPU at its default precision (fp16 activations).

| graph (bucket) | CPU, 4 threads | GPU fp32 | GPU fp16 | HTP (NPU, JIT) |
|---|---|---|---|---|
| text (L 64) | 0.7 ms | 1.4 ms | wrong (see parity) | 0.70 ms (49 dB) |
| align (L 64 / T 96) | 0.2 ms | 0.7 ms | 0.6 ms | 0.49 ms (43 dB) |
| step (T 96), one of four | 2.5 ms | 3.2 ms | 2.4 ms (31 dB, on the bar) | **1.56 ms** (41 dB) |
| decoder (P 48 = 1.9 s of audio) | 39.1 ms | 22.2 ms | 14.6 ms | 127.7 ms (31 dB; 518 s to compile) |
| decoder (P 120 = 4.8 s of audio) | 98.8 ms | 54.9 ms | 31.0 ms | did not compile in 600 s |

HTP = LiteRT's NPU accelerator with the Qualcomm compiler plugin and runtime libraries (JIT, one HTP partition per
graph); its step compile takes 1 s per process. The GPU compiles each graph once per process (0.3–0.7 s).

One 3.5 s sentence (60 letters, 88 frames: text + align + 4 steps + decoder windows P48 + P120):

| configuration | total | RTF | first audio |
|---|---|---|---|
| text + align on the CPU, the 4 steps on the HTP, decoder on the GPU at fp16 (the fastest measured) | 52.5 ms | 0.015 | 21.7 ms |
| acoustic graphs on the CPU, decoder on the GPU at fp16 (no vendor NPU libraries needed) | 56.5 ms | 0.016 | 25.5 ms |
| everything on the GPU at fp32 | 92 ms | 0.026 | 37 ms |
| everything on the CPU (4 threads) | 149 ms | 0.042 | 50 ms |

The small acoustic graphs run faster on the CPU than on the GPU (per-launch overhead); the decoder is where the GPU
pays off, and it is the wrong workload for the HTP. The base card's RTF figures are the authors' own (an RTX 4090 and
a cloud CPU) and are not repeated here.
Mac (M4 Max, ai-edge-litert 2.2.0 XNNPACK 4 threads, the same host loop): RTF 0.02–0.04, first audio 40–60 ms.

## Parity with the PyTorch package

Reference = the pip package `ema-lightning` 1.1.2 on a CPU (fp32, seed 0, speed 1), six sentences (the base card's
three "Listen" samples and three of our own: numbers, a date, a long sentence). Bar: the same frame timeline, latents
corr ≥ 0.9999, audio SNR ≥ 30 dB on every sentence.

| where | form | frames | latents | audio (min over the six) |
|---|---|---|---|---|
| Mac CPU (XNNPACK) | fp32 graphs, split | identical | corr 1.000000 | 66 dB |
| Mac CPU | **the shipped set** (text fp32, align fp32, steps fp32, decoder fp16 weights) | identical | corr 1.000000 | 56.4 dB |
| Mac CPU | the shipped set through `ema_host.py` (seed 0, readme1) | identical | — | 56.3 dB vs the package's wav |
| Galaxy S26 CPU | fp32 graphs | — | bit-exact vs the Mac graphs on every graph | — |
| Galaxy S26 GPU fp32 | every graph | — | ≥ 98 dB vs the Mac graphs | — |
| Galaxy S26 GPU fp16 | decoder P48 / P120 | — | — | 46.4 / 40.8 dB (vs the fp32 graph) |
| Galaxy S26 GPU fp16 | align / one step / text / unrolled 4 steps | — | 48 dB / 31 dB / 3 dB (wrong) / 21 dB | — |

Forms that failed the bar and are not shipped: dynamic int8 weights anywhere (the four-step distilled DiT amplifies
weight noise: 6–24 dB), fp16 weights on the DiT (one sentence at 26–28 dB), fp16 GPU activations on the text graph or the
DiT steps. The published Hub clips (`assets/sample-*.wav`) have the same length as the package's CPU output but a
different waveform (a different noise stream); they are not the reference.

## Limitations

- Turkish only, one voice, no voice cloning (the base model's limitations).
- The text frontend (normalizer-tr) is a separate component; the graphs do not read digits or abbreviations.
- Fixed buckets: a piece longer than 256 letters / 512 frames is cut by the chunker first; speeds far from 1.0 may move a
  piece to a larger bucket.
- The shipped checkpoint has no self-attention in its text encoder (`text_attn: 0` in `ema.pt`); the base card's
  architecture text describes two attention blocks.

## License and attribution

Weights and code of EMA Lightning: Apache-2.0 (canberkkkkkk). normalizer-tr: Apache-2.0 (Erdem Tuna). This conversion:
Apache-2.0. The base card asks that listeners be told they hear an AI-generated voice; that applies here too.
