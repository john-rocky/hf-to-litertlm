---
license: apache-2.0
base_model: Edge0/Audio8-TTS-Preview-0.6b
pipeline_tag: text-to-speech
language:
- en
- zh
- yue
- ja
- ko
- de
- fr
- es
- it
- nl
- pl
tags:
- litert
- tflite
- text-to-speech
- voice-cloning
- on-device
base_model_relation: quantized
---

# Audio8-TTS-Preview-0.6b — LiteRT

[Edge0/Audio8-TTS-Preview-0.6b](https://huggingface.co/Edge0/Audio8-TTS-Preview-0.6b) (Apache-2.0) converted to LiteRT (`.tflite`) for on-device text-to-speech with zero-shot voice cloning in 11 languages, at the model's native 44.1 kHz.

Audio8 TTS is a DualAR speech LM in the Fish Audio S2 Pro lineage: a 24-layer "slow" transformer predicts one semantic token per 46 ms frame, a 4-layer "fast" transformer predicts the frame's ten codec codebooks one at a time, and a neural codec (RVQ, an 8-layer windowed transformer and a causal upsampling ConvNet) turns the frames into audio. This repository holds the four graphs plus a Python host loop that reproduces the vendor's generation loop and sampler.

## Quick start (Python, desktop)

```bash
pip install ai-edge-litert tokenizers soundfile scipy numpy huggingface_hub
hf download litert-community/Audio8-TTS-Preview-0.6b --local-dir audio8
cd audio8
python audio8_tts_litert.py --model-dir . --text "Hello from LiteRT, fully on device." \
    --voice voices/en_librispeech_1272 --out hello.wav
# Japanese with the bundled Japanese voice
python audio8_tts_litert.py --model-dir . --text "今日は天気が良いので、公園まで散歩に行きましょう。" \
    --voice voices/ja_funasr_example --out konnichiwa.wav
# clone your own voice from a 0.5-10 s clip and its exact transcript
python audio8_tts_litert.py --model-dir . --register-voice me.wav --ref-text "exact transcript of me.wav" --voice-out voices/me
python audio8_tts_litert.py --model-dir . --text "..." --voice voices/me --out out.wav
# generation without a reference voice
python audio8_tts_litert.py --model-dir . --text "This utterance does not use a reference voice." --out noref.wav
```

`--slow slow_ar_int4.tflite` selects the smaller slow AR. `--threads` sets the CPU thread count (default 4).

## Android

Sample app (Android): [`android/`](https://huggingface.co/litert-community/Audio8-TTS-Preview-0.6b/tree/main/android) — type a sentence, pick one of the two sample voices or record your own for 10 s, tap Speak. The two autoregressive graphs run on the CPU (4 threads); the codec decoder runs on the GPU when its output passes a check against the same file on the CPU, and otherwise that file runs on the CPU. Galaxy S26, LiteRT 2.2.0, 2026-11-02, the codec decoder on the GPU: an 18-token English sentence became 5.99 s of speech at a real-time factor (RTF: generation time divided by the length of the speech) of 0.857 (median of 6 runs in two app processes, 0.828 to 1.229) and a 14-token Japanese sentence 3.44 s at 0.900 (0.880 to 0.904). Loading took 10.9 to 12.4 s, of which the decoder check took 7.8 to 8.2 s; in that check one 128-frame call of the decoder on the CPU took 2.84 to 2.85 s (four app processes), against 0.91 to 0.96 s for a sentence's 128-frame call on the GPU. When the check fails, the decoder runs on the CPU: on a Pixel 11 Pro with the first-published decoders put back on purpose, at thermal status 1 with its CPU frequency capped, right after other runs, one launch with a remembered failure loaded in 18.7 s, 13.5 s of it the CPU decoder's first call at load (a warm-up on all-zero codes).

## Files

| File | Size | Role |
|---|---|---|
| `slow_ar_int8.tflite` | 552 MB | Slow AR (24 layers). Signatures `prefill_256` and `decode`; KV cache 2048 (the model's max_seq_len) as graph I/O; outputs the 4097 logits the sampler can pick (4096 semantic tokens + end-of-speech) and the normalized hidden state that conditions the fast AR. Dynamic int8 (per-channel) projections, int8 embedding tables. |
| `slow_ar_int4.tflite` | 386 MB | Same graph, blockwise-32 OCTAV int4 projections. |
| `fast_ar_int8.tflite` | 68 MB | Fast AR step (4 layers, 10-slot KV cache as I/O). Called 10 times per frame: position 0 takes the slow hidden state, positions 1-9 take the previous codebook token. Dynamic int8. |
| `codec_decoder_fp16_T128.tflite` | 262 MB | Codec decoder for up to 128 frames (5.9 s) per call; fp16 weights. Runs on the mobile GPU since 2026-11-02, when it was replaced by a rewritten decoder (see [Codec decoder on the GPU](#codec-decoder-on-the-gpu)). |
| `codec_decoder_fp16_T192.tflite` | 262 MB | Same decoder for up to 192 frames (8.9 s) per call; the host loop uses it for longer text as 128-frame-context windows. Runs on the mobile GPU since 2026-11-02, rewritten like the 128-frame decoder. |
| `codec_decoder_int8_T128.tflite` | 132 MB | Codec decoder, all convolutions and projections int8 (export-time PT2E, fp32 codebooks); the CPU option. |
| `codec_encoder_fp16_10s.tflite` | 419 MB | Codec encoder for voice registration: 10.03 s of 44.1 kHz audio -> 10 x 216 codes. |
| `tokenizer.json` | 12 MB | The vendor's tokenizer (Qwen2 BPE + the `<\|semantic:N\|>` and role tokens), unchanged. |
| `voices/*/codes.npy`, `meta.json` | 7 KB | Two bundled reference voices (codec codes + transcript): an English LibriSpeech dev-clean speaker (CC BY 4.0) and the Japanese example clip from FunAudioLLM/Fun-ASR-Nano-2512 (Apache-2.0). |
| `audio8_tts_litert.py` | 13 KB | Host loop: prompt construction, chunked prefill, the vendor's sampler (top-k / top-p / temperature, repetition-aware re-draw), fast-AR loop, windowed codec decode, voice registration. |

The sampler and the prompt format are the vendor's, so the loop accepts the same generation parameters (temperature 0.7, top-p 0.9, top-k 50, max 512 frames by default).

## Codec decoder on the GPU

On 2026-11-02 the two fp16 codec decoders were replaced, under the same names, by rewritten decoders; the bytes first published are in [commit 56fcb491](https://huggingface.co/litert-community/Audio8-TTS-Preview-0.6b/tree/56fcb491ebf03eb3a1a66615fedf6744015dd4c7), and `android/` changed with them. As first published, the two decoders give a wrong output on the LiteRT GPU accelerator: on a Pixel 11 Pro and a Galaxy S26 (OpenCL, LiteRT 2.2.0 and 2.3.2) and on an Apple M4 Max (Metal), at the GPU's default precision (float16) and at float32. If you downloaded the repository before 2026-11-02, download the two decoders and `android/` again: the app remembers a failed check together with the decoder file's size, so at its next launch it checks the new files and runs them on the GPU when they pass (seen on a Pixel 11 Pro where the first-published files had failed). The rewritten decoders carry three graph changes:

- Before two MUL operators whose non-constant operand (computed at run time) has rank 2 and the output rank 3, a RESHAPE gives the operand the output's shape. The GPU accelerator lays such a rank-2 tensor out differently from a rank-3 one, and the MUL then multiplied the wrong elements, at float32 too. The RESHAPE keeps the values and their order.
- Each RMSNorm squares its input scaled by 1/16, with epsilon scaled by 1/256 and the result scaled back. Unscaled, one square reaches 121,972.6 and the 1,024-channel sums of squares about 76,500, both more than float16 holds (65,504), and the output stops depending on the codes. Powers of two only move the exponent, so the CPU result is the same.
- The attention mask's −1e9 becomes −30000. In float16 −1e9 is −inf: with the other two changes in place, the 192-frame decoder's attention then gave NaN after the softmax on the GPU of an Apple M4 Max (Metal), and the output was constant.

On the Mac CPU the rewritten decoders give the original's output byte for byte on all 32 test code sequences (15 for 128 frames, 17 for 192). On the GPU at the default precision, their output correlates with that CPU output at 0.9996 or more on every speech test sequence of a Pixel 11 Pro and of an Apple M4 Max (Metal), and on the English and the Japanese one of a Galaxy S26. Decoded on the Pixel 11 Pro's GPU, every test sequence transcribes (whisper large-v3-turbo) with the same number of errors as the Mac CPU's output, and UTMOS22 is 3.956 against 3.960 (128 frames, 14 sequences) and 3.905 against 3.913 (192 frames, 16 sequences). The rewrite is [`codec_gpu_rewrite.py`](https://github.com/john-rocky/hf-to-litertlm/blob/main/audio8_tts_work/codec_gpu_rewrite.py) in hf-to-litertlm.

## Accuracy

Measured against the vendor's PyTorch implementation (transformers 4.57, CPU fp32) on 14 seeded sentences (6 English and 6 Japanese with a cloned reference voice, one of each without a reference), with the same random draws:

- The fp32 graphs reproduce the reference code sequence frame for frame on all 14 sentences; the codec decoder is bit-exact at torch level and within 2e-6 as `.tflite`.
- Speech recognition of the output (whisper large-v3-turbo) against the input text, and speaker similarity (TitaNet-L cosine) against the reference clip:

| configuration | en WER | ja CER | speaker cosine en / ja |
|---|---|---|---|
| PyTorch reference | 1.1% | 0.0% | 0.66 / 0.74 |
| `slow_ar_int8` + `fast_ar_int8` + `codec_decoder_fp16` | 1.1% | 0.0% | 0.68 / 0.76 |
| `slow_ar_int4` + `fast_ar_int8` + `codec_decoder_fp16` | 1.1% | 0.0% | 0.63 / 0.77 |
| `codec_decoder_int8` (reference codes decoded) | 1.1% | 0.0% | 0.67 / 0.73 |

The `codec_decoder_fp16` rows were measured with the decoders as first published; on the Mac CPU the rewritten decoders give the same output byte for byte on 32 test code sequences (15 for 128 frames, 17 for 192). The one English error is shared with the reference (the model drops the first word of one sentence). Quantized graphs sample a different but valid trajectory; per-frame agreement with the reference is not a meaningful metric for a sampled model, so the gate is the transcript and the voice.

## Performance

Measured, not estimated. The host loop is Python; the Android sample under `android/` runs the same graphs from Kotlin through the LiteRT CompiledModel API (its numbers are in the Android section above).

### Apple silicon Mac (CPU, 4 threads, ai-edge-litert 2.2.0, `audio8_tts_litert.py`)

Measured with the codec decoders as first published. The rewritten decoders give the same output byte for byte on the CPU of an Apple M4 Max (32 test code sequences); their times were not measured again.

| configuration | slow AR / frame | fast AR / frame (10 calls) | codec (T128 call) | RTF (median of 14) |
|---|---|---|---|---|
| int8 / int8 / fp16 | 9.0 ms | 9.2 ms | 1.43 s | 0.81 (0.73-0.98) |
| int4 / int8 / fp16 | 12.2 ms | 9.2 ms | 1.43 s | 0.92 (0.81-1.07) |
| int8 / int8 / int8 codec | 8.9 ms | 9.4 ms | 0.96 s | 0.70 (0.64-0.82) |

### Galaxy S26 (SM-S942Q, Snapdragon 8 Elite Gen 5, Android 16; LiteRT `benchmark_model`, CPU 4 threads; frequency caps verified absent before each row)

| graph | signature | backend | inference (avg) | init / overall memory |
|---|---|---|---:|---:|
| `slow_ar_int8.tflite` | `decode` | CPU | 12.3 ms | 1077 / 1176 MB |
| `slow_ar_int8.tflite` | `prefill_256` | CPU | 241 ms | 1077 / 1249 MB |
| `slow_ar_int4.tflite` | `decode` | CPU | 10.0 ms | 610 / 709 MB |
| `slow_ar_int4.tflite` | `prefill_256` | CPU | 417 ms | 610 / 783 MB |
| `fast_ar_int8.tflite` | `step` | CPU | 0.97 ms | 127 / 127 MB |
| `codec_encoder_fp16_10s.tflite` | `encode` | CPU | 2365 ms per 10 s | 1206 / 1899 MB |

Per generated frame the two AR graphs cost 12.3 + 9.7 = 22 ms on the CPU, i.e. an autoregressive real-time factor of about 0.5 at the codec's 21.5 frames/s, before the codec. A 256-frame decoder does not prepare on the S26 GPU (`Dilated im2col buffer size overflowed`) and is not shipped; int8 convolutions do not run on that GPU; the AR graphs stay on the CPU (their codebook gather is not delegated). Memory figures are the benchmark tool's footprint; the multi-signature slow graph packs its weights once per signature.

### Galaxy S26, one call of the codec decoder (a C++ runner on the LiteRT CompiledModel API, not published)

The same phone on 2026-11-02, LiteRT 2.2.0 CompiledModel API. A value is the median time per call (writing the input codes, running, reading the samples back) after two warm-up calls, with the number of timed calls and the range in parentheses. CPU: XNNPACK, 4 threads. GPU: the LiteRT GPU accelerator (OpenCL) at its default precision (float16), with the CPU for the operators it does not take. Each row started at thermal status 0 with no CPU frequency cap.

| graph | backend | time per call |
|---|---|---:|
| `codec_decoder_fp16_T128.tflite` | GPU (OpenCL, 972 of 1,105 ops) | 937.6 ms per 5.9 s (8, 915.2 to 953.9) |
| `codec_decoder_fp16_T192.tflite` | GPU (OpenCL, 972 of 1,105 ops) | 1,426.1 ms per 8.9 s (8, 1,406.1 to 1,444.8) |
| `codec_decoder_int8_T128.tflite` (the app does not use this file) | CPU | 2,974.6 ms per 5.9 s (5, 2,958.3 to 3,014.2) |

- On the GPU, the output correlates with the Mac CPU output at 0.9998 or more on the English and the Japanese test codes, and at 0.015 or less with the GPU's own output for all-zero codes. The CPU row gives the Mac CPU output exactly.
- During the CPU row the large cores were capped at 4.07 GHz at the lowest (4.74 GHz uncapped).

## Limitations

- The codec decoder is causal but its transformer stacks eight 128-frame windows, so decoding in windows is not sample-exact against a single offline decode; the host loop makes one T128 call for utterances up to 5.9 s, one T192 call up to 8.9 s, and beyond that T192 windows with 128 frames of left context (the same context the vendor's ONNX runtime uses).
- Voice registration accepts up to 10 s of reference audio (one static bucket); the reference transcript must match the audio.
- Prompt length + generated frames must stay under 2048 positions; the default cap is 512 frames (about 24 s) per call.
- Preview checkpoint: the vendor documents limited dialect coverage and sensitivity to noisy or mis-transcribed references. Generated speech can be misused for impersonation; obtain consent before cloning a voice and disclose synthetic audio.

## License

Apache-2.0, inherited from the base model by Edge0.
