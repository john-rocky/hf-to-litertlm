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

## Files

| File | Size | Role |
|---|---|---|
| `slow_ar_int8.tflite` | 552 MB | Slow AR (24 layers). Signatures `prefill_256` and `decode`; KV cache 2048 (the model's max_seq_len) as graph I/O; outputs the 4097 logits the sampler can pick (4096 semantic tokens + end-of-speech) and the normalized hidden state that conditions the fast AR. Dynamic int8 (per-channel) projections, int8 embedding tables. |
| `slow_ar_int4.tflite` | 386 MB | Same graph, blockwise-32 OCTAV int4 projections. |
| `fast_ar_int8.tflite` | 68 MB | Fast AR step (4 layers, 10-slot KV cache as I/O). Called 10 times per frame: position 0 takes the slow hidden state, positions 1-9 take the previous codebook token. Dynamic int8. |
| `codec_decoder_fp16_T128.tflite` | 262 MB | Codec decoder for up to 128 frames (5.9 s) per call; fp16 weights. Runs on the mobile GPU. |
| `codec_decoder_fp16_T192.tflite` | 262 MB | Same decoder for up to 192 frames (8.9 s) per call; the host loop uses it for longer text as 128-frame-context windows. Runs on the mobile GPU. |
| `codec_decoder_int8_T128.tflite` | 132 MB | Codec decoder, all convolutions and projections int8 (export-time PT2E, fp32 codebooks); the CPU option. |
| `codec_encoder_fp16_10s.tflite` | 419 MB | Codec encoder for voice registration: 10.03 s of 44.1 kHz audio -> 10 x 216 codes. |
| `tokenizer.json` | 12 MB | The vendor's tokenizer (Qwen2 BPE + the `<\|semantic:N\|>` and role tokens), unchanged. |
| `voices/*/codes.npy`, `meta.json` | 6 KB | Two bundled reference voices (codec codes + transcript): an English LibriSpeech dev-clean speaker (CC BY 4.0) and the Japanese example clip from FunAudioLLM/Fun-ASR-Nano-2512 (Apache-2.0). |
| `audio8_tts_litert.py` | 14 KB | Host loop: prompt construction, chunked prefill, the vendor's sampler (top-k / top-p / temperature, repetition-aware re-draw), fast-AR loop, windowed codec decode, voice registration. |

The sampler and the prompt format are the vendor's, so the loop accepts the same generation parameters (temperature 0.7, top-p 0.9, top-k 50, max 512 frames by default).

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

The one English error is shared with the reference (the model drops the first word of one sentence). Quantized graphs sample a different but valid trajectory; per-frame agreement with the reference is not a meaningful metric for a sampled model, so the gate is the transcript and the voice.

## Performance

Measured, not estimated. The host loop is Python; on a phone the same graphs would run from Kotlin/C++ through the LiteRT Compiled Model API.

### Apple silicon Mac (CPU, 4 threads, ai-edge-litert 2.2.0, `audio8_tts_litert.py`)

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
| `codec_decoder_fp16_T128.tflite` | `decode` | GPU (OpenCL, 936 of 1069 ops) | 923 ms per 5.9 s | 1050 MB |
| `codec_decoder_fp16_T192.tflite` | `decode` | GPU (OpenCL) | 1425 ms per 8.9 s | 1109 MB |
| `codec_decoder_fp16_T128.tflite` | `decode` | CPU | 3664 ms per 5.9 s | 862 / 1433 MB |
| `codec_decoder_int8_T128.tflite` | `decode` | CPU | 1959 ms per 5.9 s | 259 / 836 MB |
| `codec_encoder_fp16_10s.tflite` | `encode` | CPU | 2287 ms per 10 s | 1206 / 1952 MB |

Per generated frame the two AR graphs cost 12.3 + 9.7 = 22 ms on the CPU, i.e. an autoregressive real-time factor of about 0.5 at the codec's 21.5 frames/s, before the codec. A 256-frame decoder does not prepare on the S26 GPU (`Dilated im2col buffer size overflowed`) and is not shipped; int8 convolutions do not run on that GPU; the fp32 decoder delegates 971/971 ops but runs at the same 924 ms with a 1462 MB footprint; the AR graphs stay on the CPU (their codebook gather is not delegated). Memory figures are the benchmark tool's footprint; the multi-signature slow graph packs its weights once per signature.

## Limitations

- The codec decoder is causal but its transformer stacks eight 128-frame windows, so decoding in windows is not sample-exact against a single offline decode; the host loop makes one T128 call for utterances up to 5.9 s, one T192 call up to 8.9 s, and beyond that T192 windows with 128 frames of left context (the same context the vendor's ONNX runtime uses).
- Voice registration accepts up to 10 s of reference audio (one static bucket); the reference transcript must match the audio.
- Prompt length + generated frames must stay under 2048 positions; the default cap is 512 frames (about 24 s) per call.
- Preview checkpoint: the vendor documents limited dialect coverage and sensitivity to noisy or mis-transcribed references. Generated speech can be misused for impersonation; obtain consent before cloning a voice and disclose synthetic audio.

## License

Apache-2.0, inherited from the base model by Edge0.
