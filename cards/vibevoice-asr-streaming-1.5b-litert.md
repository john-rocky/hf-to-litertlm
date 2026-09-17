---
license: mit
base_model: microsoft/VibeVoice-ASR-Streaming-1.5B
pipeline_tag: automatic-speech-recognition
library_name: litert-lm
language:
- en
- zh
- es
- pt
- de
- ja
- ko
- fr
- ru
- it
tags:
  - litert
  - litert-lm
  - litertlm
  - on-device
  - edge
  - asr
  - speech-recognition
  - streaming
  - audio
  - vibevoice
---

# VibeVoice-ASR-Streaming-1.5B — LiteRT-LM

[microsoft/VibeVoice-ASR-Streaming-1.5B](https://huggingface.co/microsoft/VibeVoice-ASR-Streaming-1.5B) converted to the **LiteRT-LM** (`.litertlm`) format for on-device **streaming** speech recognition with Google's [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) runtime. Audio in, text out, one bundle: the runtime runs the bundled audio encoder on each chunk and the LLM emits that chunk's transcript while the next chunk is still being recorded.

VibeVoice-ASR-Streaming is Microsoft's streaming, speaker-attributed variant of VibeVoice-ASR: the σ-VAE acoustic + semantic tokenizers (24 kHz → 7.5 latent frames/s) feed a **Qwen2.5-1.5B-shaped language model** that was trained to transcribe **2.93 s chunks with 0.53 s of look-ahead**, prefixing each segment with a speaker label (`Speaker 0: …`). 10 languages, MIT-licensed.

**This bundle is driven as a multi-turn conversation, one audio window per turn** — that is how the model was trained and the only way it transcribes correctly (fed a whole clip in one turn it returns the last 3.5 s only). A chat app that sends one clip per message (e.g. the AI Edge Gallery) is not the right host; the loop below is ~15 lines with the Python, Kotlin or Swift Conversation API, or the CLI's `--multi_turns` mode.

| File | Recipe | Size |
|---|---|---|
| `VibeVoice-ASR-Streaming-1.5B.litertlm` | LM int4 blockwise-128 (min-max), int8 embedding · audio encoder int8 dynamic linears, fp32 convs, 3.47 s window | 1.99 GB |
| `VibeVoice-ASR-Streaming-1.5B_int8.litertlm` | LM int8 dynamic (desktop build, +0.6 pp accuracy on the fixture set) · same encoder | 2.77 GB |

Sections (int4 file): prefill/decode 802 MB, embedder 237 MB, audio encoder 951 MB, tokenizer; the int8 file swaps in a 1581 MB prefill/decode section.

## How the streaming loop works

The bundle's template is the vendor's streaming protocol, verbatim: a bare prompt (`You are a helpful assistant that transcribes audio input into text output. Please transcribe the following audios streamingly with these keys: speaker, content`), then per turn `<|object_ref_start|>` + 26 audio embeddings + `<|object_ref_end|>` → the model writes the chunk's text and stops on `<|text_chunk_end|>`, which the runtime appends to the history before the next turn. Nothing is re-encoded between turns; the KV cache grows by ~40 tokens per 2.93 s of speech (2048-token budget ≈ 2.5 minutes of continuous audio per conversation).

Per turn the app sends **one window of 83 200 samples at 24 kHz (26 frames × 3200 = 3.467 s)** and then advances by **70 400 samples (22 frames = 2.933 s)**, keeping the last 4 frames as the next window's look-ahead. Zero-pad the final window to 83 200 samples. Hotwords go into a system message: `… with these keys: speaker, content and extra info: <comma-separated words>`.

## Correctness

20 LibriSpeech dev-clean clips (448 words, 2–29 s, 73 windows), greedy decoding, the loop above, WER after uppercasing, stripping punctuation and the `Speaker N:` label:

| Configuration | int4 file | int8 file |
|---|---|---|
| PyTorch fp32 reference (vendor `streaming_generate`, mean latents) | 7.59 % (34/448) | 7.59 % (34/448) |
| **LiteRT-LM 0.16.1, Apple M4 Max, CPU** | 7.81 % (35/448) | **7.14 % (32/448)** |
| LiteRT-LM 0.16.1, Apple M4 Max, LM on Metal GPU, audio on CPU | 6.92 % (31/448) | 7.37 % (33/448) |
| LiteRT-LM v0.16.1 CLI `--multi_turns`, **Galaxy S26** (SM-S942Q), CPU | 7.81 % (35/448) | 7.14 % (32/448) |
| LiteRT-LM v0.16.1 CLI, Galaxy S26, LM on OpenCL GPU, audio on CPU | 7.14 % (32/448) | 6.92 % (31/448) |

The runtime rows sit within ±0.7 pp of the fp32 reference in both directions (the int8 LM gets one clip right that the fp32 reference misses); the residual ~7 % on this set is the model's own streaming trade-off — the non-streaming [VibeVoice-ASR-BitNet](https://huggingface.co/litert-community/VibeVoice-ASR-BitNet) bundle scores 2.68 % on the same 20 clips with a 30 s window. Microsoft's card gives no LibriSpeech number for the streaming model. LiteRT-LM 0.17.1 (Python) produced transcripts identical to 0.16.1 on the clips checked.

## Usage

```python
import litert_lm
from litert_lm import Message, Contents, Content
from litert_lm.interfaces import CPU, GPU

SR, HOP, WIN, ADV = 24000, 3200, 26 * 3200, 22 * 3200   # 3.467 s window, 2.933 s advance
engine = litert_lm.Engine("VibeVoice-ASR-Streaming-1.5B.litertlm", backend=GPU(), audio_backend=CPU())
conv = engine.create_conversation(sampler_config=litert_lm.SamplerConfig(top_k=1, top_p=1.0, temperature=0.0),
                                  max_output_tokens=256)
pcm = load_mono_float_24k("clip.wav")          # your decoder; any length
pos, text = 0, []
while pos < len(pcm):
    window = pcm[pos:pos + WIN]
    window = window + [0.0] * (WIN - len(window))  # zero-pad the last window
    write_wav("window.wav", window, SR)           # 16-bit PCM mono
    resp = conv.send_message(Message.user(Contents.of([Content.AudioFile("window.wav")])))
    text.append("".join(c.text for c in resp.contents.contents))
    print(text[-1], flush=True)                    # appears ~0.3–0.5 s after each window
    pos += ADV
conv.close()
```

```bash
# CLI (litert_lm_advanced_main): one [audio:…] line per window on stdin, an empty line ends the conversation
printf '[audio:w00.wav]\n[audio:w01.wav]\n[audio:w02.wav]\n\n' | \
  ./litert_lm_advanced_main --multi_turns=true --backend=gpu --audio_backend=cpu \
      --model_path=VibeVoice-ASR-Streaming-1.5B.litertlm --max_num_tokens=2048
```

Notes:

- Greedy decoding (top_k 1) is what the vendor runtime does. Each turn's text begins with the speaker label on its first emission (`Speaker 0:`) and continues mid-sentence on later turns — concatenate the turns as-is.
- The **audio encoder must run on the CPU** (`audio_backend`); the LM runs on CPU or GPU (recommended).
- Windows shorter than 3.467 s are fine for the last chunk only (pad with zeros); send full windows otherwise — the runtime treats the encoder window as fixed.

## Performance

`litert-lm benchmark` (litert-lm 0.16.0), Apple M4 Max, `-p 256 -d 256 --runs 3 --cache no`, text prompt (LM only), quiet machine, ≥ 300 s rest before each GPU reading:

| File | Backend | Prefill (256) | Decode | TTFT | Init |
|---|---|---|---|---|---|
| int4 | **GPU (Metal)** | **1766 tok/s** | **113.7 tok/s** | 0.15 s | 1.5 s |
| int4 | CPU | 305 tok/s | 56.0 tok/s | 0.86 s | 1.2 s |
| int8 | GPU (Metal) | 1977 tok/s | 139.6 tok/s | 0.14 s | 1.6 s |
| int8 | CPU | 251 tok/s | 46.8 tok/s | 1.04 s | 6.1 s |

End-to-end streaming on the fixture set (engine loaded, one conversation per clip, 73 turns): time per 2.93 s turn (encoder + prefill of 28 tokens + ~10 decoded tokens) — int4: **0.33 s with the LM on Metal** (real-time factor 0.13), 0.51 s on CPU (0.20); int8: 0.29 s / 0.48 s. The audio encoder is ~0.15–0.19 s of that (8 threads).

Galaxy S26 (SM-S942Q, Snapdragon SM8850, Adreno), `litert_lm_advanced_main` built from the litert-lm v0.16.1 tag, `--benchmark --benchmark_prefill_tokens=256 --benchmark_decode_tokens=256`, one reading per cell, CPU uncapped and SKIN < 40 °C before each leg (LM only):

| File | Backend | Prefill (256) | Decode | TTFT | Peak private footprint (streaming run) |
|---|---|---|---|---|---|
| int4 | **GPU (OpenCL)** | 628 tok/s | 46.3 tok/s | 0.43 s | 1483 MB |
| int4 | CPU | 317 tok/s | 46.5 tok/s | 0.83 s | 2511 MB |
| int8 | GPU (OpenCL) | 610 tok/s | 30.5 tok/s | 0.45 s | 1613 MB |
| int8 | CPU | 514 tok/s | 32.8 tok/s | 0.53 s | 3159 MB |

On the phone the streaming loop costs about **1.1 s per 2.93 s turn with the LM on the GPU** and 0.6 s on the CPU (int4 file; int8: 1.1 s / 0.8 s) — the slope of process wall-clock over window count across the 20 clips (intercept = engine load, 3–6 s per process; ±0.2 s on the slope — the per-process walls scatter by 2–3 s), i.e. a real-time factor of roughly 0.2–0.4, so the transcript keeps up with live speech.

## Conversion notes

Converted with `litert-torch` 0.9.3/0.9.4, ai-edge-quantizer 0.9.0, litert-lm-builder 0.16.1, transformers 5.14.1 (native `vibevoice_asr` classes). Scripts and the full recipe: [hf-to-litertlm](https://github.com/john-rocky/hf-to-litertlm) `vibevoice_asr_streaming_work/`.

- **Protocol, not prompt.** The checkpoint shares its weight layout with VibeVoice-ASR-BitNet but was trained on the interleaved chunk protocol of the vendor's `streaming_generate` (bare prompt without ChatML, per-window speech markers, `<|text_chunk_end|>` after each text segment). The bundle encodes that as prompt templates with empty prefixes/suffixes, the model suffix `<|text_chunk_end|>`, and stop tokens `<|text_chunk_end|>` / `<|endoftext|>`; the runtime's conversation prefix-caching then reproduces the vendor loop exactly (the sampled stop token never enters the cache, the suffix is prefilled with the next turn).
- **Audio encoder** = acoustic + semantic conv encoders + projector as one single-signature tflite, `audio` f32 `[1, 26, 3200]` → `features` f32 `[1, 26, 1536]`, raw 24 kHz PCM framed by the runtime (the checkpoint's `preprocessor_config.json` sets `normalize_audio: false`, so no RMS normaliser this time). Acoustic latents are the mean (the vendor Python demo samples noise at inference; the mean transcribes equally on the fixtures). Each window is encoded independently — the 4-frame look-ahead overlap is the app's job because the runtime's built-in windowing has no overlap without a Gemma-3n-style adapter.
- **LM**: dense bf16 Qwen2.5-1.5B-shaped weights (no BitNet ternary here), tied head, exported through the same driver as every dense LLM in the collection: int4 blockwise-128 for the phone file, int8 dynamic for the desktop file.
- **Tokenizer**: the upstream `tokenizer.json` already has strings for every marker (`<|object_ref_start|>`, `<|object_ref_end|>`, `<|box_start|>`, `<|text_chunk_end|>` = 151646/151647/151648/151665), so it ships unmodified.
- **GPU**: the LM runs on Metal (macOS) and the Android OpenCL delegate; the audio encoder returns an empty transcript on both GPU delegates (also with fp32 activations — a conv-stack issue, not the fp16 range), so keep `audio_backend` on CPU.

## License and changes

Distributed under the **MIT License** (inherited from the base model; see `LICENSE`). **Changes from the original work:** weights converted from safetensors bf16 to LiteRT flatbuffers with the quantization described above; the VAE decoder (synthesis half) and the diffusion head are not included; the streaming prompt protocol and audio-preprocessing parameters embedded as LiteRT-LM metadata.
