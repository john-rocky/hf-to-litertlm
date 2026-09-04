---
license: mit
base_model: microsoft/VibeVoice-ASR-BitNet
pipeline_tag: automatic-speech-recognition
library_name: litert-lm
language:
- en
- zh
- fr
- it
- ko
- pt
- vi
tags:
  - litert
  - litert-lm
  - litertlm
  - on-device
  - edge
  - asr
  - speech-recognition
  - audio
  - bitnet
  - vibevoice
---

# VibeVoice-ASR-BitNet — LiteRT-LM

[microsoft/VibeVoice-ASR-BitNet](https://huggingface.co/microsoft/VibeVoice-ASR-BitNet) converted to the **LiteRT-LM** (`.litertlm`) format for on-device speech recognition with Google's [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) runtime. **Audio in, text out, in one bundle**: the runtime decodes the clip, runs the bundled audio encoder, and the LLM transcribes — no host-side feature extraction.

VibeVoice-ASR-BitNet is Microsoft's edge variant of VibeVoice-ASR: the same σ-VAE acoustic + semantic tokenizers (24 kHz → 7.5 latent frames/s) feeding a **Qwen2.5-1.5B-shaped language model trained with BitNet b1.58 ternary weights**, multilingual (en, zh, fr, it, ko, pt, vi), MIT-licensed. This is the first audio-in `.litertlm` in this collection.

| File | Recipe | Size |
|---|---|---|
| `VibeVoice-ASR-BitNet.litertlm` | LM: per-tensor ternary weights stored as int4 blockwise-128 (exact), int8 embedding · audio encoder: int8 dynamic linears, fp32 convs, 30 s window | 1.98 GB |

Sections: prefill/decode 793 MB, embedder 237 MB, audio encoder 951 MB, tokenizer.

## Correctness

20 LibriSpeech dev-clean clips (448 words, 2–29 s), greedy decoding, the vendor's prompt (`This is a X.XX seconds audio, please transcribe it.`), WER after uppercasing and stripping punctuation:

| Configuration | WER |
|---|---|
| PyTorch fp32 reference (ternarized LM, deterministic latents) | 2.68 % (12/448) |
| **LiteRT-LM 0.16.1, Apple M4 Max, CPU** | **2.68 % (12/448)** — transcripts identical to the reference |
| LiteRT-LM 0.16.1, Apple M4 Max, LM on Metal GPU, audio on CPU | 2.46 % (11/448) |
| LiteRT-LM v0.16.1 CLI, **Galaxy S26** (SM-S942Q, Snapdragon SM8850), CPU | 3.12 % (14/448) |
| LiteRT-LM v0.16.1 CLI, Galaxy S26, LM on OpenCL GPU, audio on CPU | 2.46 % (11/448)* |

\* measured on the preceding build of this file: identical LM sections, the audio encoder differing only in its normalisation arithmetic, whose CPU transcripts are the same on every platform (the re-run on the shipped build agreed on its first two clips before the phone dropped off USB).

The Android CPU row differs from the Mac by one clip ("On the general principles…" → "Under general principles…"), an int8 dynamic-range numerics flip on Arm (a Pixel 8a produced the identical transcript); the GPU LM row matches the Mac GPU row word for word. For reference, Microsoft reports 2.41 % on the full LibriSpeech test-clean set with their VibeASR.cpp runtime.

Sending the clip **without** the duration sentence works (the bundle's template then sends "Please transcribe it.") at 3.57 % on the same set — include the duration for best results.

## Usage

The audio goes in as an audio content item; any container/sample rate the runtime's decoder reads (wav, mp3, flac — it resamples to 24 kHz). The **audio encoder must run on the CPU** (`audio_backend`), the LM may run on CPU or GPU.

```python
import litert_lm
from litert_lm import Message, Contents, Content
from litert_lm.interfaces import CPU, GPU

engine = litert_lm.Engine("VibeVoice-ASR-BitNet.litertlm", backend=CPU(), audio_backend=CPU())
dur = 5.86  # clip length in seconds
conv = engine.create_conversation(sampler_config=litert_lm.SamplerConfig(top_k=1, top_p=1.0, temperature=0.0),
                                  max_output_tokens=256)
resp = conv.send_message(Message.user(Contents.of([
    Content.AudioFile("/abs/path/clip.wav"),
    Content.Text(f"This is a {dur:.2f} seconds audio, please transcribe it."),
])))
print("".join(c.text for c in resp.contents.contents))
conv.close()
```

```bash
# CLI (litert_lm_advanced_main / litert-lm run); the [audio:…] tag goes first
./litert_lm_advanced_main --backend=cpu --audio_backend=cpu --model_path=VibeVoice-ASR-BitNet.litertlm \
  --input_prompt='[audio:/path/clip.wav] This is a 5.86 seconds audio, please transcribe it.'
```

Notes:

- Greedy decoding (top_k 1) is what the vendor runtime does; the model emits plain text and stops on `<|im_end|>`.
- The encoder window is **30 s**. Longer clips are chunked by the engine (30 s windows, no overlap), which resets the encoder's context at each boundary; a 10 s window measured +1.6 pp WER on this set from those resets, which is why the shipped window is 30 s.
- KV budget 2048 tokens: 30 s of audio is 225 tokens plus the ~40-token prompt, leaving room for the transcript.

## Performance

`litert-lm benchmark` (litert-lm 0.16.0), Apple M4 Max, `-p 256 -d 256 --runs 3 --cache no`, text prompt (LM only — the audio encoder is a fixed cost per 30 s window: 1.3–1.7 s on the M4 Max CPU at 8 threads), quiet machine, ≥ 300 s rest before the GPU reading:

| Backend | Prefill (256) | Decode | TTFT | Init |
|---|---|---|---|---|
| **GPU (Metal)** | **2045 tok/s** | **138.6 tok/s** | 0.13 s | 1.4 s |
| CPU | 357 tok/s | 59.3 tok/s | 0.81 s | — |

End-to-end on the fixture set (engine already loaded, one conversation per clip): real-time factor **0.24 on CPU** and **0.19 with the LM on Metal**, i.e. a 10 s clip transcribes in about 2 s.

Galaxy S26 (SM-S942Q, Snapdragon SM8850, Adreno), `litert_lm_advanced_main` built from the litert-lm v0.16.1 tag, `--benchmark` with a 264-token text prompt, one reading per cell (LM only):

| Backend | Prefill (264) | Decode | TTFT |
|---|---|---|---|
| **GPU (OpenCL)** | **652 tok/s** | **36.1 tok/s** | 0.43 s |
| CPU | 163 tok/s | 35.4 tok/s | 1.65 s |

On the phone a 5–20 s clip transcribes in roughly 7–10 s of wall-clock from a cold process (CPU), including engine load; peak private footprint 3.2 GB (CPU) / 2.3 GB (LM on GPU).

## Conversion notes

Converted with `litert-torch` 0.9.3/0.9.4, ai-edge-quantizer 0.9.0, litert-lm-builder 0.16.1, transformers 5.14.1 (native `vibevoice_asr`). Scripts and the full recipe: [hf-to-litertlm](https://github.com/john-rocky/hf-to-litertlm) `vibevoice_asr_work/`.

- **Ternarization is applied at conversion, per tensor, exactly as the vendor runtime does.** The Hub checkpoint stores fp32 latent weights; VibeASR.cpp quantizes the seven projections per LM layer with a per-tensor absmean scale at GGUF-conversion time. The same values are stored here as int4 blockwise-128 min-max, which represents a per-tensor ternary exactly (every block is {−α, 0, +α}); the embedding/head stays int8.
- **Audio front-end folded into the encoder graph.** Input is raw 24 kHz PCM framed by the runtime (3200 samples = one latent frame); the vendor's −25 dBFS RMS normalisation and peak clip run in-graph. Acoustic latents are the mean (no sampling noise), as in the vendor runtime.
- **Tokenizer**: the upstream `tokenizer.json` (Qwen2) has no string for the speech-marker ids the model was trained with; `<|object_ref_start|>` / `<|object_ref_end|>` / `<|box_start|>` are added as special tokens on ids 151646/151647/151648 so the runtime can emit them around the audio embeddings. Ordinary text tokenizes identically.
- **Prompt**: the vendor's system prompt is baked into the bundle template; the user turn renders as `<|object_ref_start|>` + audio embeddings + `<|object_ref_end|>` + newline + your text.
- **GPU**: the LM runs on Metal (macOS) and on the Android OpenCL delegate; the audio encoder is CPU-only — on Metal/WebGPU the 30 s window exceeds the delegate's 65 535-workgroup dispatch limit, and on Adreno the fp16 delegate path returns an empty transcript. Keep `audio_backend` on CPU.

## License and changes

Distributed under the **MIT License** (inherited from the base model; see `LICENSE`). **Changes from the original work:** weights ternarized (LM) and converted from safetensors fp32 to LiteRT flatbuffers with the quantization described above; the VAE decoder (synthesis half) is not included; three special tokens added to the tokenizer; chat/prompt template and audio-preprocessing parameters embedded as LiteRT-LM metadata.
