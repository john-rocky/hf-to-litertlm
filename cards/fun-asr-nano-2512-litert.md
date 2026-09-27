---
license: apache-2.0
base_model: FunAudioLLM/Fun-ASR-Nano-2512
pipeline_tag: automatic-speech-recognition
library_name: litert-lm
language:
- zh
- en
- ja
tags:
  - litert
  - litert-lm
  - litertlm
  - on-device
  - edge
  - asr
  - speech-recognition
  - audio
  - funasr
  - qwen3
---

# Fun-ASR-Nano-2512 — LiteRT-LM

[FunAudioLLM/Fun-ASR-Nano-2512](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512) converted to the **LiteRT-LM** (`.litertlm`) format for on-device speech recognition with Google's [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) runtime. Audio in, text out, one file.

Fun-ASR-Nano-2512 is Tongyi Lab's end-to-end speech recognition model. A SenseVoice SAN-M audio encoder (70 layers) and a 2-layer adaptor feed a **Qwen3-0.6B** language model, which writes the transcript. It covers Chinese (with 7 dialects and 26 regional accents), English and Japanese, and accepts hotwords in the prompt. License: Apache-2.0.

The bundle runs on LiteRT-LM's generic audio path. The runtime cuts 16 kHz PCM into 60 ms frames, runs the bundled encoder, and puts the audio embeddings into a ChatML prompt. It needs no change to the runtime or the converter.

| File | Recipe | Size |
|---|---|---|
| `Fun-ASR-Nano-2512.litertlm` | LM int8 dynamic (weights) with an int8 embedding table, fp32 activations declared for the GPU · audio encoder fp16 weights, fp32 compute, 30.24 s window | 1.26 GB |

| Section | Content | Size |
|---|---|---|
| `tf_lite_audio_encoder_hw` | fbank + LFR front end, SAN-M encoder, adaptor | 473 MB |
| `tf_lite_prefill_decode` | Qwen3-0.6B, int8 | 621 MB |
| `tf_lite_embedder` | token embedding table, int8 | 159 MB |
| `HF_Tokenizer_Zlib` | Qwen3 tokenizer | 2.1 MB |
| `LlmMetadataProto` | prompt template, stop tokens, audio settings | 1.7 KB |

## How to send audio

**One message = one utterance of up to 30 s.** The encoder reads a fixed 30.24 s window (504 frames of 960 samples at 16 kHz) and returns up to 63 embeddings. A longer clip in one message is cut by the runtime into several windows. On a 60.2 s test clip the model then transcribed only the first window's speech (WER 83/143).

**Longer audio: split it and use a new conversation per piece.** Cut the audio into pieces of at most 30 s (at pauses if you can), send each piece in a fresh conversation, and join the texts. On the same 60.2 s clip, cut at 30.24 s, this scored WER 10/143, the same as the original model on the whole clip (10/143). Sending the two pieces as two turns of one conversation scored 22/143: the second turn repeats the start of the first.

**Prompt.** A message with audio only gets the model's default instruction `语音转写：` ("transcribe the speech") from the bundle's template. To change the task, put one of the model's own instructions in the same message as a text item; the template places it before the audio:

| Task | Text item |
|---|---|
| Transcribe (default, no text item needed) | `语音转写：` |
| Transcribe without inverse text normalization | `语音转写，不进行文本规整：` |
| Transcribe in a given language (`中文`, `英文`, `日文`) | `语音转写成英文：` |
| Hotwords | see below |

With the instruction that skips inverse text normalization, the 20 LibriSpeech clips (Mac CPU) scored English WER 17/448. The original model with the same instruction scored 16/448, and 16 of 20 transcripts are identical.

Hotwords use the full instruction below, with the words in the list:

```text
请结合上下文信息，更加准确地完成语音转写任务。如果没有相关信息，我们会留空。


**上下文信息：**


热词列表：[Birket Foster, John Collier]
语音转写：
```

On a LibriSpeech clip with these two hotwords, the bundle wrote "John Collier" instead of "Joan Collier". The clip's WER went from 10/68 to 8/68 (the original model: 11/68 to 6/68). "Birket" still came out as "Burkett", as it does in the original model.

Notes:

- Send 16 kHz mono PCM WAV. The runtime can also decode MP3 itself, but on 3 of 5 official MP3 samples its text differed from the text of the same audio decoded to WAV by ffmpeg.
- The encoder takes the clip length from the last non-zero sample (the runtime passes no length), so digital silence at the very end is treated as padding. On the test clips this changed no transcript.
- No timestamps: the checkpoint has no CTC decoder weights, so this bundle has no CTC branch.

## Correctness

25 clips: the model's 5 example clips (zh, en, ja, ko, yue) and 20 LibriSpeech dev-clean clips (448 words). Reference = the original model in `funasr` 1.4.16, fp32, greedy decoding. All runs below use greedy decoding (top-k 1), one conversation per clip, and the audio encoder on the CPU.

| Runtime | LM backend | Same text as the reference | English WER |
|---|---|---|---|
| Reference: `funasr`, fp32 PyTorch | — | — | 4.46 % (20/448) |
| LiteRT-LM 0.17.1 (Python), Apple M4 Max | CPU | 21/25 | 4.24 % (19/448) |
| LiteRT-LM 0.17.1 (Python), Apple M4 Max | GPU (WebGPU over Metal) | 24/25 | 4.46 % (20/448) |
| LiteRT-LM v0.16.1 CLI, Galaxy S26 (SM-S942Q) | CPU | 21/25 | 4.46 % (20/448) |
| LiteRT-LM v0.16.1 CLI, Galaxy S26 (SM-S942Q) | GPU (OpenCL) | 24/25 | 4.46 % (20/448) |

No run produced an empty or looping output. With the encoder and the language model left in fp32, the runtime's text equals the reference on 25/25 clips, so the differences above come from quantization. On the CPU they are one surname's spelling ("Birket" on the Mac, "Burkett" on the S26, "Burkitt" in the reference), punctuation on two clips, and kana on the Korean example (the model card lists Chinese, English and Japanese for this model; the reference model also returns kana for that clip). On the GPU the only difference is the punctuation of one quote.

FLEURS test, 50 clips per language, same reference model and settings:

| Language | Metric | This bundle (Mac CPU) | Reference (`funasr` fp32) |
|---|---|---|---|
| English | WER | 5.48 % | 5.13 % |
| Mandarin | CER | 6.86 % | 6.86 % |
| Japanese | CER | 6.81 % | 6.95 % |

WER counts words after uppercasing and removing punctuation. CER counts characters after NFKC normalization and lowercasing, with all punctuation, separators and whitespace removed. Ground truth: FLEURS `raw_transcription`.

## Usage

Python (`pip install litert-lm-api`, 0.17.1):

```python
import os
import litert_lm
from litert_lm import Content, Contents, Message
from litert_lm.interfaces import CPU, GPU

engine = litert_lm.Engine("Fun-ASR-Nano-2512.litertlm", backend=GPU(), audio_backend=CPU())
sampler = litert_lm.SamplerConfig(top_k=1, top_p=1.0, temperature=0.0)

def transcribe(wav_path, instruction=None):
    items = [Content.AudioFile(os.path.abspath(wav_path))]  # 16 kHz mono WAV, up to 30 s
    if instruction:                                          # e.g. "语音转写，不进行文本规整："
        items.insert(0, Content.Text(instruction))
    conv = engine.create_conversation(sampler_config=sampler, max_output_tokens=512)
    try:
        resp = conv.send_message(Message.user(Contents.of(items)))
    finally:
        conv.close()
    return "".join(part["text"] for part in resp["content"] if part["type"] == "text")

print(transcribe("clip.wav"))
```

Kotlin (LiteRT-LM Android/JVM API):

```kotlin
val engine = Engine(EngineConfig(
    modelPath = "/path/to/Fun-ASR-Nano-2512.litertlm",
    backend = Backend.GPU(),        // or Backend.CPU()
    audioBackend = Backend.CPU(),   // the audio encoder runs on the CPU
    cacheDir = context.cacheDir.path,
))
engine.initialize()

// One conversation per utterance (up to 30 s).
val text = engine.createConversation(ConversationConfig(
    samplerConfig = SamplerConfig(topK = 1, topP = 1.0, temperature = 0.0),
    maxOutputToken = 512,
)).use { conversation ->
    conversation.sendMessage(Contents.of(Content.AudioFile("/path/to/clip.wav"))).toString()
}
```

On Android the GPU backend also needs `<uses-native-library android:name="libOpenCL.so" android:required="false"/>` inside `<application>` in the app manifest.

CLI (`litert_lm_advanced_main`, built from the LiteRT-LM v0.16.1 tag; the command used for the Galaxy S26 rows):

```bash
./litert_lm_advanced_main --backend=gpu --audio_backend=cpu --sampler_backend=cpu \
    --model_path=Fun-ASR-Nano-2512.litertlm --max_num_tokens=2048 --max_output_tokens=512 \
    --input_prompt='[audio:/path/to/clip.wav]'
```

Keep `audio_backend` on the CPU. The language model runs on the CPU or the GPU.

## Performance

`litert-lm benchmark` (litert-lm 0.17.1), Apple M4 Max, `-p <P> -d 256 --runs 3 --cache no`, text prompt (language model only), one reading per cell, at least 300 s of rest before each GPU cell. Measured with a background load average of 5.8–16.0 (1-minute, at the start of each measurement); not a quiet machine.

| Backend | Prefill, p=256 | Prefill, p=512 | Decode | TTFT (p=256) | Init |
|---|---|---|---|---|---|
| GPU (WebGPU over Metal) | 3,091 tok/s | 6,170 tok/s | 143.0 tok/s | 0.10 s | 1.1 s |
| CPU | 1,027 tok/s | 1,350 tok/s | 48.9 tok/s | 0.48 s | 2.1 s |

The prefill signatures are 512, 128 and 32 tokens. At p=256 the runtime runs one padded 512-token call on the GPU and two 128-token calls on the CPU, so the p=512 column shows the GPU's unpadded rate.

End to end on the Mac (audio in, text out; engine loaded once, one conversation per clip, the 25 clips above, 217.5 s of audio): 0.37 s per clip with the language model on the GPU (real-time factor 0.046), 0.68 s on the CPU (0.091). The audio encoder alone takes 164 ms per 30.24 s window (8 threads). With the runtime cache from earlier runs in place, creating the engine took 0.1 s (CPU) / 0.2 s (GPU), and the first clip 1.45 s / 0.51 s.

Galaxy S26 (SM-S942Q, Snapdragon SM8850), `litert_lm_advanced_main` built from the LiteRT-LM v0.16.1 tag, `--benchmark --benchmark_prefill_tokens=256 --benchmark_decode_tokens=256`, text prompt (language model only), one reading per backend, CPU uncapped and SKIN below 40 °C before each leg:

| Backend | Prefill (256) | Decode | TTFT | Peak memory (25-clip run) |
|---|---|---|---|---|
| GPU (OpenCL) | 386 tok/s | 26.3 tok/s | 0.70 s | 2,233 MB |
| CPU | 331 tok/s | 31.6 tok/s | 0.80 s | 2,411 MB |

With the engine loaded once (a Kotlin app on LiteRT-LM Android 0.17.1, language model and audio encoder on the CPU), the three official samples took 1.03 s (5.6 s of Chinese), 1.24 s (7.2 s of English) and 1.42 s (7.2 s of Japanese) from sendMessage to the text; engine load 0.64 s with the runtime cache present. No such measurement was made with the language model on the GPU.

## Conversion notes

Converted with `litert-torch` 0.9.4 (audio encoder) and 0.9.3 (language model), `ai-edge-quantizer` 0.9.0 / 0.8.0, `litert-lm-builder` 0.16.1 and `transformers` 5.14.1, with the converter scripts of `hf-to-litertlm`. Source: the tensors of `FunAudioLLM/Fun-ASR-Nano-2512-vllm`, which equal the official `model.pt` bit for bit.

- **Audio encoder = one graph from raw PCM.** The Kaldi fbank, the 7-frame low-frame-rate stacking (LFR), the 70-layer SAN-M encoder and the adaptor are exported together: `audio` f32 `[1, 504, 960]` → `features` f32 `[1, 63, 1024]` + `mask` uint8 `[1, 63]`. The runtime keeps the embeddings up to the last non-zero `mask` entry, so the graph decides how many tokens a clip gets. In fp32 the graph's features match the original model's within cos ≥ 0.99999 on 23 of 25 clips. The other two clips end in digital silence and lose one 60 ms feature frame (see Notes above); their transcripts are unchanged.
- **fp16 encoder weights.** With the weights cast to fp16 (compute stays fp32), the encoder plus an fp32 language model gives the reference text on 25/25 clips (worst cosine 0.9999983 against the fp32 encoder). Int8 dynamic weights on the encoder's linear layers changed 6 of 25 transcripts (English WER 17/448), so this bundle does not use them.
- **Language model.** Int8 dynamic weights with an int8 embedding table, stored as a separate embedder section. An int4 variant (blockwise, 978 MB bundle) matched the reference on 14/25 clips with English WER 17/448. It is not shipped.
- **fp32 activations on the GPU.** With the GPU's default fp16 activations the language model emits only `!` tokens, also for a text-only prompt. The prefill/decode section therefore declares `prefer_activation_type: fp32`, which LiteRT-LM applies when the app sets no activation type. The CPU transcripts are the same with and without it.
- **Prompt contract in metadata.** The ChatML template, the default instruction `语音转写：`, the stop tokens (`<|im_end|>`, `<|endoftext|>`) and the audio settings (16 kHz, one channel, 960-sample frames, no mel extraction in the runtime) are stored in the bundle.

## Known issues

- The audio encoder does not run on the GPU. On the Mac the GPU delegate does not take three of its op types (`DEQUANTIZE`, `PAD`, `SELECT_V2`) and the conversation fails to start. Keep `audio_backend` on the CPU.
- One message holds at most 30.24 s of audio. Split longer audio and use one conversation per piece (see above).
- MP3 decoded by the runtime can give different text than the same audio as WAV; send WAV.
- Quantization changes a few transcripts slightly: a surname's spelling and some punctuation (see Correctness).
- No timestamps or CTC output.

## License and changes

Distributed under the **Apache License 2.0**, the license declared by the base model. The official model repository has no LICENSE file; its model card metadata declares Apache-2.0. The `LICENSE` file here is the Apache-2.0 text from `FunAudioLLM/Fun-ASR-Nano-2512-vllm`. `NOTICE` names the base model (Tongyi Lab / FunAudioLLM) and Qwen3-0.6B (Alibaba Cloud, Apache-2.0).

**Changes from the original work:** weights converted from the PyTorch checkpoint to LiteRT flatbuffers; the language model quantized to int8 and the audio encoder weights cast to fp16; the audio front end, encoder and adaptor exported as one graph; the prompt format stored as LiteRT-LM metadata. No CTC decoder is included (the checkpoint has no CTC decoder weights).

Other ports of this model exist for MLX (mlx-community), ONNX (sherpa-onnx, csukuangfj) and llama.cpp (the official GGUF files).
