---
license: apache-2.0
base_model: Qwen/Qwen3-ASR-1.7B-hf
base_model_relation: quantized
pipeline_tag: automatic-speech-recognition
library_name: litert-lm
language:
- zh
- en
- yue
- ar
- de
- fr
- es
- pt
- id
- it
- ko
- ru
- th
- vi
- ja
- tr
- hi
- ms
- nl
- sv
- da
- fi
- pl
- cs
- fil
- fa
- el
- hu
- mk
- ro
tags:
  - litert
  - litert-lm
  - litertlm
  - on-device
  - edge
  - asr
  - speech-recognition
  - audio
  - qwen3-asr
---

# Qwen3-ASR-1.7B — LiteRT-LM

[Qwen/Qwen3-ASR-1.7B-hf](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) converted to the **LiteRT-LM** (`.litertlm`) format for on-device speech recognition with Google's [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) runtime. Audio in, text out, one file.

Qwen3-ASR-1.7B is Qwen's speech recognition model (2.04 B parameters). A 24-layer audio encoder feeds a 28-layer Qwen3 language model, which writes `language <name><asr_text><transcript>`. The model card lists language identification and recognition for 30 languages and 22 Chinese dialects; this file was scored on English, Mandarin and Japanese.

The files run on the generic audio path of the released runtime (the 30 s file measured with `litert-lm-api` 0.17.1, the 15 s file with 0.18.0). The runtime cuts 16 kHz PCM into 10 ms frames and hands the bundled encoder a window of 30 s or 15 s, depending on the file; the encoder computes the log-mel spectrogram itself. No runtime change is needed.

![The same sentence read in Mandarin, English and Japanese, transcribed on a Galaxy S26: each screen shows the transcript, the language the model named and the seconds the transcription took](assets/demo_s26.png)

Galaxy S26 (SM-S942Q), CPU with 4 threads, airplane mode on, LiteRT-LM 0.17.1 through the Kotlin Conversation API (`litertlm-android`), the 30 s file. The clips are FLEURS test sentence 1698 read in Mandarin, English and Japanese (CC BY 4.0; one linear gain per clip evens out the recording levels). The app plays each clip through the phone's speaker, then sends it in a new conversation. The seconds are the app's wall time of the call, with the language model and audio encoder on the CPU.

| File | Recipe | Size |
|---|---|---|
| `Qwen3-ASR-1.7B-15s.litertlm` | LM int8 dynamic (weights) with an int8 embedding table · audio encoder fp16 weights, fp32 compute, 15 s window · prefill signatures for 128 and 256 tokens | 2.70 GB |
| `Qwen3-ASR-1.7B.litertlm` | LM int8 dynamic (weights) with an int8 embedding table · audio encoder fp16 weights, fp32 compute, 30 s window | 2.69 GB |

The two files hold the same weights. The 15 s file encodes a clip in 15 s windows (195 audio tokens per window), so a clip of up to 15 s costs half the encoder and prefill work of the 30 s file; its measured response times are in the Performance section. The 30 s file is the one measured on the released 0.17.1 runtime and in the Android sample below.

| Section | Content | Size |
|---|---|---|
| `tf_lite_audio_encoder_hw` | log-mel front end (the original feature extractor's STFT, Slaney filterbank and normalization), 24-layer encoder, projector | 637 MB |
| `tf_lite_prefill_decode` | Qwen3 language model, int8, KV cache 1024 tokens | 1.74 GB (30 s file), 1.75 GB (15 s file) |
| `tf_lite_embedder` | token embedding table, int8 | 313 MB |
| `HF_Tokenizer_Zlib` | tokenizer | 2.1 MB |
| `LlmMetadataProto` | prompt template, stop tokens, audio settings | 0.9 kB |

## How to send audio

**One message holds one clip.** The runtime zero-fills the window after the clip, and the encoder turns every window into a fixed number of audio tokens: 390 for the 30 s file, 195 for the 15 s file. Encoding and prefill therefore cost the same for a 4 s clip as for a 14 s clip on the 15 s file (or a 29 s clip on the 30 s file); only decoding grows with the transcript. With the 15 s file a clip longer than 15 s goes through two windows: the 28 FLEURS clips of 15.1–29.3 s in the set below were scored that way. Clips longer than 30 s were not tested with either file.

**The answer is `language <name><asr_text><transcript>`.** Split it at `<asr_text>`; the scores below use the text after the tag. On the FLEURS clips below every Mandarin and Japanese clip was tagged correctly; one English clip (two on the GPU run) came back as `language None` followed by the English transcript.

**Forced language.** Add a text item `language Chinese<asr_text>` (or another language name) to the same message. The template places it at the start of the answer, as the `qwen-asr` package does for a forced language, and the model then writes the transcript only (11 of 11 Mandarin and Japanese clips tried).

**Prompt.** The template renders `<|im_start|>user<|audio_start|>` + audio + `<|audio_end|><|im_end|><|im_start|>assistant\n`, the prompt of litert-torch's Qwen3-ASR export. The checkpoint's own chat template adds an empty system turn and newlines. With that template this file scored Mandarin CER 6.65 % and Japanese CER 6.06 % on the set below, and 8 of 50 Mandarin clips came back tagged `English` (with a Chinese transcript). System messages are ignored by the template.

## Correctness

FLEURS test, 50 clips per language, ground truth `raw_transcription`. Greedy decoding, one conversation per clip.

| Run | English WER | Mandarin CER | Japanese CER |
|---|---|---|---|
| Original model, PyTorch fp32 (transformers 5.14.1), this file's prompt | 4.35 % (50/1150) | 6.04 % (118/1954) | 5.24 % (141/2689) |
| Original model, PyTorch fp32, the checkpoint's chat template | 4.35 % (50/1150) | 6.86 % (134/1954) | 5.50 % (148/2689) |
| **15 s file**, LiteRT-LM 0.18.0 (Python), Apple M4 Max, CPU | **4.35 %** (50/1150) | **6.19 %** (121/1954) | **5.84 %** (157/2689) |
| 15 s file, LiteRT-LM 0.18.0 (Python), Apple M4 Max, LM on the GPU (WebGPU over Metal), audio on the CPU | 4.44 % (51/1150) | 6.24 % (122/1954) | 5.21 % (140/2689) |
| **30 s file**, LiteRT-LM 0.17.1 (Python), Apple M4 Max, CPU | **4.43 %** (51/1150) | **6.19 %** (121/1954) | **5.73 %** (154/2689) |
| 30 s file, LiteRT-LM 0.17.1 (Python), Apple M4 Max, LM on the GPU (WebGPU over Metal), audio on the CPU | 4.52 % (52/1150) | 6.19 % (121/1954) | 5.32 % (143/2689) |

No clip came back empty. WER counts words after uppercasing and removing punctuation. CER counts characters after NFKC normalization and lowercasing, with all punctuation, separators and whitespace removed. The 30 s file on LiteRT-LM 0.18.0 scored 4.43 % / 6.40 % / 5.99 % on the CPU and 4.52 % / 6.19 % / 5.32 % on the GPU (the CPU transcripts of 20 of the 155 clips differ between the two runtime versions).

## Performance

### Response time per clip (both files)

One clip, warm process (the model loaded and the runtime caches present), greedy, the answer complete: the seconds cover the audio encoder, the prefill and the decoding of the transcript. The clip is the 6.4 s English FLEURS sentence 1698 shown above (23 output tokens); the Mandarin (9.7 s, 25 tokens) and Japanese (11.0 s, 38 tokens) clips of the same sentence are in parentheses. Mac: median of 5 calls in one process, LiteRT-LM 0.18.0 (Python), the machine's measurement window held but other processes running. Phones: the LiteRT-LM v0.18.0 prebuilt command-line tool, one process per clip, the phone rested to an unthrottled state before each process; on the phones the audio encoder runs on the CPU and its share moves with the phone's thermal state (on the S26, 0.25–0.76 s for the same clip and bytes within one session), so the phone rows are ranges over the processes measured. On the phones the Mandarin clip's process ran first in each session and wrote the runtime caches.

| Device | Backend | 15 s file | 30 s file |
|---|---|---|---|
| Apple M4 Max | LM on the GPU (WebGPU over Metal), audio encoder CPU | 0.34 s (0.36 / 0.46) | 0.54 s (0.56 / 0.65) |
| Apple M4 Max | CPU, 4 threads | 0.85 s (0.89 / 1.14) | 1.19 s (1.24 / 1.50) |
| Galaxy S26 (SM-S942Q) | LM on the GPU (OpenCL), audio encoder CPU | 1.40–1.51 s (1.60 / 2.07) | 2.11–2.81 s (2.82 / 2.77) |
| Galaxy S26 (SM-S942Q) | CPU, 4 threads | 1.55 s (2.18 / 2.17) | 2.11 s (2.18 / 2.70) |
| Pixel 8a | LM on the GPU (OpenCL), audio encoder CPU | 5.03–6.33 s (7.03 / 6.49) | 9.86–9.98 s (11.89 / 10.01) |
| Pixel 8a | CPU, 4 threads | 5.10 s (6.85 / 6.21) | 8.39 s (11.95 / 9.48) |

Galaxy S26, the 15 s file, LiteRT-LM v0.18.0 command-line tool with `--benchmark`, one process per clip, the prompt 203 tokens (195 audio + 8 text), the weight caches present:

| Backend | Prefill | Decode | Time to first token | Peak private footprint |
|---|---|---|---|---|
| LM on the GPU (OpenCL, `prefill_256` and `decode` fully delegated), audio encoder CPU | 762–787 tokens/s | 23.5–26.0 tokens/s | 0.30–0.31 s | 2.41 GB |
| CPU, 4 threads | 511–571 tokens/s (one process 243) | 22.9–25.7 tokens/s | 0.39–0.44 s (one process 0.88 s) | 4.10 GB |

Time to first token covers the prefill and the first decode step; the audio encoder's time comes before it (0.25–0.29 s on these runs). On the GPU the whole 203-token prompt goes through the `prefill_256` signature in one call; on the CPU the runtime feeds it through `prefill_128` in two calls, so the CPU numbers of the 15 s file equal the 30 s file's per token.

### The 30 s file on the Mac, 155 clips

Apple M4 Max, LiteRT-LM 0.17.1 (Python), the 150 FLEURS clips above plus 5 example clips (155 clips, 4–29 s, 1,836 s of audio). Other jobs were running on the machine (load average 4.5–8.5), so read the times as an upper bound:

| Backend | Processing time | Real-time factor | Per clip | Peak RSS |
|---|---|---|---|---|
| CPU, 4 threads | 272 s | 0.148 (6.7× faster than real time) | 1.04–3.17 s, median 1.70 s | 4.54 GB ¹ |
| LM on the GPU (WebGPU over Metal), audio encoder on the CPU | 105 s | 0.057 (17.5× faster than real time) | 0.53–1.06 s, median 0.66 s | 4.79 GB ² |

¹ With the CPU weight caches already written. Peak RSS includes the memory-mapped weights. The first load writes about 3.0 GB of CPU weight caches (encoder 1.27 GB, LM 1.73 GB) into `cache_dir`, or next to the model file when none is set; a first load that wrote them peaked at 6.62 GB.

² This run was the first GPU load and wrote the GPU caches (1.72 GB) during the run.

### The 30 s file on the Galaxy S26

Galaxy S26 (SM-S942Q), CPU, 4 threads, LiteRT-LM command-line tool built from the v0.16.1 release tag with `--benchmark`, one process per clip: an English, a Mandarin and a Japanese FLEURS clip, the English one twice. The prompt is 398 tokens (390 audio + 8 text). The CPU frequency limit was at its maximum before every run:

| Run | Prefill | Decode | Time to first token | Peak private footprint |
|---|---|---|---|---|
| First run, writes the weight caches | 329 tokens/s | 22.8 tokens/s | 1.25 s | 3.47 GB |
| Next 3 runs, caches loaded | 538–547 tokens/s | 25.0–25.6 tokens/s | 0.77–0.78 s | 4.23 GB |

All four transcripts equal the Mac CPU run.

## Usage

Python (`pip install litert-lm-api`; the 30 s file was run on 0.17.1, the 15 s file on 0.18.0). `transcribe` takes a 16 kHz mono WAV of up to 30 s; `prefix` is text the answer starts with, for example `language Chinese<asr_text>` to force the language, and the return value is the answer after it. Open the file you downloaded; the code is the same for both:

```python
import litert_lm
from litert_lm import Content, Contents, Message
from litert_lm.interfaces import CPU

engine = litert_lm.Engine("Qwen3-ASR-1.7B-15s.litertlm", backend=CPU(), audio_backend=CPU())
sampler = litert_lm.SamplerConfig(top_k=1, top_p=1.0, temperature=0.0)


def response_text(resp):
    # litert-lm-api 0.15-0.17.1 return a dict of content parts; later builds return a Message whose str() is its text
    if type(resp) is dict:
        return "".join(p.get("text", "") for p in resp.get("content", []) if isinstance(p, dict))
    return str(resp)


def transcribe(wav_path, prefix=""):
    items = [Content.AudioFile(wav_path)]
    if prefix:
        items.append(Content.Text(prefix))
    conv = engine.create_conversation(sampler_config=sampler)
    try:
        return response_text(conv.send_message(Message.user(Contents.of(*items))))
    finally:
        conv.close()


raw = transcribe("clip.wav")
print(raw.partition("<asr_text>")[2].strip())
```

For the Chinese example clip the 30 s file returns `language Chinese<asr_text>开放时间：早上九点至下午五点。` and the script prints the text after the tag; the 15 s file returns the same sentence in traditional characters, `開放時間：早上九點至下午五點。` (both files run with this block on litert-lm-api 0.18.0; on the 50 Mandarin FLEURS clips, whose ground truth is simplified, the 15 s file scores CER 6.19 %). `backend=GPU()` (from `litert_lm.interfaces`) puts the language model on the GPU. Keep `audio_backend=CPU()`: the audio encoder does not run on the GPU delegate (`create_conversation` fails). The Kotlin Conversation API takes the same calls. In the Galaxy S26 app above (`litertlm-android`), the engine is built with `Backend.CPU` for both the model and the audio, and each clip is sent as `Content.AudioFile` in a new conversation.

Sample app (Android): [`android/`](https://huggingface.co/litert-community/Qwen3-ASR-1.7B/tree/main/android) — one app for the 30 s file and two sibling bundles (Qwen3-ASR-1.7B, Fun-ASR-Nano-2512, Confucius4-R2T2): a model picker, hold-to-talk up to 30 s and three FLEURS clips, with a device check. Galaxy S26, language model and audio encoder on the CPU (4 threads), the 6.4 s English clip in 2.49 s and the 9.7 s Mandarin clip in 2.38 s of wall time per call. The 15 s file was not run in the sample app.

On Android the S26 rows above came from the command-line tool: `litert_lm_advanced_main --backend=cpu --audio_backend=cpu --sampler_backend=cpu --num_cpu_threads=4 --model_path=Qwen3-ASR-1.7B-15s.litertlm --cache_dir=cache --max_num_tokens=1024 --max_output_tokens=256 --benchmark --input_prompt='[audio:/path/clip.wav]'` (`--backend=gpu` for the GPU rows; the 30 s file with its own file name).

## License

Apache 2.0, the license of the original model.

## Conversion

The conversion differs from a default export in three places:

1. **Encoder attention windows.** The original encoder attends within 104-token (8 s) windows. The default litert-torch export for Qwen3-ASR attends within 13-token (1 s) chunks. These files use the original windows through a constant block-diagonal attention mask; the exported encoder then matches the original encoder's output exactly in fp32 (maximum difference 0.0 on 5 s and 30 s inputs; with 13-token chunks the cosine similarity of the encoder output is 0.76–0.81).
2. **Log-mel inside the encoder.** The runtime's built-in mel front end uses a different filterbank and log scale than the original feature extractor, so the files ask the runtime for raw PCM frames and compute the original log-mel in the graph. On 10 FLEURS clips the in-graph mel of the 30 s file differs from the original by at most 0.028 (mean 0.00006, on features in about −1.5 to 1.6), which comes from the fp16 weights of the DFT and filterbank; the fp32 graph differs by at most 0.00003.
3. **Quantization.** int8 dynamic LM weights, an int8 embedding table and fp16 encoder weights: the recipe chosen for [Confucius4-R2T2](https://huggingface.co/mlboydaisuke/Confucius4-R2T2-LiteRT), a fine-tune of this model with the same audio encoder, and checked here on the 155 clips above (every language within 0.6 points of the original model in PyTorch with the same prompt).

The 15 s file (2026-10-13) came out of a speed ladder run on the 30 s file: the audio encoder re-exported with a 15 s window (input `[1, 1500, 160]`, output `[1, 195, 2048]`; same weights, same fp16 recipe) and the language model re-exported with prefill signatures for 128 and 256 tokens (`--prefill_lengths 128,256`; same int8 recipe). Of the other forms tried, int4 (blockwise 32) LM weights failed the correctness check (the model stops after the language tag on half of the 155 clips), a dynamic-int8 encoder with the log-mel kept in float passed it with no gain on the GPU, and a 512-token KV cache passed on the CPU but failed on the GPU for Japanese.

These files are made for the Conversation API. The `omni/asr` runner in the LiteRT-LM repository was not run with them.

Scripts and a one-command reproduction: https://github.com/john-rocky/hf-to-litertlm/tree/main/qwen3_asr17_work.
