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

The file runs on the generic audio path of the released runtime (measured with `litert-lm-api` 0.17.1). The runtime cuts 16 kHz PCM into 10 ms frames and hands the bundled encoder a 30 s window; the encoder computes the log-mel spectrogram itself. No runtime change is needed.

| File | Recipe | Size |
|---|---|---|
| `Qwen3-ASR-1.7B.litertlm` | LM int8 dynamic (weights) with an int8 embedding table · audio encoder fp16 weights, fp32 compute, 30 s window | 2.69 GB |

| Section | Content | Size |
|---|---|---|
| `tf_lite_audio_encoder_hw` | log-mel front end (the original feature extractor's STFT, Slaney filterbank and normalization), 24-layer encoder, projector | 637 MB |
| `tf_lite_prefill_decode` | Qwen3 language model, int8, KV cache 1024 tokens | 1.74 GB |
| `tf_lite_embedder` | token embedding table, int8 | 313 MB |
| `HF_Tokenizer_Zlib` | tokenizer | 2.1 MB |
| `LlmMetadataProto` | prompt template, stop tokens, audio settings | 0.9 kB |

## How to send audio

**One message holds one clip of up to 30 s.** The runtime zero-fills the window after the clip, and the encoder turns every window into 390 audio tokens. Encoding and prefill therefore cost the same for a 4 s clip as for a 29 s clip; only decoding grows with the transcript. Clips longer than 30 s were not tested with this file.

**The answer is `language <name><asr_text><transcript>`.** Split it at `<asr_text>`; the scores below use the text after the tag. On the FLEURS clips below every Mandarin and Japanese clip was tagged correctly; one English clip (two on the GPU run) came back as `language None` followed by the English transcript.

**Forced language.** Add a text item `language Chinese<asr_text>` (or another language name) to the same message. The template places it at the start of the answer, as the `qwen-asr` package does for a forced language, and the model then writes the transcript only (11 of 11 Mandarin and Japanese clips tried).

**Prompt.** The template renders `<|im_start|>user<|audio_start|>` + audio + `<|audio_end|><|im_end|><|im_start|>assistant\n`, the prompt of litert-torch's Qwen3-ASR export. The checkpoint's own chat template adds an empty system turn and newlines. With that template this file scored Mandarin CER 6.65 % and Japanese CER 6.06 % on the set below, and 8 of 50 Mandarin clips came back tagged `English` (with a Chinese transcript); the original model in PyTorch shows the same pattern (6.86 % / 5.50 %, 9 of 50). The template ignores system messages.

## Correctness

FLEURS test, 50 clips per language, ground truth `raw_transcription`. Greedy decoding, one conversation per clip.

| Run | English WER | Mandarin CER | Japanese CER |
|---|---|---|---|
| Original model, PyTorch fp32 (transformers 5.14.1), this file's prompt | 4.35 % (50/1150) | 6.04 % (118/1954) | 5.24 % (141/2689) |
| Original model, PyTorch fp32, the checkpoint's chat template | 4.35 % (50/1150) | 6.86 % (134/1954) | 5.50 % (148/2689) |
| **This file**, LiteRT-LM 0.17.1 (Python), Apple M4 Max, CPU | **4.43 %** (51/1150) | **6.19 %** (121/1954) | **5.73 %** (154/2689) |
| This file, LiteRT-LM 0.17.1 (Python), Apple M4 Max, LM on the GPU (WebGPU over Metal), audio on the CPU | 4.52 % (52/1150) | 6.19 % (121/1954) | 5.32 % (143/2689) |

No clip came back empty. WER counts words after uppercasing and removing punctuation. CER counts characters after NFKC normalization and lowercasing, with all punctuation, separators and whitespace removed.

## Performance

Apple M4 Max, LiteRT-LM 0.17.1 (Python), the 150 FLEURS clips above plus 5 example clips (155 clips, 4–29 s, 1,836 s of audio). Other jobs were running on the machine (load average 4.5–8.5), so read the times as an upper bound:

| Backend | Processing time | Real-time factor | Per clip | Peak RSS |
|---|---|---|---|---|
| CPU, 4 threads | 272 s | 0.148 (6.7× faster than real time) | 1.04–3.17 s, median 1.70 s | 4.54 GB ¹ |
| LM on the GPU (WebGPU over Metal), audio encoder on the CPU | 105 s | 0.057 (17.5× faster than real time) | 0.53–1.06 s, median 0.66 s | 4.79 GB ² |

¹ With the CPU weight caches already written. Peak RSS includes the memory-mapped weights. The first load writes about 3.0 GB of CPU weight caches (encoder 1.27 GB, LM 1.73 GB) into `cache_dir`, or next to the model file when none is set; a first load that wrote them peaked at 6.62 GB.

² This run was the first GPU load and wrote the GPU caches (1.72 GB) during the run.

Galaxy S26 (SM-S942Q), CPU, 4 threads, LiteRT-LM command-line tool built from the v0.16.1 release tag with `--benchmark`, one process per clip: an English, a Mandarin and a Japanese FLEURS clip, the English one twice. The prompt is 398 tokens (390 audio + 8 text). The CPU frequency limit was at its maximum before every run:

| Run | Prefill | Decode | Time to first token | Peak private footprint |
|---|---|---|---|---|
| First run, writes the weight caches | 329 tokens/s | 22.8 tokens/s | 1.25 s | 3.47 GB |
| Next 3 runs, caches loaded | 538–547 tokens/s | 25.0–25.6 tokens/s | 0.77–0.78 s | 4.23 GB |

All four transcripts equal the Mac CPU run.

## Usage

Python (`pip install litert-lm-api`, run on 0.17.1). `transcribe` takes a 16 kHz mono WAV of up to 30 s; `prefix` is text the answer starts with, for example `language Chinese<asr_text>` to force the language, and the return value is the answer after it:

```python
import litert_lm
from litert_lm import Content, Contents, Message
from litert_lm.interfaces import CPU

engine = litert_lm.Engine("Qwen3-ASR-1.7B.litertlm", backend=CPU(), audio_backend=CPU())
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

For the Chinese example clip `raw` is `language Chinese<asr_text>开放时间：早上九点至下午五点。` and the script prints the text after the tag. `backend=GPU()` (from `litert_lm.interfaces`) puts the language model on the GPU. Keep `audio_backend=CPU()`: the audio encoder does not run on the GPU delegate (`create_conversation` fails). The Kotlin and Swift Conversation APIs take the same calls; they were not run for this file.

On Android the S26 rows above came from the command-line tool: `litert_lm_advanced_main --backend=cpu --audio_backend=cpu --sampler_backend=cpu --num_cpu_threads=4 --model_path=Qwen3-ASR-1.7B.litertlm --cache_dir=cache --max_num_tokens=1024 --max_output_tokens=256 --benchmark --input_prompt='[audio:/path/clip.wav]'`.

## License

Apache 2.0, the license of the original model.

## Conversion

The conversion differs from a default export in three places:

1. **Encoder attention windows.** The original encoder attends within 104-token (8 s) windows. The default litert-torch export for Qwen3-ASR attends within 13-token (1 s) chunks. This file uses the original windows through a constant block-diagonal attention mask; the exported encoder then matches the original encoder's output exactly in fp32 (maximum difference 0.0 on 5 s and 30 s inputs; with 13-token chunks the cosine similarity is 0.81–0.82 on 5 s and 0.76 on 30 s).
2. **Log-mel inside the encoder.** The runtime's built-in mel front end uses a different filterbank and log scale than the original feature extractor, so the file asks the runtime for raw PCM frames and computes the original log-mel in the graph. On 10 FLEURS clips the in-graph mel of this file differs from the original by at most 0.028 (mean 0.00006, on features in about −1.5 to 1.6), which comes from the fp16 weights of the encoder section.
3. **Quantization.** int8 dynamic LM weights, an int8 embedding table and fp16 encoder weights: the recipe chosen for [Confucius4-R2T2](https://huggingface.co/mlboydaisuke/Confucius4-R2T2-LiteRT), a fine-tune of this model with the same audio encoder, and checked here on the 155 clips above (every language within 0.5 points of the original model in PyTorch with the same prompt).

This file is made for the Conversation API. The `omni/asr` runner in the LiteRT-LM repository was not run with it.

Scripts and a one-command reproduction: https://github.com/john-rocky/hf-to-litertlm/tree/main/qwen3_asr17_work.
