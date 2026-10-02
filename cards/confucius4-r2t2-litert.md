---
license: other
license_name: netease-model-use-license-agreement
license_link: https://huggingface.co/mlboydaisuke/Confucius4-R2T2-LiteRT/blob/main/MODEL_LICENSE
base_model: netease-youdao/Confucius4-R2T2
base_model_relation: quantized
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
  - streaming
  - audio
  - qwen3-asr
---

# Confucius4-R2T2 — LiteRT-LM

[netease-youdao/Confucius4-R2T2](https://huggingface.co/netease-youdao/Confucius4-R2T2) converted to the **LiteRT-LM** (`.litertlm`) format for on-device speech recognition with Google's [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) runtime. Audio in, text out, one file.

Confucius4-R2T2 is NetEase Youdao's streaming speech recognition model, built on Qwen3-ASR-1.7B (2.04 B parameters). A 24-layer audio encoder feeds a 28-layer Qwen3 language model, which writes `language <name><asr_text><transcript>`. The model is optimized for Chinese and English and lists 30 languages. Its streaming mode re-reads the audio received so far at every chunk and continues from the text it has already committed.

The file runs on the generic audio path of the released runtime (measured with `litert-lm-api` 0.17.1). The runtime cuts 16 kHz PCM into 10 ms frames and hands the bundled encoder a 30 s window; the encoder computes the log-mel spectrogram itself. No runtime change is needed.

| File | Recipe | Size |
|---|---|---|
| `Confucius4-R2T2.litertlm` | LM int8 dynamic (weights) with an int8 embedding table · audio encoder fp16 weights, fp32 compute, 30 s window | 2.69 GB |

| Section | Content | Size |
|---|---|---|
| `tf_lite_audio_encoder_hw` | log-mel front end (the original feature extractor's STFT, Slaney filterbank and normalization), 24-layer encoder, projector | 637 MB |
| `tf_lite_prefill_decode` | Qwen3 language model, int8, KV cache 1024 tokens | 1.74 GB |
| `tf_lite_embedder` | token embedding table, int8 | 313 MB |
| `HF_Tokenizer_Zlib` | tokenizer | 2.1 MB |
| `LlmMetadataProto` | prompt template, stop tokens, audio settings | 1.4 kB |

## How to send audio

**One message holds one clip of up to 30 s.** The runtime zero-fills the window after the clip, and the encoder turns every window into 390 audio tokens. Encoding and prefill therefore cost the same for a 4 s clip as for a 29 s clip; only decoding grows with the transcript. Clips longer than 30 s were not tested with this file.

**The answer is `language <name><asr_text><transcript>`.** Split it at `<asr_text>`; the scores below use the text after the tag.

**Forced language.** Add a text item `language Chinese<asr_text>` (or another language name) to the same message. The template places it at the start of the answer, as the original code does for a forced language, and the model then writes the transcript only.

**Prompt.** The template renders `<|im_start|>user<|audio_start|>` + audio + `<|audio_end|><|im_end|><|im_start|>assistant\n`. The original repository's prompt adds an empty system turn. With that prompt this file scored Mandarin CER 10.75 % and Japanese CER 9.33 % on the FLEURS set below. Chinese clips came back tagged `English` or `None`, numbers came out as Chinese numerals, and one Mandarin and one Japanese clip were translated into English. The original model in PyTorch also scores worse with that prompt (7.78 % / 7.47 % against 6.24 % / 6.21 %). The template ignores system messages, so the original repository's context prompt is not available in this file.

## Correctness

FLEURS test, 50 clips per language, ground truth `raw_transcription`. Greedy decoding, one conversation per clip.

| Run | English WER | Mandarin CER | Japanese CER |
|---|---|---|---|
| Original model, PyTorch fp32 (transformers 5.14.1), this file's prompt | 4.43 % (51/1150) | 6.24 % (122/1954) | 6.21 % (167/2689) |
| Original model, PyTorch fp32, the original repository's prompt | 4.35 % (50/1150) | 7.78 % (152/1954) | 7.47 % (201/2689) |
| **This file**, LiteRT-LM 0.17.1 (Python), Apple M4 Max, CPU | **4.43 %** (51/1150) | **6.40 %** (125/1954) | **6.21 %** (167/2689) |
| This file, LiteRT-LM 0.17.1 (Python), Apple M4 Max, LM on the GPU (WebGPU over Metal), audio on the CPU | 4.52 % (52/1150) | 6.45 % (126/1954) | 6.06 % (163/2689) |

No clip came back empty. WER counts words after uppercasing and removing punctuation. CER counts characters after NFKC normalization and lowercasing, with all punctuation, separators and whitespace removed.

## Performance

Apple M4 Max, LiteRT-LM 0.17.1 (Python), the 150 FLEURS clips above plus 5 example clips (155 clips, 4–29 s, 1,836 s of audio):

| Backend | Processing time | Real-time factor | Per clip | Peak RSS |
|---|---|---|---|---|
| CPU, 4 threads | 227 s | 0.124 (8.1× faster than real time) | 1.08–2.64 s, median 1.42 s | 4.55 GB ¹ |
| LM on the GPU (WebGPU over Metal), audio encoder on the CPU | 97 s | 0.053 (18.9× faster than real time) | 0.49–1.06 s, median 0.61 s | 2.68 GB |

¹ Peak RSS includes the memory-mapped weights and was read from a 155-clip run with the caches already written. The first load writes about 3.0 GB of CPU weight caches (encoder 1.27 GB, LM 1.73 GB) into `cache_dir`, or next to the model file when none is set; the 155-clip run that wrote them peaked at 6.62 GB.

Galaxy S26 (SM-S942Q), LiteRT-LM command-line tool built from the v0.16.1 release tag, one process per clip, the 20 shortest clips of the set (128 s of audio). During the CPU run the phone's frequency limit fell from 3.63 / 4.74 GHz to 1.79 / 1.75 GHz, and the GPU run started under that limit, so read these as reference values:

| Backend | Same text as the Mac CPU run | Empty | Wall per process, including engine load | Peak private footprint |
|---|---|---|---|---|
| CPU, 4 threads, no weight cache | 15 of 20 | 0 | 10.4–19.6 s | 7.47 GB |
| LM on the GPU (OpenCL), audio encoder on the CPU, 5 example clips | 4 of 5 | 0 | 5.6–9.3 s | 1.93 GB |

On the FLEURS clips among the 20, the S26 CPU run made as many errors as the Mac run (English 5/142 words, Mandarin 8/173 characters); the 5 differing English texts differ in a number (fifty / 50), a name, punctuation, capitalization and one phrase.

## Streaming

The original streaming procedure (`streaming_transcribe` in the original repository's `r2t2/r2t2_asr.py`, default settings) runs through the same API without changes to its logic. Every 2 s the app sends all audio received so far in a new conversation; the clips here are at most 29.3 s, so every call fits one window. After the first two chunks it adds the previous answer, minus its last 5 tokens, as a text item; the model continues from that text. The text that is 5 tokens behind the latest answer is the committed transcript.

Measured on 35 clips (5 example clips and 10 FLEURS clips per language), Apple M4 Max CPU, 211 calls:

| | English WER | Mandarin CER | Japanese CER | Committed text revised | Seconds per call (mean / max) |
|---|---|---|---|---|---|
| This file, this file's prompt | 3.70 % | 9.16 % | 7.43 % | 2 times | 0.99 / 2.34 |
| This file, the original repository's prompt | 3.24 % | 9.16 % | 8.91 % | 0 times | 0.96 / 1.25 |
| Original model, PyTorch fp32, the original repository's prompt | 2.78 % | 10.24 % | 6.60 % | 0 times | 0.79 / 1.69 |

Each call re-encodes a full 30 s window, so on this Mac CPU a call takes about 1 s, inside the 2 s chunk. The scores are for the 30 FLEURS clips in the set; with this file's prompt they equal the offline scores of the same clips.

The committed text is the latest answer minus its last 5 tokens, so it shrinks when a call adds fewer than 5 tokens. That happened twice in 211 calls with this file's prompt, each time on a Japanese clip near its end and by 1–2 characters. The original code's committed text can also be the bare tag (`language`, `language English`) while fewer than 5 tokens follow it; the counts above are for the transcript part.

## Usage

Python (`pip install litert-lm-api`, run on 0.17.1). `transcribe` takes a 16 kHz mono WAV of up to 30 s; `prefix` is text the answer starts with, for example `language Chinese<asr_text>` to force the language, and the return value is the answer after it:

```python
import litert_lm
from litert_lm import Content, Contents, Message
from litert_lm.interfaces import CPU

engine = litert_lm.Engine("Confucius4-R2T2.litertlm", backend=CPU(), audio_backend=CPU())
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

## License

This file is a quantized Derivative Work of Confucius4-R2T2 and is distributed under the **NetEase Youdao Model Use License Agreement**. The agreement is included in this repository in English (`MODEL_LICENSE`) and Chinese (`MODEL_LICENSE_zh`); clause 9 makes the Chinese version prevail.

Statement required by clause 4.1(a):

> Any modifications made to the original model in this Derivative Work are not endorsed, warranted, or guaranteed by the original right-holder of the original model, and the original right-holder disclaims all liability related to this Derivative Work.
>
> 该衍生品对原模型所作的任何改动与原模型原始权利人无关，原始权利人对该衍生品不背书、不担保、不承担责任

Points to check before you use it (the agreement is the authority, not this summary):

- **Clause 2.2.** You need a separate license from NetEase Youdao if your or your affiliates' products had more than 100 million monthly active users in the preceding month, or if annual revenue in the preceding year exceeded the threshold. The English text says RMB 1 billion; the Chinese text, which prevails, says RMB 100 million (1亿人民币). Contact: Youdao Zhiyun Business Team, AIcloud_Business@corp.youdao.com (clause 2.3).
- **Clause 3.4(c).** "You may not Use the NetEase Youdao Confucius4-R2T2 or any Derivative Work to improve any AI model, except for the NetEase Youdao Confucius4-R2T2 itself, its Derivative Works，or non-commercial AI models."
- **Clause 3.4(a)–(b).** Pass the agreement on to anyone you distribute this file to, and keep the copyright notices and a copy of the agreement with every copy.
- **Clause 4.2.** No deployment in high-risk scenarios such as medical diagnosis, autonomous driving, military use, critical-infrastructure control, large-scale biometric surveillance or automated decision-making.

## Conversion

The conversion differs from a default export in three places:

1. **Encoder attention windows.** The original encoder attends within 104-token (8 s) windows. The default litert-torch export for Qwen3-ASR attends within 13-token (1 s) chunks, which on 5 s clips moved the transcripts by 1.6 / 5.0 / 9.0 points (English / Mandarin / Japanese) against the original. This file uses the original windows through a constant block-diagonal attention mask. With it, an fp32 export matched the original model's transcripts on 60 five-second clips (no word or character edits after normalization).
2. **Log-mel inside the encoder.** The runtime's built-in mel front end uses a different filterbank and log scale than the original feature extractor, so the file asks the runtime for raw PCM frames and computes the original log-mel in the graph. On 10 FLEURS clips the in-graph mel of this file differs from the original by at most 0.028 (mean 0.00006, on features in about −1.5 to 1.6).
3. **Quantization chosen on the 155 clips.** int8 dynamic LM weights, an int8 embedding table and fp16 encoder weights. Two other candidates each lost one clip outright: an all-int8 file translated a Mandarin clip into English, and an fp16 LM stopped after the language tag on an English clip.

Scripts and a one-command reproduction: https://github.com/john-rocky/hf-to-litertlm/tree/main/confucius4_r2t2_work.
