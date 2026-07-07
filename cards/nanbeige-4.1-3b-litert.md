---
license: apache-2.0
base_model: Nanbeige/Nanbeige4.1-3B
tags:
  - litert
  - litert-lm
  - litertlm
  - on-device
  - edge
  - nanbeige
  - agentic
  - tool-use
pipeline_tag: text-generation
library_name: litert-lm
---

# Nanbeige4.1-3B — LiteRT-LM (blockwise int4)

[Nanbeige/Nanbeige4.1-3B](https://huggingface.co/Nanbeige/Nanbeige4.1-3B)
converted to the **LiteRT-LM** (`.litertlm`) format for on-device inference with
Google's [LiteRT-LM](https://github.com/google-ai-edge/litert-lm) runtime (the
engine behind the official `litert-community/*` models).

Text-only conversion of the Nanbeige4.1-3B decoder. Nanbeige4.1-3B is a small
general model with native support for **deep-search / agentic tool-use** (500+
rounds of tool invocations), which makes it an interesting on-device agent base.

| | |
|---|---|
| **File** | `model.litertlm` (~__SIZE__ GB) |
| **Quantization** | int4 weights — **blockwise (block 32) + OCTAV**, symmetric; embeddings INT8 |
| **Compute** | integer |
| **Context (KV cache)** | 4096 |
| **Base model** | Nanbeige/Nanbeige4.1-3B (Apache-2.0) |
| **Architecture** | `LlamaForCausalLM` — 32 layers, hidden 2560, GQA (20 Q / 4 KV heads, head_dim 128), SwiGLU FFN 10496, RoPE θ=70,000,000, RMSNorm, vocab 166,144, untied embeddings |
| **Decode speed** | ~__TOKS__ tok/s (Mac, LiteRT-LM, greedy) |

## Usage

Run with the LiteRT-LM runtime:

```bash
# build litert-lm from https://github.com/google-ai-edge/litert-lm, then:
litert_lm_main \
  --model_path model.litertlm \
  --backend gpu \
  --input_prompt "Explain on-device AI in one sentence."
```

The `.litertlm` bundle carries the tokenizer (SentencePiece) and the prompt
template (ChatML: `<|im_start|>role\n … <|im_end|>`, stop token `<|im_end|>`),
so no separate tokenizer files are needed.

## Run on Android

> **Update (July 2026):** [Google AI Edge Gallery](https://github.com/google-ai-edge/gallery) **v1.0.16+** can import litert-lm models **directly from Hugging Face** inside the app (tap **+**) — no computer or `adb` needed. The manual steps below are only required on older builds or for sideloading a local file.

The easiest way to try this model on a phone is the official
**[Google AI Edge Gallery](https://github.com/google-ai-edge/gallery)** app — it
runs `.litertlm` models fully on-device and can import your own:

1. Install a **recent** Gallery (package `com.google.ai.edge.gallery`, APK from the repo's
   [releases](https://github.com/google-ai-edge/gallery/releases) — 1.0.15+ supports
   `.litertlm`). Older 1.0.x builds (package `com.google.aiedge.gallery`) only accept the
   legacy MediaPipe `.task` format and reject `.litertlm`.
2. Download `model.litertlm` from this repo and push it to the device:
   ```bash
   adb push model.litertlm /sdcard/Download/
   ```
3. In the app, tap the **+** button (bottom-right), pick the file, and choose a backend.
4. Chat. Nothing else to configure — the `.litertlm` bundle already carries the
   tokenizer and prompt template, so the model uses its native ChatML chat format
   automatically.

> **Device memory note.** A ~2 GB int4 3B model needs a **12 GB+** phone for the
> GPU (ML Drift) backend, which builds a device-laid-out weight cache ≈ the model
> size next to it. On an **8 GB** phone (e.g. Pixel 8a) use the **CPU** backend and
> free RAM first; the GPU option is greyed out by the RAM ceiling, not a model bug.
> KV cache is not the constraint (~0.4 GB @ 4096 tokens). For a smooth GPU demo on a
> small phone prefer a 1–2 B model.

To embed the model in **your own** Android app instead, use the LiteRT-LM Kotlin API
(Gradle artifact `com.google.ai.edge.litertlm:litertlm-android`,
[getting started](https://github.com/google-ai-edge/LiteRT-LM/blob/main/docs/api/kotlin/getting_started.md)).

## Quality

Verified on the Mac with the LiteRT-LM runtime (`swift-litert-lm`): the int4 build
answers a fixed factual/reasoning check set correctly and terminates cleanly at
`<|im_end|>` (no degenerate looping or special-token spam). __GSM8K_LINE__

## Conversion

Converted with [`litert-torch`](https://github.com/google-ai-edge/litert)'s
HuggingFace-export path. Nanbeige4.1-3B is a standard `LlamaForCausalLM`
architecture (dense GQA, SwiGLU, RoPE, RMSNorm — no qk-norm / sliding-window /
partial-rope / NoPE), so it rides the existing converter and runtime with **no
custom code**. It ships a real SentencePiece `tokenizer.model`, so the bundle
embeds an `SP_Tokenizer` directly.

The int4 recipe is **blockwise (block 32) + OCTAV** optimal-clipping (data-free),
with the vocabulary embedding / LM head kept at **INT8**. Blockwise (not
channelwise) int4 is what preserves reasoning accuracy on small models; OCTAV
clipping reduces int4 PTQ degradation without any calibration data.

## Reproduce (official tools only)

Built with **stock `litert-torch`** — no graph patches. The only non-default
choice is the int4 recipe (the tool's default named int4 is *channelwise*, which
degrades small models):

```python
from litert_torch.generative.export_hf.export import export
export(
    model="Nanbeige/Nanbeige4.1-3B",
    output_dir="out",
    quantization_recipe="nanbeige_int4_block32_octav.json",  # included in this repo
    cache_length=4096,
    use_jinja_template=False,   # embed the structured ChatML template the runtime renders
    trust_remote_code=True,
)
```

`nanbeige_int4_block32_octav.json` is included in this repo. (If the export errors
with a missing `ai_edge_quantizer/recipes/` directory, create it empty — a
packaging gap in some releases that trips the `.json`-recipe path.)

## License

Apache-2.0, inherited from the base model
[Nanbeige/Nanbeige4.1-3B](https://huggingface.co/Nanbeige/Nanbeige4.1-3B).
