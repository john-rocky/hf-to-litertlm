# hf-to-litertlm

Convert open-weight Hugging Face LLMs and VLMs into `.litertlm` bundles for Google's LiteRT-LM
runtime (Android, iOS, macOS, Windows, Linux): 62 published conversions below, each with
measured speeds and a one-command reproduction.

**Want a model converted?** [Open a model request](https://github.com/john-rocky/hf-to-litertlm/issues/new?template=model-request.yml)
with its Hugging Face link. That is the whole ask. The bundle, the recipe, and the measured
numbers come back on the issue, or the reason it could not be converted.

**Run a published model on Android:** start with
[LFM2.5-1.2B-Instruct](https://huggingface.co/litert-community/LFM2.5-1.2B-Instruct) and the
[hfmodels 0.1.1 install and first-reply guide](https://github.com/john-rocky/hfmodels-android#add-it).
The guide fixes the model revision and GPU profile, includes the imports and cleanup,
and shows the expected **42** reply and where to report a failure. Its tested target is
a **Pixel 8a with 8 GB RAM, Android 16**; installation needs **API 31+, compileSdk 36**
and a 736 MB download plus cache space. [Other SDK entry points](readers/README.md#sdk-entry-points)
are listed separately with their verification scope.

## Converted models

Decode speed is tokens per second, read from each repo's `litertlm_manifest.json`: the fastest
verified backend per device class, compiled-model cache off, prefill and conditions in the
manifest. A range is the spread across runs. A dash means no measured row for that class yet.

<!-- models-table:start -->
| Model | Params | Task | Phone: decode tok/s | Mac: decode tok/s | Recipe |
|---|---:|---|---|---|---|
| [LFM2.5-230M](https://huggingface.co/litert-community/LFM2.5-230M) | 0.23B | chat | iPhone 17 Pro GPU 161.7 | M4 Max GPU 561.2 | [recipe](REPRODUCE.md#lfm25-230m-the-smallest-decoder--a-template-the-runtime-cannot-parse-and-a-shape-that-kills-the-gpu-shader-compile) |
| [granite-4.0-h-350m](https://huggingface.co/litert-community/granite-4.0-h-350m) | 0.35B | chat | Galaxy S26 CPU 97.6 | — | [recipe](REPRODUCE.md#granite-40-h-350m-fp16--int8--and-the-start_token-lesson) |
| [Falcon-H1-0.5B-Instruct](https://huggingface.co/litert-community/Falcon-H1-0.5B-Instruct) | 0.5B | chat | Galaxy S26 CPU 32.1–40.5 | M4 Max GPU 127.5 | [recipe](REPRODUCE.md#falcon-h1-attention--mamba2-in-parallel-every-layer--first-fully-hybrid-family-in-litert-form) |
| [sarashina2.2-0.5b-instruct-v0.1](https://huggingface.co/litert-community/sarashina2.2-0.5b-instruct-v0.1) | 0.5B | chat, Japanese | Galaxy S26 GPU 36.2–36.5 | M4 Max GPU 197.8 | [recipe](REPRODUCE.md#sarashina22-05b--1b-instruct-sb-intuitions-japanese--a-sentencepiece-vocab-whose-chat-specials-are-control-pieces-and-a-bos-the-model-never-saw) |
| [granite-4.0-h-1b](https://huggingface.co/litert-community/granite-4.0-h-1b) | 1B | chat | Galaxy S26 GPU 24.9 | M4 Max GPU 134.7 | [recipe](REPRODUCE.md#granite-40-h-mamba2--attention-hybrid--first-mamba2-hybrid-on-the-released-runtime) |
| [OLMo-2-1B-Instruct](https://huggingface.co/litert-community/OLMo-2-1B-Instruct) | 1B | chat | Galaxy S26 GPU 23.1 | M4 Max GPU 150.1 | [card](cards/olmo2-1b-litert.md) |
| [sarashina2.2-1b-instruct-v0.1](https://huggingface.co/litert-community/sarashina2.2-1b-instruct-v0.1) | 1B | chat, Japanese | Galaxy S26 GPU 27.4–27.5 | M4 Max GPU 159.4 | [recipe](REPRODUCE.md#sarashina22-05b--1b-instruct-sb-intuitions-japanese--a-sentencepiece-vocab-whose-chat-specials-are-control-pieces-and-a-bos-the-model-never-saw) |
| [LFM2.5-1.2B-Instruct](https://huggingface.co/litert-community/LFM2.5-1.2B-Instruct) | 1.2B | chat | Galaxy S26 GPU 54.5 | M4 Max GPU 318.3 | [recipe](REPRODUCE.md#lfm25-family-hybrid-shortconv--attention) |
| [LFM2.5-1.2B-JP](https://huggingface.co/litert-community/LFM2.5-1.2B-JP) | 1.2B | chat, Japanese | iPhone 17 Pro GPU 70.0 | M4 Max GPU 316.0 | [recipe](REPRODUCE.md#lfm25-family-hybrid-shortconv--attention) |
| [Zamba2-1.2B-instruct](https://huggingface.co/litert-community/Zamba2-1.2B-instruct) | 1.2B | chat | Galaxy S26 GPU 11.6 | M4 Max GPU 74.0 | [recipe](REPRODUCE.md#zamba2-mamba2-backbone--a-shared-lora-specialized-transformer-block--and-the-metaspace-tokenizer-trap) |
| [Falcon-H1-1.5B-Deep-Instruct](https://huggingface.co/litert-community/Falcon-H1-1.5B-Deep-Instruct) | 1.5B | chat | Galaxy S26 CPU 11.3–12.0 | M4 Max GPU 51.7 | [recipe](REPRODUCE.md#falcon-h1-attention--mamba2-in-parallel-every-layer--first-fully-hybrid-family-in-litert-form) |
| [Falcon-H1-1.5B-Instruct](https://huggingface.co/litert-community/Falcon-H1-1.5B-Instruct) | 1.5B | chat | Galaxy S26 GPU 20.8 | M4 Max GPU 102.9 | [recipe](REPRODUCE.md#falcon-h1-attention--mamba2-in-parallel-every-layer--first-fully-hybrid-family-in-litert-form) |
| [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/litert-community/Qwen2.5-Coder-1.5B-Instruct) | 1.54B | code | Galaxy S26 CPU 27.0–29.7 | M4 Max GPU 137.8 | [recipe](REPRODUCE.md#qwen25-coder-15b-instruct--a-15b-code-model-at-112-gb-and-why-the-size-is-the-recipe) |
| [Falcon-H1-3B-Instruct](https://huggingface.co/litert-community/Falcon-H1-3B-Instruct) | 3B | chat | Galaxy S26 GPU 11.6 | M4 Max GPU 65.3 | [recipe](REPRODUCE.md#falcon-h1-attention--mamba2-in-parallel-every-layer--first-fully-hybrid-family-in-litert-form) |
| [Ministral-3-3B-Instruct-2512](https://huggingface.co/litert-community/Ministral-3-3B-Instruct-2512) | 3B | chat | iPhone 17 Pro GPU 14.0–18.0 | M4 Max GPU 95.4 | [card](cards/ministral3-3b-litert.md) |
| [granite-4.1-3b](https://huggingface.co/litert-community/granite-4.1-3b) | 3.4B | chat, tool calling | Galaxy S26 GPU 16.1 | M4 Max GPU 86.3 | [recipe](REPRODUCE.md#granite-41-3b-dense--and-the-bos-a-converted-bundle-must-not-prepend) |
| [FastContext-1.0-4B-SFT](https://huggingface.co/litert-community/FastContext-1.0-4B-SFT) | 4B | chat | iPhone 17 Pro GPU 14.0 | M4 Max GPU 73.8 | [card](cards/fastcontext-4b-litert.md) |
| [Qwen3.5-4B](https://huggingface.co/litert-community/Qwen3.5-4B) | 4B | chat | iPhone 17 Pro GPU 11.4 | M4 Max GPU 68.5 | [recipe](REPRODUCE.md#qwen35-gateddeltanet--attention-hybrid--first-qwen35-in-litert-form) |
| [Falcon-H1-Tiny-R-0.6B](https://huggingface.co/litert-community/Falcon-H1-Tiny-R-0.6B) | 0.62B | reasoning | iPhone 17 Pro CPU 29.4 | M4 Max GPU 97.8 | [recipe](REPRODUCE.md#2026-09-01--falcon-h1-tiny-r-06b-the-familys-first-reasoning-ship-and-the-size-where-two-family-assumptions-break) |
| [LFM2.5-1.2B-Thinking](https://huggingface.co/litert-community/LFM2.5-1.2B-Thinking) | 1.2B | reasoning | iPhone 17 Pro GPU 69.7 | M4 Max GPU 317.9 | [recipe](REPRODUCE.md#lfm25-family-hybrid-shortconv--attention) |
| [Spark-X2.5-1.7B](https://huggingface.co/litert-community/Spark-X2.5-1.7B) | 1.71B | reasoning | Galaxy S26 GPU 17.4–17.8 | M4 Max GPU 103.6 | [recipe](REPRODUCE.md#spark-x25-17b--4b-sparkllm-team-thinking--a-remote-code-architecture-exported-by-patching-the-vendor-file-not-re-implementing-it) |
| [MiniCPM5-2B](https://huggingface.co/mlboydaisuke/MiniCPM5-2B-LiteRT) | 2.52B | chat, hybrid thinking | Galaxy S26 GPU 16.1–18.6 | M4 Max GPU 92.8 | [card](cards/minicpm5-2b-litert.md) |
| [LFM2.5-2.6B](https://huggingface.co/litert-community/LFM2.5-2.6B) | 2.6B | reasoning | Galaxy S26 GPU 20.9 | M4 Max GPU 161.6 | [recipe](REPRODUCE.md#lfm25-26b-the-thinking-flagship) |
| [Ministral-3-3B-Reasoning-2512](https://huggingface.co/litert-community/Ministral-3-3B-Reasoning-2512) | 3B | reasoning | Galaxy S26 GPU 13.8 | M4 Max GPU 95.8 | [card](cards/ministral3-3b-reasoning-litert.md) |
| [Nanbeige4.1-3B](https://huggingface.co/litert-community/Nanbeige4.1-3B) | 3B | reasoning | Galaxy S26 GPU 10.7 | M4 Max GPU 90.1 | [card](cards/nanbeige4.1-3b-litert.md) |
| [Nanbeige4.2-3B](https://huggingface.co/litert-community/Nanbeige4.2-3B) | 3B | reasoning | Galaxy S26 CPU 4.1 | M4 Max GPU 39.7 | [card](cards/nanbeige4.2-3b-litert.md) |
| [SmolLM3-3B](https://huggingface.co/litert-community/SmolLM3-3B) | 3B | chat, optional thinking | iPhone 17 Pro GPU 22.5 | M4 Max GPU 93.2 | [card](cards/smollm3-3b-litert.md) |
| [VibeThinker-3B](https://huggingface.co/litert-community/VibeThinker-3B) | 3B | math reasoning | Galaxy S26 GPU 14.6 | M4 Max GPU 94.1 | [card](cards/vibethinker-3b-litert.md) |
| [Mordant-3B-Think](https://huggingface.co/mlboydaisuke/Mordant-3B-Think-LiteRT) | 3.4B | image-prompt writing, thinking | Galaxy S26 GPU 9.6–10.1 | M4 Max GPU 71.8 | [recipe](REPRODUCE.md#granite-41-finetune-intake--the-bos-guard-goes-generic-and-the-family-fact-meets-its-first-exception) |
| [granite-4.2-3b](https://huggingface.co/litert-community/granite-4.2-3b) | 3.66B | reasoning | Galaxy S26 GPU 12.3–15.6 | M4 Max GPU 85.5 | [card](cards/granite-4.2-3b-litert.md) |
| [Phi-4-mini-reasoning](https://huggingface.co/litert-community/Phi-4-mini-reasoning) | 3.8B | math reasoning | Galaxy S26 GPU 11.8 | M4 Max GPU 82.8 | [card](cards/phi4-mini-reasoning-litert.md) |
| [Nemotron-3-Nano-4B](https://huggingface.co/litert-community/Nemotron-3-Nano-4B) | 3.97B | reasoning | Galaxy S26 CPU 11.8–12.9 | M4 Max GPU 83.3 | [recipe](REPRODUCE.md#nemotron-h-mamba2--mlp--attention-three-layer-kinds--and-the-registry-trap) |
| [Jan-nano](https://huggingface.co/litert-community/Jan-nano) | 4B | tool-use agent (MCP) | iPhone 17 Pro GPU 14.0 | M4 Max GPU 69.0 | [card](cards/jan-nano-litert.md) |
| [Polaris-4B-Preview](https://huggingface.co/litert-community/Polaris-4B-Preview) | 4B | reasoning | Galaxy S26 GPU 9.2 | M4 Max GPU 69.1 | [card](cards/polaris-4b-litert.md) |
| [Qwen3-4B-Thinking-2507](https://huggingface.co/litert-community/Qwen3-4B-Thinking-2507) | 4B | reasoning | Galaxy S26 GPU 15.3 | M4 Max GPU 68.5 | [card](cards/qwen3-4b-thinking-litert.md) |
| [Spark-X2.5-4B](https://huggingface.co/litert-community/Spark-X2.5-4B) | 4.11B | reasoning | Galaxy S26 CPU 5.5–5.6 | M4 Max GPU 53.6 | [recipe](REPRODUCE.md#spark-x25-17b--4b-sparkllm-team-thinking--a-remote-code-architecture-exported-by-patching-the-vendor-file-not-re-implementing-it) |
| [DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/litert-community/DeepSeek-R1-Distill-Qwen-7B) | 7B | reasoning | Galaxy S26 GPU 10.0 | M4 Max GPU 65.9 | [card](cards/r1-distill-qwen-7b-litert.md) |
| [granite-docling-258M](https://huggingface.co/litert-community/granite-docling-258M) | 0.26B | document to DocTags | Galaxy S26 CPU 28.6–33.5 | M4 Max CPU 64.0 | [card](cards/granite-docling-258m-litert.md) |
| [LFM2.5-VL-450M](https://huggingface.co/litert-community/LFM2.5-VL-450M) | 0.45B | chat + image | Galaxy S26 GPU 103.1 | M4 Max GPU 360.0 | [recipe](REPRODUCE.md#lfm25-vl-3b--16b--450m--lfm2-hybrid-text--siglip2-vision-native-runtime-image-support) |
| [LLaVA-OneVision-0.5B](https://huggingface.co/litert-community/LLaVA-OneVision-0.5B) | 0.5B | chat + image | Galaxy S26 GPU 83.9 | M4 Max GPU 239.3 | [card](cards/llava-onevision-0.5b-litert.md) |
| [SmolVLM2-500M](https://huggingface.co/litert-community/SmolVLM2-500M) | 0.5B | chat + image | Galaxy S26 GPU 76.3 | M4 Max CPU 63.9 | [card](cards/smolvlm2-500m-litert.md) |
| [Qwen3.5-0.8B](https://huggingface.co/litert-community/Qwen3.5-0.8B) | 0.8B | chat; separate image file | iPhone 17 Pro GPU 65.8 | M4 Max GPU 161.8 | [recipe](REPRODUCE.md#qwen35-gateddeltanet--attention-hybrid--first-qwen35-in-litert-form) |
| [OvisOCR2](https://huggingface.co/mlboydaisuke/OvisOCR2-LiteRT) | 0.85B | document OCR | iPhone 17 Pro GPU 48.5 | M4 Max GPU 140.9 | [recipe](REPRODUCE.md#ovisocr2--an-ocr-finetune-rides-the-08b-vision-rail-unchanged) |
| [PaddleOCR-VL-1.6](https://huggingface.co/litert-community/PaddleOCR-VL-1.6) | 0.9B | OCR, 109 languages | — | M4 Max GPU 208.3 | [card](cards/paddleocr-vl-1.6-litert.md) |
| [InternVL3-1B](https://huggingface.co/litert-community/InternVL3-1B) | 1B | chat + image | Galaxy S26 GPU 84.4 | M4 Max CPU 94.0 | [card](cards/internvl3-1b-litert.md) |
| [InternVL3_5-1B](https://huggingface.co/litert-community/InternVL3_5-1B) | 1B | chat + image | Galaxy S26 GPU 42.9 | M4 Max GPU 176.2 | [card](cards/internvl3_5-1b-litert.md) |
| [LFM2.5-VL-1.6B](https://huggingface.co/litert-community/LFM2.5-VL-1.6B) | 1.6B | chat + image | Galaxy S26 GPU 54.2 | M4 Max GPU 275.3 | [recipe](REPRODUCE.md#lfm25-vl-3b--16b--450m--lfm2-hybrid-text--siglip2-vision-native-runtime-image-support) |
| [InternVL3-2B](https://huggingface.co/litert-community/InternVL3-2B) | 2B | chat + image | Galaxy S26 GPU 40.4 | M4 Max CPU 50.0 | [card](cards/internvl3-2b-litert.md) |
| [InternVL3_5-2B](https://huggingface.co/litert-community/InternVL3_5-2B) | 2B | chat + image | Galaxy S26 GPU 21.6 | M4 Max GPU 137.2 | [card](cards/internvl3_5-2b-litert.md) |
| [Ovis2.5-2B](https://huggingface.co/litert-community/Ovis2.5-2B) | 2B | chat + image | Galaxy S26 GPU 28.3 | M4 Max GPU 142.6 | [card](cards/ovis2_5-2b-litert.md) |
| [Qwen2-VL-2B](https://huggingface.co/litert-community/Qwen2-VL-2B) | 2B | chat + image | Galaxy S26 GPU 36.7 | M4 Max GPU 139.1 | [card](cards/qwen2-vl-2b-litert.md) |
| [SmolVLM2-2.2B](https://huggingface.co/litert-community/SmolVLM2-2.2B) | 2.2B | chat + image | Galaxy S26 GPU 21.4 | M4 Max GPU 134.7 | [card](cards/smolvlm2-2.2b-litert.md) |
| [Qwen3.5-2B](https://huggingface.co/litert-community/Qwen3.5-2B) | 2.27B | chat; separate image file | iPhone 17 Pro GPU 33.9 | M4 Max GPU 114.3 | [recipe](REPRODUCE.md#qwen35-gateddeltanet--attention-hybrid--first-qwen35-in-litert-form) |
| [North-Micro-Vision-Instruct](https://huggingface.co/litert-community/North-Micro-Vision-Instruct) | 2.48B | chat + image, 11 languages | Galaxy S26 GPU 13.3 | M4 Max GPU 80.6 | [card](cards/north-micro-vision-instruct-litert.md) |
| [LFM2.5-VL-3B](https://huggingface.co/litert-community/LFM2.5-VL-3B) | 3B | chat + image | Galaxy S26 GPU 27.0 | M4 Max GPU 143.2 | [recipe](REPRODUCE.md#lfm25-vl-3b--16b--450m--lfm2-hybrid-text--siglip2-vision-native-runtime-image-support) |
| [InternVL3_5-4B](https://huggingface.co/litert-community/InternVL3_5-4B) | 4B | chat + image | Galaxy S26 GPU 13.3 | M4 Max GPU 86.5 | [card](cards/internvl3_5-4b-litert.md) |
| [Mage-VL](https://huggingface.co/litert-community/Mage-VL) | 4.7B | chat + image | Galaxy S26 GPU 17.1 | M4 Max GPU 80.0 | [card](cards/magevl-litert.md) |
| [Tashkeel-350M-v2](https://huggingface.co/mlboydaisuke/Tashkeel-350M-v2-LiteRT) | 0.34B | Arabic diacritization | Galaxy S26 CPU 67.2–69.0 | M4 Max CPU 97.2 | [recipe](REPRODUCE.md#granite-40-h-finetune-intake-tashkeel-350m-v2--the-recipe-rides-derivatives-unchanged) |
| [S1-mini](https://huggingface.co/mlboydaisuke/S1-mini-LiteRT) | 0.6B | ASR transcript normalization | iPhone 17 Pro GPU 32.0 | M4 Max GPU 144.6 | [card](cards/s1-mini-litert.md) |
| [Hy-MT2-1.8B](https://huggingface.co/litert-community/Hy-MT2-1.8B) | 2.04B | translation, 33 languages | Galaxy S26 GPU 20.4–20.8 | M4 Max GPU 105.8 | [recipe](REPRODUCE.md#2026-08-27--hy-mt2-18b-intake-one-config-bake-closes-the-sweeps-real-gap-and-the-engines-start_token-prepend-gets-proven) |
| [VibeVoice-ASR-BitNet](https://huggingface.co/litert-community/VibeVoice-ASR-BitNet) | 2.2B | speech to text | Galaxy S26 GPU 36.1 | M4 Max GPU 138.6 | [card](cards/vibevoice-asr-bitnet-litert.md) |
| [Shieldstral-1.0-3B](https://huggingface.co/litert-community/Shieldstral-1.0-3B) | 3B | safety classifier, text + image | Galaxy S26 GPU 10.8 | — | [recipe](REPRODUCE.md#shieldstral-10-3b-a-single-token-safety-classifier-not-a-chat-model) |
<!-- models-table:end -->

Conversions published without a manifest (personal-namespace mirrors, desktop-only files) and
the 11 non-chat conversions (encoders, embeddings, TTS, image generation) are in
[REPRODUCE.md](REPRODUCE.md).

## One command

```bash
pip install litert-torch ai-edge-quantizer "transformers==5.14.*" huggingface_hub litert-lm
python scripts/convert.py <org>/<model>                    # -> out/<model>/ (bundle + convert_report.json)
litert-lm run out/<model>/*.litertlm --prompt "Hello"      # same bundle runs on a phone: see below
```

`convert.py` refuses, with a JSON reason, what it cannot convert honestly (gated, remote-code,
pre-quantized repos) and gates every bundle before calling it done. To rebuild a published model
instead: `bash scripts/reproduce_llm.sh <key>` or `bash scripts/reproduce_vlm.sh <key>`.

## What lives here

1. **A finetune converter.** `python scripts/convert.py <org>/<model>`: one command from Hub
   id to a gated bundle. It covers finetunes of Qwen3.5, LFM2.5, MiniCPM5, granite-4.0-h,
   Falcon-H1, Nemotron-H/Nemotron-3-Nano, and every dense architecture the stock exporter
   handles, about **2,670 tagged derivatives** on the Hub as of 2026-08-26. LoRA/PEFT repos
   merge automatically. Broken models are refused with a machine-readable reason.
2. **One-command reproductions** of every model in the table, with the full recipe record in
   [REPRODUCE.md](REPRODUCE.md).
3. **A deployment manifest**, `litertlm_manifest.json`, that every published repo ships, with
   reference readers and a Google Play packer ([below](#deployment-manifests)).

## Setup

```bash
pip install litert-torch ai-edge-quantizer "transformers==5.14.*" huggingface_hub litert-lm
export PY=python    # scripts default to ~/venvs/ltconv040dev/bin/python; override with PY
```

One family needs a different stack: **Qwen3.5** exports only on litert-torch *main*
(`pip install 'litert-torch @ git+https://github.com/google-ai-edge/litert-torch.git'` in a
fresh venv — the released 0.9.4 ships the exportables but its output is degenerate).
`convert.py` refuses with the exact install command when the installed toolchain can't
convert the model honestly, so you can start without reading further.

## Convert a finetune

```bash
python scripts/convert.py <org>/<model>              # -> out/<model>/ bundle + convert_report.json
python scripts/convert.py <org>/<model> --int4       # proven int4 recipe (blockwise-32 OCTAV)
python scripts/convert.py <org>/<model> --gate-script my_gate.py   # task-specific models
```

One run does five things:

- **Entry gate.** Gated, remote-code, and pre-quantized repos — and pre-port Zamba2
  serializations no modern stack can load — are refused with a structured JSON reason
  before anything downloads. A repo that declares `auto_map` for a model_type transformers
  now registers natively is *not* remote-code: the pinned library implementation loads and
  the repo's Python is never imported, so it converts (measured on Nemotron-3-Nano-4B).
- **Adapter merge.** A LoRA/PEFT repo is merged into its base first (subprocess-isolated);
  the adapter's own tokenizer, chat template, and generation config win over the base's.
- **Export.** Stock litert-torch defaults for dense models; pinned family recipes for the
  architectures no released exporter converts (table below). The derivative's own chat
  template is embedded verbatim. ≥3B models also get the reduced 7-signature prefill ladder,
  on both the stock and the family-recipe path.
- **Post-export guards.** A missing turn-end stop token is added; a spurious start token is
  dropped when the tokenizer says `add_bos_token: False` (or bos == eos) and the template
  never renders a leading BOS; the ExecutorMetadata section is retrofitted where the
  exporter omits it.
- **Exit gate.** `verify_quality.py` (8 questions, bar 6/8, think-aware budget) — or your
  `--gate-script` for models the generic gate cannot certify (an Arabic diacritizer answers
  no trivia). Exit 0 = converted and gated, 1 = converted but gate failed, 2 = refused.

Routing is automatic, by `config.json` model_type. What the converter accepts, with the
Hub derivative counts behind the coverage claim (recounted 2026-08-26 via `base_model`
tags over the bases named in each row; mirrors included):

| base family | bases | Hub finetunes + adapters | toolchain | path |
|---|---|---:|---|---|
| any dense arch the stock exporter handles | llama 3.x, qwen 2/2.5/3, smollm3, olmo2, phi, ministral, … | open-ended | default stack | stock export |
| MiniCPM5 | 1B | 53 + 54 | default stack | stock export (plain llama rail) |
| granite-4.1 (dense) | 3b | 20 + 15 | default stack | stock export; spurious-BOS guard fires (bos == eos) |
| Hy-MT2 (hunyuan_v1_dense) | 1.8B | 10 + 1 | default stack | stock export after a bitwise-equal rope bake; duplicate-BOS guard fires |
| Qwen3.5 | 0.8B / 2B / 4B | 1,214 + 928 | litert-torch *main* | stock export; CPU gate |
| LFM2.5 | 350M / 1.2B / 2.6B | 210 + 94 | released 0.9.3/0.9.4 | stock export + ExecutorMetadata retrofit |
| granite-4.0-h | 350m / 1b | 24 + 4 | pinned checkout | family recipe (`HYBRID_RECIPE`) |
| Falcon-H1 | 0.5B / 1.5B / 1.5B-Deep / 3B | 14 + 2 | pinned checkout | family recipe (`HYBRID_RECIPE`) |
| Zamba2 | 1.2B / 2.7B | 3 + 0; the only real one is pre-port-serialized | pinned checkout | routed; pre-port checkpoints are refused with re-serialization instructions |
| Nemotron-H | Nemotron-H-4B / Nemotron-3-Nano-4B | 19 + 14 | pinned checkout | family recipe (`HYBRID_RECIPE`); measured on Nemotron-3-Nano-4B ([shipped](https://huggingface.co/litert-community/Nemotron-3-Nano-4B)) |

For `HYBRID_RECIPE` families the one-time checkout setup command is printed on refusal.
The measurements behind each row — template byte-equality, greedy A/B against the HF
reference, and three gate refusals of genuinely defective derivatives — are in
[REPRODUCE.md](REPRODUCE.md).

**iPhone.** A converted bundle loads in an existing iOS app through swift-litert-lm: [recipe](https://github.com/john-rocky/swift-litert-lm/blob/main/docs/recipe-hf-finetune-to-iphone.md) (one SwiftPM dependency, `LiteRTChat(huggingFaceRepo:fileName:)` or `LiteRTChat(modelFileURL:)`, stop and release, a verify command with its expected output).

**VLM derivatives.** `bash scripts/ship_qwen2vl_derivative.sh <org>/<model>` builds the full
bundle. Qwen2-VL derivatives train the vision tower too (measured: 339/391 vision tensors
differ on the top derivative), so vision re-exports from the derivative's own weights. The
script ends with a tokenizer-parity gate (`scripts/gate_specials.py`: every added-token special,
Latin-1, emoji — the engine's encode against the upstream `tokenizer.json`). The 2026-08 failure
on a format-exact derivative, first blamed on the runtime, was the bundle's own SentencePiece
section; the gate is what catches it, and REPRODUCE.md carries the correction.

## Reproduce a published model

```bash
bash scripts/reproduce_llm.sh --list          # 25 LLM keys
bash scripts/reproduce_llm.sh olmo2-1b        # -> out/olmo2-1b/model.litertlm
bash scripts/reproduce_vlm.sh --list          # 13 VLMs
bash scripts/reproduce_vlm.sh ovis2.5-2b     # -> out/*-bundle/Ovis2.5-2B.litertlm
```

The 2026-08 verification sweep executed every then-current LLM recipe end-to-end and gated
it: **16/18 reproduced and passed** (the two exceptions are documented — one source repo
went gated, one thinking model the strict gate over-flags). For one model the reproduced
weights are **bit-identical** to the published artifact. Per-model recipes, caveats, and
device measurements: [REPRODUCE.md](REPRODUCE.md); per-model cards: `cards/`.

What the lists contain:

- **`reproduce_llm.sh` (25)**: `llama32-3b`, `qwen3-1.7b`, `qwen3-4b-thinking`,
  `qwen25-3b`, `ministral3-3b` (+`-reasoning`), `olmo2-1b`/`7b`, `smollm3-3b`, `twil-lm3`,
  `phi4-mini-reasoning`, `r1-distill-qwen-1.5b`/`7b`, `nanbeige4.1-3b`, `nanbeige4.2-3b`,
  `polaris-4b`, `vibethinker-3b`, `jan-nano`, `fastcontext-4b`, `falcon3-3b`, `s1-mini`,
  `granite42-3b`, `minicpm5-2b`, `spark-x2.5-1.7b`, `spark-x2.5-4b`.
- **`reproduce_vlm.sh` (13)**: `granite-docling-258m`, `internvl3-1b`,
  `internvl3.5-1b`/`2b`/`4b`, `llava-onevision-0.5b`, `mage-vl`, `north-micro-vision`,
  `ovis2.5-2b`, `paddleocr-vl-1.6`, `qwen2-vl-2b`, `smolvlm2-500m`, `smolvlm2-2.2b`.
- **Family recipes (22)** (one command each, documented per family in REPRODUCE.md):
  granite-4.0-h-1b/-350m, granite-4.1-3b, Falcon-H1-0.5B/1.5B/1.5B-Deep/3B-Instruct,
  Zamba2-1.2B/2.7B-instruct, Nemotron-H-4B-Instruct-128K, **Qwen3.5-0.8B/2B/4B**,
  LFM2.5-1.2B-Instruct/-Thinking/-JP and 2.6B, MiniCPM5-1B / MiniCPM4-0.5B / MiniCPM4.1-8B,
  Qwen2.5-Coder-1.5B-Instruct, Shieldstral-1.0-3B.
- **Beyond chat (11)**: LFM2.5-Encoder-350M/-230M and the four 350M task encoders
  (PII / policy-linter / prompt-router / spellchecker), LFM2.5-Embedding-350M,
  LFM2.5-ColBERT-350M, granite-embedding-311m — plain LiteRT `.tflite` encoders — plus
  Qwen3-TTS-12Hz-0.6B (speech) and Bonsai-Image-ternary-4B (FLUX.2-klein image generation),
  which run as LiteRT graphs under host loops rather than `.litertlm` bundles.

## Convert a new architecture

- **A dense LLM** not listed: run the engine directly —
  `EXTERNALIZE_EMBEDDER=1 CACHE=4096 $PY scripts/export_simple_template.py <hf_id> out/<name> templates/<t>.jinja BOCTAV4`
  (pick a template from `templates/`; `FORCE_SPM=1` for thinking models with added tokens).
  Then add a `case` to `scripts/reproduce_llm.sh` to keep it reproducible.
- **A single-image VLM**: copy the closest `scripts/ship_*.sh` with its
  `convert_*_vision.py` / `prep_*_decoder.py`, adjust dims and the image token.
  `ovis_work/` shows how to make a dynamic-resolution (NaViT) vision tower export-able.
- **A hybrid (SSM/attention) family**: the four pinned recipes under `granite_work/`,
  `falcon_h1_work/`, `zamba2_work/`, and `nemotron_h_work/` are the working examples — each
  is one litert-torch patch plus one driver script.

## int4 recipes

Defined in `scripts/export_simple_template.py`:

| recipe | what | when |
|---|---|---|
| `BOCTAV4` | blockwise-32 int4 + OCTAV optimal clipping + int8 embedding | best quality (Mac/Android) |
| `BOCTAV4_128` | blockwise-128 variant | 4B models / iOS (~2 GiB section limit) |
| `BMIX4[_128]` | blockwise int4 min-max + int8 embedding | GPTQ ingest, or when OCTAV isn't needed |

OCTAV is data-free (no calibration set). `EXTERNALIZE_EMBEDDER=1` splits the embedding so 3B+
models load under the iOS section limit; reasoning models use a thinking template and
`CACHE=4096`.

## Deployment manifests

A `.litertlm` bundle carries the conversation contract in its header, but nothing
machine-readable says which of a repo's files fits which device, backend, and RAM budget, at
what measured speed. `manifest/` defines a repo-level `litertlm_manifest.json` for that
deployment layer: [`manifest/SCHEMA.md`](manifest/SCHEMA.md) (spec),
[`manifest/make_manifest.py`](manifest/make_manifest.py) (generator),
[`manifest/examples/`](manifest/examples/) (finished manifests for two of the published repos — 60 repos ship one as of 2026-09-05), and [`readers/`](readers/) (dependency-free TypeScript and Dart reference readers).
The schema is registered in [SchemaStore](https://www.schemastore.org/litertlm_manifest.json), so editors that use its catalog (VS Code, JetBrains, and others) validate and autocomplete any file named `litertlm_manifest.json` with no setup.

The same manifest drives Google Play delivery: [`tools/play_ai_pack/`](tools/play_ai_pack/) turns
it into a Play for On-device AI pack — one device group per manifest recommendation, the
device-targeting XML, and a host app that fetches the pack and runs the file on the backend the
manifest names.

## Layout

| path | what |
|---|---|
| `scripts/convert.py` | the finetune converter (entry gate → export → guards → exit gate) |
| `scripts/export_simple_template.py` | the LLM engine (template + quant recipe + env knobs) |
| `scripts/reproduce_llm.sh` · `scripts/reproduce_vlm.sh` | one-command reproductions |
| `scripts/ship_*.sh`, `convert_*_vision.py`, `prep_*_decoder*.py`, `build_*_bundle.py` | the VLM pipeline |
| `granite_work/` `falcon_h1_work/` `zamba2_work/` `nemotron_h_work/` | pinned hybrid-family recipes |
| `qwen35_work/` `lfm_work/` `minicpm_work/` … | per-family recipes and gates |
| `templates/`, `recipes/` | chat templates + quant recipe JSONs |
| `cards/` | model cards for the converted bundles |
| [REPRODUCE.md](REPRODUCE.md) | the full measurement record behind every claim above |

## License

Code: Apache-2.0. Converted model bundles inherit their base model's license.

This is an independent open-source project, not affiliated with or endorsed by Google.
LiteRT and LiteRT-LM are Google projects; `.litertlm` names their runtime's bundle format.
