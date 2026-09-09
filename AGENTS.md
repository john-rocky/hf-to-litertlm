# AGENTS.md — for coding agents converting a Hugging Face model to LiteRT-LM

Read this first if someone asked you to "convert *some model* to LiteRT / `.litertlm`", "run a
Hugging Face model on Android or iPhone with LiteRT-LM", or "rebuild one of the published
bundles". This file routes; the README has the commands.

**What this is.** A converter from open-weight Hugging Face LLMs and VLMs to `.litertlm` bundles
for Google's [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) runtime, plus the recipe
and the measurement record for 62 published conversions
([litert-community](https://huggingface.co/litert-community) on Hugging Face). Independent
converter, not affiliated with Google.

## Route

| Task | Do | Where |
|---|---|---|
| Check whether the model is already converted | Look it up in the model table; each row links the Hugging Face repo and its recipe | README "Converted models" |
| Convert a finetune, or any supported model, to a gated bundle | `python scripts/convert.py <org>/<model>` (`--int4` for the proven int4 recipe) | README "Convert a finetune" |
| Run the bundle on a Mac to check it | `litert-lm run out/<model>/*.litertlm --prompt "Hello"` | README top |
| Rebuild a published bundle | `bash scripts/reproduce_llm.sh <key>` or `bash scripts/reproduce_vlm.sh <key>` (`--list` for keys) | [REPRODUCE.md](REPRODUCE.md) |
| Run a published bundle in an Android app | [hfmodels-android](https://github.com/john-rocky/hfmodels-android): one Gradle line, a few lines of Kotlin; the walkthrough pins LFM2.5-1.2B-Instruct on a Pixel 8a | hfmodels-android README "Add it" |
| Run a bundle in an iPhone app | [swift-litert-lm](https://github.com/john-rocky/swift-litert-lm): `LiteRTChat(huggingFaceRepo:fileName:)` or `LiteRTChat(modelFileURL:)` | [recipe: a fine-tuned model on iPhone](https://github.com/john-rocky/swift-litert-lm/blob/main/docs/recipe-hf-finetune-to-iphone.md) |
| A hybrid family no released exporter handles (granite-4.0-h, Falcon-H1, Zamba2, Nemotron-H) | `convert.py` routes to the pinned family recipe by `model_type`; the one-time checkout command is printed on refusal | README "Convert a finetune" table, `<family>_work/` |
| A dense architecture that is not listed | `scripts/export_simple_template.py`, then a `case` in `scripts/reproduce_llm.sh` | README "Convert a new architecture" |
| A new VLM | Copy the closest `scripts/ship_*.sh` with its `convert_*_vision.py` / `prep_*_decoder.py` | README "Convert a new architecture" |
| Which file fits which device, backend and RAM budget, at what speed | `litertlm_manifest.json` in every published repo; schema, generator and readers in this repo | README "Deployment manifests", [manifest/SCHEMA.md](manifest/SCHEMA.md), [readers/](readers/) |
| GPU or NPU on Android for these bundles | Measured notes | [docs/android-gpu.md](docs/android-gpu.md), [docs/android-npu.md](docs/android-npu.md) |
| Ask for a model to be converted | Open a model request with the Hugging Face link; that is the whole ask | [CONTRIBUTING.md](CONTRIBUTING.md) |

## Rules

1. **A bundle that loads is not done.** The exit gate decides: `scripts/verify_quality.py`, 8 fixed
   questions, bar 6/8. `convert.py` exits 0 = converted and gated, 1 = converted but the gate
   failed, 2 = refused. Report the exit code and `out/<model>/convert_report.json`, not "it ran".
2. **A refusal is an answer.** Gated, remote-code and pre-quantized repos are refused with a JSON
   reason before anything downloads. Quote the reason; do not work around the gate.
3. **Pin the toolchain from README "Setup".** `transformers==5.14.*`; Qwen3.5 exports only on
   litert-torch `main` (the released 0.9.4 output is degenerate). When the stack is wrong,
   `convert.py` prints the exact install command.
4. **Numbers carry a device, a backend and a date.** The table's speeds are read from each repo's
   `litertlm_manifest.json`. A number without those three is not a measurement; do not extrapolate
   a Galaxy S26 GPU figure to another phone.
5. **The derivative's own chat template is embedded verbatim.** Do not substitute the base model's
   template; the adapter's tokenizer, template and generation config win over the base's.
6. **Model files are not in this repository.** Bundles live on Hugging Face; `out/` is local.

## Not this repo

- Single-graph `.tflite` models for Android (vision, audio, TTS) → [LiteRT-Models](https://github.com/john-rocky/LiteRT-Models).
- Apple's Core AI runtime (`.aimodel`) → [coreai-model-zoo](https://github.com/john-rocky/coreai-model-zoo).
- Whether a `.tflite` op runs on a given delegate, measured → [edge-compat](https://github.com/john-rocky/edge-compat).

Maintainer: john-rocky (GitHub), mlboydaisuke (Hugging Face). Issues: https://github.com/john-rocky/hf-to-litertlm/issues
