# Converting a model

## From a Hugging Face model to a phone

```bash
uv sync                                                    # the pinned toolchain (uv.lock) into .venv
uv run python scripts/convert.py <org>/<model>             # -> out/<model>/<model>_int8.litertlm + <model>.lock.json
uv run python scripts/verify_lock.py out/<model>/<model>.lock.json   # rebuild from the lock: MATCH = the same bytes
uv run litert-lm run out/<model>/*.litertlm --prompt "Hello"         # same bundle runs on a phone: see below
```

The same scripts run from PyPI without a clone: `uvx hf-to-litertlm <org>/<model>` converts on the `uv.lock` of the
commit the package was built from and writes under `./out`; `uvx hf-to-litertlm verify <lock>` and
`uvx hf-to-litertlm check <bundle>` run `verify_lock.py` and `check_bundle.py` (litert-lm 0.18.0, CPU: load + one echo
prompt). The hybrid families and `--curated` need a clone; the package says so and names it
([pypi/README.md](../pypi/README.md)).

`convert.py` refuses, with a JSON reason, what it cannot convert honestly (gated, remote-code,
pre-quantized repos) and gates every bundle before calling it done. The lock records the Hub
commit, the sha256 of every file the export read, the export call and every package version
([docs/one-command-lock.md](one-command-lock.md)); a failure ends in one line,
`FAIL <stage>: <op or first error line> -> next: <hint>`. When litert-lm 0.18 cannot render the model's chat template the way transformers does,
`convert.py` edits it (`.get` → indexing, text parts → one string) and keeps the edit only if the prompt
renders byte-identical on three test conversations (four when the template reads `enable_thinking`); the lock records
the edit counts. Qwen3-1.7B, SmolLM2-1.7B-Instruct and
gemma-4-E2B-it rebuild from their locks byte for byte (`bash scripts/verify_refs.sh`,
[locks/refs/CHANGELOG.md](../locks/refs/CHANGELOG.md)). To rebuild a published model
instead: `bash scripts/reproduce_llm.sh <key>` or `bash scripts/reproduce_vlm.sh <key>`.

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
python scripts/convert.py <org>/<model> --check      # entry gate only, exits 0/2, nothing downloaded
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
[REPRODUCE.md](../REPRODUCE.md).


**VLM derivatives.** `bash scripts/ship_qwen2vl_derivative.sh <org>/<model>` builds the full
bundle. Qwen2-VL derivatives train the vision tower too (measured: 339/391 vision tensors
differ on the top derivative), so vision re-exports from the derivative's own weights. The
script ends with a tokenizer-parity gate (`scripts/gate_specials.py`: every added-token special,
Latin-1, emoji — the engine's encode against the upstream `tokenizer.json`). The 2026-08 failure
on a format-exact derivative, first blamed on the runtime, was the bundle's own SentencePiece
section; the gate is what catches it, and REPRODUCE.md carries the correction.

## Speech recognition models

A standalone speech recognition checkpoint converts with the same command. Today that is transformers'
`ParakeetForTDT` (model_type `parakeet_tdt`): nvidia/parakeet-tdt-0.6b-v3 and finetunes saved with
transformers. The model_type picks the route; `--task asr` names it.

```bash
uvx hf-to-litertlm nvidia/parakeet-tdt-0.6b-v3 --smoke               # the package from 0.1.2 on
uv run python scripts/convert.py nvidia/parakeet-tdt-0.6b-v3 --smoke  # the same from a clone, today
# -> out/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3_int8.litertlm (614 MB) + .lock.json + convert_report.json
```

The file has the form of litert-torch [PR #1426](https://github.com/google-ai-edge/litert-torch/pull/1426):
a Speech2TextMetadata section, the repo's own `tokenizer.json` and one `.tflite` with the `encode` /
`decode` / `decode_1` signatures of litert-community's
[parakeet-tdt-0.6b-v3](https://huggingface.co/litert-community/parakeet-tdt-0.6b-v3) file. MediaPipe's
`AudioTranscriber` runs it (`pip install mediapipe==1.2.0 soundfile`), and LiteRT-LM's C++ `OmniEngine`
reads the same sections; the `litert-lm` 0.19.0 Python package and CLI have no speech-to-text API.

```python
from mediapipe.tasks.python.audio import audio_transcriber
from mediapipe.tasks.python.core import base_options
import soundfile as sf

audio, rate = sf.read("speech.wav", dtype="float32")
if audio.ndim > 1:
    audio = audio.mean(axis=1)  # the task takes mono samples
options = audio_transcriber.AudioTranscriberOptions(
    base_options=base_options.BaseOptions(model_asset_path="out/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3_int8.litertlm"),
    sample_rate_hz=rate)
with audio_transcriber.AudioTranscriber.create_from_options(options) as transcriber:
    print(transcriber.transcribe(audio, end_of_stream=True).text)
```

The gate transcribes LibriSpeech clips through that task on the CPU: `--smoke` one clip, by default 73
clips with a word error rate of at most 0.10 (on a Mac, litert-community's own file scores 0.0609 on them
and this route's file 0.0626). The export needs no NeMo. CTC, RNN-T, Whisper, Moonshine and the streaming
Nemotron models are refused with the reason. The design, the field-by-field match with PR #1426 and the
measurements: [docs/asr-route.md](asr-route.md). `verify <lock>` rebuilds an ASR file to the same bytes
like any other.

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
device measurements: [REPRODUCE.md](../REPRODUCE.md); per-model cards: `cards/`.

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
- **Beyond chat (13)**: LFM2.5-Encoder-350M/-230M and the four 350M task encoders
  (PII / policy-linter / prompt-router / spellchecker), LFM2.5-Embedding-350M,
  LFM2.5-ColBERT-350M, granite-embedding-311m — plain LiteRT `.tflite` encoders — plus
  Qwen3-TTS-12Hz-0.6B (speech), Bonsai-Image-ternary-4B (FLUX.2-klein image generation)
  and Kev-0.8B/4B (typed questions answered with option probabilities, `kev_work/`),
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

## What lives here

1. **A finetune converter.** `python scripts/convert.py <org>/<model>`: one command from Hub
   id to a gated bundle. It covers finetunes of Qwen3.5, LFM2.5, MiniCPM5, granite-4.0-h,
   Falcon-H1, Nemotron-H/Nemotron-3-Nano, and every dense architecture the stock exporter
   handles, about **2,670 tagged derivatives** on the Hub as of 2026-08-26. LoRA/PEFT repos
   merge automatically. Broken models are refused with a machine-readable reason.
2. **One-command reproductions** of every model in the table, with the full recipe record in
   [REPRODUCE.md](../REPRODUCE.md).
3. **A deployment manifest**, `litertlm_manifest.json`, that every published repo ships, with
   reference readers and a Google Play packer ([manifests.md](manifests.md)).
4. **An on-device check for converted graphs.** `tools/devicebench/` runs a graph on a shared Android
   phone and keeps the record of every leg; [the agent skill](../skills/android-device-gate/SKILL.md) walks through it.

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
| [REPRODUCE.md](../REPRODUCE.md) | the full measurement record behind every claim above |
