# hf-to-litertlm

Convert open-weight Hugging Face models to **`.litertlm`** bundles for the **LiteRT-LM**
runtime (CPU/GPU on iOS, Android, desktop). Two things live here:

1. **A finetune converter.** `python scripts/convert.py <org>/<model>` — one command from Hub
   id to a gated bundle. It covers finetunes of Qwen3.5, LFM2.5, MiniCPM5, granite-4.0-h,
   Falcon-H1, Nemotron-H/Nemotron-3-Nano, and every dense architecture the stock exporter
   handles — about **2,670 tagged derivatives** on the Hub as of 2026-08-26. LoRA/PEFT repos
   merge automatically. Every bundle is gated before it is called done; broken models are
   refused with a machine-readable reason.
2. **One-command reproductions** of the published litert-community models: **56 chat/task
   models** (dense LLMs, hybrid families, single-image VLMs) plus 11 more conversions
   (encoder/embedding, TTS, image generation), with the full recipe record in
   [REPRODUCE.md](REPRODUCE.md).

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

**VLM derivatives.** `bash scripts/ship_qwen2vl_derivative.sh <org>/<model>` builds the full
bundle. Qwen2-VL derivatives train the vision tower too (measured: 339/391 vision tensors
differ on the top derivative), so vision re-exports from the derivative's own weights. The
script ends with a tokenizer-parity gate (`scripts/gate_specials.py`: every added-token special,
Latin-1, emoji — the engine's encode against the upstream `tokenizer.json`). The 2026-08 failure
on a format-exact derivative, first blamed on the runtime, was the bundle's own SentencePiece
section; the gate is what catches it, and REPRODUCE.md carries the correction.

## Reproduce a published model

```bash
bash scripts/reproduce_llm.sh --list          # 21 LLM keys
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

- **`reproduce_llm.sh` (21)**: `llama32-3b`, `qwen3-1.7b`, `qwen3-4b-thinking`,
  `qwen25-3b`, `ministral3-3b` (+`-reasoning`), `olmo2-1b`/`7b`, `smollm3-3b`, `twil-lm3`,
  `phi4-mini-reasoning`, `r1-distill-qwen-1.5b`/`7b`, `nanbeige4.1-3b`, `nanbeige4.2-3b`,
  `polaris-4b`, `vibethinker-3b`, `jan-nano`, `fastcontext-4b`, `falcon3-3b`, `s1-mini`.
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
[`manifest/examples/`](manifest/examples/) (finished manifests for two of the published repos — 27 repos ship one as of 2026-09-02), and [`readers/`](readers/) (dependency-free TypeScript and Dart reference readers).

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
