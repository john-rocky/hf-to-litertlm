# hf-to-litertlm

Convert **open-weight Hugging Face models → `.litertlm`** bundles for on-device inference with the
**LiteRT-LM** runtime (CPU/GPU, iOS/Android/desktop). Covers **dense/reasoning LLMs** and
**single-image VLMs**, with a one-command reproduction for every model listed here.

## Install

```bash
# a Python env with litert_torch + ai_edge_quantizer (conversions are toolchain-version sensitive)
pip install litert-torch ai-edge-quantizer transformers huggingface_hub
export PY=python            # or point at your env, e.g. ~/clipconv/bin/python
```

The scripts default to `~/clipconv/bin/python`; override by editing the `PY=` line or exporting `PY`.

## Convert (one command per model)

### Dense / reasoning LLMs → int4 `.litertlm`

```bash
bash scripts/reproduce_llm.sh --list          # 18 models
bash scripts/reproduce_llm.sh olmo2-1b        # -> out/olmo2-1b/model.litertlm
python scripts/verify_quality.py out/olmo2-1b/model.litertlm      # 8-question local gate
```

Covered: `llama32-3b`, `qwen3-1.7b`, `qwen3-4b-thinking`, `ministral3-3b(+reasoning)`, `olmo2-1b/7b`,
`smollm3-3b`, `phi4-mini-reasoning`, `r1-distill-qwen-1.5b/7b`, `nanbeige4.1-3b`, `polaris-4b`,
`vibethinker-3b`, `jan-nano`, `fastcontext-4b`, `falcon3-3b`, `qwen25-3b`. Full recipe table +
per-model caveats in **[REPRODUCE.md](REPRODUCE.md)**.

All recipes export with `use_jinja_template=False` (plain prefix/suffix turn markers, no embedded Jinja) — vendor HF chat templates often call Python-style methods (`.get()`, `.startswith()`) that the runtime's minijinja renderer rejects, crashing on the first message. See the template-safety note in [REPRODUCE.md](REPRODUCE.md).

### Vision-language models (`fast_vlm` single image) → `.litertlm`

```bash
bash scripts/reproduce_vlm.sh --list          # 8 VLMs
bash scripts/reproduce_vlm.sh ovis2.5-2b      # -> out/*-bundle/Ovis2.5-2B.litertlm
```

Covered: `internvl3-1b`, `internvl3.5-1b/2b/4b`, `llava-onevision-0.5b`, `ovis2.5-2b`,
`smolvlm2-500m`, `smolvlm2-2.2b`. Each downloads the source, converts the vision tower
(encoder + adapter) and the decoder (int4), and assembles the fast_vlm bundle; see the matching
`cards/<name>-litert.md`.

## The int4 recipe (why it holds quality)

Defined in `scripts/export_simple_template.py`:

| recipe | what | when |
|---|---|---|
| `BOCTAV4` | blockwise-32 int4 + **OCTAV** optimal-clipping + int8 embedding | best quality (Mac/Android) |
| `BOCTAV4_128` | blockwise-128 variant | 4B models / iOS (fits the ~2 GiB section) |
| `BMIX4[_128]` | blockwise int4 min-max + int8 embedding | GPTQ ingest / when OCTAV isn't needed |

OCTAV is **data-free** (no calibration set) and recovers a large chunk of the naive-int4 gap.
`EXTERNALIZE_EMBEDDER=1` splits the embedding so 3B+ models load under the iOS section limit;
reasoning models use a thinking template and `CACHE=4096`.

## Convert your own model

- **A dense LLM** not listed: run the engine directly —
  `EXTERNALIZE_EMBEDDER=1 CACHE=4096 $PY scripts/export_simple_template.py <hf_id> out/<name> templates/<template>.jinja BOCTAV4`
  (pick a template from `templates/`; add `FORCE_SPM=1` for thinking models with special added tokens).
  Then add a `case` to `scripts/reproduce_llm.sh` to keep it reproducible.
- **A single-image VLM**: copy the closest `scripts/ship_*.sh` and its `convert_*_vision.py` /
  `prep_*_decoder.py`, adjust dims + image token. Ovis's `ovis_work/` shows how to make a
  dynamic-resolution (NaViT) vision tower export-able.

## Layout

| path | what |
|---|---|
| `scripts/export_simple_template.py` | the LLM engine (template + quant recipe + env knobs) |
| `scripts/reproduce_llm.sh` · `REPRODUCE.md` | one-command LLM reproduction + recipe table |
| `scripts/reproduce_vlm.sh` · `scripts/ship_*.sh` | one-command VLM reproduction (the fast_vlm pipeline) |
| `scripts/convert_*_vision.py`, `prep_*_decoder*.py`, `build_*_bundle.py` | VLM building blocks |
| `ovis_work/` | Ovis2.5 static-NaViT vision rewrite (dynamic-res → export-able) |
| `templates/`, `recipes/` | ChatML/thinking templates + quant recipe JSONs |
| `cards/` | model cards for the converted bundles |

## Reproducibility

Every LLM recipe in `REPRODUCE.md` was executed end-to-end and gated: **16/18 reproduce and pass**
(the 2 exceptions are documented — a source repo that went gated, and a thinking model the strict gate
over-flags). For one model the reproduced bundle was byte-compared to the published artifact: the
**model weights are bit-identical** (only ~377 bytes of container metadata differ).

## Deployment manifests (`litertlm_manifest.json`)

The `.litertlm` bundle carries the conversation contract (templates, stop tokens, thinking channels) in its header, but nothing machine-readable says which of a repo's several files fits which device, on which backend, with how much RAM, at what measured speed. `manifest/` defines a repo-level `litertlm_manifest.json` for exactly that deployment layer: variant selection, per-platform backend recommendations, requirements, and measured performance rows with their conditions.

- [`manifest/SCHEMA.md`](manifest/SCHEMA.md) — the spec and the layering rule (nothing the bundle already carries is duplicated by hand; bundle-derived fields are generated by reading the published file's header with two HTTP range requests).
- [`manifest/make_manifest.py`](manifest/make_manifest.py) — the generator (HF LFS identity + bundle header + curated deployment data, validated against [`manifest/litertlm_manifest.schema.json`](manifest/litertlm_manifest.schema.json)).
- [`manifest/examples/`](manifest/examples/) — finished manifests for [LFM2.5-1.2B-Instruct](https://huggingface.co/litert-community/LFM2.5-1.2B-Instruct) (a repo where one int4 file can create a GPU engine and its sibling cannot) and [Qwen3-4B-Thinking-2507](https://huggingface.co/litert-community/Qwen3-4B-Thinking-2507) (block-128 vs block-32 int4, and a reasoning model that needs `max_output_tokens >= 2048`).

## License

Code: Apache-2.0. Converted model bundles inherit their base model's license.
