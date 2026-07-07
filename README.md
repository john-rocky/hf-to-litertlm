# litertlm-convert

A custom converter that turns **open-weight Hugging Face models into `.litertlm` bundles**
for on-device inference with the **LiteRT-LM** runtime. It wraps `litert_torch` /
`ai_edge_quantizer` with the recipes, templates, and prep steps needed to ship real
LLMs and VLMs — and to **reproduce** each conversion with one command.

## What it converts

- **Dense / reasoning LLMs** — Llama, Qwen3, Mistral/Ministral, OLMo-2, SmolLM3, Phi-4-mini,
  DeepSeek-R1-Distill, and more → int4 (`.litertlm`) via a single ChatML-structured template.
- **Vision-language models** (the `fast_vlm` single-image path) — InternVL, LLaVA-OneVision,
  SmolVLM2, and **Ovis2.5** (a static rewrite of its dynamic-resolution NaViT vision tower).
- **Other** — audio codecs (DAC), CV encoders (SigLIP2, etc.).

## Quick start

```bash
# one command per shipped LLM -> out/<key>/model.litertlm
bash scripts/reproduce_llm.sh --list
bash scripts/reproduce_llm.sh olmo2-1b

# verify a bundle (8-question local gate)
python scripts/verify_quality.py out/olmo2-1b/model.litertlm
```

The int4 recipe legend (defined in `scripts/export_simple_template.py`):
`BOCTAV4` = blockwise-32 int4 + OCTAV optimal-clipping + int8 embedding (best quality) ·
`BOCTAV4_128` = blockwise-128 (fits the ~2 GiB iOS section for 4B) · `BMIX4[_128]` = blockwise
int4 min-max. OCTAV is data-free (no calibration set) and recovers a large chunk of the int4 gap.

## Layout

| path | what |
|---|---|
| `scripts/export_simple_template.py` | the LLM engine (template + quant recipe + env knobs) |
| `scripts/reproduce_llm.sh` + `REPRODUCE.md` | one-command reproduction of every shipped LLM + recipe table |
| `scripts/convert_*_vision.py`, `prep_*_decoder*.py`, `build_*_bundle.py`, `ship_*.sh` | the VLM (`fast_vlm`) pipeline |
| `ovis_work/` | Ovis2.5 static-NaViT vision rewrite (dynamic-res → export-able) |
| `templates/`, `recipes/` | ChatML/thinking templates + quant recipe JSONs |
| `cards/` | model cards for the converted bundles |

## Reproducibility

Every LLM recipe in `REPRODUCE.md` was executed end-to-end and gated: **16/18 reproduce and
pass** (the two exceptions are a source repo that went gated, and a thinking model the strict
gate over-flags — both documented). For one model the reproduced bundle was byte-compared to the
published artifact: **the model weights are bit-identical** (only ~377 bytes of container
metadata differ).

## Requirements

A working `litert_torch` + `ai_edge_quantizer` environment (the conversions are toolchain-version
sensitive; see `REPRODUCE.md` caveats). Conversions run on CPU; the `.litertlm` runs on the
LiteRT-LM runtime (CPU/GPU, iOS/Android/desktop).

## License

Code: Apache-2.0. Converted model bundles inherit their base model's license.
