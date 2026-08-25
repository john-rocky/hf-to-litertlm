# hf-to-litertlm

Convert open-weight Hugging Face models to **`.litertlm`** bundles for the **LiteRT-LM**
runtime (CPU/GPU on iOS, Android, desktop). Two things live here:

1. **A finetune converter.** `python scripts/convert.py <org>/<model>` — one command from Hub
   id to a gated bundle. It covers finetunes of Qwen3.5, LFM2.5, MiniCPM5, granite-4.0-h,
   Falcon-H1, and every dense architecture the stock exporter handles — about **2,470 tagged
   derivatives** on the Hub as of 2026-08-25. LoRA/PEFT repos merge automatically. Every bundle
   is gated before it is called done; broken models are refused with a machine-readable reason.
2. **One-command reproductions** of the published litert-community models: **18 LLMs and
   13 single-image VLMs**, with the full recipe record in [REPRODUCE.md](REPRODUCE.md).

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
  before anything downloads.
- **Adapter merge.** A LoRA/PEFT repo is merged into its base first (subprocess-isolated);
  the adapter's own tokenizer, chat template, and generation config win over the base's.
- **Export.** Stock litert-torch defaults for dense models; pinned family recipes for the
  architectures no released exporter converts (table below). The derivative's own chat
  template is embedded verbatim. ≥3B hybrids also get the reduced 7-signature prefill ladder.
- **Post-export guards.** A missing turn-end stop token is added; the granite family recipes
  drop the spurious start token; the ExecutorMetadata section is retrofitted where the
  exporter omits it.
- **Exit gate.** `verify_quality.py` (8 questions, bar 6/8, think-aware budget) — or your
  `--gate-script` for models the generic gate cannot certify (an Arabic diacritizer answers
  no trivia). Exit 0 = converted and gated, 1 = converted but gate failed, 2 = refused.

Routing is automatic, by `config.json` model_type:

| family | path | note |
|---|---|---|
| dense (llama, qwen, granite, …) | stock export | includes MiniCPM5 finetunes — no routing needed |
| `qwen3_5` | stock export, litert-torch *main* | CPU gate; GPU-delegable ships come from `qwen35_work/` |
| `lfm2` (LFM2.5) | stock export, released 0.9.3/0.9.4 | ExecutorMetadata retrofitted automatically |
| `granitemoehybrid`, `falcon_h1`, `zamba2`, `nemotron_h` | pinned family recipe (`HYBRID_RECIPE`) | one-time checkout; the setup command is printed on refusal |

The measurements behind each row — template byte-equality, greedy A/B against the HF
reference, and three gate refusals of genuinely defective derivatives — are in
[REPRODUCE.md](REPRODUCE.md).

**VLM derivatives.** `bash scripts/ship_qwen2vl_derivative.sh <org>/<model>` builds the full
bundle. Qwen2-VL derivatives train the vision tower too (measured: 339/391 vision tensors
differ on the top derivative), so vision re-exports from the derivative's own weights. Honest
limit: the fast_vlm runtime currently mis-encodes ChatML special tokens, so format-exact
derivatives fail an HF-parity gate — measured and documented in REPRODUCE.md.

## Reproduce a published model

```bash
bash scripts/reproduce_llm.sh --list          # 18 LLMs
bash scripts/reproduce_llm.sh olmo2-1b        # -> out/olmo2-1b/model.litertlm
bash scripts/reproduce_vlm.sh --list          # 13 VLMs
bash scripts/reproduce_vlm.sh ovis2.5-2b     # -> out/*-bundle/Ovis2.5-2B.litertlm
```

Every LLM recipe was executed end-to-end and gated: **16/18 reproduce and pass** (the two
exceptions are documented — one source repo went gated, one thinking model the strict gate
over-flags). For one model the reproduced weights are **bit-identical** to the published
artifact. Per-model recipes, caveats, and device measurements: [REPRODUCE.md](REPRODUCE.md);
per-model cards: `cards/`.

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
[`manifest/examples/`](manifest/examples/) (finished manifests for two published repos).

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
