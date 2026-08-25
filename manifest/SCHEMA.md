# litertlm_manifest.json — a deployment manifest for `.litertlm` model repos

**Status: v0.1 draft (2026-08-24).** One `litertlm_manifest.json` at the root of a Hugging Face
model repo describes every `.litertlm` file the repo ships: which backends each file targets,
which file a given device should pick, what it needs (runtime version, RAM), and how fast it
actually runs — with measured numbers, not claims.

## Why this file exists

The `.litertlm` bundle already carries the **conversation contract** (templates, stop tokens,
thinking-channel declaration, sampler defaults, per-section `backend_constraint`) inside its
header, and the engine applies it. The manifest deliberately does **not** duplicate any of that —
fields derived from the bundle are generated mechanically by reading the bundle header, never
written by hand.

What the bundle cannot carry, and no wrapper (flutter_gemma, react-native-litert-lm, …) or
upstream file (`~/.litert-lm/config.json` is a user-side invocation preference file) provides
today:

- **variant selection** — repos ship several files (CPU-lineage vs GPU-optimized, int4 vs int8,
  per-SoC NPU builds); nothing machine-readable says which file fits which device;
- **backend recommendation** — `backend_constraint` says what *can* run; nothing says what is
  *fastest* on a device class, with evidence;
- **requirements** — minimum runtime version, peak RAM, platform caveats;
- **measured performance** — reproducible numbers with conditions and provenance;
- **file identity** — sha256/size before download;
- **capability flags readable before download** — vision/audio/thinking are inside the binary
  header; wrappers today hardcode or filename-sniff them.

## Layering rule

| Layer | Home | Examples |
|---|---|---|
| Conversation contract | bundle (`LlmMetadata`) | template, stop tokens, channels, sampler defaults |
| Runtime compatibility | bundle (section `backend_constraint`) | "this file loads on cpu,gpu" — engine-enforced |
| User preference | `~/.litert-lm/config.json` | "I run this model on gpu with temp 0.6" |
| **Deployment** | **manifest (this file)** | variant choice, recommendation, requirements, measured perf, identity |

Fields marked *(derived)* below are read out of the bundle header (two HTTP range requests per
file — no weight download) and are therefore safe mirrors, not hand-kept copies. Everything else
is curated and must carry evidence.

## Top level

```json
{
  "manifest_schema": "0.1.0",
  "repo": "litert-community/LFM2.5-1.2B-Instruct",
  "generated": "2026-08-24",
  "generator": "make_manifest.py",
  "model": { ... },
  "variants": [ { ... } ]
}
```

## `model`

| field | source | meaning |
|---|---|---|
| `display_name` | curated | human name |
| `base_model` | curated | HF id of the source model |
| `architecture` | curated | free text (e.g. `lfm2-hybrid-shortconv`, `qwen3-dense`) |
| `parameters_b` | curated | parameter count in billions |
| `license` | curated | SPDX id or pointer |
| `context_length` | *(derived)* | bundle `max_num_tokens` |
| `capabilities.vision` / `.audio` | *(derived)* | from bundle `llm_model_type` |
| `capabilities.thinking` | *(derived)* | `{declared, channel:{start,end}}` from bundle `channels` |
| `session_defaults` | curated | knobs a wrapper should set that the engine cannot infer, e.g. `{"max_output_tokens_min": 2048}` for reasoning models |

## `variants[]` — one entry per `.litertlm` file

| field | source | meaning |
|---|---|---|
| `file` | — | file name in the repo |
| `sha256`, `size_bytes` | *(derived)* | from HF LFS metadata — verify after download |
| `quantization` | curated | recipe, honestly stated (e.g. `int4 block-32 linears, fp32 activations`) |
| `backends` | curated (checked against bundle `backend_constraint` when present) | backends this file is *verified to generate* on — not merely load |
| `default_backend` | curated | what to pick with no device knowledge |
| `min_runtime_version` | curated | earliest LiteRT-LM release the file is verified on |
| `recommended[]` | curated | `{platform, device_class?, backend, reason}` — the fastest *verified* choice per platform/device class |
| `requirements` | curated | `{peak_ram_mb?, platform_notes[]}` |
| `measured[]` | curated | see below |
| `known_issues[]` | curated | short, factual, with upstream links where they exist |
| `sections` | *(derived)* | bundle section table: type, size, `model_type`, `backend_constraint` |

## `measured[]` rows — the honesty rules

Every row states its conditions and its provenance; a row without them does not go in.

```json
{
  "device": "Pixel 8a (Tensor G3)", "backend": "gpu",
  "runtime": "litert-lm v0.16.0 (litert_lm_main, release-tag build)",
  "prompt_tokens": 263, "decode_tokens": 115,
  "prefill_tps": "188.3-192.6", "decode_tps": "21.0-21.2", "ttft_s": "1.41-1.44",
  "max_num_tokens": 4096, "cache": "no", "runs": 3,
  "date": "2026-08-12", "source": "ship-lane bench log; summarized in the model card"
}
```

- Speeds may be a number or a `"lo-hi"` range string; ranges are for run spread, and CPU rows on
  phones spread because of thermal throttling — say so in the card, keep the range here.
- `cache: "no"` matters: compiled-model caches mask load regressions and inflate disk cost.
- Rows come from real generation-verified backends only (a benchmark can report numbers on a
  backend that cannot generate text).
- `evidence` (optional, string) points at the primary log; it is stripped from published
  manifests (`--public`) and kept in the converter's own records.

## Versioning

`manifest_schema` is semver-style; the current line is `0.1.x`. While the schema is 0.x, the
minor version is the compatibility line: a 0.1.x release may only add optional fields — nothing
is removed, renamed, or changed in meaning short of `0.2.0`. Readers should pin a supported
range; the JSON Schema enforces the 0.1 line via the `manifest_schema` pattern, so a 0.2
manifest fails validation rather than half-parsing, and both reference readers refuse it at
parse time.

## What deliberately stays OUT of v0.1

- Anything the bundle carries (templates, stop tokens, sampler params) — read the bundle.
- Quality/parity scores — they belong on the model card with their own methodology; a
  `quality` block may be added in a later version once the row format is settled.
- Download URLs — the manifest lives in the repo it describes; `repo` + `file` is the address.
