# litertlm_manifest.json — a deployment manifest for `.litertlm` model repos

**Status: v0.1 draft (2026-08-24; 0.1.3 as of 2026-10-07).** One `litertlm_manifest.json` at the root of a Hugging Face
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

`repo` is required and identifies the repo described by the file. A reader with fetch context
uses the **repo and revision it fetched from** for download URLs: a fork may carry a copied
manifest that still names its origin. Without source context, use the file's `repo` and default
to revision `main`. An explicit resolution revision overrides the fetch revision. Fetch context
is reader state, not additional manifest fields; the required `repo` and variant `file` must
still be valid. Reference readers reject missing, empty or non-string values with a descriptive
parse error.

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
| `capabilities.thinking.control` | *(derived, 0.1.3+)* | `switch`, `always`, `never` or `model` — what the bundle's template does about thinking; see [below](#capabilitiesthinkingcontrol--what-the-template-does-about-thinking) |
| `capabilities.channels` | *(derived, 0.1.1+)* | the bundle's **full** declared channel set, `[{name, start, end, is_reasoning?}]` — the header's `channels` block is a generic named list (a model may declare e.g. tool-call markers there, though nothing shipped today does); `thinking` keeps mirroring the first channel for 0.1.0 readers |
| `session_defaults` | curated | knobs a wrapper should set that the engine cannot infer. An **open object**; declared keys (all optional): `max_output_tokens_min` (integer — a FLOOR on the output-token budget, e.g. `2048` for reasoning models, never a cap), `notes` (string — curated guidance worth surfacing), `temperature`/`top_k`/`top_p` (sampler hints). Readers take keys by name and ignore what they don't consume |

## `capabilities.thinking.control` — what the template does about thinking

`thinking.declared` says the bundle declares a thought channel. It does not say whether an app can
turn thinking on or off. `control` *(derived, 0.1.3+, optional)* says that:

| value | the bundle's template | for an app |
|---|---|---|
| `switch` | reads `enable_thinking`, and renders a different prompt for `true` and `false`; `true` does not close the thought channel and `false` does not open it | set `enable_thinking` per turn to turn thinking on or off |
| `always` | has no switch; its generation prompt ends with the thought channel's start marker | the template ignores `enable_thinking`: every reply starts inside the thought channel |
| `never` | has no switch; the bundle declares no thought channel, or the generation prompt ends with the channel's end marker | the template ignores `enable_thinking`: no reply starts inside a thought channel |
| `model` | has no switch, and neither opens nor closes the declared thought channel | the template ignores `enable_thinking`: the model opens the channel or does not |

`enable_thinking` is a template input. A caller sets it through the runtime's `extra_context`, or
the runtime sets it from its thinking config. What a `switch` template does when nothing sets it
differs per model and is not recorded. `control` covers the template only; what the runtime's
decoder does with a thinking budget is outside it.

The generator reads the template the runtime renders. That is the bundle's
`jinja_prompt_template`. For a bundle without one, it is the template the runtime builds from the
`prompt_templates` affixes. The generator renders one user turn with a generation prompt in a
sandbox, with `enable_thinking` unset, `true` and `false`. It compares the end of the prompt with
the markers of the declared channel (the one `thinking.channel` mirrors), ignoring whitespace. The
runtime decides whether a prompt leaves a channel open by looking for the exact markers. The
generator writes a value only when that reading agrees — for `switch`, in both states.

- The source model's Hugging Face template is not consulted. A bundle carries its own template,
  and the runtime renders only that one.
- `control` describes the bundle, not the weights. A model whose bundle declares no thought
  channel can still print its reasoning as plain text.
- A switch that is not `enable_thinking` — a flag in the system message, for example — is not
  `switch`.
- The bundle's `supports_thinking` flag is not consulted. It says whether the model can think,
  not what the template does, and bundles built before the field existed leave it unset.
- One value per model. The generator stops when the files of a repo give different answers.

The key is absent when the generator cannot derive a value:

- the bundle has no jinja template, and the runtime builds none from its affixes;
- the template does not render one user turn;
- it reads `enable_thinking`, but the prompt does not change, or one state contradicts the switch;
- the runtime would read the prompt differently (a marker with other whitespace, an empty marker);
- the bundle declares more than one channel;
- a string and a list of parts as the message content give different answers.

Readers treat an absent or unknown value as no statement.

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

**Resolution rule:** an explicitly requested backend is a *filter*, not a preference — a resolver
only considers variants listing it, and reports failure (`null`) rather than substituting a backend
the caller didn't ask for.
Variants with no listed backends cannot be candidates, even if a caller constructed or mutated
the manifest without parsing. Return `null` when no candidates remain; never invent a backend
from a default or recommendation.

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
- `load_s` (optional, number, 0.1.2+) is the engine load time in seconds under the row's `cache`
  condition; `peak_memory_mb` (optional, number, 0.1.2+) is the peak resident memory during the
  run, in MB. Both are measured, never estimated.
- `evidence` (optional, string) points at the primary log; `--public` strips every `evidence` key
  under a variant (measured rows and any other curated block) and the converter keeps them in its
  own records.

## Versioning

`manifest_schema` is semver-style; the current line is `0.1.x`. While the schema is 0.x, the
minor version is the compatibility line: a 0.1.x release may only add optional fields — nothing
is removed, renamed, or changed in meaning short of `0.2.0`. `0.1.1` adds
`model.capabilities.channels[]` (the full bundle channel mirror) and declares
`session_defaults`' in-the-wild keys; both are optional, so 0.1.0 manifests stay valid. `0.1.2` adds
`measured[].load_s` and `measured[].peak_memory_mb`, also optional. `0.1.3` adds
`model.capabilities.thinking.control`, optional as well. Readers should pin a supported
range; the JSON Schema enforces the 0.1 line via the `manifest_schema` pattern, so a 0.2
manifest fails validation rather than half-parsing, and both reference readers refuse it at
parse time.

## What deliberately stays OUT of v0.1

- Anything the bundle carries (templates, stop tokens, sampler params) — read the bundle.
- Quality/parity scores — they belong on the model card with their own methodology. A variant may
  carry a curated `quality[]` block today (`{task, score, reference, date}` rows); its format is
  not settled, so the schema leaves it unvalidated and readers must treat it as opaque.
- Download URLs — the manifest lives in the repo it describes; `repo` + `file` is the address.
