# One command, one lock: `<org>/<model>` on the Hub → a `.litertlm` whose bytes are pinned

Status: implemented 2026-10-19 in `scripts/convert.py`, `scripts/verify_lock.py`, `scripts/check_bundle.py`,
`scripts/chat_template_dual.py`, `scripts/failure_line.py` + `scripts/failure_hints.json`, `scripts/verify_refs.sh`,
`pyproject.toml` + `uv.lock`, `locks/refs/`. Three things the first draft of this page did not have: the export runs under a fixed
`PYTHONHASHSEED`, a template the 0.18 runtime cannot render the way transformers does is edited (`.get` → indexing, the dual form) under a byte-for-byte proof, and the 8-question gate runs on
the pinned litert-lm CLI (sections below).

## The command

```bash
uv sync                                              # pinned toolchain into .venv (uv.lock)
uv run python scripts/convert.py <org>/<model>       # -> out/<model>/<model>_<recipe>.litertlm + <model>.lock.json + convert_report.json
uv run python scripts/verify_lock.py out/<model>/<model>.lock.json   # rebuild into a temp dir, compare sha256: MATCH / DIFF (per section)
uv run python scripts/check_bundle.py out/<model>/*.litertlm           # litert-lm 0.18.0 CPU: load + one echo prompt -> PASS / FAIL
```

`convert.py` keeps its shape (entry gate → stock `litert_torch` export → exit gate). New: `--revision <sha>`
(default: `main` resolved to its commit sha and recorded), `--recipe int8|int4` (`--int4` stays as the alias),
`--smoke` (one-prompt check instead of the 8-question gate, `--max-minutes` cap, structured failure line), and the lock.
The source is downloaded at the pinned commit, top-level files only (config, tokenizer, template, the weights:
`model*.safetensors`), and every file's size and hash is checked against the Hub's tree before the export reads it
from that local snapshot. The exit gate is `scripts/verify_quality.py` (8 questions, 6/8) on the pinned litert-lm
CLI, CPU, greedy (`scripts/litert_lm_cli_verifier.py` wraps the CLI for it: one engine per call, greedy).

## The lock (`<model>.lock.json`)

| key | what it pins | source |
|---|---|---|
| `source.repo`, `source.revision` | the Hub repo at one commit | `HfApi.model_info(revision=…)` |
| `source.files[]` | path, size, sha256 of every file the export read | `/api/models/<id>/tree/<rev>` (LFS oid); non-LFS files hashed after download |
| `recipe.id`, `recipe.export_kwargs`, `recipe.quantization_recipe_sha256` | the export call, fully resolved (prefill lengths, cache length, recipe file) | the call `convert.py` made |
| `toolchain.id` = `lt<litert-torch>-b<litert-lm-builder>-t<torch>` + the version of every installed package, `python`, `uv_lock_sha256` | the exporter stack | `importlib.metadata`, `uv.lock` |
| `bundle.sha256`, `bundle.size`, `bundle.sections[]` (name, sha256) | the output | `scripts/section_identity.py` |
| `bundle.system_metadata.uuid`, `.creation_timestamp` | the two header fields the builder would otherwise randomize | derived, see below |
| `process.PYTHONHASHSEED` | the hash seed of the export process | `convert.py` (`0` unless `HF_TO_LITERTLM_HASH_SEED` is set) |
| `template` | `form` (unchanged / get / dual / get+dual), `edits`, `original_sha256`, `final_sha256`, the per-case `classes`, `basis` and render hashes, the runtime check | `chat_template_dual.py`, see below |
| `check` | engine version, backend, the echo prompt, answer head, PASS/FAIL; `arithmetic` (17 + 25: expected 42, the answer) as information | `check_bundle.py` |
| `converter`, `built_at`, `host` | informational only — `convert.py`'s sha256 and the repo commit (`verify_lock.py` warns when it differs), date, platform | `sha256`, `git`, `platform` |

**Same command, same bytes.** `litert_lm_builder` writes a fresh `uuid4` and `now()` into every bundle header
(`litertlm_builder.populate_system_metadata`, 0.18.0), so two honest runs can never be byte-identical by default.
`convert.py` replaces that function in its own process: `uuid` = `uuid5(NAMESPACE_URL, "<repo>@<revision>#<recipe.id>#<toolchain.id>")`,
`creation_timestamp` = the source revision's commit date (UTC ISO 8601). Everything else in the pipeline (torch export on CPU,
the converter, ai-edge-quantizer, the tokenizer conversion) is expected to be deterministic; `verify_lock.py` is the test,
and a DIFF names the section that moved.

The first test moved one section. Two builds of Qwen3-1.7B with the uuid and timestamp pinned differed in LlmMetadata
only (4,449 B both times; the 1.74 GB TFLite section, the tokenizer and the executor metadata byte-identical): the same
stop tokens in two orders. litert-torch 0.9.4's `build_llm_metadata` collects them in a `set` of strings and writes them
in iteration order, which Python's per-process string-hash randomization changes. `convert.py` therefore re-executes
itself under `PYTHONHASHSEED=0` before anything else: seed 0 twice gave the same file, seed 1 changed LlmMetadata alone.

**The check.** `check_bundle.py` passes a bundle when the pinned CLI loads it and the answer to "Repeat this word
exactly and say nothing else: pineapple" (the thought removed) is non-empty, not degenerate and carries the word
("Reply with exactly one word: pineapple" reads two ways: gemma-4-E2B-it answered "Fruit"). Arithmetic measures the
model rather than the conversion (SmolLM2-1.7B-Instruct answers 32 to "What is 17 + 25? Answer with the number only."
in transformers fp32 too), so its answer is recorded and never decides; the echo catches the bundle whose template
drops the user's text on litert-lm 0.18, which answers fluently about an empty message.

## Generic path and curated recipes

The default is the stock exporter with no per-architecture layer (measured on untouched finetunes across the
model_ext and generic paths; the stock export embeds the derivative's own chat template). `recipes/curated_index.json`
maps Hub ids that this repo ships with a hand-made recipe to their `scripts/reproduce_llm.sh` key; `convert.py` prints
that pointer and continues on the generic path unless `--curated` is given. Hybrid families without a released
exporter route to the pinned family recipe (the checkout command of the family recipe is printed when the installed exporter lacks the family). Curated and family outputs carry no lock yet.

## Chat template: the form the 0.18 runtime renders

litert-lm 0.18.0 renders the chat template with minijinja on native values: every message `content` is a list of text
parts, never a string ([tools/chat_template_parts/README.md](../tools/chat_template_parts/README.md)), and a map has no
Python methods. The stock export embeds the model's own template, written for transformers, and on the pinned runtime:

- Qwen3-1.7B's sets `content = ''` when the content is not a string: the runtime's own render of a user turn
  (`Conversation.render_message_to_string`) came back as `<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n`, the
  same bundle answering on litert-lm 0.17.1;
- SmolLM2-1.7B-Instruct's concatenates `str + list` and raises;
- gemma-4-E2B-it's calls `.get(…)` 25 times (`message.get('reasoning')` at line 238 runs for every message): `Failed to apply template: unknown method: map has no method named
  get (in template:238)` on every message, with or without tools (the CLI answers "An error occurred").

`convert.py` renders the template before the export the runtime's way (`tools/chat_template_prefix/desk_render.py`:
minijinja 2.14.0, the runtime's rewrites of Python idioms, its `none` test, `bos_token` = `eos_token` = `""`, and the
messages handed in as minijinja literals so that no Python method leaks in) and compares it with transformers' own
render of the original (`render_jinja_template`, the same bos / eos) on four conversations — c1 user, c2 system + user,
c3 user / assistant / user, c4 user with `enable_thinking` true and false when the template reads it. Two edits, each
only when the template cannot render without it:

1. `X.get('k')` → `X['k']` (a missing key is undefined, which the runtime's `none` test reads as the None `.get`
   returns; before an `is none` test the form is `(X['k'] if X['k'] is defined else none)`);
2. the dual form: LiteRT-LM's `format_content` macro (`a8178d7d`, `models/qwen3/chat_template.jinja`) prepended and
   every direct content read (`message.content`, `message['content']`, `messages[0].content`) routed through it.

Each case is compared with transformers' string render, or — when the template renders parts itself and its parts
render keeps every text — with transformers' parts render (gemma-4's c2: its list branch writes `item['text'] | trim +
' '`, one space after the system text, which its string branch does not write; the per-message list branch adds no space). The edited template
goes into the export as `jinja_chat_template_override` (one build) only if every case is byte-identical; otherwise
`FAIL template: …` and no bundle. The lock's `template` records the form (unchanged / get / dual / get+dual), the
edit counts, the original runtime class of each case (equal / drop / raise / leak / differs), the basis and the
hashes. After the export, `convert.py` reads the runtime's render of c1–c3 (a separate process, compile cache in a
temporary dir) and requires it to equal transformers' render on the same basis (the bundle's start token in front
when it keeps one).

## When it fails

One line on stderr and the same fields in `convert_report.json["failure"]`:

```
FAIL <stage>: <op or first error line> -> next: <one hint>
```

Stages: `fetch` (404 / 401 / gated, revision not found, a file whose size or hash differs from the Hub's), `gate`
(remote-code, pre-quantized, an adapter that names no base), `template` (the runtime's render of the template, edited or
not, differs from transformers'), `load` (transformers), `trace` (torch.export), `convert` (the LiteRT converter: unsupported op, named),
`quantize`, `bundle`, `engine-load`, `generate` (empty or wrong output, a failed gate). The stage of an export failure is
the step the export was in (`convert.py` records it as the export moves through `load_model`, `torch.export.export`,
the flatbuffer conversion, `quantize_model` and the packaging). The op is taken from the traceback (`aten.*`,
`stablehlo.*`, `tfl.*`, "Unsupported op …"); `scripts/failure_hints.json` maps a regex on the stage and the text to the
hint, and a row is added only with a measured case (hybrid model_type without model_ext → the family recipe or
litert-torch main; a data-dependent guard in torch.export; a 429 from the Hub), else a refusal's own action, else a
model request on this repository with the failure line. `convert_report.json["failure"]` also carries the seconds and
the toolchain id. Exit codes: 2 = refused or stopped before any download, 3 = any later failure (a file whose hash differs from the Hub's included), 1 = gate or check.

## Raising the toolchain (the only direction)

```bash
# first raise the == pins of litert-torch, litert-lm-builder and litert-lm in pyproject.toml (torch moves in its own step)
uv lock
uv sync
bash scripts/verify_refs.sh        # the three reference models, twice each: new lock files land next to the old ones under locks/refs/
```

`locks/refs/CHANGELOG.md` gets one row per reference model and toolchain: same bytes / changed bytes against the previous
toolchain, the rebuild (MATCH / DIFF), check PASS / FAIL, 17 + 25 (information), the compat shims applied, seconds and
peak RSS of both builds. A new toolchain becomes the pinned one when all three references pass; old lock files are
never deleted (they are the record of what a toolchain produced). One shim rides on the pinned pair: litert-torch
0.9.4's gemma4 metadata builder sets `use_template_for_fc_format`, which litert-lm-builder 0.18.0's `Gemma4` message no
longer has, so every gemma4 export of the pair dies at packaging (google-ai-edge/litert-torch#1329, fixed on main by
#1241); `convert.py` skips that one assignment only when both hold, as main does, and records it in `recipe.compat`. The
CHANGELOG's compat column shows when a raise has made it a no-op. The reference models: `Qwen/Qwen3-1.7B`
(the qwen3 export path), `HuggingFaceTB/SmolLM2-1.7B-Instruct` (llama, the generic path) and `google/gemma-4-E2B-it`
(the gemma4 export path): one model per export path a new finetune of a small model takes.

A batch of models can be run the same way: `--smoke` exits non-zero with the failure line, and
`convert_report.json["failure"]` carries stage, op, seconds and toolchain for a table.
