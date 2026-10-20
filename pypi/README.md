# hf-to-litertlm

One command from a Hugging Face model repo to a `.litertlm` bundle for Google's
[LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) runtime, on a pinned toolchain. Rebuilding from
the lock file it writes gives the same bytes.

```bash
uvx hf-to-litertlm <org>/<model>                              # -> out/<model>/: <model>_int8.litertlm, <model>.lock.json, convert_report.json
uvx hf-to-litertlm verify out/<model>/<model>.lock.json       # rebuild from the lock: MATCH = the same bytes
uvx hf-to-litertlm check out/<model>/<model>_int8.litertlm    # litert-lm 0.18.0, CPU: load + an echo prompt -> PASS / FAIL
```

`uvx` needs [uv](https://docs.astral.sh/uv/); `pip install hf-to-litertlm` pulls uv in as a dependency
and gives the same `hf-to-litertlm` command. Tested on macOS arm64 only; Linux is untested. On an M4 Max
(128 GB), a 1.7B model took 3 to 4 minutes, the gate included, and peaked at 25 GB of RAM;
gemma-4-E2B-it took 7 minutes and 64 GB
([locks/refs/CHANGELOG.md](https://github.com/john-rocky/hf-to-litertlm/blob/main/locks/refs/CHANGELOG.md)).

- **Pinned toolchain.** litert-torch 0.9.4, litert-lm-builder 0.18.0, litert-lm 0.18.0, torch 2.13.0 and
  transformers 5.14.1, and every other package at the version in the repository's `uv.lock`. The package
  carries the converter scripts with that `pyproject.toml` and `uv.lock` and runs them with
  `uv run --frozen` in `~/.cache/hf-to-litertlm/<version>/`, which is the environment `uv sync` makes in
  a clone of the repository. So a lock written by the package rebuilds in a clone, and the other way
  round, on the same Python 3.12.x. The first run installs the environment (about 1.6 GB on macOS arm64;
  uv also fetches a Python 3.12 if none is installed). Model files download into the Hugging Face cache.
- **The lock** (`<model>.lock.json`) records the Hub commit, the size and sha256 of every file the export
  read, the export call, every package version, and the sha256 of the bundle and of each of its sections.
  litert-lm-builder writes a random uuid and the current time into every bundle header
  (google-ai-edge/LiteRT-LM#4064); the converter takes the timestamp from the source commit and the uuid
  from the source, the recipe and the toolchain instead. The export runs under `PYTHONHASHSEED=0`,
  because litert-torch writes the stop tokens in set iteration order (google-ai-edge/litert-torch#1346).
  A lock verifies only on the toolchain that wrote it: when the Python version or a package version
  differs, `verify` stops with `FAIL toolchain` and lists the differences.
- **Checked before it is called done.** The gate: eight short questions with known answers (17 + 25, the
  capital of Japan, …) on the pinned litert-lm CLI, CPU, greedy, scored for the answer and for degenerate
  output; pass = 6 of 8. Refused with a JSON reason, among others: repos gated behind an access request,
  architectures that need the repo's own Python code, pre-quantized weights, repos without full weights.
  A LoRA / PEFT adapter repo is merged into its base and then exported (that route writes no lock yet).
- **A chat template litert-lm 0.18 can render.** litert-lm 0.18 renders chat templates on native values:
  a map has no Python methods, and a message's content is a list of text parts. When the model's template
  does not render the way transformers renders it, the converter edits it (`.get` → indexing, the parts
  → one string) and keeps the edit only if the prompt renders byte-identical to transformers on three
  test conversations (four when the template reads `enable_thinking`). The lock records the edits.
- **One line when it fails:** `FAIL <stage>: <op or first error line> -> next: <hint>`. Exit 0 =
  converted and passed the gate, 1 = the gate or the check failed, 2 = refused or stopped before any
  download, 3 = any later failure.

Rebuilt byte for byte from their locks in a clone, on macOS arm64: `Qwen/Qwen3-1.7B`,
`HuggingFaceTB/SmolLM2-1.7B-Instruct` and `google/gemma-4-E2B-it` (gate 8/8, 7/8 and 8/8). From this
package, `Qwen/Qwen3-1.7B` gave the bytes its reference lock records, and the lock the package wrote
rebuilt to the same bytes in a clone.
The hybrid families that need a patched litert-torch checkout (granite-4.0-h, Falcon-H1, Zamba2,
Nemotron-H) and the hand-made recipes of the repository's published conversions run from a clone of
[the repository](https://github.com/john-rocky/hf-to-litertlm); the package refuses them before any
download and names the clone.

The design, the lock format and how to raise the toolchain:
[docs/one-command-lock.md](https://github.com/john-rocky/hf-to-litertlm/blob/main/docs/one-command-lock.md).
