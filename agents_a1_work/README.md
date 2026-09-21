# Agents-A1-4B: reproduce the text decoder and its checks

Build the two bundles from the same float export, then check the answers on the
runtime you will use. This is the Qwen3.5-4B hybrid recipe applied to
[InternScience/Agents-A1-4B](https://huggingface.co/InternScience/Agents-A1-4B),
**text decoder only (vision tower dropped)**. The checkpoint's `config.json` is
byte-identical to Qwen/Qwen3.5-4B's. Bundles:
[litert-community/Agents-A1-4B](https://huggingface.co/litert-community/Agents-A1-4B).

## Environments and inputs

Run commands from the repository root. Keep conversion, template rendering,
large-model parity and the two runtime versions in separate environments.
The pins below are the measured software versions; package installation and a
full conversion were not repeated when this portable entry was staged.

```bash
python3.10 -m venv .venv-agents-convert
.venv-agents-convert/bin/python -m pip install -r agents_a1_work/requirements-converter.txt
python3.12 -m venv .venv-agents-template
.venv-agents-template/bin/python -m pip install -r agents_a1_work/requirements-template.txt
python3.14 -m venv .venv-agents-parity
.venv-agents-parity/bin/python -m pip install -r agents_a1_work/requirements-parity.txt
python3.14 -m venv .venv-agents-0171
.venv-agents-0171/bin/python -m pip install 'litert-lm==0.17.1'
python3.12 -m venv .venv-agents-nightly
.venv-agents-nightly/bin/python -m pip install 'litert-lm-nightly==0.18.0.dev20260919'

export PY=.venv-agents-convert/bin/python
export TEMPLATE_PYTHON=.venv-agents-template/bin/python
export LT_PY=.venv-agents-parity/bin/python
export RUNTIME_PYTHON=.venv-agents-0171/bin/python
export NIGHTLY_PYTHON=.venv-agents-nightly/bin/python
export PACKAGER=.venv-agents-convert/bin/litert-lm
export LITERT_LM=.venv-agents-0171/bin/litert-lm
export NIGHTLY_LITERT_LM=.venv-agents-nightly/bin/litert-lm
export AGENTS_A1_CHECKPOINT=src_models/Agents-A1-4B
export AGENTS_A1_OUTPUT=out/agents-a1-4b
export LITERT_TORCH_DIR=qwen35_work/litert-torch-qwen35
```

Defaults: scripts use their current interpreter, `litert-lm` on PATH, checkpoint
`src_models/Agents-A1-4B`, and output `out/agents-a1-4b`. Flags override those
values; `--help` lists them. The packager and builder default to the executables
beside the conversion interpreter; `PACKAGER` and `LITERT_LM_BUILDER` override
them. `CONVERTER_PYTHON` selects the build worker; `TEMPLATE_PYTHON` selects
minijinja. Gate programs accept `--model`, `--cli`, `--runtime` and `--out`;
GSM8K additionally accepts `--dataset` or `GSM8K_DATA`.

Downloads are pinned to checkpoint revision
`945c40a4aa6f534d434a353207b8d42ecf7a5293`; the two shards and tokenizer have
SHA256 checks in `export_float.py`. It sets `HF_HUB_DISABLE_XET=1` and keeps
`HF_HOME` under the output directory, with at most four download workers. An
existing partial download can resume. Do not run two runtime processes on the
same bundle: compiled caches are adjacent to that file.

The float bundle is 16,909,161,888 bytes. The measured CPU LiteRT parity stage
used approximately 73 GB peak RSS; run it alone with sufficient memory and
scratch disk. The parity loader must be `CompiledModel` from ai-edge-litert
2.2.0, because the float file is too large for the older Interpreter path.

## Build, in order

After adding the launcher case, the single conversion command is:

```bash
bash scripts/reproduce_llm.sh agents-a1-4b
```

It prepares a dedicated patched checkout, downloads the pinned checkpoint,
exports float once, builds and checks the template, then builds both quantized
files serially. The explicit equivalent is:

```bash
git clone --no-checkout --depth 1 https://github.com/google-ai-edge/litert-torch "$LITERT_TORCH_DIR"
git -C "$LITERT_TORCH_DIR" fetch --depth 1 https://github.com/google-ai-edge/litert-torch 115a13607c730c81018bb9789138a3e5e5119e3d
git -C "$LITERT_TORCH_DIR" checkout --detach 115a13607c730c81018bb9789138a3e5e5119e3d
git -C "$LITERT_TORCH_DIR" apply "$(pwd)/qwen35_work/qwen35_hybrid_litert_torch.patch"
git -C "$LITERT_TORCH_DIR" apply "$(pwd)/agents_a1_work/qwen35_export_compat.patch"
"$PY" -B agents_a1_work/export_float.py --download --model "$AGENTS_A1_CHECKPOINT" --output "$AGENTS_A1_OUTPUT/float" --litert-torch-dir "$LITERT_TORCH_DIR"
"$PY" -B agents_a1_work/build_bundles.py --model "$AGENTS_A1_CHECKPOINT" --float-bundle "$AGENTS_A1_OUTPUT/float/model.litertlm" --output "$AGENTS_A1_OUTPUT"
```

`export_float.py` reproduces **step 1** of the existing
`qwen35_work/convert_qwen35_hybrid.py`: the `litert_torch.cli.main` export entry,
empty quantization recipe, cache 4096, prefill ladder 1024/256/64/16/4/1. It does
not run that driver's older template or int8-only steps. A plain abbreviated
checkout no longer retrieves the required base; use the full fetch-by-hash
command above.

`qwen35_work/qwen35_hybrid_litert_torch.patch` in this repository predates the export measured here:
its no-token-id path lacks the position-derived padding mask used for an
externalized embedder. `qwen35_export_compat.patch` restores that exact branch;
normal text input still uses the same token-id mask. The preparation step
verifies SHA256 values for all seven patched source files against
`patch_state.json`, so the existing patch plus this small supplement exactly
matches the executed source. No shared patch file is replaced.

`build_bundles.py` reuses `minicpm5_work/quantize_minicpm5.py`,
`scripts/add_executor_metadata.py` and `scripts/set_activation_type.py` from
this repository. It passes explicit interpreter/CLI paths to those helpers.
It runs HF/minijinja render checks before packing, preserves structured prompt
metadata, declares stops 248044 and 248046, TOP_P k20/p0.95/temperature0.85,
cache 4096, generic model type and the `thought` channel, then adds state-buffer
bindings and fp32 activations. Presence penalty has no runtime equivalent.
Temporary build files are removed after each verified final file; the float
parent and checkpoint are retained for parity and inspection.

| File | Recipe | Measured bytes |
|---|---|---:|
| Agents-A1-4B_int8.litertlm | wi8fc: dynamic int8 linears + embedding; float convs/delta rule; fp32 activations | 4,407,426,400 |
| Agents-A1-4B_mixed_int4.litertlm | wi4b32_wi8 min-max: int4 block-32 linears, int8 channelwise embedding + lm_head; fp32 activations | 2,754,365,536 |

Both have one physical int8 vocabulary table of shape [248320, 2560], with no
`_duplicated_` tensors. A [2560, 9216] linear has 2,560 scales in int8 and 737,280
scales in mixed int4. A changed size/layout is recorded for investigation;
there is no recipe search or automatic tuning.

## Template check

`resources/qwen3_5_801acdef.jinja` is the pinned canonical source, so template
construction needs no runtime-source checkout. The generated template must be
byte-identical to `chat_template_agents_a1.jinja` in this directory. The vendor
default system block is copied verbatim, including `Current date: 2026-07-14`;
an explicit system message replaces it. String content and content parts are
both supported. Assistant history renders independently of its position, and
thinking defaults on with the `thought` channel. Native XML tool calls reach
the prompt and output, but the tested runtimes do not turn them into tool-call
events; an application parses the block with `parse_xml_tool_call.py`.

The build runs these commands automatically; they can be run separately:

```bash
"$PY" -B agents_a1_work/build_chat_template.py --model "$AGENTS_A1_CHECKPOINT" --out "$AGENTS_A1_OUTPUT/chat_template_agents_a1.jinja" --report "$AGENTS_A1_OUTPUT/results/template_source.json"
"$PY" -B agents_a1_work/render_check.py hf --model "$AGENTS_A1_CHECKPOINT" --template "$AGENTS_A1_OUTPUT/chat_template_agents_a1.jinja" --output "$AGENTS_A1_OUTPUT/results"
"$TEMPLATE_PYTHON" -B agents_a1_work/render_check.py minijinja --model "$AGENTS_A1_CHECKPOINT" --template "$AGENTS_A1_OUTPUT/chat_template_agents_a1.jinja" --output "$AGENTS_A1_OUTPUT/results"
```

Cases (i)–(iv) must be byte-identical to HF; all four prefix-contract variants
must pass. Cases (v)/(vi) may differ only in JSON serialization inside the tool
JSON payloads; literal diffs are saved. The recorded run met that tolerance.

## Gates, in order

First download the fixed GSM8K test set and check float CPU logits. No HF
`generate()` is used here; if adding one, pass `eos_token_id=[248044, 248046]`.

```bash
"$PY" -B agents_a1_work/download_gsm8k.py
"$PACKAGER" unpack "$AGENTS_A1_OUTPUT/float/model.litertlm" --output-dir "$AGENTS_A1_OUTPUT/float/unpacked"
"$PY" -B agents_a1_work/parity.py pt --hf "$AGENTS_A1_CHECKPOINT" --out "$AGENTS_A1_OUTPUT/results/parity_pt.npz"
"$LT_PY" -B agents_a1_work/parity.py lt --tflite "$AGENTS_A1_OUTPUT/float/unpacked/"*TFLiteModel*.tflite --ids "$AGENTS_A1_OUTPUT/results/parity_pt.npz" --out "$AGENTS_A1_OUTPUT/results/parity_lt.npz"
"$PY" -B agents_a1_work/parity.py cmp --pt "$AGENTS_A1_OUTPUT/results/parity_pt.npz" --lt "$AGENTS_A1_OUTPUT/results/parity_lt.npz" --out "$AGENTS_A1_OUTPUT/results/parity_cmp.json"
```

Parity acceptance: all logits finite, top-1 48/48, top-5 48/48, minimum
per-position Pearson ≥0.9999, mean KL ≤0.001 nats. Top-5 means LiteRT's top-1
is in HF's top-5. Measured: 48/48 and 48/48 (top-5 sets also identical), minimum
Pearson 0.999999999862, mean KL −1.64e−8 nats (floating-point roundoff), maximum
absolute logit difference 0.000221253. `parity.py` imports the existing
`scripts/parity_logits_bigmodel.py`; only dataset selection and the machine
readable acceptance report are added.

The runtime matrix is sequential. Each question/length gets one process,
closed stdin, cache off, temperature 0, seed 0 and thinking disabled.

```bash
for runtime in 0.17.1 nightly; do
  cli="$LITERT_LM"
  if [ "$runtime" = nightly ]; then cli="$NIGHTLY_LITERT_LM"; fi
  for variant in int8 mixed_int4; do
    for backend in cpu gpu; do
      "$PY" -B agents_a1_work/gate8q.py --model "$AGENTS_A1_OUTPUT/Agents-A1-4B_${variant}.litertlm" --cli "$cli" --runtime "$runtime" --backend "$backend" --out "$AGENTS_A1_OUTPUT/results/gate8q_${variant}_${backend}_${runtime}.json"
    done
  done
done
for variant in int8 mixed_int4; do
  for backend in cpu gpu; do
    "$PY" -B agents_a1_work/banana_hermetic_sweep.py --model "$AGENTS_A1_OUTPUT/Agents-A1-4B_${variant}.litertlm" --cli "$LITERT_LM" --runtime 0.17.1 --backend "$backend" --out "$AGENTS_A1_OUTPUT/results/banana_${variant}_${backend}.json"
  done
done
```

Read every 8Q answer before closing its result: a fluent off-prompt answer
fails even when a regex matches. After inspection, record the judgment with
`gate8q.py --review <result.json> --on-topic yes` (or `no`). The automatic
verdict alone remains `REVIEW_REQUIRED`. Acceptance and the recorded result:
**8/8 with zero degenerate answers in all eight cells** (0.17.1 and
0.18.0.dev20260919); BANANA **40/40 CPU fills 12–51 and 20/20 GPU fills 12–31
per file**, containing BANANA in fewer than 200 characters.

Thinking and conversation checks use int8/CPU. Run each API probe with that
runtime's own Python:

```bash
for runtime in 0.17.1 nightly; do
  cli="$LITERT_LM"; runtime_py="$RUNTIME_PYTHON"
  if [ "$runtime" = nightly ]; then cli="$NIGHTLY_LITERT_LM"; runtime_py="$NIGHTLY_PYTHON"; fi
  "$PY" -B agents_a1_work/think_probe.py --cli "$cli" --runtime "$runtime" --model "$AGENTS_A1_OUTPUT/Agents-A1-4B_int8.litertlm" --out "$AGENTS_A1_OUTPUT/results/think_probe_${runtime}.json"
  for mode in off on; do
    "$runtime_py" -B agents_a1_work/multiturn_probe.py --runtime "$runtime" --model "$AGENTS_A1_OUTPUT/Agents-A1-4B_int8.litertlm" --mode "$mode" --out "$AGENTS_A1_OUTPUT/results/multiturn_${runtime}_${mode}.json"
  done
done
"$PY" -B agents_a1_work/run_tools.py --bundles "$AGENTS_A1_OUTPUT" --cli "$LITERT_LM" --nightly-cli "$NIGHTLY_LITERT_LM" --runtime-python "$RUNTIME_PYTHON" --template-python "$TEMPLATE_PYTHON" --output "$AGENTS_A1_OUTPUT/results/tools_xml"
```

Acceptance: reasoning has channel markers and no literal think tags leak into
the answer; budget 64 has less reasoning and still answers 42/Tokyo. The
three-turn conversation recalls Osaka then Ken without an exception or garbage,
with thinking both off and on. Both runtimes passed. All **12/12 XML tool
requests** emitted exactly one expected call, plausible arguments and no suffix;
the preset uses deterministic fixtures, with no live weather/search I/O.
Inspect multi-turn text for garbage as well as the recorded checks.

Then run the original GSM8K protocol: first 100 questions, int8 then mixed
int4, GPU 0.17.1, one process/question, thinking off, greedy, seed 0. Disk cache
is permitted only after that exact file passes its cache-off GPU gates. There
is no decode-token cap; EOS or the 4096-token context ends generation.

```bash
for variant in int8 mixed_int4; do
  "$PY" -B agents_a1_work/gsm8k_litertlm.py --variant "$variant" --model "$AGENTS_A1_OUTPUT/Agents-A1-4B_${variant}.litertlm" --cli "$LITERT_LM" --gate8q "$AGENTS_A1_OUTPUT/results/gate8q_${variant}_gpu_0.17.1.json" --sweep "$AGENTS_A1_OUTPUT/results/banana_${variant}_gpu.json" --out "$AGENTS_A1_OUTPUT/results/gsm8k_${variant}.json"
done
"$PY" -B agents_a1_work/rescore_gsm8k.py --int8 "$AGENTS_A1_OUTPUT/results/gsm8k_int8.json" --mixed-int4 "$AGENTS_A1_OUTPUT/results/gsm8k_mixed_int4.json" --out "$AGENTS_A1_OUTPUT/results/gsm8k_ab_corrected.json" --original-out "$AGENTS_A1_OUTPUT/results/gsm8k_ab.json"
```

Raw text, answer lengths and native context occupancy are saved per question;
progress prints every ten questions. `--resume` appends missing rows without
rerunning completed questions. Newly created cache files are deleted after
each variant and listed in the result; pre-existing files are preserved.
Measured: **93/100 int8 and 93/100 mixed int4** under the original extractor,
with **1 and 2 context-limit answers**. Numeric normalization gives **94/100
and 94/100** from the same saved text. The original `rstrip(".0")` removes
characters rather than a decimal suffix (`20.00` becomes `2`); corrected
extraction retains the same order: `####`, boxed value, answer phrase, last
number. Keep both score columns; the corrected script never invokes a model.
There is no tuned score target for another run: report its measured counts.

## Optional Mac benchmark

Reserve the GPU on the host before timing; `--lock` can name its agreed lock
file. The default is a lock under the output directory. The script rests at
least 300 seconds before each GPU cell, checks load <3.0 and matching compute
processes before every cell, waits up to 20 minutes in 60-second polls, and
then labels a remaining busy cell contended. `--busy-pattern` can cover other
compute programs on the host. It never terminates them.

```bash
"$PY" -B agents_a1_work/bench_mac.py --bundles "$AGENTS_A1_OUTPUT" --cli "$LITERT_LM" --device 'Apple M4 Max' --out "$AGENTS_A1_OUTPUT/results/bench_mac.json"
```

Protocol: `benchmark -p 256 -d 256 --runs 3 --cache no`, default warmup, all three
measured returns retained, arithmetic means. Order: int8 GPU, mixed int4 GPU,
int8 CPU, mixed int4 CPU. Unexpected newly generated caches are recorded and
removed. The following are observations, not performance thresholds:

| File / backend | Prefill tok/s | Decode tok/s | TTFT s |
|---|---:|---:|---:|
| int8 / GPU | 672 | 61.2 | 0.40 |
| mixed int4 / GPU | 676 | 67.2 | 0.39 |
| int8 / CPU | 255 | 20.6 | 1.05 |
| mixed int4 / CPU | 99.1 | 20.2 | 2.63 |

These rows use Apple M4 Max and litert-lm 0.17.1. All four rows were taken on a quiet machine (1-minute load under 3.0, at least 300 s of rest before a GPU cell); three of them are a second pass that agreed within 2 % with a first pass taken on a busier machine. Source numbers and artifact fingerprints are in
[measured_results.json](measured_results.json).

## Phone check

Phone check (Galaxy S26, SM8850, Android 16, LiteRT-LM Android CLI build of 2026-09-18; the same 8 questions with thinking off, one fresh session per question, the prompt rendered from the bundled template): mixed int4 **8/8 on CPU** (peak 4.94 GB; 47.1 prefill / 11.9 decode tok/s, TTFT 5.52 s at 256/256, thermal status 0, frequencies uncapped) and **8/8 on the OpenCL GPU** with every subgraph delegated (peak 4.16 GB; speed not measured). With thinking on, the final answer arrives after a 350–400-token thought. The int8 file is a desktop-class file: on this phone its CPU answers are correct but 2 of 8 run on into repetitive self-talk, the thinking-on probe gives no final answer, and starting its GPU engine took the phone off USB — the published card tells Android users to take the mixed int4 file. The Android CLI has no thinking switch and no sampler flags, so a phone gate that leaves thinking on needs a token cap in the thousands and a watchdog in minutes, not seconds.

## Script map

| File | Role |
|---|---|
| export_float.py | Pinned checkout/download and float export only |
| qwen35_export_compat.patch, patch_state.json | Small source-parity supplement and hashes of all seven patched files |
| build_bundles.py | Both recipes, metadata, fp32 declaration, final verification |
| build_chat_template.py, chat_template_agents_a1.jinja | Canonical dual-form template plus exact vendor system block |
| render_check.py | HF/minijinja byte comparisons and prefix contracts |
| gate8q.py, banana_hermetic_sweep.py | Fixed quality questions and fresh-engine length sweep |
| think_probe.py, multiturn_probe.py | Thought channels, budgets and one-conversation history |
| tools_preset.py, parse_xml_tool_call.py, run_tools.py | Fixtures, app-side XML parsing and twelve tool probes |
| gsm8k_litertlm.py, rescore_gsm8k.py | Original scoring, saved answers and corrected paired analysis |
| download_gsm8k.py, parity.py | Dataset fingerprint and existing big-model parity integration |
| bench_mac.py | Optional Mac benchmark with explicit contention labels |
| bundle_header.py, inspect_layout.py | Metadata sections, physical vocab buffer and quantization layout |
| runtime_helpers.py | Serial execution, closed stdin and raw evidence |
| gsm_observer/sitecustomize.py, bench_observer/sitecustomize.py | Opt-in observation of unchanged runtime return values |
