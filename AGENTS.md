# AGENTS.md — for coding agents converting a Hugging Face model to LiteRT-LM

Read this first if someone asked you to "convert *some model* to LiteRT / `.litertlm`", "run a
Hugging Face model on Android or iPhone with LiteRT-LM", or "rebuild one of the published
bundles". This file routes; the README has the commands.

**What this is.** A converter from open-weight Hugging Face LLMs and VLMs to `.litertlm` bundles
for Google's [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) runtime, plus the recipe
and the measurement record for 84 published conversions (78 in
[litert-community](https://huggingface.co/litert-community) on Hugging Face, 6 under
[mlboydaisuke](https://huggingface.co/mlboydaisuke)). Independent converter, not affiliated
with Google.

## Route

| Task | Do | Where |
|---|---|---|
| Check whether the model is already converted | Look it up in the model table; each row links the Hugging Face repo and its recipe | README "Converted models" |
| Convert a finetune, or any supported model, to a gated bundle | `python scripts/convert.py <org>/<model>` (`--int4` for the proven int4 recipe) | README "Convert a finetune" |
| Convert with the pinned toolchain and record a lock (Hub commit, sha256 of every source file, the export call, every package version, the bundle's section sha256) | `uv sync`, then `uv run python scripts/convert.py <org>/<model> [--revision <sha>] [--recipe int8\|int4]` → `out/<model>/<model>_<recipe>.litertlm` + `<model>.lock.json` (or, without a clone, `uvx hf-to-litertlm <org>/<model>` from PyPI: the same scripts and the `uv.lock` of the commit it was built from, output under `./out`); a failure ends in one `FAIL <stage>: … -> next: …` line | [docs/one-command-lock.md](docs/one-command-lock.md) |
| Check that a lock gives the same bytes, or raise the toolchain | `uv run python scripts/verify_lock.py <lock.json>` (or `uvx hf-to-litertlm verify <lock.json>` from PyPI) rebuilds into a fresh dir: MATCH, or DIFF per section; after raising the pins in `pyproject.toml` and `uv lock`, `bash scripts/verify_refs.sh` runs the three reference models twice each into `locks/refs/` | [docs/one-command-lock.md](docs/one-command-lock.md) "Raising the toolchain", `locks/refs/CHANGELOG.md` |
| Convert a finetune of a model this repo converts with a family script (a Qwen3.5-9B finetune on `qwen35_work/convert_qwen35_9b.sh`) | Before running the base's script with the finetune's id and revision, diff the text config key by key (`jq -S .text_config` of both config.json files: every shared key equal, or one the export sets itself), then generation_config (each eos id becomes a stop token), the checkpoint's key list, `tokenizer.json` (rule 7) and the chat template (rule 5); a difference that reaches the graph means the family script is not the recipe | REPRODUCE.md "Ornith-1.0-9B" |
| Run the bundle on a Mac to check it | `litert-lm run out/<model>/*.litertlm --prompt "Hello"` | README top |
| Read a Mac GPU (WebGPU) benchmark or gate log | Count `Validation error` and `Shape mismatch` lines in the whole stderr; a process with any is invalid even when it exits 0 and prints rates. `litert-lm benchmark` builds a fresh Engine for every run, the warmup included; for Qwen3.8-27B on litert-lm 0.18.1 (Mac M4 Max) the errors came only in the third engine of a process (2 of 44 engines in 11 benchmark processes, none in 130 single-engine processes). The CLI prints only ERROR lines unless `--verbose` is given, and these are ERROR lines | REPRODUCE.md "Qwen3.8-27B" |
| Rebuild a published bundle | `bash scripts/reproduce_llm.sh <key>` or `bash scripts/reproduce_vlm.sh <key>` (`--list` for keys) | [REPRODUCE.md](REPRODUCE.md) |
| Re-pack a published bundle so its weights sit in TFLiteWeights sections at the end of the file (LiteRT-LM Web form) | `bash tools/externalize/fetch_tools.sh`, then `bash tools/externalize/externalize_repo.sh <org>/<repo> <file.litertlm> --revision <sha>`; read the staged REPORT (dtypes, graph size, smoke) before any upload | [tools/externalize/README.md](tools/externalize/README.md), [REPRODUCE.md](REPRODUCE.md) "External weights" |
| Check that a bundle's chat template survives the second message (the runtime re-renders the history before every message and requires the new render to extend the previous one) | `python tools/chat_template_prefix/desk_prefix.py table <template.jinja> tools/chat_template_prefix/examples/conv_three_turns.json` — every cell PASS; fix the template and re-pack with `scripts/set_prompt_template.py`; then `python tools/chat_template_parts/two_turn_increment.py <template.jinja> <tokenizer.json>` — the cache after the second message must equal the upstream two-turn ids (the newline after the stop token) | [tools/chat_template_prefix/](tools/chat_template_prefix/) |
| Check that a bundle's tokenizer is the model's own | Compare the bundle's tokenizer section with the repo's `tokenizer.json` (sha256 of the unpacked section, at least normalizer + pre-tokenizer); fix a built bundle with `python tools/tokenizer_parity/swap_tokenizer_section.py in.litertlm tokenizer.json out.litertlm` | [tools/tokenizer_parity/](tools/tokenizer_parity/) |
| A bundle returns tool calls as plain text, or answers without the tool result | Re-pack metadata only: the family's `llm_model_type` + the bundle's own template + the official template's tool-call / tool-response branch; desk-render a tool round trip; mind rule 9 (GPU defaults) and rule 10 (floor) | `scripts/set_prompt_template.py`, `scripts/section_identity.py`, [tools/chat_template_parts/](tools/chat_template_parts/) |
| Run a published bundle in an Android app | [hfmodels-android](https://github.com/john-rocky/hfmodels-android): one Gradle line, a few lines of Kotlin; the walkthrough pins LFM2.5-1.2B-Instruct on a Pixel 8a | hfmodels-android README "Add it" |
| Run a bundle in an iPhone app | [swift-litert-lm](https://github.com/john-rocky/swift-litert-lm): `LiteRTChat(huggingFaceRepo:fileName:)` or `LiteRTChat(modelFileURL:)` | [recipe: a fine-tuned model on iPhone](https://github.com/john-rocky/swift-litert-lm/blob/main/docs/recipe-hf-finetune-to-iphone.md) |
| A hybrid family no released exporter handles (granite-4.0-h, Falcon-H1, Zamba2, Nemotron-H) | `convert.py` routes to the pinned family recipe by `model_type`; the one-time checkout command is printed on refusal | README "Convert a finetune" table, `<family>_work/` |
| Plan the phone rows of a 7–9B hybrid (Mamba2, GatedDeltaNet) | The int4 GPU engine of such a file does not build on a 12 GB Galaxy S26 or on a Pixel 8a; measure the Mac rows and say so on the card | [docs/android-gpu.md](docs/android-gpu.md) "7–9B models on a 12 GB phone" |
| A Mamba2-hybrid bundle scores above its transformers reference on CPU / MPS | Split the score at the config's `chunk_size` and re-run the long prompts with `chunk_size` ≥ the longest prompt (the naive scan loses context across chunks); take the dt arithmetic from the vendor's modeling code, not transformers' torch path | REPRODUCE.md "Nemotron-H" |
| A thinking switch on a Qwen3.5-family hybrid | `qwen35_work/use_switch_template.sh` re-packs the template; gate thinking on with `thinking_token_budget=300` over three fresh CPU processes (without a budget a CPU thought can run past 2,048 tokens a turn) and in a three-turn conversation at the runtime's default conversation settings that asks at turn 3 for a fact given at turn 1 (three `send_message` calls on one `create_conversation()` without a `filter_channel_content_from_kv_cache` argument, e.g. "My name is Kenji." / "What is 17 times 23?" / "What is my name?"); if that fails, ship thinking off only (`qwen35_work/use_noswitch_template.sh`). By default litert-lm 0.18.1 takes the thought out of the KV cache before the next turn (`filter_channel_content_from_kv_cache`), and Qwen3.8-27B then lost the user's name at turn 3 in 3 of 3 GPU processes and 1 of 1 CPU process; it kept the name with the filter off and as the bf16 model in transformers (one run each; Mac M4 Max, 2026-10-22 and 23). The published Qwen3.5-9B and 4B files kept the name in the turn-3 reply at the default in 30 of 30 processes (CPU and GPU, 2026-10-24; 29 answered it, one 4B int8 CPU run repeated the turns to the 2,048-token cap), so the check is per file | REPRODUCE.md "Qwen3.5-9B", "Qwen3.8-27B" |
| A dense architecture that is not listed | `scripts/export_simple_template.py`, then a `case` in `scripts/reproduce_llm.sh` | README "Convert a new architecture" |
| Export a dense 7–9B model on a machine with about 128 GB of RAM (the default path reached 98.8 GB in the converter passes for granite-4.1-8b) | `granite41_work/export_lightweight.py` = `scripts/export_simple_template.py` with `experimental_lightweight_conversion` (`LIGHTWEIGHT=1`) and `RESOURCE_CONSTANT_NUMEL_THRESHOLD=100000000` (bytes: above the weights the converter folds with a scale, below the MLP weights); accept only with no constant-only op and every FULLY_CONNECTED weight an int8 / int4 constant | REPRODUCE.md "granite-4.1-8b", "Ministral-3-8B-Instruct-2512" |
| Export and quantize a model whose float32 weights nearly fill the machine's memory (Qwen3.8-27B: 111 GB in float32 on a 128 GiB Mac) | `bash qwen35_work/convert_qwen38_27b.sh`: `upcast_fp32.py` writes an F32 copy of the pinned checkpoint, which the export reads through mmap instead of converting bf16 in RAM; `convert_qwen38_hybrid_lw.py` replaces litert-torch's in-place `_clamp_inf_values` with a read-only ±inf check; `make_int4_27b.py` quantizes with `serialize_to_path`. Peak memory footprint: 11.8 GB for the export, 46.2 GB for the int4 build | REPRODUCE.md "Qwen3.8-27B" |
| Faster GPU decode for a dense bundle without re-exporting (KV writes and attention as GPU composites) | `python tools/gpu_graph/gpu_graph_retrofit.py in.litertlm out.litertlm --decomp exporter --mask add_bcast --report r.json` (weights and the other sections byte-identical); check the CPU answers against the input file and the GPU answers, then time both files in both orders on each device | [tools/gpu_graph/README.md](tools/gpu_graph/README.md), REPRODUCE.md "granite-4.1-8b" |
| A blockwise int4 bundle the Mac CPU refuses (XNNPACK prepares a blockwise int4 FULLY_CONNECTED only when every fp16 block scale is a normal float) | Floor the block scales at 2^-14: at export, `olmo3_work/export_clamped_lw.py` (the clamp-before-quantize monkeypatch on the lightweight path), or at quantization, `falcon_h1_work/quantize_clamped.py`; then `granite_work/scan_scales.py` must report zero 0 and subnormal 0. Which models produce such scales is not predictable: Olmo-3-7B-Think wrote 3,606,703 of 215 M, granite-4.1-8b none | REPRODUCE.md "Olmo-3-7B-Think", "Falcon-H1R-7B" |
| A rebuilt bundle whose TFLite section differs from the published one | Compare sections first; when a graph section differs, `python tools/tfl_compare/tfl_struct.py A B SECTION out.json` compares operators and constants without tensor indices or names (run a must-differ control beside it). An export can store the RoPE constants and the signature subgraphs' operators in either of two orders: the same computation, a different sha256 | [tools/tfl_compare/README.md](tools/tfl_compare/README.md), REPRODUCE.md "Olmo-3-7B-Think" |
| A new VLM | Copy the closest `scripts/ship_*.sh` with its `convert_*_vision.py` / `prep_*_decoder.py` | README "Convert a new architecture", [docs/vision-tower-reauthor.md](docs/vision-tower-reauthor.md) (the vision tower in plain torch for the fast_vlm contract) |
| A Qwen3-VL-family VLM with DeepStack (vision features added to the decoder after layers 0–2) and M-RoPE, whose answers are coordinates (boxes, points, OCR boxes) | Keep DeepStack exact in a wide image embedding: the adapter emits the main embedding, the DeepStack features and an image flag per token (8,196 values for a 2,048-wide decoder), the decoder adds each part after its layer and computes M-RoPE in the graph from the 1-D position and the flag (`wevisone_work/tools/wv_patch.py`, `scripts/ship_wevisone.sh`). Adding the features into one embedding changed WeVisOne-VL-2B's coordinate answers (boxes 41/55 against the unchanged model); text-only conversations then stay exact up to P + N − 1 tokens | [docs/deepstack-wide-embedding.md](docs/deepstack-wide-embedding.md), REPRODUCE.md "WeVisOne-VL-2B" |
| A bundle with an EMBEDDER section (every fast_vlm VLM, every `externalize_embedder=True` export) answers a conversation's second message differently from the model | LiteRT-LM (v0.16.0 to main of 2026-10-23) passes the previous answer's stop token with a zero embedding; `wevisone_work/tools/r6_twoturn_probe.py` replays the file's graphs with that row embedded and zeroed and compares with the runtime; ask one question per conversation | [docs/deepstack-wide-embedding.md](docs/deepstack-wide-embedding.md) "A runtime defect" |
| A raw graph that is right at GPU fp32 and collapses at fp16 or on the NPU (RMSNorm sums past 65,504) | Rewrite each norm with a per-site power-of-two rescale (exact in fp32), clamp the rsqrt for padding rows, keep masks float | [docs/fp16-safe-norms.md](docs/fp16-safe-norms.md) |
| A raw-graph pipeline too big to stay resident on the phone (a 12.9 GB text-to-image model on a Galaxy S26 or a Pixel 8a) | Load, run and close one graph at a time with one shared Environment and a MemAvailable guard between graphs; do not enable the GPU program cache on every graph (it stores the weights too) | [docs/android-gpu.md](docs/android-gpu.md) "A pipeline larger than the phone's memory" |
| A VLM whose text tower is a model the LLM rail already exports (Qwen3.5 text under MiniCPM-V-4.7) | `python tools/vlm_text_ckpt/make_text_ckpt.py <snapshot> <out> --prefix model.language_model. --model-type qwen3_5_text --architectures Qwen3_5ForCausalLM --prove`, then the LLM rail on the derived directory | [tools/vlm_text_ckpt/README.md](tools/vlm_text_ckpt/README.md) |
| A Ministral 3 checkpoint (Mistral3ForConditionalGeneration, text only) | `python scripts/extract_ministral3_text.py <snapshot> <out>` (the decoder and its lm_head as a standalone Ministral3ForCausalLM; an 8B tower is written in fp32, so it needs the RAM of an export), then the dense rail with `ministral_work/ministral3_8b_dual.jinja` (string and content-parts messages, the vendor system message, no BOS in the template) | REPRODUCE.md "Ministral-3-8B-Instruct-2512" |
| Gate a VLM bundle's image answers against the model's own fp32 outputs | `python minicpmv47_work/suite_mv47.py run <bundle> --label <name>` (one process per image question, keyword grounding) then `score <suite dir>` (first divergence against the reference arms) | REPRODUCE.md "2026-10-13 — MiniCPM-V-4.7-1B" |
| Which file fits which device, backend and RAM budget, at what speed | `litertlm_manifest.json` in every published repo; schema, generator and readers in this repo | README "Deployment manifests", [manifest/SCHEMA.md](manifest/SCHEMA.md), [readers/](readers/) |
| GPU or NPU on Android for these bundles | Measured notes | [docs/android-gpu.md](docs/android-gpu.md), [docs/android-npu.md](docs/android-npu.md) |
| Check a converted graph on an Android phone (parity rows, ms, memory) and keep the record | Run `bash tools/devicebench/dryrun.sh <lane.env>` on a fake phone first, then `bash tools/devicebench/chain.sh check\|run <round> <lane.env>` | [skills/android-device-gate/SKILL.md](skills/android-device-gate/SKILL.md) (the steps, on a public example), [tools/devicebench/README.md](tools/devicebench/README.md) |
| Ask for a model to be converted | Open a model request with the Hugging Face link; that is the whole ask | [CONTRIBUTING.md](CONTRIBUTING.md) |

## Rules

1. **A bundle that loads is not done.** The exit gate decides: `scripts/verify_quality.py`, 8 fixed
   questions, bar 6/8. `convert.py` exits 0 = converted and gated, 1 = converted but the gate or
   the one-prompt check (`scripts/check_bundle.py`) failed, 2 = refused or stopped before any
   download, 3 = any later failure. Report the
   exit code and `out/<model>/convert_report.json`, not "it ran".
2. **A refusal is an answer.** Gated, remote-code and pre-quantized repos are refused with a JSON
   reason before anything downloads. Quote the reason; do not work around the gate.
3. **Pin the toolchain from README "Setup".** `transformers==5.14.*`; Qwen3.5 exports only on
   litert-torch `main` (the released 0.9.4 output is degenerate). When the stack is wrong,
   `convert.py` prints the exact install command.
4. **Numbers carry a device, a backend and a date.** The table's speeds are read from each repo's
   `litertlm_manifest.json`. A number without those three is not a measurement; do not extrapolate
   a Galaxy S26 GPU figure to another phone.
5. **The derivative's own chat template is embedded verbatim.** Do not substitute the base model's
   template; the adapter's tokenizer, template and generation config win over the base's.
   When the bundle carries a thinking-off form instead (Ornith-1.0-9B ships the 701-byte `qwen35_text_current.jinja`),
   render it the way the runtime does (`tools/chat_template_prefix/desk_render.py`) and render the derivative's own
   template with transformers' `apply_chat_template(..., enable_thinking=False)`; ship it only if the two are
   byte-identical in every conversation shape. A family template that matches the base but not the derivative (the
   Qwen3.5-4B switch template drops the think block Ornith writes into past assistant turns) is a substitution.
6. **Model files are not in this repository.** Bundles live on Hugging Face; `out/` is local.
7. **The tokenizer section is the repo's own `tokenizer.json`.** Bundle the file from the Hub at a
   pinned revision and compare the unpacked section with it by sha256. `AutoTokenizer.save_pretrained`
   rewrites `tokenizer.json` when `tokenizer_config.json` names a dedicated class (transformers'
   `CohereTokenizer` splits every digit; the trained tokenizer groups digits in threes), and an HF
   reference loaded the same way shares the defect. The litert-torch Qwen3.5 rail writes transformers'
   `Qwen2Tokenizer` pipeline the same way (a pre-tokenizer without the `\p{M}` class: Hindi, Thai, Tamil
   and Arabic with harakat tokenize differently); `tools/tokenizer_parity/swap_tokenizer_section.py`
   replaces that section with the repo file and leaves the other sections byte-identical. A section that
   differs from the repo file only in `post_processor` (an identity TemplateProcessing where the repo has
   none) gave the same ids on 15 probe texts with and without special tokens (Falcon-H1R-7B).
8. **A tool-calling model is not packed as `generic_model`.** The runtime chooses the tool-call parser
   by `llm_model_type` (`lfm2` for `<|tool_call_start|>` calls from litert-lm 0.18.0, `qwen2p5` / `qwen3`
   for JSON in `<tool_call>`); `generic_model` returns every call as text. The template must also render
   the tool result (0.18 passes it as a `tool_response` content part): desk-render a tool round trip
   and find the result in the second turn's prompt.
9. **A non-generic `llm_model_type` changes three GPU-executor defaults** that no bundle field or public
   API overrides. Re-measure the phone GPU after a re-pack that changes the type; see
   [docs/android-gpu.md](docs/android-gpu.md).
10. **The official LiteRT-LM chat templates read content parts only.** A bundle re-packed with them
   (`scripts/repack_official_metadata.sh`) needs litert-lm 0.18.0 or newer: 0.17.x sends string content
   and the user turn renders empty, exit 0. Keep the bundle's own template and add only the official
   tool branch when an older runtime must keep working.
11. **A hybrid bundle (Mamba2, GatedDeltaNet) declares fp32 GPU activations.** With the GPU executor's
   default fp16 activations the hybrid conversions of this repository answered nothing (the pad token,
   `!` or `<unk>` repeated from the first token: Falcon-H1R-7B, Qwen3.5-9B, Ornith-1.0-9B, Ornith-1.5-9B,
   NVIDIA-Nemotron-Nano-9B-v2); `scripts/set_activation_type.py --type fp32` fixes it at 1–8 % of the
   GPU time (Falcon-H1R-7B 1 %, Nemotron 1.5 %, Qwen3.5-9B 8 %). Dense conversions keep fp16: the
   declaration costs them about 15 % (granite-4.1-8b 15.4 %, Ministral-3-8B 14.7 %) and their answers
   do not change.

## Not this repo

- Single-graph `.tflite` models for Android (vision, audio, TTS) → [LiteRT-Models](https://github.com/john-rocky/LiteRT-Models).
- Apple's Core AI runtime (`.aimodel`) → [coreai-model-zoo](https://github.com/john-rocky/coreai-model-zoo).
- Whether a `.tflite` op runs on a given delegate, measured → [edge-compat](https://github.com/john-rocky/edge-compat).

Maintainer: john-rocky (GitHub), mlboydaisuke (Hugging Face). Issues: https://github.com/john-rocky/hf-to-litertlm/issues
