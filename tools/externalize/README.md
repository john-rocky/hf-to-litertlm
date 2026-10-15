# externalize — move a `.litertlm` file's TFLite weights into TFLiteWeights sections, and check the result

`externalize_repo.sh` takes one `.litertlm` file from a Hugging Face model repo and builds the same file with the weights of its TFLiteModel sections moved into TFLiteWeights sections at the end of the container. It uses LiteRT's own tools for the move (`litert/tools/externalize_tflite_flatbuffer.py` and `externalize_litertlm_sections.py`), copies every other section (metadata, chat template, tokenizer) byte for byte, checks each TFLite section tensor by tensor against the original, runs one CPU prompt through the result, and stages the file with a report.

LiteRT-LM pairs a TFLiteWeights section with the TFLiteModel section of the same model type when it loads a file (`runtime/util/litert_lm_loader.cc`) and hands it to the compiled model as a weight file named `tflite_weights`. LiteRT's guide for large models describes these external weights (`docs/instructions/LARGE_TFLITE_MODELS.md` in google-ai-edge/LiteRT).

## Requirements

- bash 3.2 or newer, curl, perl, awk.
- Python 3.11 or newer with `ai-edge-litert` (its `schema_py_generated` module must carry the ExternalBuffer types; 2.1.6 to 2.4.0.dev20261011 do), `flatbuffers`, `numpy` and `litert-lm-builder` 0.18.0 (it brings `litert-lm-peek` and protobuf).
- The `litert-lm` CLI 0.18.0 (unpack and the smoke run) and the `hf` CLI from `huggingface_hub` (download; not needed with `--source`).

Checked on macOS 27 (arm64) with Python 3.12.13, ai-edge-litert-nightly 2.4.0.dev20261011, flatbuffers 25.12.19, numpy 2.5.4, protobuf 7.36.2, litert-lm-builder 0.18.0, litert-lm 0.18.0 and huggingface_hub 2.2.0 (Qwen3-0.6B and LFM2.5-VL-1.6B files), and with the install line below in a fresh Python 3.14.6 environment (ai-edge-litert 2.3.0 and the same versions of the rest; a small test bundle).

## Run

From the repository root:

```sh
pip install ai-edge-litert flatbuffers numpy litert-lm==0.18.0 litert-lm-builder==0.18.0 huggingface_hub
bash tools/externalize/fetch_tools.sh
export XR_TOOLS_PATH=$HOME/.cache/litert-externalize/c5c0d28f27abddec81bcc14bbd05bdb84eb5c556
bash tools/externalize/externalize_repo.sh litert-community/Qwen3-0.6B Qwen3-0.6B_dynamic_wi4b32_afp32.litertlm --revision a3c5d805
```

`fetch_tools.sh` downloads the two LiteRT tools at commit `c5c0d28f` into `~/.cache/litert-externalize/<commit>/` (sha256 checked against the pins in the script) and adds a one-line module, `litert/python/schema_py_generated.py` = `from ai_edge_litert.schema_py_generated import *`: the tools import LiteRT's in-repo flatbuffer bindings, and the ai-edge-litert wheel ships the same generated module, so no flatc build is needed.

For a vision bundle, add `--smoke-attachment some.png` so the smoke run asks about an image. `--source PATH` uses a local copy instead of downloading (its size and sha256 are still compared with the Hub's), `--keep-inline-under-mb N` leaves TFLiteModel sections of N MiB or less as they are, `--no-smoke` skips the smoke run and `--smoke-only` runs it later on the staged file. `bash tools/externalize/externalize_repo.sh --help` lists every option.

## What the script does

1. Reads the repo's model info at the revision (`/api/models/<repo>/revision/<rev>?blobs=true`) and refuses to start when the free disk space is less than four times the file size (exit 3).
2. Downloads the file at that revision (or takes `--source`) and compares its size and sha256 with the Hub's (exit 4 on a mismatch).
3. `litert-lm unpack` → `externalize_litertlm_sections.py manifest` → `externalize_tflite_flatbuffer.py --group_name tflite_weights --num_elements_threshold 256` on every TFLiteModel section → a results table (a section whose weights file comes out empty stays as it is) → `externalize_litertlm_sections.py rewrite` → `litert-lm-builder toml … output …` → `litert-lm-peek` → `externalize_litertlm_sections.py verify-peek`.
4. Runs the checks below (exit 5 when one fails; nothing is staged).
5. Smoke: `litert-lm run <file> --backend cpu --top-k 1 --thinking false --cache no --prompt "What is the capital of France? Answer in one word."` (exit 6 on a non-zero exit, an empty reply or "An error occurred").
6. Stages `<file>`, `<file>.sha256`, `<file>.REPORT.md` and `<file>.model.toml` (the TOML the file was packed from), and removes the download and the unpacked and externalized copies. Logs and check outputs stay in the work directory.

## The checks

| check | passes when | why |
|---|---|---|
| verify-peek (LiteRT's own) | the file has as many TFLiteModel and TFLiteWeights sections as expected, and the TFLiteWeights sections form one block at the end of the file | the layout the tool documents |
| sections | every section that is not a TFLite section has the same raw bytes as in the original, in the same order; every TFLite section is byte-equal to the file the results table names (the tool's graph and weights, or the original section) | metadata, chat template and tokenizer must not change when only the weights move |
| tensors | every externalized tensor's bytes in the weights section equal the original buffer it used; every other tensor keeps its original buffer; original payload = payload kept in the graph + payload of the cleared buffers; every external buffer is in the group `tflite_weights`, 64-byte aligned, inside the weights section, and the weights section ends at the last buffer | the move must carry every weight byte, and LiteRT-LM binds only the group name `tflite_weights` |
| smoke | the CPU run exits 0 with a non-empty reply that is not "An error occurred" | the litert-lm 0.18.0 CLI prints "An error occurred" and exits 0 when it cannot create a conversation; a person reads the reply in the report |

The report also writes, without failing on them:

- `FLOAT_EXTERNAL=yes` and the tensors involved when a float tensor (FLOAT32 / FLOAT16) went into a weights section. LiteRT's guide says WebGPU's streamed loading takes only INT8, INT4 and INT2 external weights, and the tool does not filter by dtype. `LFM2.5-VL-1.6B_int4_fixB.litertlm` (litert-community/LFM2.5-VL-1.6B) is an example: its ten ShortConv `Conv1d` kernels ([1, 3, 1, 2048], FLOAT32, 245,760 bytes in all) are inputs of DEPTHWISE_CONV_2D and go into the weights section.
- The graph size after the move, marked above 50 MiB. The tool moves only input 1 of FULLY_CONNECTED, CONV_2D, DEPTHWISE_CONV_2D (depth multiplier 1) and EMBEDDING_LOOKUP. The FLOAT16 scales of blockwise-quantized INT4 weights are not op inputs and stay in the graph: Qwen3-0.6B's dynamic INT4 graph is 44,497,528 bytes (42.44 MiB) after the move, LFM2.5-VL-1.6B's text graph 76,390,032 bytes (72.85 MiB).

## Things to know

- The packed file always differs from the original by more than the weights: litert-lm-builder 0.18.0 writes container format 1.7.0, a new uuid and a new creation timestamp. So the sha256 of a rebuilt file changes on every build; compare files section by section (the report's section table).
- On CPU at the default thread count, greedy replies can differ between two processes of the same file (seen with Qwen3-0.6B on litert-lm 0.18.0 and 0.19.0.dev20261013); with `--cpu-thread-count 1` they repeat. The smoke reply is read, not compared byte for byte.
- A file that already has TFLiteWeights sections is refused (exit 2).
- Runtimes that have run an externalized Qwen3-0.6B file built this way: litert-lm 0.17.1, 0.18.0 and 0.19.0.dev20261013 on macOS (CPU and WebGPU), a LiteRT-LM main build (b7c40ea81) on macOS (Metal) and on a Pixel 8a (GPU and CPU). An Android `litert_lm_advanced_main` built at tag v0.18.0 crashes (SIGSEGV) while building the GPU engine for the externalized file on the Pixel 8a, while its CPU path runs it (google-ai-edge/LiteRT-LM#3986): on Android GPU, use a LiteRT-LM build at or after b7c40ea81, and say so on the card.
- `--executor-metadata auto` adds an ExecutorMetadata section to a file that has none, generated from the prefill / decode signatures (KV caches named `kv_cache_k_<n>` / `kv_cache_v_<n>` only; anything else is refused). It is off by default.

## Exit codes

0 staged · 1 a step of the chain failed (the log is named) · 2 usage or precondition · 3 disk guard · 4 the input does not match the Hub, or the Hub could not be read · 5 a check failed, or nothing was externalized · 6 the smoke run failed.
