# GPU graph tools

These scripts rewrite the prefill/decode graph of an existing `.litertlm` bundle without re-exporting the model. Every original weight buffer keeps its bytes. [REPRODUCE.md](../../REPRODUCE.md) (section "2026-10-06 — GPU graph re-ship") lists the eight files re-shipped with them, the checks and the Mac GPU speeds.

## What the scripts do

`gpu_graph_retrofit.py` converts a graph exported without litert-torch's `--apply_gpu_composites` into the shape the flag produces. Both shapes have the same KV layout and the same weights. The script makes three changes:

- The two DYNAMIC_UPDATE_SLICE KV-cache writes per layer become one STABLEHLO_COMPOSITE `odml.cache_update`.
- Each BATCH_MATMUL(adjY) attention product becomes a STABLEHLO_COMPOSITE `odml.runtime_bmm`.
- A signature input `param_tensor` INT32[1,1,1,7] is added, and the runtime fills in start and end.

The re-shipped files use `--mask add_bcast`. With it, the attention mask stays a FLOAT32 input and is added, by broadcast, to a [bk, g, T, C] view of the logits. The script's docstring describes the other `--mask` and `--decomp` modes.

`prefill_bucket_clone.py build` copies an existing prefill signature (`--source`, e.g. `prefill_128`) at the lengths given in `--lengths`. The copy shares every weight buffer, so the file grows by graph structure only. The new subgraphs go after the existing ones, and every existing subgraph and SignatureDef keeps its index and its bytes.

## Commands

Step 1 takes a bundle exported without `--apply_gpu_composites`; step 2 adds a 16-token prefill signature copied from `prefill_128`.

```sh
export LITERT_LM_CLI=/path/to/litert-lm
python tools/gpu_graph/gpu_graph_retrofit.py previous.litertlm retrofit.litertlm --decomp exporter --mask add_bcast
python tools/gpu_graph/prefill_bucket_clone.py build retrofit.litertlm new.litertlm --lengths 16 --source prefill_128 --report bucket.json
```

`LITERT_LM_CLI` is the path to a `litert-lm` CLI (the Python package `litert-lm` from [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM)). On a `.litertlm`, each script uses it to unpack the bundle, rewrites only the prefill/decode section, and packs the bundle again. It then checks that every other section is byte-identical. The Mac speed numbers in REPRODUCE.md were measured with the `litert-lm` 0.17.1 CLI from pip.

In the re-ship, the Phi-4-mini-reasoning file went through step 1 only. `SmolLM3-3B.litertlm` has only `prefill_256`, so its step 2 used `--source prefill_256`.

## Rebuild check

The copies of `gpu_graph_retrofit.py`, `prefill_bucket_clone.py` and `shape_variant.py` in this directory rebuild all eight files from the previous files (Hub commits in REPRODUCE.md). Every bundle section of each rebuild is byte-identical to the shipped file (sha256 per section). Only the bundle header differs because `litert-lm pack` writes a new uuid and timestamp, so the whole-file sha256 differs. Each rebuild took 6–32 s on an M4 Max.

## Limits

- The retrofit with `--mask add_bcast` was verified only on the eight files in REPRODUCE.md. The 16-token prefill signature was verified only on seven of them. All eight were exported without `--apply_gpu_composites`. Nothing is claimed for other models or architectures.
- On Phi-4-mini-reasoning, the file with `prefill_16` got fewer final answers right on the Mac GPU with fp16 activations: 7 vs 8 of 8 questions, and 26 vs 28 of 30 GSM8K questions. That file ships without it.
- A bundle whose ExecutorMetadata declares a `TYPE_LINEAR_ATTENTION` state buffer, such as the LFM2 short-convolution state `kv_cache_c_*`, gives broken text on the GPU with LiteRT-LM 0.18.0 and 0.18.1 after the retrofit. Every subgraph is still delegated, no validation error is printed, and the CPU output is unchanged. The cause is the `param_tensor` input that the retrofit adds: with it, the GPU executor switches to its in-place state mode and does not pass state buffers as inputs, so these layers lose their state at every step. The log then prints `GPU TensorBuffer not found for <state buffer> Tensor` twice per state buffer (with `--verbose` on the `litert-lm` CLI). This was measured on LFM2.5-350M (int4 and int8, 0.18.0) and on a Qwen3.5-0.8B int8 file (0.18.1, GatedDeltaNet state `kv_cache_lc_*` / `kv_cache_lr_*`) on the Mac GPU (WebGPU); the retrofit with `--mask select_bool` fails the same way. LiteRT-LM 615d193f is not in 0.18.1, and main reworked this path in 0a7de4d7cb (2026-10-19); no release contains it yet (checked 2026-10-24: the latest, v0.18.1, does not). Before a retrofit, read the state buffer types in the bundle's ExecutorMetadata, and check the GPU output with a free-form prompt: a one-word answer can still come out right on a broken run. `gpu_graph_retrofit.py` stops on such a bundle and names those state buffers (its report lists every state buffer with its type); `--allow-linear-attention-state` runs it anyway.

## Vision encoders on the GPU: `tfl_rank4.py`

The GPU delegate lays out a rank-2 or rank-3 runtime tensor batch-first (`ExtractTensorShape`, model_builder_helper.cc) but a rank-2 or rank-3 constant right-aligned (`ParseInputsWithConstTensorImpl`, model_builder.cc). An elementwise op that mixes the two reads the wrong rows of the constant: a 2D RoPE multiply `[N,h,d] × const[N,1,d]` uses the constant's first row for every token, and a position-table add `[N,C] + const[N,C]` uses row 0. `tfl_rank4.py rewrite IN OUT` lifts such ops to rank 4: elementwise ops with a constant, rank-3 BATCH_MATMUL, a MEAN that drops a rank, a STRIDED_SLICE on the leading axis, a SOFTMAX whose leading dimension is above 1. Every inserted RESHAPE keeps the data order, so the CPU output does not change. `tfl_census.py IN` lists the op forms the rewrite looks for.

It was used on the vision encoders of Muse-Glimmer-30B ([`muse_glimmer_work/`](../../muse_glimmer_work/)) and Qwen2.5-VL-72B-Instruct. Before the rewrite their GPU output was at cosine 0.25 to 0.58 against HF; after it, 1.00000 at fp32 for both models on two pictures each. It does not fix fp16: at the delegate's default precision these encoders still drift, so their bundles declare fp32 for vision.

```sh
python tools/gpu_graph/tfl_rank4.py rewrite vision_encoder.tflite vision_encoder_r4.tflite --report r4.json
python tools/gpu_graph/tfl_census.py vision_encoder.tflite
```
