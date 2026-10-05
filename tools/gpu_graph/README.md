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
