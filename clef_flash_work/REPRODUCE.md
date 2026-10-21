# Reproduce the LiteRT files and the checks

Everything here ran on one Mac (Apple M4 Max, 128 GB, macOS 27.0) from 2026-10-20 to 2026-10-21. The scripts are in [`conversion/`](conversion/README.md), which gives the commands in order and the Python environments.

## Sources

| What | Where | Pin | File | Bytes | SHA-256 |
|---|---|---|---|---:|---|
| clef-flash | `Cloudflare/clef-flash` | revision `f0b7d3cd65296abfe459e6c228b618f5d8b26d2f` | `model-00001-of-00004.safetensors` | 4,942,706,120 | `8b45a8e968141cdcc58fb71c9adfc258e2c77b5f062bc636c1fd5bc5d916b565` |
| | | | `model-00002-of-00004.safetensors` | 4,987,757,928 | `7590856c713eed844a2dcf48e6c43c4de165b788bc3f80e328311183cdbc7db8` |
| | | | `model-00003-of-00004.safetensors` | 4,954,810,240 | `e6eac2467952c33361ed7dcb3c7959d1086bbe57201cd3749c3d769fdc17fe63` |
| | | | `model-00004-of-00004.safetensors` | 3,934,446,832 | `9fcecc6556b39171238373a465f409794b7f821fb4cd1e6459e3a9c0fe317af7` |
| | | | `joint_head.safetensors` | 243,538,016 | `19cdcec8c81dc9212be320fff47462ab342fbc1278be4368fb3da71241cf5ba0` |
| | | | `joint_head_config.json` | 119 | `77efe959a38b5b17b241543e129e695f3c77465ece55a25985279bd8176279a0` |
| | | | `joint_schema_model.py` | 23,263 | `0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3` |
| | | | `tokenizer.json` | 19,989,325 | `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523` |
| | | | `config.json` | 2,832 | `66f87f6fb2616b46604daf2a9c67ddc87938296d07156efa34d59b5be49e3238` |
| | | | `processor_config.json` | 1,191 | `d89ef49ce9cd37fbf510158e13c1ef063d9286411c1ec9049932dbe0487143b1` |
| | | | `LICENSE` | 11,544 | `bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a` |

The provider's own code (`joint_schema_model.py`: `load_release_model()`, `systemone()`, `encode_record()`, the joint schema head) runs from that snapshot in the reference.

## Environments

- The provider's reference: Python 3.12.11, `conversion/requirements-venv-ref.txt` (torch 2.9.0, transformers 5.17.0, torchvision 0.24.0).
- Export, quantization, the LiteRT checks and the timing of the graphs alone: Python 3.14.6, `conversion/requirements-lt094dev.txt` (litert-torch 0.9.4, litert-converter 0.4.0, torch 2.13.0, transformers 5.14.1, ai-edge-litert 2.2.0, ai-edge-quantizer 0.9.0).
- The Python host, the example and the end-to-end checks of this folder: Python 3.14.6, `host/requirements-host.txt` (ai-edge-litert 2.2.0, numpy 2.5.3, tokenizers 0.23.2, pillow 12.3.0); the host's own test also ran in Python 3.12.11 with numpy 2.5.3, tokenizers 0.23.2 and pillow 12.3.0 only.

## Steps

The commands are in `conversion/README.md`.

1. Checkpoint and reference: `run_dl.sh` downloads the six LFS files and checks each SHA-256 against its LFS oid. `make_red_arms.py` writes the three control records before the reference runs. `clef_ref.py` runs the provider's own code (`load_release_model(device="cpu", dtype=torch.float32)` and `systemone()`, 10 threads) on the 214 fixture runs and the three control records. `tok_compare.py` checks the copy of `encode_record` the checks use (`clef_tok.py`) against the provider's, and `host_selftest.py` checks the NumPy embedding rows and rotary tables (`clef_host.py`) against transformers.
2. The row graph in PyTorch: `clef_torch_check.py` runs the graph's module (`clef_graph.py`: the Qwen3.5 text stack of the checkpoint with embeddings, a valid mask and the M-RoPE cos / sin tables as inputs and the final-norm hidden state of every position as the output; the GatedDeltaNet kernel in the form `R64+sp+ec+dd+vs6` of `clef_kernel.py`) in float32 against the reference, and `clef_score.py` scores it with the provider's head.
3. Row graphs: `clef_export.py` (L1024) and `r2_export.py` (the other lengths) convert the module with litert-torch's lightweight path in float32; `r2_quant.py` stores every FULLY_CONNECTED weight in float16 (ai-edge-quantizer float casting, explicit DEQUANTIZE); `r2_check.py` runs each file's pre-registered set on the CPU and on Metal at float32 and times it, and `r2_score.py` scores it. The float32 files were deleted after their float16-weight file passed.
4. Pictures: `r3_tower_torch.py` checks the tower rewrite (`r3_tower.py`: the vision tower and merger at one fixed tile) against the provider's module and writes the per-tile constants; `r3_metal_bisect.py` chooses the form Metal computes right (the position rows as a graph input); `r3_tower_tfl.py` converts the two towers and stores their FULLY_CONNECTED weights in float16; `r3_tower_rt.py` and `r3b_rt_verdict.py` check the towers on the CPU and Metal; `r3_e2e.py` and `r3_score.py` run the picture records end to end.
5. Host: `r5a_tables.py` copies the two vocabulary tables out of the checkpoint; `test_host.py` checks the host package against the provider's code on 257 runs.
6. This folder: `r5b_stage.py` lays it out under the names below; `r5b_fixtures.py` and `r5b_expected.py` write `fixtures/` and the example's expected answers; `r5b_e2e.py` and `r5b_score.py` run the folder's own host and graphs end to end; `r5b_sums.py` reads every large file back from its copy, `r5b_sha256sums.py` writes `SHA256SUMS`, `r5b_contract.py` writes `contract.json` and `r5b_public_results.py` writes `conversion/results/`.

The scripts are the ones that ran, with their comments and docstrings edited so that they name no path of the machine the conversion ran on, and a few lines of code changed for the same reason: the measurement lock's file, the folder of the held-out reference, the earlier port's fixture file, the Python that runs a prelint tool, three record descriptions and one result key (`conversion/README.md`). No graph, table or quantization was made again from the published scripts to compare SHA-256 values: the files here are the ones the steps above made, and `SHA256SUMS` gives their SHA-256.

## The published files and the work directory

| In this repository | In the work directory |
|---|---|
| `clef-flash_rowprefill_embeds_L256_fp16fc.tflite` | `out/r2/clef_rowprefill_L256_v2_fp16fc.tflite` |
| `clef-flash_rowprefill_embeds_L512_fp16fc.tflite` | `out/r2/clef_rowprefill_L512_v2_fp16fc.tflite` |
| `clef-flash_rowprefill_embeds_L1024_fp16fc.tflite` | `out/r2/clef_rowprefill_L1024_v2_fp16fc.tflite` |
| `clef-flash_rowprefill_embeds_L2048_fp16fc.tflite` | `out/r2/clef_rowprefill_L2048_v2_fp16fc.tflite` |
| `clef-flash_rowprefill_embeds_L4096_fp16fc.tflite` | `out/r2/clef_rowprefill_L4096_v2_fp16fc.tflite` |
| `clef-flash_vision_tower_g256_fp16fc.tflite` | `out/r3/clef_tower_g256_v2_fp16fc.tflite` |
| `clef-flash_vision_tower_g448_fp16fc.tflite` | `out/r3/clef_tower_g448_v2_fp16fc.tflite` |
| `tables/vision_pos_g256.safetensors`, `tables/vision_pos_g448.safetensors` | `cache/r3/consts_g256.npz`, `cache/r3/consts_g448.npz` (`pos_embeds`, written as one float32 tensor `pos`) |
| `tables/embed_tokens.safetensors`, `tables/lm_head.safetensors` | `out/r5a/embed_tokens.safetensors`, `out/r5a/lm_head.safetensors` |
| `head/joint_head.safetensors`, `head/joint_head_config.json`, `tokenizer/tokenizer.json` | the checkpoint's files |

## Gate

The reference is the provider's own code in float32 on the CPU (`conversion/clef_ref.py`): 214 fixture runs, the three control records and the 40 held-out runs, with each question's probabilities in `fixtures/reference_probs.json`. A file or a folder passes when, on every question whose two most likely options are more than 0.02 apart, it gives the reference's most likely option; every probability is within 0.02 of the reference's; the mean over the runs of each run's mean |Δp| is at most 0.002; and each control record moves some option of its base record by more than 0.02. On Metal every node must be on the delegate in one partition.

- Each row graph alone (`conversion/results/gates_mac.json`): every length passes on the CPU (XNNPACK, 8 threads) and on Metal at float32 with every node in one partition.
- This folder end to end (`conversion/results/e2e_mac.json`): the host package of `host/` with these graphs, on every run it can make (text, and pictures at the 256 and 448 px tiles; the provider's own resolution for pictures is not a graph here): on Metal at float32, 243 runs, 545/545 plus 5/5 near ties, max |Δp| 1.65e-3 (a picture through the float16-weight tower; 1.08e-5 on the text runs), mean 7.3e-6, the three controls moving 0.907, 0.036 and 0.975; on the CPU, the first 20 runs of each length and the controls (69 runs), 207/207 plus 3/3, max |Δp| 1.62e-3, mean 1.3e-5. The responses have the provider's structure and the same decision on every question that is not a near tie (243/243 and 69/69 runs); 158 of 2,210 and 45 of 842 of the responses' four-decimal values differ from the provider's.
- The example (`conversion/results/example_mac.json`): `examples/README.md`'s commands, run as written in this folder, print PASS on Metal and on the CPU.
- The timings (`conversion/results/timing_mac.json`): one call of each row graph, each tower, and one decision request with the host's work, median of 20 calls after 5 warm-up calls inside a machine-wide measurement lock.
