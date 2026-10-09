# Reproduce the LiteRT files and the checks

Everything here ran on a Mac (Apple M4 Max, 128 GB, macOS 27.0) and one Galaxy S26 (SM-S942Q, Android 16) on 2026-10-08 and 2026-10-09. The scripts are in [`conversion/`](conversion/README.md), which gives the commands in order, the work-directory layout and the two Python environments.

## Sources

| What | Where | Pin | File | Bytes | SHA-256 |
|---|---|---|---|---:|---|
| d1-3B | `LiquidAI/d1-3B` | revision `da1fe36a861f24690f27f622dca1d8688503d113` | `model.safetensors` | 6,247,065,504 | `50e03317847caf6df9a9aee27ed40f20554a86a21e60d1d47ba41a422b546c0c` |
| | | | `tokenizer.json` | 17,905,750 | `8096ecb9f54599d756c8de728a598a340bc1e43c0deb77ddd62456c38349fcee` |
| | | | `config.json` | 2,773 | `0cbac0f580bd8036ef94ac83aa632a57a170e66d20f381ef47c3f6486c9a3e8c` |
| | | | `processor_config.json` | 824 | `b9b82ad34a69e2f70a9033741e11f82b8a98bb5f8e6c13b3b2d5efdd3935d916` |
| | | | `LICENSE` | 10,574 | `4d28ca14dedc0b3d0fcc2b3339f0e79931faa33874f3d24f522183a8fc70068c` |
| The card's photo (measurement only, not redistributed) | `http://images.cocodataset.org/val2017/000000039769.jpg` | | `000000039769.jpg` | 173,131 | `dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e` |

The later commits on the source repository's main branch, up to `051bcc46` on 2026-10-07, change `README.md` only: every other file has the same blob id at both revisions. The provider's model and prompt code (`modeling_d1.py`, `hybrid.py`, `lfm2_vl.py`, `prompt.py`, `runner.py`, `api.py`) runs from that snapshot in the reference and in the graph export.

## Environments

- The provider's reference and the fixtures' rows: Python 3.12.11, `conversion/requirements-venv-ref.txt` (torch 2.14.1, transformers 5.14.1).
- Export, quantization, LiteRT checks and timing: Python 3.14.6, `conversion/requirements-lt094dev.txt` (litert-torch 0.9.4, torch 2.13.0, transformers 5.14.1, ai-edge-litert 2.2.0, ai-edge-quantizer 0.9.0).
- The Python host and the example: Python 3.14.6, `host/requirements-host.txt` (ai-edge-litert 2.2.0, numpy 2.5.2, tokenizers 0.22.2).
- Phone: LiteRT 2.2.0 from Maven through the Kotlin CompiledModel API, in a measurement app (not part of this repository).

## Steps

The commands are in `conversion/README.md`.

0. Test requests: `fixtures/rebuild_requests.py` restores `requests.json` (382 records) and the four control requests (`red_arms.json`) from `fixtures/requests_public.json`, `fixtures/red_arms_public.json` and the kev repository at tag `kev-1.0`, and checks their SHA-256; `build_rows.py` writes the rows the checks read (`rows.json`) with the provider's code and tokenizer. `fixtures/README.md` describes the records and why 221 of them are listed by reference.
1. Tables: `d1_tables.py` cuts the read-out rows (1,234 ids, float32), the tower's position table (float32) and the whole tied embedding table (bfloat16, its bytes as in the checkpoint) out of `model.safetensors`, reading only the bytes it needs.
2. Reference: `d1_reference.py` runs the provider's code in float32 on the CPU on every fixture question, one row per question (415 questions), and the four control requests; `d1_vision_ref.py` does the same for the three requests with pictures.
3. Row graphs: `d1_graph_check.py` compares the export form of the row graph with the provider's modules in PyTorch; `d1_export.py` exports it with litert-torch; `d1_storage.py --variant v2e` stores its FULLY_CONNECTED weights in float16 (ai-edge-quantizer float casting, explicit DEQUANTIZE); `d1_check.py` runs the gate. `d1_run_L.sh` chains these per length bucket.
4. Shared-state pairs: `d1_shared_state.py --check --embeds` compares the pair with the row form in PyTorch on every question (906 runs, max |Δp| 3.8e-6); `d1_export_pair.py --embeds` exports each pair (two signatures over one copy of the weights); `d1_storage.py --pair --variant v2e` stores it, keeping the rotary-table FULLY_CONNECTED in float32; `d1_check_pair.py` runs the gate with both state hand-overs.
5. Picture graphs: `d1_vision_graph.py` exports the tower and the projector, `d1v_storage.py` stores their FULLY_CONNECTED weights in float16, and the vision checks compare them with transformers; `d1_run_vision.sh` chains these. A rerun of that chain in a new folder gave the shipped tower and projector files byte for byte.
6. Contract and host: `d1_contract.py`, `d1v_contract.py`, `d1_image_contract.py` and `d1_export_pair.py --contract-r10` write the contract. `test_d1_host.py` checks the host against the provider's code; `test_d1_vision_real.py` runs the three picture requests end to end; `d1_check_pair.py --host` compares the pair route with the row route.
7. Timing: `d1_clock_pair.py` (rows and pairs per request) and `d1_image_timing.py` (pictures), on Metal at float32 and on the CPU with 8 threads, each inside a measurement lock that the machine's other heavy jobs waited for.
8. The shipped set: `d1_ship_set.py` hashes every file again and compares it with the record that made it. The published files are the work directory's files under new names:

| In this repository | In the work directory |
|---|---|
| `d1-3b_rowprefill_embeds_L128_fp16fc.tflite` | `exports/real_rowprefill_embeds_L128_v2e_fp16fc.tflite` |
| `d1-3b_rowprefill_embeds_L256_fp16fc.tflite` | `exports/real_rowprefill_embeds_L256_v2e_fp16fc.tflite` |
| `d1-3b_rowprefill_embeds_L512_fp16fc.tflite` | `exports/real_rowprefill_embeds_L512_v2e_fp16fc.tflite` |
| `d1-3b_rowprefill_embeds_L1024_fp16fc.tflite` | `exports/real_rowprefill_embeds_L1024_v2e_fp16fc.tflite` |
| `d1-3b_rowprefill_embeds_L2048_fp16fc.tflite` | `exports/real_rowprefill_embeds_L2048_v2e_fp16fc.tflite` |
| `d1-3b_rowprefill_embeds_L4096_fp16fc.tflite` | `exports/real_rowprefill_embeds_L4096_v2e_fp16fc.tflite` |
| `d1-3b_sharedstate_embeds_Ls64_Lq64_fp16fc.tflite` | `exports/real_sharedstate_embeds_Ls64_Lq64_v2e_fp16fc.tflite` |
| `d1-3b_sharedstate_embeds_Ls128_Lq64_fp16fc.tflite` | `exports/real_sharedstate_embeds_Ls128_Lq64_v2e_fp16fc.tflite` |
| `d1-3b_sharedstate_embeds_Ls256_Lq128_fp16fc.tflite` | `exports/real_sharedstate_embeds_Ls256_Lq128_v2e_fp16fc.tflite` |
| `d1-3b_vision_tower_fp16fc.tflite` | `exports/real_vision_tower_v2_fp16fc.tflite` |
| `d1-3b_projector_fp16fc.tflite` | `exports/real_projector_v2_fp16fc.tflite` |
| `tables/readout_table.safetensors` | `cache/real/tables/readout_table.safetensors` |
| `tables/embed_table.safetensors` | `cache/real/tables/embed_table.safetensors` |
| `tables/vision_position_table.safetensors` | `cache/real/tables/vision_position_table.safetensors` |
| `tokenizer/tokenizer.json` | `hf_small/tokenizer.json` |

`contract.json` and `host/*.py` are the work directory's `host/contract.json` and `host/*.py` with this repository's file names: in the contract, every string value that names a shipped file (18 values); in the host, `D1Host.from_dir` reads the tokenizer at the contract's `tokenizer.file`. Their comments and docstrings, one description in the contract, and the scripts in `conversion/` are edited so that they name no path of the machine the conversion ran on and no file of the work directory that is not published; the code is otherwise the same, and `conversion/README.md` lists what the scripts read in place of those files.

## Gate

The reference is the provider's code in float32 on the CPU, one row per question. A file passes when, over the questions whose rows it holds: the most likely option is the reference's on every question that is not a near tie (the reference's two most likely options more than 0.02 apart), the largest absolute difference of any option's probability is at most 0.02, the mean absolute difference over all options is at most 0.002, and each of the four control requests moves some option by more than 0.02 against the request it changes.

| File | Runtime | Questions | Same most likely option | Near tie kept | Max \|Δp\| | Mean \|Δp\| | Controls moved | Delegated |
|---|---|---:|---:|---:|---:|---:|---:|---|
| L128 | Mac CPU, 8 threads | 311 | 311/311 | — | 6.5e-6 | 3.7e-7 | 4/4 | 1,691/1,691 |
| L128 | Metal, float32 | 311 | 311/311 | — | 1.0e-5 | 3.6e-7 | 4/4 | 1,691/1,691 |
| L256 | Mac CPU, 8 threads | 388 | 387/387 | 1/1 | 6.5e-6 | 3.6e-7 | 4/4 | 1,691/1,691 |
| L256 | Metal, float32 | 388 | 387/387 | 1/1 | 8.4e-6 | 3.7e-7 | 4/4 | 1,691/1,691 |
| L512 | Mac CPU, 8 threads | 395 | 394/394 | 1/1 | 6.5e-6 | 3.5e-7 | 4/4 | 1,691/1,691 |
| L512 | Metal, float32 | 395 | 394/394 | 1/1 | 7.6e-6 | 3.5e-7 | 4/4 | 1,691/1,691 |
| L1024 | Mac CPU, 8 threads | 398 | 397/397 | 1/1 | 6.5e-6 | 3.5e-7 | 4/4 | 1,691/1,691 |
| L1024 | Metal, float32 | 398 | 397/397 | 1/1 | 7.6e-6 | 3.5e-7 | 4/4 | 1,691/1,691 |
| L2048 | Mac CPU, 8 threads | 411 | 410/410 | 1/1 | 6.5e-6 | 3.5e-7 | 4/4 | 1,691/1,691 |
| L2048 | Metal, float32 | 411 | 410/410 | 1/1 | 7.6e-6 | 3.4e-7 | 4/4 | 1,691/1,691 |
| L4096 | Mac CPU, 8 threads | 415 | 414/414 | 1/1 | 6.5e-6 | 3.5e-7 | 4/4 | 1,691/1,691 |
| L4096 | Metal, float32 | 415 | 414/414 | 1/1 | 7.6e-6 | 3.4e-7 | 4/4 | 1,691/1,691 |
| Pair Ls64+Lq64 | Mac CPU, 8 threads | 201 | 201/201 | — | 6.5e-6 | 3.7e-7 | 3/3 | 1,682/1,682 + 1,717/1,717 |
| Pair Ls64+Lq64 | Metal, float32 | 201 | 201/201 | — | 4.4e-6 | 3.2e-7 | 3/3 | 1,682/1,682 + 1,717/1,717 |
| Pair Ls128+Lq64 | Mac CPU, 8 threads | 290 | 290/290 | — | 6.5e-6 | 3.6e-7 | 4/4 | 1,682/1,682 + 1,717/1,717 |
| Pair Ls128+Lq64 | Metal, float32 | 290 | 290/290 | — | 4.0e-6 | 3.0e-7 | 4/4 | 1,682/1,682 + 1,717/1,717 |
| Pair Ls256+Lq128 | Mac CPU, 8 threads | 390 | 390/390 | — | 7.0e-6 | 3.9e-7 | 4/4 | 1,682/1,682 + 1,717/1,717 |
| Pair Ls256+Lq128 | Metal, float32 | 390 | 390/390 | — | 4.2e-6 | 3.5e-7 | 4/4 | 1,682/1,682 + 1,717/1,717 |

On Metal at the default precision (float16 storage) every file stays finite and misses the tolerance: max |Δp| 0.044 to 0.086. The pairs gave the same probabilities with both state hand-overs, bit for bit.

The fp16 norm note: float16 storage keeps an RMSNorm's sum of squares in float16, which overflows past 65,504. Over the 415 text rows the largest sum of squares at any norm was 1,050 (`d1_norm_range.py`). Scaling the norms' inputs down before the sum (one factor per site or per layer, `D1_NORM_SCALE`) left the default-precision error of the earlier 256-token row graphs at max |Δp| 0.062 to 0.077, against 0.051 and 0.075 without it. The shipped graphs carry no such scaling, and the GPU runs at float32.

Pictures end to end (`test_d1_vision_real.py`, the shipped tower, projector and row graphs, three requests): the token ids, pixel values and insertion order equal the provider's processor bit for bit; max |Δp| 3.4e-5 (mean 1.4e-5) on the CPU and 3.1e-5 (mean 1.4e-5) on Metal at float32, the most likely option kept on all three.

The example of this repository (`examples/run_example.py --check`, run from this repository's files): on the CPU and on Metal at float32, every probability within 1e-5 of the reference, the same input token counts, routes and refusal. The host's unit test (`test_d1_host.py`) on this repository's host: 415 rows equal to the provider's text, ids, read-out groups, option codes and answer slots.

## Galaxy S26

The phone runs used one Galaxy S26 (SM-S942Q, Android 16) and an earlier form of the 256-token row graph: float16 FULLY_CONNECTED weights and an int8 embedding table inside the graph (5.14 GB). On the GPU (OpenCL, FP16_WITH_FP32_ACCUM) and on the CPU (XNNPACK, 4 threads) a memory guard stopped the app 8 s into the compile, when MemAvailable fell below 1,500,000 kB. A form with dynamic int8 FULLY_CONNECTED weights (2.71 GB) loaded on the CPU and moved the probabilities by up to 0.157 on 100 questions. `conversion/results/galaxy_s26.json` holds every sample.

## Timing

A request's time is the median of 20 requests after 5 warm-up requests; every graph call is write + run + read-back of the whole output. `conversion/results/timing_mac.json` holds every set.

## Where the numbers come from

| Number | Produced by | Record (in the work directory; summary in `conversion/results/`) |
|---|---|---|
| The test requests | `make_d1_fixtures.py`, `d1_image_fixture.py`, `d1_reference.py --make-red-arms`, `build_rows.py` | `fixtures/requests.json`, `fixtures/red_arms.json`, `fixtures/rows.json` (`fixtures/requests_public.json`, `fixtures/red_arms_public.json`; `fixtures/rebuild_requests.py` restores them) |
| Reference probabilities | `d1_reference.py`, `d1_vision_ref.py` | `results/reference_real.json`, `results/real_vision_e2e_ref.json` (`fixtures/reference_probs.json`) |
| Row-graph gates | `d1_check.py` (through `d1_run_L.sh`) | `results/real_rowprefill_embeds_L<L>_v2e_fp16fc_<cpu, gpu_f32, gpu_default>_check.json` (`gates_mac.json`) |
| Pair gates | `d1_check_pair.py` | `results/real_sharedstate_embeds_Ls<Ls>_Lq<Lq>_v2e_fp16fc_<accel>_check.json` (`gates_mac.json`) |
| The pairs against the row form in PyTorch | `d1_shared_state.py --check --embeds` | `results/real_sharedstate_embeds_torch_check.json` (`pair_vs_row_torch.json`) |
| Pictures end to end | `test_d1_vision_real.py` | `results/real_vision_e2e_ship_cpu.json`, `results/real_vision_e2e_r10_ship_gpu.json`, `results/real_vision_e2e_ship_gpu_default.json` (`picture_e2e_mac.json`) |
| The host's two routes | `d1_check_pair.py --host` | `results/real_sharedstate_embeds_host_check.json` (`host_check.json`) |
| Mac timing, compile times and memory | `d1_clock_pair.py`, `d1_image_timing.py` | `results/timing_mac_r10.json`, `results/timing_mac_image.json` (`timing_mac.json`) |
| Galaxy S26 | the measurement app's reports and samples | `results/device_r7_s26.json` (`galaxy_s26.json`) |
| Operators and storage | `d1_export.py`, `d1_export_pair.py`, `d1_storage.py`, `tflite_scan.py` | `results/real_export_embeds_L<L>.json`, `results/real_sharedstate_embeds_export_*.json`, `results/*_v2e_fp16fc_quant.json` |
| The fp16 norm note | `d1_norm_range.py`, `d1_check.py` | `results/real_norm_range.json`, `results/realns_rowprefill_L256_*_gpu_default_check.json`, `results/realnsl_rowprefill_L256_*_gpu_default_check.json`, `results/real_rowprefill_L256_*_gpu_default_check.json` |
| SHA-256 of the published files | `sha256` | `SHA256SUMS` |
