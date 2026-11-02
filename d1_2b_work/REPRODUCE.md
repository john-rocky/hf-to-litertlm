# Reproduce the LiteRT files and the checks

The conversion ran on a Mac (Apple M4 Max, 128 GB, macOS 27.0) on 2026-11-01 and 2026-11-02, and the phone checks on a Pixel 11 Pro (Android 17) and a Galaxy S26 (SM-S942Q, Android 16) on 2026-11-02. The scripts are in [`conversion/`](conversion/README.md), which gives the commands in order, the work-directory layout and the environments.

## Sources

| What | Where | Pin | File | Bytes | SHA-256 |
|---|---|---|---|---:|---|
| d1-2B | `LiquidAI/d1-2B` | revision `10e7671d44e4b37700020b1eed3f594c5f2af0b4` | `model.safetensors` | 3,836,417,920 | `b5e573687315e82ea80f4f7dd8d76b5cec3b6ee97e3509fd0493cfebb203719f` |
| | | | `tokenizer.json` | 17,908,258 | `a39eed58008cbcbc09e5b629b81338a654269c64cd13348aaa6090f4db75c7e7` |
| | | | `config.json` | 2,903 | `ffdc4c996152d23dbe8a2659a8c6aabe2992aa3fa3a41601dd0fca8771d6f332` |
| | | | `processor_config.json` | 1,254 | `37396ef3370083a04381ef25f9e1b50dec184b8cdcf08cf718feb8944ad8f3bc` |
| | | | `LICENSE` | 10,574 | `4d28ca14dedc0b3d0fcc2b3339f0e79931faa33874f3d24f522183a8fc70068c` |
| The card's photo (for the checks, not redistributed) | `http://images.cocodataset.org/val2017/000000039769.jpg` | | `000000039769.jpg` | 173,131 | `dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e` |

The revision was the head of the source repository's main branch on 2026-11-02. The provider (Liquid AI's model code in the source repository: its model, prompt and processor code) runs from that snapshot in the reference and in the graph export; `provider/` holds the files of it that the host reads.

## Environments

- Export, quantization and the LiteRT checks on the Mac: Python 3.12.13, `conversion/requirements-export.txt` (litert-torch 0.10.0, torch 2.14.1, ai-edge-litert 2.3.0, ai-edge-quantizer 0.10.0).
- The provider's reference: Python 3.12.11, `conversion/requirements-ref.txt` (torch 2.14.1, torchvision 0.29.1, transformers 5.19.0).
- The Python host and the example: Python 3.12.11, `host/requirements-host.txt` (ai-edge-litert 2.3.0, numpy 2.5.4, tokenizers 0.23.3).
- The NPU compile: the image of `conversion/npu/Dockerfile` (linux/amd64: Python 3.12.13, ai-edge-litert 2.2.0 and its Qualcomm compiler plugin, QAIRT 2.47.0.260601114230, which is Qualcomm's SDK and is not in this repository).
- The phones: LiteRT 2.2.0 through the Kotlin CompiledModel API, in a test app that is not part of this repository.

## Steps

The commands are in `conversion/README.md`. Every export ran with `PYTHONHASHSEED=0`.

1. Snapshot: an anonymous `hf download` of the pinned revision, checked against the Hub's file list.
2. Table: `write_embed_table` of `host/d1_host.py` writes `tables/embed_table.npy`, the tied embedding table's bfloat16 bits as they are in `model.safetensors` (uint16 `[128000, 2048]`).
3. Decision files: `export_decide.py --L <L>` for 128, 256, 512, 1,024 and 2,048 tokens, and `--L <L> --qblock 1024` for 4,096 and 6,144 (the full-attention layers' queries in blocks of 1,024, which bounds the memory of their scores). Each exports `d1_graph.py` with litert-torch, then stores its FULLY_CONNECTED weights in float16 with ai-edge-quantizer (float casting; the activations stay float32).
4. The eight-sequence file: `export_batch.py --B 8 --causal-runtime`. The eight-signature file: `export_multi8.py`, the seven length graphs and the eight-sequence graph over one copy of the weights.
5. Towers: `save_pixels.py` saves the provider processor's pixel values of the photo for the export checks; `export_tower.py --canvas <GH>x<GW>` for 24x24, 32x48, 48x32, 48x64 and 64x48 patches, and `--canvas <GH>x<GW> --qblock 1152` for 72x96 and 96x72 (the queries in blocks of 1,152).
6. NPU files: `conversion/npu/aot_compile_sm8850.py` compiles a decision file for the SM8850 in the image of `conversion/npu/Dockerfile`, and `conversion/npu/aot_check.py` reads the compile (see NPU files below).
7. Checks and timing: see `conversion/README.md`.
8. The shipped set: the graphs are the work directory's `out/graphs/` files under the same names; `npu/sm8850/<name>_SM8850.tflite` is the compile's `<name>_Qualcomm_SM8850_apply_plugin.tflite`. `SHA256SUMS` lists every file.

## The rebuild on 2026-11-02

On 2026-11-02 the steps ran again from a clean copy of this repository's text files at one commit, in a new work directory, with an anonymous download of the pinned snapshot (checked against the Hub's file list: 42 files, no difference) and the photo (its SHA-256 checked). One file of each kind was rebuilt, the 128-token NPU file too; the other files come from the same scripts with other arguments.

<!-- dayof:begin -->
| File | Rebuilt SHA-256 | Shipped SHA-256 | Result |
|---|---|---|---|
| `d1-2b_decide_L128_fp16fc.tflite` | `51cb7b9a3a90a6295e4514a8b232c72cdf86e688dae3dfa1abaa60fa7727cd52` | `51cb7b9a3a90a6295e4514a8b232c72cdf86e688dae3dfa1abaa60fa7727cd52` | the same bytes |
| `d1-2b_decide_b8_L128_fp16fc.tflite` | `85e108da8d07683b92c39d0ee1020c4b643d605df9b019a2a4c80983bdbdbdd6` | `85e108da8d07683b92c39d0ee1020c4b643d605df9b019a2a4c80983bdbdbdd6` | the same bytes |
| `d1-2b_decide_multi8_fp16fc.tflite` | `afeea7f20aaffedf1f9d71a8bd3cda20d80c2bd442db94d2465f2cd32a5845a1` | `afeea7f20aaffedf1f9d71a8bd3cda20d80c2bd442db94d2465f2cd32a5845a1` | the same bytes |
| `d1-2b_tower_G24x24_fp16fc.tflite` | `672bd5154bbe9517082e1fd97dfc1ffe78caaa46bb9d1c50a1f44c0a482ecc18` | `672bd5154bbe9517082e1fd97dfc1ffe78caaa46bb9d1c50a1f44c0a482ecc18` | the same bytes |
| `tables/embed_table.npy` | `e8d0c7462ce32986eb4414ee30a152be0bc893a62cd32dc3a2baf5a1363fc07a` | `e8d0c7462ce32986eb4414ee30a152be0bc893a62cd32dc3a2baf5a1363fc07a` | the same bytes |
| `npu/sm8850/d1-2b_decide_L128_fp16fc_SM8850.tflite` | `de6abc09c34edc1ceeacc6c1164370493c9f8b6e7c13a6ba733d77b06019f596` | `de6abc09c34edc1ceeacc6c1164370493c9f8b6e7c13a6ba733d77b06019f596` | the same bytes |
<!-- dayof:end -->

## NPU files

The five files in `npu/sm8850/` are the files that ran on the Galaxy S26: their SHA-256 are the ones the phone read before it ran them. On 2026-11-02 a compile of the 128-token file in the image of `conversion/npu/Dockerfile`, from the rebuilt decision file, gave the shipped file's SHA-256 (the table above); the other four were not compiled again. `conversion/npu/aot_check.py` reads the plugin's log of a compile: every op in one partition; (i) the plugin's memory estimate for the graph within its limit of 3,670,016 KiB (the init graph option `memory_alloc_limit`, 3,584 MiB); (ii) the context binary plus the graphs' memory beyond their constants plus 64 MiB within the 3,750,000,000 bytes of one compute-DSP session, which a context cannot span.

<!-- npu:begin -->
| File | Input graph (SHA-256) | Ops in one partition | (i) estimate, KiB | (ii) session, bytes | Bytes | SHA-256 |
|---|---|---:|---:|---:|---:|---|
| `d1-2b_decide_L128_fp16fc_SM8850.tflite` | `d1-2b_decide_L128_fp16fc.tflite` (`51cb7b9a3a90a629…`) | 2,062 / 2,062 | 2,905,324 | 3,053,187,072 | 2,970,355,376 | `de6abc09c34edc1ceeacc6c1164370493c9f8b6e7c13a6ba733d77b06019f596` |
| `d1-2b_decide_L256_fp16fc_SM8850.tflite` | `d1-2b_decide_L256_fp16fc.tflite` (`175147ab5e366a69…`) | 2,062 / 2,062 | 2,916,432 | 3,069,456,384 | 2,976,356,016 | `873ea1d67bc13971d12580a320a27baf196b2f7b0db7bcc88e9460256287a9e1` |
| `d1-2b_decide_L512_fp16fc_SM8850.tflite` | `d1-2b_decide_L512_fp16fc.tflite` (`62dccd241bd5890d…`) | 2,123 / 2,123 | 2,952,900 | 3,131,666,432 | 3,000,534,704 | `c5e1b297b1c7cffe62053d623d9c919dc2dd51a4c857686a6db235d6288efb08` |
| `d1-2b_decide_L1024_fp16fc_SM8850.tflite` | `d1-2b_decide_L1024_fp16fc.tflite` (`1e098e038f76b751…`) | 2,123 / 2,123 | 3,068,352 | 3,279,179,776 | 3,031,398,064 | `d08d75fc489f1e9525f48a2767cddea4c3cf393da0df07e1f14516d269d6aade` |
| `d1-2b_decide_b8_L128_fp16fc_SM8850.tflite` | `d1-2b_decide_b8_L128_fp16fc.tflite` (`85e108da8d07683b…`) | 2,104 / 2,104 | 3,051,000 | 3,319,877,632 | 3,087,439,584 | `12ff74c978968b206d5e46be4a6aa5a718a8f1f4efadb4c5c271c8f0f7ef0c4c` |
<!-- npu:end -->

## Where the numbers come from

Each number on the model card comes from a file that a command of `conversion/README.md` writes, from the example, or from the phones' reports:

| Numbers on the card | Command | File |
|---|---|---|
| Agreement on the Mac: the single files and towers, on the CPU and on Metal at float32 | `run_gate.py`, then `score.py` | `results/gate_cpu_fp32.json`, `results/gate_gpu_fp32.json` |
| Agreement: the eight-sequence file, 8 sequences a call | `check_batch.py`; `run_gate_batch.py`, then `score.py` | `results/gate_L128_b8_cpu.json`, `results/gate_L128_b8_gpu32.json` |
| Agreement: the seven length graphs in one file, with constant tensor sharing | `run_gate.py --multi … --sharing`, then `score.py` | `results/gate_gpu_fp32_multi_sharing.json` |
| The eight-signature file gives the outputs of the seven-signature file and of the eight-sequence file | `multi8_check.py` | `results/multi8_check_gpu.json`, `results/multi8_check_cpu.json` |
| GPU precision: the answers at Metal's default precision | `run_gate.py --precision default`, then `score.py` | `results/gate_gpu_default.json` |
| The source card's ticket: probabilities, token counts and the largest difference from the reference | `examples/run_example.py --check --out <file>` on the CPU and with `--accel gpu`; the reference's answers are `examples/run_example.expected.json`, from `ref_run.py` | the `--out` files |
| The line that the card's own code prints | the card's code block, run from the repository root | its output |
| Speed on the Mac, the compile seconds, the 64 states | `bench_host.py`; `bench_batch.py --mode batch --arm b8` | `results/bench_host_gpu_fp32.json`, `results/bench_host_cpu_fp32.json`, `results/bench_batch_b8_gpu32.json`, `results/bench_batch_b8_cpu.json` |
| Agreement on the phones | `run_replay.py` or `replay_b8.py` over a phone's outputs, then `score.py` | `results/line_<name>.json` |
| Speed and loads on the phones, memory readings, Android versions | the phones' reports (below) | not in this repository |
| File sizes, operators, data types and SHA-256 | the export scripts; `tflite_scan_d1.py`, `rank_dump.py`, `bucket_table.py` | `results/export_*.json`; `SHA256SUMS` |
| The NPU files: ops in one partition and the two memory lines | `npu/aot_check.py` | `out/npu/<tag>/aot_check.json` |

The phones ran the graph files in programs that are not in this repository. On the Pixel 11 Pro, and on the Galaxy S26's GPU and CPU, a test app ran them through the Kotlin CompiledModel API of LiteRT 2.2.0. On the Galaxy S26's NPU, a test runner on the C API of LiteRT 2.2.0 ran the files of `npu/sm8850/`, outside the app. Each wrote a report per run with its times and its load (the compile, or for the NPU the load of the compiled context); a time is one graph call in both: writing all the inputs, running, and reading the output back. A picture request on the Galaxy S26 is a tower call on the GPU at float32 in the app and a decision call on the NPU in the runner, timed apart. The inputs of every phone run were laid out on the Mac by the scripts of this repository, and the outputs were read back and scored on the Mac by its scripts (`conversion/README.md`, The phones); in the Galaxy S26 NPU rows, the pictures' decision inputs are the Pixel 11 Pro's GPU tower outputs at float32.
