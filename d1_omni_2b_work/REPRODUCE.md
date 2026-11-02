# Reproduce the LiteRT files and the checks

The conversion ran on a Mac (Apple M4 Max, 128 GB, macOS 27.0) on 2026-11-02, and the phone checks on a Pixel 11 Pro (Android 17) and a Galaxy S26 (SM-S942Q, Android 16) on 2026-11-02. The scripts are in [`conversion/`](conversion/README.md), which gives the commands in order, the work-directory layout and the environments.

## Sources

| What | Where | Pin | File | Bytes | SHA-256 |
|---|---|---|---|---:|---|
| d1-Omni-2B | `LiquidAI/d1-Omni-2B` | revision `431f1d667de5afcc9b0c58b2d9c170a7b38b358b` | `model.safetensors` | 4,333,941,640 | `1553fff04c54377ff527d2016fc479425ef60a0b3bc7a2e61d32f80a75c32aaf` |
| | | | `tokenizer.json` | 17,908,634 | `41927bd314d17d7d7fadb44e10cd0919f7b71528d7741596d8cbd15c6945ec44` |
| | | | `config.json` | 5,802 | `cebf53b6bb8835cefe9b415d9619c23398ab8326d6385ee80f98fef0caad7169` |
| | | | `processor_config.json` | 1,928 | `2288d34b7c0579593258b1c083de4ab8328a842fdb0190977363e573dbdcd0e5` |
| | | | `LICENSE` | 10,574 | `4d28ca14dedc0b3d0fcc2b3339f0e79931faa33874f3d24f522183a8fc70068c` |
| The card's photo (for the checks, not redistributed) | `http://images.cocodataset.org/val2017/000000039769.jpg` | | `000000039769.jpg` | 173,131 | `dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e` |

The revision was the head of the source repository's main branch on 2026-11-02. The provider (Liquid AI's model code in the source repository: its model, prompt, processor and feature-extractor code) runs from that snapshot in the reference and in the graph export; `provider/` holds the files of it that the host reads.

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
4. The eight-sequence file: `export_b8_omni.py`. The eight-signature file: `export_multi8_omni.py`, the seven length graphs and the eight-sequence graph over one copy of the weights.
5. Towers: `save_pixels.py` saves the provider processor's pixel values of the photo for the export checks; `export_tower.py --canvas <GH>x<GW>` for 24x24, 32x48, 48x32, 48x64 and 64x48 patches, and `--canvas <GH>x<GW> --qblock 1152` for 72x96 and 96x72 (the queries in blocks of 1,152).
6. Audio encoders: `audio_check.py --synthetic --save-feats out/feats` saves the provider feature extractor's features of synthetic clips for the export checks; `export_audio.py --T <T>` for 504, 1,000, 2,000 and 3,000 frames.
7. NPU files: `conversion/npu/aot_compile_sm8850.py` compiles a decision file for the SM8850 in the image of `conversion/npu/Dockerfile`, and `conversion/npu/aot_check.py` reads the compile (see NPU files below).
8. Checks and timing: see `conversion/README.md`.
9. The shipped set: the graphs are the work directory's `out/graphs/` files under the same names; `npu/sm8850/<name>_SM8850.tflite` is the compile's `<name>_Qualcomm_SM8850_apply_plugin.tflite`. `SHA256SUMS` lists every file.

## The rebuild on 2026-11-02

On 2026-11-02 the steps ran again from a clean copy of this repository's text files at one commit, in a new work directory, with an anonymous download of the pinned snapshot (checked against the Hub's file list: 47 files, no difference) and the photo (its SHA-256 checked). One file of each kind was rebuilt; the other files come from the same scripts with other arguments.

<!-- dayof:begin -->
| File | Rebuilt SHA-256 | Shipped SHA-256 | Result |
|---|---|---|---|
| `d1-omni-2b_decide_L128_fp16fc.tflite` | `0c9814d08d07f7ecf542df85e613a16988db5b1c31ad29c5785405b44f0359c5` | `0c9814d08d07f7ecf542df85e613a16988db5b1c31ad29c5785405b44f0359c5` | the same bytes |
| `d1-omni-2b_decide_b8_L128_fp16fc.tflite` | `21948cd632b1b9f25694d8c26bd9190126a630c096746d359fcdcdda590b1b13` | `21948cd632b1b9f25694d8c26bd9190126a630c096746d359fcdcdda590b1b13` | the same bytes |
| `d1-omni-2b_decide_multi8_fp16fc.tflite` | `47c5a1e4bc470e57f02f3a5161b0f9673c468d927ada1efa11cf26e3ec95684a` | `47c5a1e4bc470e57f02f3a5161b0f9673c468d927ada1efa11cf26e3ec95684a` | the same bytes |
| `d1-omni-2b_tower_G24x24_fp16fc.tflite` | `b1ea62f74651f4d99c9d26e52d78a698bd05687b81cbd54c4ae9cdad72b6463b` | `b1ea62f74651f4d99c9d26e52d78a698bd05687b81cbd54c4ae9cdad72b6463b` | the same bytes |
| `d1-omni-2b_audio_T504_fp16fc.tflite` | `088e8e544e447996abbc6fad5ec958396e357d6ea73ed62f53ecc341b4fe1aa3` | `088e8e544e447996abbc6fad5ec958396e357d6ea73ed62f53ecc341b4fe1aa3` | the same bytes |
| `tables/embed_table.npy` | `0c5cecb5808e459cc5fce45ea85644056294728417e2e2348d447d32e33c27ef` | `0c5cecb5808e459cc5fce45ea85644056294728417e2e2348d447d32e33c27ef` | the same bytes |
<!-- dayof:end -->

## NPU files

The four files in `npu/sm8850/` are the files that ran on the Galaxy S26: their SHA-256 are the ones the phone read before it ran them. They were not compiled again on 2026-11-02. `conversion/npu/aot_check.py` reads the plugin's log of a compile: every op in one partition; (i) the plugin's memory estimate for the graph within its limit of 3,670,016 KiB (the init graph option `memory_alloc_limit`, 3,584 MiB); (ii) the context binary plus the graphs' memory beyond their constants plus 64 MiB within the 3,750,000,000 bytes of one compute-DSP session, which a context cannot span. The 1,024-token decision file compiled within both lines too, but its answers on the phone moved one box edge outside the tolerance, so it is not in this repository: on the Galaxy S26 that length runs on the GPU.

<!-- npu:begin -->
| File | Input graph (SHA-256) | Ops in one partition | (i) estimate, KiB | (ii) session, bytes | Bytes | SHA-256 |
|---|---|---:|---:|---:|---:|---|
| `d1-omni-2b_decide_L128_fp16fc_SM8850.tflite` | `d1-omni-2b_decide_L128_fp16fc.tflite` (`0c9814d08d07f7ec…`) | 2,062 / 2,062 | 2,905,324 | 3,053,187,072 | 2,970,355,376 | `f7d29aa961ed15bce5e3c5141ed608d5eace85c30f141c57502bf697290f9901` |
| `d1-omni-2b_decide_L256_fp16fc_SM8850.tflite` | `d1-omni-2b_decide_L256_fp16fc.tflite` (`ee1e204dd83d772f…`) | 2,062 / 2,062 | 2,916,432 | 3,069,456,384 | 2,976,356,016 | `ed4fbb447c93e619df79a697df79a1f560b1b2eba092c2575c9590313e012071` |
| `d1-omni-2b_decide_L512_fp16fc_SM8850.tflite` | `d1-omni-2b_decide_L512_fp16fc.tflite` (`6464159ade002f94…`) | 2,123 / 2,123 | 2,952,900 | 3,131,666,432 | 3,000,534,704 | `5ccb7e3fe8ea5381264338a45589b6e36b548618ecb0c53473df7db5b101549f` |
| `d1-omni-2b_decide_b8_L128_fp16fc_SM8850.tflite` | `d1-omni-2b_decide_b8_L128_fp16fc.tflite` (`21948cd632b1b9f2…`) | 2,104 / 2,104 | 3,051,000 | 3,319,877,632 | 3,087,439,584 | `d76bae81d96c36ce186c1a455f290cfc7569e2df80103dbc3d1bbf36e6b911ac` |
<!-- npu:end -->

## Where the numbers come from

Each number on the model card comes from a file that a command of `conversion/README.md` writes, from the example, or from the phones' reports:

| Numbers on the card | Command | File |
|---|---|---|
| Agreement on the Mac: the single files, on the CPU and on Metal at float32, text and pictures, and audio | `run_gate.py`, then `score.py` (two rows) | `results/gate_cpu_fp32.json`, `results/gate_cpu_fp32_audio.json`, `results/gate_gpu_fp32.json`, `results/gate_gpu_fp32_audio.json` |
| Agreement: the eight-signature file with constant tensor sharing | `run_gate_m8.py --mode full`, then `score.py` | `results/gate_full_m8_gpu32_textpic.json`, `results/gate_full_m8_gpu32_audio.json` |
| Agreement: the eight-sequence graph, 8 sequences a call | `run_gate_m8.py --mode b8`, then `score.py` | `results/gate_b8_b8single_gpu32_textpic.json`, `results/gate_b8_m8_cpu_textpic.json` |
| The eight-signature file gives the outputs of the seven-signature file and of the eight-sequence file | `multi8_check_omni.py` | `results/multi8_check_gpu.json`, `results/multi8_check_cpu.json` |
| GPU precision: the answers at Metal's default precision | `run_gate.py --precision default`, then `score.py` | `results/gate_gpu_default.json`, `results/gate_gpu_default_audio.json` |
| The source card's examples and the recording: probabilities, token counts and the largest difference from the reference | `examples/run_example.py --check --out <file>` on the CPU and with `--accel gpu`; the reference's answers are `examples/run_example.expected.json`, from `ref_run.py` | the `--out` files |
| The line that the card's own code prints | the card's code block, run from the repository root | its output |
| Speed on the Mac, the compile seconds, the 64 states | `bench_host.py`; `bench_b8_omni.py --arm b8single` | `results/bench_host_gpu_fp32.json`, `results/bench_host_cpu_fp32.json`, `results/bench_b8single_gpu32.json`, `results/bench_b8single_cpu.json` |
| Agreement on the phones | `run_replay.py` or `replay_b8_omni.py` over a phone's outputs, then `score.py` | `results/line_<name>.json`, `results/line_<name>_audio.json` |
| The Galaxy S26's GPU runs against the Mac's Metal run of the same rows | `mac_generic.py`, `leg_check_omni.py`, `leg_table_omni.py` | `results/legs.json` |
| Speed and loads on the phones, memory and thermal readings, Android versions | the phones' reports (below); `leg_table_omni.py` reads the app's reports into one table | not in this repository |
| File sizes, operators, data types and SHA-256 | the export scripts; `tflite_scan_d1.py`, `rank_dump.py`, `bucket_table.py` | `results/export_*.json`; `SHA256SUMS` |
| The NPU files: ops in one partition and the two memory lines | `npu/aot_check.py` | `out/npu/<tag>/aot_check.json` |

The phones ran the graph files in programs that are not in this repository. On the Pixel 11 Pro and on the Galaxy S26's GPU, a test app ran them through the Kotlin CompiledModel API of LiteRT 2.2.0. On the Galaxy S26's NPU, a test runner on the C API of LiteRT 2.2.0 ran the files of `npu/sm8850/`, outside the app. Each wrote a report per run with its times and its load (the compile, or for the NPU the load of the compiled context); a time is one graph call in both: writing all the inputs, running, and reading the output back. A picture or clip request on the Galaxy S26 is a tower or audio-encoder call on the GPU at float32 in the app and a decision call on the NPU in the runner, timed apart. The inputs of every phone run were laid out on the Mac by the scripts of this repository, and the outputs were read back and scored on the Mac by its scripts (`conversion/README.md`, The phones); in the Galaxy S26 NPU rows, the pictures' and clips' decision inputs are the Pixel 11 Pro's GPU tower and audio-encoder outputs at float32.
