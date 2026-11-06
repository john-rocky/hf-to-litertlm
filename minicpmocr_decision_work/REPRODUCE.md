# Reproduce the LiteRT files and the checks

The conversion ran on a Mac (Apple M4 Max, 128 GB, macOS 27.0) on 2026-11-05 and 2026-11-06. The scripts are in [`conversion/`](conversion/README.md), with the commands. They are the scripts that ran, with the machine's paths replaced by the folders of one work directory and the comments edited to name no file of that machine.

## Sources

The source is `openbmb/MiniCPM-OCR-Jev` at revision `a0ed278eb6aa1f333a711665ab610055a4a816ab`:

<!-- sources:begin -->
| File | Bytes | SHA-256 |
|---|---:|---|
| `model.safetensors` | 4,426,567,144 | `6e8e43049340a9d18f711634dcc88eb5880a3cd69c852815e76d5de3aea28dcc` |
| `tokenizer.json` | 19,989,325 | `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523` |
| `config.json` | 2,775 | `fd3d48eacdf595b290afc504ac05aa4a4e44343a227406e60eb61516dc772b34` |
| `chat_template.jinja` | 7,755 | `273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80` |
| `preprocessor_config.json` | 390 | `27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516` |
| `processor_config.json` | 1,191 | `d89ef49ce9cd37fbf510158e13c1ef063d9286411c1ec9049932dbe0487143b1` |
| `tokenizer_config.json` | 1,165 | `792fa3f0cb88b111e54ef3134c873531008c4df471d108da17903426e308aa7b` |
| `generation_config.json` | 176 | `daa38670fac2e808a88a68c925f3af81e62b5447bcd81c048756753c0b1cf8d6` |
| `scripts/minicpm-ocr-jev/eval/score.py` (the author's repository) | 12,863 | `de26bfc2b92b2c8df64caa826f8606784948bc24a59438c50605af4f34198fa2` |
| `scripts/minicpm-ocr-jev/eval/evaluate.py` (the author's repository) | 2,948 | `f56309e00c7756804e798046c83ea6f947fd684739025a28a9dd00fe0dfb0d49` |
<!-- sources:end -->

The author's `eval/score.py` and `eval/evaluate.py` and the three official examples (`examples/`) are at commit `8177040f0d6c824ccf87bc18b938b6a0f91b6d0e` of `OpenBMB/MiniCPM-OCR`, in `scripts/minicpm-ocr-jev/`. The benchmark is `openbmb/JevOCRBench`, of which only the olmOCR family is used (446 records, 892 files); `conversion/fetch_inputs.py` holds its pinned revision. None of its pictures or texts are in this repository: `conversion/sets/` holds record ids and sizes only.

## Environments

- Export, float16 weight casting and the LiteRT checks on the Mac: Python 3.12.13, `conversion/requirements-export.txt` (litert-torch 0.10.0, torch 2.14.1, transformers 5.14.1, ai-edge-litert 2.3.0, ai-edge-quantizer 0.10.0).
- The author's code in float32 on the CPU: Python 3.12.11, `conversion/requirements-ref.txt` (torch 2.8.0, torchvision 0.23.0, transformers 5.10.4).
- The Python host and the examples: Python 3.12.11, `host/requirements-host.txt` (ai-edge-litert 2.3.0, numpy 2.5.4, pillow 13.0.0, tokenizers 0.22.2). No PyTorch, torchvision or transformers.
- The NPU compile: the image of `conversion/npu/Dockerfile` (linux/amd64: Python 3.12.13, ai-edge-litert 2.2.0 and its Qualcomm compiler plugin, QAIRT 2.47.0.260601114230, which is Qualcomm's SDK and is not in this repository).

## Steps

The commands are in `conversion/README.md`. Every export ran with `PYTHONHASHSEED=0`.

1. Snapshot: an anonymous `hf download` of the pinned revision, checked against the Hub's file list.
2. Tables: `make_tables.py` writes `tables/tables.safetensors`, three tensors as bfloat16 bits as they are in the source's `model.safetensors`: the token embeddings, the picture position table and the score weights. The host reads them; no graph holds them.
3. Decision files: `export_decide.py --L <L> --form R64+sp+ec+dd+vs6+bt+cv --variant f16s` for 256, 384, 512, 768, 1,024, 1,536, 2,048 and 3,072 tokens, and `--L 4096 --qblock 1024 --form R64+sp+ec+dd+vs6+bt+cv --variant f16s` for 4,096 (the full-attention layers' queries in blocks of 1,024, which bounds the memory of their scores). Each exports `decide_graph.py` at float32 with its final norm in the f16s form (`apply_f16s`: the norm's input times 2^-4, its epsilon times 4^-4, its reciprocal square root capped at the value for a zero input) and the gated linear attention layers in the kernel forms `bt` and `cv` (`kernel_q35.py`: the delta rule over all chunks at once; `conv_layout.py`: the causal convolution in the converter's layout), and casts its fully connected weights to float16 with ai-edge-quantizer (`fp16fc.py`); the activations stay float32. They write `decide_L<L>[_qb1024]_f16sfc.tflite`. Without `--variant f16s` the same commands write the plain form, `decide_L<L>[_qb1024]_fp16fc.tflite`, which this repository does not ship; at float32 the two forms gave the same outputs bit for bit on the records the card names. Without `--form` they write the default kernel form, without `bt` and `cv`: the form of the first update's decision files, of step 5 and of step 6's 384- and 768-token files.
4. Towers: `export_tower.py --N <N>` for 400, 576, 784, 1,024, 1,296, 1,600, 1,936, 2,304 and 3,136 patches, and `--N 4096 --qblock 1024` for 4,096. `tower_graph.py`, the same cast.
5. The two multi-signature files: `export_multi.py --variant f16s` (the decision graphs for 256, 512, 1,024, 2,048 and 4,096 tokens over one copy of the weights, on the default kernel form) and `export_multi_tower.py` (the towers for 400, 576, 1,024, 1,600, 2,304 and 4,096 patches). They write `multi_decide_L4096_f16sfc.tflite` and `multi_tower_N4096_fp16fc.tflite`.
6. NPU files: `export_decide.py --L <L> --variant f16s --sum-after in_proj_z,q_proj --suffix _z` writes `decide_L<L>_z_f16sfc.tflite`, with `--form R64+sp+ec+dd+vs6+bt+cv` for 256, 512 and 1,024 tokens and on the default kernel form, without it, for 384 and 768: the decision graph of step 3 with each of those 24 bias-free linears as a batch-1 matrix product, so that the converter puts a size-1 SUM between the linear and the sigmoid that reads it; this repository does not ship these graphs. `conversion/npu/aot_compile_sm8850.py` compiles one for the SM8850 in the image of `conversion/npu/Dockerfile`, and `conversion/npu/aot_check.py` reads the compile. `export_tower.py --N <N> --variant f16s` writes a tower with scaled norms for the same compile, a form this repository does not ship.
7. Checks and timing: see `conversion/README.md`.
8. The shipped set: the graphs are the work directory's `out/graphs/` files under the same names; `npu/sm8850/<name>_SM8850.tflite` is the compile's `<name>_Qualcomm_SM8850_apply_plugin.tflite`. `SHA256SUMS` lists every file.

## The rebuilds

With no fixtures in place an export takes a blank record of seeded random values as its sample and compares its float32 file with its module alone. With the fixtures in place it takes a fixture record and compares both with the author's numbers. The sample only carries the shapes through the export.

### 2026-11-06: the decision files, a tower and the NPU files

On 2026-11-06 the steps ran again in five new work directories, on the snapshot downloaded on 2026-11-05 (its files checked again against the Hub's file list), each from a copy of this repository's `conversion/`, `host/` and `examples/` folders made that day. The first three copies did not have the kernel forms `bt` and `cv` yet (earlier `conversion/kernel_q35.py` and `conversion/patch_q35.py`, no `conversion/conv_layout.py`) and had an earlier `conversion/README.md`; the first also had a `conversion/export_decide.py` without `--sum-after`. The multi-signature decision file was rebuilt from the first copy, and the 784-patch tower was exported again from the third; the other half-step towers come from the same script with another `--N`. The fourth copy differs from the files here in `conversion/README.md` only, and the fifth in the same file only: from the fourth the 384-token decision file was exported again with the command of step 3, and from the fifth the 384-token NPU file's input graph was exported and compiled again with the commands of step 6; the other files of these steps come from the same commands with another `--L`. The other files of each copy are the ones here.

<!-- dayof7:begin -->
| File | Export sample | Rebuilt SHA-256 | Shipped SHA-256 | Result |
|---|---|---|---|---|
| `multi_decide_L4096_f16sfc.tflite` | blank record | `14f6fbb19a11b6c91bb87e11bed87095f160e5c2ffd1b16b34ed2597e9ab4ada` | `14f6fbb19a11b6c91bb87e11bed87095f160e5c2ffd1b16b34ed2597e9ab4ada` | the same bytes |
| `tower_N784_fp16fc.tflite` | blank record | `4d6cdc7130641db90a9036b8ec18187703de2218e631ca95db1a43ce080e6642` | `4d6cdc7130641db90a9036b8ec18187703de2218e631ca95db1a43ce080e6642` | the same bytes |
| `decide_L384_f16sfc.tflite` | blank record | `c93ddd7239325ca92aa84f8fa681e610f07709228ae792afb20a16e12794e832` | `c93ddd7239325ca92aa84f8fa681e610f07709228ae792afb20a16e12794e832` | the same bytes |
| `decide_L384_z_f16sfc.tflite` | blank record (the input of the compile below, not in this repository) | `a25f595b24ff433676e631d57af7a661af9076745c7625d89189b509f12e6c3c` | `a25f595b24ff433676e631d57af7a661af9076745c7625d89189b509f12e6c3c` | the same bytes |
| `npu/sm8850/decide_L384_z_f16sfc_SM8850.tflite` | the compile of the row above | `ecd1787e6e05e7e790ecd836e6b2f2436dd00e35d0f239e71706c35a32c8608e` | `ecd1787e6e05e7e790ecd836e6b2f2436dd00e35d0f239e71706c35a32c8608e` | the same bytes |
<!-- dayof7:end -->

### 2026-11-05: the other files

On 2026-11-05, and past midnight into 2026-11-06 for the last checks and the multi-signature files' second export, the steps ran from a clean copy of the scripts of that day in a new work directory, with an anonymous download of the pinned snapshot (with `HF_HUB_DISABLE_XET=1`, checked against the Hub's file list). The table keeps the rows of the files in this repository: the tables, one tower and the multi-signature tower file. The scripts of that day are this repository's but for `export_decide.py`'s new option `--sum-after`, `export_multi.py`'s new option `--variant`, the kernel forms `bt` and `cv` (`kernel_q35.py`, `patch_q35.py` and the new `conv_layout.py`), and the host's precision per graph (`host/decide_host/load.py`, and the two examples that print it or name their cache by it), and its `conversion/README.md` was an earlier version.

<!-- dayof:begin -->
| File | Export sample | Rebuilt SHA-256 | Shipped SHA-256 | Result |
|---|---|---|---|---|
| `tables/tables.safetensors` | not an export | `8fa765826c3676521fa3cc6af61f1cc2c783b129c9564051a0abc28d51678986` | `8fa765826c3676521fa3cc6af61f1cc2c783b129c9564051a0abc28d51678986` | the same bytes |
| `tower_N400_fp16fc.tflite` | blank record | `4e66c14e26780c61b7b18f62469cfe41054268912973881e862e8ec2b194f244` | `4e66c14e26780c61b7b18f62469cfe41054268912973881e862e8ec2b194f244` | the same bytes |
| `multi_tower_N4096_fp16fc.tflite` | blank record | `6b071a7b40c367134578229d5cad9c8a3f9d85578cf80612ff437cc35aad3181` | `6b071a7b40c367134578229d5cad9c8a3f9d85578cf80612ff437cc35aad3181` | the same bytes |
| `tower_N400_fp16fc.tflite` | fixture record | `4e66c14e26780c61b7b18f62469cfe41054268912973881e862e8ec2b194f244` | `4e66c14e26780c61b7b18f62469cfe41054268912973881e862e8ec2b194f244` | the same bytes |
| `multi_tower_N4096_fp16fc.tflite` | fixture record | `6b071a7b40c367134578229d5cad9c8a3f9d85578cf80612ff437cc35aad3181` | `6b071a7b40c367134578229d5cad9c8a3f9d85578cf80612ff437cc35aad3181` | the same bytes |
<!-- dayof:end -->

From that copy the checks ran again on a few records each: the reference on the official examples and three records; `check_graph.py` and `multi_check.py` on the rebuilt files; `judge.py p1` and `compare`; the separation test; the control arms on three records; `speed.py` with one warm-up and two timed requests; the phone tools, with the Mac in the phone's place; `pixel_check.py` on 60 pictures and the examples, `fp16_probe.py` on its default records, the fixed-size arms on six records and the bfloat16 floor on two. `examples/score_olmocr_set.py` scored the family's 446 records with the plain decision files, this repository's towers and the host alone on Metal at float32: on the 417 it scores, `prob_1`, the logits, the SHA-256 of the token ids and of the hidden row equal the run behind that day's checks, and it returns `too_large` for the other 29. The card's numbers come from the full runs of the same scripts, on 2026-11-05 and on 2026-11-06 for the decision files at the GPU's default precision.

## NPU files

The files in `npu/sm8850/` are the files that ran on the Galaxy S26: their SHA-256 are the ones the phone read before it ran them. `conversion/npu/aot_check.py` reads the plugin's log of a compile: every op in one partition; (i) the plugin's memory estimate for the graph within its limit of 3,670,016 KiB (the init graph option `memory_alloc_limit`, 3,584 MiB); (ii) the context binary plus the graphs' memory beyond their constants plus 64 MiB within the 3,750,000,000 bytes of one compute-DSP session.

<!-- npu:begin -->
| File | Input graph (SHA-256) | Ops in one partition | (i) estimate, KiB | (ii) session, bytes | Bytes | SHA-256 |
|---|---|---:|---:|---:|---:|---|
| `decide_L256_z_f16sfc_SM8850.tflite` | `decide_L256_z_f16sfc.tflite` (`5d98340c1b598572…`) | 4,337 / 4,337 | 2,742,644 | 2,901,176,832 | 2,782,812,032 | `614816e97404285f9308e605ccc7be95e3a3430072d610875d2530bb6156b61f` |
| `decide_L384_z_f16sfc_SM8850.tflite` | `decide_L384_z_f16sfc.tflite` (`a25f595b24ff4336…`) | 7,199 / 7,199 | 2,775,444 | 2,962,976,768 | 2,811,529,088 | `ecd1787e6e05e7e790ecd836e6b2f2436dd00e35d0f239e71706c35a32c8608e` |
| `decide_L512_z_f16sfc_SM8850.tflite` | `decide_L512_z_f16sfc.tflite` (`8f33bfa6c03f3926…`) | 5,345 / 5,345 | 2,784,940 | 2,977,820,672 | 2,816,538,496 | `4ebd605aed89a67248768b21df1d96e0701f790522cd22ff8d00448482cee54e` |
| `decide_L768_z_f16sfc_SM8850.tflite` | `decide_L768_z_f16sfc.tflite` (`0c30e0bfcce9d7d7…`) | 10,979 / 10,979 | 2,842,516 | 3,078,684,672 | 2,859,440,000 | `b18e5658be3a497491f9a9221fa9b06c0795302813f4a3bca9939be27611d660` |
| `decide_L1024_z_f16sfc_SM8850.tflite` | `decide_L1024_z_f16sfc.tflite` (`687a98cf57468093…`) | 7,361 / 7,361 | 2,905,352 | 3,160,297,472 | 2,877,286,272 | `384399a74405fe8753fade2835226bed2ffff33f5e8a35159031acd63a878853` |

The input graphs `decide_L256_z_f16sfc.tflite`, `decide_L384_z_f16sfc.tflite`, `decide_L512_z_f16sfc.tflite`, `decide_L768_z_f16sfc.tflite` and `decide_L1024_z_f16sfc.tflite` are not in this repository: `export_decide.py --L <L> --variant f16s --sum-after in_proj_z,q_proj --suffix _z` writes them, for 256, 384, 512, 768 and 1,024 tokens, with `--form R64+sp+ec+dd+vs6+bt+cv` for 256, 512 and 1,024 (step 6). On 2026-11-06 a compile in the image of `conversion/npu/Dockerfile`, from the rebuilt input graph, gave the shipped SHA-256 of `decide_L384_z_f16sfc_SM8850.tflite`. The others were not compiled again. The compiles of these files ran with container memory limits (`--memory`) of 26, 28, 30, 30 and 32 GB, for 256, 384, 512, 768 and 1,024 tokens. A compile can come out with other bytes: of 4 compiles of one input graph in that image (the 384-token decision graph with `bt` and `cv`, not in this repository), at 26 and 28 GB, 3 gave the same bytes and one differed from them in 688 bytes, in 28 ranges of numeric data.
<!-- npu:end -->

## Where the numbers come from

Each number on the model card comes from a file that a command of `conversion/README.md` writes, from the example, or, for the phones' times, from the reports of the test programs:

| Numbers on the card | Command | File |
|---|---|---|
| The reference: the author's code in float32 on the CPU, its AUROC and the official examples | `ref_run.py --set A`, `ref_run.py --examples`, then `judge.py summary` | `results/ref/A__base__fp32__dw.jsonl`, `results/ref/examples__fp32__dw.json` |
| Agreement on the Mac, one family of records | `run_set.py --set A`, then `judge.py compare` | `results/gate/<tag>.jsonl` and the JSON that `judge.py` prints |
| Two sets of files on the same records: the same outputs bit for bit, or the largest difference of the score | `run_set.py --set A` for each set, or `run_replay.py` for each phone run, then the two sets' rows compared by record (no script in this repository) | `results/gate/<tag>.jsonl`; the replays' `.jsonl` |
| The three official examples, on the CPU and on Metal | `examples/run_example.py --check --out <file>`; `run_set.py --set examples`, then `judge.py p1` | that file; `results/gate/examples__*.json` |
| A graph against the author's numbers, the host not in the path; GPU precision | `check_graph.py` with `--acc cpu,gpu_f32,gpu_default` | `results/check_<file stem>.json` |
| A multi-signature file against the single files, its load and its memory | `multi_check.py` | `results/multi_check_*.json` |
| The separation test (the host's computations against the author's) | `run_set.py --backend torch`, then `judge.py host` | the JSON that `judge.py` prints |
| Speed on the Mac | `speed.py` | `results/speed/*.json` |
| File sizes, operators, data types and SHA-256 | the export scripts | `results/export_*.json`; `SHA256SUMS` |
| The NPU files: ops in one partition and the two memory limits | `npu/aot_check.py` | `out/npu/<tag>/aot_check.json` |
| The host's resize against the author's and Pillow's (the card's number: all 4,461 pictures of the benchmark and the examples; `fetch_inputs.py` downloads the olmOCR family) | `pixel_check.py` | `results/pixel_check.json` |
| float16's range: what the final norm sums | `fp16_probe.py` | `results/fp16_probe.json` |
| A fixed picture size against the reference | `ref_run.py --set device --arm S448`, `--arm P448`, then `fixed_arms.py` | `results/fixed_size_arms.json` |
| The source's bfloat16 form against its float32 form | `ref_run.py --set floor`, both dtypes, then `reference_floor.py` | `results/reference_floor.json` |
| Agreement on the phones | `make_rows.py`, the test programs (not in this repository), `run_replay.py`, `judge_device.py` | `results/line_<name>.json` |
| The phones' times, loads and memory | the reports of the test programs (not in this repository) | |
