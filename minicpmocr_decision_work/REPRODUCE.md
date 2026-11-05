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
3. Decision files: `export_decide.py --L <L> --variant f16s` for 256, 512, 1,024 and 2,048 tokens, and `--L 4096 --qblock 1024 --variant f16s` for 4,096 (the full-attention layers' queries in blocks of 1,024, which bounds the memory of their scores). Each exports `decide_graph.py` at float32 with its final norm in the f16s form (`apply_f16s`: the norm's input times 2^-4, its epsilon times 4^-4, its reciprocal square root capped at the value for a zero input) and casts its fully connected weights to float16 with ai-edge-quantizer (`fp16fc.py`); the activations stay float32. They write `decide_L<L>[_qb1024]_f16sfc.tflite`. Without `--variant f16s` the same commands write the plain form, `decide_L<L>[_qb1024]_fp16fc.tflite`, which this repository does not ship; at float32 the two forms gave the same outputs bit for bit on the records the card names.
4. Towers: `export_tower.py --N <N>` for 400, 576, 1,024, 1,600 and 2,304 patches, and `--N 4096 --qblock 1024` for 4,096. `tower_graph.py`, the same cast.
5. The two multi-signature files: `export_multi.py --variant f16s` (the decision graphs for 256, 512, 1,024, 2,048 and 4,096 tokens over one copy of the weights) and `export_multi_tower.py` (the towers for 400, 576, 1,024, 1,600, 2,304 and 4,096 patches). They write `multi_decide_L4096_f16sfc.tflite` and `multi_tower_N4096_fp16fc.tflite`. The tower file was named `multi_tower_phone_fp16fc.tflite` in the reports of the checks: the same bytes.
6. NPU files: `conversion/npu/aot_compile_sm8850.py` compiles a decision file of step 3 for the SM8850 in the image of `conversion/npu/Dockerfile`, and `conversion/npu/aot_check.py` reads the compile. `export_tower.py --N <N> --variant f16s` writes a tower with scaled norms for the same compile, a form this repository does not ship.
7. Checks and timing: see `conversion/README.md`.
8. The shipped set: the graphs are the work directory's `out/graphs/` files under the same names; `npu/sm8850/<name>_SM8850.tflite` is the compile's `<name>_Qualcomm_SM8850_apply_plugin.tflite`. `SHA256SUMS` lists every file.

## The rebuilds

With no fixtures in place an export takes a blank record of seeded random values as its sample and compares its float32 file with its module alone. With the fixtures in place it takes a fixture record and compares both with the author's numbers. The sample only carries the shapes through the export.

### 2026-11-06: the decision files

On 2026-11-06 the steps ran again from a clean copy of this repository's `conversion/`, `host/` and `examples/` folders, in a new work directory, on the pinned snapshot (its files checked against the Hub's file list). One single decision file and the multi-signature decision file were rebuilt; the other decision files come from the same script with another `--L`.

<!-- dayof7:begin -->
| File | Export sample | Rebuilt SHA-256 | Shipped SHA-256 | Result |
|---|---|---|---|---|
| `decide_L256_f16sfc.tflite` | blank record | `bc22d87912df27ab8b7faec4707794067d108fce1a6667e1ff457334fc9fceaa` | `bc22d87912df27ab8b7faec4707794067d108fce1a6667e1ff457334fc9fceaa` | the same bytes |
| `multi_decide_L4096_f16sfc.tflite` | blank record | `14f6fbb19a11b6c91bb87e11bed87095f160e5c2ffd1b16b34ed2597e9ab4ada` | `14f6fbb19a11b6c91bb87e11bed87095f160e5c2ffd1b16b34ed2597e9ab4ada` | the same bytes |
<!-- dayof7:end -->

### 2026-11-05: the other files

On 2026-11-05, and past midnight into 2026-11-06 for the last checks and the multi-signature files' second export, the steps ran from a clean copy of the scripts of that day in a new work directory, with an anonymous download of the pinned snapshot (with `HF_HUB_DISABLE_XET=1`, checked against the Hub's file list). The table keeps the rows of the files in this repository: the tables, one tower, the multi-signature tower file, and the 256-token decision file with its NPU compile; the multi-signature tower file's blank-record row ran under its earlier name. The scripts of that day are this repository's but for `export_multi.py`'s new option `--variant` and the host's precision per graph (`host/decide_host/load.py`, and the two examples that print it or name their cache by it).

<!-- dayof:begin -->
| File | Export sample | Rebuilt SHA-256 | Shipped SHA-256 | Result |
|---|---|---|---|---|
| `tables/tables.safetensors` | not an export | `8fa765826c3676521fa3cc6af61f1cc2c783b129c9564051a0abc28d51678986` | `8fa765826c3676521fa3cc6af61f1cc2c783b129c9564051a0abc28d51678986` | the same bytes |
| `tower_N400_fp16fc.tflite` | blank record | `4e66c14e26780c61b7b18f62469cfe41054268912973881e862e8ec2b194f244` | `4e66c14e26780c61b7b18f62469cfe41054268912973881e862e8ec2b194f244` | the same bytes |
| `multi_tower_N4096_fp16fc.tflite` | blank record | `6b071a7b40c367134578229d5cad9c8a3f9d85578cf80612ff437cc35aad3181` | `6b071a7b40c367134578229d5cad9c8a3f9d85578cf80612ff437cc35aad3181` | the same bytes |
| `tower_N400_fp16fc.tflite` | fixture record | `4e66c14e26780c61b7b18f62469cfe41054268912973881e862e8ec2b194f244` | `4e66c14e26780c61b7b18f62469cfe41054268912973881e862e8ec2b194f244` | the same bytes |
| `decide_L256_f16sfc.tflite` | fixture record (the input of the compile below) | `bc22d87912df27ab8b7faec4707794067d108fce1a6667e1ff457334fc9fceaa` | `bc22d87912df27ab8b7faec4707794067d108fce1a6667e1ff457334fc9fceaa` | the same bytes |
| `npu/sm8850/decide_L256_f16sfc_SM8850.tflite` | the compile of the row above | `646be785d0f7e26c89d85259d875df608b15e535fa13fa642666741bb35c04ed` | `646be785d0f7e26c89d85259d875df608b15e535fa13fa642666741bb35c04ed` | the same bytes |
| `multi_tower_N4096_fp16fc.tflite` | fixture record | `6b071a7b40c367134578229d5cad9c8a3f9d85578cf80612ff437cc35aad3181` | `6b071a7b40c367134578229d5cad9c8a3f9d85578cf80612ff437cc35aad3181` | the same bytes |
<!-- dayof:end -->

From that copy the checks ran again on a few records each: the reference on the official examples and three records; `check_graph.py` and `multi_check.py` on the rebuilt files; `judge.py p1` and `compare`; the separation test; the control arms on three records; `speed.py` with one warm-up and two timed requests; the phone tools, with the Mac in the phone's place; `pixel_check.py` on 60 pictures and the examples, `fp16_probe.py` on its default records, the fixed-size arms on six records and the bfloat16 floor on two. `examples/score_olmocr_set.py` scored the family's 446 records with the plain decision files, this repository's towers and the host alone on Metal at float32: on the 417 it scores, `prob_1`, the logits, the SHA-256 of the token ids and of the hidden row equal the run behind that day's checks, and it returns `too_large` for the other 29. The card's numbers come from the full runs of the same scripts, on 2026-11-05 and on 2026-11-06 for the decision files at the GPU's default precision.

## NPU files

The files in `npu/sm8850/` are the files that ran on the Galaxy S26: their SHA-256 are the ones the phone read before it ran them. `conversion/npu/aot_check.py` reads the plugin's log of a compile: every op in one partition; (i) the plugin's memory estimate for the graph within its limit of 3,670,016 KiB (the init graph option `memory_alloc_limit`, 3,584 MiB); (ii) the context binary plus the graphs' memory beyond their constants plus 64 MiB within the 3,750,000,000 bytes of one compute-DSP session.

<!-- npu:begin -->
| File | Input graph (SHA-256) | Ops in one partition | (i) estimate, KiB | (ii) session, bytes | Bytes | SHA-256 |
|---|---|---:|---:|---:|---:|---|
| `decide_L256_f16sfc_SM8850.tflite` | `decide_L256_f16sfc.tflite` (`bc22d87912df27ab…`) | 5,915 / 5,915 | 2,742,904 | 2,902,213,120 | 2,783,922,048 | `646be785d0f7e26c89d85259d875df608b15e535fa13fa642666741bb35c04ed` |
| `decide_L1024_f16sfc_SM8850.tflite` | `decide_L1024_f16sfc.tflite` (`2fa06fec44afd9b8…`) | 13,475 / 13,475 | 2,920,688 | 3,180,470,784 | 2,882,094,976 | `b6ce69bfd9346a2c96599b59171075ba051bcb865d4dd4484c7ffea05f5d1c37` |

The input graphs are this repository's `decide_L256_f16sfc.tflite` and `decide_L1024_f16sfc.tflite`, the same SHA-256. On 2026-11-05 a compile in the image of `conversion/npu/Dockerfile`, from the rebuilt input graph, gave the shipped SHA-256 of `decide_L256_f16sfc_SM8850.tflite`; the others were not compiled again.

A 512-token file, `decide_L512_f16sfc_SM8850.tflite` (2,819,958,656 bytes, SHA-256 `e1556ac9380d1a828abdf714fa87f784b98eee19016825b062bc3b8f322b1203`), was compiled the same way and ran on the phone. It is not in this repository: its scores are outside the tolerance (see the model card).
<!-- npu:end -->

## Where the numbers come from

Each number on the model card comes from a file that a command of `conversion/README.md` writes, from the example, or, for the phones' times, from the reports of the test programs:

| Numbers on the card | Command | File |
|---|---|---|
| The reference: the author's code in float32 on the CPU, its AUROC and the official examples | `ref_run.py --set A`, `ref_run.py --examples`, then `judge.py summary` | `results/ref/A__base__fp32__dw.jsonl`, `results/ref/examples__fp32__dw.json` |
| Agreement on the Mac, one family of records | `run_set.py --set A`, then `judge.py compare` | `results/gate/<tag>.jsonl` and the JSON that `judge.py` prints |
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
