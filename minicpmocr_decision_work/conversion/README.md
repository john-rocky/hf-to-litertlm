# Conversion and check scripts

These scripts made the LiteRT graphs of this repository and ran the checks reported for it, on an Apple M4 Max (128 GB, macOS 27.0) on 2026-11-05 and 2026-11-06. Every LiteRT run goes through the CompiledModel API of `ai_edge_litert` (CPU = XNNPACK, GPU = Metal on the Mac).

They are the scripts that ran. The paths of the machine they ran on are replaced by the folders of one work directory (below), comments and docstrings are edited so that they name no file of that machine, and an export with no fixtures takes a blank sample (see "The graphs"); the code is otherwise the same.

## Work directory

The scripts find their work directory as the parent of their own folder (or as their first argument), and run from it. It is a new folder with a copy of `conversion/`, `host/` and `examples/`:

```text
conversion/   this folder
conversion/sets/   record ids and sizes of the benchmark's olmOCR family (no picture, no text)
host/         host/ of this repository (the exports and the checks import its formulas)
snapshot/     openbmb/MiniCPM-OCR-Jev at revision a0ed278eb6aa1f333a711665ab610055a4a816ab
author/       the author's score.py and evaluate.py and the three official examples (fetch_inputs.py, not part of this repository)
bench/        openbmb/JevOCRBench rows and pictures (fetch_inputs.py, not part of this repository)
out/graphs/   the graphs, under the names of this repository
out/fix/      the author's numbers for the graph checks (fixtures_author.py)
out/npu/      the NPU compiles
results/, logs/
```

## Environments

Below, `$EXPORT`, `$REF` and `$HOST` are the interpreters of three Python environments:

- `$EXPORT` (export, float16 weight casting, the LiteRT checks): Python 3.12.13 and `requirements-export.txt` (litert-torch 0.10.0, torch 2.14.1, transformers 5.14.1, ai-edge-litert 2.3.0, ai-edge-quantizer 0.10.0).
- `$REF` (the author's code in float32 on the CPU): Python 3.12.11 and `requirements-ref.txt` (torch 2.8.0, torchvision 0.23.0, transformers 5.10.4, the versions the source's `docs/finetune_jev_en.md` names).
- `$HOST` (the PyTorch-free host): Python 3.12.11 and `host/requirements-host.txt`.

## The graphs

Every export ran with `PYTHONHASHSEED=0`, one process at a time. A decision export peaks at 16.5 to 18.7 GiB of footprint, the multi-signature decision file near 26.5 GiB, a tower export at 6.2 to 10 GiB.

```bash
# 0. the snapshot and the tables
hf download openbmb/MiniCPM-OCR-Jev --revision a0ed278eb6aa1f333a711665ab610055a4a816ab --local-dir snapshot
python3 conversion/make_tables.py

# 1. the single decision files, in the f16s form with the kernel forms bt and cv (out/graphs/decide_L<L>[_qb1024]_f16sfc.tflite, results/export_decide_*_f16s.json)
for L in 256 384 512 768 1024 1536 2048 3072; do PYTHONHASHSEED=0 $EXPORT conversion/export_decide.py --L $L --form R64+sp+ec+dd+vs6+bt+cv --variant f16s; done
PYTHONHASHSEED=0 $EXPORT conversion/export_decide.py --L 4096 --qblock 1024 --form R64+sp+ec+dd+vs6+bt+cv --variant f16s

# 2. the towers (out/graphs/tower_N<N>[_qb1024]_fp16fc.tflite, results/export_tower_*.json)
for N in 400 576 784 1024 1296 1600 1936 2304 3136; do PYTHONHASHSEED=0 $EXPORT conversion/export_tower.py --N $N; done
PYTHONHASHSEED=0 $EXPORT conversion/export_tower.py --N 4096 --qblock 1024

# 3. the two multi-signature files (the decision file in the f16s form on the default kernel form, without bt and cv; the tower file in the plain form)
PYTHONHASHSEED=0 $EXPORT conversion/export_multi.py --variant f16s
PYTHONHASHSEED=0 $EXPORT conversion/export_multi_tower.py
```

`--variant f16s` multiplies the final norm's input by 2^-4 and its epsilon by 4^-4, and caps its reciprocal square root at the value for a zero input (`apply_f16s` in `decide_graph.py`): the plain graph's final norm sums squares up to 600,543, more than float16 holds, so at a GPU's default precision the plain file answers wrong and this form does not. Without `--variant f16s` the same commands write the plain form (`_fp16fc`). At float32 the two forms gave the same outputs bit for bit on the records we compared: on the CPU, all 417 records of the olmOCR set that the files hold; on Metal at float32, the 83 records of the phones' test set that the 256- to 2,048-token files hold (12, 29, 28 and 14).

`--form R64+sp+ec+dd+vs6+bt+cv` adds two kernel forms to the default `R64+sp+ec+dd+vs6` (`FORM` in `decide_graph.py`; `kernel_q35.py`): `bt` computes the GatedDeltaNet's delta rule with every chunk as one batch row and the terms that do not read the state once over all chunks, and `cv` runs the layer's causal depthwise conv in the layout the converter keeps (`conv_layout.py`), without the transposes around it: the same computation in fewer operators. The multi-signature decision file of step 3 and the NPU files below are on the default kernel form.

Each export writes its record to `results/` (seconds, bytes, SHA-256, peak memory, versions, operator and dtype counts) and never overwrites an existing output. With fixtures in `out/fix/` (below), an export also compares its module and its float32 file with the author's numbers, and takes its sample from a fixture; with none, the sample is a blank record of seeded random values and the float32 file is compared with the module alone. The sample only carries the shapes through the export: REPRODUCE.md compares the bytes.

`quant_arms.py` makes the integer arms from a float32 file (`--keep-fp32` keeps it). `tiny_check.py` and `tiny_multi.py` run tiny random graphs through the whole path in a few minutes, with no checkpoint: run them first after a change of versions.

## The NPU files

The files in `npu/sm8850/` are compiled ahead of time for the NPU of the Qualcomm SM8850 (Galaxy S26), from the `--variant f16s` form of the graph, exported with `--sum-after in_proj_z,q_proj --suffix _z` (the 256-, 512- and 1,024-token files also with step 1's `--form R64+sp+ec+dd+vs6+bt+cv`, the 384- and 768-token files on the default kernel form, without it): each of those bias-free linears (18 `in_proj_z` and 6 `q_proj`) becomes a batch-1 matrix product, so that the converter puts a size-1 SUM between the linear and the sigmoid that reads it, and the export writes `out/graphs/decide_L<L>_z_f16sfc.tflite`, a graph this repository does not ship. The f16s form scales the input of the norms that leave float16's range by a power of two (`apply_f16s` in `decide_graph.py`; for the tower, `F16S_BLOCK_LOG2` and `F16S_MERGER_LOG2` in `tower_graph.py`, a form this repository does not ship), so a graph in float32 gives the same values. The compile runs in a linux/amd64 image with ai-edge-litert 2.2.0, its Qualcomm compiler plugin and QAIRT 2.47.0, Qualcomm's SDK, which is not in this repository (`npu/Dockerfile` says where it goes). `npu/aot_check.py` checks a compile against the NPU's two memory limits.

```bash
docker build --platform linux/amd64 -t npu-aot-qc:2.2.0-full conversion/npu      # with QAIRT at conversion/npu/qairt_data/

PYTHONHASHSEED=0 $EXPORT conversion/export_decide.py --L 256 --form R64+sp+ec+dd+vs6+bt+cv --variant f16s --sum-after in_proj_z,q_proj --suffix _z      # out/graphs/decide_L256_z_f16sfc.tflite
docker run --rm --platform linux/amd64 --cpus 4 --memory 26g --memory-swap 26g -v "$PWD/out/graphs":/m:ro -v "$PWD/conversion/npu":/s:ro \
  -v "$PWD/out/npu":/w npu-aot-qc:2.2.0-full \
  python /s/aot_compile_sm8850.py /m/decide_L256_z_f16sfc.tflite /w/z_L256 qualcomm_log_level=verbose
python3 conversion/npu/aot_check.py out/npu/z_L256

PYTHONHASHSEED=0 $EXPORT conversion/export_tower.py --N 400 --variant f16s       # out/graphs/tower_N400_f16sfc.tflite
docker run --rm --platform linux/amd64 --cpus 4 --memory 16g --memory-swap 16g -v "$PWD/out/graphs":/m:ro -v "$PWD/conversion/npu":/s:ro \
  -v "$PWD/out/npu":/w npu-aot-qc:2.2.0-full \
  python /s/aot_compile_sm8850.py /m/tower_N400_f16sfc.tflite /w/w_N400 qualcomm_log_level=verbose
python3 conversion/npu/aot_check.py out/npu/w_N400
```

The other files take another `--L` (384, 512, 768 and 1,024 for the decision files, with `--form R64+sp+ec+dd+vs6+bt+cv` for 512 and 1,024) or `--N`. `--memory` is the container's limit, and a longer file needs a higher one: the files in `npu/sm8850/` were compiled with 26, 28, 30, 30 and 32 GB, for 256, 384, 512, 768 and 1,024 tokens. The file to ship is `out/npu/<tag>/<name>_Qualcomm_SM8850_apply_plugin.tflite`, as `npu/sm8850/<name>_SM8850.tflite`.

## The checks

```bash
python3 conversion/fetch_inputs.py author
python3 conversion/fetch_inputs.py fixtures      # the records of sets/fixtures.json
python3 conversion/fetch_inputs.py family        # the 446 records of the olmOCR family (892 files, about 40 MB)

# the author's numbers: the three official examples, then the fixtures, in the reference environment
$REF -I conversion/ref_run.py . --examples --dtype fp32 --fast-dwconv
$REF -I conversion/ref_run.py . --set A --dtype fp32 --fast-dwconv
$REF -I conversion/fixtures_author.py . --pass 1
$REF -I conversion/fixtures_author.py . --pass 2      # the large records and pictures, 15.3 GiB of footprint

# a graph against the author's numbers, the host not in the path
$EXPORT conversion/check_graph.py . decide --file out/graphs/decide_L256_f16sfc.tflite --size 256 --acc cpu,gpu_default,gpu_f32
$EXPORT conversion/check_graph.py . tower --file out/graphs/tower_N400_fp16fc.tflite --size 400 --acc cpu,gpu_f32

# a multi-signature file against the single files, bit for bit
$EXPORT conversion/multi_check.py . decide --multi out/graphs/multi_decide_L4096_f16sfc.tflite --form f16sfc --acc gpu_default --sharing
$EXPORT conversion/multi_check.py . tower --multi out/graphs/multi_tower_N4096_fp16fc.tflite --acc gpu_f32 --sharing

# the host over the set, then the lines against the reference: the decision graph at the GPU's default precision, the tower at float32
$EXPORT conversion/run_set.py . --backend litert --form f16sfc --tower-form fp16fc --acc gpu_default --tower-acc gpu_f32 --set A --tag A__base__f16s__gpu_default_towerf32
$EXPORT conversion/run_set.py . --backend litert --form f16sfc --tower-form fp16fc --acc gpu_default --tower-acc gpu_f32 --set examples --tag examples__f16s__gpu_default_towerf32
python3 -I conversion/judge.py . compare A__base__f16s__gpu_default_towerf32
python3 -I conversion/judge.py . p1 examples__f16s__gpu_default_towerf32
# on the CPU: --form f16sfc --tower-form fp16fc --acc cpu --tag A__base__f16s__cpu
# the Mac's rows that the phones' lines compare with: the plain decision files (export_decide.py without --variant) and the towers, at float32 on Metal
$EXPORT conversion/run_set.py . --backend litert --form fp16fc --acc gpu_f32 --set A --tag A__base__fp16fc__gpu_f32

# the controls on 36 records: another candidate's text (R1), the two pictures swapped (R2), another original picture (R4).
# judge.py red also prints R3, read from the base rows with no run of its own: the logits one position before the pooled token. It does not move the reference's AUROC, so it is not one of the controls.
for arm in R1 R2 R4; do
  $REF -I conversion/ref_run.py . --set red --arm $arm --dtype fp32 --fast-dwconv
  $EXPORT conversion/run_set.py . --backend litert --form f16sfc --tower-form fp16fc --acc gpu_default --tower-acc gpu_f32 --set red --arm $arm --tag red__${arm}__f16s__gpu_default_towerf32
done
python3 -I conversion/judge.py . red f16s__gpu_default_towerf32

# one timed cell, on a quiet machine: one process per workload (text, formula, table), the single files; for the two multi-signature files with constant tensor sharing add
# --multi out/graphs/multi_decide_L4096_f16sfc.tflite --multi-tower out/graphs/multi_tower_N4096_fp16fc.tflite --sharing
$EXPORT conversion/speed.py . --form f16sfc --tower-form fp16fc --acc gpu_default --tower-acc gpu_f32 --workload text --suffix __towerf32
```

`run_set.py --backend torch` in `$REF` is the separation test: the author's float32 modules behind the host's interfaces, so that everything the host computes is under test and no graph is. `examples/score_olmocr_set.py` is the short way to score the same family with this repository's files and the host alone.

## The checks behind the form

Four checks behind the host's picture path, the f16s form and the choice of no fixed picture size. The first reads the pictures that are in `bench/`, so with `fetch_inputs.py` it reads the olmOCR family; the card's number comes from all 4,461 pictures of the benchmark and the examples.

```bash
$REF -I conversion/pixel_check.py .                                  # the host's resize against the author's and Pillow's: results/pixel_check.json
PYTHONHASHSEED=0 $EXPORT conversion/fp16_probe.py .                  # in PyTorch on the fixtures: what the norms sum, against float16's largest value: results/fp16_probe.json
for ARM in S448 P448; do $REF -I conversion/ref_run.py . --set device --arm $ARM --dtype fp32 --fast-dwconv; done
$REF -I conversion/fixed_arms.py .                                   # a fixed picture size against the reference: results/fixed_size_arms.json
$REF -I conversion/ref_run.py . --set floor --dtype fp32 --fast-dwconv
$REF -I conversion/ref_run.py . --set floor --dtype bf16
$REF -I conversion/reference_floor.py .                              # the source's bfloat16 form against its float32 form: results/reference_floor.json
```

## The phones

The phones ran the graph files in programs that are not in this repository: a test app on the Kotlin CompiledModel API of LiteRT 2.2.0 (the Pixel 11 Pro and the Galaxy S26's GPU) and, for the Galaxy S26's NPU, a test runner on the C API of LiteRT 2.2.0. Both run rows files: each row is one graph call's input tensors, laid out on the Mac by the host, and they write each output tensor to a file, little-endian float32, with a report per run. The rows hold the pictures' values, so they stay under `out/`.

```bash
$EXPORT conversion/make_rows.py static --out-dir out/rows       # the 90 records of sets/set_device.json, per patch bucket; the fixed workload
```

A decision row for the GPU holds the phone's own tower output at the picture positions, so it is made on the Mac after the tower runs, from the phone's folder of reports and outputs (in the NPU's runs only the decision graph ran on the NPU, and its rows hold the Mac's tower output):

```bash
$EXPORT conversion/make_rows.py pic --rows-dir out/rows --round-dir <phone folder> --src-prefix <tower runs> --bucket <L> --name <rows.json> --out-dir <phone folder>/gen
$EXPORT conversion/make_rows.py timing --rows-dir out/rows --round-dir <phone folder> --src-prefix <tower runs> --out-dir out/rows      # the fixed workload's decision rows, tmL<L>.json
```

The host on the Mac then reads the phone's outputs back: `run_replay.py` finds each call's output by the SHA-256 of the call's input bytes, and refuses a call no run answered. `judge_device.py` scores the pass against the Mac's form and the reference:

```bash
$EXPORT conversion/run_replay.py --round-dir <phone folder> --prefix <runs> --tag <name>
python3 -I conversion/judge_device.py . <name> --mac A__base__fp16fc__gpu_f32 --out results/line_<name>.json
```

`mac_generic.py` does a phone run's work on the Mac from the same rows files, which tells the replay's own error from a phone's: `$EXPORT conversion/mac_generic.py --graph out/graphs/<file> --leg <tag>=<rows.json> --out-dir <folder> --acc gpu_f32`.

The phones' folders are not in this repository, so these rows can be scored again only from a new phone run.
