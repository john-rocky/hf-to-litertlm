# Conversion and check scripts

These scripts made the LiteRT graphs of this repository and ran the checks reported for it, on an Apple M4 Max (128 GB, macOS 27.0) on 2026-11-05. Every LiteRT run goes through the CompiledModel API of `ai_edge_litert` (CPU = XNNPACK, GPU = Metal on the Mac).

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
- `$REF` (the author's code in float32 on the CPU): Python 3.12.11 and `requirements-ref.txt` (torch 2.8.0, torchvision 0.23.0, transformers 5.10.4, the versions the source's card names).
- `$HOST` (the PyTorch-free host): Python 3.12.11 and `host/requirements-host.txt`.

## The graphs

Every export ran with `PYTHONHASHSEED=0`, one process at a time. A decision export peaks near 16.5 GiB of footprint, the multi-signature decision file near 26.5 GiB, a tower export near 10 GiB.

```bash
# 0. the snapshot and the tables
hf download openbmb/MiniCPM-OCR-Jev --revision a0ed278eb6aa1f333a711665ab610055a4a816ab --local-dir snapshot
python3 conversion/make_tables.py

# 1. the decision files (out/graphs/decide_L<L>[_qb1024]_fp16fc.tflite, results/export_decide_*.json)
for L in 256 512 1024 2048; do PYTHONHASHSEED=0 $EXPORT conversion/export_decide.py --L $L; done
PYTHONHASHSEED=0 $EXPORT conversion/export_decide.py --L 4096 --qblock 1024

# 2. the towers (out/graphs/tower_N<N>[_qb1024]_fp16fc.tflite, results/export_tower_*.json)
for N in 400 576 1024 1600 2304; do PYTHONHASHSEED=0 $EXPORT conversion/export_tower.py --N $N; done
PYTHONHASHSEED=0 $EXPORT conversion/export_tower.py --N 4096 --qblock 1024

# 3. the two multi-signature files
PYTHONHASHSEED=0 $EXPORT conversion/export_multi.py
PYTHONHASHSEED=0 $EXPORT conversion/export_multi_tower.py
```

Each export writes its record to `results/` (seconds, bytes, SHA-256, peak memory, versions, operator and dtype counts) and never overwrites an existing output. With fixtures in `out/fix/` (below), an export also compares its module and its float32 file with the author's numbers, and takes its sample from a fixture; with none, the sample is a blank record of seeded random values and the float32 file is compared with the module alone. The sample only carries the shapes through the export: REPRODUCE.md compares the bytes.

`quant_arms.py` makes the integer arms from a float32 file (`--keep-fp32` keeps it). `tiny_check.py` and `tiny_multi.py` run tiny random graphs through the whole path in a few minutes, with no checkpoint: run them first after a change of versions.

## The NPU files

The files in `npu/sm8850/` are compiled ahead of time for the NPU of the Qualcomm SM8850 (Galaxy S26), from the `--variant f16s` form of the graph. That form scales the input of the norms that leave float16's range by a power of two (`apply_f16s` in `decide_graph.py`, `F16S_BLOCK_LOG2` and `F16S_MERGER_LOG2` in `tower_graph.py`), so a graph in float32 gives the same values. The compile runs in a linux/amd64 image with ai-edge-litert 2.2.0, its Qualcomm compiler plugin and QAIRT 2.47.0, Qualcomm's SDK, which is not in this repository (`npu/Dockerfile` says where it goes). `npu/aot_check.py` checks a compile against the NPU's two memory limits.

```bash
docker build --platform linux/amd64 -t npu-aot-qc:2.2.0-full conversion/npu      # with QAIRT at conversion/npu/qairt_data/

PYTHONHASHSEED=0 $EXPORT conversion/export_decide.py --L 256 --variant f16s      # out/graphs/decide_L256_f16sfc.tflite
docker run --rm --platform linux/amd64 --cpus 4 --memory 26g --memory-swap 26g -v "$PWD/out/graphs":/m:ro -v "$PWD/conversion/npu":/s:ro \
  -v "$PWD/out/npu":/w npu-aot-qc:2.2.0-full \
  python /s/aot_compile_sm8850.py /m/decide_L256_f16sfc.tflite /w/s_L256 qualcomm_log_level=verbose
python3 conversion/npu/aot_check.py out/npu/s_L256

PYTHONHASHSEED=0 $EXPORT conversion/export_tower.py --N 400 --variant f16s       # out/graphs/tower_N400_f16sfc.tflite
docker run --rm --platform linux/amd64 --cpus 4 --memory 16g --memory-swap 16g -v "$PWD/out/graphs":/m:ro -v "$PWD/conversion/npu":/s:ro \
  -v "$PWD/out/npu":/w npu-aot-qc:2.2.0-full \
  python /s/aot_compile_sm8850.py /m/tower_N400_f16sfc.tflite /w/w_N400 qualcomm_log_level=verbose
python3 conversion/npu/aot_check.py out/npu/w_N400
```

The other files take another `--L` or `--N`. `--memory` is the container's limit, and a longer file needs a higher one. The file to ship is `out/npu/<tag>/<name>_Qualcomm_SM8850_apply_plugin.tflite`, as `npu/sm8850/<name>_SM8850.tflite`.

## The checks

```bash
python3 conversion/fetch_inputs.py author
python3 conversion/fetch_inputs.py fixtures      # the records of sets/fixtures.json
python3 conversion/fetch_inputs.py family        # the 446 records of the olmOCR family (892 files, about 40 MB)

# the author's numbers: the three official examples, then the fixtures, in the reference environment
$REF -I conversion/ref_run.py . --examples --dtype fp32 --fast-dwconv
$REF -I conversion/ref_run.py . --set A --dtype fp32 --fast-dwconv
$REF -I conversion/fixtures_author.py . --pass 1
$REF -I conversion/fixtures_author.py . --pass 2      # the large records and pictures, near 21 GiB

# a graph against the author's numbers, the host not in the path
$EXPORT conversion/check_graph.py . decide --file out/graphs/decide_L256_fp16fc.tflite --size 256 --acc cpu,gpu_f32
$EXPORT conversion/check_graph.py . tower --file out/graphs/tower_N400_fp16fc.tflite --size 400 --acc cpu,gpu_f32

# a multi-signature file against the single files, bit for bit
$EXPORT conversion/multi_check.py . decide --multi out/graphs/multi_decide_phone_fp16fc.tflite --acc gpu_f32 --sharing
$EXPORT conversion/multi_check.py . tower --multi out/graphs/multi_tower_phone_fp16fc.tflite --acc gpu_f32 --sharing

# the host over the set, then the lines against the reference
$EXPORT conversion/run_set.py . --backend litert --form fp16fc --acc gpu_f32 --set A
$EXPORT conversion/run_set.py . --backend litert --form fp16fc --acc gpu_f32 --set examples --tag examples__fp16fc__gpu_f32
python3 -I conversion/judge.py . compare A__base__fp16fc__gpu_f32
python3 -I conversion/judge.py . p1 examples__fp16fc__gpu_f32

# the controls on 36 records: another candidate's text (R1), the two pictures swapped (R2), another original picture (R4)
for arm in R1 R2 R4; do
  $REF -I conversion/ref_run.py . --set red --arm $arm --dtype fp32 --fast-dwconv
  $EXPORT conversion/run_set.py . --backend litert --form fp16fc --acc gpu_f32 --set red --arm $arm
done
python3 -I conversion/judge.py . red fp16fc__gpu_f32

# one timed cell, on a quiet machine: one process per workload (text, formula, table), the two multi-signature files with constant tensor sharing, or leave the three flags out for the single files
$EXPORT conversion/speed.py . --form fp16fc --acc gpu_f32 --workload text --multi out/graphs/multi_decide_phone_fp16fc.tflite --multi-tower out/graphs/multi_tower_phone_fp16fc.tflite --sharing --suffix __multi_sharing
```

`run_set.py --backend torch` in `$REF` is the separation test: the author's float32 modules behind the host's interfaces, so that everything the host computes is under test and no graph is. `examples/score_olmocr_set.py` is the short way to score the same family with this repository's files and the host alone.

## The checks behind the form

Four checks behind the host's picture path, the float32 precision and the choice of no fixed picture size. The first reads the pictures that are in `bench/`, so with `fetch_inputs.py` it reads the olmOCR family; the card's number comes from all 4,461 pictures of the benchmark and the examples.

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
