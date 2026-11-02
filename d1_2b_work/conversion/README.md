# Conversion and check scripts

These scripts made the LiteRT graphs of this repository and ran the checks reported for it, on an Apple M4 Max (128 GB, macOS 27.0) on 2026-11-01 and 2026-11-02. Every LiteRT run goes through the CompiledModel API of `ai_edge_litert` (CPU = XNNPACK, GPU = Metal). `REPRODUCE.md` at the repository root lists the sources, the steps and the rebuild of 2026-11-02.

They are the scripts that ran. The paths of the machine they ran on are replaced by the folders of one work directory (below), and comments and docstrings are edited so that they name no file of that machine; the code is otherwise the same.

## Work directory

The scripts find their work directory `W` as the parent of their own folder, and run from `W`:

```text
W/conversion/   this folder
W/host/         host/ of this repository (the table step and the checks import it)
W/snapshot/     LiquidAI/d1-2B at revision 10e7671d44e4b37700020b1eed3f594c5f2af0b4 (code, config, tokenizer, weights)
W/tables/       tables/embed_table.npy
W/out/graphs/   the graphs, under the names of this repository
W/out/media/    the source card's photo (downloaded, not part of this repository)
W/out/pixels/   the processor's pixel values of the photo (save_pixels.py), for the tower's export check
W/out/npu/      the NPU compiles
W/results/, W/logs/
```

## Environments

Below, `$EXPORT`, `$REF` and `$HOST` are the interpreters of three Python environments:

- `$EXPORT` (export, float16 weight casting, the LiteRT checks and timing): Python 3.12.13 and `requirements-export.txt` (litert-torch 0.10.0, torch 2.14.1, transformers 5.14.1, ai-edge-litert 2.3.0, ai-edge-quantizer 0.10.0).
- `$REF` (the provider's code in float32 on the CPU): Python 3.12.11 and `requirements-ref.txt` (torch 2.14.1, torchvision 0.29.1, transformers 5.19.0).
- `$HOST` (the PyTorch-free host): Python 3.12.11 and `host/requirements-host.txt`.

## The graphs

Every export ran with `PYTHONHASHSEED=0`, one process at a time (an export peaks near three times the float32 weights in memory).

```bash
# 0. the snapshot and the photo
hf download LiquidAI/d1-2B --revision 10e7671d44e4b37700020b1eed3f594c5f2af0b4 --local-dir snapshot
mkdir -p out/media tables results logs
curl -o out/media/coco_val2017_000000039769.jpg http://images.cocodataset.org/val2017/000000039769.jpg
shasum -a 256 out/media/coco_val2017_000000039769.jpg   # dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e

# 1. the table: the tied embedding's bfloat16 bits, read out of model.safetensors
$HOST -c "import sys; sys.path.insert(0, 'host'); from d1_host import write_embed_table; write_embed_table('snapshot', 'tables/embed_table.npy')"

# 2. the decision files (out/graphs/d1-2b_decide_L<L>[_qb1024]_fp16fc.tflite, results/export_decide_*.json)
for L in 128 256 512 1024 2048; do PYTHONHASHSEED=0 $EXPORT conversion/export_decide.py --L $L; done
for L in 4096 6144; do PYTHONHASHSEED=0 $EXPORT conversion/export_decide.py --L $L --qblock 1024; done

# 3. the eight-sequence file, then the eight-signature file (the seven length graphs and the eight-sequence graph)
PYTHONHASHSEED=0 $EXPORT conversion/export_batch.py --B 8 --causal-runtime
PYTHONHASHSEED=0 $EXPORT conversion/export_multi8.py

# 4. the towers (the export check reads the photo's pixel values)
$REF conversion/save_pixels.py out/media/coco_val2017_000000039769.jpg
for C in 24x24 32x48 48x32 48x64 64x48; do PYTHONHASHSEED=0 $EXPORT conversion/export_tower.py --canvas $C; done
for C in 72x96 96x72; do PYTHONHASHSEED=0 $EXPORT conversion/export_tower.py --canvas $C --qblock 1152; done
```

Each export writes its record to `results/` (seconds, bytes, SHA-256, peak memory, versions, operator and dtype counts) and never overwrites an existing output. The tower and eight-sequence exports also check their files on the CPU against the PyTorch module. `tflite_scan_d1.py`, `rank_dump.py`, `rank_dump8.py` and `bucket_table.py` read the graphs' operators, operand ranks and largest buffers.

## The NPU files

The NPU step runs in a linux/amd64 image with ai-edge-litert 2.2.0, its Qualcomm compiler plugin and QAIRT 2.47.0, Qualcomm's SDK, which is not in this repository (`npu/Dockerfile` says where it goes). `npu/aot_check.py` checks a compile against the NPU's two memory limits. The files of this repository are the files that ran on the phone; a compile of the 128-token file on 2026-11-02 gave its SHA-256 again (`REPRODUCE.md`).

```bash
docker build --platform linux/amd64 -t npu-aot-qc:2.2.0-full conversion/npu      # with QAIRT at conversion/npu/qairt_data/
for L in 128 256 512 1024; do
  docker run --rm --platform linux/amd64 --memory 32g -v "$PWD/out/graphs":/m:ro -v "$PWD/conversion/npu":/s:ro \
    -v "$PWD/out/npu":/w npu-aot-qc:2.2.0-full \
    python /s/aot_compile_sm8850.py /m/d1-2b_decide_L${L}_fp16fc.tflite /w/L$L qualcomm_log_level=verbose
  python3 conversion/npu/aot_check.py out/npu/L$L
done
docker run --rm --platform linux/amd64 --memory 32g -v "$PWD/out/graphs":/m:ro -v "$PWD/conversion/npu":/s:ro \
  -v "$PWD/out/npu":/w npu-aot-qc:2.2.0-full \
  python /s/aot_compile_sm8850.py /m/d1-2b_decide_b8_L128_fp16fc.tflite /w/b8_L128 qualcomm_log_level=verbose
python3 conversion/npu/aot_check.py out/npu/b8_L128
```

`out/npu/<tag>/<name>_Qualcomm_SM8850_apply_plugin.tflite` is the compile; this repository names it `npu/sm8850/<name>_SM8850.tflite`.

<!-- checks:begin -->
## The checks

The checks run from `W` too, with this repository's `fixtures/` at `W/fixtures/` and the graphs in `W/out/graphs/` (the exports above, or this repository's files linked there under the same names). The commands are in the order they ran. `ref_run.py` and `run_gate.py` resume from their `--out` file when it exists, so a command run again adds only what is missing.

### Inputs

```bash
(cd fixtures && python3 rebuild_requests.py --out requests.json)    # fixtures/requests.json: the 526 test requests
curl -o fixtures/media/coco_val2017_000000039769.jpg http://images.cocodataset.org/val2017/000000039769.jpg
python3 conversion/set_files.py --requests fixtures/requests.json  # set/: requests.json, red_arms.json, extra_g96x72.json,
                                                                   # card.json, media.json, media_extra.json, media_all.json
$REF conversion/check_set.py                                       # set/token_counts.json
PYTHONHASHSEED=0 $EXPORT conversion/export_multi.py                # out/graphs/d1-2b_decide_multi7_fp16fc.tflite: the seven
                                                                   # length graphs in one file (not in this repository)
```

`set_files.py` writes the files the checks read from the test requests: 506 test requests, 19 control requests, the one request of the 96 × 72 canvas, the card's 5 examples, and the media tables. The photo must be in `fixtures/media/` for them and in `out/media/` for the tower export (step 0 above): the same file.

### The reference

```bash
$REF conversion/ref_run.py --files set/requests.json set/red_arms.json --media set/media.json --out out/ref/ref_fp32.json
$REF conversion/ref_run.py --files set/extra_g96x72.json --media set/media_extra.json --out out/ref/ref_fp32.json
$REF conversion/ref_run.py --files set/card.json --out out/ref/card_fp32.json --card
python3 conversion/pick_arms.py      # conversion/red_arms_chosen_arms.json: the six control requests
python3 conversion/make_subsets.py   # conversion/ids_L128.json (the 382 requests that fit 128 tokens), red_arms_L128.json
```

`out/ref/ref_fp32.json` (with its answer slots in `out/ref/ref_fp32_slots.npz`) holds the provider's answers in float32 on the CPU to the 526 requests: 520 answered, 5 refused, 1 too long for the largest graph. `pick_arms.py` and `make_subsets.py` read no output of the graphs: they ran before any check of the graphs.

### Agreement on the Mac

```bash
for spec in "gpu fp32" "cpu fp32" "gpu default"; do      # Metal at float32, the CPU (8 threads), Metal at its default precision
  set -- $spec
  $EXPORT conversion/run_gate.py --accel $1 --precision $2 --files set/requests.json set/red_arms.json \
    --media set/media.json --out out/gate/$1_$2.json
  $EXPORT conversion/run_gate.py --accel $1 --precision $2 --files set/requests.json set/red_arms.json \
    set/extra_g96x72.json --media set/media_extra.json --out out/gate/$1_$2.json
  $REF conversion/score.py --ref out/ref/ref_fp32.json --cand out/gate/$1_$2.json \
    --files set/requests.json set/red_arms.json set/extra_g96x72.json \
    --red-arms conversion/red_arms_chosen_arms.json --out results/gate_$1_$2.json
done

# the eight-sequence file, 8 sequences a call: first every batch row against the single 128-token file (stops above 1e-3)
$EXPORT conversion/check_batch.py --accel gpu --out results/check_batch_gpu32.json
$EXPORT conversion/check_batch.py --accel cpu --out results/check_batch_cpu.json
IDS=$(python3 -c 'import json; print(",".join(json.load(open("conversion/ids_L128.json"))["ids"]))')
for a in "gpu gpu32" "cpu cpu"; do
  set -- $a
  $EXPORT conversion/run_gate_batch.py --accel $1 --batch-file out/graphs/d1-2b_decide_b8_L128_fp16fc.tflite \
    --ids-file conversion/ids_L128.json --files set/requests.json set/red_arms.json --out out/gate/L128_b8_$2.json
  $REF conversion/score.py --ref out/ref/ref_fp32.json --cand out/gate/L128_b8_$2.json \
    --files set/requests.json set/red_arms.json --red-arms conversion/red_arms_L128.json --ids "$IDS" \
    --out results/gate_L128_b8_$2.json
done

# the seven length graphs in one file with constant tensor sharing, on Metal at float32
$EXPORT conversion/run_gate.py --accel gpu --precision fp32 --multi out/graphs/d1-2b_decide_multi7_fp16fc.tflite \
  --sharing --files set/requests.json set/red_arms.json set/extra_g96x72.json --media set/media_all.json \
  --out out/gate/gpu_fp32_multi_sharing.json
$REF conversion/score.py --ref out/ref/ref_fp32.json --cand out/gate/gpu_fp32_multi_sharing.json \
  --files set/requests.json set/red_arms.json set/extra_g96x72.json --red-arms conversion/red_arms_chosen_arms.json \
  --out results/gate_gpu_fp32_multi_sharing.json

# the eight-signature file against the seven-signature file and the eight-sequence file, output for output
$EXPORT conversion/multi8_check.py --accel gpu --out results/multi8_check_gpu.json
$EXPORT conversion/multi8_check.py --accel cpu --out results/multi8_check_cpu.json
```

`results/gate_<accelerator>_<precision>.json`, `gate_L128_b8_<accelerator>.json` and `gate_gpu_fp32_multi_sharing.json` hold the agreement rows: per kind of question the number of questions, max and mean |Δ| against the reference, the near ties, and the controls. `gate_gpu_default.json` is the run at the GPU's default precision.

### Speed on the Mac

```bash
$EXPORT conversion/bench_host.py --accel gpu --precision fp32 --out results/bench_host_gpu_fp32.json
$EXPORT conversion/bench_host.py --accel cpu --precision fp32 --workloads one_question,3_questions,5k_state,384px,box,boxes \
  --out results/bench_host_cpu_fp32.json
$EXPORT conversion/bench_batch.py --mode batch --arm b8 --accel gpu --out results/bench_batch_b8_gpu32.json
$EXPORT conversion/bench_batch.py --mode batch --arm b8 --accel cpu --out results/bench_batch_b8_cpu.json
```

One process per command, with nothing else heavy on the machine. `bench_host_*` hold the requests from one question to boxes and the compile seconds; `bench_batch_b8_*` the 64 states.

### The phones

The phones ran the graph files in programs that are not in this repository: a test app on the Kotlin CompiledModel API of LiteRT 2.2.0 (the Pixel 11 Pro, and the Galaxy S26's GPU and CPU) and, for the Galaxy S26's NPU, a test runner on the C API of LiteRT 2.2.0. Both run inputs that the host laid out on the Mac (the rows files of the scripts below), one graph signature per call, and write the outputs and a report with the times; a time is one graph call: writing the inputs, running, and reading the output back. These scripts make the rows:

```bash
$EXPORT conversion/make_rows.py static --set conversion/set_ids.json --out-dir out/rows   # the device set: 42 requests
$EXPORT conversion/make_rows_b8.py --out-dir out/rows_b8                                  # the 64 states: 8 calls of 8
```

A picture's decision rows are made on the Mac from the phone's own tower outputs: `$EXPORT conversion/make_rows.py pic --rows-dir out/rows --round-dir <phone folder> --src-tag <tower run> --bucket <L> --name <rows> --out-dir <phone folder>/gen --with-text --allow-missing`.

The host on the Mac reads the phones' outputs (`run_replay.py`: it finds each call's output by the SHA-256 of the call's input bytes), and `score.py` scores the requests whose calls all ran:

```bash
phone_pass() {   # phone_pass <name> <phone folders...> -- <run prefixes...>
  local name=$1; shift; local dirs=() pfx=()
  while [ "$1" != -- ]; do dirs+=("$1"); shift; done; shift
  for p in "$@"; do pfx+=(--prefix "$p"); done
  $EXPORT conversion/run_replay.py --round-dir "${dirs[@]}" "${pfx[@]}" --out out/replay/$name.json
  IDS=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1]))["records"]; print(",".join(r for r, v in d.items() if not v.get("error")))' out/replay/$name.json)
  $REF conversion/score.py --ref out/ref/ref_fp32.json --cand out/replay/$name.json \
    --files set/requests.json set/red_arms.json set/extra_g96x72.json --red-arms conversion/red_arms_chosen_arms.json \
    --ids "$IDS" --out results/line_$name.json
}
```

One pass per phone row: the Pixel 11 Pro's seven-signature file with sharing on the GPU and its single files and towers on the CPU (42 requests each); the Galaxy S26's 128-token file on the GPU and on the CPU (16 requests); the Galaxy S26's NPU files (the 35 requests that fit 1,024 tokens). In the Galaxy S26 rows, the pictures' decision inputs are the Pixel 11 Pro's GPU tower outputs at float32. A row per NPU file is `score.py --ids` with the requests whose decision call ran on that file, without `--red-arms` (a control request and its base can fall in two files). The 64 states run on the eight-sequence graph:

```bash
$EXPORT conversion/replay_b8.py --round-dir <phone folder> --prefix <eight-sequence run> --rows-dir <the rows it ran> \
  --out out/replay/<name>.json
W64=$(python3 -c 'print(",".join(f"wl_64_states_{i:02d}" for i in range(64)))')
$REF conversion/score.py --ref out/ref/ref_fp32.json --cand out/replay/<name>.json --files set/requests.json \
  --ids "$W64" --out results/line_<name>.json
```

The phones' folders are not in this repository, so these rows can be scored again only from a new phone run.
<!-- checks:end -->
