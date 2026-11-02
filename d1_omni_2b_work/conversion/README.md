# Conversion and check scripts

These scripts made the LiteRT graphs of this repository and ran the checks reported for it, on an Apple M4 Max (128 GB, macOS 27.0) on 2026-11-02. Every LiteRT run goes through the CompiledModel API of `ai_edge_litert` (CPU = XNNPACK, GPU = Metal). `REPRODUCE.md` at the repository root lists the sources, the steps and the rebuild of 2026-11-02.

They are the scripts that ran. The paths of the machine they ran on are replaced by the folders of one work directory (below), and comments and docstrings are edited so that they name no file of that machine; the code is otherwise the same. The decision graph and the tower are the d1-2B conversion's modules (`d1_graph.py`, `d1_graph_b.py`, `d1_tower.py`), which read this model's weights from `W/snapshot/`.

## Work directory

The scripts find their work directory `W` as the parent of their own folder, and run from `W`:

```text
W/conversion/   this folder
W/host/         host/ of this repository (the table step and the checks import it)
W/snapshot/     LiquidAI/d1-Omni-2B at revision 431f1d667de5afcc9b0c58b2d9c170a7b38b358b (code, config, tokenizer, weights)
W/tables/       tables/embed_table.npy
W/out/graphs/   the graphs, under the names of this repository
W/out/media/    the source card's photo (downloaded, not part of this repository)
W/out/pixels/   the processor's pixel values of the photo (save_pixels.py), for the tower's export check
W/out/feats/    the provider feature extractor's features of synthetic clips (audio_check.py), for the audio export check
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
hf download LiquidAI/d1-Omni-2B --revision 431f1d667de5afcc9b0c58b2d9c170a7b38b358b --local-dir snapshot
mkdir -p out/media tables results logs
curl -o out/media/coco_val2017_000000039769.jpg http://images.cocodataset.org/val2017/000000039769.jpg
shasum -a 256 out/media/coco_val2017_000000039769.jpg   # dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e

# 1. the table: the tied embedding's bfloat16 bits, read out of model.safetensors
$HOST -c "import sys; sys.path.insert(0, 'host'); from d1_host import write_embed_table; write_embed_table('snapshot', 'tables/embed_table.npy')"

# 2. the decision files (out/graphs/d1-omni-2b_decide_L<L>[_qb1024]_fp16fc.tflite, results/export_decide_*.json)
for L in 128 256 512 1024 2048; do PYTHONHASHSEED=0 $EXPORT conversion/export_decide.py --L $L; done
for L in 4096 6144; do PYTHONHASHSEED=0 $EXPORT conversion/export_decide.py --L $L --qblock 1024; done

# 3. the eight-sequence file, then the eight-signature file (the seven length graphs and the eight-sequence graph)
PYTHONHASHSEED=0 $EXPORT conversion/export_b8_omni.py
PYTHONHASHSEED=0 $EXPORT conversion/export_multi8_omni.py

# 4. the towers (the export check reads the photo's pixel values)
$REF conversion/save_pixels.py out/media/coco_val2017_000000039769.jpg
for C in 24x24 32x48 48x32 48x64 64x48; do PYTHONHASHSEED=0 $EXPORT conversion/export_tower.py --canvas $C; done
for C in 72x96 96x72; do PYTHONHASHSEED=0 $EXPORT conversion/export_tower.py --canvas $C --qblock 1152; done

# 5. the audio encoders (the export check reads the features of synthetic clips)
$REF conversion/audio_check.py --synthetic --save-feats out/feats --out results/audio_check_synthetic.json
for T in 504 1000 2000 3000; do PYTHONHASHSEED=0 $EXPORT conversion/export_audio.py --T $T; done
```

Each export writes its record to `results/` (seconds, bytes, SHA-256, peak memory, versions, operator and dtype counts) and never overwrites an existing output. The tower and audio exports also check their files on the CPU against the PyTorch module. `tflite_scan_d1.py`, `rank_dump.py`, `rank_dump8.py` and `bucket_table.py` read the graphs' operators, operand ranks and largest buffers.

## The NPU files

The NPU step runs in a linux/amd64 image with ai-edge-litert 2.2.0, its Qualcomm compiler plugin and QAIRT 2.47.0, Qualcomm's SDK, which is not in this repository (`npu/Dockerfile` says where it goes). `npu/aot_check.py` checks a compile against the NPU's two memory limits. The files of this repository are the files that ran on the phone; they were not compiled again on 2026-11-02.

```bash
docker build --platform linux/amd64 -t npu-aot-qc:2.2.0-full conversion/npu      # with QAIRT at conversion/npu/qairt_data/
for L in 128 256 512; do
  docker run --rm --platform linux/amd64 --memory 32g -v "$PWD/out/graphs":/m:ro -v "$PWD/conversion/npu":/s:ro \
    -v "$PWD/out/npu":/w npu-aot-qc:2.2.0-full \
    python /s/aot_compile_sm8850.py /m/d1-omni-2b_decide_L${L}_fp16fc.tflite /w/L$L qualcomm_log_level=verbose
  python3 conversion/npu/aot_check.py out/npu/L$L
done
docker run --rm --platform linux/amd64 --memory 32g -v "$PWD/out/graphs":/m:ro -v "$PWD/conversion/npu":/s:ro \
  -v "$PWD/out/npu":/w npu-aot-qc:2.2.0-full \
  python /s/aot_compile_sm8850.py /m/d1-omni-2b_decide_b8_L128_fp16fc.tflite /w/b8_L128 qualcomm_log_level=verbose
python3 conversion/npu/aot_check.py out/npu/b8_L128
```

`out/npu/<tag>/<name>_Qualcomm_SM8850_apply_plugin.tflite` is the compile; this repository names it `npu/sm8850/<name>_SM8850.tflite`. The 1,024-token file is not in this repository: its compile passes `aot_check.py`, but on the phone one box edge moved outside the tolerance.

<!-- checks:begin -->
## The checks

The checks run from `W` too, with this repository's `fixtures/` and `examples/` at `W/fixtures/` and `W/examples/`, and the graphs in `W/out/graphs/` (the exports above, or this repository's files linked there under the same names). The commands are in the order they ran. `ref_run.py`, `run_gate.py` and `run_gate_m8.py` resume from their `--out` file when it exists, so a command run again adds only what is missing. Every gate and scorer below gets the same six request files and three media tables:

```bash
FILES="set/requests.json set/extra_g96x72.json set/red_arms.json set/requests_audio.json set/red_arms_audio.json set/card_omni.json"
MEDIA="set/media.json set/media_extra.json set/media_audio.json"
```

### Inputs

```bash
(cd fixtures && python3 rebuild_requests.py --out requests.json)    # fixtures/requests.json: the 597 test requests
cp out/media/coco_val2017_000000039769.jpg fixtures/media/          # the photo of step 0 above
$HOST -c "import sys; sys.path.insert(0, 'examples'); import numpy as np, run_example; np.save('fixtures/media/card_beeps_4s.npy', run_example.recording()[0])"
python3 conversion/set_files.py --requests fixtures/requests.json  # set/: the six request files and three media tables
PYTHONHASHSEED=0 $EXPORT conversion/export_multi.py                # out/graphs/d1-omni-2b_decide_multi7_fp16fc.tflite: the
                                                                   # seven length graphs in one file (not in this repository)
```

`set_files.py` writes the files the checks read from the test requests: 506 text and picture test requests, the one request of the 96 × 72 canvas, 19 control requests, 55 audio test requests, 12 audio control requests, the card's 4 examples, and the media tables. The card's recording is made by the card's own code (`fixtures/README.md`).

### The reference

```bash
$REF conversion/ref_run.py --card-exact --out out/ref/card_exact_fp32.json   # the card's four code blocks
$REF conversion/ref_run.py --files set/requests.json set/extra_g96x72.json set/red_arms.json \
  --media set/media.json set/media_extra.json --out out/ref/ref_fp32.json
$REF conversion/ref_run.py --files set/requests_audio.json set/red_arms_audio.json set/card_omni.json \
  --media set/media_audio.json --out out/ref/ref_fp32.json
$REF conversion/pick_arms.py --arms set/red_arms.json --prefix conversion/red_arms_chosen
$REF conversion/pick_arms.py --arms set/red_arms_audio.json --prefix conversion/red_arms_audio_chosen
$REF conversion/id_sets.py           # conversion/ids_textpic.json (529 requests), conversion/ids_audio.json (68)
$HOST conversion/make_ids_L128.py    # conversion/ids_L128*.json and red_arms*_L128.json: the requests that fit 128 tokens
```

`out/ref/ref_fp32.json` (with its answer slots in `out/ref/ref_fp32_slots.npz`) holds the provider's answers in float32 on the CPU to the 597 requests: of the 529 text and picture requests, 523 answered, 5 refused and 1 too long for the largest graph; of the 68 audio requests, 67 answered and 1 clip too long. `pick_arms.py`, `id_sets.py` and `make_ids_L128.py` read no output of the graphs: they ran before any check of the graphs.

### Agreement on the Mac

```bash
for spec in "gpu fp32" "cpu fp32" "gpu default"; do      # Metal at float32, the CPU (8 threads), Metal at its default precision
  set -- $spec
  $EXPORT conversion/run_gate.py --accel $1 --precision $2 --files $FILES --media $MEDIA \
    --decide-keep 1 --tower-keep 0 --save-every 5 --out out/gate/$1_$2.json
  $REF conversion/score.py --ref out/ref/ref_fp32.json --cand out/gate/$1_$2.json --files $FILES \
    --red-arms conversion/red_arms_chosen_arms.json --ids-file conversion/ids_textpic.json --out results/gate_$1_$2.json
  $REF conversion/score.py --ref out/ref/ref_fp32.json --cand out/gate/$1_$2.json --files $FILES \
    --red-arms conversion/red_arms_audio_chosen_arms.json --ids-file conversion/ids_audio.json \
    --out results/gate_$1_$2_audio.json
done

# the eight-signature file against the seven-signature file and the eight-sequence file, output for output
$EXPORT conversion/multi8_check_omni.py --accel gpu --out results/multi8_check_gpu.json
$EXPORT conversion/multi8_check_omni.py --accel cpu --out results/multi8_check_cpu.json   # also writes out/xnn_cache/multi8_fp16fc.xnncache

# the eight-signature file with constant tensor sharing, on Metal at float32
$EXPORT conversion/run_gate_m8.py --mode full --accel gpu --out out/gate/full_m8_gpu32.json
$REF conversion/score.py --ref out/ref/ref_fp32.json --cand out/gate/full_m8_gpu32.json --files $FILES \
  --red-arms conversion/red_arms_chosen_arms.json --ids-file conversion/ids_textpic.json \
  --out results/gate_full_m8_gpu32_textpic.json
$REF conversion/score.py --ref out/ref/ref_fp32.json --cand out/gate/full_m8_gpu32.json --files $FILES \
  --red-arms conversion/red_arms_audio_chosen_arms.json --ids-file conversion/ids_audio.json \
  --out results/gate_full_m8_gpu32_audio.json

# the eight-sequence graph, 8 sequences a call: the eight-sequence file on Metal, the eight-signature file on the CPU
$EXPORT conversion/run_gate_m8.py --mode b8 --accel gpu --b8single out/graphs/d1-omni-2b_decide_b8_L128_fp16fc.tflite \
  --out out/gate/b8_b8single_gpu32.json
$EXPORT conversion/run_gate_m8.py --mode b8 --accel cpu --weight-cache out/xnn_cache/multi8_fp16fc.xnncache \
  --out out/gate/b8_m8_cpu.json
for tag in b8_b8single_gpu32 b8_m8_cpu; do
  $REF conversion/score.py --ref out/ref/ref_fp32.json --cand out/gate/$tag.json --files $FILES \
    --red-arms conversion/red_arms_L128.json --ids-file conversion/ids_L128_textpic.json \
    --out results/gate_${tag}_textpic.json
done
```

`results/gate_<accelerator>_<precision>.json` and `gate_<accelerator>_<precision>_audio.json`, `gate_full_m8_gpu32_*.json` and `gate_b8_*_textpic.json` hold the agreement rows: per kind of question the number of questions, max and mean |Δ| against the reference, the near ties, and the controls. `gate_gpu_default*.json` are the runs at the GPU's default precision.

### Speed on the Mac

```bash
$EXPORT conversion/bench_host.py --accel gpu --precision fp32 --out results/bench_host_gpu_fp32.json
$EXPORT conversion/bench_host.py --accel cpu --precision fp32 --workloads one_question,3_questions,5k_state,384px,audio_10s \
  --out results/bench_host_cpu_fp32.json
$EXPORT conversion/bench_host.py --accel cpu --precision fp32 --workloads 64_states --out results/bench_host_cpu_fp32.json
$EXPORT conversion/bench_b8_omni.py --arm b8single --accel gpu --out results/bench_b8single_gpu32.json
$EXPORT conversion/bench_b8_omni.py --arm b8single --accel cpu --out results/bench_b8single_cpu.json
```

One process per command, with nothing else heavy on the machine; the third command adds its workload to the same file. `bench_host_*` hold the requests from one question to the 10 s clip and the compile seconds; `bench_b8single_*` the 64 states.

### The phones

The phones ran the graph files in programs that are not in this repository: a test app on the Kotlin CompiledModel API of LiteRT 2.2.0 (the Pixel 11 Pro, and the Galaxy S26's GPU) and, for the Galaxy S26's NPU, a test runner on the C API of LiteRT 2.2.0. Both run inputs that the host laid out on the Mac (the rows files of the scripts below), one graph signature per call, and write the outputs and a report with the times; a time is one graph call: writing the inputs, running, and reading the output back. These scripts make the rows:

```bash
$EXPORT conversion/make_rows.py static --set conversion/set_ids.json --out-dir out/rows   # the device set: 55 requests
$EXPORT conversion/make_rows_b8_omni.py --out-dir out/rows_b8   # the timed rows of the eight-signature file and the 64 states
```

The media requests' decision rows are made on the Mac from the phone's own tower and audio-encoder outputs: `$EXPORT conversion/make_rows.py prep --kind aud|pa --rows-dir out/rows --round-dir <phone folder> --src-tag <tower or audio run> --bucket <L> --name <rows> --out-dir <phone folder>/gen --allow-missing`.

The host on the Mac reads the phones' outputs (`run_replay.py`: it finds each call's output by the SHA-256 of the call's input bytes), and `score.py` scores the requests whose calls all ran, in two rows: text and pictures, and audio:

```bash
phone_pass() {   # phone_pass <name> <phone folder> <run prefix>
  $EXPORT conversion/run_replay.py --round-dir $2 --prefix $3 --out out/replay/$1.json
  for row in ":conversion/ids_textpic_set.json:conversion/red_arms_chosen_arms.json" \
             "_audio:conversion/ids_audio_set.json:conversion/red_arms_audio_chosen_arms.json"; do
    IFS=: read -r suf ids arms <<< "$row"
    IDS=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1]))["records"]; print(",".join(r for r in json.load(open(sys.argv[2])) if r in d and not d[r].get("error")))' out/replay/$1.json $ids)
    $REF conversion/score.py --ref out/ref/ref_fp32.json --cand out/replay/$1.json --files $FILES --red-arms $arms \
      --ids "$IDS" --out results/line_$1$suf.json
  done
}
```

`conversion/ids_textpic_set.json` (43 requests) and `conversion/ids_audio_set.json` (12) are the device set split into the two rows. One pass per phone row: the Pixel 11 Pro's seven-signature file with sharing on the GPU and its single files on the CPU; the Galaxy S26's NPU files. A row per NPU file is `score.py --ids` with the requests whose decision call ran on that file, without `--red-arms` (a control request and its base can fall in two files). In the Galaxy S26 NPU rows, the pictures' and clips' decision inputs are the Pixel 11 Pro's GPU tower and audio-encoder outputs at float32. The 64 states run on the eight-sequence graph:

```bash
$EXPORT conversion/replay_b8_omni.py --round-dir <phone folder> --prefix <eight-sequence run> --rows-dir <the rows it ran> \
  --out out/replay/<name>.json
$REF conversion/score.py --ref out/ref/ref_fp32.json --cand out/replay/<name>.json --files $FILES \
  --ids-file conversion/ids_L128_64states.json --out results/line_<name>.json
```

The Galaxy S26's GPU runs are compared with the Mac's Metal run of the same rows at float32, row by row with the same input bytes:

```bash
tow() { case $1 in 72x96|96x72) echo d1-omni-2b_tower_G$1_qb1152_fp16fc.tflite ;; *) echo d1-omni-2b_tower_G$1_fp16fc.tflite ;; esac; }
for c in 24x24 32x48 48x32 48x64 64x48 72x96 96x72; do
  $EXPORT conversion/mac_generic.py --graph out/graphs/$(tow $c) --rows out/rows/twG$c.json --tag m32_twG$c \
    --out-dir out/mac/m32 --accel gpu --precision fp32 --sig tower
done
for T in 504 1000 2000 3000; do
  $EXPORT conversion/mac_generic.py --graph out/graphs/d1-omni-2b_audio_T${T}_fp16fc.tflite --rows out/rows/auT$T.json \
    --tag m32_auT$T --out-dir out/mac/m32 --accel gpu --precision fp32 --sig audio
done
$EXPORT conversion/mac_generic.py --graph out/graphs/d1-omni-2b_decide_multi7_fp16fc.tflite --rows out/rows/decL128.json \
  --tag m32_m7_decL128 --out-dir out/mac/m32 --accel gpu --precision fp32 --sharing
for r in b8s64 tmL128q tmL256 tmL6144; do
  $EXPORT conversion/mac_generic.py --graph out/graphs/d1-omni-2b_decide_multi8_fp16fc.tflite --rows out/rows_b8/$r.json \
    --tag m32_m8_$r --out-dir out/mac_b8/m32 --accel gpu --precision fp32 --sharing
done
# per phone run (<run>, its rows file <rows>): the comparison (the table finds it by the name *_leg_<run>_*.json), then the table
$EXPORT conversion/leg_check_omni.py --report <phone folder>/<run>.json --out-file <phone folder>/out_<run>.f32 \
  --rows out/rows_b8/<rows>.json --run-txt <phone folder>/<run>.run.txt \
  --logcat <phone folder>/<run>.logcat_pkg_all_buffers.txt --out results/device/s26_leg_<run>_<folder>.json
$EXPORT conversion/leg_table_omni.py <phone folder> --results-dir results/device --json results/legs.json
```

The phones' folders are not in this repository, so these rows can be scored again only from a new phone run.
<!-- checks:end -->
