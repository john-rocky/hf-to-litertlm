# Conversion scripts

The scripts that made and checked the files of this repository (2026-10-20 to 2026-10-21, one Mac: Apple M4 Max, 128 GB, macOS 27.0), grouped by what they make; within a group the commands are in the order they ran (the groups overlapped in time). Each one is a copy of the script as it ran; `../REPRODUCE.md` lists the sources, the environments and where each number comes from. A script's name keeps the prefix of the step of the work it belongs to (`r2_` = the row graphs at every length, `r3_` = the pictures, `r5a_` = the host, `r5b_` = this folder).

## Work directory

Every command runs from one work directory `K` with the scripts in `K/scripts/` (`conversion/*.py`, `conversion/run_dl.sh`), the host package in `K/host/clef_litert/` with `test_host.py` beside it in `K/host/`, and these folders: `hf/` (the Hugging Face cache, `HF_HOME=K/hf`), `fixtures/`, `ref/`, `out/r1`, `out/r2`, `out/r3`, `out/r5a`, `cache/`, `results/`, `logs/`, `tmp/`. The scripts find `K` as the parent of their own folder and refuse to overwrite a result. Set `HF_HUB_DISABLE_XET=1` for the download and `HF_HUB_OFFLINE=1` after it; the conversions ran with `TMPDIR=K/tmp`.

Three Python environments:
- `venv-ref`: Python 3.12.11 with `requirements-venv-ref.txt` (torch 2.9.0, transformers 5.17.0): the provider's reference and every scoring with the provider's head.
- `lt094dev`: Python 3.14.6 with `requirements-lt094dev.txt` (litert-torch 0.9.4, litert-converter 0.4.0, torch 2.13.0, transformers 5.14.1, ai-edge-litert 2.2.0, ai-edge-quantizer 0.9.0): the graph code in PyTorch, the conversions, the quantization and the LiteRT checks of the graphs.
- `venv-host`: Python 3.12.11 with numpy 2.5.3, tokenizers 0.23.2, pillow 12.3.0 and safetensors 0.8.0 (the host's tables and test); the folder's end-to-end checks used Python 3.14.6 with `../host/requirements-host.txt`.

Below, `REF` = `venv-ref/bin/python`, `EXP` = the `lt094dev` Python, `HOST` = `venv-host/bin/python`, `RT` = a Python with `../host/requirements-host.txt`.

## 1. The checkpoint, the fixtures and the reference

```bash
bash scripts/run_dl.sh                       # the six LFS files of Cloudflare/clef-flash at f0b7d3cd (aria2c), sha256 = oid
$REF scripts/make_red_arms.py                # fixtures/red_arms.json: three records with one word of one question changed
HF_HUB_OFFLINE=1 $REF scripts/clef_ref.py --resume --deadline-s 1500                     # 214 runs -> ref/
HF_HUB_OFFLINE=1 $REF scripts/clef_ref.py --records-file fixtures/red_arms.json --tag _red --no-independent
HF_HUB_OFFLINE=1 $REF scripts/tok_compare.py                                              # the provider's encode_record vs clef_tok.py
$EXP scripts/host_selftest.py                                                             # clef_host.py vs transformers 5.14.1
```

`run_dl.sh` reads `results/hub_tree_f0b7d3cd.json`, the Hub's tree listing of that revision (`https://huggingface.co/api/models/Cloudflare/clef-flash/tree/f0b7d3cd65296abfe459e6c228b618f5d8b26d2f`); the command that saved it was not recorded. The small files (`*.json`, `*.py`, `*.jinja`, `LICENSE`, `README.md`, `.gitattributes`) came with `hf download` and one `--include` per pattern; that command line was not recorded either. `clef_ref.py` runs the checkpoint's own `joint_schema_model.py` (`load_release_model(device="cpu", dtype=torch.float32)` and `systemone()`) with 10 threads; it ran in two parts (`--resume`).

The fixture files the scripts read (`fixtures/records.json`, `fixtures/heldout/records.json`, `fixtures/red_arms.json`, `fixtures/images/`, `fixtures/heldout/images/`) are the fixtures of an earlier port of this model in its own file form; `../fixtures/requests_public.json` holds the same requests (the same `request_sha256` for the 214 published ones), `../fixtures/images/` the drawn pictures and `../fixtures/fetch_photos.py` the photographs. Five requests are not published (`../fixtures/README.md`). The held-out reference (`ref/heldout/`, 40 runs) is that earlier port's run of the same code on the same weights; two of its runs were run again here with `clef_ref.py --fixtures fixtures/heldout --records ho_t07,ho_l02 --arms text --no-independent --threads 10` and matched bit for bit. The same re-run over every held-out record would be `clef_ref.py --fixtures fixtures/heldout --out-dir ref/heldout --arms text,g256,g448`, the folder and file name `r2_sets.py`, `r5b_sets.py` and `test_host.py` read here (that command was not run in this conversion).

## 2. The row graph in PyTorch and the first file (L1024, float32)

```bash
$EXP scripts/clef_torch_check.py --L 1024 --threads 8
$REF scripts/clef_score.py --hidden-dir cache/r1/torch --tag torch_L1024
TMPDIR=K/tmp $EXP scripts/clef_export.py --L 1024 --threshold 100000000           # out/r1/clef_rowprefill_L1024_fp32.tflite
$EXP scripts/clef_cpu_check.py --tflite out/r1/clef_rowprefill_L1024_fp32.tflite --threads 8
$REF scripts/clef_score.py --hidden-dir cache/r1/cpu --tag cpu_L1024
$EXP scripts/clef_ms.py --tflite out/r1/clef_rowprefill_L1024_fp32.tflite --threads 8
```

`clef_export.py` converts with litert-torch's lightweight path: `RESOURCE_CONSTANT_NUMEL_THRESHOLD` = 100,000,000 bytes is set before `litert_torch` is imported, and `litert_torch.convert(..., lightweight_conversion=True, runtime_constant_folding=False)` (with litert-converter 0.4.0, `runtime_constant_folding=True` stops with `ModuleNotFoundError: No module named 'litert_converter.mlir.dialects.tfl'`). It refuses to run unless `results/score_torch_L1024.json` passed. The float32 file was deleted after step 3.

## 3. fp16 FULLY_CONNECTED weights, every length, CPU and Metal

```bash
TMPDIR=K/tmp $EXP scripts/r2_quant.py --src out/r1/clef_rowprefill_L1024_fp32.tflite --form v2_fp16fc
TMPDIR=K/tmp $EXP scripts/r2_check.py --tflite out/r2/clef_rowprefill_L1024_v2_fp16fc.tflite --acc cpu --ms
$REF scripts/r2_score.py --check results/r2_check_v2_fp16fc_L1024_cpu.json --compare-score results/score_cpu_L1024.json --compare-cache cache/r1/cpu --compare-tag fp32_cpu_r1
TMPDIR=K/tmp $EXP scripts/r2_check.py --tflite out/r2/clef_rowprefill_L1024_v2_fp16fc.tflite --acc gpu_f32 --ms
$REF scripts/r2_score.py --check results/r2_check_v2_fp16fc_L1024_gpu_f32.json
# then for L = 256, 2048, 512, 4096 (the order they ran):
TMPDIR=K/tmp $EXP scripts/r2_export.py --L $L
TMPDIR=K/tmp $EXP scripts/r2_quant.py --src out/r2/clef_rowprefill_L${L}_fp32.tflite --form v2_fp16fc
TMPDIR=K/tmp $EXP scripts/r2_check.py --tflite out/r2/clef_rowprefill_L${L}_v2_fp16fc.tflite --acc cpu --ms
$REF scripts/r2_score.py --check results/r2_check_v2_fp16fc_L${L}_cpu.json
TMPDIR=K/tmp $EXP scripts/r2_check.py --tflite out/r2/clef_rowprefill_L${L}_v2_fp16fc.tflite --acc gpu_f32 --ms
$REF scripts/r2_score.py --check results/r2_check_v2_fp16fc_L${L}_gpu_f32.json
```

`r2_quant.py` applies ai-edge-quantizer's float casting to every FULLY_CONNECTED weight (16 bits, channelwise, explicit DEQUANTIZE) and accepts the file when its 296 FULLY_CONNECTED read float16 weight buffers through DEQUANTIZE (FLOAT16 to FLOAT32): 250 buffers, the 248 linear layers and the two constant matrices of the GatedDeltaNet kernel that 48 of them share, which must stay bit-equal. The drivers also passed `--budget-s` and `--est-call-s` to `r2_check.py` (and `--deadline-s` to the other checks): they only decide when a run stops early, and the scripts resume where they stopped. `r2_check.py` runs the pre-registered set of each length (`r2_sets.py`) and then the timing (5 warm-up and 20 timed calls) in the same process; on Metal (`GpuOptions(enforce_f32=True)`) every node must be delegated in one partition. The float32 files were deleted after their fp16 FC file passed.

## 4. The pictures

```bash
HF_HUB_OFFLINE=1 $REF scripts/r3_pixpos.py
$REF scripts/r3_embeds_check.py
HF_HUB_OFFLINE=1 $REF scripts/r3_tower_torch.py         # the tower rewrite vs the provider's module; cache/r3/consts_g{256,448}.npz
$REF scripts/r3_forms_check.py
TMPDIR=K/tmp $EXP scripts/r3_metal_bisect.py             # writes cache/r3/tower_form.json (the position rows as an input)
TMPDIR=K/tmp $EXP scripts/r3_tiny_tfl.py
TMPDIR=K/tmp $EXP scripts/r3_tower_tfl.py                # out/r3/clef_tower_g{256,448}_{fp32,v2_fp16fc}.tflite
TMPDIR=K/tmp $EXP scripts/r3_tower_rt.py
$REF scripts/r3b_rt_verdict.py
TMPDIR=K/tmp $EXP scripts/r3_e2e.py --tag L1024 --L 1024 --row out/r2/clef_rowprefill_L1024_v2_fp16fc.tflite --towers v2_fp16fc,fp32 --arms g256,g448,zero,native --accel gpu_f32 --ms
$REF scripts/r3_score.py --tag L1024
TMPDIR=K/tmp $EXP scripts/r3_e2e.py --tag L1024cpu --L 1024 --row out/r2/clef_rowprefill_L1024_v2_fp16fc.tflite --towers v2_fp16fc,fp32 --arms g256,g448,zero,native --accel cpu
$REF scripts/r3_score.py --tag L1024cpu
TMPDIR=K/tmp $EXP scripts/r3_e2e.py --tag L2048 --L 2048 --row out/r2/clef_rowprefill_L2048_v2_fp16fc.tflite --towers v2_fp16fc --arms native_bucket --accel gpu_f32
$REF scripts/r3_score.py --tag L2048
```

`r3_metal_bisect.py` found that ai-edge-litert 2.2.0 on Metal computes a FULLY_CONNECTED output plus a same-shape constant (the tower's position rows) wrongly while it delegates every node; the shipped towers take the position rows as their second input `pos` (`../tables/vision_pos_g{256,448}.safetensors`). `r3_tower_tfl.py` converts with litert-torch's default path and applies the same fp16 FC recipe as step 3. It also runs a rank-3 prelint tool that is not published; without it the step records the tool's exit code and goes on (its warnings were only recorded).

## 5. The host

```bash
$HOST scripts/r5a_tables.py                       # out/r5a/embed_tokens.safetensors, out/r5a/lm_head.safetensors
HF_HUB_OFFLINE=1 $REF scripts/r5a_torch_side.py
VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 $HOST host/test_host.py
VECLIB_MAXIMUM_THREADS=1 $HOST scripts/r5a_vs_r1.py
```

`r5a_tables.py` copies the bytes of `model.language_model.embed_tokens.weight` and `lm_head.weight` (bfloat16 [248320, 4096]) from `model-00001-of-00004.safetensors` into one-key files without metadata.

## 6. This folder

```bash
$HOST scripts/r5b_stage.py core                    # the folder: names, symlinks to the files above, the towers' position rows
$HOST scripts/r5b_fixtures.py                      # fixtures/ (CLEF_ZOO_FIXTURE = the earlier port's fixtures-clef-flash.json)
$HOST scripts/r5b_expected.py                      # examples/run_example.expected.json, examples/request.json
TMPDIR=K/tmp $RT scripts/r5b_e2e.py --folder staging/upload --accelerator gpu --length $L --ms --sums   # per L, Metal
TMPDIR=K/tmp $RT scripts/r5b_e2e.py --folder staging/upload --accelerator cpu --length $L --subset cpu20 [--ms]
$HOST scripts/r5b_score.py --accelerator gpu
$HOST scripts/r5b_score.py --accelerator cpu
$RT scripts/r5b_sums.py --file <staged file> ... --out results/r5b_sums_misc.json
python3 scripts/r5b_sha256sums.py --write
$HOST scripts/r5b_contract.py
python3 scripts/r5b_public_results.py
```

`r5b_e2e.py` runs the folder's own host package (`ClefLiteRT.decide`) on its graphs, one row-graph length per process, on every fixture run the host can make (the picture runs at the 256 and 448 px tiles), and times one decision request in the same compile.

## Not published

- The drivers that queued these commands into a machine-wide measurement lock and stopped a job when swap grew (they add no step of their own).
- The comparison of the reference with the earlier port's reference (`ref/` matched it bit for bit on 214 runs and 405 questions), which reads that port's files.
- The forms that were measured and not shipped: dynamic int8 FULLY_CONNECTED weights, Metal's default precision with and without a pre-scale of the norms, and two state forms (a chunk step and a pair of graphs).

## Lines of code that differ from the scripts as they ran

Every other difference is in comments and docstrings (the code of each other script is the same: its Python syntax tree without docstrings is equal).

- `clef_ms.py`: the measurement lock's path: the work directory read a file of this machine; the copy reads MEASUREMENT_LOCK_FILE (default /dev/null = an empty line). The line is only recorded, never used for a decision. The recorded key that says whether this run's label held the lock at every timed call is named lock_held_by_this_run_every_timed_call.
- `r2_sets.py`: the held-out reference: a folder on a disk of this machine (file records_oracle.json, written by the earlier port's copy of clef_ref.py) in the work directory; K/ref/heldout/records_ref.json in the copy = what clef_ref.py --fixtures fixtures/heldout --out-dir ref/heldout writes (conversion/README.md)
- `r2_check.py`: the measurement lock's path: the work directory read a file of this machine; the copy reads MEASUREMENT_LOCK_FILE (default /dev/null = an empty line). The line is only recorded, never used for a decision. The recorded key that says whether this run's label held the lock at every timed call is named lock_held_by_this_run_every_timed_call.
- `r3_tower_tfl.py`: the ledger's description string (record text only); the Python that runs the rank-3 prelint (a venv path of this machine -> this interpreter). The prelint tool (../tools/rank3_prelint.py) is not published: without it the step records its exit code and goes on (its warnings were recorded only)
- `r3_tower_rt.py`: the measurement lock's path: the work directory read a file of this machine; the copy reads MEASUREMENT_LOCK_FILE (default /dev/null = an empty line). The line is only recorded, never used for a decision. The recorded key that says whether this run's label held the lock at every timed call is named lock_held_by_this_run_every_timed_call.
- `r3b_rt_verdict.py`: the result's description string (record text only)
- `r3_e2e.py`: the measurement lock's path: the work directory read a file of this machine; the copy reads MEASUREMENT_LOCK_FILE (default /dev/null = an empty line). The line is only recorded, never used for a decision. The recorded key that says whether this run's label held the lock at every timed call is named lock_held_by_this_run_every_timed_call.
- `r5a_tables.py`: the ledger's description string (record text only)
- `test_host.py`: the name of one summary key of the result (now four_named); the held-out reference's folder and file name (as r2_sets.py)
- `r5b_fixtures.py`: the earlier port's published fixture file (its SemIf license notice is copied from it) and the folders of the pinned photo thumbnails as that port fetched them: paths of this machine in the work directory, CLEF_ZOO_FIXTURE (default ./fixtures-clef-flash.json) and CLEF_PHOTOS_SRC (default photos_src:heldout/photos_src) in the copy
- `r5b_sets.py`: the held-out reference's folder and file name (as r2_sets.py)
- `r5b_e2e.py`: the measurement lock's path: the work directory read a file of this machine; the copy reads MEASUREMENT_LOCK_FILE (default /dev/null = an empty line). The line is only recorded, never used for a decision. The recorded key that says whether this run's label held the lock at every timed call is named lock_held_by_this_run_every_timed_call.
