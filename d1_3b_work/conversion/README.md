# Conversion and check scripts

These scripts made the LiteRT graphs, the tables, the contract and the fixtures of this repository, and ran every check reported for it. They ran on an Apple M4 Max (128 GB, macOS 27.0) from 2026-10-08 to 2026-10-09. Every LiteRT run goes through the CompiledModel API of `ai_edge_litert` (CPU = XNNPACK, GPU = Metal). `REPRODUCE.md` at the repository root lists the sources, the checks and where each published number comes from.

These are the scripts that ran, with comments, docstrings and a few labels edited so that they name no path of the machine they ran on. Their code is the same, except that the inputs that are not published are read from `W/external/` and `W/evidence/`, the GPU lock file from `D1_GPU_LOCK` and the two lock wrappers of the chains from `D1_QW` and `D1_QH` (below), two names are renamed, and `make_d1_fixtures.py` writes the header and provenance text of the published `requests.json` (`fixtures/README.md`).

## Work directory

The scripts find their work directory `W` as the parent of their own folder (`W/scripts/` is a copy of this folder). They read and write these folders of `W`:

```text
W/scripts/            this folder: the *.py and *.sh files (test_d1_host.py and test_d1_vision_real.py go to W/host/)
W/host/               host/d1_litert.py, host/d1_shared_state.py and host/d1_vision.py of this repository, the two tests,
                      and host/contract.json, which the steps below write with the work directory's file names
W/hf_small/           the files of LiquidAI/d1-3B at revision da1fe36a861f24690f27f622dca1d8688503d113 except
                      model.safetensors (the model code, config, tokenizer and LICENSE): the scripts import the
                      provider's model and prompt code from here
W/fixtures/           requests.json and red_arms.json (fixtures/rebuild_requests.py of this repository writes them),
                      rows.json and token_probes.json (build_rows.py)
W/cache/real/tables/  the tables (d1_tables.py)
W/exports/            the graphs, named real_rowprefill_embeds_L<L>_v2e_fp16fc.tflite and so on (REPRODUCE.md maps them
                      to the names of this repository)
W/results/, W/logs/   every result json and log
W/evidence/           the Hub's safetensors header of model.safetensors (hub_safetensors_header_da1fe36a.json) and the
                      header copy that came with the source files (d1-3b_safetensors_header.json), for --check-header
W/external/           inputs that are not published: kev/ (the Kev-0.8B LiteRT conversion's fixture file, which
                      make_d1_fixtures.py reads) and other_conversion/ (another conversion of d1-3B: the fixture file of
                      its two long records, its control file and its reference, which make_d1_fixtures.py and
                      d1_reference.py read; the comparison with that reference is skipped when it is missing)
```

The weights are the snapshot of `hf download LiquidAI/d1-3B --revision da1fe36a861f24690f27f622dca1d8688503d113`; below, `$SNAP` is its directory.

## Environments

Two Python environments. Below, `$EXPORT` and `$REF` are their interpreters.

- Export, quantization, LiteRT checks and timing (`$EXPORT`): Python 3.14.6 and `requirements-lt094dev.txt` (litert-torch 0.9.4, litert-converter 0.4.0, torch 2.13.0, transformers 5.14.1, ai-edge-litert 2.2.0, ai-edge-quantizer 0.9.0, numpy 2.5.2).
- The provider's reference and the fixtures' rows (`$REF`): Python 3.12.11 and `requirements-venv-ref.txt` (torch 2.14.1, torchvision 0.29.1, transformers 5.14.1, tokenizers 0.22.2, numpy 2.5.3).

The host of this repository needs only `host/requirements-host.txt`.

## Steps

Run from `W`. Each script refuses to overwrite its outputs. The commands make the files of this repository; the conversion also built and gated forms that are not shipped (an int8 embedding table inside the graph, int8 FULLY_CONNECTED weights), which `REPRODUCE.md` lists.

```bash
# 0. The test requests and the control arms (fixtures/README.md of this repository), then their rows
python3 <this repository>/fixtures/rebuild_requests.py --out fixtures/requests.json --red-arms fixtures/red_arms.json
$REF scripts/build_rows.py                                                  # fixtures/rows.json + fixtures/token_probes.json

# 1. Tables: the read-out rows and the picture tower's position table, then the whole tied embedding table in
#    bfloat16 (its bytes copied from the snapshot); --verify reads them back with safetensors + torch.
python3 -I scripts/d1_tables.py $SNAP --out cache/real/tables
python3 -I scripts/d1_tables.py $SNAP --out cache/real/tables --full-embed --verify

# 2. The provider's reference: its own code (D1Model, SystemOne) in float32 on the CPU, one row per question and its
#    multi-question path, on every fixture question and the four control arms (fixtures/red_arms.json).
$REF scripts/d1_reference.py --source $SNAP --tag real                      # results/reference_real.json + hidden npz
$REF scripts/d1_reference.py --summarize results/reference_real.json        # results/reference_summary.json
$REF scripts/d1_vision_host_check.py                                        # the synthetic pictures + the processor's tensors
$REF scripts/d1v_extra_pictures.py coco_cats:cache/realv/coco_cats.jpg      # the card's photo (fetched, never committed)
$REF scripts/d1_vision_ref.py --source $SNAP --tag real --threads 8         # results/real_vision_e2e_ref.json
```

The row graphs: `d1_run_L.sh` runs the steps of one length bucket L in order, each into `logs/real[_embeds]_L<L>_<step>.log`. The shipped files are the embeds form v2e (float16 FULLY_CONNECTED weights, no embedding table in the graph). Its chain compares with the CPU run of the bucket's float32 ids graph, so that chain runs before it:

```bash
R="D1_REFERENCE=results/reference_real.json D1_TABLE=cache/real/tables/readout_table.safetensors"
R="$R D1_REFERENCE_HIDDEN=results/reference_real_hidden.npz D1_NEAR_TIE_FROM=results/reference_summary.json"
for L in 256 512 1024 2048 4096; do
  # 3. graph check (the graph vs the provider's modules in torch), export, prelint, float32 file on the CPU
  env $R D1_GRAPH_BAR=2.5e-4 D1_GRAPH_ROWS=2 D1_FORMS=fp32 D1_CPU_ROWS=smallest-head bash scripts/d1_run_L.sh $SNAP real $L
  # 4. the embeds form: export, prelint, float32 on the CPU, v2e, v2e on the CPU and on Metal (float32 and default)
  env $R D1_EMBEDS=1 D1_EMBED_TABLE=cache/real/tables/embed_table.safetensors D1_FORMS=fp32,v2e \
    D1_CPU_ROWS=smallest-head D1_VCPU_ROWS=all D1_RETIRE_FP32=1 bash scripts/d1_run_L.sh $SNAP real $L
done
# L128 has no ids file: its graph check runs alone, and its chain compares with the L256 ids file's CPU run
$EXPORT scripts/d1_graph_check.py --source $SNAP --tag real --L 128 --rows 2 --bar 2.5e-4
env $R D1_EMBEDS=1 D1_EMBED_TABLE=cache/real/tables/embed_table.safetensors D1_FORMS=fp32,v2e \
  D1_COMPARE=cache/real/litert_real_rowprefill_L256_fp32_cpu.npz D1_VCPU_ROWS=all D1_RETIRE_FP32=1 \
  bash scripts/d1_run_L.sh $SNAP real 128
```

`d1_run_L.sh` and `d1_run_vision.sh` wrap the CPU-heavy steps in `$D1_QW -- <command>` and the GPU steps in `$D1_QH <label> -- <command>`: on the conversion's Mac these waited for, or held, a measurement lock so that no two heavy jobs overlapped. Both variables must be set; a wrapper that only runs the command after its `--` will do:

```bash
printf '#!/bin/sh\nwhile [ $# -gt 0 ] && [ "$1" != "--" ]; do shift; done; shift; exec "$@"\n' > run_after_dashes.sh
chmod +x run_after_dashes.sh
export D1_QW=$PWD/run_after_dashes.sh D1_QH=$PWD/run_after_dashes.sh
```
 The prelint step runs a warning-only lint of the rank-3 operator forms (`W/../tools/rank3_prelint.py`, not in this folder); it changes no file, and with `D1_RESUME=1` a step whose result json exists is skipped. The Metal steps with `--stop-on-bar` stop the chain when a file misses the tolerance.

```bash
# 5. The shared-state pairs (embeds form): the pair against the row form in torch on every question, then per pair the
#    export (two signatures, one copy of the weights), v2e, and the gates on the CPU and on Metal (no sharing).
$EXPORT scripts/d1_shared_state.py --check --embeds --threads 8         # results/real_sharedstate_embeds_torch_check.json
for P in "64 64" "128 64" "256 128"; do
  set -- $P
  $EXPORT scripts/d1_export_pair.py --embeds --Ls $1 --Lq $2
  $EXPORT scripts/d1_storage.py --pair --variant v2e --tflite exports/real_sharedstate_embeds_Ls$1_Lq$2_fp32.tflite
  f=exports/real_sharedstate_embeds_Ls$1_Lq$2_v2e_fp16fc.tflite
  $EXPORT scripts/d1_check_pair.py --tflite $f --accel cpu --embed-table cache/real/tables/embed_table.safetensors --stop-on-bar
  $EXPORT scripts/d1_check_pair.py --tflite $f --accel gpu --f32 --embed-table cache/real/tables/embed_table.safetensors --stop-on-bar
done

# 6. The picture graphs: the tower and the projector in float32, checked against transformers, then v2 (float16
#    FULLY_CONNECTED weights); d1_run_vision.sh runs these steps with the same wrappers as d1_run_L.sh.
D1V_RTAG=realv D1V_TABLE=cache/real/tables/vision_position_table.safetensors D1V_STORAGE=v2 D1V_REL_BAR=1e-4 \
  D1V_EXTRA="coco_cats:cache/realv/coco_cats.jpg" bash scripts/d1_run_vision.sh $SNAP real

# 7. The contract, the host and the pictures end to end
$EXPORT scripts/d1_contract.py; $EXPORT scripts/d1v_contract.py --vision-only; $EXPORT scripts/d1_image_contract.py --write
$EXPORT scripts/d1_export_pair.py --contract-r10                       # the text path on the embeds files
$REF host/test_d1_host.py                                              # the host against the provider's code
$EXPORT host/test_d1_vision_real.py --files ship --accel cpu            # and --accel gpu: three picture requests
$EXPORT scripts/d1_check_pair.py --host --single --pick-contract       # the pair route against the row route
$EXPORT scripts/d1_export_pair.py --contract-r10 --host-check results/real_sharedstate_embeds_host_check.json

# 8. Mac timing (each run inside a measurement window): rows and pairs per request, then the pictures
$EXPORT scripts/d1_clock_pair.py --label row-L128-e-gpu-f32 --tflite exports/real_rowprefill_embeds_L128_v2e_fp16fc.tflite --accel gpu_f32
$EXPORT scripts/d1_clock_pair.py --label pair-Ls64-e-gpu-f32-noshare --tflite exports/real_sharedstate_embeds_Ls64_Lq64_v2e_fp16fc.tflite --accel gpu_f32 --no-share
$EXPORT scripts/d1_clock_pair.py --merge results/real_sharedstate_embeds_timing_*.json --out results/timing_mac_r10.json
$EXPORT scripts/d1_image_timing.py --accel gpu_f32 --sets one_picture,split_picture --out cache/real/image/timing_gpu_f32.json
$EXPORT scripts/d1_image_timing.py --accel cpu --sets one_picture --out cache/real/image/timing_cpu.json
$EXPORT scripts/d1_image_timing.py --merge cache/real/image/timing_gpu_f32.json cache/real/image/timing_cpu.json --out results/timing_mac_image.json

# 9. The shipped set: every file hashed again and compared with the record that made it
$EXPORT scripts/d1_ship_set.py                                         # results/ship_set.json
```

The test requests ship in `fixtures/` with the transfer-v4 records by reference; step 0 restores `requests.json` and `red_arms.json`. In the conversion, `make_d1_fixtures.py` (with `own_mid_records.py`) wrote `requests.json` from the fixture file of the Kev-0.8B LiteRT conversion, the source card and three records written for these conversions, and `d1_image_fixture.py` added the card's photo to `card_cats_001` by URL and SHA-256; `d1_reference.py --make-red-arms` wrote `red_arms.json` from another conversion's control file. `build_rows.py` writes `rows.json` and `token_probes.json` with the provider's code and tokenizer: on the restored `requests.json` it gives the same 416 rows (text, ids, answer slots, read-out groups) as the rows the checks read, and only the input file's SHA-256 and the picture field of `card_cats_001` differ.

Step 8 shows two of the timed files; every row graph and pair was timed the same way (`results/timing_mac_r10.json` lists them). `d1_clock_mac.py` holds the timing helpers that `d1_clock_pair.py` imports.

## Files

| File | Role |
|---|---|
| `d1_common.py` | Paths, the provider's code as a package, the provider's prompt settings |
| `d1_tables.py` | The tables: read-out rows, the position table, the whole embedding table in bfloat16 |
| `d1_reference.py`, `d1_vision_ref.py` | The reference: the provider's code in float32 on the CPU (text; pictures) |
| `d1_prefill_graph.py` | The row graph: the provider's LFM2 modules with the attention re-expressed for export (batched matrix products with an additive mask, the grouped keys and values copied by concat, the rotary tables as constants) |
| `d1_graph_check.py`, `d1_tiny_rows.py` | The row graph against the provider's modules in torch before export; the stand-ins for the tiny test model |
| `d1_export.py`, `tflite_scan.py` | Export with litert-torch and the scan of the operator table (no CUSTOM operator, no int64 tensor, no tensor of rank above 4) |
| `d1_storage.py`, `d1v_storage.py` | The storage variants with ai-edge-quantizer: v2e = float16 FULLY_CONNECTED weights (the pairs keep their rotary-table FULLY_CONNECTED in float32); the same for the picture graphs |
| `d1_check.py` | The gate of a row graph through the CompiledModel API: probabilities against the reference, the near ties, the control arms, the delegation |
| `d1_shared_state.py`, `d1_export_pair.py`, `d1_check_pair.py` | The shared-state pair in torch (against the row form), its export with two signatures, its gate (both state hand-overs); the contract's pair section |
| `d1_vision_graph.py`, `d1_vision_host_check.py`, `d1_vision_torch_check.py`, `d1_vision_lrt_check.py`, `d1v_extra_pictures.py` | The picture tower and projector: the graphs, the host's preprocessing against the processor, the graphs against transformers in torch and in LiteRT |
| `d1_norm_range.py` | The float16 headroom of the norms |
| `d1_run_L.sh`, `d1_run_vision.sh` | The chains of steps 3, 4 and 6 |
| `d1_contract.py`, `d1v_contract.py`, `d1_image_contract.py` | `host/contract.json`: token ids, graph inputs and outputs, buckets, tables, the picture path |
| `d1_clock_mac.py`, `d1_clock_pair.py`, `d1_image_timing.py` | Mac timing: graph calls and whole requests, rows and pairs; requests with pictures |
| `d1_ship_set.py` | The shipped file set with its records |
| `make_d1_fixtures.py`, `own_mid_records.py`, `build_rows.py`, `d1_image_fixture.py` | The fixtures |
| `test_d1_host.py`, `test_d1_vision_real.py` | The host's unit test against the provider's code; the picture path end to end against the provider |
| `requirements-lt094dev.txt`, `requirements-venv-ref.txt` | The two environments, as frozen |
| `results/` | Summaries of the records behind the published numbers (gates, timing, the picture path, the pairs in torch, the host check, the Galaxy S26 runs); each names the record and its SHA-256 |
