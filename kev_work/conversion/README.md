# Conversion and check scripts

These scripts made the LiteRT graphs, the head files and the fixtures of this repository, and ran every check reported for it. The same folder ships with Kev-0.8B-LiteRT and Kev-4B-LiteRT: a script that takes `--model` works on Kev-0.8B by default and on Kev-4B with `--model 4b`. They ran on macOS arm64 (Apple M4 Max, 128 GB, macOS 27.0) from 2026-10-03 to 2026-10-05; the phone runs they prepared and scored used the sources in `android/measure/` (the measurement activity, and the NPU runner).

The graphs in this repository use the final kernel: `r17_export.py` made the 64-, 128- and 256-token Kev-0.8B files (form C7, which also runs on the phone's NPU), `r13_export.py` and `r14_export_shared.py` the other Kev-0.8B files (form C), `r15_export.py` and `r15_export_shared.py` the Kev-4B files. `export_kev.py` made the files of the earlier release (the loop kernel); it is kept with its checks for reference.

Every LiteRT run goes through the CompiledModel API (`ai_edge_litert.compiled_model`; on the phone the Kotlin API and, for the NPU, the C API). `tflite_scan.py`, `quantize_kev.py`, `quantize_shared_state.py`, `r16_head_json.py` and `r18_head_json.py` read the flatbuffer with the schema module and never run the model.

## Work directory

The scripts find their work directory `W` as the parent of their own folder (`W/scripts/` is a copy of this folder). They write `W/fixtures/`, `W/oracle/`, `W/merged/`, `W/exports/`, `W/results/`, `W/cache/` and `W/logs/`. `env.sh`, sourced from `W`, puts the Hugging Face cache in `W/hf` and every other cache in `W/cache/`.

```text
W/scripts/                      a copy of this folder
W/kev/                          github.com/jaredpalmer/kev at tag kev-1.0 (commit 6b719c3c3f367295f6ef336f4f751cf5ff970abc)
W/src/semif/authored144.jsonl   benchmarks/data/authored144.jsonl of github.com/TheoLeeCJ/SemIf at commit ca3ba65f142967030ecb453346e94d6f476a69df
W/src/semif/LICENSE             that repository's LICENSE (fixtures/LICENSE-SemIf-MIT.txt here)
W/host/                         host/kev_litert.py of this repository, and head/kev_<model>_pointer_head.{safetensors,json}
W/device/r14/                   the phone files: the inputs r14_device_rows.py, r14_device_rows_req.py and r18_req2.py
                                write, and each phone run's report, read-out hidden states and 2 s samples
W/device/r11/, W/device/r17/    the NPU runner's rows files (r11_npu_fixtures.py, r17_npu_fixtures.py) and runs
W/npu/                          the NPU runner's fixtures
W/staging/Kev-4B-LiteRT/        (r15_timing_mac.py only) a copy of the earlier release of Kev-4B-LiteRT: the loop-kernel
                                files that its loopL512 / loopL2048 graphs time next to the final ones
```

## Environments

Three Python environments. `env.sh` expects their interpreters in `$ORACLE`, `$EXPORT` and `$HOST`.

- Reference (`$ORACLE`): Python 3.12.11 and `requirements-oracle.txt`, the author's lock file at `kev-1.0` (torch 2.8.0, transformers 5.17.0, peft 0.21.0, huggingface-hub 1.32.0, datasets 5.0.1, tokenizers 0.23.2) plus the kev package from `W/kev`.
- Export and quantization (`$EXPORT`): Python 3.14.6 and `requirements-export.txt` (litert-torch 0.9.4, litert-converter 0.4.0, torch 2.13.0, transformers 5.14.1, ai-edge-litert 2.2.0, ai-edge-quantizer 0.9.0, numpy 2.5.2).
- Python host (`$HOST`): Python 3.12.11 and `host/requirements-host.txt` of this repository (ai-edge-litert 2.2.0, numpy 2.5.2, tokenizers 0.23.2, safetensors 0.8.0).

```bash
cd W && source scripts/env.sh
mkdir -p fixtures oracle merged exports results logs device host
uv venv --python 3.12 .venv-oracle && uv pip install --python .venv-oracle -r scripts/requirements-oracle.txt
export ORACLE=$W/.venv-oracle/bin/python EXPORT=... HOST=...
```

The PyTorch checks start at most 4 workers while the file named by the environment variable `KEV_GPU_LOCK` holds a line with "timing" (a lock another job holds while it times the GPU). Leave it unset to always start the asked number. `r14_timing_mac.py`, `r14_timing_mac_req.py`, `r18_timing_mac.py`, `r18_timing_mac_req.py` and `r17_timing_mac.py` use that file as their GPU lock (`W/gpu.lock` when `KEV_GPU_LOCK` is unset; create it empty) and write such a line while they time; they also take `W/.b_heavy.lock`, the lock that `r14_flock.py` holds around a heavy CPU job, so no such job runs beside them. `r15_timing_mac.py` and `r15_timing_shared.py` take no lock.

## The final kernel

The graph is the Qwen3.5 text model of transformers 5.14.1 with the patch classes of `kev_qwen35_patch.py` (`kev_graph.py`). The final kernel changes how the Gated DeltaNet layers compute, not what they compute: every change is exact in fp32 or equal up to rounding, and `r14_torch_parity.py` / `r15_torch_parity.py` check that on all 402 questions. A form is a `+`-joined list of these changes (`r13_kernel.py`, `r15_form.py`):

| Token | What it changes |
|---|---|
| `R64` | The in-chunk inverse (I − A)⁻¹ of the chunk kernel by recursive doubling (a masked copy, then 5 levels of two matrix products per 64-token chunk) instead of the loop kernel's 63-step forward substitution. |
| `sp` | softplus as relu(x) + log1p(exp(x − 2 relu(x))): the EXP argument stays ≤ 0, so it cannot overflow float16. |
| `ec` | Every EXP argument in the chunk kernel is clamped at −80 (exp(−80) = 1.8e-35, below fp32 resolution of the sums it enters). |
| `dd` | The decays inside a chunk are sums over their own window, formed with constant 0/1 matrices, instead of differences of a long cumulative sum (which loses precision in float16). |
| `vs6` | The kernel scales v by 2⁶ and the gated norm after it takes eps × 4⁶: the norm's input leaves the float16 subnormal range; exact in fp32. A shared-state pair carries the recurrent state at 2⁶ times its stock value between its two signatures, so a state from one file belongs to that file only. |
| `in1` | (Kev-4B) Every decoder layer's input_layernorm runs on 2⁻¹ x with eps × 4⁻¹: exact in fp32; that norm's float16 headroom goes from 2.98× to 11.93×. |
| `fn5` | (Kev-4B) The final RMSNorm runs on 2⁻⁵ x with eps × 4⁻⁵: exact in fp32; Kev-4B's final norm input has a sum of squares above the float16 maximum on every test row without it. |
| `bk1024@in_proj_z.q_proj` | (`r17_kernel.py`) The Gated DeltaNet in_proj_z and the attention q_proj linears as a batch-1 BATCH_MATMUL + SUM; the converter turns each back into a FULLY_CONNECTED and keeps a SUM over an axis of size 1 after it (an exact identity). That SUM sits between the FC and the sigmoid of the gated norm's gate and of the attention output gate: on the phone's NPU a sigmoid that reads an FC output directly lost about 2.3e-3 (absolute, root mean square), and with the SUM in between the files pass. The owners may also be spelled with ','. |

Kev-0.8B uses `R64+sp+ec+dd+vs6+bk1024@in_proj_z.q_proj` (form C7) for its 64-, 128- and 256-token files and `R64+sp+ec+dd+vs6` (form C) for the other files and the pairs; on the CPU and on Metal at float32 the two forms give the same probabilities bit for bit. Kev-4B uses `R64+sp+ec+dd+vs6+in1+fn5`. With the final form the 512-token Kev-0.8B fp32 graph has 8,327 operators instead of the loop kernel's 20,873. `r12_kernel_product.py` holds the other forms of the inverse that were tried (`R64` is copied from it into `r13_kernel.py`); `r13_kernel.py` also holds further rewrites that the final files do not use.

### Float16 headroom of the norms

The converter lowers every RMS-type norm's mean(x²) to a SUM, so under float16 storage a sum of squares above 65,504 becomes inf and the row comes out as zeros. `r14_norm_range_probe.py` / `r15_norm_range_probe.py` measured the largest sum of squares on all 402 questions (fp32, real positions) and its margin to 65,504:

| Norm input | Kev-0.8B (`R64+sp+ec+dd+vs6`) | Kev-4B (`R64+sp+ec+dd+vs6+in1+fn5`) |
|---|---:|---:|
| Final norm, as its reduction sees it | 5,971.6 (10.97×) | 142.5 (459.64×) |
| Gated DeltaNet output norm (after the 2⁶ scale) | 2,587.5 (25.32×) | 5,086.7 (12.88×) |
| Chunk kernel l2norm (q, k) | 2,000.9 (32.74×) | 932.2 (70.27×) |
| Attention q_norm | 608.1 (107.71×) | 5,179.9 (12.65×) |
| Attention k_norm | 431.8 (151.71×) | 1,411.6 (46.41×) |
| post_attention_layernorm | 467.9 (140.01×) | 10,004.1 (6.55×) |
| input_layernorm, as its reduction sees it | 303.0 (216.20×) | 5,490.6 (11.93×) |

Without `fn5` and `in1`, Kev-4B's final norm input reaches 145,930.4 (0.45×, over the maximum on all 402 questions) and its input_layernorm 21,962.3 (2.98×). The results are `results/r14_norm_range_R64-sp-ec-dd-vs6.json` and `results/r15_norm_range_4b_R64-sp-ec-dd-vs6-in1-fn5.json`.

## Steps

The commands are for Kev-0.8B unless a comment says otherwise.

```bash
# 1. Fixtures: 377 requests, 402 questions (fixtures/rebuild_requests.py of this repository rebuilds the same file)
$ORACLE scripts/make_fixtures.py

# 2. Reference: the author's code on the CPU in fp32, one torch thread. The author's resolver downloads the
#    checkpoint and its base model into W/hf when they are missing.
$ORACLE scripts/oracle_kev.py                                                  # [--model 4b]

# 3. Merge with the author's script, then check the merge
$ORACLE kev/scripts/merge_lora_checkpoint.py --lora jaredpalmer/kev-0.8b@788ddbdd65715bb03a56788c822f6c632c9a551d \
  --out merged/kev-0.8b-v1.0      # 4B: --lora jaredpalmer/kev-4b@591dcb5bd6d05eb0b5131ea6608f93f10243335c --out merged/kev-4b-v1.0
$ORACLE scripts/merge_check.py                                                 # [--model 4b]
$ORACLE scripts/oracle_kev.py --ckpt merged/kev-0.8b-v1.0/checkpoint --tag merged
#    4B: --model 4b --ckpt merged/kev-4b-v1.0/checkpoint --tag merged_4b
$ORACLE scripts/merge_parity.py                                                # [--model 4b]

# 4. The merged checkpoint in stock transformers 5.14.1 (the baseline of the export guards)
$EXPORT scripts/tf514_baseline.py                                              # [--model 4b]
```

### Kev-0.8B (the files of this repository)

```bash
F=R64+sp+ec+dd+vs6; TAG=r14B-vs6; V=v2_fp16fc_i8emb_$TAG                     # form C
F7=R64+sp+ec+dd+vs6+bk1024@in_proj_z.q_proj; TAG7=r17C7-bkzq; V7=v2_fp16fc_i8emb_$TAG7   # form C7 (L64 / L128 / L256)

# 5. The forms in PyTorch: fp32 invariance (all 402 questions), the shared-state pair vs the row form, float16 headroom
$EXPORT scripts/r14_torch_parity.py --form $F --shards 8
$EXPORT scripts/r17_torch_parity.py --form $F7 --shards 8
$EXPORT scripts/r14_shared_torch_parity.py --form $F --workers 8
$EXPORT scripts/r14_norm_range_probe.py --form $F --shards 8

# 6. Export: fp32 graph -> op scan -> V2 (float16 FULLY_CONNECTED weights + int8 embedding table); the fp32 file is
#    deleted after the V2 checks. The pair export needs step 5's results/r14_shared_torch_parity_R64-sp-ec-dd-vs6.json.
for L in 64 128 256; do $EXPORT scripts/r17_export.py --L $L --form $F7 --tag $TAG7; done
for L in 64 128 256 512 1024 2048; do $EXPORT scripts/r13_export.py --L $L --form $F --tag $TAG; done
#    (the form C files of L64 / L128 / L256 are not published: they are the reference of the C7 checks and timings)
for Ls in 128 256; do $EXPORT scripts/r14_export_shared.py --Ls $Ls --Lq 64 --form $F --tag $TAG; done

# 7. The published names (R = the repository root)
for L in 64 128 256; do cp exports/kev08b_rowprefill_L${L}_$V7.tflite $R/kev-0.8b_rowprefill_L${L}_fp16fc_i8emb.tflite; done
for L in 512 1024 2048; do cp exports/kev08b_rowprefill_L${L}_$V.tflite $R/kev-0.8b_rowprefill_L${L}_fp16fc_i8emb.tflite; done
for Ls in 128 256; do cp exports/kev08b_sharedstate_Ls${Ls}_Lq64_$V.tflite $R/kev-0.8b_sharedstate_Ls${Ls}_Lq64_fp16fc_i8emb.tflite; done

# 8. Desktop gate: Mac CPU (8 threads) and Metal at float32 precision (in a child process), every question that fits;
#    form C on every L, form C7 on L64 / L128 / L256 (the same lines with $V7 and cache/r17/mac_hsel)
for L in 64 128 256 512 1024 2048; do
  f=exports/kev08b_rowprefill_L${L}_$V.tflite
  $EXPORT scripts/litert_parity.py --tflite $f --L $L --variant $V --accel cpu --cache cache/r14/mac_hsel
  $EXPORT scripts/gpu_gate_subprocess.py --expect results/litert_gpu_f32_parity_L${L}_$V.json \
    --log logs/litert_gpu_f32_parity_L${L}_$V.log -- --tflite $f --L $L --variant $V --accel gpu --f32 --cache cache/r14/mac_hsel
  $EXPORT scripts/gpu_vs_cpu.py --L $L --variant $V --tag gpu_f32 --cache cache/r14/mac_hsel
done
#    the GPU's default precision (float16 activations), L=128 and L=2048: the same gpu_gate_subprocess.py line without
#    --f32 and with results/litert_gpu_f16_parity_L${L}_$V.json
for Ls in 128 256; do
  f=exports/kev08b_sharedstate_Ls${Ls}_Lq64_$V.tflite; npz=cache/r14/torch_two_phase_hsel_R64-sp-ec-dd-vs6.npz
  $EXPORT scripts/r14_litert_shared_parity.py --tflite $f --Ls $Ls --Lq 64 --variant $V --accel cpu --torch-hsel $npz
  $EXPORT scripts/r14_litert_shared_parity.py --tflite $f --Ls $Ls --Lq 64 --variant $V --accel gpu --f32 --torch-hsel $npz
  $EXPORT scripts/r14_litert_shared_parity.py --tflite $f --Ls $Ls --Lq 64 --variant $V --accel gpu --f32 --share --torch-hsel $npz
done

# 9. Head contract: the graph entry of head/kev_0.8b_pointer_head.json, read from the files in $R (with graph.npu)
$HOST scripts/r18_head_json.py --repo $R --model 0.8b --src <the earlier head json> --npu-L 64,128,256 --out $R/head/kev_0.8b_pointer_head.json

# 10. Python host: token rows, head and answers against the reference and the author's live code (no graph), then the
#     host on the files of $R against step 8's rows (row, pair and auto routes)
$HOST scripts/host_parity.py --skip-graph --out results/host_parity_contract.json
RT=64,128,256=$V7
$HOST scripts/r18_host_parity.py --repo $R --phase row --accel cpu --threads 4 --row-tag $RT --suffix _C7
$HOST scripts/r16_host_parity.py --repo $R --phase pair --accel cpu --threads 4
$HOST scripts/r18_host_parity.py --repo $R --phase auto --accel cpu --threads 4 --row-tag $RT --suffix _C7
$HOST scripts/r18_host_parity.py --repo $R --phase row --accel gpu --row-tag $RT --suffix _C7
$HOST scripts/r16_host_parity.py --repo $R --phase pair --accel gpu --share on    # and --share off

# 11. Phone inputs (W/device/r14/): the gate rows of every bucket, the timing rows, the pairs' requests, the request
#     points, and the two Ls 256 request files of the request runs. Without device/r6/rows_L2048.json (the earlier
#     release's 30 phone rows) r14_device_rows.py skips its check of those rows.
python3 scripts/r14_device_rows.py --form C --pairs 128:64 256:64
python3 scripts/r14_device_rows_req.py
python3 scripts/r18_req2.py

# 12. Run the measurement activity (android/measure/README.md) with the files of step 6 under their work-directory names
#     and the inputs of step 11; put each run's report, hidden-state dumps and samples into W/device/r14/. Score a gate run
#     (a pair's run: --hsel device/r14/hsel_<tag>_<host|direct>.f32 --rows device/r14/rows_Ls<Ls>_Lq64_full.json --L <Ls>,
#     one output per hand-over):
$EXPORT scripts/r12_device_compare.py --report device/r14/<tag>.json --hsel device/r14/hsel_<tag>.f32 --L <L> \
  --rows device/r14/rows_L<L>.json --variant $V --mac-hsel-dir cache/r14/mac_hsel --out device/r14/device_parity_<tag>.json
#     (a run of a form C7 file: --variant $V7 --mac-hsel-dir cache/r17/mac_hsel)

# 13. Phone timing: every timed call split by the clock caps at its start, then the request table
python3 scripts/r18_burst.py > cache/r18/burst.md             # results/r18_burst.json
python3 scripts/r18_crossover.py > cache/r18/crossover.md     # results/r18_crossover.json

# 14. Mac timing: Metal float32 and the CPU (8 threads), each bucket's row and the Ls 128 pair (form C), then the
#     request points (the pairs, and the rows of form C); C7 next to C at L64 / L128 / L256 and the request rows on C7;
#     the C7 CPU rows again while the 1-minute load average was below 2.5
$EXPORT scripts/r14_timing_mac.py --forms C --out results/timing_mac_r14_vs6.json
$EXPORT scripts/r14_timing_mac_req.py --out results/timing_mac_r14_req.json
$EXPORT scripts/r18_timing_mac.py --forms CD --Ls 64 128 256 --no-pair --out results/timing_mac_r18_C7.json
$EXPORT scripts/r18_timing_mac_req.py --out results/timing_mac_r18_C7_req.json
$EXPORT scripts/r18_timing_mac.py --forms CD --Ls 64 128 256 --no-pair --accels cpu8 --max-load 2.5 --wait-s 1800 \
  --out results/timing_mac_r18_C7_cpu.json
#     the pairs' memory on Metal float32, one new process per leg and nothing else on the GPU (two runs, in this order)
for leg in "128 share 1" "128 noshare 1" "256 share 1" "256 noshare 1" \
           "256 noshare 2" "256 share 2" "128 noshare 2" "128 share 2"; do
  set -- $leg; $EXPORT scripts/r14_memory_mac.py --leg pair --Ls $1 --share $2 --run $3
done

# 15. NPU (the phone's Qualcomm HTP, form C7): the runner's inputs, then per run of android/measure/kev_npu_runner
#     (android/measure/README.md; the device names c7.tflite and so on are the files of step 6) its report, the score
#     against the reference and the clock caps, and one table of the runs
python3 scripts/r17_npu_fixtures.py --L 64                 # device/r17/rows_L64.json, npu/fixtures_r17_L64/
python3 scripts/r11_npu_fixtures.py --L 128                # device/r11/rows_L128.json, npu/fixtures_L128/ (and --L 256)
export R17_REL=device/r17/B                                # the folder of the runs
$EXPORT scripts/r17_runner_report.py --stdout $R17_REL/<tag>.stdout.txt --hsel-in $R17_REL/<tag>.hsel.f32 \
  --manifest npu/fixtures_L128/manifest.json --tag <tag> --accel npu --precision htp-fp16 --graph c7.tflite \
  --logcat $R17_REL/<tag>.logcat_all.txt
$EXPORT scripts/r17_device_compare.py --report $R17_REL/<tag>.json --hsel $R17_REL/hsel_<tag>.f32 --L 128 \
  --rows device/r11/rows_L128.json --out $R17_REL/device_parity_<tag>.json --variant $V7 \
  --same-rows results/litert_cpu_rows_L128_$V7.json --same-npz cache/r17/mac_hsel/hsel_cpu_L128_$V7.npz
python3 scripts/r17_leg_caps.py --tag <tag>
python3 scripts/r17_hold_table.py --md --dir device/r17/B --out results/r17_device_table_B.json
#     Mac Metal float32: form C7 next to form C at the three buckets (also measured by step 14)
$EXPORT scripts/r17_timing_mac.py --forms C0,C7 --out results/timing_mac_r17_C7.json
```

### Kev-4B (the files of Kev-4B-LiteRT)

A 4B fp32 export holds about 50 GB of memory at its peak; run one at a time.

```bash
F4=R64+sp+ec+dd+vs6+in1+fn5; V4=v2_fp16fc_i8emb_r15R64-sp-ec-dd-vs6-in1-fn5

# The form on Kev-4B in PyTorch: the kernel test, fp32 invariance (all 402 questions), float16 headroom, and the
# shared-state pair against the row form (every question at the smallest Ls that holds its state, so it covers both pairs)
$EXPORT scripts/r15_kernel_test.py --model 4b --out results/r15_kernel_test_4b.json
$EXPORT scripts/r15_torch_parity.py --model 4b --form $F4 --shards 8 --concurrency 4
$EXPORT scripts/r15_norm_range_probe.py --model 4b --form $F4 --shards 4
$EXPORT scripts/r15_shared_torch_parity.py --model 4b --form $F4 --workers 3

# Export and V2, one at a time; the published names (R4 = the Kev-4B-LiteRT root)
for L in 64 128 256 512 1024 2048; do $EXPORT scripts/r15_export.py --model 4b --L $L --form $F4; done
for Ls in 128 256; do $EXPORT scripts/r15_export_shared.py --model 4b --Ls $Ls --Lq 64 --form $F4; done
for L in 64 128 256 512 1024 2048; do cp exports/kev4b_rowprefill_L${L}_$V4.tflite $R4/kev-4b_rowprefill_L${L}_fp16fc_i8emb.tflite; done
for Ls in 128 256; do cp exports/kev4b_sharedstate_Ls${Ls}_Lq64_$V4.tflite $R4/kev-4b_sharedstate_Ls${Ls}_Lq64_fp16fc_i8emb.tflite; done

# Desktop gate: Metal at float32 precision on every row graph (in a child process), the Mac CPU (8 threads) on L=1024;
# Metal at its default precision (float16 activations) ran on L=128 and L=512: the same gpu_gate_subprocess.py line
# without --f32 and with gpu_f16 in the two file names
for L in 64 128 256 512 1024 2048; do
  f=exports/kev4b_rowprefill_L${L}_$V4.tflite
  $EXPORT scripts/gpu_gate_subprocess.py --expect results/litert_gpu_f32_parity_L${L}_4b_$V4.json \
    --log logs/litert_gpu_f32_parity_L${L}_4b_$V4.log -- --model 4b --tflite $f --L $L --variant $V4 --accel gpu --f32 --cache cache/r15/mac_hsel
done
$EXPORT scripts/litert_parity.py --model 4b --tflite exports/kev4b_rowprefill_L1024_$V4.tflite --L 1024 --variant $V4 \
  --accel cpu --threads 8 --cache cache/r15/mac_hsel
for Ls in 128 256; do
  p4=exports/kev4b_sharedstate_Ls${Ls}_Lq64_$V4.tflite
  $EXPORT scripts/r15_shared_parity.py --model 4b --form $F4 --Ls $Ls --Lq 64 --tflite $p4 --variant $V4 --accel cpu --threads 8
  $EXPORT scripts/r15_shared_parity.py --model 4b --form $F4 --Ls $Ls --Lq 64 --tflite $p4 --variant $V4 --accel gpu --f32 --share
  $EXPORT scripts/r15_shared_parity.py --model 4b --form $F4 --Ls $Ls --Lq 64 --tflite $p4 --variant $V4 --accel gpu --f32
done
$EXPORT scripts/r15_share_compare.py --form $F4        # Metal with and without constant tensor sharing, bit for bit

# Head contract and the Python host on the files of $R4: Metal at float32 for the row graphs and the pairs (with
# constant tensor sharing), the CPU on 20 rows of the 128-token graph
$HOST scripts/r16_head_json.py --repo $R4 --model 4b --src <the earlier head json> --out $R4/head/kev_4b_pointer_head.json
$HOST scripts/host_parity.py --model 4b --skip-graph --out results/host_parity_contract_4b.json
$HOST scripts/r16_host_parity.py --model 4b --repo $R4 --phase row --accel gpu
$HOST scripts/r16_host_parity.py --model 4b --repo $R4 --phase pair --accel gpu --share on
$HOST scripts/r16_host_parity.py --model 4b --repo $R4 --phase row --accel cpu --threads 4 --force-L 128 --limit 20

# Mac timing, one job per command, each with the GPU lock held by the caller. Row graphs (gpu_f16 = Metal's default
# precision, float16 activations; loopL512 / loopL2048 = the earlier release's files in W/staging/Kev-4B-LiteRT):
T4=r15R64-sp-ec-dd-vs6-in1-fn5
$EXPORT scripts/r15_timing_mac.py --accel gpu_f32 --tag $T4 --graphs L128,L256,L512,L1024,L2048,loopL512,loopL2048 \
  --repeat loopL512,L512 --out results/timing_mac_r15_4b_gpu_f32.json
$EXPORT scripts/r15_timing_mac.py --accel cpu8 --tag $T4 --graphs L128,L256,L512,loopL512 --out results/timing_mac_r15_4b_cpu8_a.json
$EXPORT scripts/r15_timing_mac.py --accel cpu8 --tag $T4 --graphs L1024,L2048 --out results/timing_mac_r15_4b_cpu8_b.json
$EXPORT scripts/r15_timing_mac.py --accel gpu_f16 --tag $T4 --graphs L128,L512 --out results/timing_mac_r15_4b_gpu_f16.json
$EXPORT scripts/r15_timing_mac.py --accel gpu_f32 --tag $T4 --graphs L64 --out results/timing_mac_r15_4b_l64_gpu.json
$EXPORT scripts/r15_timing_mac.py --accel cpu8 --tag $T4 --graphs L64 --out results/timing_mac_r15_4b_l64_cpu.json
# the pairs: GPU memory with sharing on / off, then per request (1 / 2 / 3 / 5 questions on Ls 128, 2 / 3 on Ls 256)
# on Metal float32 with sharing on and off and on the CPU; S = "" for Ls 128, "_Ls256_Lq64" for Ls 256
for P in Ls128_Lq64 Ls256_Lq64; do
  S=$([ $P = Ls128_Lq64 ] || echo _$P)
  for sh in share noshare; do
    $EXPORT scripts/r15_timing_shared.py --pair $P --memory-probe $sh --tag $T4 --out results/memory_mac_r15_4b_pair${S}_$sh.json
  done
  $EXPORT scripts/r15_timing_shared.py --pair $P --accel gpu_f32 --share on --tag $T4 --out results/timing_mac_r15_4b_pair${S}_gpu_f32_share.json
  $EXPORT scripts/r15_timing_shared.py --pair $P --accel gpu_f32 --share off --tag $T4 --out results/timing_mac_r15_4b_pair${S}_gpu_f32_noshare.json
  $EXPORT scripts/r15_timing_shared.py --pair $P --accel cpu8 --share on --tag $T4 --out results/timing_mac_r15_4b_pair${S}_cpu8.json
done
```

### The earlier release (the loop kernel)

```bash
for L in 512 1024 2048; do $EXPORT scripts/torch_graph_parity.py --L $L; done  # 4B: --model 4b --L 1024 only
for L in 512 1024 2048; do
  $EXPORT scripts/export_kev.py --L $L         # 4B: --model 4b; --L 512 and --L 2048 also take --guard-L 1024
  $EXPORT scripts/quantize_kev.py --L $L --variant v2                          # [--model 4b]
done
#    desktop gate and timing as in step 8 with V=v2_fp16fc_i8emb and exports/kev08b_rowprefill_L${L}_v2_fp16fc_i8emb.tflite;
#    timing_mac.py --job 0.8b:card --job 0.8b:L2048 --out-0.8b results/timing_mac_0.8b.json (4B: --job 4b:card --out-4b ...)
$ORACLE scripts/host_head_export.py                     # 4B: --model 4b --write
$ORACLE scripts/host_tokenizer_probes.py
$HOST scripts/host_publish_fixtures.py --created-by scripts/make_fixtures.py   # 4B: --model 4b --only oracle_probs
$EXPORT scripts/device_rows.py                          # phone gate rows (Galaxy S26, Kev-0.8B)
$EXPORT scripts/device_compare.py --report device/<tag>.json --hsel device/hsel_<tag>.f32 --L 512
```

`host_parity.py` reads the host from `W/host/kev_litert.py` and the head from `W/host/kev_<model>_pointer_head.safetensors`; the tokenizer is the `tokenizer.json` the author's resolver placed in `W/hf`.

## Files

| File | Role |
|---|---|
| `r2_common.py` | Model selection (`--model`), paths, the reference loader, the host readout in numpy, the gate statistics |
| `r3_common.py` | Process memory, the embedding-table dtype check, the GPU-vs-CPU comparison |
| `make_fixtures.py`, `own_records.py` | The fixtures; the 12 records written for this conversion |
| `oracle_kev.py` | The reference: the author's code at `kev-1.0`, CPU, fp32 |
| `merge_check.py`, `merge_parity.py` | An independent fp32 fold of the LoRA against peft's merge and the author's merged checkpoint; the merged checkpoint against the adapter checkpoint |
| `tf514_baseline.py` | The merged checkpoint in stock transformers 5.14.1 against the reference |
| `kev_qwen35_patch.py` | The Qwen3.5 classes for export: the Gated DeltaNet chunk kernel in rank 4 (the loop kernel), the padding guard driven by `valid`, one-dimensional RoPE |
| `kev_graph.py` | The exported row graph: ids and valid in, hidden states out; the attention with the key/value copies by concat |
| `r12_kernel_product.py` | The forms of the in-chunk inverse that were tried; `R64` comes from here |
| `r11_fp16_safe.py` | The softplus rewrite (`sp`) and the other float16-safe rewrites |
| `r13_kernel.py` | The final kernel's forms on a loaded model (`R64`, `sp`, `ec`, `dd`, `vs<k>`, and rewrites not used) |
| `r15_form.py` | The Kev-4B tokens `fn<k>` and `in<k>` on top of `r13_kernel.py` |
| `r14_torch_parity.py`, `r15_torch_parity.py` | A form against the loop kernel in PyTorch fp32, all 402 questions |
| `r15_kernel_test.py` | The forms on Kev-4B's GatedDeltaNet shapes with random inputs and real layers |
| `r14_norm_range_probe.py`, `r15_norm_range_probe.py` | The float16 headroom of every norm |
| `r13_export.py`, `r15_export.py` | Export of a row graph with a form, op scan and V2 (Kev-0.8B, Kev-4B) |
| `r14_shared_state.py`, `r15_shared_state.py` | The shared-state pair in PyTorch: `state_prefill_<Ls>` and `question_step_<Ls>_<Lq>` (Kev-0.8B, any config) |
| `r14_shared_torch_parity.py`, `r15_shared_torch_parity.py` | The pair against the row form in PyTorch, all 402 questions, and the state checks |
| `r14_export_shared.py`, `r15_export_shared.py`, `quantize_shared_state.py` | Export of a pair (one file, two signatures), op scan and V2 with the RoPE table kept in float32 |
| `r14_litert_shared_parity.py`, `r15_shared_parity.py` | Desktop gate of a pair through the CompiledModel API (host round trip and direct buffer hand-over) |
| `r15_share_compare.py` | The Kev-4B pairs on Metal with and without constant tensor sharing, compared bit for bit |
| `export_kev.py`, `tflite_scan.py` | litert-torch export of the earlier release's graphs; the op-table scan of a file |
| `torch_graph_parity.py` | The earlier release's graph in PyTorch against the baseline and the reference (its export guard) |
| `quantize_kev.py` | ai-edge-quantizer storage variants (`v2` = the published one) |
| `litert_parity.py`, `gpu_gate_subprocess.py`, `gpu_vs_cpu.py` | Desktop gate through the CompiledModel API; GPU runs in a child process; GPU against CPU on the same file |
| `timing_mac.py` | Mac timing of the earlier release's files; the timing scripts below import its helpers |
| `host_head_export.py`, `host_tokenizer_probes.py`, `host_parity.py`, `host_author_crosscheck.py` | The head files, the tokenizer probes, the Python host against the reference, the desktop rows and the author's live code |
| `r16_head_json.py` | The `graph` entry of the head contract, read from the graph files |
| `r16_host_parity.py` | The Python host on the published files against the desktop gate's rows (row, pair and auto routes) |
| `host_publish_fixtures.py` | The published fixture files |
| `device_rows.py`, `device_compare.py` | Phone gate rows and the scoring of the phone's read-out hidden states (the earlier release) |
| `r14_device_rows.py`, `r14_device_rows_req.py`, `r18_req2.py` | The phone inputs of the final files: the gate rows of every bucket, the timing rows, the pairs' requests, the request points |
| `r12_device_compare.py` | The scoring of one phone gate run (its read-out hidden states) against the reference, with the desktop gate's statistics |
| `r18_burst.py`, `r18_crossover.py` | Phone timing: every timed call classified by the GPU and CPU clock caps at its start; the request table (the row graphs against the pairs) |
| `r14_timing_mac.py`, `r14_timing_mac_req.py`, `timing_mac_shared.py` | Mac timing of the Kev-0.8B files: each bucket's row and the pair; the request points |
| `r14_memory_mac.py` | Mac GPU memory of a Kev-0.8B pair or row file, one new process per leg |
| `r15_timing_mac.py`, `r15_timing_shared.py` | Mac timing of the Kev-4B files: the row graphs; the pairs per request and their GPU memory |
| `r14_flock.py` | Runs one command under the heavy-job lock (`W/.b_heavy.lock`); the Mac timing scripts take the same lock |
| `r17_kernel.py`, `r17_export.py`, `r17_torch_parity.py` | The FC rewrites on top of `r13_kernel.py` (`bk1024@in_proj_z.q_proj` = form C7), the export of a row graph with them, and their fp32 invariance on all 402 questions |
| `r18_host_parity.py` | `r16_host_parity.py` with the gate rows of chosen buckets taken from another form's files (C7 at L64 / L128 / L256) |
| `r18_head_json.py` | `r16_head_json.py` with the `graph.npu` entry |
| `r18_timing_mac.py`, `r18_timing_mac_req.py`, `r17_timing_mac.py` | Mac timing of form C7 next to form C: each bucket's row, the request points' rows |
| `r11_npu_fixtures.py`, `r17_npu_fixtures.py` | The NPU runner's inputs (rows and fixtures) for L128 / L256 and for L64 |
| `r17_runner_report.py`, `r17_device_compare.py`, `r17_leg_caps.py`, `r17_hold_table.py` | One NPU runner run as a report, its score against the reference (and against the same file on the Mac CPU), its clock-cap record, and the table of the runs |
| `env.sh`, `requirements-oracle.txt`, `requirements-export.txt` | The environments |

The `cache/r2`, `cache/r3`, `cache/r4b`, `cache/r14` and `cache/r15` folders the scripts create hold intermediates (hidden states, stock rows) and can be deleted after a run; `cache/r18` holds the two tables of step 13, and `cache/r17` the Mac hidden states of the form C7 files.
