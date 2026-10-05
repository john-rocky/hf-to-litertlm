# Reproduce the LiteRT files and the checks

Everything here ran on a Mac (Apple M4 Max, 128 GB, macOS 27.0) and one Samsung Galaxy S26 (SM-S942Q, 12 GB, Android 16) from 2026-10-03 to 2026-10-05. The scripts are in [`conversion/`](conversion/README.md), which gives the commands in order, the work-directory layout, the three Python environments and the kernel forms. The same folder ships with Kev-4B-LiteRT.

## Sources

| What | Where | Pin | File | Bytes | SHA-256 |
|---|---|---|---|---|---|
| Kev-0.8B adapter and head | `jaredpalmer/kev-0.8b` | tag `v1.0` = commit `bf75a6a8848ea6960ff2ed108d9ed44c2941174f` | `adapter_model.safetensors` | 43,338,624 | `9b908623acb162118575f4e7a94524f9c139c335be4bfb74d6cfceca01e1885a` |
| | | | `head.pt` | 2,103,999 | `f400bd12802b2b105ae45d6b03774a158a3db4fccff42413734ddca2e5c920b6` |
| | | | `tokenizer.json` | 19,989,325 | `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523` |
| | | | `tokenizer_config.json` | 1,128 | `8671bed7c852ce9e661be94f179a7b4ffd091c2a65aea0363e5501c20318ee45` |
| Base model | `Qwen/Qwen3.5-0.8B-Base` | `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` | `model.safetensors-00001-of-00001.safetensors` | 1,746,942,600 | `c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c` |
| The author's code | github.com/jaredpalmer/kev | tag `kev-1.0` = commit `6b719c3c3f367295f6ef336f4f751cf5ff970abc` | | | |

The Hub resolves the adapter's tag object `788ddbdd65715bb03a56788c822f6c632c9a551d` to the commit above; the scripts pin it as `jaredpalmer/kev-0.8b@788ddbdd…`.

## Environments

- Reference (the author's code): Python 3.12.11 with the author's lock file at `kev-1.0`, `conversion/requirements-oracle.txt` (torch 2.8.0, transformers 5.17.0, peft 0.21.0, huggingface-hub 1.32.0, datasets 5.0.1, tokenizers 0.23.2, numpy 2.5.3), and the kev package from a clone at tag `kev-1.0`.
- Export and quantization: Python 3.14.6, `conversion/requirements-export.txt` (litert-torch 0.9.4, litert-converter 0.4.0, torch 2.13.0, transformers 5.14.1, ai-edge-litert 2.2.0, ai-edge-quantizer 0.9.0, numpy 2.5.2).
- Python host: Python 3.12.11, `host/requirements-host.txt` (ai-edge-litert 2.2.0, numpy 2.5.2, tokenizers 0.23.2, safetensors 0.8.0).
- Phone: LiteRT 2.2.0 from Maven, Kotlin CompiledModel API. `android/CardSnippet.kt` holds the calls; it compiles with Kotlin 2.2.21, Android Gradle Plugin 8.9.1 and compileSdk 35. The phone's GPU numbers come from the measurement activity in `android/measure/` (built with the same versions), the NPU numbers from the C++ runner beside it (LiteRT 2.2.0 C API, Android NDK 29.0.13113456).

## Steps

Script names are files in `conversion/`; the commands and their arguments are in `conversion/README.md`.

1. Fixtures: `make_fixtures.py` (with `own_records.py`) writes `fixtures/requests.json`: 377 requests, 402 questions. The published `fixtures/rebuild_requests.py` rebuilds the same file (SHA-256 `3bcc256671b49838ea781c2ff3388cf67745b69af6977652f7e90bc15723e2c7`).
2. Reference: `oracle_kev.py` runs the author's code on the CPU in fp32 with one torch thread and writes the token rows, readout indices, logits, probabilities and answers, and the hidden states read out per question.
3. Merge: the author's `scripts/merge_lora_checkpoint.py` (fp32 full weights, `model.safetensors` 3,009,606,928 bytes, SHA-256 `b5ebf92a9994a96d5c0c21b52f5eae23049c9e8b5a85fc625b4cea64076408db`), checked by `merge_check.py` and `merge_parity.py` (the merged checkpoint gives the reference's probabilities exactly).
4. Unpatched baseline: `tf514_baseline.py` runs the merged checkpoint in stock transformers 5.14.1 against the reference.
5. The final kernel in PyTorch (form `R64+sp+ec+dd+vs6`, described in `conversion/README.md`). `r14_torch_parity.py` shows that the form does not change the fp32 result: on all 402 questions the read-out hidden states stay within 2.3e-5 and the probabilities within 4.7e-6 of the loop kernel's, with the same most likely option. `r14_shared_torch_parity.py` shows that the shared-state pair computes the row form: probabilities within 4.7e-6 and read-out hidden states within 2.5e-5 on all 402 questions, and the pair's state step on a whole row gives the row form's hidden states bit for bit. `r14_norm_range_probe.py` measures the float16 headroom of every norm; the smallest margin is the final norm's, 10.97×. The 64-, 128- and 256-token files use form C7 (`R64+sp+ec+dd+vs6+bk1024@in_proj_z.q_proj`, which also runs on the phone's NPU); `r17_torch_parity.py` shows the same for it: 2.3e-5 and 4.7e-6, the same most likely option on all 402 questions.
6. Export: `r17_export.py --L 64` (and 128, 256) with form C7 and `r13_export.py --L 512` (and 1024, 2048) with form C write the fp32 row graph with litert-torch 0.9.4, scan its operator table and write V2: FULLY_CONNECTED weights stored as float16 (ai-edge-quantizer `float_casting`, explicit dequantize, float compute) and the embedding table int8 channelwise. `r14_export_shared.py --Ls 128 --Lq 64` (and `--Ls 256`) does the same for the shared-state pair: one file with the signatures `state_prefill_<Ls>` and `question_step_<Ls>_64`, the RoPE table kept in float32. The outputs, renamed, are the published `kev-0.8b_rowprefill_L{L}_fp16fc_i8emb.tflite` and `kev-0.8b_sharedstate_Ls{Ls}_Lq64_fp16fc_i8emb.tflite` (same bytes). Form C7 adds one SUM over an axis of size 1 after each of the 24 in_proj_z and q_proj linears: 3,912, 4,759 and 6,019 operators in the V2 files of 64, 128 and 256 tokens, 24 more than form C's. None of the files has a CUSTOM operator, an int64 tensor or a tensor of rank above 4. Rebuilt from `conversion/` alone in a new work directory (the commands below, after step 5's checks), the form C 128-token row graph, the Ls 128 pair and the form C7 128-token row graph came out byte-identical to the files they were checked against (the form C 128-token file is not published).

   ```bash
   F=R64+sp+ec+dd+vs6; V=v2_fp16fc_i8emb_r14B-vs6
   F7=R64+sp+ec+dd+vs6+bk1024@in_proj_z.q_proj; V7=v2_fp16fc_i8emb_r17C7-bkzq
   for L in 64 128 256; do
     $EXPORT scripts/r17_export.py --L $L --form $F7 --tag r17C7-bkzq
     cp exports/kev08b_rowprefill_L${L}_$V7.tflite kev-0.8b_rowprefill_L${L}_fp16fc_i8emb.tflite
   done
   for L in 512 1024 2048; do
     $EXPORT scripts/r13_export.py --L $L --form $F --tag r14B-vs6
     cp exports/kev08b_rowprefill_L${L}_$V.tflite kev-0.8b_rowprefill_L${L}_fp16fc_i8emb.tflite
   done
   for Ls in 128 256; do
     $EXPORT scripts/r14_export_shared.py --Ls $Ls --Lq 64 --form $F --tag r14B-vs6
     cp exports/kev08b_sharedstate_Ls${Ls}_Lq64_$V.tflite kev-0.8b_sharedstate_Ls${Ls}_Lq64_fp16fc_i8emb.tflite
   done
   ```

7. Desktop gate: `litert_parity.py` on each row graph with the CompiledModel API, on the Mac CPU (8 threads) and on Metal at float32 precision (`gpu_gate_subprocess.py` runs the GPU in a child process); the 128- and 2,048-token graphs also at the GPU's default precision (float16 activations). `r14_litert_shared_parity.py` runs each pair on the CPU and on Metal at float32 precision, with and without constant tensor sharing, handing the state over through the host and directly between the two signatures. `gpu_vs_cpu.py` compares the GPU and CPU runs of one file. The form C7 files ran the same desktop gate; on the Mac CPU and on Metal at float32 they give the same probabilities as form C's files of the same length, bit for bit.
8. Head file: `head/kev_0.8b_pointer_head.safetensors` holds the four float32 tensors of `head.pt` bit for bit (`host_head_export.py`); `r18_head_json.py` writes the `graph` entry of the JSON beside it from the graph files, with `graph.npu` for the files that run on the NPU.
9. Tokenizer probes: `host_tokenizer_probes.py` writes the token IDs the author's tokenizer gives 12 strings.
10. Python host: `host_parity.py --skip-graph` checks `host/kev_litert.py` without a graph: token rows and readout indices against the reference (402 of 402), the head on the reference's hidden states, the answers, and the author's live code on extra requests (`host_author_crosscheck.py`). `r18_host_parity.py` (the row and auto routes, with the form C7 files' rows for 64, 128 and 256 tokens) and `r16_host_parity.py` (the pairs) then run the host on the published files and compare with step 7's rows. Through the row graphs (each question through the smallest graph that holds its row, and 50 questions forced through the 1,024-token graph), 452 of 452 questions are bit-identical on the Mac CPU (4 threads, against the 8-thread rows) and on Metal at float32. Through each pair, 314 of 314 (Ls 128) and 341 of 341 (Ls 256) are bit-identical on the CPU and on Metal with and without constant tensor sharing, with either state hand-over, and their probabilities differ from the row route's by at most 3.7e-6 on the CPU and 4.4e-6 on Metal. In the default `auto` route on the CPU, a request's questions go through a pair when their row graphs would compute more than 1.5 times as many positions as the pair: 9 requests go through a pair and the other 368 through the row graphs. Every question's most likely option and every request's answers equal those of the row route.
11. Mac timing: `r14_timing_mac.py --forms C` times each bucket's representative row (L64 `tv4_010`, 64 tokens; L128 `tv4_007`, 80 tokens; L256 the 5 rows of `own_fiveq_09` as one request; L512 and L1024 the 1,805-token row of `own_long_log_10` cut to 300 and 1,000 tokens; L2048 that whole row) and the Ls 128 pair, and `r14_timing_mac_req.py` the request points (the pairs, and the row graphs with each question's row in the smaller of the 128- and 256-token files that holds it); `r18_timing_mac.py --forms CD` and `r18_timing_mac_req.py` time the form C7 files next to form C in the same run, and `r18_timing_mac.py --accels cpu8 --max-load 2.5` timed their CPU rows again while the 1-minute load average was below 2.5. `r14_memory_mac.py` measured the pairs' memory on Metal at float32, one new process per leg, in two runs. Metal at float32 and the CPU with 8 threads (the request points on Metal only): 5 warm-up calls or requests, then 20 timed, in two passes of alternating order; a call is write the inputs + run + read back the whole hidden output.
12. Phone gate and timing: `r14_device_rows.py --form C` writes the token rows of every bucket, the timing rows and the pairs' requests; `r14_device_rows_req.py` and `r18_req2.py` write the request points (the FP32 run of the 2,048-token file used rows 1 to 11 of `rows_L2048.json`: the 9 long rows, the control row and `tv4_000`). The measurement activity of `android/measure/` runs them on the phone and saves the read-out hidden states and the per-call times, while the host reads the phone's clocks, temperatures and memory every 2 seconds. `r12_device_compare.py` scores each gate run against the reference with the desktop gate's statistics; `r18_burst.py` sorts every timed call by the clock caps at its start and `r18_crossover.py` builds the request table. For the NPU, `r11_npu_fixtures.py` and `r17_npu_fixtures.py` write the inputs of `android/measure/kev_npu_runner.cc`, and `r17_runner_report.py`, `r17_device_compare.py`, `r17_leg_caps.py` and `r17_hold_table.py` turn its runs into reports, scores, clock-cap records and one table.
13. Published files: `host_publish_fixtures.py` writes `fixtures/requests_public.json` and `fixtures/oracle_probs.json`; `SHA256SUMS` lists every file of this repository.

## Gate

A file passes when, over the questions whose rows fit it, against the reference: the most likely option is the same for every question whose reference top-two probabilities are more than 0.02 apart (closer pairs are reported separately as near ties), the largest absolute difference of any option's probability is at most 0.02, and the mean absolute difference over all options is at most 0.002. A control request (`red_arm_000`, one word of `tv4_000`'s instructions changed) must differ from `tv4_000`'s reference by more than 0.02, which shows the tolerance can catch a one-word change; its row has 94 tokens, so the 64-token graph does not run it.

| File | Runtime | Questions | Same most likely option | Near ties kept | Max \|Δp\| | Mean \|Δp\| | Control |
|---|---|---:|---:|---:|---:|---:|---:|
| L64 | Mac CPU, 8 threads | 72 | 71/71 | 1/1 | 0.0078 | 8.8e-4 | — |
| L64 | Mac GPU (Metal), float32 | 72 | 71/71 | 1/1 | 0.0078 | 8.8e-4 | — |
| L128 | Mac CPU, 8 threads | 321 | 311/311 | 9/10 | 0.0104 | 9.7e-4 | 0.0477 |
| L128 | Mac GPU (Metal), float32 | 321 | 311/311 | 9/10 | 0.0104 | 9.7e-4 | 0.0477 |
| L256 | Mac CPU, 8 threads | 385 | 371/371 | 12/14 | 0.0104 | 9.1e-4 | 0.0477 |
| L256 | Mac GPU (Metal), float32 | 385 | 371/371 | 12/14 | 0.0104 | 9.1e-4 | 0.0477 |
| L512 | Mac CPU, 8 threads | 392 | 377/377 | 13/15 | 0.0104 | 9.1e-4 | 0.0477 |
| L512 | Mac GPU (Metal), float32 | 392 | 377/377 | 13/15 | 0.0104 | 9.1e-4 | 0.0477 |
| L1024 | Mac CPU, 8 threads | 392 | 377/377 | 13/15 | 0.0104 | 9.1e-4 | 0.0477 |
| L1024 | Mac GPU (Metal), float32 | 392 | 377/377 | 13/15 | 0.0104 | 9.1e-4 | 0.0477 |
| L2048 | Mac CPU, 8 threads | 401 | 386/386 (long rows 9/9) | 13/15 | 0.0104 | 8.9e-4 | 0.0477 |
| L2048 | Mac GPU (Metal), float32 | 401 | 386/386 (long rows 9/9) | 13/15 | 0.0104 | 8.9e-4 | 0.0477 |
| Pair Ls128 | Mac CPU; Metal float32 with and without sharing | 313 | 306/306 | 6/7 | 0.0078 | 9.5e-4 | 0.0477 |
| Pair Ls256 | Mac CPU; Metal float32 with and without sharing | 340 | 329/329 | 9/11 | 0.0078 | 9.1e-4 | 0.0477 |
| L128 | Mac GPU (Metal), default precision (float16 activations) | 321 | 311/311 | 9/10 | 0.0332 | 3.6e-3 | 0.0469 |
| L2048 | Mac GPU (Metal), default precision (float16 activations) | 401 | 385/386 | 11/15 | 0.0352 | 4.1e-3 | 0.0539 |

Every file passes at float32 on the CPU and on Metal. The 64-, 128- and 256-token rows are the form C7 files' (bit for bit the same probabilities as form C's). At the GPU's default precision the hidden states stay finite on every row, but the largest and mean differences are outside the tolerance. At float32 the near ties that changed are `tv4_023` (reference top-two gap 0.0020) and `own_sensor_08` (8.1e-5), on every file that holds their rows: the 64-token graph holds neither, the 128-token graph and the Ls 128 pair hold only `tv4_023`. The long rows are the 9 questions on the three long invented states (1,369 to 1,805 tokens), largest difference 0.0015. On the same file, the Metal GPU at float32 and the CPU agree within 6.9e-6 (row graphs) and 6.4e-6 (pairs); the two state hand-overs give the same hidden states bit for bit, and Metal with and without constant tensor sharing gives the same probabilities bit for bit. On Metal at float32, a pair and the 512-token row graph agree within 4.0e-6 on the same questions (3.99e-6 for Ls 128, 2.98e-6 for Ls 256).

## Galaxy S26

One Samsung Galaxy S26 (SM-S942Q, 12 GB, Android 16) with LiteRT 2.2.0 from Maven, through the Kotlin `CompiledModel` API in the measurement activity of `android/measure/`, on the GPU (OpenCL). The phone saved the read-out hidden states of every question; the numbers below are the published files' agreement with the reference, scored on the Mac by `r12_device_compare.py` with the tolerance of the Gate section. The delegate column is the logcat line `Replacing N out of N node(s) with delegate (LITERT_CL) node, yielding 1 partitions`, and the compile time is `CompiledModel.create` in that run.

| File | GPU precision | Questions | Same most likely option | Near ties kept | Max \|Δp\| | Mean \|Δp\| | Control | Delegate | Compile |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| L64 | FP16_WITH_FP32_ACCUM | 72 | 71/71 | 1/1 | 0.0076 | 1.0e-3 | — | 3,912 of 3,912 | 5.8 s |
| L128 | FP16_WITH_FP32_ACCUM | 321 | 311/311 | 9/10 | 0.0077 | 1.1e-3 | 0.0462 | 4,759 of 4,759 | 6.1 s |
| L256 | FP16_WITH_FP32_ACCUM | 385 | 371/371 | 13/14 | 0.0092 | 1.0e-3 | 0.0472 | 6,019 of 6,019 | 11.0 s |
| L512 | FP16_WITH_FP32_ACCUM | 392 | 377/377 | 13/15 | 0.0091 | 1.1e-3 | 0.0473 | 8,515 of 8,515 nodes | 6.7 s |
| L1024 | FP16_WITH_FP32_ACCUM | 392 | 377/377 | 13/15 | 0.0091 | 1.1e-3 | 0.0473 | 13,555 of 13,555 | 8.1 s |
| L2048 | FP16_WITH_FP32_ACCUM | 401 | 386/386 (long rows 9/9) | 13/15 | 0.0093 | 1.1e-3 | 0.0473 | 23,635 of 23,635 | 14.3 s |
| Pair Ls128, constant tensor sharing | FP16_WITH_FP32_ACCUM | 313 | 306/306 | 6/7 | 0.0101 | 1.1e-3 | 0.0467 | 4,975 of 4,975 (`state_prefill_128`), 3,965 of 3,965 (`question_step_128_64`) | 9.9 s |
| Pair Ls256, constant tensor sharing | FP16_WITH_FP32_ACCUM | 340 | 329/329 | 9/11 | 0.0110 | 1.1e-3 | 0.0467 | 6,235 of 6,235 (`state_prefill_256`), 3,965 of 3,965 (`question_step_256_64`) | 10.5 s |
| Pair Ls128, no sharing | FP16_WITH_FP32_ACCUM | 313 | 306/306 | 6/7 | 0.0101 | 1.1e-3 | 0.0467 | 4,975 of 4,975, 3,965 of 3,965 | 14.6 s |
| Pair Ls256, no sharing | FP16_WITH_FP32_ACCUM | 340 | 329/329 | 9/11 | 0.0110 | 1.1e-3 | 0.0467 | 6,235 of 6,235, 3,965 of 3,965 | 16.3 s |
| L128 | FP32 | 321 | 311/311 | 9/10 | 0.0104 | 9.7e-4 | 0.0477 | 4,759 of 4,759 | 8.5 s |
| L2048: the 9 long rows, `tv4_000` and the control row | FP32 | 10 | 10/10 (long rows 9/9) | — | 0.0015 | 4.4e-4 | 0.0477 | 23,635 of 23,635 | 22.0 s |

Every run passes, with finite hidden states on every row. The near ties that changed are `tv4_023` on every file that holds its row and `own_sensor_08` on the 512-, 1,024- and 2,048-token files and the Ls 256 pairs. Each pair ran every request with both state hand-overs, which gave the same hidden states bit for bit. The 64-, 128- and 256-token files read out the same hidden states on the phone as form C's files of the same length, byte for byte, and each pair without constant tensor sharing the same hidden states as with it. The 9 long rows come closer to the reference at FP32 than at FP16_WITH_FP32_ACCUM: largest difference 0.0015 and mean 4.3e-4 at FP32, 0.0093 and 1.8e-3 in the FP16_WITH_FP32_ACCUM run of the 2,048-token file (whose other 392 questions have a mean of 1.1e-3). In the gate runs without sharing the app's peak resident memory (VmHWM) was 6.88 and 7.00 GB, the GPU's page allocation 3.18 and 3.22 GB, and the phone's smallest MemAvailable 2.73 and 3.10 GB (Ls 128, Ls 256).

### Timing

A call is write the inputs + run + read back the whole hidden output, timed in the activity together with its start time. The phone's GPU lowers its clock cap under load, and compiling a graph is enough to start it: in a check run of the form C 128-token file at FP32, the GPU went from 41 °C to 60 °C during the 7.6 s compile, the samples taken during the calls showed the cap at 1,100 and then 1,050 MHz, and the 5 warm-up and 12 timed calls made then took 143.4 to 161.5 ms; the last 8 calls took 134.6 to 137.3 ms, and the next sample showed the cap back at 1,300 MHz. So each timing run waits after the compile until the cap and the temperature are back (`cool_ms`), and every call is classified by the clock caps the host read nearest to its start: cool = the GPU at 1,300 MHz (`thermal_pwrlevel` 0) and every CPU core at its maximum frequency, sustained = a cap lowered. After the wait the 64- and 128-token runs below (1.4 and 2.5 s of calls) ran with neither cap lowered; in the 256-, 512-, 1,024- and 2,048-token runs a cap came 5.4 to 7.4 s into the back-to-back calls (warm-up included).

| File | Row | Cool: median (calls) | Sustained: median (calls) |
|---|---|---:|---:|
| L64 | 64 tokens | 57.4 ms (20) | — |
| L128 | 80 tokens | 102.4 ms (20) | — |
| L256 | 5 rows of 128 to 142 tokens (one 5-question request) | 199.4 ms per call (32); 992.9 ms per request (7) | 220.5 ms per call (68); 1,088.3 ms per request (13) |
| L512 | 300 tokens | 387.0 ms (9) | 393.4 ms (11) |
| L1024 | 1,000 tokens | 819.8 ms (7) | 847.7 ms (13) |
| L2048 | 1,805 tokens | 1,808.2 ms (2) | 2,147.6 ms (18) |

GPU precision FP16_WITH_FP32_ACCUM, 5 warm-up calls (2 for L1024 and L2048), then 20 timed calls (L256: 20 requests of 5 calls). At FP32 with the same settings the cool medians were 85.3 ms (L64, 20 calls), 137.4 ms (L128, 18 calls), and 270.2 ms per call (30) and 1,355.6 ms per request (6) on L256. At FP32 L512 ran 5 warm-up and 20 timed calls, L1024 2 and 10, and L2048 2 and 6: cool 565.1 ms (16 calls), 1,219.0 ms (3) and 3,119.6 ms (2 calls, 2,971.9 and 3,267.3 ms), sustained 563.1 ms (4), 1,316.8 ms (7) and 3,121.2 ms (4).

### Requests: the row graphs or a pair

A request's row-graph time is the sum over its questions' rows, each through the smallest graph that holds it; a pair answers the request with `state_prefill` once and `question_step` per question, the state handed over directly. Each cell is the median of the requests timed with the GPU clock not capped, with their number in parentheses (2 warm-up and 6 timed requests per point, FP16_WITH_FP32_ACCUM).

| Request | Questions | Row graphs | Pair Ls128, sharing | Pair Ls128, no sharing | Pair Ls256, sharing | Pair Ls256, no sharing |
|---|---:|---:|---:|---:|---:|---:|
| `tv4_000` | 1 | 102.7 ms (6) | 249.1 ms (6) | 179.3 ms (6) | — | — |
| `own_ticket_01`, its questions 1 and 2 | 2 | 394.7 ms (6) | 341.8 ms (2) | 242.1 ms (6) | — | — |
| `own_order_06` | 2 | 399.0 ms (6) | — | — | 445.2 ms (5) | 345.3 ms (6) |
| `own_ticket_01` | 3 | 594.2 ms (6) | 441.5 ms (5) | 304.4 ms (4) | — | — |
| `own_email_03` | 3 | 593.5 ms (5) | — | — | 538.4 ms (2) | 407.0 ms (6) |
| `own_fiveq_09` | 5 | 800.0 ms (6 + 6) | 624.8 ms (2) | 430.9 ms (6) | — | — |

The activity loads one graph per run, so the 5-question request's row value is the sum of two runs: its 2 rows on the 128-token file (205.6 ms, 6 requests) and its 3 rows on the 256-token file (594.4 ms, 6 requests). The CPU clock was capped during the pair requests without sharing of 3 and 5 questions and during requests on the 256-token row graph (all of them for `own_ticket_01`'s questions 1 and 2, `own_order_06` and `own_email_03`, 2 of 6 for `own_ticket_01` and 4 of 6 for `own_fiveq_09`'s rows); the other cells ran with neither clock capped. Without constant tensor sharing the GPU holds the weights twice. During the pair runs the app's peak resident memory (VmHWM) was 6.56 to 6.86 GB without sharing and 3.10 to 3.16 GB with it, the GPU's page allocation 3.17 to 3.21 GB and 1.73 to 1.77 GB, and the phone's smallest MemAvailable 2.79 to 3.06 GB and 5.78 to 6.13 GB (GB = the `/proc` kB × 1,024 / 10⁹).

### NPU

The 64-, 128- and 256-token files also run on the phone's NPU (the Qualcomm HTP) with LiteRT 2.2.0, through its C API in `android/measure/kev_npu_runner.cc` (`android/measure/README.md` gives the build and run commands); the Kotlin API's NPU path was not measured. The accelerators are the NPU and the CPU: the Qualcomm compiler takes every operator but the int8 embedding lookup (`EMBEDDING_LOOKUP`), which runs on the CPU. The logcat shows `Partitioned subgraph<0>, selected 4758 ops, from a total of 4759 ops. resulted in 2 partitions.` for the 128-token file (3,911 of 3,912 and 6,018 of 6,019 ops for the 64- and 256-token files), then `Replacing 2 out of 3 node(s) with delegate (DispatchDelegate) node, yielding 3 partitions`. The files of 512 tokens and more and the pairs were not run on the NPU. Loading a file compiles it on the phone (Qualcomm's JIT compiler plugin with the QAIRT 2.47 libraries) and keeps the compiled graph in a cache folder, which later loads of the same file read. Those libraries and LiteRT's Qualcomm dispatch library are not part of this repository.

| File | Questions | Same most likely option | Near ties kept | Max \|Δp\| | Mean \|Δp\| | Control | Compile (JIT) | Cache folder | Load from the cache |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L64 | 72 | 71/71 | 1/1 | 0.0105 | 1.2e-3 | — | 55.7 s | 1.27 GB | 1.6 s |
| L128 | 321 | 311/311 | 10/10 | 0.0134 | 1.3e-3 | 0.0476 | 191.2 s | 1.28 GB | 1.5 s, 1.6 s |
| L256 | 385 | 371/371 | 13/14 | 0.0134 | 1.2e-3 | 0.0476 | 263.3 s | 1.29 GB | 1.7 s |

Every file passes, with finite hidden states on every row; the near tie that changed is `own_sensor_08` (reference top-two gap 8.1e-5) on the 256-token file. The runner saved the read-out hidden states of every question, scored on the Mac against the reference with the tolerance of the Gate section (`r17_device_compare.py`). The runner's peak resident memory (VmHWM) was 5.54 to 6.23 GB while the JIT compiled a file and 2.34 to 2.37 GB in a run from the cache.

Each timing run loaded the file from the cache, ran 20 different rows once each, and then timed one row: 5 warm-up calls, then 20 timed calls (the second 128-token run: 10 and 40). A call is write the inputs + run + read back the hidden output, timed in the runner.

| File | Timed row | One row: median (calls) | 20 different rows: median |
|---|---|---:|---:|
| L64 | `tv4x_emotion_00`, 42 tokens | 44.0 ms (20) | 44.1 ms |
| L128 | `tv4_007`, 80 tokens | 67.8 ms (20); 71.1 ms (40) | 69.9 ms; 72.4 ms |
| L256 | `tv4_006`, 237 tokens | 126.7 ms (20) | 126.4 ms |

In the host's samples during these runs the GPU clock was never capped; the CPU cores were at their maximum in the 64-token run, while in the 128- and 256-token runs the cores of `cpufreq/policy6` were capped at 4.26 to 4.65 GHz of 4.74 GHz (and in the 256-token run those of `policy0` at 3.51 of 3.63 GHz). The samples do not include the NPU's clock. On the same runner and row, the 128-token file on the GPU (FP16_WITH_FP32_ACCUM) took 101.9 ms (20 calls) and 102.3 ms (40 calls).

## Where the numbers come from

All paths are in the work directory of `conversion/README.md`. `V` = `v2_fp16fc_i8emb_r14B-vs6`.

| Number | Produced by |
|---|---|
| Reference probabilities and answers | `oracle_kev.py` (`oracle/oracle_0.8b.json`) |
| Merge agreement | `merge_check.py`, `merge_parity.py` (`results/merge_0.8b.json`, `results/merge_parity_0.8b.json`) |
| fp32 invariance of the final kernel | `r14_torch_parity.py` (`results/r14_torch_parity_R64-sp-ec-dd-vs6.json`); form C7: `r17_torch_parity.py` (`results/r17_torch_parity_R64-sp-ec-dd-vs6-bk1024_in_proj_z.q_proj.json`) |
| The pair against the row form in PyTorch | `r14_shared_torch_parity.py` (`results/r14_shared_torch_parity_R64-sp-ec-dd-vs6.json`) |
| Float16 headroom of the norms | `r14_norm_range_probe.py` (`results/r14_norm_range_R64-sp-ec-dd-vs6.json`) |
| Operators of each file | `r13_export.py`, `r14_export_shared.py` (`results/export_L{L}_r14B-vs6.json`, `results/quant_L{L}_$V.json`, `results/export_sharedstate_Ls{Ls}_Lq64_r14B-vs6.json`, `results/quant_sharedstate_Ls{Ls}_Lq64_$V.json`); form C7: `r17_export.py` (`results/export_L{64,128,256}_r17C7-bkzq.json`, `results/quant_L{64,128,256}_v2_fp16fc_i8emb_r17C7-bkzq.json`) |
| Desktop CPU and Metal agreement of the row graphs | `litert_parity.py` (`results/litert_{cpu,gpu_f32,gpu_f16}_rows_L{L}_$V.json`, and for the form C7 files `results/litert_{cpu,gpu_f32}_rows_L{64,128,256}_v2_fp16fc_i8emb_r17C7-bkzq.json`; the statistics above are recomputed from these rows) |
| Desktop agreement of the pairs | `r14_litert_shared_parity.py` (`results/litert_shared_{cpu,gpu_f32,gpu_f32_share}_rows_Ls{Ls}_Lq64_$V.json`) |
| Metal against the CPU on the same file | `gpu_vs_cpu.py` (`results/litert_gpu_f32_vs_cpu_L{L}_$V.json`), `r14_litert_shared_parity.py` (`same_questions` in `results/litert_shared_gpu_f32*_parity_Ls{Ls}_Lq64_$V.json`) |
| Python host agreement | `host_parity.py --skip-graph`, `r18_host_parity.py` (`results/host_parity_v2_{row,auto}_*_C7.json`), `r16_host_parity.py` (`results/host_parity_v2_pair_*.json`) |
| Rebuild from `conversion/` | the commands of `conversion/README.md` in an empty work directory; SHA-256 against `SHA256SUMS` |
| S26 agreement | `r12_device_compare.py` on the activity's reports and hidden-state dumps (form C7: `device/r14/device_parity_kev_s26_r14_G_C7_gpu_fp16acc32_L{64,128,256}_gate.json`, `device/r14/device_parity_kev_s26_r14_G_C7_gpu_fp32_L128_gate.json`; form C: `device/r14/device_parity_kev_s26_r14_G_C_gpu_fp16acc32_L{512,1024,2048}_gate.json`, `device/r14/device_parity_kev_s26_r14_G_C_gpu_fp32_L2048_long11_gate.json` (rows `device/r14/rows_L2048_long11.json`); the pairs: `device/r14/device_parity_kev_s26_r14_{P,P256}_C_gpu_fp16acc32_share_gate_{direct,host}.json` and `…_{PN,PN256}_C_gpu_fp16acc32_noshare_gate_{direct,host}.json`; the statistics above are recomputed from their `per_row`, and the byte comparisons are of the dumps `device/r14/hsel_<run>.f32`); the delegate lines and compile times from each run's logcat and report; the memory of the gate runs without sharing from their samples |
| S26 timing and memory | `r18_burst.py`, `r18_crossover.py` on the reports' per-call records and the host's samples (`results/r18_burst.json`, `results/r18_crossover.json`; runs `device/r14/kev_s26_r14_T4_C7_gpu_{fp16acc32,fp32}_L{64,128,256}_timing`, `device/r14/kev_s26_r14_T4_C_gpu_{fp16acc32,fp32}_L{512,1024,2048}_timing`, the request rows `device/r14/kev_s26_r14_V2R_C7_gpu_fp16acc32_L{128,256}_timing` and the pairs `device/r14/kev_s26_r14_V2{S,N}_C_gpu_fp16acc32_Ls{128,256}_timing`); the memory from the same runs' samples; the compile example from `device/r14/kev_s26_r14_T2_C_gpu_fp32_L128_timing` |
| Mac timing | `r14_timing_mac.py`, `r14_timing_mac_req.py` (`results/timing_mac_r14_vs6.json`, `results/timing_mac_r14_req.json`); the form C7 files: `r18_timing_mac.py`, `r18_timing_mac_req.py` (`results/timing_mac_r18_C7.json`, `results/timing_mac_r18_C7_req.json`; the CPU rows again: `results/timing_mac_r18_C7_cpu.json`); the pairs' memory: `r14_memory_mac.py` (`results/memory_mac_r14_pair_Ls{128,256}_{share,noshare}_run{1,2}.json`) |
| NPU | `android/measure/kev_npu_runner.cc` with the inputs of `r11_npu_fixtures.py` / `r17_npu_fixtures.py`; `r17_runner_report.py`, `r17_device_compare.py` and `r17_leg_caps.py` on each run's stdout, hidden-state dump, logcat and samples (`device/r17/B/kev_s26_r17_B_<run>.json`, `device/r17/B/device_parity_kev_s26_r17_B_<run>.json`, `device/r17/B/kev_s26_r17_B_<run>.caps.json`). Scores, JIT compile and memory: the runs `N7_64_npucpu_jit_cold_c7`, `N7_npucpu_jit_cold_c7` and `N7_256_npucpu_jit_cold_c7` (the statistics above are recomputed from their `per_row`; the cache folder's size is `du -k` in `<run>.context.txt`); timing: `N7c64_npucpu_cached_c7`, `N7c1_npucpu_cached_c7`, `N7c2_npucpu_cached_c7`, `N7c256_npucpu_cached_c7`, and on the GPU `G7t1_gpu_fp16acc32_c7`, `G7t2_gpu_fp16acc32_c7` |
| SHA-256 of the published files | `SHA256SUMS` |
