# Reproduce the LiteRT files and the checks

Everything here ran on a Mac (Apple M4 Max, 128 GB, macOS 27.0) on 2026-10-03 and 2026-10-04, except the two phone runs of the Phone section (2026-10-03 and 2026-10-05). The scripts are in [`conversion/`](conversion/README.md), which gives the commands in order, the work-directory layout, the three Python environments and the kernel forms; the Kev-4B commands take `--model 4b`. The same folder ships with Kev-0.8B-LiteRT, whose fixtures, tokenizer and Python host are the same as here.

## Sources

| What | Where | Pin | File | Bytes | SHA-256 |
|---|---|---|---|---|---|
| Kev-4B adapter and head | `jaredpalmer/kev-4b` | tag `v1.0` = commit `6cfce5c2fa4b4bd64026336ab649c5ca78857d52` | `adapter_model.safetensors` | 129,924,032 | `90e817356246e7f18bfa7ca3d31794cd4fbeb3332a66a84cb51d9ceae925f2b2` |
| | | | `head.pt` | 5,249,791 | `dd633435998ecc751ac538717a3742e32149500fabf7d7276287dbf0693f347c` |
| | | | `tokenizer.json` | 19,989,325 | `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523` |
| | | | `tokenizer_config.json` | 1,128 | `8671bed7c852ce9e661be94f179a7b4ffd091c2a65aea0363e5501c20318ee45` |
| Base model | `Qwen/Qwen3.5-4B-Base` | `1001bb4d826a52d1f399e183466143f4da7b741b` | `model.safetensors-00001-of-00002.safetensors` | 5,329,398,712 | `df547074dce70532a0493e5433152bd17a65efb89088cfabc2e7e2371a93d712` |
| | | | `model.safetensors-00002-of-00002.safetensors` | 3,990,429,344 | `590fbaac095dd31db886c322d9d2f7df47777966391acf306ddddc3e4e3a15ef` |
| The author's code | github.com/jaredpalmer/kev | tag `kev-1.0` = commit `6b719c3c3f367295f6ef336f4f751cf5ff970abc` | | | |

The Hub resolves the adapter's tag object `591dcb5bd6d05eb0b5131ea6608f93f10243335c` to the commit above; the scripts pin it as `jaredpalmer/kev-4b@591dcb5b…`. The tokenizer files are byte-identical to the ones in `jaredpalmer/kev-0.8b`.

## Environments

- Reference (the author's code): Python 3.12.11 with the author's lock file at `kev-1.0`, `conversion/requirements-oracle.txt` (torch 2.8.0, transformers 5.17.0, peft 0.21.0, huggingface-hub 1.32.0, datasets 5.0.1, tokenizers 0.23.2, numpy 2.5.3), and the kev package from a clone at tag `kev-1.0`.
- Export and quantization: Python 3.14.6, `conversion/requirements-export.txt` (litert-torch 0.9.4, litert-converter 0.4.0, torch 2.13.0, transformers 5.14.1, ai-edge-litert 2.2.0, ai-edge-quantizer 0.9.0, numpy 2.5.2).
- Python host: Python 3.12.11, `host/requirements-host.txt` (ai-edge-litert 2.2.0, numpy 2.5.2, tokenizers 0.23.2, safetensors 0.8.0).
- Kotlin: `android/CardSnippet.kt` (the same file as in Kev-0.8B-LiteRT) holds the CompiledModel calls (LiteRT 2.2.0 from Maven); it compiles with Kotlin 2.2.21, Android Gradle Plugin 8.9.1 and compileSdk 35. Kev-4B passes on the Mac's GPU at float32 precision (`Precision.FP32` in Kotlin); the snippet's default for the row graphs, `FP16_WITH_FP32_ACCUM`, has not been checked with Kev-4B, and Kev-4B has not run on a phone (the Phone section).

## Steps

Script names are files in `conversion/`; the commands and their arguments are in `conversion/README.md`.

1. Fixtures: `make_fixtures.py` (with `own_records.py`) writes `fixtures/requests.json`: 377 requests, 402 questions, the same file as for Kev-0.8B. The published `fixtures/rebuild_requests.py` rebuilds it (SHA-256 `3bcc256671b49838ea781c2ff3388cf67745b69af6977652f7e90bc15723e2c7`).
2. Reference: `oracle_kev.py --model 4b` runs the author's code on the CPU in fp32 with one torch thread and writes the token rows, readout indices, logits, probabilities and answers, and the hidden states read out per question. The token rows are the same as Kev-0.8B's (one tokenizer).
3. Merge: the author's `scripts/merge_lora_checkpoint.py` (fp32 full weights in 4 shards, 16,852,777,430 bytes; the author's `merge.json` gives `weights_sha256` `904380cbf0be134e122f2abd7c7bab52f1e2a0f373d3a2e7947a3025e882f3f3`). `merge_check.py --model 4b` folds the LoRA independently in fp32 and compares it with peft's merge and the author's checkpoint (426 of 426 tensors bit-identical); `oracle_kev.py --model 4b --ckpt merged/… --tag merged_4b` and `merge_parity.py --model 4b` show that the merged checkpoint gives the reference's probabilities exactly.
4. Unpatched baseline: `tf514_baseline.py --model 4b` runs the merged checkpoint in stock transformers 5.14.1 against the reference (the same most likely option on every row, max probability difference 1.3e-5).
5. The final kernel in PyTorch (form `R64+sp+ec+dd+vs6+in1+fn5`, described in `conversion/README.md`). `r15_kernel_test.py` runs the kernel forms on Kev-4B's Gated DeltaNet shapes (twice as many value heads as key heads) with random inputs and real layers. `r15_torch_parity.py` shows that the form does not change the fp32 result: on all 402 questions the read-out hidden states stay within 4.8e-5 and the probabilities within 5.3e-6 of the loop kernel's, with the same most likely option. `r15_shared_torch_parity.py` shows that the shared-state pair computes the row form: probabilities within 3.8e-6 and read-out hidden states within 4.6e-5 on all 402 questions, and the pair's state step on a whole row gives the row form's hidden states bit for bit. `r15_norm_range_probe.py` measures the float16 headroom of every norm; with `in1` and `fn5` the smallest margin is the post_attention_layernorm's, 6.55×.
6. Export: `r15_export.py --model 4b --L 64` (and 128, 256, 512, 1024, 2048) writes the fp32 row graph with litert-torch 0.9.4, scans its operator table and writes V2: FULLY_CONNECTED weights stored as float16 (ai-edge-quantizer `float_casting`, explicit dequantize, float compute) and the embedding table int8 channelwise. `r15_export_shared.py --model 4b --Ls 128 --Lq 64` (and `--Ls 256`) does the same for the shared-state pair: one file with the signatures `state_prefill_<Ls>` and `question_step_<Ls>_64`, the RoPE table kept in float32. The outputs, renamed, are the published `kev-4b_rowprefill_L{L}_fp16fc_i8emb.tflite` and `kev-4b_sharedstate_Ls{Ls}_Lq64_fp16fc_i8emb.tflite` (same bytes). None of the files has a CUSTOM operator, an int64 tensor or a tensor of rank above 4. An fp32 export holds about 50 GB of memory at its peak. Rebuilt from `conversion/` alone in a new work directory, with step 5's fp32 check as the export's guard, the 128-token row graph came out byte-identical to the published file.

   ```bash
   F4=R64+sp+ec+dd+vs6+in1+fn5; V4=v2_fp16fc_i8emb_r15R64-sp-ec-dd-vs6-in1-fn5
   for L in 64 128 256 512 1024 2048; do
     $EXPORT scripts/r15_export.py --model 4b --L $L --form $F4
     cp exports/kev4b_rowprefill_L${L}_$V4.tflite kev-4b_rowprefill_L${L}_fp16fc_i8emb.tflite
   done
   for Ls in 128 256; do
     $EXPORT scripts/r15_export_shared.py --model 4b --Ls $Ls --Lq 64 --form $F4
     cp exports/kev4b_sharedstate_Ls${Ls}_Lq64_$V4.tflite kev-4b_sharedstate_Ls${Ls}_Lq64_fp16fc_i8emb.tflite
   done
   ```

7. Desktop gate: `litert_parity.py --model 4b` on each row graph with the CompiledModel API on Metal at float32 precision (`gpu_gate_subprocess.py` runs the GPU in a child process), and on the Mac CPU (8 threads) for the 1,024-token graph; the 128- and 512-token graphs also at the GPU's default precision (float16 activations). `r15_shared_parity.py --model 4b` runs each pair on the CPU (8 threads) and on Metal at float32 precision with and without constant tensor sharing, handing the state over through the host and directly between the two signatures; `r15_share_compare.py` compares the two Metal runs.
8. Head file: `head/kev_4b_pointer_head.safetensors` holds the four float32 tensors of `head.pt` bit for bit (`host_head_export.py --model 4b --write`, temperature 2.406050072164233); `r16_head_json.py --model 4b` writes the `graph` entry of the JSON beside it from the graph files.
9. Python host: `host_parity.py --model 4b --skip-graph` checks `host/kev_litert.py` (the same file as for Kev-0.8B) without a graph: token rows and readout indices against the reference (402 of 402), the head on the reference's hidden states (max probability difference 1.8e-7), the answers, and the author's live code on extra requests (`host_author_crosscheck.py`). `r16_host_parity.py --model 4b` then runs the host on the published files and compares with step 7's rows. Through the row graphs on Metal at float32 (each question through the smallest graph that holds its row, and 50 questions forced through the 1,024-token graph), 452 of 452 questions are bit-identical. Through each pair on Metal at float32 with constant tensor sharing, 314 of 314 (Ls 128) and 341 of 341 (Ls 256) are bit-identical with either state hand-over, and their probabilities differ from the row route's by at most 5.6e-6. On the CPU (4 threads), the first 20 questions that fit the 128-token graph stay within 3.7e-6 of step 7's Metal rows (step 7 ran the CPU on the 1,024-token graph only).
10. Mac timing: `r15_timing_mac.py` times the row graphs, each in its own child process, on Metal at float32 and on the CPU with 8 threads (the 128- and 512-token graphs also at Metal's default precision, float16 activations), with each bucket's representative row (L64 `tv4_010`, 64 tokens; L128 `tv4_007`, 80 tokens; L256 the 5 rows of `own_fiveq_09` as one request; L512 and L1024 the 1,805-token row of `own_long_log_10` cut to 300 and 1,000 tokens; L2048 that whole row) and the earlier release's loop-kernel files beside them; `r15_timing_shared.py` times the pairs per request with and without constant tensor sharing, and their GPU memory. 5 warm-up calls or requests, then 20 timed; a call is write the inputs + run + read back the whole hidden output.
11. Published files: `host_publish_fixtures.py --model 4b --only oracle_probs` writes `fixtures/oracle_probs.json` (the Kev-4B reference); `fixtures/requests_public.json` and `fixtures/tokenizer_probes.json` are the Kev-0.8B repository's files; `SHA256SUMS` lists every file of this repository.

## Gate

A file passes when, over the questions whose rows fit it, against the reference: the most likely option is the same for every question whose reference top-two probabilities are more than 0.02 apart (closer pairs are reported separately as near ties), the largest absolute difference of any option's probability is at most 0.02, and the mean absolute difference over all options is at most 0.002. A control request (`red_arm_000`, one word of `tv4_000`'s instructions changed) must differ from `tv4_000`'s reference by more than 0.02, which shows the tolerance can catch a one-word change; its row has 94 tokens, so the 64-token graph does not run it.

| File | Runtime | Questions | Same most likely option | Near ties kept | Max \|Δp\| | Mean \|Δp\| | Control |
|---|---|---:|---:|---:|---:|---:|---:|
| L64 | Mac GPU (Metal), float32 | 72 | 71/71 | 1/1 | 0.0070 | 3.6e-4 | — |
| L128 | Mac GPU (Metal), float32 | 321 | 312/312 | 9/9 | 0.0152 | 4.4e-4 | 0.0528 |
| L256 | Mac GPU (Metal), float32 | 385 | 376/376 | 9/9 | 0.0152 | 4.6e-4 | 0.0528 |
| L512 | Mac GPU (Metal), float32 | 392 | 383/383 | 9/9 | 0.0152 | 4.5e-4 | 0.0528 |
| L1024 | Mac CPU, 8 threads | 392 | 383/383 | 9/9 | 0.0152 | 4.5e-4 | 0.0528 |
| L1024 | Mac GPU (Metal), float32 | 392 | 383/383 | 9/9 | 0.0152 | 4.5e-4 | 0.0528 |
| L2048 | Mac GPU (Metal), float32 | 401 | 392/392 (long rows 9/9) | 9/9 | 0.0152 | 4.5e-4 | 0.0528 |
| Pair Ls128 | Mac CPU; Metal float32 with and without sharing | 313 | 308/308 | 5/5 | 0.0152 | 4.1e-4 | 0.0528 |
| Pair Ls256 | Mac CPU; Metal float32 with and without sharing | 340 | 335/335 | 5/5 | 0.0152 | 4.4e-4 | 0.0528 |
| L128 | Mac GPU (Metal), default precision (float16 activations) | 321 | 312/312 | 8/9 | 0.0456 | 3.8e-3 | 0.0561 |
| L512 | Mac GPU (Metal), default precision (float16 activations) | 392 | 383/383 | 6/9 | 0.0675 | 4.0e-3 | 0.0488 |

Every file passes at float32 on the CPU and on Metal, and no near tie changes its most likely option. At the GPU's default precision the hidden states stay finite on every row, but the largest and mean differences are outside the tolerance and 1 (L128) and 3 (L512) near ties change. The long rows are the 9 questions on the three long invented states (1,369 to 1,805 tokens), largest difference 0.0010. On the same file, the Metal GPU at float32 and the CPU agree within 1.1e-5 (the 1,024-token graph) and 1.2e-5 (pairs); the two state hand-overs give the same hidden states bit for bit, and Metal with and without constant tensor sharing gives the same probabilities and read-out hidden states bit for bit (314 questions on Ls 128, 341 on Ls 256). On Metal at float32, a pair and the 128- and 256-token row graphs agree within 6.3e-6 on the same questions.

## Phone

The Kev-4B files did not fit a 12 GB phone. Both runs on one Samsung Galaxy S26 (SM-S942Q, 12 GB, Android 16), with LiteRT 2.2.0 from Maven through the Kotlin `CompiledModel` API in the measurement activity of `android/measure/` on the GPU, ran out of memory while the graph was compiling, before the first call:

| Date (JST) | File | GPU precision | What happened | Memory (the host's 2 s samples; GB = kB × 1,024 / 10⁹) |
|---|---|---|---|---|
| 2026-10-03 23:33 | the earlier release's `kev-4b_rowprefill_L1024_fp16fc_i8emb.tflite` (7,799,218,560 bytes) | FP32 | The delegate took all 34,313 nodes (1 partition); 9 s later Android's low-memory killer stopped the app (`min2x watermark is breached even after kill`). | MemAvailable 7.34 GB at the start, 1.13 GB at the lowest; the app's VmHWM 5.71 GB, VmSwap 6.49 GB; the killer's line reported 1.78 GB resident and 7.16 GB in swap. |
| 2026-10-05 00:13 | `kev-4b_rowprefill_L64_fp16fc_i8emb.tflite` (7,785,929,712 bytes) | FP16_WITH_FP32_ACCUM | The delegate took all 5,307 nodes (1 partition); 6 s later the host stopped the app because MemAvailable had fallen below 1,500,000 kB. The low-memory killer was reclaiming other apps by then, not this one. | MemAvailable 6.79 GB at the start, 1.22 GB at the stop; VmHWM 5.95 GB, VmSwap 2.39 GB; the GPU's page allocation stayed at 0.29 GB. |

The CPU was not tried, and the phone did not reboot. The Kev-0.8B files run on the same phone (the Galaxy S26 section of Kev-0.8B-LiteRT's REPRODUCE.md).

## Where the numbers come from

All paths are in the work directory of `conversion/README.md`. `V4` = `v2_fp16fc_i8emb_r15R64-sp-ec-dd-vs6-in1-fn5`.

| Number | Produced by |
|---|---|
| Reference probabilities and answers | `oracle_kev.py --model 4b` (`oracle/oracle_4b.json`) |
| Merge agreement | `merge_check.py`, `merge_parity.py` (`results/merge_4b.json`, `results/merge_parity_4b.json`) |
| fp32 invariance of the final kernel | `r15_torch_parity.py` (`results/r15_torch_parity_4b_R64-sp-ec-dd-vs6-in1-fn5.json`) |
| The pair against the row form in PyTorch | `r15_shared_torch_parity.py` (`results/r15_shared_torch_parity_4b_R64-sp-ec-dd-vs6-in1-fn5.json`) |
| Float16 headroom of the norms | `r15_norm_range_probe.py` (`results/r15_norm_range_4b_R64-sp-ec-dd-vs6-in1-fn5.json`) |
| Operators of each file | `r15_export.py`, `r15_export_shared.py` (`results/export_L{L}_4b_r15R64-sp-ec-dd-vs6-in1-fn5.json`, `results/quant_L{L}_4b_$V4.json`, `results/export_sharedstate_Ls{Ls}_Lq64_4b_r15R64-sp-ec-dd-vs6-in1-fn5.json`, `results/quant_sharedstate_Ls{Ls}_Lq64_4b_$V4.json`) |
| Desktop CPU and Metal agreement of the row graphs | `litert_parity.py` (`results/litert_{cpu,gpu_f32,gpu_f16}_rows_L{L}_4b_$V4.json`; the statistics above are recomputed from these rows) |
| Desktop agreement of the pairs | `r15_shared_parity.py` (`results/litert_shared_{cpu,gpu_f32,gpu_f32_share}_rows_Ls{Ls}_Lq64_4b_$V4.json`); with and without sharing: `r15_share_compare.py` (`results/r15_share_vs_noshare_4b_r15R64-sp-ec-dd-vs6-in1-fn5.json`) |
| Metal against the CPU on the same file | `litert_parity.py` (`gpu_vs_cpu_same_file` in `results/litert_gpu_f32_parity_L1024_4b_$V4.json`), `r15_shared_parity.py` (`same_questions` in `results/litert_shared_gpu_f32_share_parity_Ls{Ls}_Lq64_4b_$V4.json`) |
| Python host agreement | `host_parity.py --model 4b --skip-graph`, `r16_host_parity.py --model 4b` (`results/host_parity_v2_4b_{row,pair}_*.json`) |
| Rebuild from `conversion/` | the commands of `conversion/README.md` in an empty work directory; SHA-256 against `SHA256SUMS` |
| Mac timing | `r15_timing_mac.py`, `r15_timing_shared.py` (`results/timing_mac_r15_4b_*.json`, `results/memory_mac_r15_4b_pair*.json`) |
| Phone runs | the measurement activity's logcat and the host's samples (`device/r6/kev_s26_r6_E_4b_gpu_fp32_L1024_timing.*` with `device/r6/kev_s26_r6_E_4b.kill_lines.txt`; `device/r14/kev_s26_r14_T4B_C_gpu_fp16acc32_L64_timing.*`) |
| SHA-256 of the published files | `SHA256SUMS` |
