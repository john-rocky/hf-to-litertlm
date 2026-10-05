# Kev-0.8B and Kev-4B (Jared Palmer) → LiteRT: row graphs and shared-state pairs for decision models that never generate text

Published: [litert-community/Kev-0.8B-LiteRT](https://huggingface.co/litert-community/Kev-0.8B-LiteRT) and [litert-community/Kev-4B-LiteRT](https://huggingface.co/litert-community/Kev-4B-LiteRT), release of 2026-10-05. Each repo holds eight raw LiteRT graphs (`.tflite`): six row graphs, for rows of up to 64, 128, 256, 512, 1,024 and 2,048 tokens, and two shared-state pairs, for states of up to 128 and 256 tokens with questions of up to 64 tokens. The 2026-10-04 release had three row graphs (512, 1,024 and 2,048 tokens); those three files keep their names with new contents. Each repo also holds the pointer head, the tokenizer, the Python host, the Kotlin snippet, the sources behind the phone numbers, the conversion scripts and the fixtures. This folder holds byte-identical copies of the scripts, fixtures, host, example, Kotlin snippet and phone sources, and of the two repos' reproduction guides, so the conversion and its checks can be rerun from here.

## What the models are

Kev-0.8B and Kev-4B are decision models by Jared Palmer (`jaredpalmer/kev-0.8b`, `jaredpalmer/kev-4b`, both tag `v1.0`, Apache-2.0). They read a text (the state) and typed questions about it, noul (yes/no), choice (multiple choice) and score (rating), and return an answer with probabilities per question. They never generate text. Requests and responses follow the `/v1/systemone` shape of the author's server ([github.com/jaredpalmer/kev](https://github.com/jaredpalmer/kev)). Each adapts a base model with a rank-16 LoRA on 12 projection types and adds a pointer head of two 256-dimensional linear layers that reads the hidden states. The bases are `Qwen/Qwen3.5-0.8B-Base` (revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`) and `Qwen/Qwen3.5-4B-Base` (revision `1001bb4d826a52d1f399e183466143f4da7b741b`), both Apache-2.0, on the Qwen3.5 hybrid backbone: the 0.8B has 24 layers (18 Gated DeltaNet, a linear-attention form, and 6 full attention) at hidden size 1024, the 4B has 32 layers (24 + 8) at hidden size 2560, with 16 key heads and 32 value heads in its Gated DeltaNet. Vocabulary 248,320.

The published graphs have no LM head: they return hidden states after the final RMSNorm, and the host applies the pointer head in float32. All eight files of a model carry the same weights. A row graph's signature `serving_default` takes `ids` int32 [1, L] (right-padded with 248044) and `valid` float32 [1, L] and returns `hidden` float32 [1, L, d], with d = 1024 (0.8B) or 2560 (4B). Delimiter ids: state 248060, question 248061, option start 248049, option end 248050, decide 248062; no BOS. One row is the state plus one question. Positions are the constants 0 to L−1, and the mask is a constant causal mask plus (1 − valid) × (−1e4). No KV or Gated DeltaNet state goes in or out of a row graph, so every row starts from zero and a request with Q questions takes Q calls. The model never decodes, so the graph needs no cache-update operators, and the forms that mobile GPUs reject in the stock exporters (masked_fill, in-place row updates with index_put, a rank-5 repeat_interleave on the 4B, rank-4 PAD) stay out of it. A graph computes all of its positions whatever the row length, so the host runs each row on the smallest row graph that holds it. The host reads the hidden states at the row's last real token (248062) and at each option's closing 248050, applies the pointer head with the author's temperature (2.3510958125672174 for the 0.8B, 2.406050072164233 for the 4B), and takes the softmax.

A shared-state pair computes the state once per request instead of once per question. It is one file with two signatures that share the weights. `state_prefill_<Ls>` takes `[248060]` and the state's tokens as `ids` and `valid` [1, Ls]. It returns the state of every layer (the recurrent state and conv tail of each Gated DeltaNet layer, the keys and values of each attention layer): 48 tensors on the 0.8B (23,347,200 bytes per request at Ls 128) and 64 on the 4B (61,079,552 bytes at Ls 128, 69,468,160 at Ls 256). `question_step_<Ls>_64` takes one question's branch (the row from 248061 on) as `ids` and `valid` [1, 64], the state's `valid` as `state_valid` [1, Ls] and the state tensors, and returns `hidden` float32 [1, 64, d]; the branch's positions follow the state. A state belongs to the file that made it: the kernel carries the recurrent state at 2⁶ times its stock value between the two signatures.

## What this folder holds

| File | Content |
|---|---|
| `REPRODUCE-Kev-0.8B.md`, `REPRODUCE-Kev-4B.md` | Byte copies of the two Hub repos' `REPRODUCE.md`: sources with sha256, environments, the 13 / 11 numbered steps, the pass criterion, the phone sections and where each number comes from |
| `conversion/` (79 files) | The conversion, check and timing scripts with their environment files, byte-identical in both Hub repos; `conversion/README.md` gives the commands in order, the kernel forms and the float16 headroom of the norms |
| `fixtures/requests_public.json` | 156 requests with text (SemIf 144 + 12 invented); the 221 transfer-v4 records by reference |
| `fixtures/rebuild_requests.py` | Restores the full 377-request file from the author's GitHub repo at tag `kev-1.0` (output sha256 `3bcc2566…`) |
| `fixtures/tokenizer_probes.json`, `fixtures/oracle_probs.json`, `fixtures/oracle_probs_4b.json` | The 12 tokenizer probes; the reference results of Kev-0.8B and of Kev-4B for all 402 questions |
| `fixtures/README.md`, `fixtures/LICENSE-SemIf-MIT.txt` | The guide to the fixtures (the Kev-0.8B repo's copy; the Kev-4B repo's names its own checkpoint in two sentences); the MIT license of the SemIf records |
| `host/kev_litert.py`, `host/requirements-host.txt` | The Python host for both models (tokenizers + numpy + safetensors + ai-edge-litert 2.2.0; no PyTorch, no kev package) and its requirements |
| `examples/run_example.py`, `examples/run_example.expected.json`, `examples/run_example.expected_4b.json` | The host example and its expected output for Kev-0.8B and for Kev-4B |
| `android/CardSnippet.kt` | The Kotlin CompiledModel calls: `KevRowGraph` for a row graph, `KevPairGraph` for a pair, the GPU at `FP16_WITH_FP32_ACCUM` by default |
| `android/measure/` | `GateActivity.kt` and `RowCodec.kt`, the measurement activity behind the Galaxy S26 GPU numbers; `kev_npu_runner.cc`, the NPU runner on the LiteRT 2.2.0 C API; `README.md` with the build and run commands and the settings behind each number. They are not a sample app. A sample app in a separate repository ([github.com/john-rocky/LiteRT-Models/tree/main/kev](https://github.com/john-rocky/LiteRT-Models/tree/main/kev)) runs the Kev-0.8B files on the GPU and, for rows of up to 256 tokens, on the NPU |
| `NOTICE-Kev-0.8B`, `NOTICE-Kev-4B` | Attribution (code adapted from the kev package and from litert-torch) |

## How to reproduce

The order is the same for both models. [`conversion/README.md`](conversion/README.md) gives the commands, the work-directory layout and the three Python environments; the Kev-4B commands take `--model 4b`. [`REPRODUCE-Kev-0.8B.md`](REPRODUCE-Kev-0.8B.md) (13 numbered steps) and [`REPRODUCE-Kev-4B.md`](REPRODUCE-Kev-4B.md) (11) add the sources with sha256, the environments, the pass criterion, the phone sections and where each number comes from.

The published graphs use the final kernel, named by a form: a `+`-joined list of changes that `conversion/README.md` explains one by one. Kev-0.8B uses form C7 (`R64+sp+ec+dd+vs6+bk1024@in_proj_z.q_proj`, which also runs on the phone's NPU) for its 64-, 128- and 256-token files and form C (`R64+sp+ec+dd+vs6`) for the other files and the pairs; on the CPU and on Metal at float32 the two forms give the same probabilities bit for bit. Kev-4B uses `R64+sp+ec+dd+vs6+in1+fn5`.

| Step | Scripts in `conversion/` |
|---|---|
| Fixtures | `make_fixtures.py`, `own_records.py` |
| Reference: the author's code | `oracle_kev.py` |
| Merge | the author's `scripts/merge_lora_checkpoint.py`, then `merge_check.py`, `merge_parity.py` |
| Unpatched baseline | `tf514_baseline.py` |
| Graph and kernel forms | `kev_qwen35_patch.py`, `kev_graph.py`, `r13_kernel.py`, `r15_form.py` (Kev-4B), `r17_kernel.py` (form C7), `r14_shared_state.py` and `r15_shared_state.py` (pairs) |
| Kernel checks in PyTorch | Kev-0.8B: `r14_torch_parity.py`, `r17_torch_parity.py`, `r14_shared_torch_parity.py`, `r14_norm_range_probe.py`; Kev-4B: `r15_kernel_test.py`, `r15_torch_parity.py`, `r15_shared_torch_parity.py`, `r15_norm_range_probe.py` |
| Export, operator scan, storage form | Kev-0.8B: `r17_export.py` (L64, L128, L256), `r13_export.py` (L512, L1024, L2048), `r14_export_shared.py` (pairs); Kev-4B: `r15_export.py`, `r15_export_shared.py` |
| Desktop check | `litert_parity.py`, `gpu_gate_subprocess.py`, `gpu_vs_cpu.py` (Kev-0.8B); pairs: `r14_litert_shared_parity.py` (Kev-0.8B), `r15_shared_parity.py` and `r15_share_compare.py` (Kev-4B) |
| Head file | `host_head_export.py`; `r18_head_json.py` (Kev-0.8B, with the NPU entry), `r16_head_json.py` (Kev-4B) |
| Tokenizer probes, host | `host_tokenizer_probes.py`, `host_parity.py`, `host_author_crosscheck.py`; on the published files, `r18_host_parity.py` and `r16_host_parity.py` |
| Mac timing | Kev-0.8B: `r14_timing_mac.py`, `r14_timing_mac_req.py`, `r18_timing_mac.py`, `r18_timing_mac_req.py`, `r14_memory_mac.py`; Kev-4B: `r15_timing_mac.py`, `r15_timing_shared.py` |
| Galaxy S26 GPU (Kev-0.8B) | `r14_device_rows.py`, `r14_device_rows_req.py`, `r18_req2.py`; the measurement activity of `android/measure/`; then `r12_device_compare.py`, `r18_burst.py`, `r18_crossover.py` |
| Galaxy S26 NPU (Kev-0.8B) | `r11_npu_fixtures.py`, `r17_npu_fixtures.py`; `android/measure/kev_npu_runner.cc`; then `r17_runner_report.py`, `r17_device_compare.py`, `r17_leg_caps.py` |
| Published fixtures | `host_publish_fixtures.py` |

The merge folds the LoRA in float32 (186 weights on the 0.8B, 248 on the 4B); read with the author's loader, the folded checkpoint gives bit-identical probabilities to the adapter checkpoint on all 402 questions. Rebuilt from `conversion/` alone in a new work directory, the Kev-0.8B L128 file and Ls128 pair and the Kev-4B L128 file came out with the same SHA-256 as the published files. `export_kev.py`, `torch_graph_parity.py`, `device_rows.py` and `device_compare.py` made and checked the 2026-10-04 files and are kept for reference.

Environments: Python 3.12.11 with the author's lock (`conversion/requirements-oracle.txt`: torch 2.8.0, transformers 5.17.0, peft 0.21.0) and the kev package at tag `kev-1.0` for the reference; Python 3.14.6 with `conversion/requirements-export.txt` (litert-torch 0.9.4, litert-converter 0.4.0, torch 2.13.0, transformers 5.14.1, ai-edge-litert 2.2.0, ai-edge-quantizer 0.9.0) for the export, the quantization and the desktop checks; Python 3.12.11 with `host/requirements-host.txt` (ai-edge-litert 2.2.0, numpy 2.5.2, tokenizers 0.23.2, safetensors 0.8.0) for the host. Run the reference with one torch thread: with the default thread count the depthwise convolution of the Gated DeltaNet layers falls to a slow per-channel path and a row takes many times longer. A Kev-4B fp32 export takes about 50 GB of memory at its peak; run one at a time. On the phone, the measurement activity used LiteRT 2.2.0 from Maven with Kotlin 2.2.21, Android Gradle Plugin 8.9.1 and compileSdk 35, and the NPU runner the LiteRT 2.2.0 C API with the Android NDK 29.0.13113456.

## How to run the published files

`host/kev_litert.py` here is the same file as in both Hub repos.

```bash
hf download litert-community/Kev-0.8B-LiteRT --local-dir Kev-0.8B-LiteRT
cd Kev-0.8B-LiteRT
pip install -r host/requirements-host.txt
python examples/run_example.py --check
```

`examples/run_example.py` sends an invented support ticket with three questions on the CPU and prints the response. `--check` compares it with `examples/run_example.expected.json`, and `--mode row` or `--mode pair` sends it through one route only. The example needs one model file, the Ls128 pair or the L256 graph: add `--include "*_L256_*" "head/*" "tokenizer/*" "host/*" "examples/*"` to the download to fetch only the L256 graph and the files it needs. For Kev-4B, whose graph files are 7.8 GB each, fetch everything but the graphs, then the L256 graph:

```bash
hf download litert-community/Kev-4B-LiteRT --exclude "*.tflite" --local-dir Kev-4B-LiteRT
hf download litert-community/Kev-4B-LiteRT kev-4b_rowprefill_L256_fp16fc_i8emb.tflite --local-dir Kev-4B-LiteRT
```

That repo has its own `examples/run_example.expected.json` (here `examples/run_example.expected_4b.json`). Both expected responses are unchanged from the 2026-10-04 release.

In your own code:

```python
import sys
sys.path.insert(0, "host")  # run from the repository root
from kev_litert import KevLiteRT

request = {"state": "Order #1182 arrived with a cracked screen.",
           "questions": {"refund": {"type": "noul", "instructions": "Should we offer a refund?"}}}

with KevLiteRT.from_dir(".") as kev:
    print(kev.decide(request))
```

`from_dir` finds the row graphs and pairs present in a downloaded repo, the head and the tokenizer, and compiles a file only when a request needs it; any one graph file works alone. A row that no file present holds raises `RowTooLong` instead of being cut, and a non-finite hidden state raises `NonFiniteOutput`. The default is the CPU with 4 threads (`threads=` changes the count). `accelerator="gpu"` runs the GPU at float32 precision (`GpuOptions(enforce_f32=True)`), which the desktop GPU needs (see Limits).

In `mode="auto"`, the default, the host takes the smallest pair whose Ls holds the state. For the n questions whose branches fit the pair's 64 tokens, it compares R, the sum of the lengths of the smallest row graphs that fit their rows, with P = Ls + n × 64. Those questions take the pair only when R > 1.5 × P (`pair_ratio`); every other question takes its row graph. Of the 377 test requests, `auto` sends 9 through a pair (3 on Ls128, 6 on Ls256) and 368 through the row graphs. `mode="row"` uses only the row graphs, and `mode="pair"` sends every question through a pair or raises `PairDoesNotFit`. On the GPU, `constant_tensor_sharing=True` (the default) keeps one copy of a pair's weights for its two signatures; `False` is faster and uses about twice the memory (see Measurements). `handover="direct"` (the default) passes the state step's output buffers to the question step as its inputs, and `handover="host"` reads them back and writes them; both give the same bits. On the command line, `--graph` takes one or more row graphs and pairs, `--accel gpu` selects the GPU at float32 precision, `--request -` reads the request from stdin, and `--mode`, `--pair-ratio`, `--handover` and `--no-constant-tensor-sharing` set the options above.

On Android, `android/CardSnippet.kt` holds the CompiledModel calls. `KevRowGraph` runs a row graph. `KevPairGraph` runs a pair: `runState` once per request, then `readout` for each question's branch, with the state's output buffers passed to `question_step` as its inputs. Both default to `FP16_WITH_FP32_ACCUM`, and `KevPairGraph` keeps constant tensor sharing on by default. The app tokenizes with the repo's `tokenizer.json` (no special tokens added, `<|name|>` rewritten to `<¦name¦>`), joins the pieces with the delimiter ids, and applies the pointer head in float32 to the returned vectors. With Kev-4B, pass `precision = CompiledModel.GpuOptions.Precision.FP32` to both classes and `layers = 32` to `KevPairGraph`; for Kev-4B the snippet shows the call shape only, since Kev-4B has not run on a phone. The snippet compiles with Kotlin 2.2.21, Android Gradle Plugin 8.9.1 and compileSdk 35; the phone numbers come from the measurement activity in `android/measure/`.

The NPU numbers come from `android/measure/kev_npu_runner.cc` on the LiteRT 2.2.0 C API; `android/measure/README.md` gives the build and run commands. The runner needs LiteRT 2.2.0's NPU runtime libraries (the Qualcomm compiler plugin and dispatch library) and the HTP libraries of QAIRT 2.47, which neither this folder nor the repos include.

## Measurements

### How agreement is measured

Agreement is measured on post-temperature probabilities against the author's fp32 code at tag `kev-1.0` on the CPU, with the author's lock file (torch 2.8.0, transformers 5.17.0, peft 0.21.0). The test set has 377 requests with 402 questions: 220 from the author's `evals/v4/transfer-v4/development.jsonl`, SemIf authored144 (MIT) mapped to 3-option choice, 12 invented records with 37 questions, and 1 control. The rows of 392 questions have at most 366 tokens. The other 9 questions sit on 3 long invented states, with rows of 1,369 to 1,805 tokens; only the L2048 files take them. Each file runs the questions whose rows fit it.

A near tie is a question whose two most likely options in the reference are 0.02 or less apart; "377/377 + 13/15" means the reference's most likely option on all 377 questions outside the near ties and on 13 of the 15 near ties. The tolerance, set before the runs: the same most likely option on every non-near-tie question, max |Δp| ≤ 0.02, mean ≤ 0.002. The control is the request of `tv4_000` with one word of its instructions changed ("correctly" to "incorrectly"); a run detects the change when the control's max |Δp| from `tv4_000`'s reference is above 0.02. Its row has 94 tokens, so the L64 files do not run it.

### Kev-0.8B: agreement

Each cell gives the same most likely option + near ties kept, then max |Δp| and mean |Δp|. The pairs ran with and without constant tensor sharing, which gave the same probabilities bit for bit.

| File (questions) | Mac CPU (8 threads) and Metal, float32 | Galaxy S26 GPU, FP16_WITH_FP32_ACCUM | Galaxy S26 NPU | Control: Mac / S26 GPU / NPU |
|---|---|---|---|---|
| L64 (72) | 71/71 + 1/1, 0.0078, 8.8e-4 | 71/71 + 1/1, 0.0076, 1.01e-3 | 71/71 + 1/1, 0.0105, 1.18e-3 | not run |
| L128 (321) | 311/311 + 9/10, 0.0104, 9.7e-4 | 311/311 + 9/10, 0.0077, 1.08e-3 | 311/311 + 10/10, 0.0134, 1.29e-3 | 0.0477 / 0.0462 / 0.0476 |
| L256 (385) | 371/371 + 12/14, 0.0104, 9.1e-4 | 371/371 + 13/14, 0.0092, 1.04e-3 | 371/371 + 13/14, 0.0134, 1.24e-3 | 0.0477 / 0.0472 / 0.0476 |
| L512 (392) | 377/377 + 13/15, 0.0104, 9.1e-4 | 377/377 + 13/15, 0.0091, 1.06e-3 | not run | 0.0477 / 0.0473 / — |
| L1024 (392) | 377/377 + 13/15, 0.0104, 9.1e-4 | 377/377 + 13/15, 0.0091, 1.06e-3 | not run | 0.0477 / 0.0473 / — |
| L2048 (401) | 386/386 + 13/15, 0.0104, 8.9e-4 | 386/386 + 13/15, 0.0093, 1.08e-3 | not run | 0.0477 / 0.0473 / — |
| Pair Ls128 (313) | 306/306 + 6/7, 0.0078, 9.5e-4 | 306/306 + 6/7, 0.0101, 1.10e-3 | not run | 0.0477 / 0.0467 / — |
| Pair Ls256 (340) | 329/329 + 9/11, 0.0078, 9.1e-4 | 329/329 + 9/11, 0.0110, 1.08e-3 | not run | 0.0477 / 0.0467 / — |

The CPU and Metal give the same statistics on each line; on the same file they agree within 6.9e-6 (row graphs) and 6.4e-6 (pairs). A pair and the row route, with each question on its smallest row graph, differ by at most 3.7e-6 on the CPU and 4.4e-6 on Metal. At float32 the near ties that changed are the same two questions on every file that holds their rows, `tv4_023` (reference top-two gap 0.0020) and `own_sensor_08` (8.1e-5). The 2026-10-04 files gave the same numbers, because the int8 embedding table sets the difference from the reference. The 9 long rows on L2048 give max |Δp| 0.0015 on the Mac. On the S26 GPU they give 0.0015 (mean 4.3e-4) at FP32 and 0.0093 (mean 1.84e-3) at FP16_WITH_FP32_ACCUM, and L128 at FP32 gives the Mac's numbers (311/311 + 9/10, 0.0104, 9.7e-4, control 0.0477). At the GPU's default precision (float16 activations) on Metal, L128 gives 311/311 + 9/10, 0.0332, 3.6e-3 and L2048 385/386 + 11/15, 0.0352, 4.1e-3: finite on every row, but outside the tolerance.

### Kev-0.8B: Mac speed

Apple M4 Max (128 GB, macOS 27.0), ai-edge-litert 2.2.0, Python CompiledModel API, measured while no other GPU job ran. A call runs one question's row. A time is the wall clock of writing the inputs, running and reading the whole hidden output back: the median of 20 calls after 5 warm-up calls. A range spans two passes.

| Graph (row tokens) | Metal, float32 | CPU, 8 threads |
|---|---:|---:|
| L64 (64) | 24.8 to 24.9 ms | 69.0 to 69.1 ms |
| L128 (80) | 34.0 to 34.1 ms | 109.4 to 109.5 ms |
| L256 (128 to 142; one call of a 5-question request) | 54.4 ms | 187.9 to 188.2 ms |
| L512 (300) | 96.7 to 97.0 ms | 358.8 to 388.3 ms |
| L1024 (1,000) | 187.9 to 188.2 ms | 789.9 to 798.5 ms |
| L2048 (1,805) | 398.7 to 398.8 ms | 1,634 to 1,660 ms |

Whole requests on Metal at float32: through the row graphs, the sum of the calls, each row on the smallest graph that holds it; through a pair, the state once, then one step per question, with the state handed over directly.

| Request | Pair | Row graphs | Pair, sharing | Pair, no sharing |
|---|---|---:|---:|---:|
| 1 question, state 30 tokens (one row of 94 tokens: L128) | Ls128 | 33.9 ms | 75.3 to 75.7 ms | 62.3 to 62.5 ms |
| 2 questions, state 109 tokens (L256 × 2) | Ls128 | 108.5 ms | 106.3 to 106.8 ms | 87.5 to 87.8 ms |
| 3 questions, state 109 tokens (L256 × 3) | Ls128 | 162.7 to 162.9 ms | 137.2 to 137.9 ms | 113.0 to 113.3 ms |
| 5 questions, state 99 tokens (L256 × 3, L128 × 2) | Ls128 | 230.0 to 231.8 ms | 199.4 to 200.1 ms | 163.6 to 164.0 ms |
| 2 questions, state 150 tokens (L256 × 2) | Ls256 | 108.0 to 108.8 ms | 130.8 ms | 110.7 to 110.9 ms |
| 3 questions, state 167 tokens (L256 × 3) | Ls256 | 162.7 to 162.8 ms | 162.1 to 162.3 ms | 136.3 to 136.4 ms |

On the Ls128 pair without sharing, `state_prefill` took 36.8 to 37.0 ms and `question_step` 25.3 to 25.4 ms per question; with sharing, 44.2 to 44.5 ms and 30.9 to 31.1 ms. On the CPU with 8 threads, the Ls128 pair took 194.7 to 195.4, 340.5 to 341.8 and 484.4 to 487.4 ms for the requests of 1, 3 and 5 questions. On Metal, with a new process per run, the process footprint with the Ls128 pair after one request was 6.3 GB without sharing and 3.0 GB with it, and its peak during the compile 9.3 to 9.5 GB and 3.3 GB.

### Kev-0.8B: Galaxy S26 GPU

One Galaxy S26 (SM-S942Q, Snapdragon SM8850, Android 16, 12 GB) with LiteRT 2.2.0 through the Kotlin CompiledModel API on the GPU (OpenCL), in the measurement activity of `android/measure/`. The phone saved the read-out hidden states, and the Mac scored them against the reference with the same tolerance. Every file ran whole on the delegate in one partition: 3,912, 4,759, 6,019, 8,515, 13,555 and 23,635 nodes for L64 to L2048, and for the pairs 4,975 (Ls128) or 6,235 (Ls256) in `state_prefill` and 3,965 in `question_step`. In the agreement runs at FP16_WITH_FP32_ACCUM, the compile took 5.8, 6.1, 11.0, 6.7, 8.1 and 14.3 s for L64 to L2048; for the pairs (Ls128, Ls256) it took 9.9 and 10.5 s with sharing and 14.6 and 16.3 s without.

A time is write + run + `readFloat`: on the GPU, `run()` returns at once and the work finishes when the output is read. The phone lowers its clock caps under load, and a compile alone can start it. In a run on a 128-token graph at FP32, the 7.6 s compile took the GPU from 41 to 60 °C; the 17 calls made while the GPU cap was lowered took 143.4 to 161.5 ms, and the 8 made after it was lifted took 134.6 to 137.3 ms. So each timing run waits after the compile until the GPU is back to its temperature from before the compile, and every call is matched with the phone's state, read every 2 seconds. Cool is the median of the calls made while neither the GPU's clock cap (1,300 MHz) nor any CPU cap was lowered; sustained is the median of the calls made after a cap was lowered.

| Graph (row tokens) | FP16_WITH_FP32_ACCUM, cool (calls) | FP16_WITH_FP32_ACCUM, sustained (calls) |
|---|---:|---:|
| L64 (64) | 57.4 ms (20) | cap not lowered |
| L128 (80) | 102.4 ms (20) | cap not lowered |
| L256 (128 to 142; one call of a 5-question request) | 199.4 ms (32); the request: 992.9 ms (7 requests) | 220.5 ms (68); the request: 1,088.3 ms (13 requests) |
| L512 (300) | 387.0 ms (9) | 393.4 ms (11) |
| L1024 (1,000) | 819.8 ms (7) | 847.7 ms (13) |
| L2048 (1,805) | 1,808.2 ms (2) | 2,147.6 ms (18) |

At FP32, the cool medians were 85.3 ms on L64 (20 calls), 137.4 ms on L128 (18), 270.2 ms on L256 (30; the 5-question request 1,355.6 ms, 6 requests), 565.1 ms on L512 (16), 1,219.0 ms on L1024 (3) and 3,119.6 ms on L2048 (2).

The pair requests at FP16_WITH_FP32_ACCUM were timed as on the Mac, with 2 warm-up and 6 timed requests per line. A cell is the median of the requests timed while the GPU cap was not lowered, with their number in parentheses. The CPU clock was capped during the pair requests without sharing of 3 and 5 questions; the other cells ran with neither clock capped.

| Request | Pair | Pair, sharing | Pair, no sharing |
|---|---|---:|---:|
| 1 question, state 30 tokens | Ls128 | 249.1 ms (6) | 179.3 ms (6) |
| 2 questions, state 109 tokens | Ls128 | 341.8 ms (2) | 242.1 ms (6) |
| 3 questions, state 109 tokens | Ls128 | 441.5 ms (5) | 304.4 ms (4) |
| 5 questions, state 99 tokens | Ls128 | 624.8 ms (2) | 430.9 ms (6) |
| 2 questions, state 150 tokens | Ls256 | 445.2 ms (5) | 345.3 ms (6) |
| 3 questions, state 167 tokens | Ls256 | 538.4 ms (2) | 407.0 ms (6) |

Memory during these runs (the largest or smallest value of a run; GB = the `/proc` kB × 1,024 / 10⁹):

| Run | Process VmHWM | GPU (kgsl) | Smallest MemAvailable |
|---|---:|---:|---:|
| Pair with sharing (4 runs) | 3.1 to 3.2 GB | 1.7 to 1.8 GB | 5.6 to 6.1 GB |
| Pair without sharing (4 runs, no low-memory kill) | 6.6 to 7.0 GB | 3.2 GB | 2.7 to 3.1 GB |
| One row graph, FP16_WITH_FP32_ACCUM | 4.6 to 5.7 GB (L64 to L1024); 5.4 to 6.1 GB (L2048) | 1.6 to 1.8 GB (L64 to L1024); 2.1 to 2.2 GB (L2048) | 3.7 to 4.3 GB (L64 to L1024); 2.4 to 2.9 GB (L2048) |
| One row graph, FP32 | 5.5 to 6.7 GB | 2.6 to 2.9 GB (L64 to L1024); 3.4 GB (L2048) | 2.6 to 3.4 GB |

Many requests in a row lower the GPU and CPU clocks: in a run where the GPU cap fell to 578 MHz, the 5-question request on the pair with sharing took 1,081.0 ms instead of 624.8 ms. Keep the app visible with the screen on; when the screen locks, the computation stops.

### Kev-0.8B: Galaxy S26 NPU

The 64-, 128- and 256-token files also ran on the phone's NPU (the Qualcomm HTP), through `android/measure/kev_npu_runner.cc` on the LiteRT 2.2.0 C API with the accelerators NPU and CPU. The Qualcomm compiler takes every operator but the int8 embedding lookup, which runs on the CPU: 4,758 of the 4,759 operators of L128 (3,911 of 3,912 for L64, 6,018 of 6,019 for L256). The dispatch delegate then takes 2 of the 3 nodes. With the NPU alone as the accelerator, LiteRT 2.2.0 fails to compile the graphs. The L512 and longer files and the pairs were not run on the NPU.

These three files carry 24 extra SUM operators over an axis of size 1, one after each in_proj_z and q_proj linear. They sit between the FULLY_CONNECTED operators of two kinds of gate, the Gated DeltaNet output norm's gate and the attention output gate, and the sigmoids that read them. On the HTP, a sigmoid that reads a FULLY_CONNECTED output directly was off by about 2.3e-3 (absolute, root mean square), and graphs without the SUMs moved the probabilities by up to 0.085. The SUMs change no value: on the Mac CPU and GPU and on the S26 GPU, the files give the same bits as the graphs without them.

Each timing run loaded the file from its cache. A time is write + run + read, timed in the runner.

| File (timed row) | NPU, same row, 20 calls (median) | NPU, 20 different rows, once each (median) | GPU at FP16_WITH_FP32_ACCUM, same runner and row, 20 calls (median) |
|---|---:|---:|---:|
| L64 (`tv4x_emotion_00`, 42 tokens) | 44.0 ms | 44.1 ms | 56.9 ms |
| L128 (`tv4_007`, 80 tokens) | 67.8 ms; 71.1 ms (40 calls, another process) | 69.9 ms; 72.4 ms (the other process) | 101.9 ms; 102.3 ms (40 calls) |
| L256 (`tv4_006`, 237 tokens) | 126.7 ms | 126.4 ms | 194.9 ms |

NPU times move by about 5% from process to process (L128: 67.8 and 71.1 ms). The GPU times for L64 and L256 come from graphs with the same weights but without the 24 SUMs; on L128 the published file took 101.9 ms and the graph without the SUMs 101.1 ms. In the host's samples during these runs the GPU clock was never capped, and in the 128- and 256-token runs the cores of `cpufreq/policy6` were capped at 4.26 to 4.65 GHz of 4.74 GHz. The samples do not include the NPU's clock.

The initial load of a file compiles it on the phone (JIT): 55.7 s for L64, 191.2 s for L128 and 263.3 s for L256, with a process VmHWM of 5.5 to 6.2 GB and a smallest MemAvailable of 3.3 to 4.2 GB meanwhile. Each file leaves a cache of 1.27 to 1.29 GB. Later loads read it in 1.5 to 1.7 s, with a VmHWM of 2.3 to 2.4 GB.

A sample app (see What this folder holds) also ran the published L64, L128 and L256 files through the Kotlin CompiledModel API, with the accelerators NPU and CPU. It used the 181 questions whose text the repository carries (the 144 SemIf questions and the 37 invented ones). Each file stayed within the tolerance on the rows it holds: max |Δp| 0.0105 on L64 (34 questions), 0.0134 on L128 (147) and 0.0134 on L256 (172).

In the app, one call took 45.1 ms on L64, 65.8 ms on L128 (68.5 ms in another process) and 121.9 ms on L256. These times are medians of 60 calls after 5 warm-up calls, with each graph loaded from its cache. A time is write + run + read.

Inside the app, the initial compile took 81.6 s for L64, 179.0 s for L128 and 298.2 s for L256, one file per process. Later loads read the cache in 0.9 to 1.5 s. Not measured: other phones or SoCs, and files compiled ahead of time (on small graphs, AOT gave the same bits as the JIT).

### Kev-0.8B: against the 2026-10-04 files

The 2026-10-04 values are that release's, on the same devices and rows: the median of 20 calls after 5 warm-up calls, with sustained load included on the S26. The NPU time is the L256 graph's, which computes all 256 positions at any row length.

| Row | 2026-10-04 | 2026-10-05 |
|---|---|---|
| S26, about 130 tokens (one call of the 5-question request) | L512 at FP32: 630 ms | L256 at FP16_WITH_FP32_ACCUM: 199.4 ms cool, 220.5 ms sustained; at FP32: 270.2 ms cool; on the NPU: 126.7 ms |
| S26, the 5-question request | L512 × 5 at FP32: 3,150 ms | Ls128 pair at FP16_WITH_FP32_ACCUM: 624.8 ms with sharing, 430.9 ms without; row graphs: 800.0 ms |
| S26, 300 tokens | L512 at FP32: 663 ms | L512 at FP16_WITH_FP32_ACCUM: 387.0 ms cool, 393.4 ms sustained; at FP32: 565.1 ms cool, 563.1 ms sustained |
| S26, 1,000 tokens | L1024 at FP32: 1,345 ms | L1024 at FP16_WITH_FP32_ACCUM: 819.8 ms cool, 847.7 ms sustained; at FP32: 1,219.0 ms cool, 1,316.8 ms sustained |
| S26, 1,805 tokens | L2048 at FP32: 3,331 ms (one call at any row length, 25 calls) | L2048 at FP16_WITH_FP32_ACCUM: 1,808.2 ms cool, 2,147.6 ms sustained; at FP32: 3,119.6 ms cool, 3,121.2 ms sustained |
| Mac (Metal, float32), about 130 tokens | L512: 139.9 ms | L256: 54.4 ms |
| Mac, the 5-question request | L512 × 5: 699.9 ms | Ls128 pair: 163.6 to 164.0 ms without sharing, 199.4 to 200.1 ms with; row graphs: 230.0 to 231.8 ms |
| Mac, 1,805 tokens | L2048: 474.2 ms | L2048: 398.7 to 398.8 ms |

### Kev-4B

The same Mac and timing method as for Kev-0.8B. Every file ran on Metal at float32 on every test question whose row fits it. For this model, 9 of the 401 test questions are near ties.

| File (questions) | Metal, float32: same most likely option + near ties kept, max \|Δp\|, mean \|Δp\| | Control |
|---|---|---:|
| L64 (72) | 71/71 + 1/1, 0.0070, 3.6e-4 | not run |
| L128 (321) | 312/312 + 9/9, 0.0152, 4.4e-4 | 0.0528 |
| L256 (385) | 376/376 + 9/9, 0.0152, 4.6e-4 | 0.0528 |
| L512 (392) | 383/383 + 9/9, 0.0152, 4.5e-4 | 0.0528 |
| L1024 (392) | 383/383 + 9/9, 0.0152, 4.5e-4 | 0.0528 |
| L2048 (401) | 392/392 + 9/9, 0.0152, 4.5e-4 (long rows 9/9, max 0.0010) | 0.0528 |
| Pair Ls128 (313) | 308/308 + 5/5, 0.0152, 4.1e-4, with and without sharing | 0.0528 |
| Pair Ls256 (340) | 335/335 + 5/5, 0.0152, 4.4e-4, with and without sharing | 0.0528 |

No question changed its answer on any of these runs, near ties included. The numbers equal those of the earlier release's files, and on the same questions the GPU results of the two releases differ by at most 8.8e-6. The Mac CPU (8 threads) gave the same numbers on L1024 and both pairs, within 1.1e-5 (L1024) and 1.2e-5 (pairs) of Metal on the same file. Metal with and without sharing gives the same probabilities and read-out hidden states bit for bit. On Metal, `is_fully_accelerated` is true for all eight files. At Metal's default precision (float16 activations), L128 gave 312/312 + 8/9, 0.0456, 3.8e-3 (control 0.0561) and L512 383/383 + 6/9, 0.0675, 4.0e-3 (control 0.0488): finite on every row, but outside the tolerance.

| Row | File | Metal, float32 | CPU, 8 threads |
|---|---|---:|---:|
| 64 tokens | L64 | 70.8 ms | 228.7 ms |
| 80 tokens | L128 | 118.5 ms | 450.0 ms |
| 128 to 142 tokens (one question of a 5-question request) | L256 | 209.8 ms | 815.3 ms |
| The 5-question request (sum of 5 calls, all on L256) | L256 | 1,048.8 ms | 4,083.5 ms |
| 300 tokens | L512 | 396.0 to 396.5 ms (two runs) | 1,497.0 ms |
| 1,000 tokens | L1024 | 801.0 ms | 2,928.4 ms |
| 1,805 tokens | L2048 | 1,705.4 ms | 5,934.9 ms |

Requests, timed as for Kev-0.8B (GPU = Metal at float32, CPU = 8 threads):

| Request | Row graphs, GPU | Pair, GPU, sharing | Pair, GPU, no sharing | Row graphs, CPU | Pair, CPU |
|---|---:|---:|---:|---:|---:|
| 1 question; state 30 tokens; row 94 tokens (L128); Ls128 pair | 117.4 ms | 331.2 ms | 201.1 ms | 444.8 ms | 755.5 ms |
| 2 questions; state 109 tokens; rows on L256; Ls128 pair | 419.5 ms | 465.9 ms | 274.0 ms | 1,629.2 ms | 1,019.0 ms |
| 3 questions; state 109 tokens; rows on L256; Ls128 pair | 629.2 ms | 600.6 ms | 347.2 ms | 2,444.0 ms | 1,287.1 ms |
| 5 questions; state 99 tokens; 3 rows on L256, 2 on L128; Ls128 pair | 864.1 ms | 871.8 ms | 492.6 ms | 3,333.0 ms | 1,814.2 ms |
| 2 questions; state 150 tokens; Ls256 pair | 419.5 ms | 571.4 ms | 376.6 ms | 1,624.4 ms | 1,410.8 ms |
| 3 questions; state 167 tokens; Ls256 pair | 628.8 ms | 705.6 ms | 449.9 ms | 2,438.1 ms | 1,677.0 ms |

On the GPU, a pair request splits into one state step and one question step per question: 128.2 ms + 72.9 ms per question on Ls128 without sharing and 196.8 ms + 134.4 ms with it; 228.5 ms + 73.9 ms and 298.7 ms + 136.0 ms on Ls256. With sharing, the pairs on the GPU took about as long as the row graphs, or longer. Without it, they took less time than the row graphs on every request of two or more questions. On the CPU, a pair took less time on every request of two or more questions, and more on the 1-question request.

With one row graph, the process footprint was 17.0 to 20.8 GB after compile with a peak of 37.0 to 38.1 GB on the GPU, and 14.7 to 16.4 GB on the CPU. With a pair on the GPU it was 15.9 GB after compile and 17.2 GB at peak with sharing, and 36.9 GB and 54.9 GB without; on the CPU, 31 GB. The GPU compile took 29 to 49 s for a row graph, and 13 to 22 s for a pair with sharing and 62 to 74 s without.

Against the 2026-10-04 files, timed again in the same timing run on the same Mac: 300 tokens on L512 took 464.4 to 464.7 ms on the GPU (two runs) and 1,967.5 ms on the CPU, against 396.0 to 396.5 ms and 1,497.0 ms now; 1,805 tokens on L2048 took 1,824.2 ms on the GPU, against 1,705.4 ms. The earlier release had no graph below L512; an 80-token row now runs on L128 in 118.5 ms. The 2026-10-04 card gave 2,320.3 ms on the GPU for the 5-question request (five calls on L512, measured in an earlier run); it now takes 864.1 ms through the row graphs, 492.6 ms through the Ls128 pair without sharing and 871.8 ms with sharing.

Kev-4B did not fit one Galaxy S26 (12 GB). Both tries compiled one graph for the GPU and ran out of memory during the compile, before any call ran. The CPU was not tried, and the phone did not reboot.

- 2026-10-03, the earlier release's L1024 file (7,799,218,560 bytes), FP32: the delegate took all 34,313 nodes, and 9 s later the low-memory killer stopped the process. It had peaked at 5.7 GB of resident memory (VmHWM), and the killer's log line reported 7.2 GB of its memory in swap.
- 2026-10-05, this release's L64 file (7,785,929,712 bytes), FP16_WITH_FP32_ACCUM, with the phone's memory watched: the delegate took all 5,307 nodes in one partition. 6 s later MemAvailable had fallen from 6.8 GB to 1.2 GB, the process had peaked at 5.9 GB (VmHWM), and the GPU's allocation was still 0.3 GB. The host then stopped the app; by that time the low-memory killer had started reclaiming other apps.

Not measured for Kev-4B: phones with more than 12 GB, the NPU, and GPUs on Linux or Windows.

## Limits

- The model never generates text. A row holds at most 2,048 tokens: the state plus one question. A pair holds a state of up to 256 tokens and question branches of up to 64 tokens; a question that does not fit takes a row graph. Longer states are not handled; the author's server accepts states of up to 65,536 tokens.
- A choice question takes at most 255 options.
- The GPU's default precision (float16 activations) is outside the tolerance on both models. Use float32 on the desktop. On Android, use `FP16_WITH_FP32_ACCUM` or `FP32` with Kev-0.8B; rows longer than 1,024 tokens stay closer to the reference at `FP32` (0.0015 and 4.3e-4 on the 9 long rows, against 0.0093 and 1.84e-3). Kev-4B passes at float32; it has not been checked at `FP16_WITH_FP32_ACCUM`.
- Probabilities differ from the reference by up to 0.0104 (Kev-0.8B on the desktop at float32), 0.0110 (Kev-0.8B on the S26 GPU), 0.0134 (Kev-0.8B on the S26 NPU) and 0.0152 (Kev-4B at float32). In these runs, no question outside the near ties changed its answer. A question whose two most likely options are 0.002 or less apart can change its answer: on the desktop, 2 of the 401 Kev-0.8B questions did.
- The NPU runs only the Kev-0.8B L64, L128 and L256 files. Its initial compile on the phone takes 55.7 to 263.3 s, and its cache takes 1.27 to 1.29 GB per file. It was measured on one Galaxy S26 only; inside a sample app, the initial compile took 81.6 to 298.2 s.
- A pair without constant tensor sharing uses about twice the memory: on Kev-0.8B, a process footprint of 6.3 GB against 3.0 GB on the Mac after one request, and a VmHWM of 6.6 to 7.0 GB against 3.1 to 3.2 GB on the S26.
- Continuous load slows the phone; compare the cool and sustained columns.
- Kev-4B needs desktop-class memory: on the Mac GPU, 17.0 to 20.8 GB after compile with one row graph and a peak of 37.0 to 38.1 GB, and 36.9 GB with a peak of 54.9 GB with a pair without sharing. It did not fit the 12 GB Galaxy S26 in two tries.
- Agreement is to the author's fp32 code, not task accuracy: accuracy and calibration were not re-measured, and the temperature is the author's fitted value, unchanged.
- Measured on one Mac (Apple M4 Max) and one Galaxy S26.

## License and attribution

- The models `jaredpalmer/kev-0.8b` and `jaredpalmer/kev-4b` and their bases `Qwen/Qwen3.5-0.8B-Base` and `Qwen/Qwen3.5-4B-Base` are Apache-2.0; the published LiteRT files are converted from them.
- The scripts and sources in this folder (conversion, host, example, Kotlin snippet, measurement activity and NPU runner) are Apache-2.0.
- The SemIf records in `fixtures/requests_public.json` are MIT (`fixtures/LICENSE-SemIf-MIT.txt`). The transfer-v4 records are listed by reference only, because their source datasets carry different licenses (tweet_eval unknown, SciQ CC BY-NC 3.0); `fixtures/rebuild_requests.py` restores them from the author's repo.
- `NOTICE-Kev-0.8B` and `NOTICE-Kev-4B` give the attribution, including the code adapted from the kev package and from litert-torch.
