# MNN-Pocket-0.3B-Dec (taobao-mnn) → LiteRT: raw graphs for a decision model that never generates text

Published: [litert-community/MNN-Pocket-0.3B-Dec-LiteRT](https://huggingface.co/litert-community/MNN-Pocket-0.3B-Dec-LiteRT), first release 2026-11-07, updated twice on 2026-11-08. This folder follows commit `92110e853b6b7343a776a2872a3b0888d4ffbfc1` (the second update, 62 files, 14.05 GB). The repository holds 28 raw LiteRT graphs (`.tflite`) in four sets: for the CPU and the GPU, for a Qualcomm NPU that compiles them on the phone, the same graphs compiled ahead of time for the Qualcomm SM8850, and for the Google Tensor NPU. It also holds the source model's token table, head, temperature and tokenizer, the Python host, the examples, the conversion scripts and the model card. This folder holds byte-identical copies of the repository's scripts, host, examples, reproduction guide, `NOTICE` and `SHA256SUMS`, so the conversion and its checks can be rerun from here. The graphs and the source model's files are on the Hub only.

## What the model is

[`taobao-mnn/MNN-Pocket-0.3B-Dec`](https://huggingface.co/taobao-mnn/MNN-Pocket-0.3B-Dec) (revision `dfbe0aed3abd4caed175b883a4b462ed35184fad`, Apache-2.0) is a decision model built on `taobao-mnn/MNN-Pocket-0.3B`. It takes a `state` and typed `questions` (`choice`, `noul`, `score`) and returns the probability of every candidate in one forward pass. It never generates text. It adapts the Qwen3 backbone (28 layers, hidden size 896) with a rank-16 LoRA (`lora_alpha` 32) and adds a pointer head that reads the hidden states.

Each graph is the decoder with the LoRA folded in float32 and its FULLY_CONNECTED weights stored as float16. It takes `embeds` f32 [1, L, 896], `cos` and `sin` f32 [1, L, 64] and `mask` f32 [1, 1, L, L], and returns `hidden` f32 [1, L, 896], the output of the final RMSNorm at every position. The host (`host/pocket_dec_litert.py`) does the rest as the source's `inference.py` does: it encodes the request, writes the rows of the bfloat16 embedding table, the rotary tables of the source's positions (each question branch restarts at the state length) and the additive mask, runs the smallest graph that holds the request (L = 64, 96, 128, 256, 512, 768, 1,024 or 2,048), and applies the pointer head and the temperature in float32. A request longer than 2,048 tokens raises `ValueError`.

The three forms are explained in `conversion/README.md` ("The forms"): `k4` for the `_fp16w` files (CPU and GPU), `k4_exp_silu` for the `_fp16w_npu` files (SiLU without LOGISTIC, for the Qualcomm NPU), and `mean_norm` for the `_fp16w_tensor` files (each RMSNorm computed on x / (4 mean|x|), for the Google Tensor NPU, at 64 to 256 tokens). In float32 each form gives the source's winner on all 2,409 questions of `REPRODUCE-MNN-Pocket-0.3B-Dec.md`.

## What this folder holds

Every file but this README is the Hub file at the commit above, byte for byte.

| File | Bytes | SHA-256 | Content |
|---|---:|---|---|
| `conversion/README.md` | 9,883 | `eba99734f8f8ad57e9063b4b6a77bb26aed08adf42c8e067efd07982a2dbbd8c` | The commands in order, the SM8850 compile and the three forms |
| `conversion/bundle_files.py` | 2,364 | `ad3f31105b97366272c1523e378b0ec594d552a34c55be679ce7cfa0d413efb2` | Step 1: copies the source files the host reads and writes `embedding.safetensors` |
| `conversion/reference_examples.py` | 2,443 | `c6f60dbf14a7aa939131170f2e3b9984418ab27f35fc109add7defb204547154` | Step 2: the source's `inference.py` on the example requests |
| `conversion/merge_lora.py` | 3,153 | `651356bfb766819d871108be14dba898c350686f8188483fdc3674d51473ee30` | Step 3: folds the LoRA adapter into the backbone in float32 |
| `conversion/dec_graph.py` | 9,506 | `acb67b4364873fae60969cc567086b906fa79b5d1ff9e3a6cf0294955fe04e49` | The decoder in plain PyTorch, in the forms `k4`, `k4_exp_silu` and `mean_norm` |
| `conversion/export_dec.py` | 6,225 | `66ae21a556785d208668a42a926ca52f3c5aa309b5fbc08aa045a6209b943bc1` | Step 4: the export at one length, after a check of the module on the example requests |
| `conversion/tflite_scan.py` | 9,072 | `0c07a7864a681d7384da3172e7f588558866369ff3aefd3d8ca930a67a402d59` | The static scan of an exported graph (signatures, operators) |
| `conversion/quantize_dec.py` | 3,397 | `fca8ec07f45b02d1ab9e9fb29c947f05e5fa8995883c3cf2ba29895d17786c24` | Step 4: the FULLY_CONNECTED weights stored as float16 |
| `conversion/quant_scan.py` | 6,396 | `677b6362d25e077fb78c756579533fe80bee5e5b655db62227a9260021ae2e43` | The weight types of a quantized graph, for the checks of `quantize_dec.py` |
| `conversion/requirements-reference.txt` | 780 | `dbb1cb197eeacc12bc6ed7cff3ae8048a5cd3b83b4b02d2a89290095af4f44bb` | The reference environment (Python 3.12) |
| `conversion/requirements-export.txt` | 1,437 | `8ec6c9922739829034d48cee804a2b7b0dd984a1f0cbcbc2fa3a7d8601b9eb38` | The export environment (Python 3.13) |
| `conversion/npu/Dockerfile` | 1,471 | `ee4bfe807b018e172b021ac9d34f8020206ed62827be3160ef0b8a8dbd639b4a` | Step 6: the image of the SM8850 compile |
| `conversion/npu/aot_compile_sm8850.py` | 3,340 | `0bc6b24ad173511c9c2a8c1c7a341fe6f2476f01ef6be56d9d0d8e2d62ae8ffb` | Step 7: compiles one `_fp16w_npu` graph for the SM8850 |
| `conversion/npu/aot_check.py` | 3,698 | `7c865d76ed24da84d7735b644d033624d79b8b7b609b93644ff4a189d7c9bbf6` | Step 7: reads the compile's log |
| `conversion/npu/aot_compare.py` | 3,397 | `27c098fc0fa58ad7f76029ea73f6eb510ff2641953780fc22d94b167113d0ca9` | Step 8: compares a compiled file with the published one |
| `examples/run_examples.py` | 3,905 | `12984cea4d233303ffa8f012208e44795b9b669c91188ffded8b82fce9bc64d1` | Runs the example requests; `--check` compares them with the source's answers |
| `examples/readme_request.json` | 425 | `04408ea0f8bb85a742be133209e76999b759f8f735f62cef09cbc2d665228f8b` | The call in the source model's README |
| `examples/readme_request.expected.json` | 270 | `a5de9b03a97321657e5a66a67b636359d2ece363dcf078876f945dec54a061bc` | The source's answer to it |
| `examples/default_request.json` | 689 | `fae64925eb5bb1fba8f9f73abddc7b4fc46fabf9512e1e8a50c571e11e6aba6d` | The `DEFAULT_REQUEST` of the source's `inference.py` |
| `examples/default_request.expected.json` | 849 | `52a543a4a589228ca1beca7397d4096897178233e4cc13e088e0b251c861ad86` | The source's answer to it |
| `examples/long_request.json` | 4,135 | `c616f2e577a436fd5657c388a3ff343b082fec2bd6cc59415d3c1bfdf61e9406` | A longer request, invented for this conversion |
| `examples/long_request.expected.json` | 1,127 | `796acfe37ce4b7d8805656a1e87ea529393aaabe75835390fa53bb3979b00aec` | The source's answer to it |
| `host/pocket_dec_litert.py` | 21,280 | `806666959e06bdf097d7bbbb00a02949137ecf4bfe6d0697b3d9cd0096b8af09` | The Python host, with the API of the source's `inference.py` |
| `host/requirements.txt` | 301 | `613eefc2f0fa35f526c7d44234ae794a4dae6cc1d5bba0002889cd3d178ca18e` | The host's requirements |
| `REPRODUCE-MNN-Pocket-0.3B-Dec.md` | 45,537 | `7ef31f1a5002abdbcd1fb03348f286c2135fef960fd7515b213d328e076fe076` | The repository's `REPRODUCE.md` |
| `NOTICE` | 2,917 | `b717576d6bc3e299533bada2bfb275f738b61af48a024a9777087bd4e0f49490` | The attribution and what the conversion changed |
| `SHA256SUMS` | 6,112 | `9a386d4765205cf1b9765e17daf17560b173c6fee7aa7f7ccb7925b5cf1179b9` | The SHA-256 of every file of the repository except `README.md` and itself; step 5 reads it |

## How to reproduce

`conversion/README.md` gives the commands in order. They run from this folder as written, because it has the repository's layout without the graphs and the files that step 1 writes. Step 8 compares the compiled files with the published SM8850 files, which are on the Hub only, so download those into this folder first. Steps 6 and 7 need QAIRT 2.50.0.260828, Qualcomm's SDK, which is not in this folder; `conversion/README.md` gives its URL and SHA-256. `REPRODUCE-MNN-Pocket-0.3B-Dec.md` gives the sources with SHA-256, the environments, the steps, the checks and where each number comes from.

**From this folder, the commands rebuild the 20 graphs other than the SM8850 files byte for byte.** On 2026-11-08, on a Mac (Apple M4 Max, 128 GB, macOS 27.0), steps 1 to 5 ran as written in two new virtual environments (Python 3.12.13 and 3.13.13, with the versions of `conversion/requirements-*.txt`). They wrote the 20 graphs with the SHA-256 of `SHA256SUMS` (`shasum -c` printed OK for each) and the 5 files of step 1 with the Hub's bytes, in 368 seconds. The largest process reached 7.4 GB of resident memory.

**The SM8850 file: the same computation, not the same bytes.** Steps 6 to 8 ran for the 64-token file only. The compile of the rebuilt `mnn-pocket-0.3b-dec_L64_fp16w_npu.tflite` wrote 494,391,320 bytes like the published file, with SHA-256 `4c20bfc8590afd38a3b565f9e2f2856e9fa68fa1e836bd6a5581c00b4fbf1930` where the published file has `9f6eb4c5618b1b84df429d5b175819b529a1ac2c413106fcf106ec25af2398d4`. `aot_compare.py` printed `same apart from unwritten bits: 3,238 32-bit words of the context binary differ, each in its upper 16 bits only`, the difference that `REPRODUCE-MNN-Pocket-0.3B-Dec.md` explains. Docker built the image from the cached layers of an earlier build of the same `conversion/npu/Dockerfile` with the same QAIRT files on that Mac.

## How to run the published files

`host/pocket_dec_litert.py` here is the same file as on the Hub. The model card's commands download the CPU and GPU files and check them against the source model's answers:

```bash
hf download litert-community/MNN-Pocket-0.3B-Dec-LiteRT --local-dir MNN-Pocket-0.3B-Dec-LiteRT --exclude "*_npu*" --exclude "*_tensor*"
cd MNN-Pocket-0.3B-Dec-LiteRT
python3 -m venv .venv
.venv/bin/pip install -r host/requirements.txt
.venv/bin/python examples/run_examples.py --check
```

The model card gives the agreement with the source model on each runtime and phone, the speeds, and which file to run where.

## License and attribution

- `taobao-mnn/MNN-Pocket-0.3B-Dec` is Apache-2.0 (Copyright 2026 Taobao and Tmall Group of Alibaba). The published LiteRT files are converted from it.
- The scripts, the host and the example runner in this folder are Apache-2.0.
- `NOTICE` gives the attribution and what the conversion changed.
