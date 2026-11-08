# How the LiteRT files were built and checked

Everything here ran on 2026-11-07 and 2026-11-08 on a Mac (Apple M4 Max, 128 GB, macOS 27.0), one Samsung Galaxy S26 (SM-S942Q, Qualcomm SM8850, MemTotal 11,389,756 kB, Android 16, build BP4A.251205.006) and one Google Pixel 11 Pro (Tensor G6, MemTotal 16,030,796 kB, Android 17, build CD1A.260905.001.B1). The scripts and the commands in order are in [`conversion/`](conversion/README.md).

## Sources

All from [`taobao-mnn/MNN-Pocket-0.3B-Dec`](https://huggingface.co/taobao-mnn/MNN-Pocket-0.3B-Dec) at revision `dfbe0aed3abd4caed175b883a4b462ed35184fad`.

| What | File | Bytes | SHA-256 |
|---|---|---:|---|
| Base backbone | `base/model.safetensors` | 723,137,248 | `24e182cebe433e9354dc35567d66c655888e7577ca7f6f71cea196a5b7da6276` |
| LoRA adapter | `adapter/adapter_model.safetensors` | 28,723,600 | `7a3b432e65ef79a9fd0600e8d711e2e4bb4d9c6ce949935c97c725ae9d253696` |
|  | `adapter/adapter_config.json` | 1,248 | `da48ff319ebb8d3cd4cde5403cc58a7c86a9d9c35ede212584c3b888c930985e` |
| Decision head | `decision_head.safetensors` | 1,837,352 | `653cc9f7695bc142f933f19df6bf3fcdeeee875558438c3e95eeb6db065ad0a2` |
| Temperature | `temperature.json` | 418 | `337a860aa0989975a98bf81c498a815a1a36a51d9770f08022e08291abf2f71d` |
| Tokenizer | `base/tokenizer.json` | 4,766,512 | `466e9968103aed7cc58177b435d89e5c76e6acce38ea6f2e6e20bd7fff4b4784` |
| Original code | `inference.py` | 17,099 | `69567ac174b3562339e3c67a4e1e2309d4612c05061d2da5008c4a1b21c68e77` |
| License | `LICENSE.txt` | 11,363 | `c4082a47027777530d49447f8f105002466257bcb2985f5aded0736d10061729` |

## Environments

- Reference: Python 3.12, `conversion/requirements-reference.txt` (torch 2.8.0, transformers 4.57.1, peft 0.19.1, safetensors 0.7.0, accelerate 1.14.0, numpy 1.26.4: the versions in the original repository's `requirements.txt`, with their dependencies).
- Export and quantization: Python 3.13, `conversion/requirements-export.txt` (torch 2.14.1, litert-torch 0.10.0, litert-converter 0.5.0, ai-edge-litert 2.3.0, ai-edge-quantizer 0.10.1, numpy 2.5.4, tokenizers 0.23.3, safetensors 0.8.0).
- The SM8850 compile: Docker on the Mac (linux/amd64), the image of `conversion/npu/Dockerfile`: Python 3.12.13, ai-edge-litert 2.3.0 with its Qualcomm compiler plugin, and QAIRT 2.50.0.260828, Qualcomm's SDK, which is not in this repository. The published SM8850 files were compiled in an earlier image with the same ai-edge-litert and QAIRT; given the same input file, the two images write the same bytes.
- Python host: Python 3.14.6, `host/requirements.txt` (ai-edge-litert 2.3.0, numpy 2.5.4, tokenizers 0.23.3, safetensors 0.8.0, ml-dtypes 0.6.0).
- Phones: LiteRT 2.3.0 and its GPU accelerator, the same version as the Mac's `ai-edge-litert`, driven through its C API by a small C++ program that is not part of this repository. On the Galaxy S26's NPU, also LiteRT's Qualcomm dispatch and compiler plugins (Maven `litert-npu-runtime-qualcomm` 2.3.0) and the QNN runtime libraries of QAIRT 2.50. On the Pixel 11 Pro's NPU, LiteRT's Google Tensor dispatch library and compiler plugin (Maven `litert-npu-runtime-google-tensor` 2.3.0) and the phone's own `/vendor/lib64/libedgetpu_litert.so`. The phones have no Python host: the Mac computes each request's inputs with the host's code, the phone runs the graph and saves the hidden states, and the Mac applies the host's head and temperature to them.

## Steps

The commands and their arguments are in `conversion/README.md`.

1. `bundle_files.py` copies the files the host reads and writes `embedding.safetensors`: the bytes of `model.embed_tokens.weight` in `base/model.safetensors`, unchanged (bfloat16).
2. `reference_examples.py` runs the original `inference.py` on the three example requests and compares its answers with `examples/*.expected.json`.
3. `merge_lora.py` folds the adapter into the backbone in float32: `W + (B @ A) * (lora_alpha / r)`, with r 16 and `lora_alpha` 32. The result (1,211,358,392 bytes, SHA-256 `a2f58515fc25ceb1974cf95452f8cd85c72efbbbb550354db1a4118fc2ed38d3`) has the same value as peft's `merge_and_unload()` in all 310 tensors, bit for bit. Through the original model's code it gives the same winner on all 2,409 questions below, and every probability is within 1.1e-5 of the reference.
4. `dec_graph.py` writes the decoder in plain PyTorch, in three forms (`conversion/README.md`, "The forms"): `k4` for the `_fp16w` files, `k4_exp_silu` for the `_fp16w_npu` files and `mean_norm` for the `_fp16w_tensor` files. In float32 each form gives the same winner on all 2,409 questions, with every probability within 1.3e-5, 1.1e-5 and 7.3e-6 of the reference, in that order.
5. `export_dec.py` converts it with litert-torch at each length and checks the module on the example requests that fit the length first. `quantize_dec.py` stores the 196 FULLY_CONNECTED weights as float16 with ai-edge-quantizer (`FLOAT_CASTING`); the computation stays in float. `shasum -a 256 -c` compares the 20 files with `SHA256SUMS`.
6. to 8. `npu/aot_compile_sm8850.py`, in the image of `npu/Dockerfile`, compiles each `_fp16w_npu` file ahead of time for the SM8850; `npu/aot_check.py` reads the compile's log, and `npu/aot_compare.py` compares the output with the published file.

**The commands rebuild the 20 published files of steps 1 to 5 byte for byte.** On the Mac above, the commands of `conversion/README.md`, run as written from a copy of this repository without the `.tflite` files and the files step 1 writes, in two new virtual environments (Python 3.12.13 and 3.13.13), wrote 18 of the 20 `.tflite` files with the SHA-256 in the table below, the 5 files of step 1 with the same bytes, and, behind each `.tflite` file, the float32 graph with the same SHA-256 as the run that made the published file. A second run, after the L768 files were added, limited steps 4 and 5 to L768 and wrote the two files of that length the same way. Steps 1 to 5 took 357 and 58 seconds. The largest process reached 7.4 GB of resident memory. `PYTHONHASHSEED=0` on the export lines is part of the recipe. Another machine or another version of the packages may give different bytes.

**The SM8850 files: the same computation, not the same bytes.** Step 7 compiled the L64 file again from the rebuilt `mnn-pocket-0.3b-dec_L64_fp16w_npu.tflite`: 494,391,320 bytes like the published file, with SHA-256 `4c20bfc8590afd38a3b565f9e2f2856e9fa68fa1e836bd6a5581c00b4fbf1930` where the published file has `9f6eb4c5618b1b84df429d5b175819b529a1ac2c413106fcf106ec25af2398d4`. The first 1,776 bytes, the flatbuffer with the DISPATCH_OP and its options, are the same. The files differ in 6,229 bytes of the context binary, all of them in the upper 16 bits of 3,238 32-bit words whose lower 16 bits are the same, and step 8 (`aot_compare.py`) reports exactly that. The compiler does not write those bits, so they keep what was in its memory: a compile under glibc's `MALLOC_PERTURB_=165`, which fills the memory the allocator hands out and takes back, changes the upper 16 bits of 928 words and no lower half, and 493 of the 3,238 words then hold the fill values 0x5a5a and 0xa5a5. Everything else about the compile repeats: the same command wrote the same bytes twice, the image the published files were compiled in wrote the same bytes from the same file, and given the input file name the published files were compiled under, both images wrote the published bytes. The second run compiled the L768 file the same way, from its rebuilt `mnn-pocket-0.3b-dec_L768_fp16w_npu.tflite`: 517,226,520 bytes, SHA-256 `f718413fed35dd1ff8f605dc6c712e52f998391e0b458e0e8b6d48a9c8b30ba9` where the published file has `e7ef9525e25cf2ade15ef1cc47a9de5940c65c950c6b093951335ee0708809da`, the same first 1,776 bytes, and 7,345 differing bytes, all in the upper 16 bits of 3,726 words of the context binary, which step 8 reports; given the input file name its published file was compiled under, the image of `npu/Dockerfile` wrote the published bytes. On the Galaxy S26, at every length where both ran, the published SM8850 file and the same graph compiled on the phone give the same probabilities to every digit (below). The other 6 SM8850 files were made by the same command and were not compiled again here.

## Published files

`SHA256SUMS` lists every file of this repository except `README.md` and itself.

| File | For | Bytes | SHA-256 |
|---|---|---:|---|
| `mnn-pocket-0.3b-dec_L64_fp16w.tflite` | CPU and GPU | 488,880,336 | `ceaa8bd8eff2c335467ca49193f3625bf6b397f0286cef8527fe1c8b1834745e` |
| `mnn-pocket-0.3b-dec_L96_fp16w.tflite` | CPU and GPU | 488,880,528 | `4136dfe3eca3079727904b413633a2e9b25c3f1ceb3224b5360ef29d3fdaad55` |
| `mnn-pocket-0.3b-dec_L128_fp16w.tflite` | CPU and GPU | 488,880,528 | `0f93fb8791a82189105dcc709d83e5788e13127b8303891bb989ac2ae1a135f3` |
| `mnn-pocket-0.3b-dec_L256_fp16w.tflite` | CPU and GPU | 488,880,528 | `7fb2e0c29c7d93de1f6a51c919711de2a5aa2aa42fd6a9c31ffc96072ce4f92a` |
| `mnn-pocket-0.3b-dec_L512_fp16w.tflite` | CPU and GPU | 488,880,528 | `e66a712762dc6036a545c7e514a500617358907e6f7200c8b0efc87245fed48b` |
| `mnn-pocket-0.3b-dec_L768_fp16w.tflite` | CPU and GPU | 488,880,528 | `7dc7ca0596129e7dc0eec2e952b0b07bacbbd048d17d7988b3f13c4903c10034` |
| `mnn-pocket-0.3b-dec_L1024_fp16w.tflite` | CPU and GPU | 488,880,528 | `a76354c9b14333d4d55385c4d0cd86b3bfe8a6df30a99a9963b138ea71a8397c` |
| `mnn-pocket-0.3b-dec_L2048_fp16w.tflite` | CPU and GPU | 488,880,528 | `957db64c9e2b94b957c2f4c38371bd2cd38e2ce60076d64f34a9cfe3191f0c85` |
| `mnn-pocket-0.3b-dec_L64_fp16w_npu.tflite` | Qualcomm NPU, compiled on the phone | 488,905,856 | `ff5583ccc10ee2caa1a5a9f5c3991d52ec85dbdf485b1811ecc1bd71e4182fe8` |
| `mnn-pocket-0.3b-dec_L96_fp16w_npu.tflite` | Qualcomm NPU, compiled on the phone | 488,906,048 | `33d6cd6ab71545d6034bc8fe6854718aaba20da60c67f0523377de8e84cd80ef` |
| `mnn-pocket-0.3b-dec_L128_fp16w_npu.tflite` | Qualcomm NPU, compiled on the phone | 488,906,048 | `d1d6a889298cde29e11208e713ecfbe70695fe019aeb654610c6907bca515783` |
| `mnn-pocket-0.3b-dec_L256_fp16w_npu.tflite` | Qualcomm NPU, compiled on the phone | 488,906,048 | `825982b3674fe1b6ac28c852dd3c28b8b5bf6161324272bfdc13d501df45fd41` |
| `mnn-pocket-0.3b-dec_L512_fp16w_npu.tflite` | Qualcomm NPU, compiled on the phone | 488,906,048 | `9516eb4264e8ae44eff60a1807cfec08927badc94aa2b5e5d034ff0556f9450d` |
| `mnn-pocket-0.3b-dec_L768_fp16w_npu.tflite` | Qualcomm NPU, compiled on the phone | 488,906,048 | `1b60cb0030a29c2a839d6822c132389b01ce988a850aa4e7a1203a26e6ee80e5` |
| `mnn-pocket-0.3b-dec_L1024_fp16w_npu.tflite` | Qualcomm NPU, compiled on the phone | 488,906,048 | `64f94603fbbab15956661ab9ba920347aa7b0b1f9584d9a44dc08c9e90ea0079` |
| `mnn-pocket-0.3b-dec_L2048_fp16w_npu.tflite` | Qualcomm NPU, compiled on the phone | 488,906,048 | `8fb1fc76153c92a097f5d5499cd96cf7bc6b490afa48fcc8d30415c24ee72999` |
| `mnn-pocket-0.3b-dec_L64_fp16w_npu_SM8850.tflite` | Qualcomm SM8850 NPU, compiled ahead of time | 494,391,320 | `9f6eb4c5618b1b84df429d5b175819b529a1ac2c413106fcf106ec25af2398d4` |
| `mnn-pocket-0.3b-dec_L96_fp16w_npu_SM8850.tflite` | Qualcomm SM8850 NPU, compiled ahead of time | 494,903,320 | `5fbeced0c6d50bc6a160a76ad91cfa1142a94ab1f4405169475c259a65d546eb` |
| `mnn-pocket-0.3b-dec_L128_fp16w_npu_SM8850.tflite` | Qualcomm SM8850 NPU, compiled ahead of time | 495,296,536 | `01584b13d8f6a2c7b1b5982854d6935f6b0e24fa6bdc97a37dd02453e45b0299` |
| `mnn-pocket-0.3b-dec_L256_fp16w_npu_SM8850.tflite` | Qualcomm SM8850 NPU, compiled ahead of time | 498,327,576 | `b5c25051c96cc5f396b1f017bc4823afc9c2df0353687ca609218c1589d3350a` |
| `mnn-pocket-0.3b-dec_L512_fp16w_npu_SM8850.tflite` | Qualcomm SM8850 NPU, compiled ahead of time | 509,464,600 | `31d729102125cd2214e9e647c15efae627f1bdb0262c74a03c7a953b201a0583` |
| `mnn-pocket-0.3b-dec_L768_fp16w_npu_SM8850.tflite` | Qualcomm SM8850 NPU, compiled ahead of time | 517,226,520 | `e7ef9525e25cf2ade15ef1cc47a9de5940c65c950c6b093951335ee0708809da` |
| `mnn-pocket-0.3b-dec_L1024_fp16w_npu_SM8850.tflite` | Qualcomm SM8850 NPU, compiled ahead of time | 529,834,008 | `d2dd03180b78bd2228e0b966032575aafcb888411b8993a5eb7b613c463a1d60` |
| `mnn-pocket-0.3b-dec_L2048_fp16w_npu_SM8850.tflite` | Qualcomm SM8850 NPU, compiled ahead of time | 605,974,552 | `6b667dad5ac71498fda153c81aa6a55bc2f91c8a17035edadb53061327791d83` |
| `mnn-pocket-0.3b-dec_L64_fp16w_tensor.tflite` | Google Tensor NPU | 488,975,184 | `649e80b81cb07bad6f0f4cf209bddaefea3d305597d0926a9ab8450da861fd47` |
| `mnn-pocket-0.3b-dec_L96_fp16w_tensor.tflite` | Google Tensor NPU | 488,975,376 | `71462f61f7808f4acb55ddc2e07930c5586e6d2760034a4ac346035a2f74e041` |
| `mnn-pocket-0.3b-dec_L128_fp16w_tensor.tflite` | Google Tensor NPU | 488,975,376 | `3d0cbf0d8d05bf8738e8dacc5e2d2bb2fec26c1ef4a31b2449351c2165fdb8e7` |
| `mnn-pocket-0.3b-dec_L256_fp16w_tensor.tflite` | Google Tensor NPU | 488,975,376 | `10a2f75325ea645a26cca863cb88e7f132b36b976ce3bd71c6271d9f8804fcdc` |
| `embedding.safetensors` | the host | 117,440,616 | `da8f89d90c9feace23085f49a7f8eb0780f76b9c6203aa4ec2323f852cd055ff` |
| `tokenizer.json` | the host | 4,766,512 | `466e9968103aed7cc58177b435d89e5c76e6acce38ea6f2e6e20bd7fff4b4784` |
| `decision_head.safetensors` | the host | 1,837,352 | `653cc9f7695bc142f933f19df6bf3fcdeeee875558438c3e95eeb6db065ad0a2` |
| `temperature.json` | the host | 418 | `337a860aa0989975a98bf81c498a815a1a36a51d9770f08022e08291abf2f71d` |
| `LICENSE` |  | 11,363 | `c4082a47027777530d49447f8f105002466257bcb2985f5aded0736d10061729` |

Each graph takes `embeds` float32 `[1, L, 896]`, `cos` and `sin` float32 `[1, L, 64]` and `mask` float32 `[1, 1, L, L]`, and returns `hidden` float32 `[1, L, 896]`, the output of the final RMSNorm. The host reads the embedding table, the head and the temperature from their own files, and loads only the `_fp16w` files.

Operators of the `_fp16w` files: ADD 253, BATCH_MATMUL 56, CONCATENATION 112, DEQUANTIZE 196, FULLY_CONNECTED 196, LOGISTIC 28, MEAN 113, MUL 636, RESHAPE 449, RSQRT 113, SLICE 112, SOFTMAX 28, TRANSPOSE 140. Of the `_fp16w_npu` files: ABS 28, ADD 281, BATCH_MATMUL 56, CONCATENATION 112, DEQUANTIZE 196, DIV 28, EXP 56, FULLY_CONNECTED 196, MEAN 113, MINIMUM 28, MUL 664, RESHAPE 449, RSQRT 113, SLICE 112, SOFTMAX 28, TRANSPOSE 140. Of the `_fp16w_tensor` files: ABS 113, ADD 253, BATCH_MATMUL 56, CONCATENATION 112, DEQUANTIZE 196, DIV 113, FULLY_CONNECTED 196, LOGISTIC 28, MEAN 226, MUL 1,043, RESHAPE 336, RSQRT 113, SLICE 112, SOFTMAX 28, TRANSPOSE 140. Every length of a form has the same operators, and none of these files has a custom op. Each `_SM8850` file is one DISPATCH_OP whose context binary holds the whole graph.

## The reference

The reference is the original `inference.py`, unchanged, on the original files: float32, CPU, one torch thread, `predict(state, questions)`, the probabilities after the temperature at full precision. A second run gave the same probabilities, logits and read-out hidden states bit for bit. With the adapter disabled, the answers move by up to 0.998 and the winner changes on 1,363 of 2,409 questions, so these requests can see the adapter.

## The requests

| Set | Requests | Questions | Near ties in the reference | Tokens per request | Where it comes from |
|---|---:|---:|---:|---:|---|
| Kev-0.8B-LiteRT's requests | 377 | 402 | 24 | 36 to 1,926 | `fixtures/rebuild_requests.py` in [`litert-community/Kev-0.8B-LiteRT`](https://huggingface.co/litert-community/Kev-0.8B-LiteRT/tree/7e21f8d4b0b23e9ef8fd455ceab99b3f864abf99/fixtures) at commit `7e21f8d4b0b23e9ef8fd455ceab99b3f864abf99` rebuilds them (SHA-256 of the result `3bcc256671b49838ea781c2ff3388cf67745b69af6977652f7e90bc15723e2c7`) |
| typed-decisions test split | 400 | 2,000 | 61 | 363 to 838 | [`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) at revision `f5affa5f05859bd8486bc15e1e352a04f67994cf`, `all/test-00000-of-00001.parquet`, SHA-256 `4f294f218ea1da27f3efef936359389c62ea4d3973a41457732990f1d31b647c`. Each row is `predict(json.loads(row["state"]), json.loads(row["questions"]))`, in file order |
| The examples | 3 | 7 | 1 | 60, 85, 1,024 | `examples/readme_request.json`, `default_request.json`, `long_request.json` in this repository |

That is 780 requests and 2,409 questions; 2,402 of the questions have a gold answer. Kev-0.8B-LiteRT's requests quote public datasets under several licenses, so they stay by reference in that repository, whose `fixtures/README.md` lists them. The checks read the same 377 records with a different `created_by` string in the file's header, so that file's SHA-256 is `dfe55fb145df7a3967ed7213ae24d67b5d4ec42da0315fdb409d5e4b702e3a48`. `long_request.json` is an invented incident log written for this conversion.

## The bar

A file passes on a runtime when, over the questions whose request fits its length (the others are skipped for that file), against the reference:

- the winner (the option with the highest logit) is the same on every question whose reference top-two probabilities are more than 0.02 apart. Closer pairs are near ties, counted apart;
- the largest absolute difference of any option's probability is at most 0.02, near ties included;
- the mean absolute difference over all options is at most 0.002, near ties included;
- the control moves: the file's answer to `red_arm_000` (among Kev-0.8B-LiteRT's requests: `tv4_000` with one word of the instructions changed) differs from the reference's answer to `tv4_000` by more than 0.02, on every file that ran that request;
- the hidden state is finite at every position the head reads;
- on a phone, the winner also equals the same file's on the Mac CPU on every question that is not a near tie;
- on an NPU, the whole graph runs on the NPU: one partition that holds every op.

## The Mac

`ai-edge-litert` 2.3.0, Python `CompiledModel`, all 780 requests where they fit. Metal at float32 is `GpuOptions(enforce_f32=True)` with `HardwareAccelerator.GPU` alone; default is the same without the option. On Metal every run is fully accelerated.

| File | Runtime | Questions | Same winner | Near ties kept | Max \|Δp\| | Mean \|Δp\| | Control | Result |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `L64_fp16w` | CPU, 4 threads | 71 | 68/68 | 3/3 | 7.9e-4 | 1.9e-4 | — | Pass |
| `L64_fp16w` | GPU (Metal), float32 | 71 | 68/68 | 3/3 | 7.8e-4 | 1.9e-4 | — | Pass |
| `L96_fp16w` | CPU, 4 threads | 278 | 267/267 | 11/11 | 0.0014 | 1.9e-4 | 0.0401 | Pass |
| `L96_fp16w` | GPU (Metal), float32 | 278 | 267/267 | 11/11 | 0.0014 | 1.9e-4 | 0.0401 | Pass |
| `L128_fp16w` | CPU, 4 threads | 322 | 305/305 | 17/17 | 0.0021 | 1.9e-4 | 0.0401 | Pass |
| `L128_fp16w` | GPU (Metal), float32 | 322 | 305/305 | 17/17 | 0.0021 | 1.9e-4 | 0.0401 | Pass |
| `L128_fp16w` | GPU (Metal), default (float16) | 322 | 305/305 | 14/17 | 0.0376 | 0.0052 | 0.0291 | Fail: max, mean |
| `L256_fp16w` | CPU, 4 threads | 379 | 356/356 | 23/23 | 0.0021 | 1.9e-4 | 0.0401 | Pass |
| `L256_fp16w` | GPU (Metal), float32 | 379 | 356/356 | 23/23 | 0.0021 | 1.9e-4 | 0.0401 | Pass |
| `L256_fp16w` | GPU (Metal), default (float16) | 379 | 353/356 | 16/23 | 0.0375 | 0.0066 | 0.0409 | Fail: winner, max, mean |
| `L512_fp16w` | CPU, 4 threads | 1,232 | 1,186/1,186 | 46/46 | 0.0021 | 1.2e-4 | 0.0401 | Pass |
| `L512_fp16w` | GPU (Metal), float32 | 1,232 | 1,186/1,186 | 46/46 | 0.0021 | 1.2e-4 | 0.0401 | Pass |
| `L512_fp16w` | GPU (Metal), default (float16) | 1,232 | 1,182/1,186 | 32/46 | 0.0399 | 0.0050 | 0.0452 | Fail: winner, max, mean |
| `L768_fp16w` | CPU, 4 threads | 2,362 | 2,276/2,276 | 86/86 | 0.0021 | 1.1e-4 | 0.0401 | Pass |
| `L768_fp16w` | GPU (Metal), float32 | 2,362 | 2,276/2,276 | 86/86 | 0.0021 | 1.1e-4 | 0.0401 | Pass |
| `L1024_fp16w` | CPU, 4 threads | 2,400 | 2,314/2,314 | 86/86 | 0.0021 | 1.1e-4 | 0.0401 | Pass |
| `L1024_fp16w` | GPU (Metal), float32 | 2,400 | 2,314/2,314 | 86/86 | 0.0021 | 1.1e-4 | 0.0401 | Pass |
| `L1024_fp16w` | GPU (Metal), default (float16) | 2,400 | 2,305/2,314 | 64/86 | 0.0463 | 0.0045 | 0.0403 | Fail: winner, max, mean |
| `L2048_fp16w` | CPU, 4 threads | 2,409 | 2,323/2,323 | 86/86 | 0.0021 | 1.1e-4 | 0.0401 | Pass |
| `L2048_fp16w` | GPU (Metal), float32 | 2,409 | 2,323/2,323 | 86/86 | 0.0021 | 1.1e-4 | 0.0401 | Pass |
| `L2048_fp16w` | GPU (Metal), default (float16) | 2,409 | 2,318/2,323 | 60/86 | 0.0580 | 0.0047 | 0.0410 | Fail: winner, max, mean |
| `L64_fp16w_npu` | CPU, 4 threads | 71 | 68/68 | 3/3 | 7.8e-4 | 1.9e-4 | — | Pass |
| `L64_fp16w_npu` | GPU (Metal), float32 | 71 | 68/68 | 3/3 | 7.8e-4 | 1.9e-4 | — | Pass |
| `L96_fp16w_npu` | CPU, 4 threads | 278 | 267/267 | 11/11 | 0.0014 | 1.9e-4 | 0.0401 | Pass |
| `L96_fp16w_npu` | GPU (Metal), float32 | 278 | 267/267 | 11/11 | 0.0014 | 1.9e-4 | 0.0401 | Pass |
| `L128_fp16w_npu` | CPU, 4 threads | 322 | 305/305 | 17/17 | 0.0021 | 1.9e-4 | 0.0401 | Pass |
| `L128_fp16w_npu` | GPU (Metal), float32 | 322 | 305/305 | 17/17 | 0.0021 | 1.9e-4 | 0.0401 | Pass |
| `L128_fp16w_npu` | GPU (Metal), default (float16) | 322 | 303/305 | 13/17 | 0.0605 | 0.0056 | 0.0388 | Fail: winner, max, mean |
| `L256_fp16w_npu` | CPU, 4 threads | 379 | 356/356 | 23/23 | 0.0021 | 1.9e-4 | 0.0401 | Pass |
| `L256_fp16w_npu` | GPU (Metal), float32 | 379 | 356/356 | 23/23 | 0.0021 | 1.9e-4 | 0.0401 | Pass |
| `L512_fp16w_npu` | CPU, 4 threads | 1,232 | 1,186/1,186 | 46/46 | 0.0021 | 1.2e-4 | 0.0401 | Pass |
| `L512_fp16w_npu` | GPU (Metal), float32 | 1,232 | 1,186/1,186 | 46/46 | 0.0021 | 1.2e-4 | 0.0401 | Pass |
| `L512_fp16w_npu` | GPU (Metal), default (float16) | 1,232 | 1,183/1,186 | 37/46 | 0.0540 | 0.0049 | 0.0460 | Fail: winner, max, mean |
| `L768_fp16w_npu` | CPU, 4 threads | 2,362 | 2,276/2,276 | 86/86 | 0.0021 | 1.1e-4 | 0.0401 | Pass |
| `L768_fp16w_npu` | GPU (Metal), float32 | 2,362 | 2,276/2,276 | 86/86 | 0.0021 | 1.1e-4 | 0.0401 | Pass |
| `L1024_fp16w_npu` | CPU, 4 threads | 2,400 | 2,314/2,314 | 86/86 | 0.0021 | 1.1e-4 | 0.0401 | Pass |
| `L1024_fp16w_npu` | GPU (Metal), float32 | 2,400 | 2,314/2,314 | 86/86 | 0.0021 | 1.1e-4 | 0.0401 | Pass |
| `L2048_fp16w_npu` | CPU, 4 threads | 2,409 | 2,323/2,323 | 86/86 | 0.0021 | 1.1e-4 | 0.0401 | Pass |
| `L2048_fp16w_npu` | GPU (Metal), float32 | 2,409 | 2,323/2,323 | 86/86 | 0.0021 | 1.1e-4 | 0.0401 | Pass |
| `L64_fp16w_tensor` | CPU, 4 threads | 71 | 68/68 | 3/3 | 7.8e-4 | 1.9e-4 | — | Pass |
| `L64_fp16w_tensor` | GPU (Metal), float32 | 71 | 68/68 | 3/3 | 7.8e-4 | 1.9e-4 | — | Pass |
| `L64_fp16w_tensor` | GPU (Metal), default (float16) | 71 | 67/68 | 3/3 | 0.0388 | 0.0051 | — | Fail: winner, max, mean |
| `L96_fp16w_tensor` | CPU, 4 threads | 278 | 267/267 | 11/11 | 0.0014 | 1.9e-4 | 0.0401 | Pass |
| `L96_fp16w_tensor` | GPU (Metal), float32 | 278 | 267/267 | 11/11 | 0.0014 | 1.9e-4 | 0.0401 | Pass |
| `L128_fp16w_tensor` | CPU, 4 threads | 322 | 305/305 | 17/17 | 0.0021 | 1.9e-4 | 0.0401 | Pass |
| `L128_fp16w_tensor` | GPU (Metal), float32 | 322 | 305/305 | 17/17 | 0.0021 | 1.9e-4 | 0.0401 | Pass |
| `L128_fp16w_tensor` | GPU (Metal), default (float16) | 322 | 303/305 | 12/17 | 0.0518 | 0.0054 | 0.0332 | Fail: winner, max, mean |
| `L256_fp16w_tensor` | CPU, 4 threads | 379 | 356/356 | 23/23 | 0.0021 | 1.9e-4 | 0.0401 | Pass |
| `L256_fp16w_tensor` | GPU (Metal), float32 | 379 | 356/356 | 23/23 | 0.0021 | 1.9e-4 | 0.0401 | Pass |

Every file passes on the CPU and on Metal at float32. Metal's default float16 activations fail on every file where they ran (`L128_fp16w`, `L256_fp16w`, `L512_fp16w`, `L1024_fp16w`, `L2048_fp16w`, `L128_fp16w_npu`, `L512_fp16w_npu`, `L64_fp16w_tensor`, `L128_fp16w_tensor`), so the host asks for float32. The L64 files hold no request of the control pair, so their control is not counted. On `L2048_fp16w`, on the CPU and on Metal at float32, 1,713 of the 2,402 questions with a gold answer get it right, the same questions the reference gets right.

## The phones: CPU and GPU

The `_fp16w` files with a fixed set of requests: all of Kev-0.8B-LiteRT's requests and, in file order, the first 24 typed-decisions requests of 257 to 512 tokens and the first 24 of 513 to 1,024 tokens, each on the smallest of the L128, L256, L512, L1024 and L2048 files that holds it, plus the examples on every one of these files that holds them. The L64 and L96 files run the requests of the L128 set that fit them, and the L768 files the first 24 typed-decisions requests of 513 to 768 tokens and the examples that fit them. That gives L64 71 requests and 71 questions, L96 276 requests and 278 questions, L128 320 requests and 322 questions, L256 48 requests and 61 questions, L512 36 requests and 142 questions, L768 26 requests and 124 questions, L1024 27 requests and 127 questions, L2048 6 requests and 16 questions. The control is in the L96 and L128 sets.

| Phone | Runtime | File | Questions | Same winner | Near ties kept | Max \|Δp\| | Mean \|Δp\| | Control | Same winner as the Mac CPU | Result |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Galaxy S26 | CPU, 4 threads | `L64_fp16w` | 71 | 68/68 | 3/3 | 7.9e-4 | 1.9e-4 | — | 68/68 | Pass |
| Galaxy S26 | CPU, 4 threads | `L96_fp16w` | 278 | 267/267 | 11/11 | 0.0014 | 1.9e-4 | 0.0401 | 267/267 | Pass |
| Galaxy S26 | CPU, 4 threads | `L128_fp16w` | 322 | 305/305 | 17/17 | 0.0021 | 1.9e-4 | 0.0401 | 305/305 | Pass |
| Galaxy S26 | CPU, one thread | `L256_fp16w` | 61 | 54/54 | 7/7 | 5.7e-4 | 1.4e-4 | — | 54/54 | Pass |
| Galaxy S26 | CPU, one thread | `L512_fp16w` | 142 | 137/137 | 5/5 | 0.0010 | 8.5e-5 | — | 137/137 | Pass |
| Galaxy S26 | CPU, 4 threads | `L768_fp16w` | 124 | 119/119 | 5/5 | 9.1e-4 | 9.9e-5 | — | 119/119 | Pass |
| Galaxy S26 | CPU, one thread | `L1024_fp16w` | 127 | 122/122 | 5/5 | 9.1e-4 | 9.7e-5 | — | 122/122 | Pass |
| Galaxy S26 | CPU, one thread | `L2048_fp16w` | 16 | 15/15 | 1/1 | 5.4e-4 | 1.5e-4 | — | 15/15 | Pass |
| Galaxy S26 | GPU, FP32 | `L64_fp16w` | 71 | 68/68 | 3/3 | 7.8e-4 | 1.9e-4 | — | 68/68 | Pass |
| Galaxy S26 | GPU, FP32 | `L96_fp16w` | 278 | 267/267 | 11/11 | 0.0014 | 1.9e-4 | 0.0401 | 267/267 | Pass |
| Galaxy S26 | GPU, FP32 | `L128_fp16w` | 322 | 305/305 | 17/17 | 0.0021 | 1.9e-4 | 0.0401 | 305/305 | Pass |
| Galaxy S26 | GPU, FP32 | `L256_fp16w` | 61 | 54/54 | 7/7 | 5.6e-4 | 1.4e-4 | — | 54/54 | Pass |
| Galaxy S26 | GPU, FP32 | `L512_fp16w` | 142 | 137/137 | 5/5 | 0.0010 | 8.5e-5 | — | 137/137 | Pass |
| Galaxy S26 | GPU, FP32 | `L768_fp16w` | 124 | 119/119 | 5/5 | 9.1e-4 | 9.9e-5 | — | 119/119 | Pass |
| Galaxy S26 | GPU, FP32 | `L1024_fp16w` | 127 | 122/122 | 5/5 | 9.1e-4 | 9.7e-5 | — | 122/122 | Pass |
| Galaxy S26 | GPU, FP32 | `L2048_fp16w` | 16 | 15/15 | 1/1 | 5.4e-4 | 1.5e-4 | — | 15/15 | Pass |
| Galaxy S26 | GPU, FP16 with FP32 accumulation | `L64_fp16w` | 71 | 68/68 | 3/3 | 0.0126 | 0.0018 | — | 68/68 | Pass |
| Galaxy S26 | GPU, FP16 with FP32 accumulation | `L96_fp16w` | 278 | 267/267 | 10/11 | 0.0126 | 0.0016 | 0.0379 | 267/267 | Pass |
| Galaxy S26 | GPU, FP16 with FP32 accumulation | `L128_fp16w` | 322 | 305/305 | 15/17 | 0.0126 | 0.0017 | 0.0379 | 305/305 | Pass |
| Galaxy S26 | GPU, FP16 with FP32 accumulation | `L256_fp16w` | 61 | 54/54 | 6/7 | 0.0067 | 0.0015 | — | 54/54 | Pass |
| Galaxy S26 | GPU, FP16 with FP32 accumulation | `L512_fp16w` | 142 | 137/137 | 4/5 | 0.0043 | 9.0e-4 | — | 137/137 | Pass |
| Galaxy S26 | GPU, FP16 with FP32 accumulation | `L768_fp16w` | 124 | 119/119 | 3/5 | 0.0048 | 7.9e-4 | — | 119/119 | Pass |
| Galaxy S26 | GPU, FP16 with FP32 accumulation | `L1024_fp16w` | 127 | 122/122 | 3/5 | 0.0074 | 8.4e-4 | — | 122/122 | Pass |
| Galaxy S26 | GPU, FP16 with FP32 accumulation | `L2048_fp16w` | 16 | 15/15 | 0/1 | 0.0077 | 0.0016 | — | 15/15 | Pass |
| Galaxy S26 | GPU, FP16 | `L128_fp16w` | 322 | 300/305 | 14/17 | 0.0676 | 0.0073 | 0.0460 | 300/305 | Fail: winner, max, mean, same as Mac |
| Galaxy S26 | GPU, FP16 | `L256_fp16w` | 61 | 54/54 | 5/7 | 0.0469 | 0.0065 | — | 54/54 | Fail: max, mean |
| Galaxy S26 | GPU, FP16 | `L512_fp16w` | 142 | 137/137 | 3/5 | 0.0278 | 0.0046 | — | 137/137 | Fail: max, mean |
| Galaxy S26 | GPU, FP16 | `L1024_fp16w` | 127 | 122/122 | 4/5 | 0.0246 | 0.0038 | — | 122/122 | Fail: max, mean |
| Galaxy S26 | GPU, FP16 | `L2048_fp16w` | 16 | 15/15 | 1/1 | 0.0246 | 0.0083 | — | 15/15 | Fail: max, mean |
| Galaxy S26 | GPU, no precision set | `L128_fp16w` | 322 | 300/305 | 14/17 | 0.0676 | 0.0073 | 0.0460 | 300/305 | Fail: winner, max, mean, same as Mac |
| Pixel 11 Pro | CPU, 6 threads | `L64_fp16w` | 71 | 68/68 | 3/3 | 7.9e-4 | 1.9e-4 | — | 68/68 | Pass |
| Pixel 11 Pro | CPU, 6 threads | `L96_fp16w` | 278 | 267/267 | 11/11 | 0.0014 | 1.9e-4 | 0.0401 | 267/267 | Pass |
| Pixel 11 Pro | CPU, 6 threads | `L128_fp16w` | 322 | 305/305 | 17/17 | 0.0021 | 1.9e-4 | 0.0401 | 305/305 | Pass |
| Pixel 11 Pro | CPU, one thread | `L256_fp16w` | 61 | 54/54 | 7/7 | 5.7e-4 | 1.4e-4 | — | 54/54 | Pass |
| Pixel 11 Pro | CPU, one thread | `L512_fp16w` | 142 | 137/137 | 5/5 | 0.0010 | 8.5e-5 | — | 137/137 | Pass |
| Pixel 11 Pro | CPU, 5 threads | `L768_fp16w` | 124 | 119/119 | 5/5 | 9.1e-4 | 9.9e-5 | — | 119/119 | Pass |
| Pixel 11 Pro | CPU, one thread | `L1024_fp16w` | 127 | 122/122 | 5/5 | 9.1e-4 | 9.7e-5 | — | 122/122 | Pass |
| Pixel 11 Pro | CPU, one thread | `L2048_fp16w` | 16 | 15/15 | 1/1 | 5.4e-4 | 1.5e-4 | — | 15/15 | Pass |
| Pixel 11 Pro | GPU, FP32 | `L64_fp16w` | 71 | 68/68 | 3/3 | 7.8e-4 | 1.9e-4 | — | 68/68 | Pass |
| Pixel 11 Pro | GPU, FP32 | `L96_fp16w` | 278 | 267/267 | 11/11 | 0.0014 | 1.9e-4 | 0.0401 | 267/267 | Pass |
| Pixel 11 Pro | GPU, FP32 | `L128_fp16w` | 322 | 305/305 | 17/17 | 0.0021 | 1.9e-4 | 0.0401 | 305/305 | Pass |
| Pixel 11 Pro | GPU, FP32 | `L256_fp16w` | 61 | 54/54 | 7/7 | 5.7e-4 | 1.4e-4 | — | 54/54 | Pass |
| Pixel 11 Pro | GPU, FP32 | `L512_fp16w` | 142 | 137/137 | 5/5 | 0.0010 | 8.5e-5 | — | 137/137 | Pass |
| Pixel 11 Pro | GPU, FP32 | `L768_fp16w` | 124 | 119/119 | 5/5 | 9.1e-4 | 9.9e-5 | — | 119/119 | Pass |
| Pixel 11 Pro | GPU, FP32 | `L1024_fp16w` | 127 | 122/122 | 5/5 | 9.1e-4 | 9.7e-5 | — | 122/122 | Pass |
| Pixel 11 Pro | GPU, FP32 | `L2048_fp16w` | 16 | 15/15 | 1/1 | 5.4e-4 | 1.5e-4 | — | 15/15 | Pass |
| Pixel 11 Pro | GPU, FP16 with FP32 accumulation | `L128_fp16w` | 322 | 305/305 | 14/17 | 0.0166 | 0.0025 | 0.0416 | 305/305 | Fail: mean |
| Pixel 11 Pro | GPU, FP16 with FP32 accumulation | `L256_fp16w` | 61 | 54/54 | 5/7 | 0.0159 | 0.0024 | — | 54/54 | Fail: mean |
| Pixel 11 Pro | GPU, FP16 with FP32 accumulation | `L512_fp16w` | 142 | 137/137 | 3/5 | 0.0079 | 0.0014 | — | 137/137 | Pass |
| Pixel 11 Pro | GPU, FP16 with FP32 accumulation | `L768_fp16w` | 124 | 119/119 | 4/5 | 0.0120 | 0.0015 | — | 119/119 | Pass |
| Pixel 11 Pro | GPU, FP16 with FP32 accumulation | `L1024_fp16w` | 127 | 122/122 | 4/5 | 0.0075 | 0.0015 | — | 122/122 | Pass |
| Pixel 11 Pro | GPU, FP16 with FP32 accumulation | `L2048_fp16w` | 16 | 15/15 | 0/1 | 0.0125 | 0.0024 | — | 15/15 | Fail: mean |
| Pixel 11 Pro | GPU, FP16 | `L128_fp16w` | 322 | 304/305 | 12/17 | 0.0285 | 0.0058 | 0.0421 | 304/305 | Fail: winner, max, mean, same as Mac |
| Pixel 11 Pro | GPU, FP16 | `L256_fp16w` | 61 | 54/54 | 7/7 | 0.0223 | 0.0053 | — | 54/54 | Fail: max, mean |
| Pixel 11 Pro | GPU, FP16 | `L512_fp16w` | 142 | 137/137 | 4/5 | 0.0290 | 0.0037 | — | 137/137 | Fail: max, mean |
| Pixel 11 Pro | GPU, FP16 | `L1024_fp16w` | 127 | 122/122 | 5/5 | 0.0195 | 0.0034 | — | 122/122 | Fail: mean |
| Pixel 11 Pro | GPU, FP16 | `L2048_fp16w` | 16 | 15/15 | 1/1 | 0.0242 | 0.0056 | — | 15/15 | Fail: max, mean |
| Pixel 11 Pro | GPU, no precision set | `L128_fp16w` | 322 | 304/305 | 12/17 | 0.0285 | 0.0058 | 0.0421 | 304/305 | Fail: winner, max, mean, same as Mac |

The CPU rows of the L64, L96 and L128 files ran with 4 threads on the Galaxy S26 and 6 on the Pixel 11 Pro. They equal the one-thread runs of the same files to every digit, and the two phones equal each other. The L768 rows ran with 4 threads on the Galaxy S26 and 5 on the Pixel 11 Pro, from another build of the same C++ program with the same CPU code, and the two phones equal each other there too. The CPU rows of L256, L512, L1024 and L2048 ran on one thread: the C++ program passed the thread count under an option name that LiteRT 2.3.0 does not read, so it used its default. An earlier version of this page called them 4 threads.

On the GPU, every run is fully accelerated. The CPU and FP32 runs pass on both phones, and the CPU's largest and mean differences are the same on the two phones to every digit. FP16 with FP32 accumulation passes on all 8 files on the Galaxy S26. On the Pixel 11 Pro it passes on 3 of 6 files (L512, L768 and L1024); on L128, L256 and L2048 its mean difference is above the bar, so it does not pass on that phone: a runtime passes on a phone only when it passes on every file it ran. FP16 passes on none of the 5 files it ran on either phone. An app that sets no GPU precision gets FP16 on both phones: on the L128 file, all 322 questions have the same probabilities to every digit. A phone app has to set the precision; FP32 passes on both phones.

## The Galaxy S26's NPU (Qualcomm SM8850)

Two files per length run on this NPU: the `_fp16w_npu` file, which LiteRT compiles for the NPU on the phone when it loads it, and the `_fp16w_npu_SM8850` file compiled ahead of time on the Mac (steps 6 to 8). Each runs the requests of the CPU and GPU rows of its length.

| L | File | Questions | Same winner | Near ties kept | Max \|Δp\| | Mean \|Δp\| | Same winner as the Mac CPU | a (ms) | b (ms) | c (ms) | d (ms) | Compile or load (s) | Peak memory (VmHWM, kB) | Ops on the NPU |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | `L64_fp16w_npu`, compiled on the phone | 71 | 68/68 | 3/3 | 0.0188 | 0.0016 | 68/68 | 11.40 | — | — | — | 9.42 | 2,735,684 | 2,600 / 2,600 |
| 64 | `L64_fp16w_npu_SM8850`, SM8850 file | 71 | 68/68 | 3/3 | 0.0188 | 0.0016 | 68/68 | 11.28 | — | — | — | 0.19 | 537,576 | 2,600 / 2,600 |
| 96 | `L96_fp16w_npu`, compiled on the phone | 278 | 267/267 | 10/11 | 0.0190 | 0.0017 | 267/267 | 15.76 | 15.92 | — | — | 10.05 | 2,779,020 | 2,600 / 2,600 |
| 96 | `L96_fp16w_npu_SM8850`, SM8850 file | 278 | 267/267 | 10/11 | 0.0190 | 0.0017 | 267/267 | 16.21 | 15.96 | — | — | 0.21 | 538,480 | 2,600 / 2,600 |
| 128 | `L128_fp16w_npu`, compiled on the phone | 322 | 305/305 | 15/17 | 0.0190 | 0.0017 | 305/305 | 20.90 | 20.69 | — | — | 12.89 | 2,830,228 | 2,600 / 2,600 |
| 128 | `L128_fp16w_npu_SM8850`, SM8850 file | 322 | 305/305 | 15/17 | 0.0190 | 0.0017 | 305/305 | 20.55 | 20.58 | — | — | 0.20 | 539,312 | 2,600 / 2,600 |
| 256 | `L256_fp16w_npu`, compiled on the phone | 61 | 54/54 | 5/7 | 0.0083 | 0.0016 | 54/54 | 39.32 | 39.22 | — | — | 50.29 | 3,066,084 | 2,600 / 2,600 |
| 256 | `L256_fp16w_npu_SM8850`, SM8850 file | 61 | 54/54 | 5/7 | 0.0083 | 0.0016 | 54/54 | 39.58 | 39.49 | — | — | 0.24 | 543,832 | 2,600 / 2,600 |
| 512 | `L512_fp16w_npu`, compiled on the phone | 142 | 137/137 | 3/5 | 0.0060 | 0.0011 | 137/137 | 113.06\* | 109.89\* | — | — | 69.15 | 3,827,688 | 2,600 / 2,600 |
| 512 | `L512_fp16w_npu_SM8850`, SM8850 file | 142 | 137/137 | 3/5 | 0.0060 | 0.0011 | 137/137 | 100.35 | 98.28 | — | — | 0.32 | 557,960 | 2,600 / 2,600 |
| 768 | `L768_fp16w_npu`, compiled on the phone | 124 | 119/119 | 3/5 | 0.0103 | 0.0010 | 119/119 | — | — | — | 142.47 | 107.16 | 4,653,960 | 2,600 / 2,600 |
| 768 | `L768_fp16w_npu_SM8850`, SM8850 file | 124 | 119/119 | 3/5 | 0.0103 | 0.0010 | 119/119 | — | — | — | 140.07 | 0.43 | 569,144 | 2,600 / 2,600 |
| 1024 | `L1024_fp16w_npu`, compiled on the phone | 127 | 122/122 | 3/5 | 0.0103 | 9.5e-4 | 122/122 | 206.04 | 206.69 | 205.55 | — | 314.70 | 4,882,904 | 2,600 / 2,600 |
| 1024 | `L1024_fp16w_npu_SM8850`, SM8850 file | 127 | 122/122 | 3/5 | 0.0103 | 9.5e-4 | 122/122 | 203.41 | 203.97 | 203.44 | — | 0.50 | 586,472 | 2,600 / 2,600 |
| 2048 | `L2048_fp16w_npu`, compiled on the phone | stopped | — | — | — | — | — | — | — | — | — | stopped after 101 | 6,814,564 | 2,600 / 2,600 |
| 2048 | `L2048_fp16w_npu_SM8850`, SM8850 file | 16 | 15/15 | 0/1 | 0.0094 | 0.0016 | 15/15 | 876.78 (capped) | 882.79 (capped) | 878.24 (capped) | — | 1.20 | 684,616 | 2,600 / 2,600 |

Every run that finished passes, with all 2,600 ops in one partition on the NPU. At L64, L96, L128, L256, L512, L768 and L1024, the SM8850 file and the compile on the phone give the same probabilities to every digit on all 1,125 questions. Loading an SM8850 file took 0.19 to 1.20 s, with a peak of 537,576 to 684,616 kB. Compiling on the phone took 9.42 to 314.70 s, with a peak of 2,735,684 to 4,882,904 kB. At L2048, 101 s into the compile on the phone, the memory watch of these checks stopped it: it stops a run when the phone's available memory falls below 1,500,000 kB, and the process had reached 6,814,564 kB. Whether that compile finishes without the watch was not tried, so here L2048 runs on this NPU from its SM8850 file only.

a, b, c and d are `readme_request` (60 tokens), `default_request` (85 tokens), `long_request` (1,024 tokens) and the typed-decisions test row `customer_service_000006` (586 tokens). Each number is the median of 20 calls after 5 warm-up calls in one process: a, b and c on each file that holds them except the L768 files, d on the L768 files only. The rows compiled on the phone were timed from LiteRT's cache of that compile. "(capped)" marks a run whose CPU clock samples show a cap (L2048). \* The L512 row compiled on the phone was timed in an earlier session, with the same program and libraries. The L768 rows ran with another build of the same C++ program.

All of these runs set the HTP performance mode to Burst. Without that option, the same SM8850 files take 5.47 to 6.74 times as long, with the same probabilities on the timed request to every digit:

| Request | File | Burst (ms) | No option (ms) | No option / Burst |
|---|---|---:|---:|---:|
| `readme_request` (60 tokens) | `L64_fp16w_npu_SM8850` | 11.33 | 76.29 | 6.74 |
| `default_request` (85 tokens) | `L96_fp16w_npu_SM8850` | 16.11 | 93.66 | 5.81 |
| `long_request` (1,024 tokens) | `L1024_fp16w_npu_SM8850` | 205.15 | 1,122.66 | 5.47 |

Each number is the mean of the medians of two runs of 20 calls, in the order Burst, no option, no option, Burst. The setting, as named in LiteRT 2.3.0: in Kotlin, `CompiledModel.QualcommOptions(htpPerformanceMode = CompiledModel.QualcommOptions.HtpPerformanceMode.BURST)`, set as `qualcommOptions` of `CompiledModel.Options`; in C, `LrtQualcommOptionsSetHtpPerformanceMode(options, kLiteRtQualcommHtpPerformanceModeBurst)`.

## The Pixel 11 Pro's NPU (Google Tensor)

The `_fp16w_tensor` files run on this NPU; LiteRT compiles each for the NPU on the phone when it loads it. In these checks this NPU computed in float16, and results below 2^-14 became zero: the module dump that the phone's NPU service printed while compiling a test graph shows float16 activations, and a simulation of a norm with that flush matches the phone, while the same simulation without the flush does not. So in the other forms the squares of small values in the norms were lost and the answers moved. The `mean_norm` form computes each norm on the row divided by 4 times its mean absolute value, which keeps those squares in range (`conversion/README.md`, "The forms"). Each file runs the requests of the CPU and GPU rows of its length.

| File | Options | Questions | Same winner | Near ties kept | Max \|Δp\| | Mean \|Δp\| | Control | Same winner as the Mac CPU | Result |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `L64_fp16w_tensor` | none | 71 | 68/68 | 3/3 | 0.0110 | 0.0015 | — | 68/68 | Pass |
| `L96_fp16w_tensor` | none | 278 | 267/267 | 11/11 | 0.0110 | 0.0016 | 0.0390 | 267/267 | Pass |
| `L128_fp16w_tensor` | none | 322 | 305/305 | 17/17 | 0.0110 | 0.0016 | 0.0390 | 305/305 | Pass |
| `L256_fp16w_tensor` | none | 61 | 54/54 | 7/7 | 0.0055 | 0.0013 | — | 54/54 | Pass |

Every run is one partition with all ops on this NPU. The `_fp16w_npu` files of the Qualcomm NPU do not pass here: the same winner on L64 62 of 68, L96 244 of 267, L128 274 of 305, L256 49 of 54 questions that are not near ties, and differences up to 0.1762. Two graphs of 512 tokens, the published `L512_fp16w_npu` file and one other form, were stopped by the memory watch before they answered, so the files for this NPU stop at 256 tokens.

## The Python host

`host/pocket_dec_litert.py` ran from this repository's files in a new virtual environment (Python 3.14.6, `pip install -r host/requirements.txt` and nothing else), on the Mac, against the reference. It chooses the smallest `_fp16w` file that holds the request: L64 for `readme_request`, L96 for `default_request`, L768 for the typed-decisions test row `customer_service_000006` and L1024 for `long_request`. Each request's token IDs equal the original code's, and the embedding table it reads equals the original checkpoint's bytes.

| Runtime | Request | Tokens | File | Same winner | Max \|Δp\| |
|---|---|---:|---|---:|---:|
| CPU, 4 threads | `examples/readme_request.json` | 60 | `L64_fp16w` | 1/1 | 1.2e-4 |
| CPU, 4 threads | `examples/default_request.json` | 85 | `L96_fp16w` | 3/3 | 3.3e-4 |
| CPU, 4 threads | typed-decisions test row `customer_service_000006` | 586 | `L768_fp16w` | 5/5 | 5.1e-4 |
| CPU, 4 threads | `examples/long_request.json` | 1,024 | `L1024_fp16w` | 3/3 | 2.0e-4 |
| GPU (Metal), float32 | `examples/readme_request.json` | 60 | `L64_fp16w` | 1/1 | 1.2e-4 |
| GPU (Metal), float32 | `examples/default_request.json` | 85 | `L96_fp16w` | 3/3 | 3.3e-4 |
| GPU (Metal), float32 | typed-decisions test row `customer_service_000006` | 586 | `L768_fp16w` | 5/5 | 5.1e-4 |
| GPU (Metal), float32 | `examples/long_request.json` | 1,024 | `L1024_fp16w` | 3/3 | 2.0e-4 |

`examples/run_examples.py --check` runs the same comparison with `examples/*.expected.json`, the original code's answers.

The host takes up to 2,048 tokens, the longest graph. A request of exactly 2,048 tokens ran on L2048 on both runtimes (largest difference 2.0e-4, same winners). A request of 2,049 tokens raises `ValueError`, while the original code answers it: its own limit is 32,768 tokens, above which it raises `ValueError` too. The host never truncates. On 7 malformed requests (an unknown question type, choice criteria that are not an object, an unsupported JSON value, `questions` not an object, empty `questions`, a temperature of zero, a boolean temperature) the host raises the same exception type with the same message as the original code.

## Timing

A call is: write the four inputs, run, read back `hidden`. Each number is the median of 20 calls after 5 warm-up calls in one process, except the median of 8 calls after 2 warm-up calls in the first table below, in the Pixel 11 Pro's row of the second, and in the README's `long_request` (1,024 tokens) numbers on the Galaxy S26's GPU at FP32, its GPU at FP16 with FP32 accumulation, the Pixel 11 Pro's GPU at FP32 and its CPU at 5 threads.

- Mac: through the Python `CompiledModel`, with no other model job running. The host's own time (tokenize, build the inputs, head) is measured apart.
- Phones: the C++ program copies the inputs into the locked input buffers, calls `LiteRtRunCompiledModel` and copies `hidden` out. CPU and GPU clocks were sampled every 1 to 3 seconds while each run ran. A run whose samples show a CPU cap or a raised GPU thermal level is marked as capped.

The typed-decisions test row `customer_service_000006` (586 tokens) on the `L768_fp16w` file, through the phones' CPU and GPU. Each runtime ran it two or three times; the table shows the fastest run without a cap, or the fastest run, marked "(capped)", when every run shows one.

| Phone | Runtime | File | d (ms) |
|---|---|---|---:|
| Galaxy S26 | CPU, 4 threads | `L768_fp16w` | 673.48 (capped) |
| Galaxy S26 | GPU, FP32 | `L768_fp16w` | 352.43 |
| Galaxy S26 | GPU, FP16 with FP32 accumulation | `L768_fp16w` | 247.84 |
| Pixel 11 Pro | CPU, 5 threads | `L768_fp16w` | 1,900.88 |
| Pixel 11 Pro | GPU, FP32 | `L768_fp16w` | 716.41 |

The Pixel 11 Pro's GPU at FP16 with FP32 accumulation is left out: it does not pass on that phone (above).

d on the L1024 and L768 files, one run after the other in the order L1024, L768, L768, L1024 (A1, B1, B2, A2) on each runtime:

| Where | Runtime | A1: L1024 (ms) | B1: L768 (ms) | B2: L768 (ms) | A2: L1024 (ms) | B1 / A1 − 1 | B2 / A2 − 1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Mac | CPU, 4 threads | 392.63 | 222.24 | 221.45 | 382.04 | −43.40 % | −42.03 % |
| Mac | GPU (Metal), float32 | 89.76 | 60.34 | 60.05 | 89.66 | −32.78 % | −33.03 % |
| Galaxy S26 | NPU, SM8850 file | 204.21 (capped) | 140.30 (capped) | 140.07 | 205.10 (capped) | −31.30 % | −31.71 % |
| Pixel 11 Pro | GPU, FP32 | 1,032.34 | 718.48 | 716.41 | 1,033.25 | −30.40 % | −30.66 % |

On each runtime both pairs move the same way, by more than 3 % and by more than its two L1024 runs differ. The Metal row is a second session: in the first, the second L1024 run took 47 % longer than the first.

The Galaxy S26's NPU numbers are in its table above, and the other CPU and GPU numbers are in the README.

The script that ran the full set of checks and the phones' C++ program are not in this repository. The requests and the bar above define the checks.
