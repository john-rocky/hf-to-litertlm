# Qwen3.6-35B-A3B on LiteRT-LM: the text decoder with each layer's experts as one `moe` op, int8 experts

[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (revision `345d467b`, Apache License 2.0) is Qwen's hybrid MoE model: 40 layers in the layout 10 × (3 × Gated DeltaNet + 1 × gated attention), each followed by a mixture of 256 experts of which 8 routed and 1 shared are used for each token; 35B parameters, 3B of them activated (source model card). Its text configuration is the same as Qwen3.5-35B-A3B's (model type `qwen3_5_moe`), so this directory is [`../qwen35_35b_a3b_work/`](../qwen35_35b_a3b_work/) with other weights: it converts the text decoder with litert-torch's `export_hf` (main `67f11e89`) and the out-of-tree extension `q35moe_ext.py`, which writes each layer's experts as one TFLite custom op `moe`. The results are the three files of [litert-community/Qwen3.6-35B-A3B](https://huggingface.co/litert-community/Qwen3.6-35B-A3B):

- `qwen36_35b_a3b_mixed_int8_cpu_ekv4096.litertlm` (42.15 GB): `export_hf` without GPU composites; the Gated DeltaNet step is the custom op `gated_delta_update`.
- `qwen36_35b_a3b_int8_gpu_ekv4096.litertlm` (36.43 GB): the GPU export below with every linear layer of at least 524,289 weights int8 per channel.
- `qwen36_35b_a3b_mixed_int8_gpu_ekv4096.litertlm` (42.15 GB): `export_hf` with GPU composites (`apply_gpu_composites`); the Gated DeltaNet step is `odml.gated_delta_block` after `odml.short_conv_step`, and each attention layer is one fused linear layer for q (with the attention's output gate), k and v, then `odml.qkv_norm_rope`, `odml.cache_update` and `odml.sdpa_transposed`. The export loads the q / k norm weights unpermuted, so that the norms get litert-torch's partial-rotary permutation once (`export_q35.py`, `install_norm_perm_once`).

In all three:

- The Gated DeltaNet and attention layers, their caches and the signatures (`prefill_128`, `prefill_1024`, `decode`) are litert-torch's own Qwen3.5 export; the extension replaces the dense MLP with the MoE block.
- Each layer's 256 experts are one `moe` op with SiLU: only the 8 routed experts are computed for each token. The router, its top 8 and their renormalization, and the shared expert are ordinary ops.
- Weights: the experts int8 per channel (one fp32 scale for each output row of each expert). In the CPU file and the mixed int8 GPU file every other weight is fp32 (attention, the Gated DeltaNet projections, the shared expert and its gate, the router, lm_head, the token table). The int8 GPU file also has every linear layer of at least 524,289 weights int8 per channel (attention, the Gated DeltaNet projections, the shared expert, lm_head; litert-quantizer's min-max, `recipes/fc8_gpu_int8.json`); the router (256 × 2,048), the shared expert's gate (1 × 2,048) and the token table stay fp32. The KV cache and the recurrent and convolution states are fp32; KV 4,096 tokens. The CPU file and the mixed int8 GPU file ask the runtime for fp32 activations, the int8 GPU file for fp16.
- The bundle carries LiteRT-LM's own `models/qwen3_5/LlmMetadataProto.pbtext` at v0.19.1 verbatim, the checkpoint's own `tokenizer.json` bytes and `export_hf`'s executor metadata, and declares the activation type (`make_bundle.py --act`).

## The runtime it needs

LiteRT-LM 0.19.2, the latest release on PyPI on 2026-11-13, does not run these files. The `moe` kernels of LiteRT-LM 0.19.2 accept only GELU: its PyPI package stops each file on the CPU (`moe node #94 unsupported activation='silu'` for the CPU file, `moe node #39 unsupported activation='silu'` for the GPU files) and the GPU files on the GPU (the delegate refuses the `moe` op, `moe only supports activation='gelu_tanh' (or its legacy spelling 'gelu')`, and `odml.gated_delta_block`); 0.19.1 stopped the CPU file on the CPU with `moe node #94 unsupported activation='silu'` and on the GPU with the `moe` message. SiLU in the `moe` op is in two LiteRT pull requests, open on 2026-11-13: [#11192](https://github.com/google-ai-edge/LiteRT/pull/11192) (the GPU kernel) and [#11188](https://github.com/google-ai-edge/LiteRT/pull/11188) (the XNNPACK kernel). Every check below ran on LiteRT-LM `02d7e8d8` (main, 2026-11-07) built with the LiteRT commit its WORKSPACE pins, `bf49eb62`, and the two pull requests applied without their test files (one test file of #11192 does not apply at `bf49eb62`; no test is built):

```bash
git clone https://github.com/google-ai-edge/LiteRT-LM.git && cd LiteRT-LM
git checkout 02d7e8d80cb75cbb03d8cba4c63f51e2eb318ec5
git lfs pull --include "prebuilt/macos_arm64/*"
cd ..
git clone https://github.com/google-ai-edge/LiteRT.git && git -C LiteRT checkout bf49eb62406d15948261a840c9dcd8e4f13b478c
curl -fLO https://raw.githubusercontent.com/john-rocky/hf-to-litertlm/84336e8d00a567464e834854bdc1987a5c6656ca/qwen3_30b_a3b_work/apply_ws_patch_cmds.py
python3 apply_ws_patch_cmds.py LiteRT-LM/WORKSPACE litert LiteRT
curl -L https://github.com/google-ai-edge/LiteRT/commit/ebacd381a0d223a585ec66c1e880cffa640b29be.diff | git -C LiteRT apply --exclude='*_test.cc'
curl -L https://github.com/google-ai-edge/LiteRT/commit/9860c14d7b63f6eaa85b9bfbe807afc971b85fe4.diff | git -C LiteRT apply --exclude='*_test.cc'
cd LiteRT-LM
bazel build --config=macos_arm64 -c opt --define=litert_runtime_link_mode=dynamic \
  --override_repository=litert=$PWD/../LiteRT \
  //runtime/engine:litert_lm_advanced_main @litert//litert/c:litert_runtime_c_api_dylib \
  @litert//litert/runtime/accelerators/gpu:ml_drift_webgpu_accelerator_dylib
```

Run the block in an empty directory. `--override_repository` skips the patch commands that LiteRT-LM's WORKSPACE gives the LiteRT archive, so `apply_ws_patch_cmds.py` (fetched by the block from this repository's `qwen3_30b_a3b_work/`) applies them by hand; it needs GNU sed installed as `gsed`. The block also needs Git LFS and Bazel (LiteRT-LM's `.bazelversion` names 7.6.1, which Bazelisk runs); it ran here with Xcode 27.0 on macOS 27.0. The two commits are the pull requests' heads on 2026-11-13 (each pull request has one commit): `ebacd381` is #11192 and `9860c14d` is #11188. For the CPU, copy `bazel-bin/runtime/engine/litert_lm_advanced_main` and `bazel-bin/external/litert/litert/c/libLiteRt.dylib` into one directory with `prebuilt/macos_arm64/libwebgpu_dawn.dylib` and `prebuilt/macos_arm64/libGemmaModelConstraintProvider.dylib` (paths from `LiteRT-LM`, where the block ends): the CLI links the last two and does not start without them, also on the CPU. For the GPU, also copy `bazel-bin/external/litert/litert/runtime/accelerators/gpu/libLiteRtWebGpuAccelerator.dylib` (the third target) into that directory. Put the file for your backend in the same directory and run its command there:

```bash
./litert_lm_advanced_main --backend=gpu --model_path=qwen36_35b_a3b_int8_gpu_ekv4096.litertlm \
  --max_output_tokens=2048 --sampler_top_k=1 --cache_dir=:nocache --sampler_handles_input=false \
  --input_prompt="What is 17 + 25? Answer briefly."
./litert_lm_advanced_main --backend=cpu --model_path=qwen36_35b_a3b_mixed_int8_cpu_ekv4096.litertlm \
  --max_output_tokens=2048 --sampler_top_k=1 --cache_dir=:nocache --num_cpu_threads=12 \
  --input_prompt="What is 17 + 25? Answer briefly."
```

## Files

| file | what it does |
|---|---|
| `export_q35.py` | runs `export_hf` with the extension: fp32, `plain` (no GPU composites) or `gpu` (`apply_gpu_composites`, the q / k norm weights loaded unpermuted), the prefill lengths and the KV length as arguments; skips `export_hf`'s own bundle and keeps its executor metadata. |
| `q35moe_ext.py`, `q35moe_ext_layout.py` | the extension: the MoE block, the `moe` op (SiLU, the expert weights in the op's layout or 4-element placeholders for the graft), the router as ordinary ops; its environment switches are in its docstring. |
| `quant_tflite.py`, `recipes/fc8_gpu_int8.json` | the int8 GPU file only: litert-quantizer (nightly) on the float export with the recipe, the `moe` ops left alone (the same script as `../qwen35_35b_a3b_work/quant_tflite.py`). |
| `graft_moe.py`, `q35_quant_common.py` | replaces each `moe` op's placeholders with the experts quantized from the bf16 checkpoint (`--bits 8 --block 0`: int8 per channel), one projection of one layer at a time, with the quantizer's own functions. |
| `make_bundle.py` | the `.litertlm`: the official metadata, the tokenizer bytes (sha256 checked), the executor metadata, the declared activation type (`--act`, fp32 unless given). |
| `reproduce.sh`, `requirements.txt` | the steps below in one script, and the Python packages they ran with. |

## Steps

The Python steps ran with Python 3.12 in the environment that `requirements.txt` lists (the same file as in `../qwen35_35b_a3b_work/`): all 79 packages, frozen from it, litert-torch's own dependencies included. litert-torch itself is installed from source at `67f11e89`, with `--no-deps` so that pip keeps those versions:

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
git clone https://github.com/google-ai-edge/litert-torch.git
git -C litert-torch checkout 67f11e89e7ca14ac222638c88e707f8a2e755273
pip install --no-deps -e litert-torch
bash reproduce.sh WORK_DIR cpu    # or gpu, or gpu_int8 (REPRODUCE.md compares the bytes with the published files)
```

`reproduce.sh` runs, in order (peak memory about 45 GB; disk about 72 GB for the checkpoint plus 52 GB at the peak: it deletes the export's `model.tflite` after step 5, and on macOS the bundle takes the grafted model's blocks over):

1. LiteRT-LM's `models/qwen3_5/LlmMetadataProto.pbtext` at v0.19.1 into `template/` (sha256 `2bdb4bcb…`, checked).
2. The checkpoint at revision `345d467b117d1fd5eda9b879a5988de9ae5f47a7` (or `SNAP=<snapshot dir>`).
3. `PYTHONHASHSEED=0 Q35MOE_EXPERTS=graft python3 export_q35.py SNAPSHOT WORK/export plain 128,1024 4096` (`gpu` for the GPU files)
4. The int8 GPU file only: `python3 quant_tflite.py WORK/export/model.tflite WORK/q8.tflite recipes/fc8_gpu_int8.json`
5. `python3 graft_moe.py --tflite WORK/export/model.tflite --ckpt SNAPSHOT --out WORK/model.tflite --bits 8 --block 0` (`WORK/q8.tflite` as `--tflite` for the int8 GPU file)
6. `python3 make_bundle.py --tflite WORK/model.tflite --out WORK/qwen36_35b_a3b_mixed_int8_cpu_ekv4096.litertlm --tokenizer SNAPSHOT/tokenizer.json --executor-metadata WORK/export/executor_metadata.pb --consume-tflite` (the GPU files' names for the GPU files; `--act fp16` for the int8 GPU file)

## How it was checked

- **The `moe` op against Transformers, before the whole model.** A cut of two real layers (layer 0, Gated DeltaNet, and layer 3, attention, each with its MoE block) with fp32 experts (`Q35MOE_EXPERTS=fp32`): three prompts through the decode signature (540 positions) and through both prefill signatures (228 positions after the prefill). The CPU picks Transformers' top-1 token at every position with a largest logit difference of 4.4e-5, and each Gated DeltaNet layer's recurrent and convolution states match Transformers' to a cosine of 1.000000 (largest difference 1.34e-5). The GPU form of the same cut on WebGPU (fp32) picks the same tokens at every position, with a largest logit difference of 6.0e-5.
- **Why int8 experts.** On the same cut, the median over 768 positions of the largest logit difference per position against Transformers was 0.865 on the CPU with int4 experts (blockwise, block 32) and int8 attention and lm_head, 0.728 with int8 experts and int8 attention and lm_head, and 0.035 with int8 experts and every other weight fp32.
- **The published files.** 12 questions and a 3-turn conversation with thinking on, greedy, up to 2,048 tokens per answer, judged against Transformers (the bf16 checkpoint computed in fp32 on the CPU) by the same regular expression on the answer after the thought: each file 12 / 12 with the same verdict and 12 / 12 ending with the stop token, the CPU file on the CPU (12 threads) and the GPU files on WebGPU; all three pass the 3 turns with `filter_channel_content_from_kv_cache` false (the model card has the default Conversation's result). Teacher-forced over the first 24 positions of one prompt, the CPU file and the mixed int8 GPU file pick Transformers' top token at 23, the int8 GPU file at 22. The int8 GPU file with its fp16 activations also copied two 40-digit codes and recalled two numbers placed past position 2,048 of a 2,600-token prompt, as the same weights did with fp32.
- **Speed.** Apple M4 Max (128 GB), `--benchmark --benchmark_prefill_tokens=1024 --benchmark_decode_tokens=256 --cache_dir=:nocache --max_num_tokens=4096`, the median of 3 processes: the int8 GPU file on WebGPU (`--backend=gpu --sampler_handles_input=false --num_iterations=4`, the mean of iterations 2 to 4 in each process) decodes 76.99 tokens/s and prefills 1,336.22 tokens/s at a peak physical footprint of 49 GB (2026-11-13); the CPU file (`--backend=cpu --num_cpu_threads=12 --num_iterations=2`, iteration 2) decodes 7.74 tokens/s and prefills 101.45 tokens/s at 20 GB (2026-11-12, while a shell left by other work kept one CPU core at about 100 %).

## What did not ship

- **int4 experts.** With the experts in int4 (blockwise, block 32, the quantization of the Qwen3.5-35B-A3B files) and attention and lm_head in int8, the file picked Transformers' top token at 21 of the 24 teacher-forced positions (the conversion asks for 22), on the CPU and on the GPU. Blocks of 16 are not offered by litert-quantizer-nightly 0.12.0.dev20261109 (`ValueError: 'BLOCKWISE_16' is not a valid QuantGranularity`). With that quantizer's block-size table patched to 16 (not in this directory), the int8 GPU file's form with int4 experts in blocks of 16 (28.25 GB) picked Transformers' top token at 22 of the 24 positions and passed the 12 questions with fp32 activations, but prefilled 1,159.77 tokens/s against the published int8 GPU file's 1,336.22 (decode 76.38 against 76.99, the same session), and picked 21 of 24 with fp16 activations.
- **A 2,048-token KV cache.** The int8 GPU file's form exported with a 2,048-token KV cache decoded 75.43 tokens/s against 75.03 with 4,096 (both with fp32 activations, the same session): not faster.
- **fp16 activations in the mixed int8 GPU file.** The mixed int8 GPU file asking for fp16 (2 bytes of its header differ) gave Transformers' verdict on 11 of the 12 questions: on "Which is larger: 0.9 or 0.11?" its thought ran to the 2,048-token limit without an answer.
