# Qwen3.5-35B-A3B on LiteRT-LM: the text decoder with each layer's experts as one `moe` op

[Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) (revision `59d61f3c`, Apache License 2.0) is
Qwen's hybrid MoE model: 40 layers in the layout 10 × (3 × Gated DeltaNet + 1 × gated attention), each followed by a
mixture of 256 experts of which 8 routed and 1 shared are used for each token; 35B parameters, 3B of them activated
(source model card). This directory converts its text decoder with litert-torch's `export_hf` (main `67f11e89`) and an
out-of-tree extension, `q35moe_ext.py`, that writes each layer's experts as one TFLite custom op `moe`. The result is
the CPU file of [litert-community/Qwen3.5-35B-A3B](https://huggingface.co/litert-community/Qwen3.5-35B-A3B),
`qwen35_35b_a3b_mixed_int4_cpu_ekv4096.litertlm` (27.61 GB).

- The Gated DeltaNet and attention layers, their caches and the signatures (`prefill_128`, `prefill_1024`, `decode`)
  are litert-torch's own Qwen3.5 export; the extension replaces the dense MLP with the MoE block. The Gated DeltaNet
  step is the custom op `gated_delta_update` (no GPU composites).
- Each layer's 256 experts are one `moe` op with SiLU: only the 8 routed experts are computed for each token. The
  router, its top 8 and their renormalization, and the shared expert are ordinary ops.
- Weights: experts int4 blockwise (block 32, one fp32 scale per block, scales below 2^-14 raised to 2^-14), attention
  q / k / v / o and lm_head int8 per channel, every other weight fp32 (the Gated DeltaNet projections, the shared
  expert and its gate, the router, the token table). Activations, the KV cache and the recurrent and convolution
  states are fp32; KV 4,096 tokens.
- The bundle carries LiteRT-LM's own `models/qwen3_5/LlmMetadataProto.pbtext` at v0.19.1 verbatim, the checkpoint's
  own `tokenizer.json` bytes and `export_hf`'s executor metadata, and declares fp32 activations.

## The runtime it needs

LiteRT-LM 0.19.1, the latest release on 2026-11-10, does not run this file. The `moe` kernels of LiteRT-LM 0.19.1
accept only GELU: its PyPI package stops on the CPU with `moe node #94 unsupported activation='silu'` and on the GPU
with `moe only supports activation='gelu_tanh' (or its legacy spelling 'gelu')`. SiLU in the `moe` op is in two LiteRT
pull requests, open on that day: [#11192](https://github.com/google-ai-edge/LiteRT/pull/11192) (the GPU kernel) and
[#11188](https://github.com/google-ai-edge/LiteRT/pull/11188) (the XNNPACK kernel). Every check below ran on
LiteRT-LM `02d7e8d8` (main, 2026-11-07) built with the LiteRT commit its WORKSPACE pins, `bf49eb62`, and the two pull
requests applied without their test files (one test file of #11192 does not apply at `bf49eb62`; no test is built):

```bash
git clone https://github.com/google-ai-edge/LiteRT-LM.git && cd LiteRT-LM
git checkout 02d7e8d80cb75cbb03d8cba4c63f51e2eb318ec5
git lfs pull --include "prebuilt/macos_arm64/*"
cd ..
git clone https://github.com/google-ai-edge/LiteRT.git && git -C LiteRT checkout bf49eb62406d15948261a840c9dcd8e4f13b478c
python3 path/to/hf-to-litertlm/qwen3_30b_a3b_work/apply_ws_patch_cmds.py LiteRT-LM/WORKSPACE litert LiteRT   # GNU sed as gsed
curl -L https://github.com/google-ai-edge/LiteRT/commit/ebacd381a0d223a585ec66c1e880cffa640b29be.diff | git -C LiteRT apply --exclude='*_test.cc'
curl -L https://github.com/google-ai-edge/LiteRT/commit/9860c14d7b63f6eaa85b9bfbe807afc971b85fe4.diff | git -C LiteRT apply --exclude='*_test.cc'
cd LiteRT-LM
bazel build --config=macos_arm64 -c opt --define=litert_runtime_link_mode=dynamic \
  --override_repository=litert=$PWD/../LiteRT \
  //runtime/engine:litert_lm_advanced_main @litert//litert/c:litert_runtime_c_api_dylib
```

`--override_repository` skips the patch commands that LiteRT-LM's WORKSPACE gives the LiteRT archive, so
`apply_ws_patch_cmds.py` applies them by hand. The two commits are the pull requests' heads on 2026-11-10. Put
`litert_lm_advanced_main` and `libLiteRt.dylib` from `bazel-bin` in one directory with `libwebgpu_dawn.dylib` and
`libGemmaModelConstraintProvider.dylib` from `prebuilt/macos_arm64/`: the CLI links the last two and does not start
without them, also on the CPU.

```bash
./litert_lm_advanced_main --backend=cpu --model_path=qwen35_35b_a3b_mixed_int4_cpu_ekv4096.litertlm \
  --max_output_tokens=2048 --sampler_top_k=1 --cache_dir=:nocache --input_prompt="What is 17 + 25? Answer briefly."
```

## Files

| file | what it does |
|---|---|
| `export_q35.py` | runs `export_hf` with the extension: fp32, `plain` (no GPU composites) or `gpu` (`apply_gpu_composites`), the prefill lengths and the KV length as arguments; skips `export_hf`'s own bundle and keeps its executor metadata. |
| `q35moe_ext.py`, `q35moe_ext_layout.py` | the extension: the MoE block, the `moe` op (SiLU, the expert weights in the op's layout or 4-element placeholders for the graft), the router as ordinary ops; its environment switches are in its docstring. |
| `quant_tflite.py`, `recipes/fc8_attn_lmhead.json` | the nightly quantizer on the float export: int8 per channel for attention q / k / v / o and lm_head, every other op untouched. |
| `graft_moe.py`, `q35_quant_common.py` | replaces each `moe` op's placeholders with the experts quantized int4 blockwise from the bf16 checkpoint, one projection of one layer at a time, with the quantizer's own functions. |
| `make_bundle.py` | the `.litertlm`: the official metadata, the tokenizer bytes (sha256 checked), the executor metadata, fp32 activations declared. |
| `reproduce.sh`, `requirements.txt` | the steps below in one script, and the Python packages they ran with. |

## Steps

The Python steps ran with Python 3.12 in the environment that `requirements.txt` lists: all 79 packages, frozen
from it, litert-torch's own dependencies included (jax, torchvision, torchaudio, torchao, litert-lm, ...). litert-torch
itself is installed from source at `67f11e89`, with `--no-deps` so that pip keeps those versions:

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
git clone https://github.com/google-ai-edge/litert-torch.git
git -C litert-torch checkout 67f11e89e7ca14ac222638c88e707f8a2e755273
pip install --no-deps -e litert-torch
bash reproduce.sh WORK_DIR        # KV 4096, block 32: the published file's computation (REPRODUCE.md compares the bytes)
```

`reproduce.sh` runs, in order (peak memory about 34 GB; disk about 67 GB for the checkpoint plus 35 GB at the
peak: it deletes the export's `model.tflite` after step 4 and the int8 file after step 5):

1. LiteRT-LM's `models/qwen3_5/LlmMetadataProto.pbtext` at v0.19.1 into `template/` (sha256 `2bdb4bcb…`, checked).
2. The checkpoint at revision `59d61f3ce65a6d9863b86d2e96597125219dc754` (or `SNAP=<snapshot dir>`).
3. `PYTHONHASHSEED=0 Q35MOE_EXPERTS=graft python3 export_q35.py SNAPSHOT WORK/export plain 128,1024 4096`
4. `python3 quant_tflite.py WORK/export/model.tflite WORK/q8.tflite recipes/fc8_attn_lmhead.json`
5. `python3 graft_moe.py --tflite WORK/q8.tflite --ckpt SNAPSHOT --out WORK/model.tflite --bits 4 --block 32`
6. `python3 make_bundle.py --tflite WORK/model.tflite --out WORK/qwen35_35b_a3b_mixed_int4_cpu_ekv4096.litertlm --tokenizer SNAPSHOT/tokenizer.json --executor-metadata WORK/export/executor_metadata.pb`

## How it was checked

- **The `moe` op against Transformers, before the whole model.** The first four layers of the real weights with fp32
  experts (`Q35MOE_EXPERTS=fp32`): three prompts through the decode signature (540 positions) and through both prefill
  signatures (228 positions after the prefill). The CPU picks Transformers' top-1 token at every position with a
  largest logit difference of 5.0e-5, and each Gated DeltaNet layer's recurrent and convolution states match
  Transformers' to a cosine of 1.0 (largest difference 1.8e-5).
- **The published file.** 12 questions and a 3-turn conversation with thinking on, greedy, up to 2,048 tokens per
  answer, judged against Transformers (the bf16 checkpoint computed in fp32 on the CPU) by the same regular
  expression on the answer after the thought: 12 / 12 with the same verdict, 12 / 12 ending with the stop token, the 3
  turns passed, on the CPU of the runtime above.
- **Speed.** Apple M4 Max (128 GB), `--benchmark --benchmark_prefill_tokens=1024 --benchmark_decode_tokens=256
  --num_iterations=2 --cache_dir=:nocache --max_num_tokens=4096`, iteration 2: decode 5.83 tokens/s, prefill 72.46
  tokens/s, peak physical footprint 17.18 GB.

## What did not ship

- **More int8 layers.** With every linear layer of at least 524,289 weights and the token table in int8 (the
  Gated DeltaNet projections and the shared expert included), the thought on "What is the capital of Japan? Answer
  briefly." repeated itself up to the 2,048-token limit on the CPU, where Transformers ends with the stop token at 222
  tokens; the published file keeps those layers fp32 and ends at 279.
- **A GPU file.** `export_q35.py ... gpu` (export_hf's GPU form: the Gated DeltaNet as the composite
  `odml.gated_delta_block`, fused projections; the shared expert on the rank-3 input, `Q35MOE_SHARED_3D=1`, without
  which the prefill graphs do not build on WebGPU) builds and generates on WebGPU with the runtime above, but its
  quantized files did not pass: over 24 checked positions they picked Transformers' top token at 21 (the conversion
  asks for 22), and the thought looped on 2 of the 12 questions with int8 only in attention, lm_head and the token
  table. Every GPU-form file checked had its token table in int8, which the published CPU file keeps in fp32. On a
  random 2-layer model with the real dimensions, in fp32, the GPU form's logits are as close to Transformers' as the
  plain graph's (largest difference 1.8e-5 for both).
