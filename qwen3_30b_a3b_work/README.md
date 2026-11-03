# Qwen3-30B-A3B on LiteRT-LM: a 48-layer MoE decoder written with the Tensor API

[Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) (revision `ad44e777`, Apache License 2.0) is a
mixture-of-experts model: 48 layers, 128 experts per layer, 8 of them used for each token, 30.5 B parameters of which
3.3 B are active per token. This directory writes its decoder straight from the bf16 checkpoint with the
[LiteRT Tensor API](https://github.com/google-ai-edge/LiteRT/tree/main/tensor), one tensor at a time, and packs it into
a `.litertlm` that LiteRT-LM's stock executor runs (`litert_lm_advanced_main`, the C++ CLI).

- Each layer's experts are one TFLite custom op `moe`, the op that LiteRT's XNNPACK delegate and GPU delegate run:
  only the 8 routed experts are computed for each token. The router, its softmax, top-8 and the renormalization of the
  top-8 weights are ordinary ops in the graph.
- Weights: experts int4 blockwise (block 32, one fp32 scale per block: the form LiteRT #11309's quantizer writes for
  this op), attention q / k / v / o and lm_head int8 per channel, router, norms and the token table fp32. Activations
  and the KV cache are fp32.
- Signatures `prefill_128`, `prefill_1024` and `decode`, a KV cache of 4,096 or 2,048 tokens (two files).
- The bundle carries LiteRT-LM's official Qwen3 configuration verbatim (`models/qwen3` at LiteRT-LM `dab2e708`: the
  chat template with thinking on by default, the stop token `<|im_end|>`, the `qwen3` model type, a `thought`
  channel), the checkpoint's own `tokenizer.json` bytes, and declares fp32 activations.

Why the Tensor API: the converter path would export this model in fp32 first (about 122 GB, extrapolated from the
first two layers: 7.47 GB) and quantize the experts in a second step; the writer quantizes one tensor at a time and
never holds an fp32 graph (its peak memory was 2.6 GB for the pack and 0.1 GB for the writer).

## The runtime it needs

The experts use SiLU. The `moe` op of LiteRT accepts only GELU so far: LiteRT-LM 0.19.0 (the latest release on
2026-11-03, PyPI and GitHub) refuses these files on the CPU (`moe node #85 unsupported activation='silu'`) and on the
GPU (`moe only supports activation='gelu_tanh' (or its legacy spelling 'gelu')`), and LiteRT's main branch has the same
checks on that day. SiLU support is in two open pull requests of the LiteRT repository: #11192 (the GPU delegate's moe
kernel) and #11188 (the XNNPACK kernel). Until a LiteRT-LM release includes them, build LiteRT-LM with them:

```bash
git clone https://github.com/google-ai-edge/LiteRT-LM.git && cd LiteRT-LM
git checkout dab2e708dc5750549685692863039c5a954c3a35
git lfs pull --include "prebuilt/macos_arm64/*"
cd ..
git clone https://github.com/google-ai-edge/LiteRT.git && git -C LiteRT checkout dcb657e8a537b2dc944e53e14c7eb43bcd9ededa
git clone https://github.com/google-ai-edge/ml-drift.git && git -C ml-drift checkout 76492bb9412030ec93cc5b607df26ab5cf787490
# --override_repository skips the patch_cmds that LiteRT-LM's WORKSPACE gives these two archives: apply them by hand
# (GNU sed as `sed`), then the two pull requests
python3 apply_ws_patch_cmds.py LiteRT-LM/WORKSPACE litert LiteRT
python3 apply_ws_patch_cmds.py LiteRT-LM/WORKSPACE ml_drift ml-drift
gh pr diff 11192 -R google-ai-edge/LiteRT | git -C LiteRT apply
gh pr diff 11188 -R google-ai-edge/LiteRT | git -C LiteRT apply
cd LiteRT-LM
bazel build --config=macos_arm64 -c opt --define=litert_runtime_link_mode=dynamic \
  --override_repository=litert=$PWD/../LiteRT --override_repository=ml_drift=$PWD/../ml-drift \
  //runtime/engine:litert_lm_advanced_main @litert//litert/c:litert_runtime_c_api_dylib \
  @litert//litert/runtime/accelerators/gpu:ml_drift_webgpu_accelerator_dylib \
  @litert//litert/runtime/accelerators/gpu:ml_drift_metal_accelerator_dylib
```

Put `litert_lm_advanced_main`, `libLiteRt.dylib`, `libLiteRtWebGpuAccelerator.dylib` (and
`libLiteRtMetalAccelerator.dylib`) from `bazel-bin` in one directory with `libwebgpu_dawn.dylib` and
`libGemmaModelConstraintProvider.dylib` from `prebuilt/macos_arm64/`; the CLI does not start without the last two,
also on the CPU. These are the steps of the build that ran every check below (on a 128 GB Mac, 2026-10-28); they were
not run again for this page. On 2026-11-03 `gh pr diff` gave the same two diffs as on 2026-10-28 (sha256
`9b413402…` for #11192, `6a996ece…` for #11188).

Run (the GPU here is WebGPU on the Mac; with `--sampler_handles_input=false` the GPU sampler does not write the
decode inputs itself, the setting every GPU check below used):

```bash
./litert_lm_advanced_main --backend=gpu --model_path=qwen3_30b_a3b_mixed_int4_ekv4096.litertlm \
  --sampler_handles_input=false --input_prompt="What is 17 + 25? Answer briefly."
```

## Files

| file | what it does |
|---|---|
| `tapi_pack_moe.py` | reads the checkpoint one tensor at a time and writes `weights.bin` + `index.tsv` + `config.tsv` + `moe_options.bin`: experts stacked in the op's layout and quantized with ai-edge-quantizer's own functions (int4 blockwise, scales raised to at least 2^-14), attention and lm_head int8 per channel. |
| `scale_scan.py` | counts zero and non-finite scales in a pack (0 of each is the condition). |
| `graph/qwen3moe_writer.cc`, `graph/BUILD` | the Tensor API graph: `prefill_<N>` and `decode` signatures, KV cache in and out per layer (V stored transposed), q / k norms, RoPE from `input_pos`, grouped-query attention over the whole cache with the runtime's mask, the router and the `moe` op, `ModelFactory::Save` with the weights mapped from `weights.bin`. |
| `patches/blockwise_named_scales.diff` | lets the Tensor API (LiteRT `dcb657e8`) write a blockwise-quantized tensor whose scales are an existing tensor of the graph, with a block shape (the `moe` op reads the scales as an input). |
| `overlay.sh` | applies the patch to a LiteRT checkout and links `graph/` into it as `tensor/examples/qwen3moe`. |
| `make_bundle.py`, `template/qwen3_official_dab2e708.pbtext` | the `.litertlm`: the official Qwen3 metadata (sha256 `f132370a…`), the tokenizer, the decoder, fp32 activations declared; `--ids-from` copies the header's uuid and creation time from another file. |
| `bundle_sections.py` | prints a `.litertlm`'s sections with their sha256, its uuid / creation time and the whole-file sha256. |
| `apply_ws_patch_cmds.py` | applies the `patch_cmds` of one archive of a bazel WORKSPACE to a local checkout (for the runtime build above). |

## Steps

The Python steps ran with Python 3.12, torch 2.14.1, safetensors 0.8.0, numpy 2.5.4, ai-edge-quantizer 0.10.0,
flatbuffers 25.12.19, protobuf 7.36.2 and litert-lm-builder 0.19.0.

1. The checkpoint (61.1 GB):

   ```bash
   hf download Qwen/Qwen3-30B-A3B --revision ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 --local-dir SNAPSHOT
   ```

2. Pack the weights (20.6 GB; 3 to 11 minutes depending on the disk, peak memory 2.6 GB) and check the scales:

   ```bash
   python tapi_pack_moe.py SNAPSHOT PACK --experts xb4 --block 32 --fc int8 --lm-head int8
   python scale_scan.py PACK scale_scan.json
   ```

3. Build the writer in a LiteRT checkout at `dcb657e8` (the LiteRT that LiteRT-LM 0.19.0 pins):

   ```bash
   git clone https://github.com/google-ai-edge/LiteRT.git && git -C LiteRT checkout dcb657e8a537b2dc944e53e14c7eb43bcd9ededa
   bash overlay.sh LiteRT
   (cd LiteRT && bazel build -c opt //tensor/examples/qwen3moe:qwen3moe_writer)
   ```

4. Write the decoder (20.6 GB). `--moe_fusion_guard` keeps the GPU delegate from fusing the residual ADD into the
   moe kernel, which otherwise emits invalid WGSL / Metal code (it adds `+ moe * 0`, exact):

   ```bash
   W=LiteRT/bazel-bin/tensor/examples/qwen3moe/qwen3moe_writer
   $W --weights_dir=PACK --out=model_kv4096.tflite --prefill=128,1024 --cache=4096 --moe_fusion_guard
   $W --weights_dir=PACK --out=model_kv2048.tflite --prefill=128,1024 --cache=2048 --moe_fusion_guard
   ```

5. The bundles:

   ```bash
   python make_bundle.py --tflite model_kv4096.tflite --tokenizer SNAPSHOT/tokenizer.json \
     --out qwen3_30b_a3b_mixed_int4_ekv4096.litertlm
   python make_bundle.py --tflite model_kv2048.tflite --tokenizer SNAPSHOT/tokenizer.json \
     --out qwen3_30b_a3b_mixed_int4_ekv2048.litertlm --max-tokens 2048
   python bundle_sections.py qwen3_30b_a3b_mixed_int4_ekv2048.litertlm
   ```

## How it was checked

- **The op contract, on a tiny model.** A random Qwen3-MoE (3 layers, 16 experts, top-8) written by the same pack and
  writer: on the CPU the logits equal Transformers' (fp32) within 1.8e-6 with fp32 experts, and within 1.9e-6 of
  Transformers run on the dequantized weights with int4 / int8 experts; the 32-token greedy text is the same. Without the fusion guard the WebGPU run prints `Validation error` 75 times and
  generates token 0 only; with it, WebGPU and Metal give the CPU's text.
- **The real weights, first two layers.** Layers 0-1 + final norm + lm_head, fp32 experts, 185 positions of four
  prompts: the CPU picks Transformers' top-1 at 185 / 185 positions, largest logit difference 2.7e-5, and the router
  picks the same 8 experts at every position of both layers; WebGPU (fp32) 185 / 185, 4.0e-5. Quantized as shipped
  (int4 experts, int8 attention and lm_head), the two layers equal Transformers run on the dequantized weights:
  185 / 185 positions, 4.5e-5 on WebGPU.
- **The whole model.** 12 questions (8 short questions from this repository's `scripts/verify_quality.py`, a word
  problem, a Python function, an ordered list and a Japanese-to-English sentence) and a 3-turn conversation whose third
  turn asks for a number given in the first, greedy, up to 2,048 tokens per answer, thinking on, against Transformers
  with the bf16 weights: on WebGPU both files, and on the CPU the 4,096-token file, give the same verdict as
  Transformers on 12 of 12 questions, end every answer with `<|im_end|>`, and pass the 3-turn check. On WebGPU the
  2,048-token file's 12 answers are byte for byte the 4,096-token file's.
- **What did not ship.** fp16 activations (7 to 15 % faster decoding on WebGPU): the greedy thinking loops on a
  sentence and reaches the token limit without an answer (every one of the 8 questions run on the two files). int4
  attention and lm_head (with 128-wide expert blocks): no faster, and one of the 12 answers loops to the limit.
  Experts with 128-wide blocks: slower than 32-wide blocks.

Speeds: the model card.
