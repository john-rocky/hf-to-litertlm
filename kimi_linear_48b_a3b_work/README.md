# Kimi-Linear-48B-A3B-Instruct on LiteRT-LM: the 27-layer hybrid with each layer's experts as one `moe` op

[moonshotai/Kimi-Linear-48B-A3B-Instruct](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct) (revision `e1df551a`, MIT license) is Moonshot AI's hybrid linear-attention MoE model: 27 layers, 20 of them Kimi Delta Attention (KDA) and 7 MLA (no rotary embedding); every layer but the first is followed by a mixture of 256 experts of which 8 routed and 1 shared are used for each token, the first has a dense MLP; 48B parameters, 3B of them activated (source model card). This directory converts it with `tools/kimi_ext.py`, a graph writer for litert-torch that writes the model in plain TFLite ops plus LiteRT's `moe` custom op, checked against Transformers' own `kimi_linear` model. The result is the CPU file of [litert-community/Kimi-Linear-48B-A3B-Instruct](https://huggingface.co/litert-community/Kimi-Linear-48B-A3B-Instruct), `kimi_linear_48b_a3b_mixed_int8_cpu_ekv4096.litertlm` (55.29 GB).

- KDA: the prefill computes the chunked delta rule (chunks of 16 tokens) and the decode the exact recurrent step, with the per-channel decay of the source. Each layer's state = the delta-rule state (32 heads × 128 × 128, fp32) and the last 3 inputs of the short convolution.
- MLA: the absorbed form; the cache holds per position the normalized compressed latent (512 values) and the shared key part (64 values).
- MoE: the router in plain ops (sigmoid scores, the top 8 chosen on the scores plus the correction bias, the weights renormalized and scaled by 2.446, as in the source); each layer's 256 routed experts are one `moe` op with SiLU, so only the 8 routed experts are computed for each token; the shared expert and the first layer's MLP in plain ops.
- Weights: the routed experts int8 per channel (one scale per output row and expert), by LiteRT's quantizer (litert_quantizer); every other weight fp32, the token table and the output layer included. Activations, the KV cache and the KDA states are fp32; KV 4,096 tokens. Signatures `prefill_128`, `prefill_1024`, `decode`.
- The bundle carries Kimi-Linear's own metadata (LiteRT-LM has no configuration for this model): stop token `<|im_end|>` (163586), pad token `[PAD]` (163839), no start token (the source's template and tokenizer add none), the `generic_model` type, no sampler settings (the source ships none); the source's chat template with four rewrites for LiteRT-LM's template engine; a `tokenizer.json` converted from the source's tiktoken tokenizer.

## The runtime it needs

LiteRT-LM 0.19.1, the latest release on 2026-11-13, does not run this file: the `moe` kernels of LiteRT-LM 0.19.1 accept only GELU. SiLU in the `moe` op is in two LiteRT pull requests, open on that day: [#11192](https://github.com/google-ai-edge/LiteRT/pull/11192) (the GPU kernel) and [#11188](https://github.com/google-ai-edge/LiteRT/pull/11188) (the XNNPACK kernel). Every check ran on this build:

LiteRT-LM `02d7e8d8` (main, 2026-11-07) with the LiteRT commit its WORKSPACE pins, `bf49eb62`, and the two pull requests applied without their test files (one test file of #11192 does not apply at `bf49eb62`; no test is built). ml-drift is the WORKSPACE's own archive. `--override_repository` skips the patch commands that LiteRT-LM's WORKSPACE applies to the LiteRT archive, so [`apply_ws_patch_cmds.py`](https://github.com/john-rocky/hf-to-litertlm/blob/84336e8d00a567464e834854bdc1987a5c6656ca/qwen3_30b_a3b_work/apply_ws_patch_cmds.py) (kept in the hf-to-litertlm folder of the Qwen3-30B-A3B conversion) applies them by hand; it needs GNU sed installed as `gsed`. The block also needs Git LFS and Bazel (LiteRT-LM's `.bazelversion` names 7.6.1, which Bazelisk runs); it ran here with Xcode 27.0 on macOS 27.0. Run it in an empty directory:

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

The two commits are the pull requests' heads on 2026-11-13 (each pull request has one commit): `ebacd381` is #11192 and `9860c14d` is #11188. Copy `bazel-bin/runtime/engine/litert_lm_advanced_main` and `bazel-bin/external/litert/litert/c/libLiteRt.dylib` into one directory with `prebuilt/macos_arm64/libwebgpu_dawn.dylib` and `prebuilt/macos_arm64/libGemmaModelConstraintProvider.dylib` (paths from `LiteRT-LM`, where the block ends): the CLI links the last two and does not start without them, also on the CPU. The block's third target, the GPU accelerator, is not needed for this repository's CPU file. Put `kimi_linear_48b_a3b_mixed_int8_cpu_ekv4096.litertlm` in the same directory and run the model card's command there.

```bash
./litert_lm_advanced_main --backend=cpu --model_path=kimi_linear_48b_a3b_mixed_int8_cpu_ekv4096.litertlm \
  --max_output_tokens=2048 --min_log_severity=2 --sampler_top_k=1 \
  --cache_dir=:nocache --num_cpu_threads=12 --input_prompt="What is 17 + 25? Answer briefly."
```

## Files

| file | what it does |
|---|---|
| `tools/kimi_ext.py` | the graph writer: KDA, MLA, the router and the `moe` op, the shared expert; the checkpoint loader |
| `tools/kimi_export.py` | traces `prefill_128`, `prefill_1024` and `decode` and writes one fp32 `.tflite`; with `KIMI_EXPERTS=graft` the routed experts are 4-element placeholders |
| `tools/graft_kimi.py`, `tools/quant_common.py` | quantizes the routed experts from the bf16 checkpoint with litert_quantizer's own code, one projection of one layer at a time, and writes them into the placeholders |
| `tools/tok_convert.py` | the source's tiktoken tokenizer as a Hugging Face `tokenizer.json` (Transformers' `TikTokenConverter`, the pattern read from the source's tokenizer class) |
| `tools/kimi_bundle.py` | the `.litertlm` bundle: metadata, executor metadata (the state buffers), tokenizer, the graph |
| `reproduce.sh` | the steps below in order |
| `requirements.txt` | the Python environment the conversion ran in |

## Steps

The environment, in a new Python 3.12 venv (litert-torch from source at `67f11e89`, without its dependencies, which `requirements.txt` already pins):

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps "git+https://github.com/google-ai-edge/litert-torch.git@67f11e89e7ca14ac222638c88e707f8a2e755273"
```

`SNAP=<snapshot dir> bash reproduce.sh WORK_DIR` runs these steps; without `SNAP` it downloads the checkpoint (98 GB) into `WORK_DIR/hf_cache`.

1. Tokenizer: copy `tokenization_kimi.py`, `tiktoken.model` and `tokenizer_config.json` from the snapshot into `tok/` next to `tools/` and run `python3 tools/tok_convert.py tokjson`. It loads the source's tokenizer class (`trust_remote_code`: the source's own `tokenization_kimi.py`, which reads `tiktoken.model` and opens no network connection) and writes `tokjson/tokenizer.json` (sha256 `475d3a67952bcc7ece90c0288165ff4193797b3b247b1869b9abbea0946c6100`). On 6,612 texts (329,448 tokens) it gives the same ids as the source tokenizer, also with tokenizers 0.21.0, the version LiteRT-LM links.
2. Chat template: the source's `chat_template.jinja` with four rewrites for LiteRT-LM's template engine (minijinja without its Python compatibility layer), each found exactly once: `msg.get('content')` → `msg['content']`, `message.get('name')` → `message['name']`, `message.get('tool_calls')` → `message['tool_calls']`, `tools | tojson(separators=(',', ':'))` → `tools | tojson`. The result, `template/chat_template.lite.jinja`, has sha256 `8aebb7a4414511393c00c45c36a861905c8dca45bc2f688dfd5a93afb0b635a0`.
3. Export: `KIMI_EXPERTS=graft python3 tools/kimi_export.py $SNAP WORK_DIR/plain.tflite --prefill 128,1024 --cache 4096` (fp32; 8,056,377,136 B; peak footprint 26.85 GB, 399 s on an Apple M4 Max).
4. Experts: `python3 tools/graft_kimi.py --tflite WORK_DIR/plain.tflite --ckpt $SNAP --out WORK_DIR/model.tflite --bits 8 --block 0 --spill WORK_DIR/spill.bin` (55,282,341,392 B; peak footprint 10.75 GB; `--spill` keeps the finished experts on disk instead of in RAM and does not change the output bytes).
5. Bundle: `python3 tools/kimi_bundle.py --tflite WORK_DIR/model.tflite --cut $SNAP --out WORK_DIR/kimi_linear_48b_a3b_mixed_int8_cpu_ekv4096.litertlm --cache 4096 --consume-tflite` (55,285,913,104 B; `--consume-tflite` frees the model's blocks as the bundle takes them: macOS `F_PUNCHHOLE` only; elsewhere leave it out and keep the model's size free again).

## How it was checked

- The graph against Transformers' `kimi_linear` model (bf16 weights computed in fp32, on the CPU), with the bundle in LiteRT-LM's CPU path: 24 teacher-forced positions of a GSM8K prompt, the top-1 token of every position compared; 12 questions and a 3-turn conversation, greedy, the answers judged by the same rule on both sides. The results are on the model card.
- The tokenizer: the 6,612 texts above. The template: rendered with minijinja 2.14.0 (the template engine LiteRT-LM uses) against the source's template in Transformers, the same text for a user turn, a system turn, three turns with history and a history without the generation prompt; with tools, the declarations differ only in the JSON spacing (`, ` and `: ` where the source writes compact JSON).
