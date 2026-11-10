# gpt-oss-120b on LiteRT-LM: a 36-layer MoE decoder written with the Tensor API

[openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) (revision `b5c939de`, Apache License 2.0) is a mixture-of-experts reasoning model: 36 layers that alternate sliding-window (128 tokens) and full attention, 128 experts per layer of which 4 are used for each token, 117B parameters with 5.1B active (source model card), the experts stored in MXFP4. This directory writes its decoder straight from the checkpoint with the [LiteRT Tensor API](https://github.com/google-ai-edge/LiteRT/tree/main/tensor), one tensor at a time, and packs it into a `.litertlm` that LiteRT-LM 0.19.1 from PyPI runs on the CPU.

- Each layer's experts are one TFLite custom op `moe`: only the 4 routed experts are computed for each token. The router (with its bias), the top 4 and the softmax over their 4 logits are ordinary ops in the graph.
- Weights: experts int4 blockwise (block 32, one fp32 scale per block chosen by the least squared error against the MXFP4 values, raised to at least 2^-14), attention q / k / v / o and lm_head int8 per channel; router, biases, sinks, norms and the token table fp32. Activations and the KV cache are fp32.
- Signatures `prefill_128`, `prefill_1024` and `decode`, a KV cache of 4,096 tokens.
- The bundle carries the source's chat template with four edits for LiteRT-LM, the harmony headers declared as channels, the checkpoint's own `tokenizer.json` bytes, and declares fp32 activations.

## What differs from the source, and why

The source's experts compute `(clamp(up, -7, 7) + 1) · g · σ(1.702 g)` with `g = min(gate, 7)`. The `moe` op of LiteRT-LM 0.19.1 offers GELU only (exact or tanh form), with no clamp. This file computes `(up + 1) · GELU_tanh(gate)`: `x · σ(1.702 x)` is replaced by GELU in its tanh form (largest difference 0.021, at x = 2.29) and the clamps are dropped. The clamps matter, more so in the later layers. On the source's Transformers code in float32, over all 36 layers and three conversations with the source model's replies (718 tokens, 413 of them reply positions), once as written and once with this activation (the MXFP4 weights unchanged): the share of routed (token, expert) pairs that reach a clamp grows from 4 % in layer 0 to 48 % in layer 11, and is 72 to 100 % in every layer from 14 on; with this activation the hidden state differs from the source's by 1.4 to 5.5 % (relative L2) in layers 0 to 5 and by 21 to 66 % from layer 6 on; at the output it still picks the source's next token at 399 of the 413 reply positions (96.6 %, teacher-forced on the source's replies). The file adds the int4 experts on top. The whole-model check below compares replies.

The experts' biases: the op has no bias input, so each expert's input gets one more block of 32 columns that carries a constant (1/1024) and the biases (2,880 → 2,912 inputs; the expert weights grow by 2.2 %). This is exact.

## Files

| file | what it does |
|---|---|
| `convert_gpt_oss_120b.sh` | the whole conversion as steps (`fetch pack writer write bundle`); `LAYERS=N` writes the first N layers only, to test the pack, writer and bundle steps on ~6 GB of reads (`fetch` still downloads the whole snapshot). |
| `tools/pack_gptoss.py`, `tools/st_pread.py` | read the checkpoint one tensor at a time (`pread` from the safetensors shards) and write the pack: experts in the op's layout, MXFP4 → int4 blockwise (`--experts xb4 --xb4-rule mse`), the bias block, attention and lm_head int8 per channel, the op's options (`--act gelu_tanh`). |
| `tensor_api/gptoss/gptoss_writer.cc`, `tensor_api/gptoss/BUILD` | the Tensor API graph: `prefill_<N>` and `decode` signatures, KV cache in and out per layer, the sinks, the 128-token window on the even layers, YaRN RoPE, the router and the `moe` op, `ModelFactory::Save`. |
| `tensor_api/blockwise_named_scales.diff` | lets the Tensor API (LiteRT `dcb657e8`) write a blockwise-quantized tensor whose scales are an existing tensor of the graph, with a block shape (the `moe` op reads the scales as an input). |
| `tensor_api/overlay.sh` | applies the patch to a LiteRT checkout and links `tensor_api/gptoss/` into it as `tensor/examples/gptoss`. |
| `template/make_ship_template.py`, `template/make_pbtext.py`, `template/gptoss_litertlm.jinja`, `template/gptoss_litertlm.pbtext`, `template/vendor_b5c939de.jinja` | the source's chat template (`vendor_b5c939de.jinja`, its `chat_template.jinja` at that revision), the ship template with its four edits, and the LlmMetadata text (channels, stop tokens, sampler, `generic_model`). |
| `tools/make_bundle.py` | the `.litertlm`: the metadata, the tokenizer, the decoder, fp32 activations declared; `--stream` writes the header and copies the decoder in one pass, hashing both; `--ids-from` copies the header's uuid and creation time from another file. |
| `tools/bundle_sections.py` | prints a `.litertlm`'s sections with their sha256, its uuid / creation time and the whole-file sha256. |
| `tools/tfl_tensors_cmp.py` | checks a partial re-run (`LAYERS=N`) against the published file: every constant tensor of the smaller file is in the larger one with the same bytes and quantization parameters (it needs the `ai-edge-litert` and `litert-lm-builder` packages). |
| `tools/requirements.txt` | the Python packages: the full `pip freeze` of the venv that re-ran these steps on 2026-11-10; numpy, torch, transformers, flatbuffers, litert-lm-builder and protobuf are the versions the 120B file was made with. |

## Steps

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r gpt_oss_120b_work/tools/requirements.txt
PY=$PWD/.venv/bin/python bash gpt_oss_120b_work/convert_gpt_oss_120b.sh OUTDIR
```

`bazelisk` builds the writer (`BAZEL_OUTPUT_BASE=<dir>` puts Bazel's output base there). The 120B needs about 300 GB of disk at the peak (the snapshot 65.3 GB, the pack 77.2 GB, the TFLite 77.3 GB, the `.litertlm` 77.3 GB); `SNAPSHOT=<dir>` uses a snapshot you already have. On a 128 GB Mac with the checkpoint and the outputs on a USB hard disk shared with other jobs, the pack took 9,815 s (peak footprint 15.0 GB), the writer about 73 minutes (peak footprint 0.22 GB; it maps the pack, and its page faults on that disk read at a few MB/s at times, so an SSD for the pack helps), the bundle about 72 minutes (one streamed pass, peak footprint 0.34 GB).

The script as committed here has not yet run end to end from a clean copy of this directory: a run with `LAYERS=2`, its file compared tensor by tensor with the 120B file (`tools/tfl_tensors_cmp.py --subgraph all`), is queued, and this section will say what it gave. The `fetch` step's file list was checked with `hf download --dry-run` (huggingface_hub 2.4.0): the 26 top-level files of the pinned revision, 65.3 GB.

## How it was checked

- **The writer, on the first two real layers.** Layers 0-1 + final norm + lm_head with the experts' MXFP4 values kept exactly (int8 blocks) and an exact activation, against the source's Transformers code without the clamps: the top-1 token at 445 / 445 positions of three prompts, largest logit difference 2.86e-5, the router's top-4 set the same at every position (the CPU, XNNPACK). With the int4 experts: 401 / 445.
- **The template.** For 7 conversations, with and without tools, the ship template gives the source template's text through Transformers and through LiteRT-LM 0.19.1's own template engine, byte for byte.
- **The reference.** The replies of the source model come from MLX (mlx-lm 0.31.3) with the source's MXFP4 experts and the source's YaRN frequencies (mlx-lm rounds the YaRN correction range, which the source's config turns off); on the first two layers it equals Transformers at 445 / 445 positions (largest difference 3.8e-5).
- **The reference at 36 layers.** The MLX replies to three of the questions (up to 256 tokens each, 413 positions), teacher-forced through the source's Transformers code in float32 streamed one layer at a time: the same next token at all 413 positions (one position has a top-2 margin under 0.01 logits).
- **The whole model.** 12 questions and a 3-turn conversation on the CPU with LiteRT-LM 0.19.1 from PyPI, greedy, up to 2,048 tokens, each reply's `final` channel judged by the same function as the reference's: the same verdict on 12 of 12, every reply ends with a stop token, and the 3 turns pass (in the third turn the reference renamed the function, so its tests could not call it; the file's reply kept the name and passed its 4 tests).

Speeds, answers and how to run it: the model card, [litert-community/gpt-oss-120b](https://huggingface.co/litert-community/gpt-oss-120b).
