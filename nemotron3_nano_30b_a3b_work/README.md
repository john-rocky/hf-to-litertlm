# Nemotron-3-Nano-30B-A3B on LiteRT-LM: a 52-layer hybrid MoE decoder written with the Tensor API

[nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
(revision `bf77c317`, the NVIDIA Nemotron Open Model License) is a hybrid model: 52 layers (23 Mamba-2, 23
mixture-of-experts, 6 attention); each MoE layer has 128 routed experts, of which 6 are used for each token, and one
shared expert; 30B parameters, of which 3.5B are active (source model card). This directory writes its decoder straight
from the bf16 checkpoint with the [LiteRT Tensor API](https://github.com/google-ai-edge/LiteRT/tree/main/tensor), one
tensor at a time, and packs it into a `.litertlm` that LiteRT-LM's stock executor runs.

- Each MoE layer's routed experts are one TFLite custom op `moe`, the op that LiteRT's XNNPACK delegate and GPU
  delegate run. The op computes `act(x·Wg) ⊙ (x·W1) · W2` with a GELU activation; these experts are relu² MLPs without
  a gate, `relu(x·Wup)² · Wdown`. So relu² is written as the op's `gelu_tanh` with `Wg = W1 = Wup` (one tensor, one
  set of scales), the op's input scaled by 256 and each expert's output by 1/65,536:
  `gelu(256u)·256u / 256² = u²·Φ(256u)`, within 2.6e-6 of `relu(u)²` (`nh_pack.py` documents the bound).
- In fp16 the scaled input overflows (`(256u)²` for `|u| > 1`), so the bundle declares fp32 activations. With fp16
  GPU activations a random tiny model generated `<unk>` only.
- Each `moe` output gets a second consumer, an exact `moe * 0` term, so the GPU delegate does not fuse the next op into
  the expert kernel, a fusion that emits invalid shader code (`--moe_fusion_guard`, on by default).
- The router (a sigmoid, top-6 on the sigmoid plus the correction bias, the chosen weights renormalized and scaled), the
  shared expert, Mamba-2 and attention are ordinary ops. Mamba-2 runs the chunked SSD scan in prefill and the
  recurrence in decode; its convolution and SSM states ride in LiteRT-LM's cache as linear-attention state (declared in
  the bundle's ExecutorMetadata section). The runtime pads prefill on the right with token 0; the graph masks the pads
  out of the scan, the convolution window and the states.
- Weights: experts int4 blockwise (block 32, fp32 scales raised to at least 2^-14); token table, Mamba-2 in / out
  projections, attention q / k / v / o, shared experts and lm_head int8 per channel; router, norms and the Mamba-2
  convolution, A, D and dt bias fp32. Activations, the KV cache and the states are fp32.
- Signatures `prefill_128`, `prefill_1024` and `decode`, a KV cache of 4,096 tokens.
- The bundle carries the chat template of
  [litert-community/Nemotron-3-Nano-4B](https://huggingface.co/litert-community/Nemotron-3-Nano-4B)
  (`../templates/nemotron3nano_history.jinja`) with its one `messages[0]` read guarded for an empty message list, the
  checkpoint's own `tokenizer.json` bytes, the `generic_model` type, stop tokens 2, 11 and `<|im_end|>\n`, and a
  `thought` channel.

## The runtime it needs

LiteRT-LM 0.19.1. On the GPU, LiteRT-LM 0.18.0's `moe` op only supports activation='gelu'; this file uses
'gelu_tanh'. The CPU checks ran through the PyPI package `litert-lm==0.19.1`; the GPU checks and speeds through
`litert_lm_advanced_main` built from the v0.19.1 tag (`7a03724c`) with `--sampler_handles_input=false` (the PyPI
package has no such setting) and `--disable_cache=true`. On a Mac GPU, every run that loaded a GPU program cache an
earlier run had written answered wrongly (5 of 5: 4 through the PyPI package, with text unrelated to the prompt, and 1
through the command-line tool, a loop); runs that compiled answered correctly (4 of 4). `cache_dir=":nocache"` (Python)
and `--disable_cache=true` (the command-line tool) avoid it.

## Files

| file | what it does |
|---|---|
| `nh_pack.py` | reads the checkpoint one tensor at a time (`st_read.py`) and writes `weights.bin` + `index.tsv` + `config.tsv` + `moe_options.bin` + `pack.json`: experts stacked in the op's layout and quantized with ai-edge-quantizer's own functions (int4 blockwise, scales raised to at least 2^-14), the token table, attention, Mamba-2 projections, shared experts and lm_head int8 per channel, the router fp32, relu² written as above. |
| `st_read.py`, `st_read_check.py` | a safetensors reader with one large `read()` per tensor (an mmap reader was slow on a USB disk), and its check against `safe_open(...).get_tensor()`. |
| `scale_scan.py` | counts zero, tiny and non-finite scales in a pack (0 of each is the condition). |
| `graph/nemotronh_writer.cc`, `graph/BUILD` | the Tensor API graph: `prefill_<N>` and `decode` signatures, the Mamba-2 states and the KV cache in and out per layer (V stored transposed), the chunked scan with the pad guard, grouped-query attention over the whole cache with the runtime's mask, the router and the `moe` op, `ModelFactory::Save`. |
| `patches/blockwise_named_scales.diff` | lets the Tensor API (LiteRT `dcb657e8`) write a blockwise-quantized tensor whose scales are an existing tensor of the graph, with a block shape (the `moe` op reads the scales as an input); the same patch as `../qwen3_30b_a3b_work/patches/`. |
| `overlay.sh` | applies the patch to a LiteRT checkout and links `graph/` into it as `tensor/examples/nemotronh`. |
| `nh_bundle.py`, `template/nemotron3nano_history_guard.jinja` | the `.litertlm`: the metadata above, the tokenizer, an ExecutorMetadata section that declares every state of the graph, the decoder, fp32 activations declared. |
| `bundle_section_check.py` | checks a bundle against its sources: the decoder section = the standalone `.tflite` and the tokenizer section = `tokenizer.json`, byte for byte; prints the metadata it carries. |

## Steps

Python 3.13 with numpy, torch, safetensors and ai-edge-quantizer 0.10.1 for the pack (tested: numpy 2.5.4, torch
2.14.1, safetensors 0.8.0, flatbuffers 25.12.19); Python 3.12 with litert-lm-builder 0.19.0 and ai-edge-litert 2.3.0 for
the bundle; Bazel (bazelisk) for the writer.

```bash
# 0. The checkpoint at the tested revision (35 files, 63,174,967,188 bytes).
HF_HUB_DISABLE_XET=1 hf download nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --revision bf77c3174f68ad409e1c2aa60daeb46e32d1c606 --local-dir ckpt

# 1. Pack: 19 GB of weights in the writer's layout, then the scale scan (0 zero, tiny or non-finite scales).
python nh_pack.py ckpt pack --experts xb4 --block 32 --fc int8 --lm-head int8 --embed int8 \
  --relu2 gelu_in --gelu-s 256
python scale_scan.py pack

# 2. The writer, in a LiteRT checkout at dcb657e8.
git clone https://github.com/google-ai-edge/LiteRT.git
git -C LiteRT checkout dcb657e8a537b2dc944e53e14c7eb43bcd9ededa
bash overlay.sh LiteRT
(cd LiteRT && bazelisk build -c opt //tensor/examples/nemotronh:nemotronh_writer)
LiteRT/bazel-bin/tensor/examples/nemotronh/nemotronh_writer --weights_dir=pack \
  --out=nemotronh.tflite --prefill=128,1024 --cache=4096

# 3. The bundle, and its check against its sources.
python nh_bundle.py --tflite nemotronh.tflite \
  --out nemotron3_nano_30b_a3b_mixed_int4_ekv4096.litertlm --tokenizer ckpt/tokenizer.json
python bundle_section_check.py nemotron3_nano_30b_a3b_mixed_int4_ekv4096.litertlm \
  --tflite nemotronh.tflite --tokenizer ckpt/tokenizer.json
```

The model repository's second file, `nemotron3_nano_30b_a3b_mixed_int4_ekv2048.litertlm` (a 2,048-token cache), is the same pack with `--cache=2048` in step 2 and that file name in step 3.
The third file, `nemotron3_nano_30b_a3b_int4_ekv2048.litertlm` (int4 projections, a 2,048-token cache), is step 1 with `--fc int4 --fc-block 32` in place of `--fc int8`, then `--cache=2048` in step 2 and that file name in step 3.

On an Apple M4 Max the pack took 24 minutes (peak footprint 4.4 GB) and the writer and the bundle about 2 hours
together, the checkpoint and the outputs on a USB disk.

## How it was checked

- **The reference is NVIDIA's arithmetic.** A reference implementation of the model (float32 on the bf16 weights)
  was held to the checkpoint's own `modeling_nemotron_h.py`: on a random tiny checkpoint, logits within 5.3e-7 of the
  largest logit and the same top token at 40 of 40 positions; on the real weights, the same top token at 64 of 64
  positions of a 64-token chat prompt (within 1.5e-6), and the modeling code's top token equals the reference's at
  every position of three of the reference's greedy replies.
- **The graph, on a random tiny model.** A random 5-layer NemotronH (Mamba-2, MoE, Mamba-2, attention, MoE; 16
  experts, 3 used per token) through the same pack, writer and bundle: on the CPU (XNNPACK) the logits are within
  8.0e-7 of the largest logit of the reference's, with fp32, int4 and int8 experts in the `moe` op, at prefill lengths
  4, 16 and 32 (1, 2 and 4 scan chunks, with pads) and in decode; LiteRT-LM 0.19.1 from PyPI generates the reference's
  32 greedy tokens on the CPU and on WebGPU.
- **Three real layers.** Layers 0 (Mamba-2), 1 (MoE) and 5 (attention) with the embeddings, the final norm and
  lm_head, int4 experts written as above, against the reference on the same dequantized weights: the top token matches
  at every decode position after prefills of 32, 128 and 256 tokens. The logits differ by up to 5.7e-5 of the largest
  logit on the CPU (XNNPACK through ai-edge-litert 2.3.0) and 1.8e-5 on WebGPU (a LiteRT-LM source build at
  `dab2e708`), more than the 1e-5 set before the run; the relu² form alone moves them by 1.0e-5. LiteRT-LM 0.19.1
  generates the reference's 32 greedy tokens on the CPU and on WebGPU.
- **The whole model.** 12 questions (8 short questions from this repository's `scripts/verify_quality.py`, a price
  word problem, a syllogism, an average-speed question and a question in Japanese) and a 3-turn chat ("My name is
  Kenji." / "What is my name? Answer with the name only." / "What is 17 times 23? Answer with the number only."),
  greedy, up to 2,048 new tokens, one fresh process per question; the CPU through the PyPI package, the GPU through the
  v0.19.1 command-line tool. Thinking on: 12 of 12 right on both, every reply ends with a stop token, and the chat
  answers Kenji and 391. Thinking off, against the reference with thinking off (2,048-token thinking replies were too
  slow on the reference): both backends give the reference's verdict on 12 of 12 (11 right; the reference and both
  backends answer "violet" to the rhyme question), end every reply with a stop token as the reference does, and answer
  the chat with Kenji and 401. The reference answers 401 too: given each backend's own chat history, its first digit
  is '4' over '3' by 0.66 and 1.18 logits. The check as first written required 391 there; after these runs it was
  revised to require the right number only where the reference gives it.
- **What did not ship.** fp16 activations: the scaled relu² input overflows, and on the random tiny model WebGPU with
  fp16 activations generated `<unk>` only.

Speeds: the model card.
