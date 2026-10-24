# Qwen3-VL's DeepStack through LiteRT-LM's fast_vlm path: a wide image embedding

From the conversion of [`tencent/WeVisOne-VL-2B`](https://huggingface.co/tencent/WeVisOne-VL-2B) (WeChat Vision's perception model built on Qwen3-VL) to [`litert-community/WeVisOne-VL-2B`](https://huggingface.co/litert-community/WeVisOne-VL-2B). Scripts: [`wevisone_work/`](../wevisone_work/), one command: `scripts/ship_wevisone.sh`.

## The problem

Qwen3-VL feeds its decoder two things the `fast_vlm` path does not have:

- **DeepStack.** Three extra vision mergers (on ViT blocks 5, 11 and 17 for this model) produce features that transformers adds to the decoder's hidden state at the image positions after decoder layers 0, 1 and 2 (`_deepstack_process`).
- **M-RoPE.** Every token gets a three-axis position (t, h, w); image tokens take their row and column in the merged grid.

LiteRT-LM's `fast_vlm` path gives the decoder one embedding per image token from the vision adapter, and one position per token.

## What the shortcuts cost

Measured on the float32 PyTorch model at 512 × 512 (256 image tokens), against the same model unchanged, on 29 detection, referring, pointing, counting, OCR and segmentation questions. A box counts when IoU ≥ 0.75; a point when it is within 2 % of the side.

| Decoder input | Boxes | Counts | Points | OCR character error |
|---|---:|---:|---:|---:|
| The three DeepStack features added into the one image embedding, exact M-RoPE | 41/55 | 4/6 | 17/44 | 0.472 |
| The same, with the runtime's 1-D positions | 0/55 | 3/6 | 2/44 | 1.000 |
| The wide embedding below, float32 build of the shipped graphs | 55/55 | 6/6 | 44/44 | 0.000 |

Adding the features into one embedding gave the same tokens with exact M-RoPE and with the derived positions below (29 of 29 answers), so its cost is the DeepStack sum alone. With 1-D positions the coordinate answers fall apart: seven of the 29 answers run to the 1,024-token cap. The float32 build of the wide graphs gives the original's tokens on all 29 questions.

## The wide embedding

Each token carries 8,196 values:

| Values | Content |
|---|---|
| 0 … 2047 | the decoder's input embedding (a text token's lookup, or the main merger output for an image token) |
| 2048 … 4095 | DeepStack 0, added after decoder layer 0 |
| 4096 … 6143 | DeepStack 1, added after decoder layer 1 |
| 6144 … 8191 | DeepStack 2, added after decoder layer 2 |
| 8192 | the image flag: 1 from the adapter, 0 from the embedder |
| 8193 … 8195 | zeros |

- The vision encoder returns the four merger inputs in raster order, [1, 1024, 4096]. The adapter returns `merger(h23)`, the three DeepStack mergers and the flag tail, [1, 256, 8196].
- The embedder returns the text lookup [1, 1, 2048] padded with 6,148 zeros.
- The decoder takes [1, T, 8196], runs layer 0 on the first 2,048 values and adds slice i + 1 after layer i (i = 0, 1, 2). Text tokens carry zeros there, so the add changes nothing at text positions: the decoder computes `_deepstack_process` exactly (bit for bit in PyTorch).

LiteRT-LM v0.18.1 copies each image embedding at the decoder's input width (`runtime/components/embedding_lookup/embedding_lookup_multi_modal.cc` L120–123: the floats per token are the product of the input's inner dimensions) and reads only the token count from the adapter's output (`runtime/executor/vision/vision_executor_utils.cc`). The 8,196-wide files ran unchanged on the Mac (CPU, WebGPU), on a Galaxy S26 (OpenCL GPU, the Android build of v0.18.1) and on an iPhone 18 Pro (the v0.16.0 framework, an int8 build).

## Positions from the 1-D position and the flag

The decoder computes Qwen3-VL's positions in the graph from the runtime's position p and the flag f:

- image token (f = 1): (t, h, w) = (P, P + ⌊(p − P) / G⌋, P + (p − P) mod G)
- text token (f = 0): t = h = w = p − D · [p ≥ P + N]

P = 15 (the fixed system turn, `user\n` and `<|vision_start|>`), G = 16 (the merged grid side at 512 px), N = G² = 256, D = N − G = 240. The steps are `relu(x) − relu(x − 1)` on integer-valued floats, so no clamp (`RELU_0_TO_1`, which LiteRT-LM's WebGPU delegate refused in an earlier conversion: [decider2bv_work/FINDINGS.md](../decider2bv_work/FINDINGS.md) §4) reaches the GPU delegates; the division by G is a sum of G − 1 steps. The cos / sin of these positions equal transformers' `Qwen3VLModel.get_rope_index` bit for bit on 85,542 positions (29 picture questions at 512 and at 768).

The flag is needed. Positions from p alone put a text-only prompt's tokens P … P + N − 1 on the image grid: that model answered 2 of 10 text questions correctly, against 8 with plain positions.

## The cost that stays: text-only conversations past 270 tokens

A rule without state makes either picture conversations exact or long text-only conversations, not both. With the flag, a text-only conversation gets the original's positions up to token P + N − 1 = 270 and positions 240 lower after it: a 353-token summary request broke down after 7 tokens (the float32 original wrote 84). At 768 × 768 (G = 24, N = 576, D = 552) the exact range is 590 tokens. Two ways out, not taken for the published file: a separate text-only file of the same checkpoint on the LLM rail, or a decoder state that remembers whether the conversation holds a picture.

## The export

[`wevisone_work/tools/wv_patch.py`](../wevisone_work/tools/wv_patch.py) installs the wide input, the DeepStack adds and the derived rotary in the export process only (litert-torch 0.10.0 `export_hf.export` with `externalize_embedder=True`, `single_token_embedder=True`, `use_rope_composite=False`), on a text-only checkpoint derived from the VLM with [`tools/vlm_text_ckpt/make_text_ckpt.py`](../tools/vlm_text_ckpt/make_text_ckpt.py). Before the export it checks the patched cos / sin against the formula above on every cache position. Nothing on disk changes.

## A runtime defect that every file with an embedder section meets

From the second message of a conversation, LiteRT-LM passes the previous answer's stop token to the model with a zero embedding. Decode queues the token it samples without an embedding, the next prefill fills such a token as its id followed by zeros, and it never looks the embedding up (`runtime/executor/llm_litert_compiled_model_executor.cc` L1326–1341, L577–584, L413–447 at v0.18.1; the same code at v0.16.0 and on main of 2026-10-23). The NPU executor looks the embedding up before it queues the token.

[`wevisone_work/tools/r6_twoturn_probe.py`](../wevisone_work/tools/r6_twoturn_probe.py) replays a file's two graphs on the runtime's schedule with the stop token's row as its embedding, as zeros, or as the runtime writes it, and compares the second answers with the runtime's. On a float32 build of WeVisOne-VL-2B the zero row reproduces the runtime's second answer and the right row the original's. On `litert-community/Qwen2.5-Coder-1.5B-Instruct` (int4, 1 thread on the Mac CPU) the runtime answered "I asked you to remember the number 4182." where the right row gives "You asked me to remember the number 4182."; the zero row gives the runtime's text. Three of the four conversations tried answered the same either way. Until the runtime looks the token up, ask one question per conversation.
