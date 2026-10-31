---
license: apache-2.0
base_model: vllm-sr/Vela-1.0-Encoder-307M-Embedding
pipeline_tag: sentence-similarity
library_name: litert
tags:
  - litert
  - litert-lm
  - embeddings
  - modernbert
  - sentence-transformers
  - android
language:
  - multilingual
---

# Vela-1.0-Encoder-307M-Embedding — LiteRT

This repository is a conversion of [vllm-sr/Vela-1.0-Encoder-307M-Embedding](https://huggingface.co/vllm-sr/Vela-1.0-Encoder-307M-Embedding) to LiteRT and LiteRT-LM, made with litert-torch 0.10.0 for LiteRT 2.2.0 and LiteRT-LM 0.18.0: on a Galaxy S26 it embeds a text of up to 64 tokens in 6.8 ms on the NPU (2026-10-18), and at the GPU's default precision on the Galaxy S26, the Pixel 8a and a Mac its search results on SciFact stay within 0.003 of the float32 reference's (nDCG@10 0.48633 to 0.48813 against 0.48581; measured 2026-10-30 and 2026-10-31).

LiteRT (`.tflite`) and LiteRT-LM (`.litertlm`) conversions of vllm-sr/Vela-1.0-Encoder-307M-Embedding (revision `290318fd`), a 22-layer ModernBERT sentence embedder (768-d, mean pooling + L2 normalization, Matryoshka 64–768) with the Gemma 256k tokenizer. The graph takes `input_ids` + `attention_mask` and returns the normalized sentence embedding; pooling and normalization are inside the graph, so the output is the same vector sentence-transformers returns for the same text.

## Files

| file | input → output | weights | size |
|---|---|---|---|
| `vela_L64_fp16.tflite` / `vela_L128_fp16.tflite` / `vela_L256_fp16.tflite` / `vela_L512_fp16.tflite` — **CPU** (one file, token ids in) | `input_ids` int32 [1,L], `attention_mask` float32 [1,L] → `embedding` float32 [1,768] | float16 (FC weights and the token table, dequantized at load), float32 activations | 614–616 MB each |
| `vela_L64_embeds_fp16.tflite` … `vela_L512_embeds_fp16.tflite` + `tok_embeddings_fp16.bin` — **GPU / NPU** (the host looks the token rows up) | `inputs_embeds` float32 [1,L,768], `attention_mask` float32 [1,L] → `embedding` float32 [1,768]; the host looks the token rows up in the table (float16, row-major [256000,768]) | float16 | 221–223 MB + 393 MB table |
| `Vela-1.0-Encoder-307M-Embedding_fp16.litertlm` | text → embedding through LiteRT-LM `EmbeddingEngine` (embedder + `encoder_64/128/256/512` + the repo's tokenizer; the engine adds `<bos>` / `<eos>` and normalizes) | float16 | 624 MB |

One file per static window L: a text of n tokens (including `<bos>` and `<eos>`) runs in the smallest window with L ≥ n; the rows beyond n are padding (`attention_mask` 0, `input_ids` 0). Tokenize with the repo's `tokenizer.json` (post-processor adds `<bos>` = 2 and `<eos>` = 1); strip the text first — sentence-transformers strips before tokenizing, and a trailing space otherwise becomes a lone `▁` token.

## Conversion

- Direct trace of the encoder (attention written out, alternating global / sliding-window(±64) layers, the publisher's YaRN RoPE tables as constants) + mean pooling + L2 normalization with litert-torch 0.10.0 / litert-converter 0.5.0.dev20260912 → one `.tflite` per window; weights to float16 with ai-edge-quantizer (weight-only; activations float32).
- The LayerNorms of layers 12–21 and the final norm are computed on x·2⁻⁶ (eps·2⁻¹²), layers 9–11 on x·2⁻²: the sum of squares of their inputs reaches 4e7 in float32, beyond float16's 65,504, and this rewrite keeps float16-storage GPU paths finite. Exact in float32 (bit-identical to the plain graph).
- An int8 token table was tried and rejected: the per-row int8 rounding error of this table (0.75 % per row on average, 2.1 % on `<bos>`) moves short sentences by up to 0.7 % in cosine (floor 0.9929 over 3,918 texts), and no set of exactly-kept rows restores the 0.999 floor; float16 keeps it at 0.9998. Int8 FC weights fail harder (cosine floor 0.964). Everything shipped here is float16.
- Two shapes because of the GPU delegate: it does not take a float16 token table inside the graph (`EMBEDDING_LOOKUP: Empty quantization params`), so the one-file form is a CPU form, and the `_embeds_` files take the looked-up rows from the host and run fully on the GPU (Metal: every op accelerated) and on the NPU. The pooling is written as multiply / sum / rsqrt rather than a masked select, which the delegate also refuses (`SELECT_V2` with a broadcast condition).
- Reference for every number: the publisher's own stack (sentence-transformers 5.1.2, transformers 4.57.6, CPU float32, SDPA math kernel), which reproduces the card's STS-B v2 Spearman (0.811867 vs the card's 0.811864) on the same 1,362 test pairs.

## Accuracy

Agreement with the reference vectors, checked as: cosine ≥ 0.999 to the reference on every text, top-1 retrieval agreement 100 % on non-near-tie queries, Spearman of pair cosines vs the reference ≥ 0.999, the attention mask live (cutting 5 tokens off the longest text moves the vector), every output finite. Check set: PAWS-X test (en 300 pairs, es / ja / zh 100 each), STS-B v2 test (1,362 pairs), plus concatenated PAWS-X sentences for the 128 / 256 / 512 windows — 4,184 texts in all; the rows are referenced by dataset id, revision and index in the conversion record, not copied.

Results on the full check set (Mac CPU, every window): the float16 one-file and host-table files reach cosine ≥ 0.99977 to the reference on every text, top-1 agreement 100 % (1,738–1,873 queries per window), pair Spearman 0.9999995, STS-B v2 Spearman +0.000004 vs the reference; the float32 export sits at 0.99999977 (the converter's noise floor). On the phones the device subsets (140–497 texts per window) pass the same checks on the S26 NPU (cosine ≥ 0.9994), on the S26 and Pixel 8a GPUs at FP32 (≥ 0.99997) and on both CPUs (≥ 0.99997); the LiteRT-LM bundle passes the full set through `EmbeddingEngine` (cosine ≥ 0.99977), and the control with special tokens off fails (0.31), which is the engine inserting `<bos>` / `<eos>` as sentence-transformers does. Int8 weights were tried and rejected (see Conversion). GPU fp16 arithmetic (the default precision of Metal, Adreno and Mali) drops the per-row cosine floor to 0.991–0.998 on about 1 % of the texts while every top-1 and the pair ranking hold; at FP32 the floor stays at 0.99977 or above.

What that does to search was measured on a retrieval set with known answers: the SciFact test set (`mteb/scifact` revision `d56462d0`, 300 queries over 5,183 documents with relevance judgments), each text in the smallest window that holds it (none is longer than 512 tokens), the documents ranked by cosine to each query, nDCG@10 and recall@10 averaged over the queries. The reference is the publisher's stack above, run on the same texts. The phones computed the vectors on their GPUs (LiteRT 2.2.0 `CompiledModel`, the `_embeds_fp16` files and the host table) and the Mac scored them. Each row embeds the queries and the documents at its own precision; a mix, such as documents indexed at FP32 and queries at the default precision, was not measured.

| device · GPU precision | nDCG@10 | recall@10 | lowest per-text cosine vs the reference (these 5,483 texts) |
|---|---|---|---|
| reference (sentence-transformers, CPU float32) | 0.48581 | 0.60833 | — |
| Mac M4 Max · Metal, FP32 | 0.48581 | 0.60833 | 0.99985 |
| Mac M4 Max · Metal, default (FP16 arithmetic) | 0.48633 | 0.60833 | 0.99685 |
| Galaxy S26 · Adreno, default (FP16 arithmetic) | 0.48813 | 0.60722 | 0.99588 |
| Pixel 8a · Mali-G715, default (FP16 arithmetic) | 0.48699 | 0.60722 | 0.99557 |
| Pixel 8a · Mali-G715, FP16_WITH_FP32_ACCUM | 0.48575 | 0.60833 | 0.99877 |

At the default precision every GPU stays within 0.003 of the reference on both measures, so a GPU can run these files at its default precision (Mac 2026-10-30, phones 2026-10-31). The publisher's card gives 48.28 (the card's scores are ×100, i.e. 0.4828) for SciFact from MTEB 2.21.0; its stack run directly, as here, gives 0.48581 on the same dataset revision, and the cause of the 0.003 difference was not found.

## Performance

One text per call (write the inputs, run, read the 768-d vector back), median of 20 calls after 5 warm-ups, on a rested phone (thermal status 0, no CPU frequency cap) unless the row says otherwise. Every file is a static window, so the time is the window's, not the text's: an app picks the smallest window that holds the text.

| device · backend | file | L64 | L128 | L256 | L512 | cosine floor vs the reference |
|---|---|---|---|---|---|---|
| Galaxy S26 · NPU (Qualcomm HTP, LiteRT 2.2.0 C-API, JIT) | `_embeds_fp16` + host table | 6.8 ms | 13.1 ms | 28.0 ms | 68.4 ms | ≥ 0.9994 on every window, top-1 100 % (phone not rested) |
| Galaxy S26 · NPU | `_fp16` one file (the HTP takes the fp16 table) | 6.8 ms | 13.1 ms | 29.0 ms | 71.0 ms | the same vectors (phone not rested) |
| Galaxy S26 · GPU (Adreno, FP32 arithmetic) | `_embeds_fp16` + host table | 16.5 ms | 24.0 ms | 47 ms | 120 ms | ≥ 0.99997 |
| Galaxy S26 · CPU (XNNPACK, 4 threads) | `_fp16` one file | 19.8 ms | 39–41 ms | 85–88 ms | 210–237 ms | ≥ 0.99997 |
| Pixel 8a · GPU (Mali-G715, default precision = FP16 arithmetic) | `_embeds_fp16` + host table | 55 ms | 68 ms | 103 ms | 209 ms | ≥ 0.9960 (medians over the accuracy runs, phone not rested) |
| Pixel 8a · GPU (Mali-G715, FP32) | `_embeds_fp16` + host table | 76 ms | 92 ms | 181 ms | 408 ms | ≥ 0.99997 (medians over the accuracy runs, phone not rested) |
| Pixel 8a · CPU (4 threads) | `_fp16` one file | 181 ms | 473 ms | 936 ms | 2,045 ms | ≥ 0.99997 (medians over the accuracy runs) |
| Mac M4 Max · Metal (default precision = FP16 arithmetic) | `_embeds_fp16` + host table | 6.1 ms | 7.7 ms | 10.2 ms | 18.4 ms | 0.9908 on 3,918–4,184 texts |
| Mac M4 Max · Metal (FP32 arithmetic) | `_embeds_fp16` + host table | 6.1 ms | 7.3 ms | 11.0 ms | 20.2 ms | 0.99977 on 3,918–4,184 texts |
| Mac M4 Max · CPU (8 threads) | `_fp16` one file | 22 ms | 30 ms | 45 ms | 79 ms | 0.99977 |

NPU JIT compile on the S26: 4–34 s per window on the first run (cache it). A GPU can run the `_embeds_fp16` files at its default precision (float16 arithmetic) with search results within 0.003 of the reference (Accuracy, above). On the Pixel 8a that takes 55–209 ms against 76–408 ms at FP32, and on the Mac 10.2 and 18.4 ms against 11.0 and 20.2 ms at L256 and L512 (the same at L64, slower at L128). On the Galaxy S26 the NPU stays the fastest; its GPU at the default precision took 11.5 ms at L64 against 17.0 ms at FP32 in the small LiteRT C-API program that also produced the NPU rows, in runs that did not wait for a rested phone (the CPU frequency capped, and in the FP32 run the GPU clock limited too), and the other windows were not timed at that precision on a rested phone. FP16_WITH_FP32_ACCUM (float16 arithmetic with float32 accumulation, between the two in agreement with the reference) took 84 ms at L512 on the S26 (FP32: 120 ms). Int8 weights were tried and rejected (see Conversion).

## Usage

Python (LiteRT 2.2.0, `ai-edge-litert`; `tokenizers`):

```python
import numpy as np
from ai_edge_litert.compiled_model import CompiledModel, CpuOptions, HardwareAccelerator, Options
from tokenizers import Tokenizer

L = 64                                                      # the file's window: vela_L64_fp16.tflite takes up to 64 tokens
tok = Tokenizer.from_file("tokenizer.json")                 # the repo's tokenizer.json (adds <bos> = 2 and <eos> = 1)
model = CompiledModel.from_file("vela_L64_fp16.tflite", options=Options(HardwareAccelerator.CPU, cpu_options=CpuOptions(num_threads=4)))
sig = list(model.get_signature_list())[0]                  # "serving_default"
ins = {n: model.create_input_buffer_by_name(sig, n) for n in model.get_input_tensor_details(sig)}
outs = {n: model.create_output_buffer_by_name(sig, n) for n in model.get_output_tensor_details(sig)}


def embed(text: str) -> np.ndarray:
    ids = tok.encode(text.strip()).ids                      # strip first: sentence-transformers does, and a trailing space adds a token
    assert len(ids) <= L, f"{len(ids)} tokens > window {L}: use the next window file"
    input_ids = np.zeros((1, L), np.int32); input_ids[0, : len(ids)] = ids
    mask = np.zeros((1, L), np.float32); mask[0, : len(ids)] = 1.0
    ins["input_ids"].write(input_ids); ins["attention_mask"].write(mask)
    model.run_by_name(sig, ins, outs)
    return np.asarray(outs["embedding"].read(768, np.float32), dtype=np.float32).reshape(768)   # L2-normalized, 768-d


q, d1, d2 = "How do I reset my password?", "Steps to change a forgotten password", "Best pizza toppings for a party"
e = [embed(t) for t in (q, d1, d2)]
print("cos(q, d1) =", float(e[0] @ e[1]), " cos(q, d2) =", float(e[0] @ e[2]))
```

Run on 2026-10-18 on the fp16 L64 file (`card/snippet_check.py`, the same code against the files of this conversion): `cos(q, d1) = 0.842  cos(q, d2) = -0.007`, and the vectors of 8 check-set texts match the publisher's stack at cosine 1.000000.

Android: the `.litertlm` bundle runs through LiteRT-LM's `EmbeddingEngine` (`EmbeddingEngineConfig(modelPath)` → `computeEmbedding(text)`; leave `insertSpecialTokens` at its default, the bundle declares `<bos>` 2 and `<eos>` 1). The `.tflite` files run through LiteRT `CompiledModel` on the CPU or GPU with the host doing the tokenization (the repo's `tokenizer.json`) and, for the `_embeds_` files, the table lookup.

## Matryoshka

The checkpoint was trained with Matryoshka dimensions 64 / 128 / 256 / 512 / 768. The graph returns the full 768-d normalized vector; to use a smaller dimension, take the first d values and re-normalize on the host (sentence-transformers `truncate_dim` does the same).

## Source and licence

Model: [vllm-sr/Vela-1.0-Encoder-307M-Embedding](https://huggingface.co/vllm-sr/Vela-1.0-Encoder-307M-Embedding), Apache-2.0 — this conversion keeps the licence. Conversion scripts and the full record: [hf-to-litertlm](https://github.com/john-rocky/hf-to-litertlm).
