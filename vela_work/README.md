# Vela-1.0-Encoder-307M-Embedding → LiteRT: a ModernBERT sentence embedder as static-window graphs, with its Galaxy S26 NPU, GPU and CPU numbers

Published: [litert-community/Vela-1.0-Encoder-307M-Embedding-LiteRT](https://huggingface.co/litert-community/Vela-1.0-Encoder-307M-Embedding-LiteRT).

## What the model is

[vllm-sr/Vela-1.0-Encoder-307M-Embedding](https://huggingface.co/vllm-sr/Vela-1.0-Encoder-307M-Embedding) (Apache-2.0, revision `290318fd`) is a 22-layer ModernBERT encoder (768-d, 12 heads, GeGLU, the Gemma 256k tokenizer, alternating global / sliding-window attention, YaRN RoPE) trained as a sentence embedder: mean pooling over the tokens, then L2 normalization, Matryoshka 64–768. The sentence-transformers pipeline strips the text, tokenizes with `<bos>` … `<eos>`, pools and normalizes.

## What this folder holds

| File | Content |
|---|---|
| `conversion/graph.py` | The export graph: the encoder written out (attention, RoPE from the publisher's own cos / sin tables, masks with NEG = −1e4 and the diagonal always allowed), LayerNorms of layers 12–21 and the final norm computed on x·2⁻⁶ (eps·2⁻¹²; layers 9–11 on x·2⁻²) so float16 storage paths stay finite, mean pooling + normalization in the graph. Two pooling forms: `where` (torch.where + F.normalize, CPU) and `arith` (MUL / SUM / MAXIMUM / RSQRT — the GPU delegate refuses the broadcast SELECT_V2 of torch.where). Three input forms: `input_ids` (one file), `inputs_embeds` (the host looks the token rows up), and the LiteRT-LM EmbeddingEngine encoder signature. |
| `conversion/export.py`, `merge_encoders.py` | litert-torch 0.10.0 exports, one static window each (L64 / 128 / 256 / 512), the four-signature encoder of the `.litertlm` bundle; each export is smoke-tested on the CPU against the PyTorch graph. |
| `conversion/quantize.py`, `tables.py` | ai-edge-quantizer weight-only variants (fp16 FC + fp16 table; fp16 FC + int8 table; int8 FC; fp16 FC with the table on the host); the float16 host table from the checkpoint. The recipes are `phocinae_work/conversion/quantize.py`'s, loaded by file path (`peer.py`). |
| `conversion/pack_engine.sh`, `bundle_sections.py` | The EmbeddingEngine bundle (litert-lm-builder 0.18.0: fp16 embedder + the encoder + the repo's `tokenizer.json` verbatim, metadata `generic_model`, bos 2, eos 1) and the check that its tokenizer section unpacks to the repo's bytes. |
| `conversion/make_fixtures.py`, `reference.py` | The check set (PAWS-X test en / es / ja / zh, STS-B v2 test, concatenated long texts; 4,184 texts, by dataset id + revision + row index) and the reference: the publisher's own stack (sentence-transformers 5.1.2 + transformers 4.57.6, CPU fp32), which reproduces the card's STS-B v2 Spearman (0.811867 vs 0.811864). |
| `conversion/torch_parity.py`, `litert_parity.py`, `embed_gate.py`, `metrics.py`, `engine_gate.py` | The gates: the PyTorch graph vs the reference (plus the int8-table simulation), the `.tflite` files on the Mac CPU / Metal, the bundle through `EmbeddingEngine` with the special-token control. Bar: cosine ≥ 0.999 on every text, top-1 retrieval agreement 100 % on non-near-tie queries, pair Spearman ≥ 0.999, the mask live, every output finite. |
| `conversion/table_int8_analysis.py` | Why the int8 table fails: the per-row rounding error of every vocabulary row and the tokens of the failing texts. |
| `card/README.md`, `card/snippet_check.py` | The published card and its Python usage block as a runnable script. |

The phone measurements (Galaxy S26 GPU / CPU through a CompiledModel gate app, HTP through the LiteRT C-API runner; Pixel 8a GPU / CPU) were taken with a shared device harness that is not part of this repo; the numbers and conditions are in `REPRODUCE.md` and on the card.
