# EmbeddingEngine bundles (litert-lm >= 0.17.0)

litert-lm 0.17.0 is the first release whose runtime hosts embedding models directly — `EmbeddingEngine` in Python (`litert_lm.embedding_engine`), C and Kotlin (`litertlm-android`). Its bundle contract, read from `runtime/executor/embedding_litert_compiled_model_executor.cc` and `runtime/testdata/test_embedding.litertlm` and verified by running, is two graphs rather than the single `(input_ids, attention_mask)` graph the plain `.tflite` encoders in this repo use:

| section | contract |
|---|---|
| `TF_LITE_EMBEDDER` | one signature, ONE input `token` int32[1], output f32 `[1,1,D]` — the token-embedding row; the runtime calls it per token |
| `TF_LITE_TEXT_ENCODER` | signatures named `encoder_<S>`, inputs `embeddings` f32 `[1,S,D]` + `input_mask` f32 `[1,S]` (runtime writes 1/0), output `[1,dim]` — pooling and normalization **in-graph**; the smallest signature >= the token count is used |
| tokenizer | `tokenizer.json`; the runtime does **not** run its post-processor, so a model whose CLS position expects `<bos>` declares it in `EmbeddingMetadata.bos_token` and the engine's `insert_special_tokens` (default true) inserts it |

`embedding_engine_common.py` holds the shared pieces (the lookup module, a per-token driver that mimics the runtime, the trace-sample helper). Each `convert_*_engine.py` reuses its fused export's model loading and mask construction unchanged and only splits the graph; `pack_*.sh` builds the bundle with `litert-lm-builder` (>= 0.17.0).

| model | script | pooling | specials | bundle (wi8fc) |
|---|---|---|---|---|
| granite-embedding-311m-multilingual-r2 | `convert_granite_embedding_r2_engine.py` + `pack_granite_embedding_r2.sh` | CLS | `<bos>` (id 2) declared as bos | 332 MB |
| LFM2.5-Embedding-350M | `convert_lfm25_embedding_engine.py` + `pack_lfm25_embedding.sh` | CLS | `<\|startoftext\|>` (id 1) declared as bos | 370 MB |
| Nemotron-3-Embed-1B | `convert_nemotron3_embed_engine.py` + `pack_nemotron3_embed.sh` | mean (prompt included) | none | 1166 MB |

Parity gate (two venvs, because the exporter pins litert-torch 0.9.2 and the engine needs litert-lm 0.17.0): `gate_tflite_embed_half.py` embeds a text list through the shipped `.tflite` with the HF tokenizer, `gate_embedding_engine.py` embeds the same list through the bundle, `compare_embedding_gate.py` prints per-text cosine. All three bundles: cosine 1.000000 against their `.tflite` on 10 texts, on a Mac (Python) and on a Galaxy S26 (Kotlin `EmbeddingEngine`, litertlm-android 0.17.0).

Two traps worth knowing before exporting another model this way:
- Sample `embeddings` for the trace must be ordinary tensors — rows produced under `torch.inference_mode()` make `torch.export` fail ("Inference tensors cannot be saved for backward").
- The runtime fills only the first `len(tokens)` rows of the `embeddings` buffer; pad rows hold whatever was there. Graphs that zero pads by multiplying with the mask (LFM2.5's ShortConv path, mean pooling) must instead replace pad rows with `torch.where`, or a non-finite stale row becomes NaN.
