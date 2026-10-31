# Reproduce — Vela-1.0-Encoder-307M-Embedding → LiteRT

Environments: a Python 3.12 with torch 2.14.0, litert-torch 0.10.0, litert-converter 0.5.0.dev20260912, ai-edge-litert 2.2.0, ai-edge-quantizer-nightly 0.10.0.dev20260914, tokenizers 0.23.2, numpy 2.5.3 (the export and gate venv, `P` below); a second venv layered on it for the reference with `transformers==4.57.6` (the repo's own `transformers_version`), `sentence-transformers==5.1.2`, `tokenizers==0.22.2`, `huggingface-hub<1.0`, `pyarrow` (`REF` below); litert-lm 0.18.0 + numpy for the bundle gate (`LM`). Apple M4 Max, macOS 27.0. The scripts are run from a work directory `W` that holds `hf/` (the checkpoint), `fixtures/`, `oracle/`, `exports/`, `results/`; every result file is written once.

```bash
cd W; S=/path/to/hf-to-litertlm/vela_work/conversion
bash $S/dl_model.sh                                        # 1. vllm-sr/Vela-1.0-Encoder-307M-Embedding @ 290318fd into hf/ (sha256 of model.safetensors
                                                           #    0ad845c4b4d09c6cf45b5a3233cfde4dfa936a939cc8e48f7fdb6532569aa537)
$REF $S/make_fixtures.py                                   # 2. fixtures/texts.json from paws-x @ 4cd8187c (en 0-299, es/ja/zh 0-99 + long concatenations)
                                                           #    and mteb/STSBenchmarkv2 @ 93b628c3 (all 1,362 test pairs): 4,184 stripped texts
$REF $S/reference.py --threads 8                           # 3. the publisher's stack, CPU fp32, SDPA math: oracle/embeddings.npz, oracle/rope_L512.npz (the
                                                           #    YaRN cos/sin tables the graph uses as constants), oracle/fp16_headroom.json
SHIFT=attn12-21=6,mlp12-21=6,final=6,attn9-11=2,mlp9-11=2  # the LayerNorm rewrite (the scan: layers 12-21 + final reach a sum of squares of 4e7)
for L in 64 128 256 512; do
  $P $S/torch_parity.py --threads 8 --L $L --ln-shift $SHIFT --pool arith     # 4. the graph vs the reference (cos min 0.9999998, embeds bit-equal)
  for form in ids embeds engine; do $P $S/export.py --L $L --form $form --ln-shift $SHIFT --pool arith; done   # 5. fp32 exports (6-8 s each)
  $P $S/quantize.py --src exports/vela_L${L}_ids_fp32_shift_arith.tflite --variant v3_fp16fc_fp16emb          # the one-file fp16 form (614 MB)
  $P $S/quantize.py --src exports/vela_L${L}_embeds_fp32_shift_arith.tflite --variant wfp16                    # the host-table form (221 MB)
  $P $S/quantize.py --src exports/vela_L${L}_engine_fp32_shift_arith.tflite --variant wfp16
done
$P $S/tables.py                                            # exports/tables/tok_embeddings_fp16.bin (393 MB, the checkpoint's F32 table rounded to float16)
$P $S/export.py --lookup && $P $S/quantize.py --src exports/vela_lookup_fp32.tflite --variant wfp16_lookup    # the fp16 embedder of the bundle
$P $S/merge_encoders.py --Ls 64,128,256,512 --ln-shift $SHIFT --out exports/engine/vela_encoder_fp32.tflite   # encoder_64..512 in one trace
$P $S/quantize.py --src exports/engine/vela_encoder_fp32.tflite --variant wfp16
bash $S/pack_engine.sh exports/vela_lookup_wfp16_lookup.tflite exports/engine/vela_encoder_wfp16.tflite exports/engine/Vela-1.0-Encoder-307M-Embedding_fp16.litertlm
$LM $S/bundle_sections.py exports/engine/Vela-1.0-Encoder-307M-Embedding_fp16.litertlm hf/tokenizer.json     # the tokenizer section == the repo's file
# 6. gates on the Mac: every fitting fixture row of a window (3,918 at L64 ... 4,184 at L512)
for L in 64 128 256 512; do
  $P $S/litert_parity.py --tflite exports/vela_L${L}_ids_v3_fp16fc_fp16emb_shift_arith.tflite --L $L --accel cpu --threads 8
  $P $S/litert_parity.py --tflite exports/vela_L${L}_embeds_wfp16_shift_arith.tflite --L $L --accel cpu --threads 8 --table exports/tables/tok_embeddings_fp16.bin
  $P $S/litert_parity.py --tflite exports/vela_L${L}_embeds_wfp16_shift_arith.tflite --L $L --accel gpu --f32 --table exports/tables/tok_embeddings_fp16.bin --timing
  $P $S/litert_parity.py --tflite exports/vela_L${L}_embeds_wfp16_shift_arith.tflite --L $L --accel gpu --table exports/tables/tok_embeddings_fp16.bin --timing   # fp16 arithmetic: misses the per-row floor (0.991); SciFact retrieval within 0.003 of the reference (the card's Accuracy)
done
$LM $S/engine_gate.py --bundle exports/engine/Vela-1.0-Encoder-307M-Embedding_fp16.litertlm --backend cpu                  # PASS, 4,184 texts
$LM $S/engine_gate.py --bundle exports/engine/Vela-1.0-Encoder-307M-Embedding_fp16.litertlm --backend cpu --no-special-tokens --limit 40   # the control: must FAIL
$P ../card/snippet_check.py --model exports/vela_L64_ids_v3_fp16fc_fp16emb_shift_arith.tflite   # the card's usage block as a script
```

The published names: `vela_L*_fp16.tflite` = `exports/vela_L*_ids_v3_fp16fc_fp16emb_shift_arith.tflite`, `vela_L*_embeds_fp16.tflite` = `exports/vela_L*_embeds_wfp16_shift_arith.tflite`, `tok_embeddings_fp16.bin` (+ `.json`), `Vela-1.0-Encoder-307M-Embedding_fp16.litertlm`, `tokenizer.json` (the repo's file verbatim). The bundle's encoder is the `where`-pooling trace (it runs on the CPU through EmbeddingEngine); a GPU-backed engine would take a re-pack from `merge_encoders.py --pool arith`.

## What was measured, and what was rejected

- **Reference and bar.** The publisher's stack reproduces the card's STS-B v2 Spearman on the same 1,362 pairs (0.811867 vs 0.811864). The bar is per text (cosine ≥ 0.999 on every row) plus retrieval top-1 agreement, pair Spearman, mask liveness and finiteness — a vector contract, not a task metric: the task metrics (STS-B Δ ≤ 2e-4) hide every form that fails below.
- **fp32 export:** cosine 0.99999977 to the reference on every window — the converter's noise floor.
- **fp16 weights (FC + table):** 0.99977 on every window, top-1 100 %, Spearman 0.9999995. **The int8 table fails** (0.9929, 19 of 3,918 texts): a per-row symmetric int8 table rounds every row by 0.75 % on average and the control tokens by 2–3.6 %; the PyTorch graph with the simulated table reproduces the LiteRT number, and keeping the 16 / 256 / 2,048 worst rows exact does not lift the floor (0.99516) — a sensitive short text with ordinary tokens at 0.7–0.8 % each. **int8 FC weights fail harder** (0.964, 27 top-1 flips). Everything shipped is float16.
- **The GPU delegate (LiteRT 2.2.0, Metal / Adreno / Mali) refuses two idioms:** `torch.where(mask > 0.5, h, 0)` pooling (`SELECT_V2: Cond must be float or bool type, if, else tensors must be either be same the shape as output or constant, scalar`; the compile then fails on every GPU) and a float16 token table inside the graph (`EMBEDDING_LOOKUP: Empty quantization params`). Pooling written as `(h·m).sum / max(m.sum, ε)` and `x · rsqrt(max(Σx², ε²))` is bit-identical on the 0 / 1 mask and compiles everywhere; the table goes to the host for the GPU (`_embeds_` files). The Qualcomm HTP takes both forms whole (1,689 / 1,689 ops, the fp16 table included).
- **GPU fp16 arithmetic misses the per-row floor on all three GPUs** (Metal 0.991–0.993, Adreno 0.9974–0.9983, Mali 0.9960–0.9978; about 1 % of the texts, every top-1 and the pair ranking kept); FP32 arithmetic passes with the CPU's floor. FP16_WITH_FP32_ACCUM passes the full bar on L128–L256 on both phones (Mali also L64; Adreno misses L64 by one short text, Mali the L512 long-pair Spearman by 0.001). The HTP's fp16 arithmetic passes everywhere (0.9994–0.9996).
- **Measured, one text per call (write the inputs, run, read the vector back), median of 20 after 5 warm-ups, rested phone.** Galaxy S26 (SM-S942Q, Android 16, LiteRT 2.2.0): HTP (C-API runner, JIT; the phone not rested: thermal status 0–2, the CPU frequency capped in the host-table runs) 6.8 / 13.1 / 28.0 / 68.4 ms at L64 / 128 / 256 / 512 (one-file form 6.8 / 13.1 / 29.0 / 71.0; JIT compile 4–40 s per window), GPU FP32 16.5 / 24.0 / 47 / 120 ms, CPU 4 threads 19.8 / 40 / 87 / 224 ms. Pixel 8a (Mali-G715, gate-leg medians on a working phone): GPU FP32 76 / 92 / 181 / 408 ms, CPU 4 threads 181 / 473 / 936 / 2,045 ms. Mac M4 Max: Metal FP32 6.1 / 7.3 / 11.0 / 20.2 ms (fp16 6.1 / 7.7 / 10.2 / 18.4), CPU 8 threads 22 / 30 / 45 / 79 ms. The time is the window's, not the text's: the app picks the smallest window that holds the text.
- **EmbeddingEngine bundle** (litert-lm 0.18.0, 623.6 MB): 4,184 texts through `compute_embedding`, cosine min 0.99977, top-1 1,873 / 1,873; with `insert_special_tokens` off the floor is 0.308 — the bundle's bos 2 / eos 1 metadata is what makes the engine's text path equal sentence-transformers'.
