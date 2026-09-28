# Audio8-TTS-Preview-0.6b -> LiteRT (classic .tflite set + host loop) — findings

Session 2026-09-28 (lane [B] conversion #1). Source `Edge0/Audio8-TTS-Preview-0.6b` @ `f07040f3d151f1ba0253bfb92cb2f5dd38b44594`
(HF org **Edge0**; the card's code sample points at an `Audio8/` org that does not exist on the Hub). Apache-2.0, custom
`arktts` remote code, DualAR (Fish Audio S2 Pro lineage): slow AR 24L/896/14h/2kv (Qwen2.5-0.5B geometry, qkv bias),
fast AR 4L/896/14h/2kv predicting 10 codebooks x 4096 per frame, 44.1 kHz codec at 2048 samples/frame (21.5 fps).

Everything below is measured; scripts are in this directory, logs under `out/`.

## 1. Oracle
`oracle_ref.py` on `~/parakeet-env` (transformers 4.57.6 = the vendor's 4.57.x family, torch 2.12.1, CPU fp32), vendor
remote code unmodified, seeded `torch.Generator`, generation_config defaults (T 0.7 / top-p 0.9 / top-k 50 / RAS).
14 cases: 6 en + 6 ja with a cloned reference voice (`fixtures/ref_{en,ja}_44k.wav`, transcripts in `common.py`),
1 en + 1 ja without reference. Dumps per case: prompt [11,P], codes [10,T], per-step slow logits (4097 slice) and fast
hidden, first-8-frame fast logits, teacher-forced full-forward logits, wav. All 14 finish with EOS (59-110 frames).
Trap avoided: after `generate()` the vendor leaves KV caches attached to the layers, so a plain `forward()` raises
"cache_position is required" — detach `layer.attention.kv_cache` first. `freqs_cis` asserted non-zero (tf-5.x meta-load trap
does not apply on 4.57 but the assert stays).

## 2. Port (`arktts_port.py`) — what was rewritten and why it is exact
- Weights read straight from `model.safetensors` (bf16 -> fp32) and `codec.pth` (fp32, weight-norm folded via
  `parametrize.remove_parametrizations` / legacy `remove_weight_norm`). No `from_pretrained` in the export path.
- RoPE: the vendor applies interleaved-pair rotation with a **bf16-rounded** table (`_precompute_rope(...).to(bfloat16)`).
  The port permutes the q/k rows of `wqkv` to [even dims, odd dims] and uses rotate-half with the same bf16-rounded
  cos/sin held as fp32 constants -> tables bit-identical to the vendor buffers, dot products invariant to the shared
  head-dim permutation.
- lm_head sliced to 4097 rows (semantic 151678..155773 + eos 151645): the semantic logits processor masks everything
  else to -inf, so the sampler can never pick another id. Same layout as the publisher's ONNX (`semantic_then_eos`).
- Embedding = text/semantic row + sum of 10 codebook rows (offsets i*4096), masked to semantic positions (vendor `_embed`).
- KV caches are graph I/O updated with `litert_torch...dynamic_update_slice` (contiguous positions). Cache 2048 = max_seq_len.
- **GQA without repeat**: `repeat_interleave(7)` on the cache lowered to BROADCAST_TO (48/step) and made decode
  **615 ms/step**, thread-count independent (Mac fp32). Folding the 7 query heads that share a kv head into the matmul
  row dimension removed it: **16 ms/step**. The same fold is applied to the codec transformers (16h/8kv) where it also
  clears the GPU delegate.
- Codec: the vendor module is imported from the snapshot with three patches: `@torch.jit.script` snake -> plain python,
  `torch.polar` rope table -> eager-cached real (cos,sin) constants (aten.polar has no lowering), attention rewritten as
  above with the causal/window mask served as an eagerly-built additive constant (a traced `repeat()` becomes
  BROADCAST_TO; a data-dependent cache key breaks `torch.export`). Every index that reaches a gather is clamped in-graph
  so random-input benchmark tools cannot fault (`gather_nd index out of bounds` on the S26 before the clamps).

Parity, teacher-forced vs oracle (`verify_port.py`, `verify_tflite_*.py`):
| stage | torch port | fp32 .tflite |
|---|---|---|
| slow logits (scale ~50-65) | max abs diff 3e-5, argmax 100% | 1e-4, argmax 100% |
| fast logits (72 steps/case) | 6e-5, argmax 100% | 2.4e-4, argmax 100% |
| codec decoder wav | 0.0 (bit-exact) | 1.9e-6, corr 1.000000 |
| codec encoder codes | 100% equal | 100% equal |
End-to-end with the fp32 graphs (`hostloop_e2e.py`, same torch.Generator draw order as the vendor sampler: two
full-vocab draws per frame + nine 4096 draws): **all 14 cases reproduce the oracle's code sequence frame for frame**.

## 3. Graph contract (what ships)
| file | signatures | inputs | outputs |
|---|---|---|---|
| slow_ar_{int8,int4}.tflite | prefill_256, decode | codes i32 [1,11,T], input_pos i32 [T], mask f32 [1,1,T,2048] additive, k_i/v_i f32 [1,2,2048,64] x24 | logits f32 [1,4097], hidden f32 [1,1,896], k_i/v_i |
| fast_ar_int8.tflite | step | hidden [1,1,896], token i32 [1], use_hidden f32 [1], pos i32 [1], mask [1,1,1,10], k_all/v_all [4,1,2,10,64] | logits [1,4096], k_all, v_all |
| codec_decoder_fp16_T{128,192}.tflite, codec_decoder_int8_T128.tflite | decode | codes i32 [1,10,T] | wav f32 [1,1,T*2048] |
| codec_encoder_fp16_10s.tflite | encode | audio f32 [1,1,442368] | codes i32 [1,10,216] |
Host loop = `audio8_tts_litert.py` (prompt build, chunked right-padded prefill of prompt[:-1] + decode of the last
prompt token, vendor sampler, 10 fast calls per frame, windowed codec decode). Prefill trick: the last prompt token
always goes through `decode`, so a right-padded prefill chunk never has to produce logits from a pad slot.

## 4. Quantization
Post-hoc `ai_edge_quantizer` on the fp32 graphs (`quantize_ar.py`, `quantize_codec.py`):
- AR graphs: dynamic int8 per-channel FC + int8 EMBEDDING_LOOKUP (drq8); blockwise-32 OCTAV int4 FC + int8 embedding (bo4).
  Embedding int8 is free (FC-only int8 gives the same logits error, `slow_drq8fc_probe.log`).
- Teacher-forced error vs oracle: slow int8 logits max|d| 2.2 / argmax 88%; slow int4 5.5-7.7 / 66-76%; fast int8 4.9 /
  86% (each of the 4 blocks contributes ~2-3, no single culprit — it is the dynamic activation quantization); fast int4 27 /
  40%. fp16 weights are exact but XNNPACK unpacks them to fp32 at init (no speed or RAM gain for the AR graphs).
- Codec: fp16 FLOAT_CASTING = exact (corr 1.000000, max|d| 1.3e-3). aeq dynamic int8 on the first decoder conv
  ([1536,1,7,1024], 44 MB) trips a numpy broadcast bug in aeq's >32 MiB chunked path; excluding that one conv works
  (`drq8x`, 251 MB, corr 0.9984). **Export-time native int8 (PT2E, the A/B partner required by the 2026-07-22 rule)**: the
  converter's PT2E_DYNAMIC mode quantizes the *unannotated* RVQ codebook tables asymmetrically, which TFLite's
  EMBEDDING_LOOKUP kernel refuses at prepare (`zero_point == 0` check); `set_operator_type`/module split do not stop it
  (same behaviour sopro hit on an unannotated head). Fix-up = write the fp32 codebooks back into the flatbuffer
  (`fix_native_i8_tables.py`) -> `i8native` 132 MB, corr 0.998, **1.6x faster than fp16 on Mac CPU** (590 vs 967 ms @T128).
  Native vs post-hoc: same quality (corr 0.9980 vs 0.9983, identical WER, spk cos 0.674/0.724 vs 0.671/0.727), native is
  smaller (all convs int8) and faster -> native ships as the CPU option; on the S26 GPU int8 conv fails kernel init
  ("Unable to parse bc coord for BATCH axis"), so the GPU codec is fp16.

## 5. Quality gate (whisper large-v3-turbo WER/CER vs the input text, TitaNet-L speaker cosine vs the reference clip)
See NOTES.md tables. Summary: every int8 combination matches the oracle's WER (1.1% en = one oracle-shared error, 0.0%
ja) and speaker cosine (en 0.66-0.68 vs oracle 0.657, ja 0.75-0.77 vs 0.744). slow int4 keeps WER, en cosine 0.630.
fast int4 drops ja cosine to 0.691 -> not shipped. Codec int8: WER unchanged, ja cosine 0.727 (fp16 0.744).

## 6. Speed
Mac (Apple silicon, 4 threads, ai-edge-litert 2.2.0 Interpreter; quiet run 07:30-07:33, load avg 3.5-4.9, no peer process
above 20% CPU; shipped files): int8/int8/fp16-T128 -> slow 9.0 ms/frame, fast 9.2 ms/frame (10 calls), codec 1.43 s per
T128 call, RTF median 0.81 (0.73-0.98, 14 cases); int4 slow -> 12.2 ms/frame, RTF 0.92; int8 codec -> 0.96 s/call, RTF 0.70.
Long text (290 frames = 13.5 s, T192 windows with 128-frame context): RTF 0.94.
Galaxy S26 (SM-S942Q, `benchmark_model` from /data/local/tmp/litert-cli, 4 threads, frequency caps checked before each leg):
see NOTES.md / card table. AR per frame 12.3 + 9.7 ms -> AR RTF 0.51; codec fp16 T128 on GPU 923 ms per 5.94 s (936/1069 ops on CL;
the CPU remainder is the fp16 DEQUANTIZE + EMBEDDING_LOOKUP), fp16 T192 GPU 1425 ms per 8.9 s, fp32 T128 fully delegated at the same
924 ms (1462 MB); int8 codec CPU 1959 ms; encoder fp16 CPU 2365 ms per 10 s (the published g2 build; the earlier `_g` build read 2287 ms and was quoted on the card until 2026-09-28 10:5x, corrected).

## 7. Codec windowing
The decoder is causal but its post-transformer stacks 8 sliding-window (128) layers, so a fixed window is never exactly the
offline decode: with T=128 windows and 127 frames of left context the frames after the boundary differ by max|d| 0.08
(corr 0.9968); with 32 frames of context corr 0.70. The publisher's ONNX runtime uses context 128 + a 1-frame guard. Ship:
T128 (one call covers <= 5.9 s) and T192 (<= 8.9 s, or 128-frame-context windows = 64 new frames per call for longer text). T256
is not shipped: it fails GPU prepare on the S26 ("Dilated im2col buffer size overflowed").

## 8. Open items / not done
- No on-device functional run (the classic-tflite host loop is Python; S26 rows are per-graph latency + delegate coverage
  from the LiteRT benchmark tool; functional parity is Mac-only on the same files).
- GPU for the AR graphs: GATHER_ND (3-D codebook gather) and the 5-D mask reshape keep them CPU-only; a 1-D gather + 4-D
  mask tile would be the next step (CPU int8 is the intended path).
- Fast AR = 10 invokes per frame (~10 ms on S26); folding the 9 codebook steps into one graph with host-supplied noise
  would cut invoke overhead but not the 10x weight re-read (bandwidth-bound).
- SmoothQuant-style scaling into RMSNorm/w3/v rows would cut the dynamic-int8 activation error (the fast AR's 4.9) — not
  needed for the gate, noted for an int4 attempt.
