# Audio8-TTS-Preview-0.6b -> LiteRT: running notes (2026-09-28, session B)

Source: Edge0/Audio8-TTS-Preview-0.6b @ f07040f3 (HF org Edge0; the README's "Audio8/" org does not exist on the Hub).
Oracle: vendor remote code, transformers 4.57.6 / torch 2.12.1 (~/parakeet-env), CPU fp32, seeded torch.Generator,
14 cases (6 en + 6 ja with a cloned reference voice, 1 en + 1 ja without reference). audio8_tts_work/oracle_ref.py.

## Verified facts
- Port (arktts_port.py, weights straight from safetensors/codec.pth) vs oracle, teacher-forced: slow logits max|d| 3e-5
  (scale ~50), fast logits 6e-5, argmax 100%; RoPE tables bit-identical to the vendor's bf16-rounded freqs_cis;
  codec decoder wav max|d| 0.0 (bit-exact), codec encoder codes 100% equal.
- fp32 .tflite vs oracle: slow logits max|d| 1e-4 argmax 100%; fast 2.4e-4 argmax 100%; codec wav max|d| 1.9e-6.
- fp32 end-to-end host loop (same generator draw order as the vendor sampler) reproduces the oracle's code sequence
  frame-for-frame on all 14 cases (first divergence = sequence length everywhere).
- GQA: `repeat_interleave` on the KV cache became BROADCAST_TO (48/step) and made decode 615 ms/step (thread-count
  independent). Folding the 7 query heads that share a kv head into the matmul row dim removed it: 16 ms/step fp32
  (4 threads, Mac). fast AR call 2.84 -> 1.96 ms.
- lm_head is sliced to 4097 rows (semantic range + eos) = what the sampler can pick; same layout as the publisher's ONNX.
- Codec windowing: the decoder is causal but the post-transformer has a 128-frame window, so a window with only 32
  frames of left context is NOT exact (corr 0.96-0.999 vs full decode); see ctx sweep below.
- CompiledModel (litert 2.2.0) runs the graphs; buffers follow the SUBGRAPH input order (hidden, token, use_hidden,
  pos, mask, k_all, v_all), not the alphabetical signature order.

## Quantization scan (Mac, 4 threads)
- fast AR dynamic int8 (per-channel): logits max|d| 4.9 (top1-top2 gap ~1.1, std 14.5), argmax 86%; every block
  contributes ~2-3 (no single culprit); blockwise-32 OCTAV int4: max|d| 27, argmax 40%. fp16 weights: exact but no
  speedup (XNNPACK unpacks to fp32). Calls: fp32 1.96 ms / int8 0.90 / int4 0.66.
- slow AR teacher-forced: dynamic int8 logits max|d| 2.2 argmax 88%, decode 10.7 ms/step; bo4 max|d| 5.5-7.7
  argmax 66-76%, 10-14 ms/step.
- codec: fp16 weights corr 1.000000 max|d| 1.3e-3 (261 MB, 526 vs 657 ms @T64); aeq dynamic int8 on CONV_2D fails
  ("input operand has more dimensions than allowed by the axis remapping", weight [O,1,K,I]); FC-only int8 corr
  0.99916 max|d| 6.5e-2 (307 MB, 483 ms).

## Quantized end-to-end gate (Mac, whisper large-v3-turbo / TitaNet-L; 6+1 en, 6+1 ja cases; fp16 codec T128)
| variant | en WER | ja CER | spk cos vs ref en/ja | frames vs oracle |
|---|---|---|---|---|
| oracle (PyTorch fp32) | 1.1% (1/94) | 0.0% | 0.657 / 0.744 | - |
| fp32 tflite | 1.1% | 0.0% | 0.655 / 0.735 | identical code sequences |
| slow int8 + fast int8 | 1.1% | 0.0% | 0.678 / 0.755 | 100.7% |
| slow int4(b32 OCTAV) + fast int8 | 1.1% | 0.0% | 0.630 / 0.767 | 103.5% |
| slow int8 + fast fp32 | 3.2% (riverbank word-merge by ASR) | 0.0% | 0.662 / 0.753 | 100.5% |
| slow fp32 + fast int8 | 1.1% | 0.0% | 0.654 / 0.749 | 101.3% |
| slow fp32 + fast int4 | 1.1% | 0.0% | 0.634 / 0.691 | 101.8% |
The only en error is shared with the oracle (the model drops the first word of en_ref_2). fast int4 lowers ja speaker
similarity (0.744 -> 0.691) -> not shipped. slow int8 + fast int8 is the primary ship; slow int4 is the small option.

## Mac RTF (M-series, 4 threads, load avg 2.8-4.6 during the int8 run; the int4 run's tail overlapped a load spike)
int8/int8/fp16-T128: slow 9.2 ms/frame, fast 9.7 ms/frame (10 calls), codec 1.50 s per call, RTF median 0.84 (0.77-1.13).
int4/int8/fp16-T128: slow 12.6 ms/frame, RTF 0.96.

## S26 (SM-S942Q, LiteRT benchmark_model from /data/local/tmp/litert-cli, CPU 4 threads unless noted)
| graph | signature | backend | avg ms | init / overall footprint MB | delegate coverage |
|---|---|---|---|---|---|
| fast int8 (68 MB) | step | CPU | 1.02 | 127 / 127 | 262/283 XNNPACK |
| fast int8 | step | GPU | 2.77 | 183 / 186 | 58/283 CL (GATHER_ND unsupported) |
| slow int8 (558 MB, 3 sigs) | decode | CPU | 12.6 | 1441 / 1541 | 1381/1456 |
| slow int8 | prefill_64 | CPU | 59.9 | 1441 / 1559 | |
| slow int8 | prefill_256 | CPU | 239 | 1441 / 1614 | |
| slow int8 | decode | GPU | 20.5 | 1580 / 1682 | 56/1456 CL |
| slow int4 (389 MB) | decode | CPU | 10.2 | 816 / 915 | |
| slow int4 | prefill_256 | CPU | 409 | 816 / 989 | |
| codec fp16 T128 (v1 attention) | decode | CPU | 2778 | 864 / 1446 | 1184/1246 |
| codec fp16 T128 (v1 attention) | decode | GPU | 1016 | 713 / 713 | 422/1246 CL (BROADCAST_TO, 5-D CONCATENATION from repeat_interleave / interleaved rope) |
Per frame on CPU: 12.6 (slow) + 10.2 (fast) = 22.8 ms -> AR-only RTF 0.49 at 21.5 frames/s.
The init footprint of the 3-signature slow graph is ~2.6x its file size (XNNPACK packs FC weights per signature).

## S26 clean legs (frequency caps verified absent before each row; 2-signature slow graphs = the shipped files)
| graph | sig | backend | avg ms | init / overall MB | notes |
|---|---|---|---|---|---|
| fast int8 (v3) | step | CPU 4t | 0.97 | 127 / 127 | |
| slow int8 (p256) | decode | CPU 4t | 12.3 | 1077 / 1176 | 3-sig file was 1441 MB init |
| slow int8 (p256) | prefill_256 | CPU 4t | 241 | 1077 / 1249 | |
| slow int4 (p256) | decode | CPU 4t | 10.0 | 610 / 709 | |
| slow int4 (p256) | prefill_256 | CPU 4t | 417 | 610 / 783 | int4 prefill slower than int8 |
| codec fp16 T128 g2 | decode | GPU CL | 923 | 1050 | 936/1069 on GPU; CPU side = DEQUANTIZE (fp16 weights) + EMBEDDING_LOOKUP |
| codec fp16 T128 g2 | decode | CPU 4t | 3664 | 862 / 1433 | |
| codec fp16 T256 g2 | decode | GPU | FAIL | | "Dilated im2col buffer size overflowed" (CONV_2D prepare) |
| codec int8 T128 g2 (native, fp32 codebooks) | decode | CPU 4t | 1959 | 259 / 836 | GPU: int8 conv kernel init fails |
| codec encoder fp16 10 s g (unpublished build) | encode | CPU 4t | 2287 | 1206 / 1952 | GPU: ARG_MAX/CAST int64 not supported |
| codec encoder fp16 10 s g2 (= shipped file) | encode | CPU 4t | 2365 (min 2052) | 1206 / 1899 | s26_codec2_bench.log; the card row was corrected to this on 2026-09-28 |
Ship smoke (out/ship, Mac 4t): en 95 frames RTF 0.98, ja 72 frames RTF 1.14, register-voice via fp16 encoder = 99.8% codes vs oracle.

## Appendix — tables that only existed in the session transcript (2026-09-28)
Codec window left-context sweep (T128 windows vs one T256 decode of a 200-frame sequence; error over frames >= 128):
| ctx frames | max|d| | corr | first new frame max|d| | last frame max|d| |
|---|---|---|---|---|
| 32 | 0.494 | 0.698 | 0.161 | 0.062 |
| 64 | 0.334 | 0.966 | 0.052 | 0.065 |
| 96 | 0.333 | 0.973 | 0.041 | 0.062 |
| 112 | 0.074 | 0.995 | 0.019 | 0.019 |
| 120 | 0.126 | 0.995 | 0.019 | 0.050 |
| 124 | 0.078 | 0.996 | 0.009 | 0.050 |
| 127 | 0.080 | 0.997 | 0.003 | 0.050 |
No fixed context is sample-exact (8 stacked 128-frame windows compound the receptive field); the residual at ctx >= 112 is ~0.08 on a +-1 waveform.

Fast AR int8 per-group sensitivity (one group quantized at a time, 3 cases x 8 frames x 9 steps vs oracle logits; fp32 = 216/216):
| group | file MB | max|d| | mean|d| | argmax match |
|---|---|---|---|---|
| embedding only | 257 | 0.47 | 0.023 | 212/216 |
| head only | 257 | 0.21 | 0.021 | 211/216 |
| block 0 | 223 | 3.05 | 0.186 | 195/216 |
| block 1 | 223 | 2.87 | 0.162 | 202/216 |
| block 2 | 223 | 2.14 | 0.126 | 205/216 |
| block 3 | 223 | 1.66 | 0.098 | 202/216 |
| all FC (emb fp32) | 68 | 4.86 | 0.298 | 185/216 |
| all FC + emb (head fp32) | 79 | 4.86 | 0.297 | 185/216 |
Error accumulates across blocks; it is the dynamic activation quantization, not one tensor.
