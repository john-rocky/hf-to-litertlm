# Re-authoring a vision tower in plain PyTorch for the fast_vlm contract

LiteRT-LM's `fast_vlm` path feeds one fixed-size image to VISION_ENCODER, then VISION_ADAPTER, and splices
the N embeddings where the chat template writes the image token. A tower written for dynamic resolution
(window index tables, `cu_seqlens`, gathers) exports badly; a fixed grid lets every index become a constant.
These are the steps that turned MiniCPM-V-4.7-1B's 27-layer window-merger ViT (448×448 → 32×32 patches →
64 soft tokens) into two graphs that passed parity on the first export (`minicpmv47_work/mv47_vision.py`,
`vision_export.py`, `vision_parity_eager.py`; the record is the 2026-10-13 entry in REPRODUCE.md).

1. **Positions as a constant.** Slice the position-embedding table to the fixed grid once, in Python: the
   graph holds the 1024 rows it needs out of a 4900-row table, and no GATHER.
2. **Window attention without permutation.** Keep raster order and add a constant block-diagonal additive
   mask: 0 inside a window, −1e4 outside. −1e9 becomes −inf in fp16, and a softmax that meets an all-−inf
   prefix produces NaN; in fp32 the two give the same result. Check the mask against the reference's window
   partition (here the reference's window order is the raster order of the window grid).
3. **2×2 merges as strided slices.** Four strided slices and a concat on the raster layout reproduce the
   reference's view/permute. converter 0.4.0 lowers them as slices; converter 0.3.1 turned the same slices
   into GATHER_ND, which no mobile GPU delegate takes.
4. **LayerNorm in fp16.** Scale each LayerNorm input by 1/S with S a power of two from a calibration pass
   over a few images (S = 2^max(0, ceil(log2(absmax / 8)))), normalise with eps/S², keep the affine as is
   — exact in real arithmetic — and clamp the LN → FC path to ±65504. MiniCPM-V-4.7: all 58 LayerNorms
   needed S > 1; from layer 10 the input absmax is 1302–1509 and S = 2^8.
5. **Gates.** The torch re-author against the model's own image-feature function in fp32: correlation
   ≥ 0.9999 on every fixture (here max |Δ| 5.9e-8). The exported graphs: op histogram with GATHER, FLEX,
   CUSTOM and STABLEHLO at 0, and end to end against the fp32 reference ≥ 0.999 per fixture (fp16 weights:
   1.000000, per-token cosine at worst 0.999996).
6. **Quantization.** Keep the adapter fp16. A dynamic-int8 adapter loses accuracy on the CPU through
   run-time activation quantization (end to end 0.9969–0.9989); weight-only int8 keeps it (0.99974–0.99991)
   but its DEQUANTIZE → fp32 FC form fails at GPU engine creation. An int8 encoder is the 8 GB-phone form:
   on a Pixel 8a (CPU) it encodes in 1.3–1.6 s and peaks at 2.3–2.6 GB where the fp16 encoder takes
   5.3–7.0 s and 3.9–4.3 GB, and a Mali phone rebooted on an fp16 encoder in an earlier conversion
   (Qwen3.5-2B-VL).
7. **Ranks.** Converters lower Linear, mean(keepdim) and 4-D matmul to rank-2 or rank-3 tensors even when
   the module keeps a leading batch axis (537 such float activations in this encoder). Read them for the
   mobile-GPU trap classes (LiteRT #10445: a rank-2 elementwise op against a rank-2 or rank-3 constant, a
   rank-3 PAD on a non-innermost axis, an int8 FULLY_CONNECTED feeding the residual ADD), not for the rank
   of every activation; this encoder had none of the elementwise form. A static pre-flight that matches these
   forms against a measured compatibility matrix is `edge-lint` in
   [edge-compat](https://github.com/john-rocky/edge-compat); run it on the `.tflite` before the device leg.
8. **Contract cost, measured first.** The runtime squares the image to the fixed size and feeds 1-D
   positions. Run the upstream fp32 model three ways before deciding that a derived-position graph is
   needed: processor-native, the fixed square, and the fixed square with 1-D positions. MiniCPM-V-4.7, 9
   image questions: paraphrases only, no change of content, so the plain contract shipped.
