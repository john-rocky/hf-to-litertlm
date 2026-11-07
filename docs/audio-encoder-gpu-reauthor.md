# Re-authoring a speech encoder for LiteRT's GPU delegates

The stock export of granite-speech-5.0-2b's audio encoder (log-mel front end, a 16-layer conformer with 4 s block attention and Shaw relative positions, a 2-layer Q-Former) is refused by the GPU delegate on the Mac (WebGPU, LiteRT-LM 0.19.1; 228 of its 2,476 ops delegated): BROADCAST_TO, GATHER_ND, SELECT_V2, a MEAN with two constant inputs, and the block attention's rank-5 CONCATENATION / RESHAPE. LiteRT-LM asks for full delegation, so the conversation fails to start with INTERNAL (13). The rewrite in [`granite_speech_2b_work/port6.py`](../granite_speech_2b_work/port6.py) keeps the weights and runs whole on PowerVR and Adreno (`LITERT_CL 2674/2674`, one partition) and on the Mac (WebGPU, Metal).

1. Every tensor at rank 4 or lower: block attention as [1, blocks × heads, 200, 128]. After the export, `tools/gpu_graph/tfl_rank4.py rewrite --rules bmm --unique-names` turns back the 38 batch-1 matmuls the converter folds to rank 3.
2. Shaw relative positions by the Transformer-XL skew (pad and reshape) instead of a gather from the [1025, 128] table.
3. Length, frame and token masks in block coordinates by comparisons, with no division and no gather: the converter turns `x / 15` into `x · f32(1/15)`, and `ceil` then goes one up on 97 of the 120 multiples of 15 up to 1,800.
4. The STFT's reflection padding by static slices and a constant one-hot matmul.
5. 0/1 masks applied by MUL instead of SELECT_V2; the Q-Former's learned queries are constants, so their projections are computed once.
6. The window: a multiple of the attention block (200 frames), the Q-Former window (15 frames) and the shrink (10). The encoder does ceil(L / W) · W of work per clip; a 24 s window beats 36 s on clips up to 24 s.

Measured: one 24 s window takes 114 ms on the Mac's Metal against 476 ms on XNNPACK at 4 threads; on a Galaxy S26 the 9.9 s clip's first text is 1.21 s with the LM on the GPU (`granite-speech-5.0-2b_ekv2048.litertlm`). The transcripts pass the 220-clip host gate and the 38-clip phone gates.
