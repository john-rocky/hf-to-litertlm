# GPU graph tools

These scripts rewrite the prefill/decode graph of an existing `.litertlm` bundle without re-exporting the model. Every original weight buffer keeps its bytes. [REPRODUCE.md](../../REPRODUCE.md) lists the files re-shipped with them, the checks and the speeds: eight in the section "2026-10-06 — GPU graph re-ship" and two in "2026-11-14 — GPU graph re-ship".

## What the scripts do

`gpu_graph_retrofit.py` converts a graph exported without litert-torch's `--apply_gpu_composites` into the shape the flag produces. Both shapes have the same KV layout and the same weights. The script makes three changes:

- The two DYNAMIC_UPDATE_SLICE KV-cache writes per layer become one STABLEHLO_COMPOSITE `odml.cache_update`.
- Each BATCH_MATMUL(adjY) attention product becomes a STABLEHLO_COMPOSITE `odml.runtime_bmm`.
- A signature input `param_tensor` INT32[1,1,1,7] is added, and the runtime fills in start and end.

The re-shipped files use `--mask add_bcast`. With it, the attention mask stays a FLOAT32 input and is added, by broadcast, to a [bk, g, T, C] view of the logits. The script's docstring describes the other `--mask` and `--decomp` modes.

`prefill_bucket_clone.py build` copies an existing prefill signature (`--source`, e.g. `prefill_128`) at the lengths given in `--lengths`. The copy shares every weight buffer, so the file grows by graph structure only. The new subgraphs go after the existing ones, and every existing subgraph and SignatureDef keeps its index and its bytes.

## Commands

Step 1 takes a bundle exported without `--apply_gpu_composites`; step 2 adds a 16-token prefill signature copied from `prefill_128`.

```sh
export LITERT_LM_CLI=/path/to/litert-lm
python tools/gpu_graph/gpu_graph_retrofit.py previous.litertlm retrofit.litertlm --decomp exporter --mask add_bcast
python tools/gpu_graph/prefill_bucket_clone.py build retrofit.litertlm new.litertlm --lengths 16 --source prefill_128 --report bucket.json
```

`LITERT_LM_CLI` is the path to a `litert-lm` CLI (the Python package `litert-lm` from [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM)). On a `.litertlm`, each script uses it to unpack the bundle, rewrites only the prefill/decode section, and packs the bundle again. It then checks that every other section is byte-identical. The Mac speed numbers in REPRODUCE.md were measured with the `litert-lm` 0.17.1 CLI from pip.

In the re-ship, the Phi-4-mini-reasoning file went through step 1 only. `SmolLM3-3B.litertlm` has only `prefill_256`, so its step 2 used `--source prefill_256`.

## Rebuild check

The copies of `gpu_graph_retrofit.py`, `prefill_bucket_clone.py` and `shape_variant.py` in this directory rebuild all eight files of the 2026-10-06 section and the two of the 2026-11-14 section from the previous files (Hub commits in REPRODUCE.md). Every bundle section of each rebuild is byte-identical to the shipped file (sha256 per section). Only the bundle header differs because `litert-lm pack` writes a new uuid and timestamp, so the whole-file sha256 differs. Each rebuild took 6–32 s on an M4 Max.

## Limits

- The retrofit with `--mask add_bcast` was verified only on the ten files in REPRODUCE.md (the 2026-10-06 and 2026-11-14 sections). The 16-token prefill signature was verified only on seven of them, and the 1,024-token signature copied from `prefill_512` only on SmolVLM2-500M. All ten were exported without `--apply_gpu_composites`. Nothing is claimed for other models or architectures.
- A model whose sliding-window layers add their band to the mask input inside the graph (litert-torch's `create_sliding_mask(...) + mask`) stops in one of two places. With grouped-query attention, where a CONCATENATION repeats the band ADD's output over the groups, it stops at `mask consumer op <n> (ADD) is not CONCATENATION(mask x g, axis 2)` (NVIDIA-Nemotron-Labs-Dory1.0-4B, Spark-X2.6). `dory_work/retrofit_gpu.py` handles that case: it loads this script by path, unchanged, runs the `add_bcast` rewrite on the band ADD's output as well, and made the GPU files of both models (REPRODUCE.md). It reads and writes the prefill_decode `.tflite`, not the bundle: unpack the bundle with `litert-lm unpack`, run `retrofit_gpu.py` on the unpacked prefill_decode `.tflite`, then pack the result with `spark_work/x26_pack_check.py <src.litertlm> <unpack dir> <new .tflite> <dst.litertlm> <report.json>` (LiteRT-LM 0.19.0's CLI; it checks that every other section is byte-identical). `bash spark_work/x26_build.sh post in.litertlm out.litertlm` runs the three steps on a Spark-X2.6 file. Olmo-3-7B-Think's graph has the same band, but its KV cache is multi-head and this script stops earlier, at `prefill_1024: mask ADD 7 adds to [1, 1024, 4096], not a [bk, g, T, C] view` (every consumer of its mask input is an ADD). `retrofit_gpu.py` takes only a band ADD whose output is read only by group CONCATENATIONs; it has not been run on Olmo-3-7B-Think, whose published files keep the exporter's graph.
- On Phi-4-mini-reasoning, the file with `prefill_16` got fewer final answers right on the Mac GPU with fp16 activations: 7 vs 8 of 8 questions, and 26 vs 28 of 30 GSM8K questions. That file ships without it.
- A bundle whose ExecutorMetadata declares a `TYPE_LINEAR_ATTENTION` state buffer, such as the LFM2 short-convolution state `kv_cache_c_*`, gives broken text on the GPU with LiteRT-LM 0.18.0 and 0.18.1 after the retrofit. Every subgraph is still delegated, no validation error is printed, and the CPU output is unchanged. The cause is the `param_tensor` input that the retrofit adds: with it, the GPU executor switches to its in-place state mode and does not pass state buffers as inputs, so these layers lose their state at every step. The log then prints `GPU TensorBuffer not found for <state buffer> Tensor` twice per state buffer (with `--verbose` on the `litert-lm` CLI). This was measured on LFM2.5-350M (int4 and int8, 0.18.0) and on a Qwen3.5-0.8B int8 file (0.18.1, GatedDeltaNet state `kv_cache_lc_*` / `kv_cache_lr_*`) on the Mac GPU (WebGPU); the retrofit with `--mask select_bool` fails the same way. LiteRT-LM 615d193f is not in 0.18.1, and main reworked this path in 0a7de4d7cb (2026-10-19); no release contains it yet (checked 2026-10-24: the latest, v0.18.1, does not). Before a retrofit, read the state buffer types in the bundle's ExecutorMetadata, and check the GPU output with a free-form prompt: a one-word answer can still come out right on a broken run. `gpu_graph_retrofit.py` stops on such a bundle and names those state buffers (its report lists every state buffer with its type); `--allow-linear-attention-state` runs it anyway.

## A shorter KV cache without a re-export: `cache_variant.py`

`cache_variant.py SRC.tflite DST.tflite --cache N --report R.json` rewrites the KV cache length of an exported prefill/decode graph (KV caches as signature inputs and outputs). It reuses `prefill_bucket_clone.py`'s propagation: starting from the cache dims of `kv_cache_k_*` / `kv_cache_v_*` and the mask, and from a constant that lists the cache positions (litert-torch's sliding-window mask builder), it rewrites every tensor shape and every RESHAPE / SLICE constant that carries the cache length. A number that only happens to equal it is left alone. Every weight buffer is copied byte for byte and checked. The tool stops on STABLEHLO_COMPOSITE ops and on a fixed SLICE window of the cache larger than the new length. It also stops at the first op whose type its propagation does not handle (`op type not handled by the T propagation`), whether or not that op carries the cache length. Every signature of a Qwen3.5 hybrid has such ops: in a Qwen3.5-4B graph the 24 GatedDeltaNet layers put PAD, DEPTHWISE_CONV_2D, SELECT and CUMSUM into the prefill signatures and DEPTHWISE_CONV_2D and SELECT into prefill_1 and decode. On Qwen3.5-4B-3D-Coder it stopped at prefill_1024's op 34, a PAD from [1, 8192, 1028] to [1, 8192, 1032], so that model's 16,384-token file is a re-export ([REPRODUCE.md](../../REPRODUCE.md), "Qwen3.5-4B-3D-Coder").

Check the input cap before you shorten a KV cache: with one prefill signature of length L, LiteRT-LM 0.19.0 accepts at most floor(free KV entries / L) × L tokens and refuses more with `FAILED_PRECONDITION`. KV 3,584 with `prefill_1024` alone takes 3,072 tokens.

It made the 2,048-token cache of [`muse_glimmer_work/`](../../muse_glimmer_work/) from the 4,096-token int4 export. Checked on the real weights cut to 4 layers: an export with a 2,048-token cache rewritten to 1,024 gives the original file's logits bit for bit. For `prefill_bucket_clone.py` the Muse graph needed one more rule: a CONCATENATION that stacks copies of one tensor along the prefill length (the grouped-query mask repeat `[1,1,T,C] × g → [1,1,g·T,C]`) is now followed instead of stopping the tool.

```sh
python tools/gpu_graph/cache_variant.py model.tflite model_c2048.tflite --cache 2048 --report cache_variant.json
python tools/gpu_graph/prefill_bucket_clone.py tflite model_c2048.tflite model_p32c2048.tflite --lengths 32 --source prefill_128 --report clone.json
```

## Vision encoders on the GPU: `tfl_rank4.py`

The GPU delegate lays out a rank-2 or rank-3 runtime tensor batch-first (`ExtractTensorShape`, model_builder_helper.cc) but a rank-2 or rank-3 constant right-aligned (`ParseInputsWithConstTensorImpl`, model_builder.cc). An elementwise op that mixes the two reads the wrong rows of the constant: a 2D RoPE multiply `[N,h,d] × const[N,1,d]` uses the constant's first row for every token, and a position-table add `[N,C] + const[N,C]` uses row 0. `tfl_rank4.py rewrite IN OUT` lifts such ops to rank 4: elementwise ops with a constant, rank-3 BATCH_MATMUL, a MEAN that drops a rank, a STRIDED_SLICE on the leading axis, a SOFTMAX whose leading dimension is above 1. Every inserted RESHAPE keeps the data order, so the CPU output does not change. `tfl_census.py IN` lists the op forms the rewrite looks for.

By default some of the tensors the rewrite adds share a name (`r4_begin` / `r4_end` / `r4_strides` at every leading-axis STRIDED_SLICE, `<name>/r4` when one tensor is lifted twice). The published files were built with these names, and the default keeps them so that re-running their conversion steps writes the same bytes. ai-edge-quantizer 0.10.0 refuses a graph with repeated tensor names: for a file that goes on to it, add `--unique-names`, which gives a repeated added name `#2`, `#3`, ... as `tfl_unique_names.py` does. On Qwen2.5-VL-72B-Instruct's vision encoder the file it wrote was, byte for byte, the plain rewrite followed by `tfl_unique_names.py` (17 of 3,478 tensors renamed), and its int8 weights (below) were the published section.

It was used on the vision encoders of Muse-Glimmer-30B ([`muse_glimmer_work/`](../../muse_glimmer_work/)) and Qwen2.5-VL-72B-Instruct. Before the rewrite their GPU output was at cosine 0.25 to 0.58 against HF; after it, 1.00000 at fp32 for both models on two pictures each. It does not fix fp16: at the delegate's default precision these encoders still drift, so their bundles declare fp32 for vision.

```sh
python tools/gpu_graph/tfl_rank4.py rewrite vision_encoder.tflite vision_encoder_r4.tflite --report r4.json
python tools/gpu_graph/tfl_rank4.py rewrite vision_encoder.tflite vision_encoder_r4u.tflite --unique-names --report r4u.json  # for ai-edge-quantizer
python tools/gpu_graph/tfl_census.py vision_encoder.tflite
```

The same layout rule breaks a binary elementwise op whose runtime operand has a lower rank than its output and the same element count, when the two shapes get different placements. After a `FULLY_CONNECTED` with `keep_num_dims=False`, the codec decoder of Audio8-TTS-Preview-0.6b multiplies its `[256,1024]` result by a constant `[1,1,1024]` into `[1,256,1024]` (op 677 of `codec_decoder_fp16_T128.tflite`, `R[256, 1024] C[1, 1, 1024]` in the census; op 711 is the same with `[512,1024]`). The delegate lays out the runtime operand as BHWC(256,1,1,1024) and the output as BHWC(1,1,256,1024), so the op reads the wrong elements at every precision: at fp32 the decoder's GPU output on a Pixel 11 Pro and a Galaxy S26 had a correlation as low as 0.486 with its CPU output over 14 test code sequences. The default rule for an elementwise op with a constant (`elem` in `--rules`) leaves this op alone: its constant is per-channel (`[1,...,1,C]`), and the wrong layout is the runtime operand's. Rule `promote` reshapes such a runtime operand to the output shape before the op. It is not in the default set: `--promote-lower-rank` adds it to the default rules, `--rules promote` runs it alone. On `codec_decoder_fp16_T128.tflite` and `codec_decoder_fp16_T192.tflite` it changes ops 677 and 711 and writes the same file as `audio8_tts_work/codec_gpu_rewrite.py --promote`, and the CPU output stays the original's byte for byte (the 14 and 16 test code sequences and an all-zero control: 15 and 17 inputs). On the GPU of a Pixel 11 Pro at fp32 the rewritten T128 decoder's output had a correlation of 1.000000 with its CPU output on all 14 test code sequences, and was byte for byte the output of the decoder that also rescales its RMSNorms. The rule does not fix the fp16 overflow of those RMSNorms: at the GPU's default precision the decoder needs the rescale of [docs/fp16-safe-norms.md](../../docs/fp16-safe-norms.md) as well. `test_tfl_rank4_promote.py` checks the forms it leaves alone: a broadcast that adds elements, two constants, two runtime tensors, the same rank, a quantized operand.

```sh
python tools/gpu_graph/tfl_rank4.py rewrite codec_decoder_fp16_T128.tflite codec_decoder_fp16_T128_p.tflite --rules promote --report p.json
```

## Int8 weights for a vision encoder in a published bundle: `tfl_unique_names.py` (with `tfl_same_fn.py`), `tfl_int8_weights.py`, `litertlm_vision_swap.py`

Qwen2.5-VL-72B-Instruct's vision encoder (after `tfl_rank4.py`) was 2,553,783,680 bytes in fp32, over the 2 GiB that LiteRT-LM 0.19.0's capability check (`litert_lm_loaded_file_create` in the C API, `litert_lm.ModelInfo` in Python) accepts for a vision encoder section; with int8 weights it is 653,432,240 bytes. The weights were changed in the published bundle, in three steps:

1. `tfl_unique_names.py IN OUT --report R.json` gives every tensor a unique name and changes nothing else. ai-edge-quantizer 0.10.0 refuses a graph with repeated tensor names, and `tfl_rank4.py` repeats the names of some tensors it adds (`tfl_rank4.py rewrite --unique-names` writes the renamed file in one step). `tfl_same_fn.py IN OUT` checks the result: the same operators, tensors, buffers and graph inputs and outputs, and the same CPU output for a seeded random input (largest difference 0).
2. `tfl_int8_weights.py IN OUT` quantizes the weights of FULLY_CONNECTED and CONV_2D to int8 per output channel with ai-edge-quantizer 0.10.0 (dynamic range, symmetric); the activations stay float32. For the 72B, on LiteRT's Metal GPU delegate at fp32, the image embeddings were then at a relative L2 of 0.050 and 0.053 from the embeddings Transformers computes in fp32, on two pictures. On the CPU, XNNPACK also quantizes the activations to int8 at run time, and the same two numbers were 0.18 and 0.20.
3. `litertlm_vision_swap.py OLD.litertlm NEW.litertlm --venc ENC.tflite --vadp ADP.tflite [--clone] [--ids-from FILE]` writes the bundle with only its two vision sections replaced; every other section is copied byte for byte, and every section of both files is hashed afterwards (`SWAP_OK`). The header gets a new uuid and creation time, so two builds differ in their whole-file sha256 while every section is the same. `--ids-from FILE` writes FILE's uuid and creation time instead. FILE is the file you are rebuilding, or just its first 64 KiB (`curl -L -r 0-65535 <its resolve URL> -o head.bin`; the header is the first few KB, and the tool stops if FILE ends before the header does). Rebuilt that way on 2026-11-01 with `--clone`, from the file first published (commit `9ac47fea`) and int8 vision files made with `tfl_rank4.py --unique-names` and `tfl_int8_weights.py`, the 72B file was the published file byte for byte (sha256 `8a37fff5…`, 42,684,876,864 bytes).

The 72B's commands are the last step of its entry in [REPRODUCE.md](../../REPRODUCE.md).
