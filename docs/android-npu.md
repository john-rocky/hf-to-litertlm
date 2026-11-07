# Running converted models on the Qualcomm NPU (Android)

This is the recipe for running models from this repo on the Qualcomm Hexagon NPU (HTP),
compiled **on the device (JIT)**. No precompiled NPU artifact is needed — you ship the
normal model file plus a set of runtime libraries in your APK, and the first load compiles
the model for whatever SoC it lands on.

For a 4B-class LLM the on-device compile did not fit a 12 GB phone (measured on one model);
§4b compiles ahead of time for one SoC.

**Verified on:** Galaxy S26 (SM8850, Hexagon **v81**), QAIRT 2.47.0, with LiteRT 2.2.0
(the Android AAR) for classic `.tflite` models and a LiteRT-LM 0.16-line runtime for LLM
bundles. Other SoCs, other Hexagon generations, and other runtime versions are
**untested** — the mechanism should carry over, but none of the numbers below do.

---

## 1. The 10 runtime libraries

JIT needs ten `.so` files in `jniLibs/arm64-v8a/`. The six dispatch libraries alone are
**not enough** — that set runs precompiled models but cannot compile on device.

| file | from | size (B) | sha256 (first 12) |
|---|---|---:|---|
| `libLiteRtCompilerPlugin_Qualcomm.so` | **`litert_npu_runtime_libraries_jit.zip`** | 724,016 | `425e5caf007f` |
| `libQnnIr.so` | QAIRT SDK, `aarch64-android` | 1,940,848 | `79043536d4ac` |
| `libQnnSaver.so` | QAIRT SDK, `aarch64-android` | 809,832 | `d26d8002da0f` |
| `libQnnHtpV81CalculatorStub.so` | QAIRT SDK, `aarch64-android` | 250,904 | `3567bc848ae1` |
| `libLiteRtDispatch_Qualcomm.so` | `litert_npu_runtime_libraries*.zip` | 445,576 | `c4abfff6c99e` |
| `libQnnHtp.so` | `litert_npu_runtime_libraries*.zip` | 3,664,480 | `c0488f2df879` |
| `libQnnHtpPrepare.so` | `litert_npu_runtime_libraries*.zip` | 86,301,344 | `9988ce10ffee` |
| `libQnnHtpV81Skel.so` | `litert_npu_runtime_libraries*.zip` | 12,322,384 | `0b4fa7419e72` |
| `libQnnHtpV81Stub.so` | `litert_npu_runtime_libraries*.zip` | 771,024 | `479da62bd52b` |
| `libQnnSystem.so` | `litert_npu_runtime_libraries*.zip` | 3,931,536 | `077a8b20a53b` |

Where to get them:

- The LiteRT zips are release assets on
  [github.com/google-ai-edge/LiteRT/releases](https://github.com/google-ai-edge/LiteRT/releases).
  The compiler plugin is **only** in the `_jit` variant of the zip — the plain
  `litert_npu_runtime_libraries.zip` does not contain it. Pick the `qualcomm_runtime_v81/`
  folder for Hexagon v81 devices.
- QAIRT (Qualcomm AI Runtime SDK) is Qualcomm's download; the three extra libraries are in
  its `lib/aarch64-android/` directory. Hashes above are from QAIRT 2.47.0.

`libQnnHtpPrepare.so` (82 MB) is the on-device compiler itself — that is the APK size cost
of JIT.

⚠ **`libQnnSaver.so` is the one everyone forgets.** Without it, the compiler plugin fails
to `dlopen`, and LiteRT rounds the error up to
`Failed to apply compiler plugins: No compiler plugin found`. The real cause appears only
as one W-level log line:
`dlopen failed: library "libQnnSaver.so" not found: needed by libLiteRtCompilerPlugin_Qualcomm.so`.

---

## 2. Classic models (`.tflite`) — CompiledModel API

Two Environment options are required:

```kotlin
mapOf(
  Environment.Option.DispatchLibraryDir       to libDir,
  Environment.Option.CompilerPluginLibraryDir to libDir,   // ← required for JIT
)
```

⚠ **If you drop `CompilerPluginLibraryDir`, the model silently falls back to XNNPACK
(CPU).** The call returns `OK` and produces plausible-looking latencies at CPU speed.
Do not trust "it ran" — check the delegate log line:

- NPU: `Replacing N out of N node(s) with delegate (DispatchDelegate)`
- CPU fallback: the same line names `XNNPACK` instead.

**Compile cache (classic path):** LiteRT caches the JIT result on disk
(`NPU JIT compilation caching enabled with cache dir: ...` in the log). Measured on one
mid-size vision model (SM8850, ~25 ms/inference on HTP): first load **11.6 s** (compile),
second load **0.17 s**. From the second launch on, JIT and a precompiled model are
indistinguishable.

**Two graphs, two accelerators.** An app that runs two graphs per window (a speech encoder, then a CTC reader over its output) can put the encoder on the NPU and the reader on the GPU. Give each graph's CompiledModel its own thread and hand window k's encoder output to the reader while the encoder takes window k + 1 (the sample app github.com/john-rocky/eg2-transcribe, `Pipeline.kt`). On a Galaxy S26 (LiteRT 2.2.0, QAIRT 2.47 libraries in the APK) a 54-minute recording in 532 windows of 10 s took 71.1 s this way, against 137.6 s with both graphs on the GPU at fp32 precision in 30 s windows. Log the time the encoder waits for the reader: it was 21.5 s in all (encoder 91.9 ms, reader 131.0 ms per window), so the GPU reader, not the NPU, set the pace. The second thread did not change the outputs: on 20 test utterances every token's label, best score and blank score was bit-identical to the one-thread order. Judge the form at task level as well: its word error rate on the 54-minute recording was 4.33 % against 3.81 % for the GPU fp32 form, and whether the NPU's fp16 arithmetic or the shorter windows cost the half point was not separated.

**Fold a small CPU graph into the NPU graph.** A pipeline that runs a small graph on the CPU between NPU calls pays for that stage on every call. In a cache-aware streaming speech encoder (nvidia/nemotron-3.5-asr-streaming-0.6b, 320 ms chunks), moving the subsampling (`pre`: an input convolution, two depthwise-separable stages and a linear layer, 34 ops) from the CPU into the conformer's NPU graph took the median chunk on a Galaxy S26 from 27.546 ms to 25.442 ms (−7.64 %; LiteRT 2.3.0, QAIRT 2.50; 5 LibriSpeech test-clean files of 18 to 35 s, each streamed 5 times, 1,865 chunks per run, the int8 decoder on the CPU; the two forms timed alternately, split, fused, fused, split, in one sitting, each number the mean of its two runs' medians): the 2.5–2.7 ms CPU stage went away and the HTP part grew by about 0.6 ms. The first chunk has another shape (25 mel rows instead of 32, no cache yet). Padding it with zero rows to the steady shape is not the same function: the first convolution's bias is positive on 120 of its 256 channels, so its ReLU turns zero rows into non-zero rows, and chunk 0's embeddings moved by up to 845.70. Multiplying the outputs of the first two ReLUs by row masks (`pre_m0` on the input convolution's 16 rows, `pre_m1` on stage 1's 8 rows: 0 on the rows that come from the padding, 1 elsewhere, all ones after the first chunk) gives the separate first-chunk graph's output bit for bit in the PyTorch float32 model (82 utterances, 2,021 chunks); the converted fused graph then gave the same tokens and frames as the separate graphs on all 82 utterances, its encoder outputs within 5.7e-4 of theirs (not bit-equal; the op was not traced), with one signature and one NPU graph. Code: [`nemotron_asr_stream_work/scripts/graphs_r4.py`](../nemotron_asr_stream_work/scripts/graphs_r4.py).

**Put a size-1 SUM between a gate's linear and the sigmoid that reads it.** On a Galaxy S26's HTP, the 512-token decision graph of MiniCPM-OCR-Decision-2B gave output probabilities up to 0.051 away from the same graph at float32 on a Mac GPU, over 29 records. The graph is a Qwen3.5 decoder with 18 gated-delta-net layers and 6 attention layers, compiled ahead of time with ai-edge-litert 2.2.0 and QAIRT 2.47. Exporting each `in_proj_z` and `q_proj` linear as a batch-1 matrix product makes the converter (litert-torch 0.10.0, litert-converter 0.5.0) put a size-1 SUM between those 24 linears (one per layer) and the sigmoids that read them (the gated-delta-net output norm's gate and the attention output gate). On the Mac GPU at float32 the graph with the SUMs gives the same bits as the graph without them (29 records). On the HTP all 8,459 ops still compiled into one partition, and the largest difference fell to 0.0065. The graph's 42 other sigmoids that read a linear's output got no SUM. The converter keeps such a SUM after some linears and drops it after others (litert-torch 0.9.4 kept it after 78 of Kev-0.8B's 138 batch-1 linears), so in the converted `.tflite`, before the compile, follow each gate LOGISTIC's input back through the SLICE or RESHAPE in front of it and check that it reaches a SUM. The same SUMs made Kev-0.8B's NPU files pass, compiled on the phone (REPRODUCE.md "Kev-0.8B and Kev-4B"). QAIRT 2.47's HTP documentation lists a more precise sigmoid kernel (`QNN_HTP_GRAPH_CONFIG_OPTION_HIGH_PRECISION_SIGMOID`, off by default), but the Qualcomm compiler plugin of ai-edge-litert 2.2.0 and 2.3.0 has no flag for it. Code: `--sum-after` in [`minicpmocr_decision_work/conversion/export_decide.py`](../minicpmocr_decision_work/conversion/export_decide.py).

---

## 3. LLM bundles (`.litertlm`) — LiteRT-LM

### What runs where

An NPU LLM bundle from this repo is a **static-range-quantized (SRQ)** `.litertlm`. It
contains only the quantized model — no precompiled QNN context — so it is hosted and
downloaded like any other bundle, and the device compiles it at load time.

What actually lands on the NPU, measured on SM8850:

- The **transformer itself runs fully on the NPU**: every op of the prefill and decode
  subgraphs is partitioned to HTP (gemma-3-270m: 888/888 + 869/869 ops; a 0.6B model:
  1424/1424 + 1376/1376).
- The small auxiliary graphs (RoPE/mask/cache-update, embedder) fall back to CPU
  (XNNPACK): the QNN validator rejects one op they contain (`ElementWiseSelect`, lowered
  from `DynamicUpdateSlice`), and a rejected graph falls back as a whole. Measured
  end-to-end cost: ≈2.8% of decode time (cache-update 2.3%, the rest <0.5%).

Quality is not a JIT trade-off: for gemma-3-270m (cache 896), the on-device JIT build
produced **string-identical generations to the offline-compiled build on 45/45 prompts**,
and scores 29/45 on our 45-prompt gate vs 28/45 for the fp32 reference.

### Startup cost — be aware

The LLM runtime currently has **no compile cache wired up**, so **every load recompiles
the model**: ≈16 s for gemma-3-270m, ≈43 s for a 0.6B model (SM8850, LiteRT-LM 0.16-line
build). This is per process start, not per prompt. Plan your UX around it.

### App wiring

Almost nothing: LiteRT-LM derives the compiler-plugin directory from the dispatch-library
directory automatically. Point the dispatch library dir at the folder holding the 10 `.so`
files, select the NPU backend, done.

### CLI (adb) — the `LD_LIBRARY_PATH` trap

The command-line runner needs one thing the app path gives you for free: the plugin's own
dependencies (`libQnnIr.so` etc.) resolve through the **linker namespace**, not through the
dispatch dir. If `LD_LIBRARY_PATH` does not include the `.so` directory you get
`dlopen failed: library "libQnnIr.so" not found` — even though the file sits right next to
the plugin — followed by a **silent XNNPACK fallback that keeps running at CPU speed**.

```sh
# libs pushed to $D/jit, model to $D; binary = LiteRT-LM built with the NPU backend
D=/data/local/tmp/npu
adb shell "cd $D && export LD_LIBRARY_PATH=\$D:\$D/jit && \
  export ADSP_LIBRARY_PATH=\"\$D/jit;/system/lib/rfsa/adsp;/vendor/lib/rfsa/adsp;/dsp\" && \
  ./litert_lm_advanced_main --backend=npu --model_path=\$D/model_srq.litertlm \
    --litert_dispatch_lib_dir=\$D/jit --max_output_tokens=64 \
    --input_prompt='What is the capital of France? Answer in one sentence.'"
```

Check the same delegate lines as in §2, plus
`1 compiler plugins were applied successfully: Qualcomm compiler plugin`. Note that
`--min_log_severity=3` hides both; capture the verdict logs at default severity.

---

## 4. Exporting an NPU (SRQ) LLM bundle yourself

The tooling is litert-torch's experimental **`npu_export`** pipeline
(`litert_torch/generative/export_hf/experimental/npu_export/`; the gemma-3-270m SM8850
preset config ships in the repo). Four stages: ① export → ② calibrate → ③ static-range
quantize → ④ QNN AOT compile. **For a JIT bundle, stop after stage ③** — the
`model_srq.litertlm` it produces is the distributable artifact.

Two things you need to know (stage ② no longer needs a patch: the `KeyError: 'valid_mask'`
that [litert-torch PR #1178](https://github.com/google-ai-edge/litert-torch/pull/1178)
worked around is fixed on `main` since `d49f1ba6`, 2026-08-24):

1. **Keep the prefill mask concat under 1 MiB, or add the mask by broadcast.** On SM8850
   with QAIRT 2.47 we observed a size threshold: once the prefill graph's mask concat
   output crosses 1 MiB, the HTP applies the mask to the wrong rows and quality collapses
   (gemma-3-270m: 12/45 on our gate at cache length 1024 or 960, vs 29/45 at 896; fp32
   reference 28/45). The failure is identical on the AOT and JIT paths — observed on
   SM8850, reported upstream:
   [litert-torch #1184](https://github.com/google-ai-edge/litert-torch/issues/1184).
   **For gemma-3-270m, set the KV cache length to 896**, not the preset's 1024. The
   MNN-Pocket-0.3B recipe keeps a 2,048-token cache instead: its patch to
   `split_cache/attention.py` adds the prefill graph's mask by broadcast over the query
   heads, with no concat (the decode graph keeps its concat, 28,686 bytes, far under the
   line). QAIRT 2.50.0.260828 runs the issue's two-op reproducer correctly; a context
   compiled with 2.47 stays wrong on the 2.50 runtime, and we have not re-run a full
   bundle on 2.50.
2. **Check the rotary-position ranges before stage ③.** Stage ③ also quantizes the small
   rotary-position graphs (`prefill_rope_*`, `decode_rope`) to int16, at the range stage ②
   recorded. A position past the largest one the calibration reached is clipped to it, so
   every later token gets that position's rotation. Read the max of the `input_pos` and
   rotary entries in stage ②'s file for the aux section
   (`Section5_TFLiteModel_tf_lite_aux.tflite.json` for these models); if it is below the
   cache's last position, raise it before stage ③. MNN-Pocket-0.3B's calibration
   ended at 511 and 527 on a 2,048-token cache, and 31 of its first 40 GSM8K replies on a
   Galaxy S26 ran to the 1,024-token cap; with the 11 rope maxima set to 2,047.9375
   (`mnn_pocket_work/npu/rope_calib_range.py`, a scale of exactly 1/16), 10 of 40 did.
   The published gemma-3-270m file is clipped at 359 in decode and 383 in the prefill.

**Quality gate before shipping:** an SRQ export is not automatically parity. Gate it on
device against the fp32 reference before publishing. Published from this pipeline:
gemma-3-270m (cache 896) —
[mlboydaisuke/gemma-3-270m-it-NPU-LiteRT](https://huggingface.co/mlboydaisuke/gemma-3-270m-it-NPU-LiteRT)
(the card carries the measured numbers and the gate results; its rotary positions are
clipped as in item 2) — and MNN-Pocket-0.3B (JIT, cache 2,048) —
[litert-community/MNN-Pocket-0.3B](https://huggingface.co/litert-community/MNN-Pocket-0.3B)
(recipe in `mnn_pocket_work/npu/`, see `REPRODUCE.md`). Qwen3-0.6B produced garbage on
**both** the AOT and JIT paths when we ran it (a pipeline quality issue, not a JIT one)
and is withheld.

---

## 4b. 4B-class models: ahead-of-time per SoC

Measured on NVIDIA-Nemotron-3.5-Bolt-E4B (the Gemma 4 E4B architecture: 42 layers, 8 query and 2 KV heads) on a Galaxy S26
(SM8850, 12 GB), 2026-10-26.

- **The on-device compile did not fit.** For a 4B-class static-range graph the Qualcomm compiler plugin takes every op, then
  the QNN graph composition runs the phone's free memory from 6.93 to 0.92 GB in 9 s (NVIDIA-Nemotron-Labs-Dory1.0-4B); int4
  weights do not shrink it, because the plugin unpacks them into int8 copies (read in the plugin's source, not measured). Compile ahead of time instead (npu_export's
  stage 4, here ai-edge-litert 2.2.0 with QAIRT 2.47.0.260601114230 for SM8850, the auxiliary graph left uncompiled). The file
  is for that SoC only, and the runtime checks the context's SDK version against the phone-side QNN libraries: with
  the 2.47 libraries it prints `Context binary SDK version matches current SDK: 2.47.0` (other library versions were not
  tested with this file).
- **Concatenate the decode mask when there are two or more KV heads.** The plugin's "MHA optimization (fast vlm decode)" pass
  matches a sequence of op types in every graph. With the mask broadcast over the query heads in decode (RESHAPE – ADD –
  RESHAPE around the mask), it divides the query count it reads on axis 1 (1 in decode) by the KV heads (2) and stops with
  SIGFPE (`apply_plugin` exit −8, the verbose log ending at `[G2G] MHA optimization (fast vlm decode)`). Broadcast in prefill,
  concatenate in decode: then decode's 2,180 ops compile in one partition.
- **One HTP graph must stay under 3,670,016 KB of estimated allocation.** With every fully connected weight in 8 bits, Bolt's
  decode graph needed 4,695,648 KB: `graph requires estimated allocation of 4695648 KB, limit is 3670016 KB`. The per-graph
  estimates appear only with `--qualcomm_log_level=verbose` (`alloca_est_total`). Splitting decode into two graphs passed the
  compile, but the 4.71 GB context of the three graphs did not load on the S26 (`Failed to create QNN context: 1002`); contexts
  of 3.28 GB and 2.68 GB loaded. Ops inserted for the split (EXPAND_DIMS, SQUEEZE) stay on the CPU: the plugin reports them
  `not supported in Qualcomm Compiler`.
- **The whole context must also fit one cDSP session.** Every graph can pass that line and the context still fail on the phone: Spark-X2.6-4B with 8-bit weights and a 2,048-token cache (graphs estimated at 3,151,294,464 and 3,349,938,176 bytes) did not load on the S26 (`Failed to create QNN context: 1002`, `QNN_COMMON_ERROR_MEM_ALLOC`). The cause is in logcat, not in the runner's stderr: `QnnDsp <E> Failed to find available PD for contextId 1 … with context size estimate 3820573952`. QAIRT 2.47's HTP documentation ("Multiple cDSP Sessions") gives 3.75 GB of usable virtual address space per session, says one context binary cannot be split across sessions, and recommends keeping one context under 3 GB. Estimate before taking the phone: the context binary's size (`Serialized a context bin of size`) + the sum over its graphs of (`alloca_est_total` − `alloca_pickle_const_extent`, both in the verbose log) + about 60–67 MiB (two measured points; for the file above 3,082,039,296 + 360,710,144 + 315,228,160 + 62,596,352 = 3,820,573,952). The int16 KV cache is outside the graph constants and counted in each graph, so a longer cache grows the estimate by about twice the KV growth. Bolt's 3.28 GB context (about 3.65 GB by the same estimate) loaded on the S26; the smallest estimate seen to fail is the 3.82 GB above.
- **The static-range path has channelwise int4 only.** With the MLP projections, the per-layer-embedding projections and the
  final vocabulary projection (lm_head) in channelwise int4 (attention in 8 bits), GSM8K on the phone fell to 13 of 27 questions (the BF16 model
  27 of 27). Bolt's NPU file keeps 4 bits only on the MLP gate and up projections and the final vocabulary projection (lm_head), 8 bits on every
  other fully connected layer, int16 activations: 92 of 99 (the BF16 model 92 of 99), 5.04 GB, one decode graph estimated at
  3,371,794,432 bytes, a 3.28 GB context, decode 12.2 and prefill 742 tokens/s (1,024 / 256 tokens, two runs). The context keeps
  the weights' widths (2.68 GB for the 4-bit-heavy form, 3.28 GB for this one), so moving weights from 4 to 8 bits costs
  decode speed: 12.2 against 14.6 tokens/s.
- **The same compile can give other bytes.** MiniCPM-OCR-Decision-2B's 384-token decision graph (2.75 GB) was compiled four times from the same input in one image (ai-edge-litert 2.2.0, QAIRT 2.47.0, for the SM8850), twice with a 26 GB and twice with a 28 GB container memory limit. Three compiles gave the same 2,802,820,992-byte file; one, at 26 GB, differed from it in 688 bytes over 28 ranges. A rebuild whose sha256 differs once does not show that the recipe changed: compile again, ship the bytes that ran on the phone, and in a reproduce entry give the sha256 with the number of compiles that gave it ([`minicpmocr_decision_work/REPRODUCE.md`](../minicpmocr_decision_work/REPRODUCE.md)). For a `.litertlm` file compare the sections, not the file's sha256 (its header holds the uuid and creation time of each pack): `python mnn_pocket_work/npu/aot_compare.py A B` prints `SAME BYTES`, `SAME COMPUTATION` or `DIFFERENT`.
- **Runtime.** The file ran on a LiteRT-LM build from the main branch (commit `3dbb23e1`, 2026-10-02, in no release up to
  v0.18.1) with the dispatch library of that build, the QAIRT 2.47 libraries, no compiler plugin on the phone, and
  `--use_hw_cache_update_for_npu=true`. Two prefill inputs that no op reads (`kv_cache_k_23`, `kv_cache_v_23`) are renamed so
  the runtime creates that layer's cache from the decode signature; the runtime then prints `Failed to get buffer
  requirements for tensor 'dangling_k_23'` once per process, which does not change the answers.

## 4c. 16-bit weights on the HTP

Measured on LiquidAI/LFM3-VL-2B and on a tiny random-weight LFM3 (hidden size 64), both compiled ahead of time for SM8850 with QAIRT 2.47; the LFM3-VL-2B file also ran on a Galaxy S26, 2026-10-29.

- **fp16 islands, not int16 weights.** The Qualcomm compiler plugin does not put a fully connected layer with int16 weights on the HTP: it writes quantized int16 tensors as `QNN_DATATYPE_SFIXED_POINT_16`, while the HTP's FullyConnected takes 16-bit weights only as `UFIXED_POINT_16` input and weights with an `SFIXED_POINT_32` bias (`Unsupported input/output datatypes requested for the HTP Op 'FullyConnected'`), so each such layer stays on the CPU. The tiny LFM3 with every fully connected weight in int16 compiled with all 69 of them on the CPU, its two graphs in 33 and 37 partitions. The plugin's `use_qint16_as_quint16` option is a no-op (`litert/c/options/litert_qualcomm_options.h`). The same layers with fp16 weights (stage ③ with float_casting on those fully connected ops only; a DEQUANTIZE / QUANTIZE pair joins each to the int16 graph) compile into one partition per graph, and the HTP computes them in FP16 (`relaxed_precision_flag value=1`).
- **Gate the islands on the phone.** The host interpreter computes the islands in fp32 where the HTP uses FP16. In the one case measured, the host was the looser check: LFM3-VL-2B with 105 fully connected layers as fp16 islands (those of layers 8 to 23 and lm_head; the rest int8 weights with int16 activations) kept the bfloat16 source model's first 32 tokens on 21 of 23 picture questions on the Mac interpreter (the bar it had to clear), and on the Galaxy S26 NPU one of 8 picture answers changed at its first token ("Oversize load" became "OVERSIZE LOAD"; the fp16 model's top-2 gap there is 0.066), so that file was not published. The cause was not isolated: the HTP's float fully connected op takes only FP16 and BF16, so the islands cannot run in fp32 on the phone.
- **Count the 16-bit weights after stage ③.** ai-edge-quantizer's min-max policy has no static 16-bit weight entry, and its recipe manager skips an entry that the policy lacks without an error, so read the weight types in the quantized file.

## 4d. A neural audio codec (convolutions and transposed convolutions) on the NPU

Measured on the Audio8-TTS codec decoder for 128 frames (`codec_decoder_fp16_T128.tflite` of [litert-community/Audio8-TTS-Realistic-0.6b](https://huggingface.co/litert-community/Audio8-TTS-Realistic-0.6b): 10 × 128 codes in, 262,144 samples at 44.1 kHz out), 2026-11-02 and 03. For the Galaxy S26 it was compiled ahead of time for SM8850 with QAIRT 2.47 (ai-edge-litert 2.2.0); the quantized forms come from ai-edge-quantizer 0.9.0. The speech checks compare with the float16 decoder's output on the Mac CPU on 14 test utterances (whisper large-v3-turbo errors, UTMOS22). No NPU form beat the phone's GPU.

| Form of the decoder | Galaxy S26 HTP |
|---|---|
| float16 (the published file) | Runs whole, 1,105 ops in one partition. The output correlates with the Mac CPU's at 0.99988 or more, but UTMOS22 is lower on all 14 utterances (mean 3.939 against 3.954), and a call takes 1,742.6 ms against 946.2 ms on the phone's GPU |
| int8 weights, float compute (w8) | Refused: `Op Dequantize does not support per-channel quant tensor` |
| int8 weights, int16 activations (a16w8) | The HTP refuses its five int16 × int8 TRANSPOSE_CONV (`QnnBackend_validateOpConfig failed 3110`): 1,130 of 1,135 ops in 6 partitions |
| a16w8 with float TRANSPOSE_CONV, a float output and min / max calibration (a16w8tfm) | Runs whole, 1,142 ops in one partition, but its output does not match the same graph on the Mac CPU (correlation −0.04 to +0.04), which passes the checks there; on the phone UTMOS22 falls to 1.27 against 3.95, and a call takes 1,859.5–1,866.3 ms, 1.91–1.94 times the GPU's 963.7–973.7 ms |

- **The Pixel 11 Pro (Tensor G6) compile stops on the decoder.** The on-device compile (Google Maven litert 2.3.0 with litert-npu-runtime-google-tensor 2.3.0) refuses the float16 decoder and the int8-weight one alike after about 11 s with `UNIMPLEMENTED: DeduceInputSlice not non-identity InternalPadding not yet implemented.` The caller gets only `FAILED_TRANSACTION` from the compiler plugin; the reason is in logcat under `vendor.google.edgetpu_vendor_service`.
- **Check for op outputs that no op reads before compiling.** The Qualcomm compiler plugin of LiteRT 2.2.0 makes every such output a QNN graph output (`Uses().empty() || is_tensor_output` in `qnn_compose_graph.cc`, the same in v2.3.0; LiteRT's main branch dropped it on 2026-10-15), and at run time only the model's own outputs get buffers, so a graph that compiles into one partition can still fail its first call with `Tensor ID <n> clientBuf is null` (status 0x1774). Seen on a sampler graph whose TOPK_V2 values no op read: with `--qualcomm_log_level=verbose` the compile log prints one `Graph Output Tensor` line per QNN graph output, 10 there against the model's 1.
- **ai-edge-quantizer's default calibration clipped the decoder's peaks.** ai-edge-quantizer 0.9.0 calibrates with a moving average of each sample's min / max (smoothing 0.95), and `Quantizer.calibrate` takes no argument to change it. That narrowed the int16 range of the last convolution: the output stopped at ±0.5947 on 8 of the 14 utterances, whose float peaks are 0.66–0.86, and the Mac CPU output still passed the speech checks (UTMOS22 3.928 against 3.954). Compare the quantized output's maximum |x| with the float one's; a `calibrator.Calibrator` built with `qsv_update_func=qsv_utils.min_max_update` kept the peaks (0.97–0.98 of the float ones).

## 5. License note on the `.so` files — summary, not legal advice

The ten libraries are Qualcomm's proprietary binaries (three straight from the QAIRT SDK;
the rest repackaged in the LiteRT release zips). Our reading of the QAIRT license — a
summary, not legal advice: **bundling them inside your own application is the intended
use; do not redistribute them standalone.** Download them from the sources in §1 under
Qualcomm's own terms. This repo does not mirror them, and the SRQ bundles above contain
none of them.
