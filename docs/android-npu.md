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

Two things you need to know:

1. **Stage ② needs an open fix.** On current `main`, calibration dies on the first prompt
   with `KeyError: 'valid_mask'` — including for the gemma-3-270m preset that ships in the
   repo. The fix is
   [litert-torch PR #1178](https://github.com/google-ai-edge/litert-torch/pull/1178)
   (open, not merged); apply it as a patch until it lands.
2. **Keep the decode mask concat under 1 MiB.** On SM8850 we observed a size threshold:
   once the decode mask concat output crosses 1 MiB, the HTP applies the mask to the wrong
   rows and quality collapses (gemma-3-270m: 12/45 on our gate at cache length 1024 or
   960, vs 29/45 at 896; fp32 reference 28/45). The failure is identical on the AOT and
   JIT paths, so it cannot be dodged by compiling differently — observed on SM8850,
   reported upstream:
   [litert-torch #1184](https://github.com/google-ai-edge/litert-torch/issues/1184).
   **For gemma-3-270m, set the KV cache length to 896**, not the preset's 1024.

**Quality gate before shipping:** an SRQ export is not automatically parity. Gate it on
device against the fp32 reference before publishing. Of the two models we have run through
this pipeline: gemma-3-270m (cache 896) passes and is published —
[mlboydaisuke/gemma-3-270m-it-NPU-LiteRT](https://huggingface.co/mlboydaisuke/gemma-3-270m-it-NPU-LiteRT)
(the card carries the measured numbers and the gate results); Qwen3-0.6B currently
produces garbage on **both** the AOT and JIT paths (a pipeline quality issue, not a JIT
one) and is withheld.

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
- **The static-range path has channelwise int4 only.** With the MLP projections, the per-layer-embedding projections and the
  final vocabulary projection (lm_head) in channelwise int4 (attention in 8 bits), GSM8K on the phone fell to 13 of 27 questions (the BF16 model
  27 of 27). Bolt's NPU file keeps 4 bits only on the MLP gate and up projections and the final vocabulary projection (lm_head), 8 bits on every
  other fully connected layer, int16 activations: 92 of 99 (the BF16 model 92 of 99), 5.04 GB, one decode graph estimated at
  3,371,794,432 bytes, a 3.28 GB context, decode 12.2 and prefill 742 tokens/s (1,024 / 256 tokens, two runs). The context keeps
  the weights' widths (2.68 GB for the 4-bit-heavy form, 3.28 GB for this one), so moving weights from 4 to 8 bits costs
  decode speed: 12.2 against 14.6 tokens/s.
- **Runtime.** The file ran on a LiteRT-LM build from the main branch (commit `3dbb23e1`, 2026-10-02, in no release up to
  v0.18.1) with the dispatch library of that build, the QAIRT 2.47 libraries, no compiler plugin on the phone, and
  `--use_hw_cache_update_for_npu=true`. Two prefill inputs that no op reads (`kv_cache_k_23`, `kv_cache_v_23`) are renamed so
  the runtime creates that layer's cache from the decode signature; the runtime then prints `Failed to get buffer
  requirements for tensor 'dangling_k_23'` once per process, which does not change the answers.

## 5. License note on the `.so` files — summary, not legal advice

The ten libraries are Qualcomm's proprietary binaries (three straight from the QAIRT SDK;
the rest repackaged in the LiteRT release zips). Our reading of the QAIRT license — a
summary, not legal advice: **bundling them inside your own application is the intended
use; do not redistribute them standalone.** Download them from the sources in §1 under
Qualcomm's own terms. This repo does not mirror them, and the SRQ bundles above contain
none of them.
