# Running converted models on GPU (Android)

## LLM bundles (`.litertlm`)

The quickest path is the **Google AI Edge Gallery** app: menu → Models → **+** → *From
local model file* (or import straight from a Hugging Face repo).

⚠ **The import dialog defaults to CPU-only.** Scroll the dialog to the bottom:
*Compatible accelerators* is a multi-select row of CPU/GPU/NPU buttons with only CPU
pre-selected. Tick **GPU** at import time — for an already-imported model, delete it and
re-import with GPU ticked. This toggle is the only thing that decides which accelerators
Gallery offers for an imported model; nothing is read from the bundle. Some Gallery builds
have crashed on GPU for locally imported models regardless of the bundle — if CPU works
and GPU crashes, try the LiteRT-LM API path below or a newer Gallery build.

In your own app, use the LiteRT-LM Android API
(`com.google.ai.edge.litertlm:litertlm-android`) and select the GPU backend. The APK
manifest needs `<uses-native-library android:name="libOpenCL.so" android:required="false"/>`.

### What to expect from GPU

Token generation (decode) is memory-bandwidth-bound, so **decode speed on GPU is roughly
CPU speed — no gain**. The gain is **prefill**: it is compute-bound, so long prompts
process substantially faster. Budget RAM before choosing GPU: the GPU path needs roughly
**2× the model size** (weights plus the runtime's GPU weight cache), so a 3B int4 bundle
(~2.7 GB) wants a 12 GB device; on 8 GB devices prefer a 1–2B model or CPU.

**A CPU-vs-GPU decode comparison also measures the KV cache the bundle allocates.** On the CPU
path (XNNPACK) the attention step runs over the whole allocated cache, filled or not, so CPU decode
falls as `max_num_tokens` grows: Gemma 3 270M int8 on a Galaxy S26, same 1,024-token prompt, 139.9 →
80.2 tok/s from 1,280 to 4,096 entries; Qwen3-0.6B int4 (a 128-token benchmark prompt) 94.9 → 15.4 tok/s from
384 to 4,096. The OpenCL GPU decode stays flat over the same range, and the weight kernels take the
same time on both paths. A bundle whose cache is sized at export always pays that length; one that
leaves it resizable takes the engine's or the app's allocation. So when the GPU reads two or three
times the CPU at the bundle default, set `max_num_tokens` to the context you need, measure the CPU
again before calling it a GPU gain, and state the allocation beside every pair of numbers.

**The bundle's model type changes GPU defaults.** When the app sets nothing, LiteRT-LM picks three
GPU-executor settings by whether the bundle's `llm_model_type` is `generic_model`: typed bundles allow
quantized fully-connected / conv ops (the runtime's note: faster at the risk of quality), only generic
bundles make OpenCL wait for completion after each invoke (the note: a known quality issue on AMD and
Mali GPUs), and only generic bundles get a kernel batch hint (`runtime/engine/engine_settings.cc`, every
release since 0.14). Nothing in the bundle or the public APIs overrides them. A metadata re-pack that
changes the type leaves every graph byte-identical and still runs differently on Adreno and Mali:
measure the GPU again before quoting older GPU numbers.

### 7–9B models on a 12 GB phone

Measured with the prebuilt `litert_lm_advanced_main` of 2026-09-18 (OpenCL) on a Galaxy S26 with 11.4 GB of
RAM (MemTotal 11,389,756 kB), the phone in our hands and a device-lab unit of the same model line:

- **Dense 8B-class int4 files build and run** (granite-4.1-8b, Ministral-3-8B-Instruct-2512: 5.1 / 5.0 GB,
  six prefill signatures). The GPU-composite rewrite of granite-4.1-8b (`tools/gpu_graph/`) decoded 1.4×
  faster than the exported graph on this phone in both orders (5.35 vs 3.75 tok/s) and prefilled 15 % faster.
  Their int8 files are not run on it: an 8B-class int8 GPU engine warm-up rebooted the phone, which then
  stays off adb until a person unlocks it.
- **A 7–9B hybrid's int4 GPU engine does not build.** Qwen3.5-9B (Mixed INT4, 6.1 GB) and its finetunes
  Ornith-1.0-9B / Ornith-1.5-9B failed the GPU engine build in every session, with six or three prefill
  signatures (the OpenCL delegate took one or two signatures, then the phone ran out of memory); the
  Falcon-H1R-7B int4 file (4.69 GB, Mamba2 hybrid, fp32 activations declared) had every node of all seven
  signatures delegated, then memory ran out — the phone in our hands rebooted 69 s after the engine start,
  the device-lab unit's low-memory killer emptied the phone and its system services died. A Pixel 8a stops
  on memory during the same build. File size alone does not decide (the dense files above are as large):
  plan the Mac rows for a hybrid of this size and say on the card that the phone GPU engine does not build.
- **A 7B multi-head-attention int4 file builds only with a 2,048-token KV cache.** Olmo-3-7B-Think int4
  (32 KV heads, 4.3 GB) built its GPU engine with a 2,048-token cache and did not build with 4,096 (the
  low-memory killer ran down to foreground apps and the framework restarted); export the phone file with
  `CACHE=2048` and say on the card that it serves contexts of up to 2,048 tokens.

### Image prompts and timing with the LiteRT-LM Android binary

Measured with MiniCPM-V-4.7-1B (`fast_vlm` bundles) on a Galaxy S26 and a Pixel 8a with the prebuilt `litert_lm_advanced_main` of 2026-10-13 (REPRODUCE.md, 2026-10-13):

- Write the image first: `--input_prompt='[image:/data/local/tmp/x.png]What is in this image?'`. The binary turns the text before `[image:…]` into a text part, so a sentence-first prompt puts the question before the image and its token layout differs from the upstream processor's (image, then the question).
- Add `--benchmark=true`: only then does the binary print the prefill count (image tokens + text, the proof that the image was read), the time to first token and the vision encode time (`Mark Durations: vision_executor`). The time to first token starts at the prefill, so image → first token = the vision_executor mark + TTFT.
- A process without `--vision_backend` builds no vision executor: `[image:…]` is an InvalidArgument, and its peak memory is the decoder's alone (S26, CPU, the int8 text+image bundle: 1,984 MB without, 4,268 MB with the fp16 encoder on the CPU).
- For hybrid (GatedDeltaNet, Mamba) models time one process per run. A second Conversation on the same Engine starts from the first one's linear-attention state (LiteRT-LM #3165), so a loop of runs in one process measures a drifting model.

## Classic models (`.tflite`)

Use LiteRT's **CompiledModel** API with the GPU accelerator — no extra libraries beyond
the LiteRT runtime. Whether GPU or NPU is faster is per-model; the model cards in this
repo carry measured numbers for both where we have them, and the NPU recipe is in
[android-npu.md](android-npu.md).
