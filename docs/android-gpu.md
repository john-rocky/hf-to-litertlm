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

## Classic models (`.tflite`)

Use LiteRT's **CompiledModel** API with the GPU accelerator — no extra libraries beyond
the LiteRT runtime. Whether GPU or NPU is faster is per-model; the model cards in this
repo carry measured numbers for both where we have them, and the NPU recipe is in
[android-npu.md](android-npu.md).
