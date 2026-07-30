# Bonsai for macOS — DiT on the Apple GPU

macOS SwiftUI demo for **Bonsai Image 4B** (ternary DiT, int4-b32): the 3-graph LiteRT pipeline with the 2.3 GiB DiT running on the **Apple GPU** through the LiteRT Metal accelerator (fp32 forced), text encoder + VAE on CPU (XNNPACK). ~6 s per 512×512 image steady-state on an Apple-Silicon Mac (0.74 s/DiT-step ×4 + text encoder ~1.3 s + VAE ~1.2 s), after a one-time ~40 s Metal compile per launch.

## Build

```bash
./prep_resources.sh   # tokenizer tables + pipeline meta + LiteRT runtime pair + C headers
xcodegen generate
xcodebuild -project BonsaiMac.xcodeproj -scheme Bonsai -configuration Release build
```

No LiteRT code is committed here: `prep_resources.sh` downloads the C API headers from the LiteRT repo at the **v2.1.6 release tag** (into `third_party/LiteRT/`, gitignored) and embeds the runtime dylib pair from the ai-edge-litert **2.1.6** wheel — headers and binaries from the same release.

Models are looked up under `~/models/bonsai-image-4b-tflite` (flat, or in `gpu_work/` + `hf_upload/` subdirs; changeable in-app):

- `dit_gpu_int4b32.tflite` — the **GPU-shaped** DiT export (rope constants folded, gather-free; `bonsai_image_work/export_dit_gpu.py`). The CPU-shaped `dit_int4b32.tflite` does not run on the Metal delegate.
- `textenc_int4.tflite`, `vae_dec_fp32.tflite`, `pipeline_meta.json` — as published.

## Headless CLI mode

```bash
BONSAI_AUTORUN=1 BONSAI_PROMPT="a bonsai tree" BONSAI_SEED=7 BONSAI_STEPS=4 \
  ./Bonsai.app/Contents/MacOS/Bonsai
```

Prints stage timings and `AUTORUN_DONE <png path>`; images land in `~/Library/Application Support/Bonsai/`.

## Runtime notes (hard-won, do not "upgrade" casually)

- **Runtime pair = ai-edge-litert 2.1.6** (`libLiteRt.dylib` + `libLiteRtMetalAccelerator.dylib` from the same wheel, embedded in the app bundle). The pair must stay same-generation. The LiteRT-main prebuilt pair (2026-07-31, `litert/prebuilt/macos_arm64`) **SIGSEGVs in `xnn_x8_transposec_ukernel__16x16_reuse_dec_zip_neon`** inside the Metal accelerator during delegate init on this DiT (weight repack, likely a >2 GiB offset overflow) — verified with the CLI in `smoke/`.
- **fp32 precision is mandatory** for the DiT (`gpu_options` TOML `precision = 2`): default fp16 overflows the activation range and corrupts output (cos ≈ −0.02 vs CPU). The prebuilt runtime exports no `Lrt*Options` helpers — options are passed as a hand-built TOML C-string via `LiteRtCreateOpaqueOptions("gpu_options", …)`.
- **Accelerator discovery**: `kLiteRtEnvOptionTagRuntimeLibraryDir` → the app's `Contents/Frameworks`; the registry auto-scans it on macOS (no `RegisterGpuAccelerator` call, unlike iOS).
- **CPU stages via CompiledModel** (`"xnnpack"` TOML `num_threads = 6`): bit-exact vs the device fixtures (text encoder cos = 1.0, VAE PNG byte-identical). The prebuilt `libLiteRt.dylib` does not export the classic `TfLiteInterpreter*` C API.
- The compiler-cache env option does not cover Metal kernels (NPU JIT only) — the ~40 s compile happens every launch; the app keeps all three graphs resident.
- **Memory**: fp32-forced GPU weights are resident on unified memory — steady state ≈ 22 GB RSS (int4 DiT dequantized to fp32 on the GPU, ~19 GB transient during compile). Comfortable on a 32 GB+ Mac; expect swap pressure below that.

`smoke/` has a standalone CLI (`build.sh` + `bonsai_smoke.mm`) that verifies each stage against `~/models/bonsai-image-4b-tflite/device_fixtures/` — run it before suspecting the app.
