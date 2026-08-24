# Manifest readers — reference implementations

Two dependency-free readers for [`litertlm_manifest.json`](../manifest/SCHEMA.md), the deployment manifest shipped at the root of `.litertlm` model repos. They answer one question: **given this device, which file do I download, on which backend, with which settings?** They do not run models — wire the result into whatever runtime integration you already have (flutter_gemma, react-native-litert-lm, your own FFI).

| package | where | status |
|---|---|---|
| `litertlm-manifest` (TypeScript) | [`ts/`](ts/) | tested (`npm test`, 9 tests against the real shipped manifests) |
| `litertlm_manifest` (Dart) | [`dart/`](dart/) | tested (`dart test`, 8 tests against the same fixtures, Dart 3.13) |

## Usage (TypeScript)

```ts
import { fetchManifest, resolve } from "litertlm-manifest";

const manifest = await fetchManifest("litert-community/LFM2.5-1.2B-Instruct");
const r = resolve(manifest, { platform: "android", deviceClass: "midrange-2023+" });
// r.url      -> the exact .litertlm to download (sha256 in r.variant.sha256)
// r.backend  -> "gpu"   (verified-fastest for that class; "cpu" on ios for this model)
// r.thinkingChannel -> the model's exact <think> markers, whitespace included
// r.sessionDefaults -> e.g. { max_output_tokens_min: 2048 } for reasoning models
// r.notes    -> platform caveats + known issues to surface to the developer
```

Dart mirrors the API: `LitertlmManifest.fromJson(...)` then `.resolve(platform: 'ios')`.

## The v0.1 resolution algorithm (identical in both readers)

1. A variant with a `recommended` entry matching the requested platform wins; matching `device_class` too ranks higher. The recommendation's backend is used unless the caller requested one the variant also lists.
2. Otherwise a variant listing the requested backend wins.
3. Otherwise the first variant, on its `default_backend` (else `cpu`).

Ties break toward the smaller file. The resolver never returns a backend absent from the variant's verified `backends` list — that list means *verified to generate*, not merely to load.

## Live manifests to test against

- https://huggingface.co/litert-community/LFM2.5-1.2B-Instruct/resolve/main/litertlm_manifest.json
- https://huggingface.co/litert-community/Qwen3-4B-Thinking-2507/resolve/main/litertlm_manifest.json

The test suites use these two (vendored via `../../manifest/examples/`) as fixtures, so the tests break if the readers and the shipped manifests ever disagree.
