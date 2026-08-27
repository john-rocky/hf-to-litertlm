# Manifest readers — reference implementations

Two dependency-free readers for [`litertlm_manifest.json`](../manifest/SCHEMA.md), the deployment manifest shipped at the root of `.litertlm` model repos. They answer one question: **given this device, which file do I download, on which backend, with which settings?** They do not run models — wire the result into whatever runtime integration you already have (flutter_gemma, react-native-litert-lm, your own FFI).

| package | where | status |
|---|---|---|
| `litertlm-manifest` (TypeScript) | [`ts/`](ts/) | tested (`npm test`, 18 tests — fixtures include the real shipped manifests) |
| `litertlm_manifest` (Dart) | [`dart/`](dart/) | tested (`dart test`, 17 tests against the same fixtures, Dart 3.13) |

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

declaredChannels(manifest); // 0.1.1+: the bundle's full channel set (thinking, tool-call, ...)
                            // — empty for 0.1.0 manifests; Dart: manifest.declaredChannels
```

Dart mirrors the API: `LitertlmManifest.fromJson(...)` then `.resolve(platform: 'ios')`.

## The v0.1 resolution algorithm (identical in both readers)

1. An explicit `backend` request is a **filter**, not a score: only variants listing it compete, the result keeps that backend, and resolution returns `null` when no variant lists it — never a substitute backend.
2. A variant with a `recommended` entry matching the requested platform wins; matching `device_class` too ranks higher. When a backend was requested, only recommendations naming that backend count.
3. Otherwise the smallest variant (by `size_bytes`), on the requested backend (else its `default_backend`, else the first listed backend).

Ties break toward the smaller file. The resolver never returns a backend absent from the variant's verified `backends` list — that list means *verified to generate*, not merely to load — and a recommendation naming an unlisted backend is ignored.

Two more contract points: `parseManifest` rejects a variant with an empty `backends` list (the schema's `minItems: 1`), so a resolver can never invent an unverified pick; and resolution URLs follow the revision the manifest was fetched at (`fetchManifest(repo, rev)` in TS, `LitertlmManifest.fromJson(json, revision: rev)` in Dart, both overridable per resolve call) instead of hardcoding `main`.

## Live manifests to test against

Every repo the converter has shipped carries one — 20 repos (33 variants) as of 2026-08-26. The two the test suites vendor as fixtures:

- https://huggingface.co/litert-community/LFM2.5-1.2B-Instruct/resolve/main/litertlm_manifest.json
- https://huggingface.co/litert-community/Qwen3-4B-Thinking-2507/resolve/main/litertlm_manifest.json

The fixtures live in `../../manifest/examples/`, so the tests break if the readers and the shipped manifests ever disagree.
