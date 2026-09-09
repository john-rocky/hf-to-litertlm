# Manifest readers — reference implementations

Two dependency-free readers for [`litertlm_manifest.json`](../manifest/SCHEMA.md), the deployment manifest shipped at the root of `.litertlm` model repos. They answer one question: **given this device, which file do I download, on which backend, with which settings?** They do not run models — wire the result into whatever runtime integration you already have (flutter_gemma, react-native-litert-lm, your own FFI).

| package | where | status |
|---|---|---|
| `litertlm-manifest` (TypeScript) | [`ts/`](ts/) | 0.2.2 source; `npm test` (24 tests, including shipped manifests) |
| `litertlm_manifest` (Dart) | [`dart/`](dart/) | 0.2.2 source; `dart test` (21 tests against the same fixtures, Dart 3.13) |

Distribution is through this repository's source. As of 2026-09-08, neither reference package name is published on npm or pub.dev; the version fields identify the source, not a registry release. See each package README for local installation and its changelog for the changes in this source version.

## SDK entry points

**First Android inference:** follow the existing [hfmodels 0.1.1 guide](https://github.com/john-rocky/hfmodels-android#add-it)
with [LFM2.5-1.2B-Instruct](https://huggingface.co/litert-community/LFM2.5-1.2B-Instruct).
It covers OS/SDK and memory/storage conditions, a pinned model revision and GPU profile,
Maven installation, complete imports, download → initialize → generate → release,
the expected **42** answer, and failure reporting. A Pixel 8a (8 GB RAM, Android 16)
[device log](https://github.com/john-rocky/hfmodels-android/blob/6410dfc48c53d13364ab0b911f3d790e6ef8585b/litertlm/results/2026-09-08-4C131JEKB15210-0.16.1-device-check.log)
records all those steps plus Stop. This path uses the SDK's `hfmodels.json` descriptor;
installing these TS/Dart reference readers is not required. A reference-reader source
version does not identify the code inside a published mobile SDK.

**Other existing SDK procedures (not device-verified by this walkthrough):**
[Flutter Gemma installation](https://pub.dev/packages/flutter_gemma#installation) and
[React Native installation and manifest resolution](https://github.com/hung-yueh/react-native-litert-lm#installation).
Follow each SDK's published-package requirements. Their unit tests, and the tests below,
check their respective code layers; a passing JS/Dart suite alone does not establish
phone inference. For a direct Kotlin runtime integration, the existing
[Android chat recipe](https://github.com/john-rocky/on-device-recipes/blob/main/android-llm-chat/INTEGRATION.md)
uses Qwen2.5-1.5B and has its own device record and conditions.

## Usage (TypeScript)

```ts
import { declaredChannels, fetchManifest, resolve } from "litertlm-manifest";

const manifest = await fetchManifest("litert-community/LFM2.5-1.2B-Instruct");
const r = resolve(manifest, { platform: "android", deviceClass: "midrange-2023+" });
if (!r) throw new Error("No variant supports the requested device/backend");
// r.url      -> the exact .litertlm to download (sha256 in r.variant.sha256)
// r.backend  -> "gpu"   (verified-fastest for that class; "cpu" on ios for this model)
// r.thinkingChannel -> the model's exact <think> markers, whitespace included
// r.sessionDefaults -> e.g. { max_output_tokens_min: 2048 } for reasoning models
// r.notes    -> platform caveats + known issues to surface to the developer

declaredChannels(manifest); // 0.1.1+: the bundle's full channel set (thinking, tool-call, ...)
                            // — empty for 0.1.0 manifests; Dart: manifest.declaredChannels
```

Dart mirrors the selection API: `LitertlmManifest.fromJson(...)` then `.resolve(platform: 'ios')`.
The app supplies HTTP; use `LitertlmManifest.fromJson(json, sourceRepo: repo, revision: rev)` to carry the fetch source into download URLs.

## The v0.1 resolution algorithm (identical in both readers)

1. An explicit `backend` request is a **filter**, not a score: only variants listing it compete, the result keeps that backend, and resolution returns `null` when no variant lists it — never a substitute backend.
2. A variant with a `recommended` entry matching the requested platform wins; matching `device_class` too ranks higher. When a backend was requested, only recommendations naming that backend count.
3. Otherwise the smallest variant (by `size_bytes`), on the requested backend (else its `default_backend`, else the first listed backend).

Ties break toward the smaller file. The resolver never returns a backend absent from the variant's verified `backends` list — that list means *verified to generate*, not merely to load — and a recommendation naming an unlisted backend is ignored.

Parsing requires non-empty string `repo` and variant `file` fields, with a descriptive `Error` in TS or `FormatException` in Dart. It also rejects empty `backends` (the schema's `minItems: 1`) and eagerly checks string-list elements (`backends`, `requirements.platform_notes`, `known_issues`). Resolution skips empty backend lists in manually constructed or subsequently mutated manifests and returns `null` when no candidate remains.

Download URLs use the fetch source repo and revision, including when a fork's copied manifest still names its origin. TS `fetchManifest(repo, rev)` sets both; Dart accepts `sourceRepo` and `revision` in `fromJson`. Without source context, parsing uses the file's required `repo`. A per-resolve `revision` overrides the fetched revision; the default is `main`. Source context does not make a malformed manifest valid.

TS `fetchManifest` continues to throw on HTTP or parse failure. Thinking markers and declared channels retain their existing reference API behavior.

The 0.2.2 fixes incorporate the parser, fork-source and empty-backend corrections by Hugh (@hung-yueh) in [react-native-litert-lm 0.7.0](https://github.com/hung-yueh/react-native-litert-lm/releases/tag/v0.7.0). Each package includes the upstream MIT notice alongside its Apache-2.0 license.

## Live manifests to test against

Every repo the converter has shipped carries one — 60 repos (92 variants) as of 2026-09-05. The two the test suites vendor as fixtures:

- https://huggingface.co/litert-community/LFM2.5-1.2B-Instruct/resolve/main/litertlm_manifest.json
- https://huggingface.co/litert-community/Qwen3-4B-Thinking-2507/resolve/main/litertlm_manifest.json

The fixtures live in `../../manifest/examples/`, so the tests break if the readers and the shipped manifests ever disagree.
