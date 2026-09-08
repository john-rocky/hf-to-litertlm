# litertlm-manifest

Reference reader for `litertlm_manifest.json` — the deployment manifest shipped at the root of `.litertlm` model repos on Hugging Face. Given a device, it answers: which file to download, on which backend, with which settings (thinking markers, session defaults, caveats). Dependency-free; it does not run models.

Version 0.2.2 is available as source in this checkout. As of 2026-09-08, `litertlm-manifest` is not published on npm. From a checkout containing this version:

```sh
cd readers/ts
npm install
npm test
npm pack
```

`npm pack` builds the JavaScript and type declarations. Install the resulting archive from your app with `npm install /path/to/litertlm-manifest-0.2.2.tgz`. The source tests use the fixtures in `manifest/examples/` at the repository root.

```ts
import { fetchManifest, resolve } from "litertlm-manifest";

const manifest = await fetchManifest("litert-community/LFM2.5-1.2B-Instruct");
const r = resolve(manifest, { platform: "android" });
if (!r) throw new Error("No variant supports the requested device/backend");
r.url;             // exact .litertlm to download (sha256 in r.variant.sha256)
r.backend;         // verified-fastest backend for the platform
r.thinkingChannel; // the model's exact <think> markers, whitespace included
r.sessionDefaults; // e.g. { max_output_tokens_min: 2048 }
```

`resolve()` returns `null` when no candidate remains, including when an explicitly requested backend is not listed or all backend lists are empty. It only chooses declared backends.

`parseManifest()` requires non-empty string `repo` and variant `file` fields and reports malformed values at parse time. `fetchManifest(repo, revision)` throws on HTTP or parse failure, then uses that fetch source for download URLs even if the manifest names another repo. The revision defaults to `main`; `resolve(manifest, { revision })` can override it. Direct parsing uses the manifest's own `repo`.

See [CHANGELOG.md](CHANGELOG.md) for changes and credit to the downstream fixes in react-native-litert-lm. The package includes its [Apache-2.0 license](LICENSE) and the upstream [MIT notice](THIRD_PARTY_NOTICES.md).

Spec and schema: https://github.com/john-rocky/hf-to-litertlm/blob/main/manifest/SCHEMA.md
