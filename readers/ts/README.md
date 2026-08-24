# litertlm-manifest

Reference reader for `litertlm_manifest.json` — the deployment manifest shipped at the root of `.litertlm` model repos on Hugging Face. Given a device, it answers: which file to download, on which backend, with which settings (thinking markers, session defaults, caveats). Dependency-free; it does not run models.

```ts
import { fetchManifest, resolve } from "litertlm-manifest";

const manifest = await fetchManifest("litert-community/LFM2.5-1.2B-Instruct");
const r = resolve(manifest, { platform: "android" });
r.url;             // exact .litertlm to download (sha256 in r.variant.sha256)
r.backend;         // verified-fastest backend for the platform
r.thinkingChannel; // the model's exact <think> markers, whitespace included
r.sessionDefaults; // e.g. { max_output_tokens_min: 2048 }
```

Spec and schema: https://github.com/john-rocky/hf-to-litertlm/blob/main/manifest/SCHEMA.md
