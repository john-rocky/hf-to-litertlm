# litertlm_manifest

Reference reader for `litertlm_manifest.json` — the deployment manifest shipped at the root of `.litertlm` model repos on Hugging Face. Given a device, it answers: which file to download, on which backend, with which settings (thinking markers, session defaults, caveats). No dependencies beyond `dart:convert`; it does not run models — fetch the JSON with your own HTTP stack.

Version 0.2.2 is available as source in this checkout. As of 2026-09-08, `litertlm_manifest` is not published on pub.dev. In your app's `pubspec.yaml`, use a local path to a checkout containing this version:

```yaml
dependencies:
  litertlm_manifest:
    path: /path/to/hf-to-litertlm/readers/dart
```

Run `dart pub get` in the app. To run the reader's tests, use `cd readers/dart && dart pub get && dart test` from the repository root; the tests use the repository's `manifest/examples/` fixtures.

```dart
import 'package:litertlm_manifest/litertlm_manifest.dart';

// jsonString is the response body fetched from this repo and revision by your app.
const repo = 'litert-community/LFM2.5-1.2B-Instruct';
const revision = 'main';
final manifest = LitertlmManifest.fromJson(jsonString,
    sourceRepo: repo, revision: revision);
final r = manifest.resolve(platform: 'android');
if (r == null) throw StateError('No variant supports the requested device/backend');
r.url;              // exact .litertlm to download (sha256 in r.variant.sha256)
r.backend;          // verified-fastest backend for the platform
r.thinkingChannel;  // the model's exact <think> markers, whitespace included
r.sessionDefaults;  // e.g. {'max_output_tokens_min': 2048}
```

`resolve()` returns `null` when no candidate remains, including when an explicitly requested backend is not listed or backend lists have been emptied after parsing. It only chooses declared backends.

`fromJson()` requires non-empty string `repo` and variant `file` fields and reports malformed values with `FormatException`. The optional `sourceRepo` supplies the download repo when a fork's copied manifest still names its origin; without it, the manifest's own `repo` is used. The input map is not modified. `revision` defaults to `main` for URLs; `.resolve(revision: ...)` overrides the parsed source revision. Source context does not bypass validation.

See [CHANGELOG.md](CHANGELOG.md) for changes and credit to the downstream fixes in react-native-litert-lm. The package includes its [Apache-2.0 license](LICENSE) and the upstream [MIT notice](THIRD_PARTY_NOTICES.md).

Spec and schema: https://github.com/john-rocky/hf-to-litertlm/blob/main/manifest/SCHEMA.md
