# litertlm_manifest

Reference reader for `litertlm_manifest.json` — the deployment manifest shipped at the root of `.litertlm` model repos on Hugging Face. Given a device, it answers: which file to download, on which backend, with which settings (thinking markers, session defaults, caveats). No dependencies beyond `dart:convert`; it does not run models — fetch the JSON with your own HTTP stack.

```dart
import 'package:litertlm_manifest/litertlm_manifest.dart';

final manifest = LitertlmManifest.fromJson(jsonString);
final r = manifest.resolve(platform: 'android');
r.url;              // exact .litertlm to download (sha256 in r.variant.sha256)
r.backend;          // verified-fastest backend for the platform
r.thinkingChannel;  // the model's exact <think> markers, whitespace included
r.sessionDefaults;  // e.g. {'max_output_tokens_min': 2048}
```

Spec and schema: https://github.com/john-rocky/hf-to-litertlm/blob/main/manifest/SCHEMA.md
