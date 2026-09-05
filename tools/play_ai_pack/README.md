# play_ai_pack — `litertlm_manifest.json` → Google Play AI pack

`make_ai_pack.py` turns the deployment manifest a `.litertlm` model repo ships
([`manifest/SCHEMA.md`](../../manifest/SCHEMA.md)) into what
[Play for On-device AI](https://developer.android.com/google/play/on-device-ai) (beta) needs
to deliver the right file to the right phone: an AI pack module, a device-targeting XML with one
Play device group per manifest recommendation, the Gradle snippets, and optionally a complete host
app that fetches the pack and runs the model through LiteRT-LM.

The manifest already answers "which file for which device, on which backend, with what evidence".
This tool only translates that answer into Play's vocabulary (`#group_` asset directories,
`device-targeting-config` selectors) and refuses to invent anything the manifest does not say.

```
python3 tools/play_ai_pack/make_ai_pack.py plan     litert-community/LFM2.5-1.2B-JP        # or a local manifest path, or many
python3 tools/play_ai_pack/make_ai_pack.py generate litert-community/Qwen3.5-0.8B --out out/qwen3_5_0_8b --host-app
python3 tools/play_ai_pack/make_ai_pack.py fetch    --out out/qwen3_5_0_8b                # downloads, verifies sha256 + size
python3 tools/play_ai_pack/make_ai_pack.py from-adb                                       # registry entry for an attached phone
```

No third-party Python dependencies. `fetch` uses `curl -C -` so a multi-GB pull resumes.

## What a `plan` says

```
# litert-community/LFM2.5-1.2B-JP  (manifest 0.1.0, 2026-08-25)
pack: lfm2_5_1_2b_jp  delivery: fast-follow
device groups (Play serves the first match):
  flagship           LFM2.5-1.2B-JP_int8_gpu.litertlm  [gpu]  1.24 GB
                     matches: RAM>=10.20GB  OR  device in {samsung/m1q}  OR  SoC in {QTI SM8850}
                     - RAM floor 10,200,000,000 B = 12 GB nominal of the smallest verified device (galaxy-s26-sm-s942q) x (1 - 0.15)
                     - verified device ids: samsung/m1q (galaxy-s26-sm-s942q)
                     - SoC extrapolation from verified devices: QTI SM8850
  midrange           LFM2.5-1.2B-JP_int4_gpu.litertlm  [gpu]  0.74 GB
                     matches: RAM>=6.80GB  OR  device in {google/akita,samsung/m1q}  OR  SoC in {QTI SM8850}
                     ...
default (every device): empty pack (litertlm_pack_index.json only)
  excluded: LFM2.5-1.2B-JP_int8.litertlm - no Android recommendation in the manifest
  excluded: LFM2.5-1.2B-JP_int4.litertlm - device_class 'midrange' is also claimed by LFM2.5-1.2B-JP_int4_gpu.litertlm [gpu], kept because it has a measured Android row on its backend
```

Every line of evidence is repeated as a comment in the generated `device_targeting_config.xml`
and recorded in `plan.json`.

## Mapping rules

| manifest | Play |
|---|---|
| one repo | one AI pack, named from the repo (`Qwen3.5-0.8B` → `qwen3_5_0_8b`) |
| `variants[].recommended[]` entry with `platform: android` and a `device_class` | one device group, named from the class (`midrange-2023+` → `midrange_2023plus`); variant file + `litertlm_pack.json` sidecar under `assets/model#group_<name>/` |
| class-less Android recommendation | group `baseline`, last in the XML, with a RAM floor derived from the smallest device that verified the file (policy `floor_default: auto`), so phones below it get an empty pack instead of a download they cannot run; `--floor-default none` makes it the un-suffixed default for every device |
| `measured[]` rows on the recommended backend | `included-device-id` selectors for the verified devices and, when their SoC strings are known, a `system-on-chip` selector (policy `extrapolate_soc`) |
| smallest verified device's nominal RAM, or the class's nominal RAM when the file has no measured Android row | `ram-min-bytes` = nominal × (1 − `ram_margin`); `requirements.peak_ram_mb` × `peak_ram_headroom` raises it |
| estimated download > 1.5 GB | excluded — Play caps one pack at 1.5 GB compressed; the estimate is `size_bytes` × the policy's compression ratio for the stated quantization (int8 = 0.72 measured through Play; int4 = 0.85 by local deflate-6, which reproduced Play's int8 number to three digits; others 1.0) |
| no Android recommendation | excluded (opt in with `--allow-unrecommended` for CPU files; the sidecar then says so) |
| two variants claiming one class | `--prefer-backend`, else the one with a measured Android row on its backend, else the smaller file (the reference readers' tie-break); the loser is reported |

Group order in the XML is the class priority from `device_classes.json` (flagship → midrange-2023+ →
midrange → baseline): Play serves the first group a device matches. Selectors inside a group are
OR-ed; properties inside one selector are AND-ed, so RAM, device ids and SoCs each get their own
selector. Un-suffixed assets go to **every** device, so the model files live only in `#group_`
directories and the group table (`litertlm_pack_index.json`) sits in a separate un-suffixed
`assets/index/` — bundletool rejects an un-suffixed `model/` next to `model#group_*/`.

`device_classes.json` is the only place policy lives: the pack cap, the RAM margin, the headroom
factor, SoC extrapolation, the class priorities and nominal RAM tiers, and the device registry that
maps a manifest `measured[].device` string to Play's `brand`/`device` codes and `Build.SOC_MANUFACTURER`
/ `SOC_MODEL` strings. Each device entry states how its values were obtained; `from-adb` prints a new
entry from an attached phone. A device without SoC strings contributes only its device id — the plan
says so.

## What the pack carries besides the file

`litertlm_pack.json` in each group directory freezes the manifest's answer for that group:
file name, sha256, size, backend, quantization, `min_runtime_version`, the recommendation reason,
`model.{display_name, base_model, license, context_length, capabilities, session_defaults}`
(capabilities re-derived per file from the bundle's sections, so a text-only file next to a VL build does
not claim vision), platform notes and known issues. The host app reads it, verifies the sha256, and
loads the file on the named backend — nothing is hardcoded per model.

## Host app (`--host-app`)

A minimal Gradle project: `settings.gradle`, root `build.gradle` (AGP 8.13.2, Kotlin 2.2.21),
`app/` (LiteRT-LM Android 0.16.1, Play AI Delivery 0.2.0-beta01, one Activity) and the pack
module. `PACK_README.md` inside the output is the runbook: fetch files → `gradle wrapper` →
`./gradlew :app:bundleRelease` → bundletool `--local-testing` on a USB device → Play internal test
track. Without `--host-app`, `snippets/` holds the three lines to add to an existing project.

Facts the generated project depends on, read from the Play doc dated 2026-08-14: AI packs need
AGP ≥ 8.8; device targeting needs AGP ≥ 8.10 and
`android.experimental.enableDeviceTargetingConfigApi=true`; AI packs hold models only; fast-follow
and on-demand packs land as files under `AiPackLocation.assetsPath()` (LiteRT-LM needs a file path,
so those are the delivery modes to use for `.litertlm`; an install-time pack sits inside an APK and
would have to be copied out first); bundletool ≥ 1.18.0 with `--device-groups` for local testing;
selector strings are evaluated only by Play itself, so a local install cannot validate them.

## Verified run (2026-09-02)

`generate litert-community/Qwen3.5-0.8B --host-app` → `bundleRelease` (AGP 8.13.2, Gradle 8.13; 719 MB
`.aab` carrying the 963 MB int8 file) → bundletool 1.18.3 `build-apks --local-testing` →
`install-apks --device-groups=baseline` on a Galaxy S26 (SM-S942Q, QTI SM8850, Android 16). The app
fetched the pack through the AI Delivery API (pending → downloading → transferring → completed), read
the `baseline` sidecar, verified the sha256, loaded the file on CPU in 19.7 s (no compile cache) and
answered the prompt in 0.9 s.

**Through Google Play (2026-09-05).** The same AAB, uploaded to an internal-testing release
(Play App Signing, upload key registered on first upload), installed from the Play Store on the
Galaxy S26. Play evaluated the device-targeting config itself: the phone matched the `baseline`
group (RAM floor 6.8 GB) and received that split only. The fast-follow pack downloaded as
**696 MB for the 963 MB int8 file** (Play's compressed download size, so int8 weights compress to
about 72 % here — the 1.5 GB cap has more room than `size_bytes` suggests), then the app read the
sidecar, verified the sha256, loaded the file on CPU in 12.3 s and answered in 1.1 s.

**Second device through Google Play (2026-09-05, Pixel 8a).** The same internal-testing release,
installed from the Play Store on a Pixel 8a (Google Tensor G3, 7.75 GB RAM, Android 16). Play matched
the phone to the same `baseline` group (RAM floor 6.8 GB, below the phone's 7.75 GB) and delivered the
fast-follow pack within about 40 s of the install; the app read the sidecar (`group baseline`,
`Qwen3.5-0.8B_int8.litertlm`, backend cpu), verified the sha256, loaded the file on CPU in 34.2 s (no
compile cache) and answered in 2.0 s. Two phones, one release, the group chosen by Play on each.

**Two groups, two files (local testing, 2026-09-05).** `generate litert-community/LFM2.5-1.2B-Instruct
--host-app` yields two groups from that manifest: `midrange_2023plus` (Pixel 8a device id + Tensor G3
SoC) carrying `LFM2.5-1.2B-Instruct_int4_gpu.litertlm` on gpu, and `midrange` (RAM >= 6.8 GB) carrying
`LFM2.5-1.2B-Instruct_int4.litertlm` on cpu; signed AAB 1.28 GB. bundletool `install-apks
--device-groups=midrange_2023plus` on the Pixel 8a: sidecar `group midrange_2023plus`, the int4_gpu
file, backend gpu, sha256 OK, engine ready 12.8 s (GPU compile cache warm from a first run), reply
2.7 s. The first run on that phone failed at the first message with "Can not find OpenCL library on
this device" until the host app declared `libOpenCL.so` / `libOpenCL-pixel.so` with
`<uses-native-library>` (Android 12+ hides vendor libraries from apps that do not declare them; the
template now does). The other half on the Galaxy S26 (`--device-groups=midrange`): sidecar `group midrange`,
`LFM2.5-1.2B-Instruct_int4.litertlm`, backend cpu, sha256 OK, engine ready 2.5 s, reply 1.3 s — two phones,
two groups, two files, two backends from one manifest. A second pack from `litert-community/LFM2.5-1.2B-JP`
(flagship = `LFM2.5-1.2B-JP_int8_gpu.litertlm` on gpu, midrange = `LFM2.5-1.2B-JP_int4_gpu.litertlm` on gpu;
signed AAB 1.70 GB, so it also probes whether Play's 1.5 GB limit counts the whole pack or the delivered
group) ran its flagship half on the S26: sha256 OK, engine ready 4.5 s (GPU), reply 1.2 s. Both AABs sit
under `out/play_ai_pack/*/app/build/outputs/bundle/release/` for the Play upload.

**Two groups through Google Play (2026-09-05, release 2 = the LFM2.5-1.2B-Instruct pack, versionCode 2).**
Published to the internal-testing track at 23:0x JST; the Pixel 8a's store page offered the update about
20 minutes later and quoted **746 MB** — the size of *its* group only (the 736 MB int4_gpu file plus the
base), which is Play's own device-targeting evaluation visible before the download. After the update the
app logged `group midrange_2023plus`, `LFM2.5-1.2B-Instruct_int4_gpu.litertlm`, backend gpu, sha256 OK,
engine ready 12.1 s (GPU, no compile cache), reply 1.9 s. The Galaxy S26 half of the same release (expected
`group midrange`, int4 on cpu) is recorded here once that phone is back on USB.

## Limits worth knowing

- Play states that models delivered this way are for the delivering app only; an AI pack does not
  share one model across apps.
- The doc names LiteRT and MediaPipe as usable with AI packs; it does not mention the `.litertlm`
  container. A `.litertlm` is an ordinary asset file to Play.
- Which RAM value Play compares `device_ram` against is not specified; the 15 % margin keeps a
  verified device inside its own group. The Play Console Device Catalog is the reference for
  `brand`/`device` spelling.
- A pack is only as good as its manifest: a variant without an Android recommendation, or a file
  over 1.5 GB, cannot be packed. `plan` over every shipped manifest is the quickest way to see
  where curation is still missing.
