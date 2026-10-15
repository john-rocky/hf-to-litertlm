---
name: android-device-gate
description: Run a converted LiteRT graph on a shared Android phone with tools/devicebench, after a fake-phone dry run on the Mac, and keep the record of every leg - the app's report, your scorer's verdict, 2-second samples of clocks, temperatures and memory - behind an owned hold and a named waiting line. Use when a graph has to run on a phone other sessions also use, when a device number needs its record, or when a plan or lane.env is new.
---

# Android device gate

A device run is done when three things hold:

1. it went from the hold to the release: `chain.sh run` exited 0 and the phone's hold file is free again;
2. every leg has its report and its verdict: `<tag>.json` says DONE, or FAILED with the app's error, and a DONE leg has `<tag>.verdict` from your scorer;
3. the record names the phone, the runtime version and the memory left during each leg. A number without them is not a result.

`tools/devicebench/` runs one converted graph per leg through a measurement app on an Android phone that other sessions share; [tools/devicebench/README.md](../../tools/devicebench/README.md) is the reference for every plan column, `lane.env` variable and exit code. This skill is the order to use it in, on the public example in `d1_omni_work/`: the d1-omni-600M audio encoder from the Hub, the measurement app `android/d1omni_gate/` and its Mac stand-in `conversion/fake_runner.py`. Run every command from the repository root, on a Mac with `python3.12`, in one shell session: a new shell needs `LANE_HOME` exported again, and from step 3 on `ROUND`, `LANE_ENV` and `RUN_DIR`.

## Loop

**1. Lay out a lane: the graph, the rows, `lane.env` and a plan.** A lane is a directory outside the repository with `out/` for graphs and `device/` for the settings, plans, rows and APK. This makes one with its own Python for the stand-in and the scorer, and fetches the graph:

```bash
export LANE_HOME=${LANE_HOME:-$HOME/devicegate/d1-audio}
mkdir -p "$LANE_HOME/out" "$LANE_HOME/device"
python3.12 -m venv "$LANE_HOME/venv"
"$LANE_HOME/venv/bin/pip" install -r d1_omni_work/host/requirements.txt huggingface_hub
"$LANE_HOME/venv/bin/hf" download litert-community/d1-omni-600M-LiteRT d1-omni-600M_audio_T1001_fp16.tflite --local-dir "$LANE_HOME/out"
```

A generic leg runs any graph. Its rows file names the signature, each input's name, dtype, shape and file, and the outputs; an input file holds one little-endian slice per row, back to back. Save this as `$LANE_HOME/make_rows.py`. It writes the rows of the six public speech clips with the repository's own audio host:

```python
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "d1_omni_work/host")
import d1_audio_host as audio

dev = Path(os.environ["LANE_HOME"]) / "device"
recs = json.loads(Path("d1_omni_work/fixtures/public_audio.json").read_text())["records"]
names, slices, rows = ["mel", "mel_valid", "v1", "v2", "v3"], {}, []
for i, rec in enumerate(recs):
    x, _ = audio.prepare(audio.read_audio("d1_omni_work/fixtures/" + rec["media"]["file"]), bucket=1001)
    for n in names:
        slices.setdefault(n, []).append(np.ascontiguousarray(x[n], "<f4"))
    rows.append({"key": rec["id"], "index": i})
for n in names:
    (dev / f"au_{n}.f32").write_bytes(b"".join(s.tobytes() for s in slices[n]))
doc = {"kind": "generic", "signature": "audio_1001",
       "inputs": [{"name": n, "dtype": "float32", "shape": list(slices[n][0].shape), "file": f"au_{n}.f32"} for n in names],
       "outputs": [{"name": "prefix", "dtype": "float32", "shape": [1, audio.dims(1001)[2], 1024]}],
       "rows": rows, "sets": [{"name": "one_clip", "kind": "single", "rows": [rows[0]["key"]]}]}
(dev / "au_rows.json").write_text(json.dumps(doc, indent=1))
```

```bash
"$LANE_HOME/venv/bin/python" -B "$LANE_HOME/make_rows.py"
printf 'not an APK: the fake phone only records the install\n' > "$LANE_HOME/device/dry.apk"
shasum -a 256 "$LANE_HOME/device/dry.apk" > "$LANE_HOME/device/dry.apk.sha256"
```

Save the settings as `$LANE_HOME/device/lane.env`. Every value is `VAR=${VAR:-value}`, so the environment wins; README "lane.env" lists the rest.

```bash
LANE=${LANE:-d1audio}
SERIAL=${SERIAL:?set SERIAL to the adb serial of the phone}
HOLD=${HOLD:-$CA/.device_hold}
PKG=${PKG:-com.mlboydaisuke.d1omni.gate}
STAGE=${STAGE:-/data/local/tmp/d1omni_gate}
APK=${APK:-$DEVICE_DIR/d1omni_gate.apk}
APP_LOG_TAG=${APP_LOG_TAG:-D1OMNI_GATE}
PY=${PY:-$LANE_DIR/venv/bin/python}
STANDIN=${STANDIN:-$PY $BENCH/../../d1_omni_work/conversion/fake_runner.py}
DRY_PLAN=${DRY_PLAN:-$DEVICE_DIR/plan_run1.tsv}
DRY_APK=${DRY_APK:-$DEVICE_DIR/dry.apk}
DRY_LOWMEM_TAG=${DRY_LOWMEM_TAG:-au_cpu}
DRY_LOST_LEG=${DRY_LOST_LEG:-au_lost 120 gpu fp32 d1-omni-600M_audio_T1001_fp16.tflite au_rows.json generic_timing clean rest=1 warmup=2 reps=100000}
lane_score() {
  [ "$3" = generic ] || return 0
  "$PY" -c 'import sys, numpy as np; a = np.fromfile(sys.argv[1], "<f4"); print("FINITE" if a.size and np.isfinite(a).all() else "NONFINITE")' "$R/out_$1.f32" > "$R/$1.verdict"
}
```

Save the plan as `$LANE_HOME/device/plan_run1.tsv`, one leg per line, columns separated by spaces or tabs; README "The plan" gives the columns and the extras:

```
# tag  limit_s  accel  precision  graph  rows  mode  ready  extras
au_gpu  90  gpu  fp32  d1-omni-600M_audio_T1001_fp16.tflite  au_rows.json  generic  gate  rest=5
au_cpu  90  cpu  -     d1-omni-600M_audio_T1001_fp16.tflite  au_rows.json  generic  gate  rest=5 threads=4
```

FINITE says the output holds no NaN or Inf; it is not a parity check. A parity verdict compares each output with your source model's output for the same input: dump those once ([on-device-verification](https://github.com/google-ai-edge/litert-samples/blob/main/skills/on-device-verification/SKILL.md), step 1) and compare in `lane_score`. For d1-omni, `conversion/s26_rows.py` and `s26_score.py` do this, but they read reference outputs that are not published, so they do not run from a clone.

**2. Run the plan on the fake phone.**

```bash
SERIAL=fake bash tools/devicebench/dryrun.sh "$LANE_HOME/device/lane.env"
```

`fake_adb.py` plays the phone and the stand-in runs the real graph on the Mac CPU: about 2 minutes on a quiet Mac. Exit 0 means all 34 checks in `tools/devicebench/cache/dryrun/d1audio/summary.json` are ok: the GPU leg, the memory guard stopping `au_cpu`, whose app the dry run makes hang (`DRY_LOWMEM_TAG`), a phone that leaves adb 4 s into `au_lost` (exit 9, then `cleanup`), an app already running when the hold comes (exit 5), and no real adb call. The next dry run of the same `LANE` deletes that folder first. A `Terminated: 15 sleep 600` line is the fake foreign holder letting go. These two legs prove the GPU and CPU paths and the memory guard only: timing legs, `needs`, other precisions, a second graph or another bucket's rows each need a leg in your dry plan (README "The dry run").

**3. Check the inputs.** Build the app as `d1_omni_work/android/README.md` says ("Build"), copy the APK to `$LANE_HOME/device/d1omni_gate.apk` with its `shasum -a 256` line in `d1omni_gate.apk.sha256`, and export `SERIAL` (the serial `adb devices` lists). Then:

```bash
ROUND=${ROUND:-run1}
LANE_ENV=${LANE_ENV:-$LANE_HOME/device/lane.env}
RUN_DIR=$(dirname "$LANE_ENV")/$ROUND
bash tools/devicebench/chain.sh check "$ROUND" "$LANE_ENV"
```

No adb, no hold: the sha256 sums of every graph, rows file and input file go to `$RUN_DIR/inputs.sha256`, and the APK is checked against its `.sha256`; a missing file, or an APK that does not match, exits 1.

**4. Run it on the phone.** First do README "Before a run: five checks, and which phone". The chain can wait in the line for a long time: start it in the background and wait for its exit, not for a sleep.

```bash
bash tools/devicebench/chain.sh run "$ROUND" "$LANE_ENV" > "$RUN_DIR/chain.log" 2>&1; echo "chain exit $?"
```

| Exit | Meaning |
|---|---|
| 0 | Done. Legs can still be FAILED: read step 5 |
| 3 | No hold: the hold tools are missing, or the keeper gave up in the line |
| 5 | No leg: an input failed its check (no hold taken), the setup failed, or another session's transfer or app was on the phone (the hold is released) |
| 9 | Hard stop: the phone left adb or rebooted. The hold is released; clean the phone by hand (below) |

After a hard stop, this takes the phone again, removes the app and the stage dir, and releases it. It exits 4 while a chain of the same round still runs.

```bash
bash tools/devicebench/chain.sh cleanup "$ROUND" "$LANE_ENV"
```

No leg starts after 25 minutes of hold, every leg ends by 28, and the keeper releases at 30 at the latest. These are the defaults of `SKIP_MIN`, `RUN_END_MIN` and `END_MIN`; when the phone is promised to others for less time, set the three in `lane.env` to your budget so the chain, not your margin, keeps it. Count the setup into the budget: every pushed file adds its transfer and its sha256 read-back to the hold (a rows set with dozens of input files takes minutes), so push one set per hold or fold inputs into fewer files. A leg whose report says DONE is skipped by the next run of the same round, so a later hold continues the plan.

**5. Read the record.**

```bash
grep -r -H --include='*.leg_status' --include='*.verdict' --include='*.memory.txt' . "$RUN_DIR"
python3 -c 'import json, pathlib, sys; [print(p.name, d.get("status"), d.get("device"), d.get("android"), d.get("litert"), d.get("error", "")) for p in sorted(pathlib.Path(sys.argv[1]).glob("*.json")) for d in [json.load(open(p))]]' "$RUN_DIR"
```

A card or table row carries the phone (`device`, `android`), the runtime (`litert`), the leg's accel and precision, its status and verdict, the delegation line in `<tag>.logcat.txt` (`Replacing N out of N node(s) with delegate (LITERT_CL)` = the whole graph on the GPU), the memory peaks, and the phone's state at the start (`<tag>.ready.txt`, `<tag>.state_before.txt`).

## Watch for

- **A pulled file can hold the phone's error text.** `adb exec-out` has no separate stderr, so pulling a missing report returns `cat`'s error. The harness checks each pulled file's byte count and keeps a non-JSON report aside as `<tag>.report_pull.txt`; check your own pulls the same way.
- **An output that is an input passed through is not written.** The measurement app runs the graph with the CompiledModel API, which binds such an output to a buffer nothing writes: `out_<tag>.f32` then holds stale bytes for it. Compare only the outputs an op computes (a flatbuffer output whose tensor index equals an input's is a pass-through), or the verdict reads garbage.
- **Temperatures come from the current section.** `dumpsys thermalservice` prints `Cached temperatures` before the current ones, and the cached values can be old: on one Galaxy S26 the cached `SKIN` said 37.9 °C while the current one said 30.1 °C.
- **The phone decides where its cooling devices are.** A Galaxy S26's thermal service lists none, and its shell user reads 62 under `/sys/class/thermal` (the harness keeps 12); a Pixel 8a's lists nine. `s_gpu.txt` names the source of each run.
- **Memory decides the phone.** On an 8 GB Pixel 8a the GPU compile of an 896 MB graph did not fit: the memory guard stopped it 6 s in, after lowmemorykiller had killed 26 background apps. The same file runs on the 12 GB Galaxy S26. A Pixel PASS does not stand in for an S26 PASS.
- **A subset verdict does not decide a reduced-precision form.** A check set too big for one leg can run as parts (one rows file per part, one leg each); pool the parts exactly — token-weighted mean, the minimum of the minimums, counts summed — and give a verdict only when every part is complete. On a speech encoder a subset passed the NPU (fp16 arithmetic) and one phone's GPU at `FP16_WITH_FP32_ACCUM`, and the whole set, with long recordings, failed both.
- **`check` rewrites `inputs.sha256` in the round dir.** Never point it at a finished round you want to keep.
- **One hold file per phone.** Every session that uses a phone names the same `HOLD`. A second hold file for it is a second arbiter, and the hold protects nothing.
- **One process per timing run for hybrid models.** A second Conversation on one LiteRT-LM Engine starts from the first one's linear-attention state (LiteRT-LM #3165), so a leg that loops its runs inside one process measures a drifting model. Give each run its own leg.
- **Image legs with the LiteRT-LM binary** need the image first in the prompt (`[image:…]question`), `--benchmark=true` for the prefill count and `--vision_backend` for the executor: [docs/android-gpu.md](../../docs/android-gpu.md) "Image prompts and timing".

## Output layout

```
<lane>/
  out/<graph>.tflite
  device/                      lane.env, plans, rows files and their inputs, the APK and its .sha256
    <round>/                   one dir per round (ROUND_DIR)
      inputs.sha256, chain.log the check; the chain's output
      keeper.log, HOLD_RELEASED
      s_gpu.txt, s_thermal_raw.txt, s_cooling_sysfs.txt, s0_state_initial.txt, s9_state_end.txt
      <tag>.json               the app's report: status, error, device, android, litert, per-row ms
      <tag>.leg_status, <tag>.verdict, <tag>.memory.txt, <tag>.ready.txt, <tag>.logcat.txt
      <tag>.samples.txt        the 2 s samples; <tag>.state_before.txt and .state_after.txt
      out_<tag>.f32            a generic leg's outputs, rows in file order
      s_skipped.txt, CHAIN_DONE or CHAIN_STOPPED
  results/                     your scorer's files (RESULTS_DIR)
```
