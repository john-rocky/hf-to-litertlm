# Measurement activity

This is the measurement activity that produced the phone numbers of the model card and of `REPRODUCE.md`; it is not a sample app.

- `GateActivity.kt`: the activity. It loads one graph with the LiteRT Kotlin `CompiledModel` API, runs the rows or requests of a JSON file, and writes a report (and, in the gate modes, the read-out hidden states) into the app's `files/`.
- `RowCodec.kt`: right padding, the `valid` mask, the selection of the read-out rows, the little-endian float32 dump and the median.
- `kev_npu_runner.cc`: a small C++ runner on the LiteRT 2.2.0 C API that produced the Kev-0.8B NPU numbers (and the GPU numbers measured beside them on the same runner). See the last section.

Only these sources are here. The two Kotlin files were built into a debuggable Android app with LiteRT 2.2.0 from Maven (`com.google.ai.edge.litert:litert:2.2.0`), Kotlin 2.2.21, Android Gradle Plugin 8.9.1, compileSdk 35, minSdk 26 and JDK 17. The manifest declares `GateActivity` as the launcher activity and `<uses-native-library android:name="libOpenCL.so" android:required="false" />`, so the GPU accelerator can open OpenCL. The native libraries were packaged extracted (`jniLibs { useLegacyPackaging = true }`).

## Running it

Copy the graph and the input JSON into the app's `files/` (for example `adb shell run-as com.mlboydaisuke.kev.gate cp /data/local/tmp/<file> files/`), then start the activity with the extras below and wait for `files/<report>`:

```bash
adb shell am start -W -n com.mlboydaisuke.kev.gate/.GateActivity \
  --es graph kev08b_rowprefill_L512_v2_fp16fc_i8emb_r14B-vs6.tflite --es accel gpu --es precision fp16acc32 \
  --es rows rows_L512.json --es report kev_s26_r14_G_C_gpu_fp16acc32_L512_gate.json --ei L 512 --ei limit 0 \
  --es mode gate --ei threads 4
```

The graph names are the work-directory names of `conversion/README.md`; the files behind the published numbers are byte-identical to published files. The input files come from `conversion/r14_device_rows.py`, `conversion/r14_device_rows_req.py` and `conversion/r18_req2.py`, and `conversion/r12_device_compare.py` scores a gate report with its hidden-state dump. A file `files/STOP` ends a run after the current row, timing set or request; the report then says `stopped_early`.

| Extra | Type | Default | Meaning |
|---|---|---|---|
| `graph` | string | `kev08b_rowprefill_L512_v2_fp16fc_i8emb.tflite` | The `.tflite` file in `files/`. |
| `accel` | string | `gpu` | `gpu`; any other value runs the CPU (XNNPACK). |
| `precision` | string | `fp32` | GPU precision: `fp32` = `FP32`, `fp16acc32` = `FP16_WITH_FP32_ACCUM`, `fp16` = `FP16`, anything else = `DEFAULT`. |
| `rows` | string | `rows_L512.json` | The input JSON in `files/`: rows (gate), timing sets (timing) or requests (shared_gate, shared_timing). An input file with `"gpu_constant_tensor_sharing": true` turns on GPU constant tensor sharing. |
| `report` | string | `kev_gate_report.json` | The report written into `files/` at the end. The hidden-state dumps are named after it: `hsel_<report stem>.f32`, or `_host.f32` and `_direct.f32` for a pair. |
| `L` | int | 512 | The row graph's length: gate checks the rows file's `L` against it, timing runs the sets with this `L`. A pair takes Ls and Lq from its requests file (the published runs passed Ls). |
| `limit` | int | 0 | gate: stop after N rows of the file (0 = all). |
| `mode` | string | `gate` | `gate`, `timing`, `shared_gate`, `shared_timing`, or `sig_timing` (a single-signature probe; no published number comes from it). |
| `threads` | int | 4 | CPU threads when the CPU runs. |
| `warmup` | int | 5 | timing: warm-up calls; shared_timing: warm-up requests. |
| `reps` | int | 20 | Timed rounds. timing: one round is the set's rows in turn; shared_timing: one round is one host and one direct request, in alternating order. |
| `rest_ms` | int | 0 | Pause after each timed round (shared_timing: after each request). |
| `cool_ms` | int | 0 | timing and shared_timing: before each timing set or request (so also right after the compile), wait up to `cool_ms` until the GPU clock cap and temperature in `/sys/class/kgsl/kgsl-3d0` are back to the values read before the compile (temperature within 5 °C). 0 = no wait. |

A timing report holds every timed call as `[device wall-clock ms at the call's start, ms]` (`timed_calls`), and a shared_timing report holds every request as `[ms at the request's start, warmup | timed, host | direct, request ms]` (`calls`). A call's ms is write + run + read-back of the whole `hidden` output.

## Settings behind the published numbers

- Gate: `mode gate` with `rows_L<L>.json` (every question whose row fits L, and the control row) and `precision fp16acc32`; at `precision fp32` the 128-token file with `rows_L128.json` and the 2,048-token file with `rows_L2048_long11.json` (rows 1 to 11 of `rows_L2048.json`: the 9 long rows, the control row and `tv4_000`). Pair: `mode shared_gate` with `requests_Ls<Ls>_Lq64_share.json` and `requests_Ls<Ls>_Lq64_noshare.json`; both state hand-overs run on every request.
- Per-bucket timing: `mode timing` with `timing_rows_r14.json`, `warmup 5` (2 for L1024 and L2048), `reps 20` (at `fp32` 10 for L1024 and 6 for L2048), `cool_ms 20000`, at `fp16acc32` and at `fp32`.
- Request timing: `warmup 2`, `reps 6`, `rest_ms 0`, `cool_ms 20000`; rows with `timing_rows_r14_req.json` (`mode timing`) on Kev-0.8B-LiteRT's 128- and 256-token files (form C7, one file per run), pairs with `requests_Ls128_Lq64_<share|noshare>_req.json` and `requests_Ls256_Lq64_<share|noshare>_req2.json` (`mode shared_timing`).

While a run went on, the host read the phone's state with a 2-second pause between reads and appended one block per read to `<report stem>.samples.txt`: a line `=== <host time> n=<count> fg=<foreground check>` followed by the output of this command, run through `adb shell` (every 15 s the block also held the thermal status, the skin and battery temperatures and the screen state):

```sh
echo "devtime $(date +%s)"; p=$(pidof com.mlboydaisuke.kev.gate); echo "kgsl $(cat /sys/class/kgsl/kgsl-3d0/clock_mhz) $(cat /sys/class/kgsl/kgsl-3d0/max_clock_mhz) $(cat /sys/class/kgsl/kgsl-3d0/thermal_pwrlevel) $(cat /sys/class/kgsl/kgsl-3d0/temp)"; for q in /sys/devices/system/cpu/cpufreq/policy*; do echo "cpu ${q##*/} $(cat $q/scaling_cur_freq) $(cat $q/scaling_max_freq) $(cat $q/cpuinfo_max_freq)"; done; grep -E "MemAvailable|MemFree" /proc/meminfo; echo "pid $p"; [ -n "$p" ] && grep -E "VmRSS|VmHWM|VmSwap" /proc/$p/status; echo "kgsl_page_alloc $(cat /sys/class/kgsl/kgsl/page_alloc 2>/dev/null)"
```

`conversion/r18_burst.py` reads the report's per-call records with these blocks and splits each run into the calls made with neither clock capped and the calls made under a cap. The memory figures come from the same blocks: `/proc` reports kB in units of 1,024 bytes, so GB = kB × 1,024 / 10⁹; `kgsl_page_alloc` is in bytes.

The Kev-0.8B phone numbers came from builds of these sources made before their last edit, which makes the gate mode take the hidden width from the output (1,024 for Kev-0.8B, the width the earlier builds used; 2,560 for Kev-4B). The edits before it added the timing extras (`warmup`, `reps`, `rest_ms`, `cool_ms`, the per-call records); the gate and shared_gate code did not change.

## The NPU runner

`kev_npu_runner.cc` loads one or more graphs with the same accelerator and options, runs a parity pass over K fixtures per graph (one warm-up call on fixture 000, then every fixture once), and then times W warm-up and R timed calls on one fixture. Its header lists the flags. A call's ms is write the inputs + run + read back output 0 (the hidden output of these files).

It was built with the Android NDK 29.0.13113456 and the LiteRT 2.2.0 C API headers (`litert/c/*.h`); this command builds the binary that ran, byte for byte:

```bash
$NDK/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android28-clang++ -std=c++17 -O2 -fPIE -pie -Wall -Wextra \
  -Werror -static-libstdc++ -I <LiteRT 2.2.0 C API headers> kev_npu_runner.cc -L <folder with libLiteRt.so> \
  -lLiteRt -llog -ldl -Wl,-rpath,'$ORIGIN' -o kev_npu_runner
```

The libraries are not included; put them in one folder on the phone and pass it as `--libdir`: `libLiteRt.so` (LiteRT 2.2.0), `libLiteRtDispatch_Qualcomm.so` and `libLiteRtCompilerPlugin_Qualcomm.so` (the Qualcomm dispatch library and JIT compiler plugin of LiteRT's NPU runtime libraries), and the QAIRT 2.47 libraries the plugin and the HTP need (`libQnnHtp.so`, `libQnnHtpPrepare.so`, `libQnnSystem.so`, `libQnnHtpV81Stub.so`, `libQnnHtpV81Skel.so` and the others of that release). A run of the NPU numbers:

```bash
cd $STAGE && LD_LIBRARY_PATH=$STAGE ADSP_LIBRARY_PATH="$STAGE;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp" \
  ./kev_npu_runner --accel npu+cpu --burst 1 --libdir $STAGE --cache-dir $STAGE/cache_c7 --fixtures $STAGE/fx_L128 \
  --sel-dir $STAGE/fx_L128 --count 322 --inputs ids,valid --warmup 5 --rounds 20 --ab-fixture 006 --dump none \
  --out $STAGE/out --model $STAGE/c7.tflite
```

`c7.tflite` is Kev-0.8B-LiteRT's 128-token file (`kev-0.8b_rowprefill_L128_fp16fc_i8emb.tflite`); `fx_L128` holds the fixtures that `conversion/r11_npu_fixtures.py` writes (`conversion/r17_npu_fixtures.py` for the 64-token file). `--accel npu+cpu` lets the int8 embedding lookup, which the Qualcomm compiler does not take, run on the CPU. A run of a file that `--cache-dir` does not hold yet compiles the graph on the phone (the JIT) and writes the result there; a later run of the same file with the same options loads it from there. The timing runs loaded the files from the cache and used `--count 20` and fixture 006: `tv4x_emotion_00` (42 tokens) for the 64-token file, `tv4_007` (80 tokens) for the 128-token file and `tv4_006` (237 tokens) for the 256-token file; the GPU run on the same runner used `--accel gpu --gpu-precision fp16acc32 --count 1`. `conversion/r17_runner_report.py` turns the runner's stdout and `out/model0/hsel.f32` into a report that `conversion/r17_device_compare.py` scores.

While the runner ran, the host read the phone's state with a 2-second pause between reads and appended each block to `<tag>.samples.txt` (every fifth block also held the thermal status); `conversion/r17_leg_caps.py` reads them:

```sh
echo "=== $(date +%T) uptime $(cut -d' ' -f1 /proc/uptime)"; grep -E 'MemAvailable|SwapFree' /proc/meminfo; grep -E 'VmHWM|VmRSS|VmSwap' /proc/<runner pid>/status; head -3 /proc/<runner pid>/cgroup; echo "kgsl_max $(cat /sys/class/kgsl/kgsl-3d0/max_clock_mhz) pwrlevel $(cat /sys/class/kgsl/kgsl-3d0/thermal_pwrlevel)"; for q in /sys/devices/system/cpu/cpufreq/policy*; do echo "cpucap ${q##*/} $(cat $q/scaling_max_freq) $(cat $q/cpuinfo_max_freq)"; done
```
