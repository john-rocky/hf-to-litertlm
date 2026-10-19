# devicebench

One shell harness that runs a converted LiteRT graph on a shared Android phone and records everything a parity or timing claim needs. It takes an owned hold on the phone, installs a measurement app, pushes the graphs and input rows, runs each leg of a plan file with a 2-second sampler of clocks, temperatures and memory, scores every finished leg on the host, cleans the phone and releases the hold.
A model lane keeps only a settings file (`lane.env`), its plan files, rows, graphs, the app's APK (or a native runner), a host scorer and a Mac stand-in. The chains, the leg runner and the hold keeper live here once and are not copied into lanes.
A fake phone (`fake_adb.py`) runs the same chains end to end on the Mac (`dryrun.sh`, `dryrun_npu.sh`) before any phone time is spent.

## Files

| File | What it does |
|---|---|
| `chain.sh` | `check`, `mac-baseline`, `run`, `cleanup` of one round of one lane, with the measurement app |
| `npu_chain.sh` | `check`, `run`, `cleanup` with a native C-API runner instead of the app (NPU legs, and GPU legs of the same runner) |
| `lib.sh` | Shared functions: loading `lane.env`, phone reads, the plan, the hold and its clock, the ready gate, the transfer check |
| `run.sh` | One leg of the measurement app (`am start`, poll, stop at the limit, pull, logcat), plus `state` and `cleanup` |
| `keeper.sh` | Waits in the phone's named line, owns the hold, releases it on request or at its limit (`HOLD_LIMIT`; the chains set it from `END_MIN`, 30 minutes by default) |
| `fake_adb.py` | A stand-in for `adb` with a fake phone in a directory; the app's process is the lane's stand-in |
| `dryrun.sh` | Three scenarios of `chain.sh` against the fake phone, with a `summary.json` of checks |
| `test_xfer.sh` | `chain.sh`'s push (`xfer`: exec-in, then the byte count polled) against a fake phone and a fake clock |
| `dryrun_npu.sh` | Three scenes of `npu_chain.sh` against the fake phone and the lane's stand-in runner |
| `examples/pixel8a.lane.env` | Settings for a second phone (Pixel 8a) |
| `hold/` (when present) | The hold tools `CA` defaults to: `device_hold.py` (the owned hold file), `hold_cli.py` (acquire / release / read from bash), `queue_cli.py` (the phone's named waiting line) and their tests |
| `pool_parts.py` | One verdict for a check set that ran as parts, pooled exactly from the parts' scorer files ("The plan"); `test_pool_parts.py` its cases |

Host: macOS (BSD `date` and `stat`), bash 3.2 or later, the system `python3` for small JSON reads and the hold tools, the lane's Python for its scorer and stand-in. `CA` names the hold tools' directory (default: `hold/` in this folder when it has `queue_cli.py`, else this folder's parent). The keeper needs `queue_cli.py wait <hold file> <name> <pid>` (wait in the phone's named line, then take the hold for that pid) and `list`, and `hold_cli.py release <hold file> <pid>` (remove the hold only when that pid owns it); the dry run also uses `hold_cli.py acquire` and `queue_cli.py enqueue`.

## Commands

```bash
bash chain.sh check <round> <lane.env>         # the plan and its inputs with sha256; no adb, no hold
bash chain.sh mac-baseline <round> <lane.env>  # every gate leg's graph and rows through the stand-in on the Mac CPU
bash chain.sh run <round> <lane.env>           # the phone: hold, setup, legs, cleanup, release
bash chain.sh cleanup <round> <lane.env>       # by hand after a hard stop
bash dryrun.sh <lane.env>                      # the fake phone: scenarios 1-3, cache/dryrun/<LANE>/summary.json
bash npu_chain.sh check | run | cleanup <round> <lane.env>
bash dryrun_npu.sh <lane.env>                  # the fake phone: scenes normal, compile_hang, usb_lost
```

`mac-baseline` and `dryrun.sh` run real graphs on the Mac CPU: run them in the Mac's measurement window (nothing else heavy at the same time). A run writes into the round dir `<lane>/device/<round>/` (`ROUND_DIR` overrides). Do not point `check` at the round dir of a finished run you want to keep: it rewrites `inputs.sha256` there.

## The plan

The plan of round `<round>` is `PLAN` if set, else the first of `<device dir>/<round>/plan_<round>.tsv`, `<device dir>/<round>/plan.tsv`, `<device dir>/plan_<round>.tsv`. One leg per line, whitespace-separated, `#` starts a comment:

```
tag  limit_s  accel  precision  graph  rows  mode  ready  [extras]
```

| Column | Values |
|---|---|
| `tag` | The leg's name; the app's report is `<round dir>/<tag>.json` |
| `limit_s` | The app's time limit; cut so that the leg ends by `RUN_END_MIN` |
| `accel` | `gpu` or `cpu` |
| `precision` | `fp32`, `fp16acc32`, `fp16` or `default` for `gpu`; `-` for `cpu` |
| `graph` | A file in `GRAPH_DIR`; the app gets `sig=<SIG_OF applied to the name>` when `SIG_OF` is set |
| `rows` | A file in `ROWS_DIR`; the side files it names go along (`rows[].prefix_file`, `sets[].rows[].prefix_file`, a generic rows file's `inputs[].file`) |
| `mode` | `gate` (parity rows), `timing` (timed sets), `generic` / `generic_timing` (named input files, any graph) |
| `ready` | `clean` (thermal status 0, no CPU policy capped, the GPU's thermal level 0: see "GPU reads"): wait for a clean phone up to `REST_MAX`, then run warm. `gate`: do not wait, but never at thermal status 4 or above. `timing`: run only from clean, else leave the leg for the next hold. `uncapped`: run when no CPU policy is capped, the GPU's thermal level is 0 and the thermal status is at most 1, else as `timing` |

Extras (`key=value`): `rest=<s>` (rest since the previous leg, default 60), `keep=<graph,...>` (graphs that stay on the phone), `guard=<kB>` (MemAvailable below it force-stops the app; default `MEM_GUARD_KB`; 0 = off), `sample=<s>` (sampler period, default 2), `poll=<s>` (ready gate poll), `needs=<tag>` (run only when that leg is DONE), `needs_pass=<tag>` (run only when that gate leg's verdict is PASS; waits up to `VERDICT_WAIT_S` for the scorer), `once=1` (one attempt per round dir), `whole=1` (start only when the whole limit fits before `RUN_END_MIN`), `prep=<kind>:<leg>` (a generic leg whose rows and inputs `lane_prep` makes from that earlier generic leg's pulled output), `threads=<n>` (default 4), and the app's own extras listed in `APP_INT_EXTRAS` / `APP_STR_EXTRAS`. An extra reaches the app only when it has a value.

A leg whose report already says DONE is skipped, so the next hold continues the plan where the last one stopped.

A check set that does not fit one leg (the app's limit, the rows file's size, the hold clock) runs as parts: one rows file per part (`<set>a.json`, `<set>b.json`, …) and one leg each. A part's verdict is a verdict on a subset; `pool_parts.py` pools the parts' scorer files into one verdict for the whole set, with the rule given as arguments: `--wmean F:W` (the mean weighted by W, for example the tokens), `--min` / `--max` / `--sum`, `--concat` (lists joined and sorted), `--complete` (a condition every part must meet: scored, every row present) and `--bar` (the conditions on the pooled values); `--rules <file>` keeps a lane's rule in one file. List every part of the set, present or not: an absent part, a part that fails a `--complete` condition or lacks a value a rule reads makes the whole INCOMPLETE, never PASS.

```bash
python3 pool_parts.py --out results/whole_w10.json --id tag \
  --sum tokens --wmean cos_mean:tokens --min cos_min --max max_abs \
  --complete 'verdict in (PASS, FAIL)' --complete 'rows_present == @rows' \
  --bar 'cos_mean >= 0.999' --bar 'cos_min >= 0.99' \
  results/w10a.json results/w10b.json results/w10c.json
```

## lane.env

A bash file that the chain reads with `ROUND`, `LANE_DIR` (the parent of the file's directory), `DEVICE_DIR` (the file's directory), `BENCH` (this folder) and `CA` already set. Write every value as `VAR=${VAR:-value}` so that the caller's environment wins.

| Variable | Default | Meaning |
|---|---|---|
| `LANE` | required | Short lane name; the waiting-line name is `<LANE>-<round>-<DEVICE>` unless `SCRIPT_NAME` is set |
| `SERIAL` | required | The phone's adb serial |
| `HOLD` | required | The phone's hold file (one per phone) |
| `PKG` | required by `chain.sh` | The measurement app's package |
| `STAGE` | required by `chain.sh` | The phone's staging dir (`/data/local/tmp/<name>`); the cleanup removes it |
| `DEVICE` | `phone` | Short phone name in the waiting-line name |
| `GRAPH_DIR` / `ROWS_DIR` / `RESULTS_DIR` | `<lane>/out`, `<lane>/device`, `<lane>/results` | Where the plan's graphs and rows are, where the scorer writes |
| `APK` / `APK_SHA` | `<device dir>/app-debug.<round>.apk`, its `.sha256` | The app, checked before the install |
| `ACTIVITY` | `.GateActivity` | The app's activity |
| `SIG_OF` | empty | A `sed -E` script mapping a graph file name to the app's `sig` extra |
| `APP_INT_EXTRAS` / `APP_STR_EXTRAS` | `limit_rows:limit warmup rest_ms reps cool_ms cpu_cache gpu_src_quant` / empty | Plan extras passed as `--ei` / `--es` (`plan_key:app_key`) |
| `APP_OUTPUTS` | `sel full:keep_scores` | The app's output files `<kind>_<tag>.f32` (size in the report's `<kind>_bytes`); `<kind>:<extra>` is pulled only when that extra is 1. The first one decides whether a gate leg is scored |
| `APP_LOG_TAG` / `APP_LOG_RE` | empty / the package | The app's log tag; the lines kept from all logcat buffers |
| `PEER_APP_RE` | the package's first two labels | Other lanes' apps: one in front, or one whose CPU time grows, stops the run before any leg. Empty = off |
| `PY` / `ADB` | `python3` / `adb` | The scorer's and stand-in's Python; the adb client |
| `STANDIN` | empty | The command line of the app's stand-in on the Mac (`mac-baseline`, the dry run). It takes the app's extras as `--<key> <value>` plus `--files-dir`, and `--fake-root` / `--lowmem` under the fake phone |
| `MEM_GUARD_KB`, `SKIP_MIN`, `RUN_END_MIN`, `END_MIN`, `REST_MAX`, `REST_MAX_T`, `POLL`, `DATA_MARGIN`, `XFER_MAX_S`, `VERDICT_WAIT_S`, `MAC_THREADS` | 1500000, 25, 28, 30, 300, 600, 15, 2000000000, 900, 120, 8 | The memory guard, the clock, the ready gate, transfers |
| `DRY_PLAN`, `DRY_APK`, `DRY_LOWMEM_TAG`, `DRY_LOST_LEG`, `DRY_REFERENCE` | empty | The dry run's plan, APK, stuck-app leg, lost-phone leg and, optionally, an earlier `summary.json` to reproduce |
| `NPU_RUNNER_NAME`, `NPU_STAGE` | required by `npu_chain.sh` | The runner's process name on the phone (at most 15 characters: `pidof` matches the kernel's short name) and its staging dir |
| `NPU_RUNNER`, `NPU_RUNNER_SHA`, `NPU_LIBS`, `NPU_LIBS_SUMS`, `NPU_ADSP_PATH`, `FX_ROOT`, `CACHE_EST` | `<device dir>/<round>/stage/<name>`, empty (not checked), `<device dir>/<round>/stage/libs`, `<libs>/SHA256SUMS`, the stage then the vendor DSP dirs, `<device dir>/<round>`, 1400000000 | The runner file and its recorded sha256, the runtime libraries and their sums, the DSP library search path, where the fixture manifests are, the room kept per JIT cache |
| `NPU_DRY_ROUND`, `NPU_DRY_PLAN`, `NPU_STANDIN`, `NPU_DRY_LOST_LINE`, `NPU_DRY_REFERENCE`, `NPU_DRY_REFERENCE_NORMAL` | empty | The NPU dry run: the round whose fixtures it uses, its plan, the stand-in runner, the stdout line after which the phone leaves adb, earlier summaries to reproduce |

Hooks (bash functions in `lane.env`, each optional, run from `LANE_DIR`):

- `lane_score <tag> <rows> <mode> <graph>`: the host scorer of a DONE leg, in the background (`mode` is `gate`, `timing` or `generic`). A gate leg's verdict is the first word of `<round dir>/<tag>.verdict` (`PASS`, `FAIL`, ...), which `needs_pass` reads. Without the hook nothing is scored.
- `lane_score_mac <base> <graph> <rows> <mode> <rows dir>`: the scorer of a `mac-baseline` run (report `<round dir>/mac/<base>.json`).
- `lane_prep <kind> <src report> <src output> <src rows> <name> <out dir> <sha file>`: makes a `prep=` leg's rows file `<out dir>/<name>` and its inputs (and their sha256 lines) from an earlier leg's output.
- `lane_graph_file <graph>` (`npu_chain.sh`): the plan's graph name to its file name in `GRAPH_DIR`.
- `lane_npu_score <tag>` (`npu_chain.sh`): the host scorer of an NPU chain leg (`<round dir>/<tag>.leg.json`, `<tag>.stdout.txt`, `<tag>.dump/`), in the background.

The app's contract: `am start -W -n <PKG>/<ACTIVITY>` with `--es graph accel precision rows report mode`, `--ei threads` and the extras above; it reads `files/<graph>` and `files/<rows>`, writes `files/<tag>.json` (`status` DONE or FAILED, `<kind>_bytes` for each output), stops after the current row when `files/STOP` appears, and stays alive until force-stopped. Generic legs write `files/out_<tag>.f32` (`out_bytes`).

## The order of a run

1. `check` of the plan and every input (sha256 of the APK, graphs, rows and side files into `inputs.sha256`); a failure takes no hold.
2. The keeper (started from a copy in the round dir) waits in the phone's named line (`queue_cli.py wait`), takes the owned hold and writes `HOLD_ACQUIRED`. The clock starts.
3. `stay_on_while_plugged_in` is read, never changed. The GPU's driver is read once (`kgsl`, `mali` or `none`, into `s_gpu.txt` with, on Mali, which of its files the shell user can read): it decides the sampler's GPU line and the ready gate's GPU level. Where the cooling devices are read is decided once too (the thermal service's list, else `/sys/class/thermal`; a `cooling:` line in `s_gpu.txt`). The phone's `dumpsys thermalservice` is kept once as it is (`s_thermal_raw.txt`: its cached and current temperatures and its cooling devices with their types), and every `/sys/class/thermal/cooling_device*` with its type and states (`s_cooling_sysfs.txt`).
4. Transfer check: another session's adb transfer to this phone (its serial, or no serial) on the Mac, or this app, a `*runner` process or a busy app of another lane on the phone, means no leg: release. A transfer to another phone is recorded only.
5. `/proc/uptime`, the APK (sha256 checked) installed, the rows and side files sent into the app's `files/` with their sha256 read back on the phone.
6. Per leg: the graph sent by `exec-in` with its byte count polled and its sha256 checked on the phone (other graphs and their XNNPACK cache files removed unless kept, `/data` free checked) → the ready gate → the leg (`run.sh`) with the 2-second sampler (device clock, GPU clock and level, CPU caps, MemAvailable, the app's VmHWM / VmRSS / VmSwap, the GPU's memory; every ~15 s the thermal status, the skin and GPU temperatures from the current section, the cooling devices, the battery temperature and the screen state), the foreground check and the memory guard → lmkd lines naming this package, the memory guard, a reboot (`/proc/uptime` went down), other processes' CPU time → `lane_score` in the background.
7. Cleanup: force-stop, uninstall, the stage dir removed and read back; `RELEASE_HOLD`; the keeper releases the hold.

Clock, in minutes after the hold: no leg starts after `SKIP_MIN` (25), every leg ends by `RUN_END_MIN` (28), the keeper releases at `END_MIN` (30) at the latest. Exit 9 = the phone left adb or rebooted (hard stop; the hold is released, then `cleanup` by hand). Exit 5 = inputs, setup, or another job on the phone (no leg). Exit 3 = no hold.

## The dry run

```bash
bash dryrun.sh <lane.env>                 # an Adreno (kgsl) fake phone: cache/dryrun/<LANE>/
DRY_GPU=mali bash dryrun.sh <lane.env>    # an Arm Mali fake phone: cache/dryrun/<LANE>-mali/
```

It puts an `adb` tripwire first on `PATH`, sets `ADB=fake_adb.py`, a fake hold file and all output under `cache/dryrun/<LANE>/`, and runs three scenarios with the lane's real graphs on the Mac CPU:

1. Another holder keeps the fake phone for 15 seconds and an entry without a keeper heads the line; `check`, `mac-baseline`, then the plan `DRY_PLAN` (it should cover every path: each accel and precision, timing, `needs`, `needs_pass` passed and not passed, a stop at the limit, the memory guard on a stuck app, a rows file of another bucket, an extra output file, a graph swap).
2. The phone leaves adb 4 seconds into `DRY_LOST_LEG`: hard stop and release; the phone comes back; `cleanup` takes a hold, cleans, releases.
3. The app is already running when the hold comes: no leg, release; then `cleanup`.

`summary.json` lists every check with `ok`, `got` and `want`: each step's exit code, no real adb call, every fake adb call inside a hold, the chain's transfer check, ready gate, leg, pull, cleanup and release lines, every graph swap and small file checked on the phone, every plan leg run or skipped with a reason, the GPU reads of the fake phone's kind (a GPU line in every leg's samples and no failed GPU read, a `gpu:` line in every state file, the GPU level in every ready gate line; on Mali also the GPU's temperature and cooling device in the slow samples), every leg's report JSON (the fake `exec-out`, like the real one, returns `cat`'s error text for a missing report: it is kept aside as `<tag>.report_pull.txt`), the thermal service's raw dump and the sysfs cooling devices kept after the hold, the cooling source named in `s_gpu.txt` (the fake Adreno phone has a Galaxy S26's form: its thermal service lists no cooling device and its sysfs lists twelve, so `sysfs`; the fake Mali phone a Pixel 8a's: its thermal service lists nine, so `hal`, and a sysfs device of a kept type is left alone), the cooling devices from that source (each state file's `cooling:` line equals the HAL's `mType` 2 and 3 devices or the kept sysfs devices with their states; every leg's slow samples name a GPU device and nothing else), every temperature from the current section (both fakes' cached sections hold older values: each state file's `skin:` and `hal_temps:`, every slow sample's skin and, on Mali, `G3D`), the fake phone clean and the hold free afterwards. With `DRY_REFERENCE` it also compares the leg statuses, skips, verdicts, line counts, the scorer's results, timing sets and the stand-in's layout check with an earlier run (a kgsl phone's record: a Mali dry run skips the comparison, because the app's own cool wait reads kgsl and sleeps a fixed time on Mali). The script exits 0 only when every check is ok.

The fake phone writes a whole file before `exec-in` returns, so the dry run never reaches the byte-count polls of `chain.sh`'s push (`xfer`). `bash test_xfer.sh` runs that function against a fake phone and a fake clock: a file complete at once, one complete 0.7 s after `exec-in` returned (one second of polls), one stuck (stopped after 30 s without growth and removed), one slow but growing, one never complete (`XFER_MAX_S`), and a phone lost in the transfer.

`bash dryrun_npu.sh <lane.env>` does the same for `npu_chain.sh` with the lane's stand-in runner (staged under the runner's name; it dumps known outputs instead of running the graph), three scenes, each on its own fake phone and hold file: `normal` (the plan `NPU_DRY_PLAN` as it is: every leg DONE with its dumps and a score; the runner's stdout and stderr in two files, LiteRT's log lines only in `stderr.txt`, the runner's `summary` line at a line start of `stdout.txt` and in the scorer's results, `delegate.txt` opening with the runner's own delegation lines, the raw thermal dump and the sysfs cooling devices kept, every temperature from the current section and the cooling devices from sysfs), `compile_hang` (every NPU compile hangs: `COMPILE_TIMEOUT` by the compile watch, the killed runner's `stderr.txt` still pulled, the GPU legs still run, a leg that needs a timed-out leg is skipped) and `usb_lost` (the phone leaves adb during the first leg: hard stop, release, then `cleanup`). Every fake adb call must fall inside a hold and every stage must be gone at the end; `NPU_DRY_REFERENCE` / `NPU_DRY_REFERENCE_NORMAL` compare the scenes with an earlier run.

## Before a run: five checks, and which phone

Check these before starting a run (or an agent session that will start one), and do not start when one fails:

1. Disk: free space covers the run (graphs, caches, the pulled outputs).
2. The terminal workspace manager that hosts the sessions responds.
3. The account's usage left covers the length of the run.
4. The phone: `adb devices`, the waiting line (`python3 queue_cli.py list <hold file>`) and the scheduled jobs that take a free phone (for example a nightly benchmark at 02:00 and 05:30).
5. The clock: no leg starts after `SKIP_MIN` (25) minutes of hold, every leg ends by `RUN_END_MIN` (28) and the keeper releases at `END_MIN` (30). When the phone is promised to others for less time, set the three in `lane.env` to that budget. Count the setup in it: every pushed file adds its transfer and its sha256 read-back to the hold (a rows set with dozens of input files takes minutes), so push one set per hold or fold inputs into fewer files.

Which phone:

- GPU and CPU legs (gates, form comparisons, the host loop, checks of the measurement app itself) go to a free second phone such as the Pixel 8a (`examples/pixel8a.lane.env`, its own hold file and waiting line).
- NPU legs and card numbers (the final gate and timing of a shipped file) go to the Galaxy S26.
- A Pixel PASS does not stand in for an S26 PASS: GPU op support and fp16 behaviour differ between Mali and Adreno. Use the Pixel to narrow the candidates and to catch failures early.
- The GPU is read through each driver's own paths (next section). On the Pixel, gate legs are the tested use; a timing leg needs a definition of a clean Mali GPU checked on the phone first (the clock limit itself is not readable there).
- Where it has run: `chain.sh` on a Pixel 8a (Mali) and on a Galaxy S26 (Adreno), `npu_chain.sh` on the Galaxy S26 (an HTP JIT leg and the same runner's GPU legs), each end to end from the hold to the release.
- Memory: a stop by the memory guard is the harness's rule, not the phone's limit. On the Pixel 8a (8 GB), with `guard=400000`, a 1.26 GB graph compiled whole on the GPU at `FP16_WITH_FP32_ACCUM` and passed its gate (2026-10-19). During that compile MemAvailable fell to 522,628 kB, below the default guard of 1,500,000 kB, and lowmemorykiller killed 25 cached background processes but not the app. Run a large graph's GPU legs on the Pixel at `guard=400000` before moving them to the S26.

## GPU reads: Adreno (kgsl) and Mali

After the hold, the chain reads which GPU driver the phone has: `kgsl` when `/sys/class/kgsl/kgsl-3d0` exists (Adreno), `mali` when `/sys/class/misc/mali0/device` exists (Arm Mali), else `none` (no GPU reads; the ready gate then never sees a clean phone: gate legs run, clean legs run warm after `REST_MAX`, timing and uncapped legs are skipped). `GPU_KIND` in the environment skips the read. The result is in `<round dir>/s_gpu.txt`.

| | kgsl (Adreno; for example the Galaxy S26) | mali (for example the Pixel 8a: Tensor G3, Mali-G715) |
|---|---|---|
| 2 s sampler | `kgsl <clock MHz> <max clock MHz> <thermal_pwrlevel> <temp m°C>`, `kgsl_page_alloc <bytes>` | `gpu mali <cur_freq kHz> <total_gpu_mem bytes> <dma_buf_gpu_mem bytes>` (`?` for a file it cannot read) |
| The GPU's thermal level (ready gate) | `thermal_pwrlevel` | `gpu_cooling`: the largest `mValue` among the GPU cooling devices (`mType` 3; on the Pixel 8a only `thermal-gpufreq-0`; 0 = not capped), from the same `dumpsys thermalservice` as the thermal status |
| The GPU's temperature | `kgsl-3d0/temp` | the HAL sensor `G3D` (slow samples and the state files) |
| State files (`phone_state`) | `gpu: kind=kgsl`, `kgsl: temp= clock_mhz= max_clock_mhz= thermal_pwrlevel=` | `gpu: kind=mali cur_freq= total_gpu_mem= dma_buf_gpu_mem= gpu_cooling= G3D=` |
| Memory line of a leg | `kgsl_page_alloc_max_B` | `mali_total_gpu_mem_max_B` |
| Cooling devices (`s_gpu.txt`: `cooling: hal`, `sysfs` or `none`) | the thermal service's list when it has devices, else `/sys/class/thermal` (the Galaxy S26: `sysfs`) | the same rule (the Pixel 8a: `hal`) |

Temperatures come from the thermal service's current section only. `dumpsys thermalservice` prints a `Cached temperatures:` section (the values of the HAL's last callback, which can be old) before `Current temperatures from HAL:`; on one Galaxy S26 the cached section said `SKIN` 37.9 °C while the current one said 30.1 °C, and in a recorded Pixel 8a dump the cached `G3D` was 58.0 °C against 31.0 °C current. The skin temperature (`SKIN` on the S26, `VIRTUAL-SKIN` on the Pixel), the GPU's `G3D` and the state files' `hal_temps:` are read from the current section; the thermal status is a line of its own, and the battery temperature comes from `dumpsys battery`.

Cooling devices come from one source per run, decided after the hold and named in `s_gpu.txt`:

- `hal`: the thermal service lists cooling devices. The CPU and GPU ones are chosen by their type, never by their name (names differ between phones): `mType` 2 = CPU, 3 = GPU. Slow samples: `cooling <name> <mValue> type=<mType>` (for example `cooling thermal-cpufreq-0 0 type=2`); state files: `cooling: <name>=<mValue>(type <mType>);` per device. A Pixel 8a (Android 16) lists nine cooling devices: `thermal-cpufreq-0/1/2` (`mType` 2) and `thermal-gpufreq-0` (3), the four these reads keep, and five of types 1 and 5 (`dc_icl`, `usbc-port`, `fcc`, `gxp-cooling`, `tpu_cooling`).
- `sysfs`: the thermal service lists none and the shell user reads `/sys/class/thermal/cooling_device*/type`. The devices of the types `cpufreq-*` and `cpu-cluster*` (CPU), `gpu` and `kgsl` (GPU), `cdsp*` (the DSP) and `ddr-cdev` (the memory bus) are kept, in the phone's order of the directory names. Slow samples: `cooling <type> <cur_state>/<max_state>` (for example `cooling gpu 0/18`); state files: `cooling: <type>=<cur_state>/<max_state>;` per device. A Galaxy S26 (Android 16) is such a phone: its thermal service has temperatures (`AP`, `BAT`, `PA`, `SKIN`, `SUBBAT`, `USB`) and the thresholds of `BAT` and `SKIN`, but an empty cooling-device list, and the shell user reads all 62 of its sysfs cooling devices. The reads keep 12: `cpufreq-cpu0` 0/27, `cpufreq-cpu6` 0/30, `cpu-cluster0` 0/27, `cpu-cluster1` 0/30, `kgsl` and `gpu` 0/18, `cdsp_hw` 0/1, `cdsp_ss` 0/12, `cdsp` 0/12, `cdsp_sw_hvx` 0/11, `cdsp_sw_hmx` 0/12 and `ddr-cdev` 0/1 at rest; the other 50 are CPU hotplug and pause devices, modem, PA and mmWave devices, the panel backlight, `display-fps` and `ufs`.
- `none`: neither; no slow line, and the state files say `cooling: n/a`.

The chains keep the thermal service's dump as it is in `s_thermal_raw.txt` and every sysfs cooling device with its type and states (or `cat`'s error) in `s_cooling_sysfs.txt`, once per hold. The ready gate's GPU level stays `thermal_pwrlevel` on kgsl. The CPU caps are the same `cpufreq/policy*` reads on both.

On the Pixel 8a (Android 16), the shell user reads `cur_freq` (kHz; 150000 at idle), `hint_min_freq`, `dvfs_period`, `capacity_headroom`, `dma_buf_gpu_mem` and `total_gpu_mem` under `/sys/class/misc/mali0/device/` (= `/sys/devices/platform/1f000000.mali`), but not `max_freq`, `min_freq`, `scaling_max_freq`, `hint_max_freq`, `governor`, `power_policy`, `available_frequencies`, `gpu_top`, `clock_info` or `dvfs_table` (permission denied), nor `/sys/class/devfreq/*/cur_freq` or `/sys/class/thermal/thermal_zone*/type`; it has no `/sys/class/kgsl` and no `/sys/kernel/gpu/`. A run writes, for each of these files, `readable <file>` or `<file>: <cat's error>` into `s_gpu.txt` (judged by `cat`: the shell user cannot even `stat` some of them, so `[ -e ]` says absent for a file that is there). Measured there: the GPU legs' `cur_freq` rose from 150000 kHz at rest to 890000 kHz under load, and `thermal-gpufreq-0` stayed 0 through the run.

## Rules kept from earlier chains

- No `timeout adb`: it kills only the Mac's client while the phone keeps running. Waits poll the phone.
- Every remote `date` format is quoted; `mkdir -p` before a staged push; `LC_ALL=C sort`.
- lmkd counts only lines naming this package; the chain's own force-stop is `MEMGUARD_STOPPED`.
- The foreground check judges a live process only; a process gone one second after the start is decided by its report.
- Never a bare `wait`: the keeper is a child, background scorers are waited for by pid.
- Every adb call reads `/dev/null` except `exec-in`: an adb call inside a `while read` loop would otherwise eat the loop's input (the fake adb's `shell` eats stdin the same way, so a dry run shows the bug).
- `exec-out` has no separate stderr: a pull of a missing file returns the phone's error text. Every pulled file is checked (byte count against the report, the report parsed as JSON; a report that is not JSON is kept as `<tag>.report_pull.txt`).
- An extra reaches `am` only with a value: `--es key ""` makes `am` read the next flag as the value.
- Never edit a body file in place while any lane may be running it: write a new file and `mv` it over the old one (a running bash keeps the old inode). The keeper runs from a copy in the round dir.
- The time is checked before every graph transfer; a long parity leg heats the phone, so timing legs get their own rested start.

## NPU legs: npu_chain.sh

NPU legs use a native C-API runner instead of the app (pushed with its runtime libraries into `NPU_STAGE`, started with `adb shell`), and a plan with other columns:

```
tag  accel  precision  graph  fx  count  warmup  rounds  ab  cache  compile_s  ready  [extras]
```

`accel` is `npu+cpu`, `npu` or `gpu`; `precision` is `-` for the NPU; `graph` goes through `lane_graph_file`; `fx` names a fixture dir whose manifest `<FX_ROOT>/<fx>.manifest.json` gives its dir, file count, byte count, the sha256 of its files concatenated in byte order, `L`, `out_elems` and the graph's input order; `count` fixtures run the parity pass, then `warmup` and `rounds` calls on fixture `ab`; `cache` is `-`, `new:<dir>` (a cold JIT compile, the dir removed first) or `use:<dir>` (from that cache, skipped when it is not on the phone); `compile_s` is the compile watch (no `compile_end` line by then: the runner is killed, `COMPILE_TIMEOUT`, never retried); `ready` is `gate`, `clean` or `uncapped` (an `uncapped` leg not uncapped by the limit runs warm). Extras: `rest=`, `needs=`, `guard=`, `sample=`.

Its defaults differ from `chain.sh`: `REST_MAX` 180, `REST_MAX_T` 300, `POLL` 10, `MEM_GUARD_KB` 1000000 (the runner is killed, `OOM_GUARD`), `XFER_MAX_S` 300, `rest=` 30.

Per leg: the graph pushed with its byte count and sha256 checked (graphs and caches no later leg needs are removed first; `/data` keeps room for each cold leg's cache), the ready gate, a device clock mark, the runner with the 2-second sampler, the compile watch, the memory guard and the run-end watch, any runner left alive killed by `pidof`, `/proc/uptime`, logcat since the mark (the runner pid's delegation lines to `<tag>.delegate_key.txt`; lmkd naming the runner = hard stop), the runner's stderr pulled, the dumps pulled and counted, `<tag>.leg.json`, then `lane_npu_score` in the background. The runner's stdout streams to `<tag>.stdout.txt` (the compile watch reads it); its stderr, where LiteRT writes its own log lines, goes to a file in `NPU_STAGE` on the phone and is pulled to `<tag>.stderr.txt` with its byte count checked: one stream per file, so a runner line never lands behind a fragment of a LiteRT line (on one Galaxy S26 run the runner's `summary` line arrived as `INFO: summary ...` when both streams shared one file). Measured on the S26: LiteRT's own lines go to `stderr.txt`; the Qualcomm library's log (`QnnDsp` lines) and one OpenCL loading line (`INFO: Loaded OpenCL library with dlopen.`) stay on the runner's stdout, as whole lines. `<tag>.delegate.txt` holds the runner's own delegation lines from both files first (`<file>:<line>`; the HTP partition line is in no logcat), then the runner pid's logcat lines. The runtime libraries are checked against `NPU_LIBS_SUMS` on the Mac and read back on the phone; the runner against `NPU_RUNNER_SHA`.
