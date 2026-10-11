#!/bin/bash
# dryrun.sh: the chain's dry run on the Mac against a fake phone: no phone, no real adb, no real hold.
#   bash dryrun.sh <lane.env>                  an Adreno (kgsl) fake phone
#   DRY_GPU=mali bash dryrun.sh <lane.env>     an Arm Mali fake phone (fake_adb.py FAKE_GPU=mali)
# A few minutes; the lane's stand-in runs the lane's real graphs on the Mac CPU (take the Mac's measurement window for
# it). Everything goes to cache/dryrun/<LANE>/ (a Mali run: cache/dryrun/<LANE>-mali/; removed and rewritten by every
# run); the summary and its checks to summary.json there; exit 0 only when every check is ok.
# lane.env gives: STANDIN (the app's stand-in), DRY_PLAN (scenario 1's plan), DRY_APK (an APK; its sha256 in
# <DRY_APK>.sha256), DRY_LOWMEM_TAG (the leg of DRY_PLAN whose app gets stuck in its compile under low memory),
# DRY_LOST_LEG (one plan line, a long timing leg, for scenario 2) and, optionally, DRY_REFERENCE (the summary.json of an
# earlier dry run of the same plan, whose items this run must reproduce; a kgsl phone's record, so a Mali run does not
# compare with it: the app's own GPU cool wait reads kgsl and falls back to a fixed sleep on Mali).
# Guards: an `adb` tripwire first on PATH (any call of `adb` by name is written to real_adb_calls.log and fails with
# 97), ADB = fake_adb.py, the hold file, the round dirs and the results dir under cache/ (the chain refuses a dry run
# with anything else).
# Scenario 1 (DRY_PLAN): another lane holds the fake phone for 15 s and an entry without a keeper sits at the head of
#   the line; the chain's keeper waits behind the holder, takes the hold; check; mac-baseline; the legs (DRY_PLAN should
#   cover the paths: gate on each accel / precision, timing, needs, needs_pass PASS and not PASS, STOP at the limit,
#   the memory guard on a stuck app, a rows file of another bucket, an extra output file, a graph swap); cleanup;
#   release.
# Scenario 2: the phone leaves adb 4 s into DRY_LOST_LEG -> hard stop, the hold released; the phone comes back ->
#   `chain.sh cleanup` takes a hold, cleans, releases.
# Scenario 3: this app is already running on the phone when the hold comes -> no leg, release; then the cleanup.
# The checks (summary.json "checks": name, ok, got, want): the exit code of every step; 0 real adb calls; scenario 1's
# hold taken after the foreign holder's release, every fake adb call between the keeper's acquisition and its release,
# one transfer check, as many ready gate / leg start / leg status lines as legs run, no pull CHECK, a phone sha256 MATCH
# for every small file, every graph swap checked, every plan leg run or skipped with a reason, the fake phone clean, the
# GPU reads of the fake phone's kind (a GPU line in every leg's samples and no failed GPU read, a gpu: line in every
# state file, the GPU level in every ready gate line; on Mali also G3D and the GPU's cooling device in the slow
# samples), the thermal service's raw dump and the sysfs cooling devices kept after the hold, the cooling source in
# s_gpu.txt (sysfs on the kgsl fake, whose thermal service lists none as a Galaxy S26's; hal on the Mali fake, which
# lists the Pixel 8a's nine), the cooling devices from that source (every state file's cooling: line = the HAL's mType 2
# / 3 devices or the kept sysfs devices with their states, a GPU device and nothing else in every leg's slow samples),
# every temperature from the HAL's current section and none from the cached one (the state files' skin: and hal_temps:,
# the slow samples' skin and, on Mali, G3D), every leg's report JSON (a missing report's pull kept aside); scenario 2's
# hard stop in the lost leg; scenario 3's stop for the busy phone and only the other session's calls outside a hold;
# both cleanups clean; the fake hold free. With DRY_REFERENCE, also: the legs' statuses, the skips, the verdicts, the
# log and graph line counts, the host scorer's results (summary, same-file and earlier-round comparisons), the timing
# sets, the stand-in's layout check and scenarios 2 and 3 equal to the reference.
set -u
BENCH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LANE_ENV_ARG=${1:?usage: dryrun.sh <lane.env>}
export DRY_RUN=1
# shellcheck source=lib.sh
. "$BENCH/lib.sh"
ROUND=dryrun
bench_load "$LANE_ENV_ARG"
for v in STANDIN DRY_PLAN DRY_APK DRY_LOWMEM_TAG DRY_LOST_LEG; do
  [ -n "${!v:-}" ] || { echo "$LANE_ENV: $v is not set (dryrun.sh needs it)" >&2; exit 2; }
done
[ -s "$DRY_APK.sha256" ] || { echo "no $DRY_APK.sha256" >&2; exit 2; }
DRY_GPU=${DRY_GPU:-kgsl}
case "$DRY_GPU" in
  kgsl) OUT=$BENCH/cache/dryrun/$LANE ;;
  mali) OUT=$BENCH/cache/dryrun/$LANE-mali
        [ -z "${DRY_REFERENCE:-}" ] || echo "a Mali fake phone: not compared with $DRY_REFERENCE (a kgsl phone's record)"
        DRY_REFERENCE="" ;;
  *) echo "DRY_GPU is kgsl or mali (got $DRY_GPU)" >&2; exit 2 ;;
esac
export FAKE_GPU=$DRY_GPU
CHAIN=$BENCH/chain.sh
rm -rf "$OUT"; mkdir -p "$OUT/tripwire" "$OUT/results"
cat > "$OUT/tripwire/adb" <<EOF
#!/bin/bash
echo "\$(date '+%F %T') real adb called: \$*" >> "$OUT/real_adb_calls.log"
exit 97
EOF
chmod +x "$OUT/tripwire/adb"; : > "$OUT/real_adb_calls.log"; : > "$OUT/rc.txt"
export PATH="$OUT/tripwire:$PATH"
[ "$(command -v adb)" = "$OUT/tripwire/adb" ] || { echo "the tripwire is not the first adb on PATH"; exit 2; }
export ADB=$BENCH/fake_adb.py HOLD=$OUT/fake_hold RESULTS_DIR=$OUT/results APK=$DRY_APK
APK_SHA=$(cut -d' ' -f1 "$DRY_APK.sha256"); export APK_SHA
export REST_MAX=20 REST_MAX_T=30 POLL=2
export FAKE_PKG=$PKG FAKE_ACTIVITY=$ACTIVITY FAKE_SERIAL=$SERIAL FAKE_STANDIN=$STANDIN
step() {  # label out command...: run one step with its output to <out>, its exit code to rc.txt
  local label=$1 log=$2 rc
  shift 2
  "$@" > "$log" 2>&1; rc=$?
  echo "$label $rc" | tee -a "$OUT/rc.txt"
}
echo "dry run $(date '+%F %T'); lane $LANE ($LANE_ENV); fake GPU $FAKE_GPU; adb on PATH = $(command -v adb); ADB = $ADB; hold = $HOLD"

# ---- scenario 1
export FAKE_ROOT=$OUT/s1_phone ROUND_DIR=$OUT/s1 SCRIPT_NAME=$LANE-dryrun-$DEVICE PLAN=$DRY_PLAN
mkdir -p "$FAKE_ROOT/_fault"; echo "$DRY_LOWMEM_TAG" > "$FAKE_ROOT/_fault/lowmem_tag"
sleep 600 & FOREIGN=$!
python3 "$CA/hold_cli.py" acquire "$HOLD" foreign-lane-dryrun $FOREIGN
( sleep 15; python3 "$CA/hold_cli.py" release "$HOLD" $FOREIGN; kill $FOREIGN 2>/dev/null
  echo "$(date '+%F %T') foreign holder released" >> "$OUT/foreign_holder.log" ) &
RELEASER=$!
python3 "$CA/queue_cli.py" enqueue "$HOLD" other-lane-no-keeper someone 10 > /dev/null
python3 "$CA/queue_cli.py" enqueue "$HOLD" "$SCRIPT_NAME" dryrun 30 > "$OUT/s1_queue_before.txt"
cat "$OUT/s1_queue_before.txt"
step s1_check "$OUT/s1_check.txt" bash "$CHAIN" check dryrun "$LANE_ENV"
step s1_mac_baseline "$OUT/s1_mac_baseline.txt" bash "$CHAIN" mac-baseline dryrun "$LANE_ENV"
cat "$OUT/s1_mac_baseline.txt"
step s1_run "$OUT/s1_chain.log" bash "$CHAIN" run dryrun "$LANE_ENV"
wait "$RELEASER" 2>/dev/null
python3 "$CA/queue_cli.py" list "$HOLD" > "$OUT/s1_queue_after.txt"

# ---- scenario 2: the phone leaves adb 4 s into a long timing leg
export FAKE_ROOT=$OUT/s2_phone ROUND_DIR=$OUT/s2 SCRIPT_NAME=$LANE-dryrun-lost-$DEVICE PLAN=$OUT/plan_lost.tsv
mkdir -p "$FAKE_ROOT/_fault"
printf '%s\n' "$DRY_LOST_LEG" > "$PLAN"
LOST_TAG=${DRY_LOST_LEG%% *}
( t=0; until [ -s "$ROUND_DIR/$LOST_TAG.running.txt" ] || [ $t -ge 900 ]; do sleep 1; t=$((t + 1)); done
  [ -s "$ROUND_DIR/$LOST_TAG.running.txt" ] || exit 0
  sleep 4; : > "$FAKE_ROOT/_fault/gone"; echo "$(date '+%F %T') phone gone" >> "$OUT/s2_fault.log" ) &
FAULT=$!
step s2_run "$OUT/s2_chain.log" bash "$CHAIN" run dryrun-lost "$LANE_ENV"
kill "$FAULT" 2>/dev/null; wait "$FAULT" 2>/dev/null
rm -f "$FAKE_ROOT/_fault/gone"; echo "$(date '+%F %T') phone back" >> "$OUT/s2_fault.log"
step s2_cleanup "$OUT/s2_cleanup.log" bash "$CHAIN" cleanup dryrun-lost "$LANE_ENV"

# ---- scenario 3: this app already runs on the phone (a leftover, or another session's) when the hold comes
export FAKE_ROOT=$OUT/s3_phone ROUND_DIR=$OUT/s3 SCRIPT_NAME=$LANE-dryrun-busy-$DEVICE PLAN=$DRY_PLAN
mkdir -p "$FAKE_ROOT/_fault"; echo leftover > "$FAKE_ROOT/_fault/lowmem_tag"
"$ADB" -s "$SERIAL" install -r "$APK" > "$OUT/s3_setup.txt" 2>&1
"$ADB" -s "$SERIAL" shell am start -W -n "$PKG/$ACTIVITY" --es graph x.tflite --es rows x.json --es report leftover.json \
  >> "$OUT/s3_setup.txt" 2>&1
rm -f "$FAKE_ROOT/_fault/lowmem_tag"
# these first calls are the other session's, outside every hold by design; the 1 s pause keeps them out of the second in
# which the keeper takes the hold (the keeper's times have 1 s resolution)
wc -l < "$FAKE_ROOT/adb_calls.log" | tr -d ' ' > "$OUT/s3_setup_calls.txt"; sleep 1
step s3_run "$OUT/s3_chain.log" bash "$CHAIN" run dryrun-busy "$LANE_ENV"
step s3_cleanup "$OUT/s3_cleanup.log" bash "$CHAIN" cleanup dryrun-busy "$LANE_ENV"

# ---- the summary and the checks
python3 -I - "$OUT" "$HOLD" "$PKG" "$STAGE" "$DRY_PLAN" "$LOST_TAG" "${DRY_REFERENCE:-}" "$LANE" "$LANE_ENV" "$DRY_GPU" <<'PY'
import json, re, sys, time
from pathlib import Path
out, hold, pkg, stage = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4].lstrip("/")
plan, lost_tag, ref_path, lane, lane_env = Path(sys.argv[5]), sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9]
gpu_kind = sys.argv[10]
TS = r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)"
WANT_RC = {"s1_check": 0, "s1_mac_baseline": 0, "s1_run": 0, "s2_run": 9, "s2_cleanup": 0, "s3_run": 5, "s3_cleanup": 0}


def t_of(s):
    return time.mktime(time.strptime(s, "%Y-%m-%d %H:%M:%S"))


def read(p):
    return p.read_text() if p.exists() else ""


def keeper_window(log):
    text = read(log)
    acq = [t_of(m) for m in re.findall(r"HOLD ACQUIRED by pid \d+ at " + TS, text)]
    rel = [t_of(m) for m in re.findall(r"HOLD RELEASED at " + TS, text)]
    return acq, rel


def calls(phone):
    p = phone / "adb_calls.log"
    return [(float(l.split(" ", 1)[0]), l.split(" ", 1)[1].strip()) for l in p.read_text().splitlines()] if p.exists() else []


def outside(c, acq, rel):
    # the fake adb calls outside every hold; the keeper's times have 1 s resolution, so a call in the second of an
    # acquisition or a release counts as inside
    return [cmd[:100] for t, cmd in c if not any(a <= t <= r + 1 for a, r in zip(acq, rel))]


def wall(text):
    m0 = re.search(r"^chain \S+ pid \d+ " + TS, text, re.M)
    m1 = re.search(r"^chain exit \d+ " + TS, text, re.M)
    return int(t_of(m1.group(1)) - t_of(m0.group(1))) if m0 and m1 else None


def plan_tags(p):
    tags = []
    for line in read(p).splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            tags.append(line.split()[0])
    return tags


def load(p):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def host_check(d):
    s = d.get("summary") or {}
    return {"rows": s.get("rows"), "argmax": s.get("argmax_outside_near_tie"), "near_tie": s.get("near_tie"),
            "max_abs_dp": s.get("max_abs_dp"), "mean_abs_dp": s.get("mean_abs_dp"), "bar_pass": s.get("bar_pass"),
            "red_arm": s.get("red_arm_max_abs_dp"), "delegate_lines": len((d.get("delegation") or {}).get("lines") or []),
            "same_file": {k: [v.get("rows_compared"), v.get("rows_bit_equal"), v.get("max_abs_dscore_markers"), v.get("max_abs_dp")]
                          for k, v in (d.get("same_file") or {}).items()},
            "round3": {k: [v.get("rows_compared"), v.get("rows_logits_bit_equal"), v.get("max_abs_dlogit"), v.get("max_abs_dp")]
                       for k, v in (d.get("round3_same_file") or {}).items()}}


def timing(d):
    sets = {}
    for k, v in (d.get("sets") or {}).items():
        sets[k] = [((v.get("per_call_ms_write_run_read") or {}).get("n")), (((v.get("split") or {}).get("burst") or {}).get("n")),
                   ((v.get("cool") or {}).get("mode"))]
    return {"status": (d.get("run") or {}).get("status"), "stopped_early": (d.get("run") or {}).get("stopped_early"), "sets": sets}


res = {"lane": lane, "lane_env": lane_env, "plan": str(plan), "fake_gpu": gpu_kind,
       "rc": {l.split()[0]: int(l.split()[1]) for l in read(out / "rc.txt").splitlines() if len(l.split()) == 2},
       "real_adb_calls": len(read(out / "real_adb_calls.log").splitlines())}
s1, log1 = out / "s1", read(out / "s1_chain.log")
acq, rel = keeper_window(s1 / "keeper.log")
c1 = calls(out / "s1_phone")
graphs = read(s1 / "s_graphs.txt")
foreign = read(out / "foreign_holder.log").strip()
res["s1"] = {
    "hold_acquired": [time.strftime("%T", time.localtime(t)) for t in acq],
    "hold_released": [time.strftime("%T", time.localtime(t)) for t in rel],
    "foreign_release": foreign or None,
    "fake_adb_calls": len(c1),
    "calls_outside_hold": outside(c1, acq, rel),
    "all_calls_inside_hold": bool(acq and rel and c1) and not outside(c1, acq, rel),
    "wall_s": wall(log1),
    "log_lines": {k: len(re.findall(p, log1)) for k, p in (
        ("transfer_check", r"=== transfer check"), ("ready_gate", r"ready gate: "), ("leg_start", r"=== leg \S+ start"),
        ("leg_status", r"=== leg \S+ status"), ("pull_sel_match", r"sel bytes: report=\d+ pulled=\d+ MATCH"),
        ("pull_full_match", r"full bytes: report=\d+ pulled=\d+ MATCH"), ("pull_check", r"bytes: report=\S* pulled=\S* CHECK"),
        ("small_file_sha_match", r"sha256 \S+\.(?:json|f32) phone=\S+ mac=\S+ MATCH"),
        ("cleanup_pm_empty", r"pm list: ''"), ("release_written", r"RELEASE_HOLD written"), ("keeper_released", r"keeper released the hold"))},
    "graph_lines": {k: len(re.findall(p, graphs)) for k, p in (
        ("exec_in_bytes_equal", r"exec-in \S+\.tflite rc=0 bytes (\d+)/\1 "), ("sha_match", r"sha256 \S+\.tflite phone=(\S+) mac=\1 MATCH"),
        ("already_on_phone", r"already on the phone"), ("removed", r"(?m)^removed \S+\.tflite"))},
    "legs": {p.name[: -len(".leg_status")]: p.read_text().strip() for p in sorted(s1.glob("*.leg_status"))},
    "skipped": read(s1 / "s_skipped.txt").splitlines(),
    "verdicts": {p.name[: -len(".verdict")]: p.read_text().strip() for p in sorted(s1.glob("*.verdict"))},
    "host_checks": {}, "timing": {}, "layout_equals_host": {},
    "phone_after": {"package": (out / f"s1_phone/data/app/{pkg}").exists(),
                    "stage": (out / "s1_phone" / stage).exists(),
                    "app_data": (out / f"s1_phone/data/data/{pkg}").exists()},
}
# the GPU reads of the fake phone's kind: the sampler's GPU line, no failed read (a cat error, an empty kgsl field, a "?"
# Mali field, a line of the other kind), the state files' gpu: line, the ready gate's GPU level (and on Mali the slow
# samples' G3D temperature and GPU cooling device line, mType 3)
GPU_LINE = {"kgsl": r"kgsl \d+ \d+ \d+ \d+", "mali": r"gpu mali \d+ \d+ \d+"}[gpu_kind]
OTHER = {"kgsl": r"gpu mali", "mali": r"kgsl"}[gpu_kind]
LEVEL = {"kgsl": "kgsl_pwrlevel", "mali": "gpu_cooling"}[gpu_kind]
gpu = {"legs": {}, "state_files": 0, "state_files_with_gpu_line": 0, "ready_lines": 0, "ready_lines_with_level": 0,
       "s_gpu": read(s1 / "s_gpu.txt").strip()}
for p in sorted(s1.glob("*.samples.txt")):
    ls = read(p).splitlines()
    gpu["legs"][p.name[: -len(".samples.txt")]] = {
        "gpu_lines": sum(bool(re.fullmatch(GPU_LINE, l)) for l in ls),
        "failed_reads": sum(bool(re.search(r"No such file|Permission denied|^cat: ", l)) or bool(re.match(OTHER, l))
                            or (l.startswith(("kgsl ", "gpu mali")) and not re.fullmatch(GPU_LINE, l)) for l in ls),
        "g3d_lines": sum(bool(re.fullmatch(r"mValue=[-0-9.]+, mType=\d+, mName=G3D", l)) for l in ls),
        "gpu_cooling_lines": sum(bool(re.fullmatch(r"cooling .+ \d+ type=3", l)) for l in ls)}
STATE_FILES = [s1 / "s0_state_initial.txt", s1 / "s9_state_end.txt"] + sorted(s1.glob("*.state_before.txt")) + sorted(s1.glob("*.state_after.txt"))
for p in STATE_FILES:
    if p.exists():
        gpu["state_files"] += 1
        gpu["state_files_with_gpu_line"] += bool(re.search(rf"(?m)^gpu: kind={gpu_kind}\b", read(p)))
for l in read(s1 / "s_ready.txt").splitlines():
    if re.match(r"\d\d:\d\d:\d\d ready\? ", l):
        gpu["ready_lines"] += 1
        gpu["ready_lines_with_level"] += bool(re.search(rf" {re.escape(LEVEL)} \d+ rested ", l))
res["s1"]["gpu"] = gpu
raw = read(s1 / "s_thermal_raw.txt")


def section(text, start, end):
    """The lines of one section of the thermal service's dump (after the line starting with `start`, before `end`)."""
    out, on = [], False
    for l in text.splitlines():
        if on and l.startswith(end):
            break
        if on:
            out.append(l)
        on = on or l.startswith(start)
    return out


# the temperatures, from the HAL's current section only (the fake's cached section holds older, different values):
# every state file's skin: and hal_temps: lines, every slow sample's skin line and, on Mali, the state's G3D= and the slow
# samples' G3D line
cur_sec = section(raw, "Current temperatures from HAL", "Current cooling devices from HAL")
cached_sec = section(raw, "Cached temperatures", "HAL Ready")
SKIN_RE, G3D_RE = r"mValue=[-0-9.]+, mType=3, mName=[A-Za-z0-9_-]*SKIN", r"mValue=[-0-9.]+, mType=\d+, mName=G3D"


def first(lines, rx):
    for l in lines:
        m = re.search(rx, l)
        if m:
            return m.group(0)
    return None


temps = {"skin_current": first(cur_sec, SKIN_RE), "skin_cached": first(cached_sec, SKIN_RE),
         "g3d_current": first(cur_sec, G3D_RE), "g3d_cached": first(cached_sec, G3D_RE),
         "want_hal_temps": "".join(m.group(0)[len("Temperature{"):] + ";" for l in cur_sec
                                   for m in [re.search(r"Temperature\{mValue=[-0-9.]+, mType=\d+, mName=[A-Za-z0-9_-]+", l)] if m),
         "state_files": 0, "state_skin_current": 0, "state_hal_temps_current": 0, "state_g3d_current": 0,
         "slow_skin_lines": 0, "slow_skin_current": 0, "slow_g3d_lines": 0, "slow_g3d_current": 0}
g3d_val = (re.match(r"mValue=([-0-9.]+)", temps["g3d_current"] or "") or [None, None])[1]
for p in STATE_FILES:
    if p.exists():
        t = read(p)
        temps["state_files"] += 1
        temps["state_skin_current"] += bool(re.search(rf"(?m)^skin: {re.escape(temps['skin_current'] or '-')}$", t))
        temps["state_hal_temps_current"] += bool(re.search(rf"(?m)^hal_temps: {re.escape(temps['want_hal_temps'])}$", t))
        temps["state_g3d_current"] += bool(g3d_val and re.search(rf"(?m)^gpu: kind=mali .* G3D={re.escape(g3d_val)}$", t))
for p in sorted(s1.glob("*.samples.txt")):
    for l in read(p).splitlines():
        if re.fullmatch(SKIN_RE, l):
            temps["slow_skin_lines"] += 1
            temps["slow_skin_current"] += l == temps["skin_current"]
        if re.fullmatch(G3D_RE, l):
            temps["slow_g3d_lines"] += 1
            temps["slow_g3d_current"] += l == temps["g3d_current"]
res["s1"]["temperatures"] = temps
# the cooling devices from the phone's own source: the HAL's list when it has devices (the Mali fake: the CPU / GPU ones
# chosen by type, mType 2 = CPU, 3 = GPU, whatever their names), else the kept sysfs devices (the kgsl fake, a Galaxy
# S26's form: cpufreq-*, cpu-cluster*, gpu, kgsl, cdsp*, ddr-cdev, in the phone's glob order); against s_gpu.txt's
# cooling: line, every state file's cooling: line and every leg's slow-sample cooling lines
devs = re.findall(r"CoolingDevice\{mValue=(\d+), mType=(\d+), mName=([^}]*)\}", "\n".join(section(raw, "Current cooling devices from HAL", "Temperature static thresholds")))
KEPT = re.compile(r"(cpufreq-[A-Za-z0-9_.-]*|cpu-cluster[A-Za-z0-9_.-]*|gpu|kgsl|cdsp[A-Za-z0-9_.-]*|ddr-cdev)")
sysfs = []
for d in sorted((out / "s1_phone/sys/class/thermal").glob("cooling_device*")):
    sysfs.append([read(d / f).strip() for f in ("type", "cur_state", "max_state")])
kept = [x for x in sysfs if KEPT.fullmatch(x[0])]
source = "hal" if devs else "sysfs" if kept else "none"
cool = {"raw_thermal_status": bool(re.search(r"(?m)^Thermal Status: \d+$", raw)), "raw_devices": len(devs),
        "raw_types": sorted({t for _, t, _ in devs}), "sysfs_devices": len(sysfs), "sysfs_kept": len(kept),
        "s_cooling_sysfs_lines": len(re.findall(r"(?m)^cooling_device\d+ \S+ \d+/\d+$", read(s1 / "s_cooling_sysfs.txt"))),
        "want_source": source, "s_gpu_cooling": (re.search(r"(?m)^cooling: (\S+)", read(s1 / "s_gpu.txt")) or [None, None])[1],
        "want_state": ("".join(f"{n}={v}(type {t});" for v, t, n in devs if t in ("2", "3")) if source == "hal"
                       else "".join(f"{t}={c}/{m};" for t, c, m in kept) if source == "sysfs" else "n/a"),
        "state_files": 0, "state_files_equal": 0, "legs": {}}
for p in STATE_FILES:
    if p.exists():
        cool["state_files"] += 1
        m = re.search(r"(?m)^cooling: (.*)$", read(p))
        cool["state_files_equal"] += bool(m and m.group(1) == cool["want_state"])
WANT_SLOW = ({f"cooling {n} {v} type={t}" for v, t, n in devs if t in ("2", "3")} if source == "hal"
             else {f"cooling {t} {c}/{m}" for t, c, m in kept})
GPU_SLOW = (re.compile(r"cooling .+ \d+ type=3") if source == "hal" else re.compile(r"cooling (gpu|kgsl) \d+/\d+"))
for p in sorted(s1.glob("*.samples.txt")):
    ls = [l for l in read(p).splitlines() if l.startswith("cooling ")]
    cool["legs"][p.name[: -len(".samples.txt")]] = {"lines": len(ls), "gpu_lines": sum(bool(GPU_SLOW.fullmatch(l)) for l in ls),
                                                     "other_lines": sum(l not in WANT_SLOW for l in ls)}
res["s1"]["cooling"] = cool
for p in sorted((out / "results").glob("*.json")):
    d = load(p)
    if not isinstance(d, dict):
        continue
    if isinstance(d.get("summary"), dict) and "bar_pass" in d["summary"]:
        res["s1"]["host_checks"][p.stem] = host_check(d)
    elif isinstance(d.get("sets"), dict) and isinstance(d.get("run"), dict):
        res["s1"]["timing"][p.stem] = timing(d)
for tag in plan_tags(plan):
    d = load(s1 / f"{tag}.json")
    if isinstance(d, dict) and d.get("layout_equals_host_build_inputs"):
        res["s1"]["layout_equals_host"][tag] = d["layout_equals_host_build_inputs"]
for name, phone in (("s2", "s2_phone"), ("s3", "s3_phone")):
    r = out / name
    a, rl = keeper_window(r / "keeper.log")
    c = calls(out / phone)
    n = int(read(out / f"{name}_setup_calls.txt") or 0)
    res[name] = {"chain_stopped": read(r / "CHAIN_STOPPED").strip() or None,
                 "holds_acquired": len(a), "holds_released": len(rl), "fake_adb_calls": len(c),
                 # scenario 3's setup = the other session's install and start of the app, outside every hold by design
                 "setup_calls": len(c[:n]), "setup_calls_outside_hold": len(outside(c[:n], a, rl)),
                 "calls_outside_hold": outside(c[n:], a, rl), "wall_s": wall(read(out / f"{name}_chain.log")),
                 "legs": {p.name[: -len(".leg_status")]: p.read_text().strip() for p in sorted(r.glob("*.leg_status"))},
                 "phone_after_cleanup": {"package": (out / phone / f"data/app/{pkg}").exists(),
                                         "stage": (out / phone / stage).exists()}}
res["hold_file_after"] = read(hold) or None
res["queue_after"] = json.loads(read(Path(str(hold) + ".queue")) or "null")

checks = []


def check(name, ok, got=None, want=None):
    checks.append({"name": name, "ok": bool(ok), "got": got, "want": want})


for k, want in WANT_RC.items():
    check(f"exit code {k}", res["rc"].get(k) == want, res["rc"].get(k), want)
check("no real adb call", res["real_adb_calls"] == 0, res["real_adb_calls"], 0)
S1 = res["s1"]
m = re.search(TS, foreign)
check("s1 hold taken after the foreign holder released it", bool(acq and m and acq[0] >= t_of(m.group(1)) - 1),
      S1["hold_acquired"][:1], foreign)
check("s1 every fake adb call inside the hold", S1["all_calls_inside_hold"], S1["calls_outside_hold"][:5], [])
ll, legs_run = S1["log_lines"], len(S1["legs"])
check("s1 one transfer check", ll["transfer_check"] == 1, ll["transfer_check"], 1)
check("s1 ready gate = leg start = leg status = legs run", legs_run > 0 and ll["ready_gate"] == ll["leg_start"] == ll["leg_status"] == legs_run,
      [ll["ready_gate"], ll["leg_start"], ll["leg_status"]], legs_run)
check("s1 no pulled file CHECK", ll["pull_check"] == 0, ll["pull_check"], 0)
small = [l.split()[1] for l in read(s1 / "inputs.sha256").splitlines() if len(l.split()) == 2 and not l.split()[1].endswith(".tflite")]
check("s1 a phone sha256 MATCH for every small file", ll["small_file_sha_match"] == len(small) > 0, ll["small_file_sha_match"], len(small))
check("s1 cleanup, RELEASE_HOLD and the keeper's release", ll["cleanup_pm_empty"] == ll["release_written"] == ll["keeper_released"] == 1,
      [ll["cleanup_pm_empty"], ll["release_written"], ll["keeper_released"]], [1, 1, 1])
gl = S1["graph_lines"]
check("s1 every graph swap checked (sha256 MATCH = exec-in + already there)",
      gl["exec_in_bytes_equal"] >= 1 and gl["sha_match"] == gl["exec_in_bytes_equal"] + gl["already_on_phone"],
      [gl["sha_match"], gl["exec_in_bytes_equal"], gl["already_on_phone"]], "sha_match = exec_in + already")
skipped_tags = {re.match(r"(?:SKIP|DEFER) (\S+)", l).group(1).rstrip(":") for l in S1["skipped"] if re.match(r"(?:SKIP|DEFER) \S+", l)}
tags = plan_tags(plan)
check("s1 every plan leg run or skipped with a reason", set(tags) == set(S1["legs"]) | skipped_tags,
      sorted(set(tags) - set(S1["legs"]) - skipped_tags), [])
check("s1 the fake phone clean", not any(S1["phone_after"].values()), S1["phone_after"], "all false")
G1 = S1["gpu"]
gl_ok = bool(G1["legs"]) and all(v["gpu_lines"] >= 1 and v["failed_reads"] == 0 for v in G1["legs"].values())
mali_ok = gpu_kind != "mali" or all(v["g3d_lines"] >= 1 and v["gpu_cooling_lines"] >= 1 for v in G1["legs"].values())
check(f"s1 the GPU reads of a {gpu_kind} phone (sampler line, no failed read, state gpu: line, ready gate level)",
      gl_ok and mali_ok and G1["s_gpu"].find(f": {gpu_kind} (") > 0 and G1["state_files"] > 0
      and G1["state_files_with_gpu_line"] == G1["state_files"] and G1["ready_lines"] > 0 and G1["ready_lines_with_level"] == G1["ready_lines"],
      {"legs": len(G1["legs"]), "legs_ok": sum(v["gpu_lines"] >= 1 and v["failed_reads"] == 0 for v in G1["legs"].values()),
       "mali_slow_ok": mali_ok, "s_gpu": G1["s_gpu"], "state": [G1["state_files_with_gpu_line"], G1["state_files"]],
       "ready": [G1["ready_lines_with_level"], G1["ready_lines"]]},
      "every leg ok, every state file and ready line")
C1, T1 = S1["cooling"], S1["temperatures"]
WANT_SRC = {"kgsl": "sysfs", "mali": "hal"}[gpu_kind]
check("s1 the thermal service's raw dump and the sysfs cooling devices kept after the hold (s_thermal_raw.txt: the thermal "
      "status, the HAL's cooling devices on the Mali fake and none on the kgsl fake; s_cooling_sysfs.txt: every sysfs device)",
      C1["raw_thermal_status"] and (C1["raw_devices"] > 0) == (gpu_kind == "mali") and C1["sysfs_devices"] > 0
      and C1["s_cooling_sysfs_lines"] == C1["sysfs_devices"],
      [C1["raw_thermal_status"], C1["raw_devices"], C1["raw_types"], C1["s_cooling_sysfs_lines"], C1["sysfs_devices"]],
      "Thermal Status, HAL devices on mali only, every sysfs device listed")
check(f"s1 s_gpu.txt names the cooling source ({WANT_SRC} on the {gpu_kind} fake: "
      + ("the HAL lists devices, the sysfs ones are left alone)" if gpu_kind == "mali" else "the HAL lists none)"),
      C1["want_source"] == WANT_SRC and C1["s_gpu_cooling"] == WANT_SRC, [C1["want_source"], C1["s_gpu_cooling"]], [WANT_SRC, WANT_SRC])
check("s1 the cooling devices from that source (state files' cooling: = the HAL's mType 2 / 3 or the kept sysfs devices "
      "with cur/max; slow samples: a GPU device in every leg, nothing else)",
      C1["want_state"] not in ("", "n/a") and C1["state_files"] > 0 and C1["state_files_equal"] == C1["state_files"] and bool(C1["legs"])
      and all(v["gpu_lines"] >= 1 and v["other_lines"] == 0 for v in C1["legs"].values()),
      {"want_state": C1["want_state"], "state": [C1["state_files_equal"], C1["state_files"]],
       "legs_ok": sum(v["gpu_lines"] >= 1 and v["other_lines"] == 0 for v in C1["legs"].values()), "legs": len(C1["legs"])},
      "every state file equal, every leg ok")
check("s1 every temperature from the HAL's current section, not the cached one (state skin: and hal_temps:, slow samples' "
      "skin" + (", the state's G3D= and the slow samples' G3D)" if gpu_kind == "mali" else ")"),
      T1["skin_current"] and T1["skin_cached"] and T1["skin_current"] != T1["skin_cached"] and T1["state_files"] > 0
      and T1["state_skin_current"] == T1["state_hal_temps_current"] == T1["state_files"]
      and T1["slow_skin_lines"] > 0 and T1["slow_skin_current"] == T1["slow_skin_lines"]
      and (gpu_kind != "mali" or (T1["g3d_current"] and T1["g3d_cached"] and T1["g3d_current"] != T1["g3d_cached"]
                                  and T1["state_g3d_current"] == T1["state_files"] and T1["slow_g3d_lines"] > 0
                                  and T1["slow_g3d_current"] == T1["slow_g3d_lines"])),
      {k: T1[k] for k in ("skin_current", "skin_cached", "g3d_current", "g3d_cached", "state_files", "state_skin_current",
                          "state_hal_temps_current", "state_g3d_current", "slow_skin_lines", "slow_skin_current",
                          "slow_g3d_lines", "slow_g3d_current")},
      "the current values in every state file and slow sample")
# the fake exec-out returns cat's error text for a missing file, as the real one does: a leg without a report (the stuck
# app) must leave no non-JSON <tag>.json behind (run.sh keeps the text aside as <tag>.report_pull.txt)
not_json = [p.name for p in sorted(s1.glob("*.json")) if p.stem in set(tags) and load(p) is None]
S1["report_pulls_kept_aside"] = sorted(p.name for p in s1.glob("*.report_pull.txt"))
check("s1 every leg's report is JSON (a pull of a missing report is kept aside)", not not_json,
      {"not_json": not_json, "kept_aside": S1["report_pulls_kept_aside"]}, {"not_json": []})
S2, S3 = res["s2"], res["s3"]
check("s2 hard stop in the lost leg", (S2["chain_stopped"] or "").startswith(f"device_lost_in_{lost_tag} "), S2["chain_stopped"], f"device_lost_in_{lost_tag} <time>")
check("s2 two holds taken and released", S2["holds_acquired"] == S2["holds_released"] == 2, [S2["holds_acquired"], S2["holds_released"]], [2, 2])
check("s2 no fake adb call outside a hold", not S2["calls_outside_hold"], S2["calls_outside_hold"][:5], [])
check("s2 the fake phone clean after the cleanup", not any(S2["phone_after_cleanup"].values()), S2["phone_after_cleanup"], "all false")
check("s3 no leg: another job on the phone", (S3["chain_stopped"] or "").startswith("other_session_job ") and not S3["legs"],
      [S3["chain_stopped"], S3["legs"]], "other_session_job <time>, no leg")
check("s3 two holds taken and released", S3["holds_acquired"] == S3["holds_released"] == 2, [S3["holds_acquired"], S3["holds_released"]], [2, 2])
check("s3 only the other session's setup calls outside a hold", S3["setup_calls"] > 0 and S3["setup_calls_outside_hold"] == S3["setup_calls"] and not S3["calls_outside_hold"],
      [S3["setup_calls"], S3["setup_calls_outside_hold"], S3["calls_outside_hold"][:5]], "setup = outside, rest inside")
check("s3 the fake phone clean after the cleanup", not any(S3["phone_after_cleanup"].values()), S3["phone_after_cleanup"], "all false")
check("the fake hold free at the end", res["hold_file_after"] is None, res["hold_file_after"], None)
check("the line: only the keeper-less entry left", [e.get("script") for e in (res["queue_after"] or [])] == ["other-lane-no-keeper"],
      [e.get("script") for e in (res["queue_after"] or [])], ["other-lane-no-keeper"])

if ref_path:
    ref = json.loads(Path(ref_path).read_text())
    R1 = ref["s1"]
    res["reference"] = ref_path

    def same(name, got, want):
        check(f"= reference: {name}", got == want, got, want)

    same("s1 leg statuses", S1["legs"], R1["legs"])
    same("s1 skips", S1["skipped"], R1["skipped"])
    same("s1 verdicts", {k: v.split()[0] for k, v in S1["verdicts"].items()}, {k: v.split()[0] for k, v in R1["verdicts"].items()})
    same("s1 chain log line counts", S1["log_lines"], R1["log_lines"])
    same("s1 graph swap line counts", S1["graph_lines"], R1["graph_lines"])
    for k in sorted(set(S1["host_checks"]) | set(R1["host_checks"])):
        same(f"s1 host scorer {k}", S1["host_checks"].get(k), R1["host_checks"].get(k))
    for k in sorted(set(S1["timing"]) | set(R1["timing"])):
        g, w = S1["timing"].get(k), R1["timing"].get(k)
        if g and w and (g.get("stopped_early") or w.get("stopped_early")):
            # a STOP at the limit: the number of calls before the stop depends on the machine's speed
            g = {"status": g["status"], "stopped_early": g["stopped_early"], "sets": sorted(g["sets"])}
            w = {"status": w["status"], "stopped_early": w["stopped_early"], "sets": sorted(w["sets"])}
        same(f"s1 timing {k}", g, w)
    same("s1 stand-in layout = host build_inputs", S1["layout_equals_host"], R1["layout_equals_host"])
    same("s1 phone after", S1["phone_after"], R1["phone_after"])
    for name in ("s2", "s3"):
        G, W = res[name], ref[name]
        kind = lambda x: (x or "").rsplit(" ", 2)[0]
        same(f"{name} chain stop", kind(G["chain_stopped"]), kind(W["chain_stopped"]))
        for key in ("holds_acquired", "holds_released", "setup_calls", "setup_calls_outside_hold", "calls_outside_hold", "legs", "phone_after_cleanup"):
            same(f"{name} {key}", G[key], W[key])
    same("fake hold after", res["hold_file_after"], ref["hold_file_after"])
    same("the line after", [e.get("script") for e in (res["queue_after"] or [])], [e.get("script") for e in (ref["queue_after"] or [])])
    res["info_vs_reference"] = {"fake_adb_calls": {k: [res[k]["fake_adb_calls"], ref[k]["fake_adb_calls"]] for k in ("s1", "s2", "s3")},
                                "wall_s": {k: [res[k]["wall_s"], ref[k]["wall_s"]] for k in ("s1", "s2", "s3")}}

res["checks"] = checks
res["checks_ok"] = sum(c["ok"] for c in checks)
res["checks_total"] = len(checks)
res["all_ok"] = all(c["ok"] for c in checks)
(out / "summary.json").write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps({k: res[k] for k in ("lane", "rc", "real_adb_calls", "checks_ok", "checks_total", "all_ok")}, indent=1))
for c in checks:
    if not c["ok"]:
        print("NG", c["name"], "got", json.dumps(c["got"])[:300], "want", json.dumps(c["want"])[:300])
sys.exit(0 if res["all_ok"] else 1)
PY
