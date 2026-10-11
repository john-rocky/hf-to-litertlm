#!/bin/bash
# dryrun_npu.sh: the NPU chain's dry run on the Mac against a fake phone and the lane's stand-in runner: no phone, no
# real adb, no real hold. Three scenes, each on a fresh fake phone with its own hold file and line, all under
# cache/dryrun_npu/<LANE>/ (removed and rewritten by every run); the summary and its checks to summary.json there; exit
# 0 only when every check is ok.
#   bash dryrun_npu.sh <lane.env>
#   normal        NPU_DRY_PLAN as it is: every leg DONE with its dumps, every adb call inside the hold, the stage removed;
#                 the runner's stdout and stderr in two files (stderr.txt pulled with the phone's byte count, LiteRT's log
#                 lines only there, the runner's summary at a line start of stdout.txt and in the scorer's results),
#                 delegate.txt opening with the runner's own delegation lines, the thermal service's raw dump and the
#                 sysfs cooling devices kept, every temperature from the HAL's current section and the cooling devices
#                 from sysfs (the fake phone's HAL lists none, as a Galaxy S26's)
#   compile_hang  every NPU compile hangs (FAKE_ROOT/_fault/compile_hang, which the stand-in honours): the compile watch
#                 kills the runner (COMPILE_TIMEOUT; its stderr.txt is still pulled), the GPU legs run, a leg that needs a
#                 timed-out leg is skipped
#   usb_lost      the phone leaves adb once the first leg's stdout has a line matching NPU_DRY_LOST_LINE: exit 9, the hold
#                 released; the phone comes back; `npu_chain.sh cleanup` takes a hold and removes the stage
# lane.env gives: NPU_RUNNER_NAME, NPU_DRY_ROUND (the round whose fixtures, manifests and runtime libraries the plan
# uses), NPU_DRY_PLAN, NPU_STANDIN (the stand-in runner file: the chain stages it under NPU_RUNNER_NAME; it honours
# FAKE_ROOT, FAKE_COMPILE_S, FAKE_SLOW_PARITY and _fault/compile_hang), NPU_DRY_LOST_LINE and, optionally,
# NPU_DRY_REFERENCE (an earlier summary of the three scenes: its leg statuses must come back) and
# NPU_DRY_REFERENCE_NORMAL (an earlier normal scene: leg statuses, precisions, dumps and max |dp| must come back).
# Guards: the `adb` tripwire first on PATH, ADB = fake_adb.py, every hold file, round dir and results dir under cache/.
set -u
BENCH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LANE_ENV_ARG=${1:?usage: dryrun_npu.sh <lane.env>}
export DRY_RUN=1
# shellcheck source=lib.sh
. "$BENCH/lib.sh"
ROUND=${NPU_DRY_ROUND:-dryrun}
bench_load "$LANE_ENV_ARG"
for v in NPU_RUNNER_NAME NPU_DRY_ROUND NPU_DRY_PLAN NPU_STANDIN NPU_DRY_LOST_LINE; do
  [ -n "${!v:-}" ] || { echo "$LANE_ENV: $v is not set (dryrun_npu.sh needs it)" >&2; exit 2; }
done
ROUND=$NPU_DRY_ROUND
OUT=$BENCH/cache/dryrun_npu/$LANE
CHAIN=$BENCH/npu_chain.sh
rm -rf "$OUT"; mkdir -p "$OUT/tripwire"
cat > "$OUT/tripwire/adb" <<EOF
#!/bin/bash
echo "\$(date '+%F %T') real adb called: \$*" >> "$OUT/real_adb_calls.log"
exit 97
EOF
chmod +x "$OUT/tripwire/adb"; : > "$OUT/real_adb_calls.log"; : > "$OUT/rc.txt"
export PATH="$OUT/tripwire:$PATH"
[ "$(command -v adb)" = "$OUT/tripwire/adb" ] || { echo "the tripwire is not the first adb on PATH"; exit 2; }
export ADB=$BENCH/fake_adb.py NPU_RUNNER=$NPU_STANDIN PLAN=$NPU_DRY_PLAN FAKE_RUNNER_NAME=$NPU_RUNNER_NAME
export REST_MAX=6 REST_MAX_T=6 POLL=2 FAKE_COMPILE_S=${FAKE_COMPILE_S:-3}
FIRST_TAG=$(grep -v '^[[:space:]]*#' "$NPU_DRY_PLAN" | awk 'NF {print $1; exit}')
echo "npu dry run $(date '+%F %T'); lane $LANE ($LANE_ENV); round $ROUND; plan $NPU_DRY_PLAN; adb on PATH = $(command -v adb)"

scene_env() {  # scene: the scene's fake phone, round dir, results dir and hold file
  local S=$OUT/$1
  mkdir -p "$S/root/_fault" "$S/r" "$S/results"
  export FAKE_ROOT=$S/root ROUND_DIR=$S/r RESULTS_DIR=$S/results HOLD=$S/fake_hold SCRIPT_NAME=$LANE-npu-dryrun-$1
}

step() {  # label log command...: one step with its output to <log>, its exit code to rc.txt
  local label=$1 log=$2 rc
  shift 2
  "$@" > "$log" 2>&1; rc=$?
  echo "$label $rc" | tee -a "$OUT/rc.txt"
}

scene_env normal
step normal_run "$OUT/normal/chain.log" bash "$CHAIN" run "$ROUND" "$LANE_ENV"

scene_env compile_hang
: > "$FAKE_ROOT/_fault/compile_hang"
step compile_hang_run "$OUT/compile_hang/chain.log" bash "$CHAIN" run "$ROUND" "$LANE_ENV"

scene_env usb_lost
( t=0; until grep -qE "$NPU_DRY_LOST_LINE" "$ROUND_DIR/$FIRST_TAG.stdout.txt" 2>/dev/null || [ $t -ge 4500 ]; do sleep 0.2; t=$((t + 1)); done
  grep -qE "$NPU_DRY_LOST_LINE" "$ROUND_DIR/$FIRST_TAG.stdout.txt" 2>/dev/null || exit 0
  : > "$FAKE_ROOT/_fault/gone"; date '+%F %T' > "$OUT/usb_lost/gone_at" ) &
WATCH=$!
export FAKE_SLOW_PARITY=1
step usb_lost_run "$OUT/usb_lost/chain.log" bash "$CHAIN" run "$ROUND" "$LANE_ENV"
unset FAKE_SLOW_PARITY
kill "$WATCH" 2>/dev/null; wait "$WATCH" 2>/dev/null
rm -f "$FAKE_ROOT/_fault/gone"
step usb_lost_cleanup "$OUT/usb_lost/cleanup.log" bash "$CHAIN" cleanup "$ROUND" "$LANE_ENV"

python3 -I - "$OUT" "$NPU_DRY_PLAN" "$NPU_STAGE" "$ROUND" "${NPU_DRY_REFERENCE:-}" "${NPU_DRY_REFERENCE_NORMAL:-}" "$LANE" "$LANE_ENV" <<'PY'
import json, re, sys, time
from pathlib import Path
out, plan, stage, rnd, ref_all, ref_normal, lane, lane_env = (Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3].lstrip("/"),
                                                              sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8])
TS = r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)"
WANT_RC = {"normal_run": 0, "compile_hang_run": 0, "usb_lost_run": 9, "usb_lost_cleanup": 0}


def read(p):
    return p.read_text() if p.exists() else ""


def plan_legs(p):
    legs = []
    for line in read(p).splitlines():
        line = line.split("#", 1)[0].split()
        if line:
            legs.append({"tag": line[0], "accel": line[1], "count": int(line[5]), "rounds": int(line[7]), "cache": line[9],
                         "extras": line[12:]})
    return legs


LOG_LINE = re.compile(r"(?:INFO|VERBOSE|WARNING|ERROR): ")


def streams(scene):
    """The runner's two files per leg: stderr.txt pulled with the phone's byte count, LiteRT's log lines in it and none
    in stdout.txt, the runner's summary line at a line start of stdout.txt, the delegation lines at the head of
    delegate.txt (from the runner's own files) and the lane scorer's runner summary next to the median of the calls."""
    r, got = out / scene / "r", {}
    for p in sorted(r.glob("*.leg.json")):
        j = json.loads(p.read_text())
        tag = j["tag"]
        so, se, st = read(r / f"{tag}.stdout.txt"), read(r / f"{tag}.stderr.txt"), j.get("stderr") or {}
        own = [l for l in read(r / f"{tag}.delegate.txt").splitlines() if l.startswith((f"{tag}.stdout.txt:", f"{tag}.stderr.txt:"))]
        lp = None
        for q in sorted((out / scene / "results").glob(f"*{tag}*.json")):
            try:
                d = json.loads(q.read_text())
            except ValueError:
                continue
            if isinstance(d, dict) and isinstance(d.get("latency_phase"), dict) and q.stem.endswith(f"{tag}_{rnd}"):
                lp = d["latency_phase"]
        got[tag] = {"status": j["status"], "stderr_bytes": [st.get("bytes_phone"), st.get("bytes_pulled"),
                                                            (r / f"{tag}.stderr.txt").stat().st_size if (r / f"{tag}.stderr.txt").exists() else None],
                    "log_lines_stderr": sum(bool(LOG_LINE.match(l)) for l in se.splitlines()),
                    "log_lines_stdout": sum(bool(LOG_LINE.match(l)) for l in so.splitlines()),
                    "summary_at_line_start": len(re.findall(r"(?m)^summary model\[0\] ", so)),
                    "own_delegation_lines": len(own), "own_replacing": sum("Replacing" in l for l in own),
                    "own_partitioned": sum("Partitioned subgraph" in l for l in own),
                    "runner_summary_median": ((lp or {}).get("runner_summary") or {}).get("median"),
                    "calls_median": ((lp or {}).get("per_call_ms_write_run_read") or {}).get("median")}
    return got


def calls(scene):
    log = out / scene / "root/adb_calls.log"
    cs = [float(l.split()[0]) for l in read(log).splitlines()]
    kl = read(out / scene / "r/keeper.log")
    t = lambda m: time.mktime(time.strptime(m, "%Y-%m-%d %H:%M:%S"))
    acq = [t(m) for m in re.findall(r"HOLD ACQUIRED by pid \d+ at " + TS, kl)]
    rel = [t(m) for m in re.findall(r"HOLD RELEASED at " + TS, kl)]
    inside = sum(1 for c in cs if any(a - 1 <= c <= b + 1 for a, b in zip(acq, rel)))
    return {"adb_calls": len(cs), "inside_hold": inside, "holds": [len(acq), len(rel)]}


def max_dp(scene, tag):
    for p in sorted((out / scene / "results").glob(f"*{tag}*.json")):
        try:
            d = json.loads(p.read_text())
        except ValueError:
            continue
        s = d.get("summary") if isinstance(d, dict) else None
        if isinstance(s, dict) and "max_abs_dp" in s and p.stem.endswith(f"{tag}_{rnd}"):
            return s["max_abs_dp"]
    return None


legs_planned = plan_legs(plan)
res = {"lane": lane, "lane_env": lane_env, "plan": str(plan), "round": rnd,
       "rc": {l.split()[0]: int(l.split()[1]) for l in read(out / "rc.txt").splitlines() if len(l.split()) == 2},
       "real_adb_calls": len(read(out / "real_adb_calls.log").splitlines()), "scenes": {}}
for scene in ("normal", "compile_hang", "usb_lost"):
    r = out / scene / "r"
    legs = {}
    for p in sorted(r.glob("*.leg.json")):
        j = json.loads(p.read_text())
        legs[j["tag"]] = {"status": j["status"], "precision": j["precision"], "dumps": j["dumps"], "bad": j["dumps_bad_size"],
                          "killed_by": j["killed_by"], "rc": j["rc"]}
    res["scenes"][scene] = {"legs": legs, "calls": calls(scene), "max_abs_dp": {t: max_dp(scene, t) for t in legs},
                            "skipped": read(r / "s_skipped.txt").splitlines(),
                            "chain_stopped": read(r / "CHAIN_STOPPED").strip() or None,
                            "stage_left": (out / scene / "root" / stage).exists(),
                            "hold_left": read(out / scene / "fake_hold").strip() or None}
checks = []


def check(name, ok, got=None, want=None):
    checks.append({"name": name, "ok": bool(ok), "got": got, "want": want})


for k, want in WANT_RC.items():
    check(f"exit code {k}", res["rc"].get(k) == want, res["rc"].get(k), want)
check("no real adb call", res["real_adb_calls"] == 0, res["real_adb_calls"], 0)
for scene in ("normal", "compile_hang", "usb_lost"):
    S = res["scenes"][scene]
    c = S["calls"]
    check(f"{scene}: every fake adb call inside a hold", c["adb_calls"] > 0 and c["adb_calls"] == c["inside_hold"], c, "all inside")
    check(f"{scene}: the stage removed, the hold free", not S["stage_left"] and S["hold_left"] is None,
          [S["stage_left"], S["hold_left"]], [False, None])
N, H, U = (res["scenes"][s] for s in ("normal", "compile_hang", "usb_lost"))
check("normal: every plan leg DONE with its dumps (count) and none of a wrong size",
      sorted(N["legs"]) == sorted(l["tag"] for l in legs_planned)
      and all(N["legs"][l["tag"]]["status"] == "DONE" and N["legs"][l["tag"]]["dumps"] == l["count"] and N["legs"][l["tag"]]["bad"] == 0
              for l in legs_planned if l["tag"] in N["legs"]),
      {t: [v["status"], v["dumps"], v["bad"]] for t, v in N["legs"].items()}, "DONE, dumps = count, bad 0")
check("normal: every leg scored (a max |dp| in the results)", all(v is not None for v in N["max_abs_dp"].values()) and N["max_abs_dp"],
      N["max_abs_dp"], "a number per leg")
npu_tags = [l["tag"] for l in legs_planned if l["accel"].startswith("npu")]
needs_npu = [l["tag"] for l in legs_planned if any(x.startswith("needs=") and x.split("=", 1)[1] in npu_tags for x in l["extras"])]
gpu_tags = [l["tag"] for l in legs_planned if l["accel"] == "gpu" and l["tag"] not in needs_npu]
check("compile_hang: every NPU leg COMPILE_TIMEOUT", npu_tags and all(H["legs"].get(t, {}).get("status") == "COMPILE_TIMEOUT"
      for t in npu_tags if t not in needs_npu), {t: H["legs"].get(t, {}).get("status") for t in npu_tags}, "COMPILE_TIMEOUT")
check("compile_hang: every GPU leg DONE", all(H["legs"].get(t, {}).get("status") == "DONE" for t in gpu_tags),
      {t: H["legs"].get(t, {}).get("status") for t in gpu_tags}, "DONE")
check("compile_hang: a leg that needs a timed-out leg skipped", all(t not in H["legs"] and any(t in x for x in H["skipped"]) for t in needs_npu),
      needs_npu, "skipped, with a reason")
NS, HS = streams("normal"), streams("compile_hang")
res["scenes"]["normal"]["streams"], res["scenes"]["compile_hang"]["streams"] = NS, HS
rounds_of = {l["tag"]: l["rounds"] for l in legs_planned}
check("normal: the runner's stdout and stderr apart (stderr.txt pulled with the phone's bytes, LiteRT's log lines only "
      "there, a latency leg's summary at a line start of stdout.txt)",
      NS and all(v["stderr_bytes"][0] is not None and v["stderr_bytes"][0] == v["stderr_bytes"][1] == v["stderr_bytes"][2]
                 and v["log_lines_stderr"] > 0 and v["log_lines_stdout"] == 0
                 and v["summary_at_line_start"] == (1 if rounds_of.get(t, 0) > 0 else 0) for t, v in NS.items()),
      {t: [v["stderr_bytes"], v["log_lines_stderr"], v["log_lines_stdout"], v["summary_at_line_start"]] for t, v in NS.items()},
      "bytes equal, log lines > 0 in stderr and 0 in stdout, summary 1 per latency leg")
check("normal: the runner's own summary in the scorer's results (median = the calls' median within 0.001 ms)",
      all(v["runner_summary_median"] is not None and v["calls_median"] is not None
          and abs(v["runner_summary_median"] - v["calls_median"]) <= 0.001 for t, v in NS.items() if rounds_of.get(t, 0) > 0)
      and any(rounds_of.get(t, 0) > 0 for t in NS),
      {t: [v["runner_summary_median"], v["calls_median"]] for t, v in NS.items() if rounds_of.get(t, 0) > 0}, "equal, not null")
cold_npu = {l["tag"] for l in legs_planned if l["accel"].startswith("npu") and l["cache"].startswith("new:")}
check("normal: delegate.txt opens with the runner's own delegation lines (Replacing; a cold NPU leg also Partitioned)",
      NS and all(v["own_replacing"] >= 1 and (t not in cold_npu or v["own_partitioned"] >= 1) for t, v in NS.items()),
      {t: [v["own_replacing"], v["own_partitioned"]] for t, v in NS.items()}, "Replacing >= 1, Partitioned >= 1 on a cold NPU leg")
killed = {t: v for t, v in HS.items() if v["status"] == "COMPILE_TIMEOUT"}
check("compile_hang: a killed runner's stderr.txt pulled too (the phone's bytes)",
      killed and all(v["stderr_bytes"][0] is not None and v["stderr_bytes"][0] == v["stderr_bytes"][1] == v["stderr_bytes"][2]
                     and v["log_lines_stderr"] > 0 for v in killed.values()),
      {t: [v["stderr_bytes"], v["log_lines_stderr"]] for t, v in killed.items()}, "bytes equal, log lines > 0")
NR = out / "normal" / "r"
raw = read(NR / "s_thermal_raw.txt")
sysfs = sorted((out / "normal" / "root/sys/class/thermal").glob("cooling_device*"))
sysfs_lines = len(re.findall(r"(?m)^cooling_device\d+ \S+ \d+/\d+$", read(NR / "s_cooling_sysfs.txt")))
check("normal: the thermal service's raw dump and the sysfs cooling devices kept after the hold (s_thermal_raw.txt: the "
      "thermal status, the current temperatures, no HAL cooling device; s_cooling_sysfs.txt: every sysfs device)",
      bool(re.search(r"(?m)^Thermal Status: \d+$", raw)) and "Current temperatures from HAL:" in raw and "CoolingDevice{" not in raw
      and sysfs and sysfs_lines == len(sysfs),
      [bool(re.search(r"(?m)^Thermal Status: \d+$", raw)), raw.count("CoolingDevice{"), sysfs_lines, len(sysfs)],
      "Thermal Status, 0 CoolingDevice, every sysfs device listed")


def section(text, start, end):
    out_, on = [], False
    for l in text.splitlines():
        if on and l.startswith(end):
            break
        if on:
            out_.append(l)
        on = on or l.startswith(start)
    return out_


SKIN_RE = r"mValue=[-0-9.]+, mType=3, mName=[A-Za-z0-9_-]*SKIN"
skin_cur = next((m.group(0) for l in section(raw, "Current temperatures from HAL", "Current cooling devices from HAL")
                 for m in [re.search(SKIN_RE, l)] if m), None)
skin_cached = next((m.group(0) for l in section(raw, "Cached temperatures", "HAL Ready") for m in [re.search(SKIN_RE, l)] if m), None)
KEPT = re.compile(r"(cpufreq-[A-Za-z0-9_.-]*|cpu-cluster[A-Za-z0-9_.-]*|gpu|kgsl|cdsp[A-Za-z0-9_.-]*|ddr-cdev)")
kept = [[read(d / f).strip() for f in ("type", "cur_state", "max_state")] for d in sysfs]
kept = [x for x in kept if KEPT.fullmatch(x[0])]
want_state = "".join(f"{t}={c}/{m};" for t, c, m in kept)
want_slow = {f"cooling {t} {c}/{m}" for t, c, m in kept}
states = [NR / "s0_state_initial.txt"] + sorted(NR.glob("*.state_before.txt")) + sorted(NR.glob("*.state_after.txt"))
tc = {"skin_current": skin_cur, "skin_cached": skin_cached, "want_state": want_state,
      "s_gpu_cooling": (re.search(r"(?m)^cooling: (\S+)", read(NR / "s_gpu.txt")) or [None, None])[1],
      "state_files": sum(p.exists() for p in states),
      "state_skin_current": sum(bool(re.search(rf"(?m)^skin: {re.escape(skin_cur or '-')}$", read(p))) for p in states),
      "state_cooling_sysfs": sum(bool(re.search(rf"(?m)^cooling: {re.escape(want_state)}$", read(p))) for p in states), "legs": {}}
for p in sorted(NR.glob("*.samples.txt")):
    ls = read(p).splitlines()
    cl = [l for l in ls if l.startswith("cooling ")]
    sk = [l for l in ls if re.fullmatch(SKIN_RE, l)]
    tc["legs"][p.name[: -len(".samples.txt")]] = {"skin": len(sk), "skin_current": sum(l == skin_cur for l in sk), "cooling": len(cl),
                                                   "cooling_gpu": sum(bool(re.fullmatch(r"cooling (gpu|kgsl) \d+/\d+", l)) for l in cl),
                                                   "cooling_other": sum(l not in want_slow for l in cl)}
res["scenes"]["normal"]["temperatures_cooling"] = tc
check("normal: every temperature from the HAL's current section and the cooling devices from sysfs (s_gpu.txt cooling: "
      "sysfs; state skin: = current, not cached; state cooling: = the kept sysfs devices; slow samples: the current skin, "
      "a GPU device, nothing else)",
      skin_cur and skin_cached and skin_cur != skin_cached and want_state and tc["s_gpu_cooling"] == "sysfs"
      and tc["state_files"] > 0 and tc["state_skin_current"] == tc["state_cooling_sysfs"] == tc["state_files"] and tc["legs"]
      and all(v["skin"] > 0 and v["skin_current"] == v["skin"] and v["cooling_gpu"] >= 1 and v["cooling_other"] == 0
              for v in tc["legs"].values()),
      tc, "sysfs, every state file and leg current / sysfs")
check("usb_lost: hard stop, no leg record", (U["chain_stopped"] or "").startswith("device_lost") and not U["legs"],
      [U["chain_stopped"], list(U["legs"])], "device_lost..., no leg")
if ref_all:
    ref = json.loads(Path(ref_all).read_text())
    res["reference"] = ref_all
    for scene in ("normal", "compile_hang", "usb_lost"):
        W = (ref.get("scenes") or {}).get(scene) or {}
        G = res["scenes"][scene]
        common = sorted(set(W.get("legs", {})) & set(G["legs"]))
        check(f"= reference: {scene} statuses of the reference's legs", all(G["legs"][t]["status"] == W["legs"][t]["status"] for t in common)
              and set(W.get("legs", {})) <= set(G["legs"]),
              {t: G["legs"].get(t, {}).get("status") for t in W.get("legs", {})}, {t: v["status"] for t, v in W.get("legs", {}).items()})
        if "chain_rc" in W:
            key = {"normal": "normal_run", "compile_hang": "compile_hang_run", "usb_lost": "usb_lost_run"}[scene]
            check(f"= reference: {scene} chain exit code", res["rc"].get(key) == W["chain_rc"], res["rc"].get(key), W["chain_rc"])
if ref_normal:
    ref = json.loads(Path(ref_normal).read_text())
    res["reference_normal"] = ref_normal
    got = {t: {"status": v["status"], "precision": v["precision"], "dumps": v["dumps"]} for t, v in N["legs"].items()}
    check("= reference: normal legs (status, precision, dumps)", got == ref.get("legs"), got, ref.get("legs"))
    check("= reference: normal max |dp| per leg", N["max_abs_dp"] == ref.get("max_abs_dp"), N["max_abs_dp"], ref.get("max_abs_dp"))
res["checks"] = checks
res["checks_ok"] = sum(c["ok"] for c in checks)
res["checks_total"] = len(checks)
res["all_ok"] = all(c["ok"] for c in checks)
(out / "summary.json").write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps({k: res[k] for k in ("lane", "round", "rc", "real_adb_calls", "checks_ok", "checks_total", "all_ok")}, indent=1))
for c in checks:
    if not c["ok"]:
        print("NG", c["name"], "got", json.dumps(c["got"])[:300], "want", json.dumps(c["want"])[:300])
sys.exit(0 if res["all_ok"] else 1)
PY
