#!/usr/bin/env python3
"""A stand-in for `adb` that keeps a fake phone under $FAKE_ROOT, so chain.sh -> run.sh runs end to end on the Mac
without a phone (dryrun.sh drives it). The app's process is the lane's stand-in ($FAKE_STANDIN).

    FAKE_ROOT=<dir> FAKE_PKG=<package> FAKE_STANDIN='<python> <stand-in.py>' ADB=<this file> DRY_RUN=1 \
        bash chain.sh run <round> <lane.env>          (lib.sh exports FAKE_PKG / FAKE_ACTIVITY / FAKE_SERIAL /
                                                       FAKE_STANDIN / FAKE_RUNNER_NAME from lane.env when DRY_RUN=1)

Env: FAKE_ROOT (required), FAKE_PKG (the app's package), FAKE_STANDIN (the command line of the app's stand-in: it gets
--files-dir <the app's files dir> --fake-root <FAKE_ROOT>, one --<key> <value> per am extra, and --lowmem for the lowmem
fault), FAKE_ACTIVITY (default .GateActivity), FAKE_SERIAL (in the "not found" error), FAKE_RUNNER_NAME (a native
runner's process name, for npu_chain.sh: the runner is a file the chain pushes and starts with `adb shell`; on the fake
phone that file is the lane's stand-in runner, which writes $FAKE_ROOT/_run/<name>.pid while it lives), FAKE_GPU (kgsl,
the default: an Adreno phone in a Galaxy S26's form: /sys/class/kgsl, a dumpsys thermalservice with stale cached
temperatures (SKIN 37.9) before the HAL's current ones (SKIN 31.5) and an empty cooling-device list, and cooling devices
under /sys/class/thermal/cooling_device*/{type,cur_state,max_state}, kept types among others; mali: an Arm Mali phone in
a Pixel 8a's form: /sys/class/misc/mali0/device (cur_freq, total_gpu_mem, dma_buf_gpu_mem) and no kgsl, a dumpsys
thermalservice with stale cached temperatures (G3D 53.0, VIRTUAL-SKIN 30.2) before the current ones (G3D 32.6,
VIRTUAL-SKIN 30.4) and the Pixel 8a's nine cooling devices, and a sysfs cooling device of a kept type that the reads
must leave alone while the HAL lists devices).
Supported: -s SERIAL get-state | install -r <apk> | uninstall <pkg> | push <src...> <dst> | push <dir>/. <dst>/ (a file
over 64 MiB is hard-linked when it can be, else copied) | pull <src> <dst> | pull <dir>/. <dst>/ | shell [-n] <cmd...> |
exec-in <cmd...> (stdin -> the command) | exec-out <cmd...> (binary stdout, with the command's stderr in the same stream
as the real client) | logcat [-b all] -d -T <time> [--pid=N]. Like the real client, `shell` without -n takes whatever
its stdin holds (a `while read ...; do adb shell ...; done < list` loop loses its list here as it does on a phone). A
shell command runs in bash with the device paths (/data/local/tmp, /data/data, /data/app, /sys/class/kgsl,
/sys/class/misc/mali0, /sys/class/thermal, /sys/devices/system/cpu/cpufreq, /proc) mapped under $FAKE_ROOT and shims
(_shims/) for what the phone has and the Mac does not: am (start -W -n <pkg>/<activity> --es/--ei extras -> the stand-in
as a detached process = the app's process; force-stop), pm list packages, pidof, run-as (only for an installed package;
cwd = its data dir), dumpsys (thermalservice, battery, power, activity), getprop, toybox / sha256sum (also on stdin),
stat -c %s, df -k / -h, ps -A -o, top, settings get, svc, input (no-ops). One phone state: an installed app, its files/,
the stage dir, a runner pid. Faults for the failure paths (files under $FAKE_ROOT/_fault/): lowmem_tag (the leg whose
report is <tag>.json gets a stand-in stuck in its compile and MemAvailable 1,000,000 kB until it is force-stopped), gone
(get-state says the phone is not there), thermal (the thermal status), gpu_cooling (the state of a Mali phone's
thermal-gpufreq-0 cooling device), data_free_kb (df's free space). Every call is appended to $FAKE_ROOT/adb_calls.log as
"<epoch seconds> <argv>"; this script never runs the real adb.
"""
import os
import select
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ["FAKE_ROOT"]).resolve()
SHIMS = ROOT / "_shims"
PKG = os.environ.get("FAKE_PKG", "")
RUNNER_NAME = os.environ.get("FAKE_RUNNER_NAME", "")
ACTIVITY = os.environ.get("FAKE_ACTIVITY") or ".GateActivity"
SERIAL = os.environ.get("FAKE_SERIAL") or "fake-serial"
STANDIN = shlex.split(os.environ.get("FAKE_STANDIN", ""))
GPU = os.environ.get("FAKE_GPU") or "kgsl"
# /sys/class/thermal/cooling_device<n>: (n, type, cur_state, max_state). The kgsl phone's (a Galaxy S26's HAL lists no
# cooling device; its kernel's devices are read here): CPU (cpufreq-*, cpu-cluster*), GPU (gpu, kgsl), DSP (cdsp*) and
# memory bus (ddr-cdev) devices are kept, the others (cpu-pause-1, ufs, display-fps) left out; numbered so that the
# phone's glob order (0, 1, 12, 2, 23, ...) differs from the numeric one. The mali phone's: one device of a kept type
# (gpu, state 3) that must not reach the reads while the HAL lists cooling devices.
SYSFS_COOLING = {
    "kgsl": [(0, "cpufreq-cpu0", 0, 27), (1, "cpufreq-cpu6", 0, 30), (2, "cpu-cluster0", 0, 27), (12, "cpu-cluster1", 0, 30),
             (23, "cpu-pause-1", 0, 1), (31, "gpu", 0, 18), (40, "kgsl", 0, 18), (41, "cdsp", 1, 12), (55, "cdsp_hw", 0, 4),
             (7, "ddr-cdev", 0, 3), (8, "ufs", 0, 3), (9, "display-fps", 2, 5)],
    "mali": [(0, "thermal-cpufreq-0", 0, 10), (3, "gpu", 3, 5)],
}
MEMINFO_OK = "MemTotal:       11389756 kB\nMemFree:          800000 kB\nMemAvailable:    6200000 kB\nSwapFree:        4000000 kB\n"
MEMINFO_LOW = "MemTotal:       11389756 kB\nMemFree:          200000 kB\nMemAvailable:    1000000 kB\nSwapFree:        1000000 kB\n"


def dev(p: str) -> Path:
    return ROOT / p.lstrip("/")


def setup_root():
    if (ROOT / ".ready").exists():
        return
    for pol, mx in (("policy0", 3628800), ("policy6", 4742400)):
        d = dev(f"/sys/devices/system/cpu/cpufreq/{pol}")
        d.mkdir(parents=True, exist_ok=True)
        for f, v in (("scaling_max_freq", mx), ("cpuinfo_max_freq", mx), ("scaling_cur_freq", mx // 2)):
            (d / f).write_text(f"{v}\n")
    if GPU == "mali":
        g = dev("/sys/class/misc/mali0/device")
        g.mkdir(parents=True, exist_ok=True)
        for f, v in (("cur_freq", 150000), ("hint_min_freq", 150000), ("total_gpu_mem", 52428800), ("dma_buf_gpu_mem", 0)):
            (g / f).write_text(f"{v}\n")
    else:
        g = dev("/sys/class/kgsl/kgsl-3d0")
        g.mkdir(parents=True, exist_ok=True)
        for f, v in (("max_clock_mhz", 1300), ("thermal_pwrlevel", 0), ("clock_mhz", 1300), ("temp", 40000)):
            (g / f).write_text(f"{v}\n")
        dev("/sys/class/kgsl/kgsl").mkdir(parents=True, exist_ok=True)
        dev("/sys/class/kgsl/kgsl/page_alloc").write_text("220000000\n")
    for n, typ, cur, mx in SYSFS_COOLING.get(GPU, []):
        d = dev(f"/sys/class/thermal/cooling_device{n}")
        d.mkdir(parents=True, exist_ok=True)
        for f, v in (("type", typ), ("cur_state", cur), ("max_state", mx)):
            (d / f).write_text(f"{v}\n")
    dev("/proc").mkdir(parents=True, exist_ok=True)
    dev("/proc/meminfo").write_text(MEMINFO_OK)
    dev("/data/local/tmp").mkdir(parents=True, exist_ok=True)
    for d in ("_fault", "_logcat", "_run"):
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    (ROOT / ".boot").write_text(f"{time.time()}\n")
    SHIMS.mkdir(parents=True, exist_ok=True)
    adb = f"{sys.executable} -I {Path(__file__).resolve()}"
    shims = {
        # the device side of `am`: the fake adb itself handles start / force-stop (it owns the runner's pid file)
        "am": f'exec {adb} _am "$@"',
        "pm": f'exec {adb} _pm "$@"',
        "pidof": f'exec {adb} _pidof "$@"',
        "run-as": ('pkg=$1; shift; [ -f "$FAKE_ROOT/data/app/$pkg/base.apk" ] || { echo "run-as: unknown package: $pkg" >&2; '
                   'exit 1; }; mkdir -p "$FAKE_ROOT/data/data/$pkg"; cd "$FAKE_ROOT/data/data/$pkg" || exit 1; exec "$@"'),
        "dumpsys": f'exec {adb} _dumpsys "$@"',
        "getprop": f'exec {adb} _getprop "$@"',
        "toybox": 'a=$1; shift; exec "$a" "$@"',
        "sha256sum": ('if [ $# -eq 0 ]; then shasum -a 256 | sed "s/ .*/  -/"; else for f in "$@"; do if [ -f "$f" ]; then '
                      'shasum -a 256 "$f"; else echo "sha256sum: $f: No such file or directory" >&2; fi; done; fi'),
        "stat": 'if [ "$1" = "-c" ]; then shift 2; /usr/bin/stat -f %z "$@"; else /usr/bin/stat "$@"; fi',
        "df": ('free=$(cat "$FAKE_ROOT/_fault/data_free_kb" 2>/dev/null || echo 20000000); '
               'if [ "$1" = "-k" ]; then echo "Filesystem 1K-blocks Used Available Use% Mounted on"; '
               'echo "/dev/block/dm-88 231000000 $((231000000 - free)) $free 92% /data/user/0"; '
               'else echo "Filesystem Size Used Avail Use% Mounted on"; echo "/dev/block/dm-88 221G 202G $((free / 1000000))G 92% /data/user/0"; fi'),
        "ps": f'exec {adb} _ps "$@"',
        "top": 'echo "Tasks: 3 total (fake)"',
        "settings": 'echo 0',
        "svc": 'true',
        "input": 'true',
        # seconds since the fake boot (+100,000 s), so a fault that "reboots" the phone can be seen as a smaller uptime
        "uptime_fake": 'boot=$(cut -d. -f1 "$FAKE_ROOT/.boot"); echo "$(( $(date +%s) - boot + 100000 )).00 999999.99"',
    }
    for name, body in shims.items():
        p = SHIMS / name
        p.write_text("#!/bin/bash\n" + body + "\n")
        p.chmod(0o755)
    (ROOT / ".ready").write_text("ok\n")


def rewrite(cmd: str) -> str:
    for p in ("/data/local/tmp", "/data/data", "/data/app", "/sys/devices/system/cpu/cpufreq", "/sys/class/kgsl",
              "/sys/class/misc/mali0", "/sys/class/thermal"):
        cmd = cmd.replace(p, str(dev(p)))
    cmd = cmd.replace("cat /proc/uptime", "uptime_fake")
    cmd = cmd.replace("/proc/", str(dev("/proc")) + "/")
    return cmd


def env():
    return dict(os.environ, PATH=f"{SHIMS}:{os.environ['PATH']}", FAKE_ROOT=str(ROOT))


def run_device(cmd: str, stdin=None, stderr=None) -> int:
    """The phone's `sh -c <cmd>` (stdout passes through unchanged, so `exec-out ... cat` stays binary; stderr=STDOUT
    puts the command's stderr into the same stream, as the real exec-out does)."""
    return subprocess.run(["bash", "-c", rewrite(cmd)], env=env(), stdin=stdin, stderr=stderr).returncode


def take_stdin() -> None:
    """What the real client does with a `shell` call's stdin: it reads what is there (a file to its end, a pipe as far
    as it has data now) and the remote command never sees it. A terminal is left alone."""
    try:
        fd = sys.stdin.fileno()
        if os.isatty(fd):
            return
        while select.select([fd], [], [], 0)[0]:
            if not os.read(fd, 65536):
                return
    except (OSError, ValueError):
        return


def installed() -> bool:
    return (dev(f"/data/app/{PKG}") / "base.apk").exists()


def runner_pid(name=None):
    """The live pid of the app's stand-in (or of the native runner `name`), from its pid file."""
    p = ROOT / "_run" / f"{name or PKG}.pid"
    try:
        pid = int(p.read_text().strip())
        os.kill(pid, 0)
        return pid
    except (OSError, ValueError):
        return None


def force_stop():
    pid = runner_pid()
    if pid:
        try:
            os.killpg(pid, signal.SIGTERM)
        except OSError:
            pass
        for _ in range(20):
            if runner_pid() is None:
                break
            time.sleep(0.1)
        if runner_pid():
            try:
                os.killpg(pid, signal.SIGKILL)
            except OSError:
                pass
    dev("/proc/meminfo").write_text(MEMINFO_OK)


def am(args) -> int:
    if args[:1] == ["force-stop"]:
        force_stop()
        return 0
    if args[:1] != ["start"]:
        print(f"fake am: unsupported {args[:1]}", file=sys.stderr)
        return 1
    i, comp, extras = 1, None, {}
    while i < len(args):
        a = args[i]
        if a == "-W":
            i += 1
        elif a == "-n":
            comp = args[i + 1]
            i += 2
        elif a in ("--es", "--ei"):
            extras[args[i + 1]] = args[i + 2]
            i += 3
        else:
            print(f"fake am: unsupported argument {a}", file=sys.stderr)
            return 1
    if comp != f"{PKG}/{ACTIVITY}" or not installed():
        print(f"Error: Activity class {{{comp}}} does not exist.")
        return 1
    if not STANDIN:
        print("fake am: FAKE_STANDIN is not set (the app's stand-in)", file=sys.stderr)
        return 1
    force_stop()
    files = dev(f"/data/data/{PKG}/files")
    files.mkdir(parents=True, exist_ok=True)
    cmd = STANDIN + ["--files-dir", str(files), "--fake-root", str(ROOT)]
    for k, v in extras.items():
        cmd += [f"--{k}", v]
    lowmem = (ROOT / "_fault/lowmem_tag")
    if lowmem.exists() and extras.get("report") == lowmem.read_text().strip() + ".json":
        cmd.append("--lowmem")
        dev("/proc/meminfo").write_text(MEMINFO_LOW)
    (ROOT / "_run").mkdir(exist_ok=True)
    log = open(ROOT / "_run" / "runner_stdout.log", "a")
    proc = subprocess.Popen(cmd, stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True,
                            env=dict(os.environ, FAKE_ROOT=str(ROOT)))
    (ROOT / "_run" / f"{PKG}.pid").write_text(f"{proc.pid}\n")
    pdir = dev(f"/proc/{proc.pid}")
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "cgroup").write_text("5:cpuset:/top-app\n4:cpu:/top-app\n0::/uid_10345/pid_%d\n" % proc.pid)
    print(f"Starting: Intent {{ cmp={comp} (has extras) }}\nStatus: ok\nLaunchState: COLD\nActivity: {comp}\n"
          f"TotalTime: 5\nWaitTime: 6\nComplete")
    return 0


def pm(args) -> int:
    if args[:2] == ["list", "packages"]:
        if installed() and (len(args) < 3 or args[2] in PKG):
            print(f"package:{PKG}")
        return 0
    print(f"fake pm: unsupported {args}", file=sys.stderr)
    return 1


def pidof(args) -> int:
    name = args[0] if args else ""
    pid = runner_pid() if PKG and name == PKG else runner_pid(name) if RUNNER_NAME and name == RUNNER_NAME else None
    if pid:
        print(pid)
        return 0
    return 1


def getprop(args) -> int:
    print({"ro.product.model": "fake phone", "ro.build.version.release": "16",
           "ro.build.display.id": "fake"}.get(args[0] if args else "", ""))
    return 0


def dumpsys(args) -> int:
    what = args[0] if args else ""
    if what == "thermalservice":
        t = (ROOT / "_fault/thermal").read_text().strip() if (ROOT / "_fault/thermal").exists() else "0"
        if GPU == "mali":
            gc = (ROOT / "_fault/gpu_cooling").read_text().strip() if (ROOT / "_fault/gpu_cooling").exists() else "0"
            # the cached section holds old values (a recorded Pixel 8a dump: G3D 58.0 there, 31.0 current): every
            # temperature must come from the HAL's current temperatures
            print(f"IsStatusOverride: false\nThermal Status: {t}\nCached temperatures:\n"
                  "\tTemperature{mValue=74.0, mType=0, mName=BIG, mStatus=0}\n\tTemperature{mValue=53.0, mType=1, mName=G3D, mStatus=0}\n"
                  "\tTemperature{mValue=30.2, mType=3, mName=VIRTUAL-SKIN, mStatus=0}\nHAL Ready: true\n"
                  "HAL connection:\n\tThermalHAL AIDL 3  connected: yes\n"
                  "Current temperatures from HAL:\n"
                  "\tTemperature{mValue=33.1, mType=0, mName=BIG, mStatus=0}\n\tTemperature{mValue=32.4, mType=0, mName=MID, mStatus=0}\n"
                  "\tTemperature{mValue=31.9, mType=0, mName=LITTLE, mStatus=0}\n\tTemperature{mValue=32.6, mType=1, mName=G3D, mStatus=0}\n"
                  "\tTemperature{mValue=30.4, mType=3, mName=VIRTUAL-SKIN, mStatus=0}\n\tTemperature{mValue=29.0, mType=2, mName=battery, mStatus=0}\n"
                  # a Pixel 8a's list (Android 16) in its order: CPU (2) and GPU (3) devices among battery (1) and
                  # NPU / TPU (5) ones
                  "Current cooling devices from HAL:\n"
                  "\tCoolingDevice{mValue=0, mType=1, mName=dc_icl}\n\tCoolingDevice{mValue=0, mType=5, mName=gxp-cooling}\n"
                  "\tCoolingDevice{mValue=0, mType=5, mName=tpu_cooling}\n\tCoolingDevice{mValue=0, mType=1, mName=usbc-port}\n"
                  "\tCoolingDevice{mValue=0, mType=2, mName=thermal-cpufreq-1}\n\tCoolingDevice{mValue=0, mType=1, mName=fcc}\n"
                  f"\tCoolingDevice{{mValue={gc}, mType=3, mName=thermal-gpufreq-0}}\n"
                  "\tCoolingDevice{mValue=0, mType=2, mName=thermal-cpufreq-2}\n\tCoolingDevice{mValue=0, mType=2, mName=thermal-cpufreq-0}\n"
                  "Temperature static thresholds from HAL:")
            return 0
        # a Galaxy S26's form (Android 16, ThermalHAL AIDL 3): the cached temperatures are old (SKIN 37.9 there, 31.5
        # current), the HAL lists no cooling device (the reads go to /sys/class/thermal), SKIN's thresholds follow
        print(f"IsStatusOverride: false\nThermalEventListeners:\n\tcallbacks: 3\n\tkilled: false\n\tbroadcasts count: -1\n"
              f"Thermal Status: {t}\nCached temperatures:\n"
              "\tTemperature{mValue=48.3, mType=0, mName=AP, mStatus=0}\n\tTemperature{mValue=31.1, mType=2, mName=BAT, mStatus=0}\n"
              "\tTemperature{mValue=37.9, mType=3, mName=SKIN, mStatus=0}\nHAL Ready: true\n"
              "HAL connection:\n\tThermalHAL AIDL 3  connected: yes\n"
              "Current temperatures from HAL:\n"
              "\tTemperature{mValue=29.7, mType=0, mName=AP, mStatus=0}\n\tTemperature{mValue=29.0, mType=2, mName=BAT, mStatus=0}\n"
              "\tTemperature{mValue=31.5, mType=3, mName=SKIN, mStatus=0}\n\tTemperature{mValue=27.7, mType=4, mName=USB, mStatus=0}\n"
              "Current cooling devices from HAL:\n"
              "Temperature static thresholds from HAL:\n"
              "\tTemperatureThreshold{mType=3, mName=SKIN, mHotThrottlingThresholds=[38.0, 40.0, 42.0, 45.0, 47.0, 60.0, 90.0], "
              "mColdThrottlingThresholds=[NaN, NaN, NaN, NaN, NaN, NaN, NaN]}\n"
              "Temperature headroom thresholds:\n[NaN, 0.8333333, 0.9, 1.0, 1.0666667, 1.5, 2.5]")
    elif what == "battery":
        print("Current Battery Service state:\n  level: 80\n  temperature: 290")
    elif what == "power":
        print("POWER MANAGER (dumpsys power)\n  mWakefulness=Awake")
    elif what == "activity":
        top = f"{PKG}/{ACTIVITY}" if PKG and runner_pid() else "com.sec.android.app.launcher/.activities.LauncherActivity"
        print(f"  topResumedActivity=ActivityRecord{{1234 u0 {top} t1}}")
    return 0


def ps(args) -> int:
    """toybox ps -A -o <columns>: PID, ETIME / ELAPSED, TIME, NAME of a few fixed system processes (their TIME never
    grows) and of the app while its stand-in lives."""
    cols = ["PID", "TIME", "NAME"]
    if "-o" in args:
        cols = [c.strip().upper() for c in args[args.index("-o") + 1].split(",")]
    procs = [(1501, "4-02:11:09", "00:00:31", "system_server"), (2240, "4-02:10:51", "00:00:02", "com.sec.android.app.launcher"),
             (812, "4-02:11:20", "00:00:00", "installd")]
    pid = runner_pid() if PKG else None
    if pid:
        procs.append((pid, "00:05", "00:00:01", PKG))
    npid = runner_pid(RUNNER_NAME) if RUNNER_NAME else None
    if npid:
        procs.append((npid, "00:05", "00:00:01", RUNNER_NAME))
    print(" ".join("ELAPSED" if c == "ETIME" else c for c in cols))
    for p, et, t, name in procs:
        val = {"PID": str(p), "ETIME": et, "ELAPSED": et, "TIME": t, "NAME": name}
        print(" ".join(val.get(c, "?") for c in cols))
    return 0


def place(src: Path, target: Path):
    """A file onto the fake phone: a large file (a graph) as a hard link when it is on the same volume, else a copy."""
    if target.exists():
        target.unlink()
    if src.stat().st_size > (64 << 20):
        try:
            os.link(src, target)
            return
        except OSError:
            pass
    shutil.copy2(src, target)


def push(srcs, dst) -> int:
    if len(srcs) == 1 and srcs[0].endswith("/."):
        s, d = Path(srcs[0][:-2]), dev(dst.rstrip("/"))
        if not s.is_dir() or not d.is_dir():
            print(f"adb: error: cannot push '{srcs[0]}' to '{dst}'", file=sys.stderr)
            return 1
        n = 0
        for f in sorted(s.iterdir()):
            place(f, d / f.name)
            n += 1
        print(f"{srcs[0]}: {n} files pushed, 0 skipped. (fake)")
        return 0
    d = dev(dst)
    if len(srcs) > 1 and not d.is_dir():
        print(f"adb: error: target '{dst}' is not a directory", file=sys.stderr)
        return 1
    for s in srcs:
        sp = Path(s)
        target = d / sp.name if d.is_dir() else d
        if not target.parent.is_dir():
            print(f"adb: error: failed to copy '{s}' to '{dst}': remote No such file or directory", file=sys.stderr)
            return 1
        place(sp, target)
    print(f"{srcs[0]}: {len(srcs)} file pushed, 0 skipped. (fake)")
    return 0


def pull(src, dst) -> int:
    if src.endswith("/."):
        s, d = dev(src[:-2]), Path(dst.rstrip("/"))
        if not s.is_dir():
            print(f"adb: error: remote object '{src}' does not exist", file=sys.stderr)
            return 1
        d.mkdir(parents=True, exist_ok=True)
        for f in sorted(s.iterdir()):
            shutil.copy2(f, d / f.name)
        return 0
    s = dev(src)
    if not s.exists():
        print(f"adb: error: remote object '{src}' does not exist", file=sys.stderr)
        return 1
    shutil.copy2(s, dst)
    return 0


def install(args) -> int:
    apk = Path(args[-1])
    if not apk.is_file():
        print(f"adb: failed to stat {apk}: No such file or directory", file=sys.stderr)
        return 1
    d = dev(f"/data/app/{PKG}")
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy2(apk, d / "base.apk")
    dev(f"/data/data/{PKG}").mkdir(parents=True, exist_ok=True)
    print("Performing Streamed Install\nSuccess")
    return 0


def uninstall(args) -> int:
    if args[-1] != PKG or not installed():
        print("Failure [DELETE_FAILED_INTERNAL_ERROR]")
        return 1
    force_stop()
    shutil.rmtree(dev(f"/data/app/{PKG}"), ignore_errors=True)
    shutil.rmtree(dev(f"/data/data/{PKG}"), ignore_errors=True)
    print("Success")
    return 0


def logcat(args) -> int:
    pid, since = None, None
    for i, a in enumerate(args):
        if a.startswith("--pid="):
            pid = a.split("=", 1)[1]
        elif a == "-T" and i + 1 < len(args):
            since = args[i + 1]     # "MM-DD HH:MM:SS.mmm" = the first 18 characters of every fake line
    print("--------- beginning of main (fake)")
    for f in sorted((ROOT / "_logcat").glob("*.log")):
        if pid and f.stem != pid:
            continue
        for line in f.read_text().splitlines(keepends=True):
            if since is None or line[:18] >= since:
                sys.stdout.write(line)
    return 0


def main() -> int:
    a = sys.argv[1:]
    if a[:1] and a[0].startswith("_"):        # a shim calling back into this file
        setup_root()
        fn = {"_am": am, "_pm": pm, "_pidof": pidof, "_dumpsys": dumpsys, "_ps": ps, "_getprop": getprop}[a[0]]
        return fn(a[1:])
    setup_root()
    with open(ROOT / "adb_calls.log", "a") as f:
        f.write(f"{time.time():.3f} " + " ".join(a)[:400] + "\n")
    if a[:1] == ["-s"]:
        a = a[2:]
    if not a:
        return 1
    if a[0] == "get-state":
        if (ROOT / "_fault/gone").exists():
            print(f"error: device '{SERIAL}' not found", file=sys.stderr)
            return 1
        print("device")
        return 0
    if (ROOT / "_fault/gone").exists():
        print(f"error: device '{SERIAL}' not found", file=sys.stderr)
        return 1
    if a[0] == "install":
        return install(a[1:])
    if a[0] == "uninstall":
        return uninstall(a[1:])
    if a[0] == "push":
        return push(a[1:-1], a[-1])
    if a[0] == "pull":
        return pull(a[1], a[2])
    if a[0] == "shell":
        if a[1:2] == ["-n"]:
            return run_device(" ".join(a[2:]), stdin=subprocess.DEVNULL)
        take_stdin()
        return run_device(" ".join(a[1:]), stdin=subprocess.DEVNULL)
    if a[0] == "exec-out":
        # the real client's exec-out has no separate stderr: `exec-out cat <missing file>` returns cat's error text
        return run_device(" ".join(a[1:]), stderr=subprocess.STDOUT)
    if a[0] == "exec-in":
        return run_device(" ".join(a[1:]), stdin=sys.stdin)
    if a[0] == "logcat":
        return logcat(a[1:])
    print(f"fake adb: unsupported {a[0]}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
