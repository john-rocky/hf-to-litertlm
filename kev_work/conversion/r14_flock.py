"""Run one command while holding an exclusive flock(2) on a lock file (macOS has no flock(1) command): the heavy CPU jobs
of this work (export, the 402-question PyTorch checks, CPU parity, quantization) ran one at a time through it, and the
Mac timing scripts take the same lock (r14_timing_mac.py imports oldest_ticket).

    python3 r14_flock.py --lock W/.b_heavy.lock --label "export L512" -- <cmd> ...

Python's fcntl.flock is flock(2), the same lock flock(1) would take, so every process that locks this file with
flock(2) excludes the others. Blocking wait (polls with LOCK_NB, so the wait is logged), then one owner line "<label> pid
<pid> since <time>" is written into the lock file (information only: the flock is the lock), the command runs as a
child, its return code is returned, and the line is cleared before the flock is released. TERM / INT are passed to the
child. Waits and holds are appended to logs/r14_heavy_waits.log (one line each).

GPU timing windows: no job starts while the GPU lock file named by the environment variable KEV_GPU_LOCK (default
W/gpu.lock; a missing file reads as free) holds an owner line containing "timing" (another job is timing the GPU and a
heavy CPU job beside it would slow it down). --gpu-timing wait (the default) waits for that before taking the heavy
lock and checks again after taking it (a window that opened in between -> release, wait again); the wait is logged
(WAIT-TIMING lines). --gpu-timing ignore is for a holder that must take the lock during a window.

Arrival order: every waiter puts a ticket <lock>.queue/<arrival time ns, 20 digits>-<pid> (W/.b_heavy.queue for the
default lock) and tries the flock only while its ticket is the oldest one (tickets of dead pids are swept by whoever
reads the queue); the ticket is removed once the flock is held, and on exit. A process that locks the file without a
ticket still competes at each release. The GPU timing-window wait happens while holding the ticket, so the place in the
queue is kept.

--guard NAME starts a memory-watch script on the child (swap_guard.sh for labels with "export", r14_job_guard.sh
otherwise); those scripts are not part of this folder, so leave --guard out. Exit 3 = the guard killed the child."""
import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

K = Path(__file__).resolve().parents[1]
LOG = K / "logs/r14_heavy_waits.log"
GPU_LOCK = Path(os.environ.get("KEV_GPU_LOCK", str(K / "gpu.lock")))


def gpu_timing_owner():
    """The owner line of another job's GPU timing window, or None."""
    try:
        t = GPU_LOCK.read_text().strip()
    except OSError:
        return None
    return t if "timing" in t else None


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def oldest_ticket(qdir):
    """The oldest live ticket name in qdir (dead pids' tickets are removed on the way), or None."""
    for t in sorted(qdir.iterdir()):
        try:
            pid = int(t.name.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        if alive(pid):
            return t.name
        try:
            t.unlink()
            log(f"QUEUE swept the ticket of dead pid {pid} ({t.name})")
        except FileNotFoundError:
            pass
    return None


def stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def log(line):
    with open(LOG, "a") as f:
        f.write(f"{stamp()} {line}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", default=str(K / ".b_heavy.lock"))
    ap.add_argument("--label", required=True)
    ap.add_argument("--gpu-timing", choices=["auto", "wait", "ignore"], default="wait",
                    help="wait (default) = do not start while the GPU lock names a timing window; ignore = start anyway; "
                         "auto = wait only for a label with 'export'")
    ap.add_argument("--guard", default="", help="swap_guard.sh on the child once it runs (name for its log)")
    ap.add_argument("--guard-log", default="", help="guard log path (default logs/guard_r14_<guard>.log)")
    ap.add_argument("--guard-kind", choices=["auto", "export", "job"], default="auto",
                    help="export = scripts/swap_guard.sh (system swap alone kills); job = scripts/r14_job_guard.sh (kills "
                         "only when the job's own footprint grew too); auto = export iff the label says export")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    assert cmd, "no command"
    wait_timing = a.gpu_timing == "wait" or (a.gpu_timing == "auto" and "export" in a.label.lower())
    fh = open(a.lock, "a+")
    qdir = Path(a.lock).with_suffix(".queue")          # K/.b_heavy.lock -> K/.b_heavy.queue
    qdir.mkdir(exist_ok=True)
    ticket = qdir / f"{time.time_ns():020d}-{os.getpid()}"
    ticket.touch()
    signal.signal(signal.SIGTERM, lambda *_: (ticket.unlink(missing_ok=True), sys.exit(143)))
    signal.signal(signal.SIGINT, lambda *_: (ticket.unlink(missing_ok=True), sys.exit(130)))
    t0, holder, twait, queued = time.time(), None, 0.0, False
    while True:
        head = oldest_ticket(qdir)
        if head != ticket.name:
            if not queued:
                log(f"QUEUE {a.label} pid {os.getpid()}: ticket {ticket.name}, behind {head}")
                queued = True
            time.sleep(2)
            continue
        if wait_timing:
            tw0, seen = time.time(), None
            while (owner := gpu_timing_owner()) is not None:
                if seen != owner:
                    log(f"WAIT-TIMING {a.label} pid {os.getpid()}: GPU timing window ({owner}); no heavy job starts in it")
                    seen = owner
                time.sleep(10)
            twait += time.time() - tw0
            if time.time() - tw0 > 1:        # the window ended: re-read the queue before trying
                continue
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if holder is None:
                try:
                    holder = Path(a.lock).read_text().strip() or "<no owner line>"
                except OSError:
                    holder = "<unreadable>"
                log(f"WAIT {a.label} pid {os.getpid()}: lock held ({holder})")
            time.sleep(2)
            continue
        if wait_timing and gpu_timing_owner() is not None:      # a window opened while the lock was taken
            fcntl.flock(fh, fcntl.LOCK_UN)
            continue
        break
    ticket.unlink(missing_ok=True)
    waited = round(time.time() - t0, 1)
    if wait_timing and twait:
        log(f"TIMING-WAIT-TOTAL {a.label} pid {os.getpid()}: {round(twait, 1)}s waited for GPU timing windows")
    fh.seek(0)
    fh.truncate()
    fh.write(f"{a.label} pid {os.getpid()} since {stamp()}\n")
    fh.flush()
    log(f"HOLD {a.label} pid {os.getpid()} waited {waited}s: {' '.join(cmd)[:300]}")
    proc = subprocess.Popen(cmd)
    guard = None
    if a.guard:
        glog = a.guard_log or str(K / f"logs/guard_r14_{a.guard}.log")
        kind = a.guard_kind if a.guard_kind != "auto" else ("export" if "export" in a.label.lower() else "job")
        gscript = "swap_guard.sh" if kind == "export" else "r14_job_guard.sh"
        log(f"GUARD {a.label} pid {os.getpid()}: {gscript} on child {proc.pid} ({glog})")
        guard = subprocess.Popen(["zsh", str(K / "scripts" / gscript), glog, a.guard, str(proc.pid)])

    def fwd(sig, _frame):
        try:
            proc.send_signal(sig)
        except ProcessLookupError:
            pass

    signal.signal(signal.SIGTERM, fwd)
    signal.signal(signal.SIGINT, fwd)
    t1 = time.time()
    rc = proc.wait()
    fh.seek(0)
    fh.truncate()
    fh.flush()
    fcntl.flock(fh, fcntl.LOCK_UN)
    fh.close()
    log(f"FREE {a.label} pid {os.getpid()} rc {rc} held {round(time.time() - t1, 1)}s")
    if guard is not None:
        try:
            grc = guard.wait(timeout=60)
        except subprocess.TimeoutExpired:
            guard.terminate()
            grc = guard.wait()
        if grc == 3:
            log(f"GUARD-KILLED {a.label} pid {os.getpid()}: swap_guard {a.guard} killed the child (child rc {rc})")
            sys.exit(3)
    sys.exit(rc if rc >= 0 else 128 - rc)


if __name__ == "__main__":
    main()
