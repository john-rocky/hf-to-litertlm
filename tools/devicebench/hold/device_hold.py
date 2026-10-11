#!/usr/bin/env python3
"""An owned hold on a shared phone: a small JSON file that names the process holding the phone.

Every driver that uses the phone takes the hold before its first adb call and gives it back at the end:

    import device_hold
    device_hold.acquire("my_driver.py")
    try:
        ...
    finally:
        device_hold.release()

The file records the owner's pid, the driver's name, the phone's label and the start time. `acquire` refuses while a
live process other than the caller holds the phone and takes over a hold whose pid is gone. `release` removes the file
only when the caller owns it, so a driver that exits first never removes another driver's hold.

Why a pid rather than an empty flag file or a check by process name: a flag without an owner is removed by whichever
driver exits first, and a check by script name misses every driver that is not on its list. Two drivers on one phone
at once do not fail; they produce a slow, plausible number.

An empty (legacy) hold file cannot be attributed to a process: `acquire` warns and writes its own hold over it, and
`release` leaves it in place, since it is not the caller's.
"""
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
HOLD = os.path.join(HERE, ".device_hold")


def hold_path(device=None):
    """One hold file per phone: HOLD for the first phone, HOLD + "." + <device> for another, so a run on one phone never
    waits for a run on another."""
    return HOLD if not device else HOLD + "." + str(device)


def _device_label(path):
    """The phone's label written into the hold, read from the hold file's name: `.device_hold.<name>` (hold_path's name
    for a phone other than the first), `.<name>_hold` or `.hold_<name>` -> <name>; `.device_hold` (the first phone's)
    -> "phone"; any other name -> the file's name without its leading dot.

    The label comes from the file's own name (not from the module-level HOLD, which hold_cli.py sets to the same path
    before every call), so the readable field always names the file that arbitrates the phone.
    """
    base = os.path.basename(path)
    name = base[1:] if base.startswith(".") else base
    if name == "device_hold":
        return "phone"
    if name.startswith("device_hold."):
        name = name[len("device_hold."):]
    elif name.endswith("_hold"):
        name = name[: -len("_hold")]
    elif name.startswith("hold_"):
        name = name[len("hold_"):]
    return name or "phone"


def _alive(pid):
    """True if the pid exists. Signal 0 only probes; it never delivers.

    Only `ProcessLookupError` (ESRCH) means gone. `PermissionError` (EPERM) means the process is alive and owned by
    another user, so it counts as alive: treating every OSError as gone would read another user's live process as dead,
    clear its hold and take the phone. Processes of the same user never raise EPERM, so only a test against another
    user's pid (pid 1) shows the difference.
    """
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (ValueError, TypeError, OSError):
        return False
    return True


def read(device=None):
    """-> dict for an owned hold, {} for a legacy empty one, None for no hold."""
    path = hold_path(device)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            text = fh.read().strip()
    except OSError:
        return {}
    if not text:
        return {}
    try:
        data = json.loads(text)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def acquire(script, force=False, pid=None, device=None):
    """Take the device hold, or raise SystemExit naming who has it.

    `force` is for a hold left behind by a driver that was killed hard enough to
    skip its `finally` — verify the pid is really gone before reaching for it.

    `pid` records an owner other than this process. A shell script that shells
    out to acquire must pass its OWN pid: the python that wrote the file exits
    immediately, and a hold owned by a dead pid reads as stale to the very next
    acquirer, which is the same as having no hold at all.
    """
    me = int(pid) if pid else os.getpid()
    path = hold_path(device)
    cur = read(device)
    if cur is None:
        pass
    elif not cur:
        print("  device hold: an unowned (legacy/zero-byte) hold is present — "
              "cannot attribute it to a process; writing an owned hold over it",
              flush=True)
    elif cur.get("pid") == me:
        pass
    elif _alive(cur.get("pid")) and not force:
        raise SystemExit(
            f"device held by pid {cur.get('pid')} ({cur.get('script')}) since "
            f"{cur.get('started')} — wait for it, or coordinate with that "
            f"session. Re-run with force only if that pid is provably gone.")
    else:
        print(f"  device hold: clearing stale hold from pid {cur.get('pid')} "
              f"({cur.get('script')})", flush=True)

    with open(path, "w") as fh:
        # the label is read from the hold file's own name (_device_label), so the field a person reads before deciding
        # that a phone is free always names the phone this file arbitrates
        json.dump({"pid": me, "script": script, "device": _device_label(path),
                   "started": time.strftime("%Y-%m-%d %H:%M:%S")}, fh)
    return path


def release(pid=None, device=None):
    """Remove the hold only if we own it. Never strip another session's.

    Anything we cannot positively identify as ours is left alone — including an
    unowned zero-byte hold, which is what someone else's driver writes. Leaving a
    stale file behind costs one warning on the next `acquire`; removing a live
    hold costs a measurement.
    """
    cur = read(device)
    if cur is None:
        return
    if cur.get("pid") != (int(pid) if pid else os.getpid()):
        print(f"  device hold: not ours ({cur.get('pid') or 'unowned'}) — "
              f"leaving it in place", flush=True)
        return
    try:
        os.remove(hold_path(device))
        print("device hold removed", flush=True)
    except FileNotFoundError:
        pass
