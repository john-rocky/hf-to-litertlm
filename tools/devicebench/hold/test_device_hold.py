#!/usr/bin/env python3
"""Regression test for the owned device hold (device_hold.py), on a throwaway hold file.

Each case is one way a shared hold fails: a driver taking a phone that another
process already holds, or a driver removing a hold it did not place.

    python3 test_device_hold.py
"""
import contextlib
import io
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import device_hold  # noqa: E402

DEFAULT_HOLD = device_hold.HOLD
fails = []


def check(name, cond):
    print(f"{'ok  ' if cond else 'FAIL'} {name}")
    if not cond:
        fails.append(name)


tmp = tempfile.mkdtemp()
device_hold.HOLD = os.path.join(tmp, ".device_hold")

# A pid that is certainly not running: spawn a process and reap it.
dead = subprocess.Popen([sys.executable, "-c", "pass"])
dead.wait()
DEAD_PID = dead.pid

# 1. Free device -> acquire writes an owned hold.
device_hold.acquire("test_driver.py")
cur = device_hold.read()
check("acquire on a free device records our pid", cur.get("pid") == os.getpid())
check("acquire records the script name", cur.get("script") == "test_driver.py")

# 2. Re-acquiring our own hold is not an error (a retry must not deadlock).
device_hold.acquire("test_driver.py")
check("re-acquiring our own hold is allowed", device_hold.read().get("pid") == os.getpid())

# 3. We own it -> release removes it.
device_hold.release()
check("release removes a hold we own", device_hold.read() is None)

# 4. A LIVE foreign holder must stop us.
with open(device_hold.HOLD, "w") as fh:
    fh.write('{"pid": %d, "script": "foreign_driver.py", "started": "2026-08-27 18:19:22"}'
             % os.getppid())
try:
    device_hold.acquire("second_driver.py")
    check("a live foreign hold blocks acquire", False)
except SystemExit as exc:
    check("a live foreign hold blocks acquire", "foreign_driver.py" in str(exc))

# 5. ...and release must NOT remove it.
device_hold.release()
check("release leaves a foreign hold in place",
      (device_hold.read() or {}).get("script") == "foreign_driver.py")

# 6. A dead holder's stale hold must not block the device forever.
with open(device_hold.HOLD, "w") as fh:
    fh.write('{"pid": %d, "script": "killed_driver.py", "started": "x"}' % DEAD_PID)
device_hold.acquire("test_driver.py")
check("a stale hold from a dead pid is taken over",
      device_hold.read().get("pid") == os.getpid())
device_hold.release()

# 6b. A shell driver passes its own pid and must be able to re-acquire its own
#     hold: acquire compares the stored pid with the caller's pid, not with the
#     pid of the python process that writes the file.
CALLER = os.getppid()
device_hold.acquire("driver.sh", pid=CALLER)
check("acquire records the caller's pid, not python's",
      device_hold.read().get("pid") == CALLER)
try:
    device_hold.acquire("driver.sh", pid=CALLER)
    check("the same caller may re-acquire its own hold", True)
except SystemExit:
    check("the same caller may re-acquire its own hold", False)
device_hold.release(pid=CALLER)
check("the caller's release removes its own hold", device_hold.read() is None)

# 7. A legacy zero-byte hold is not ours to remove.
open(device_hold.HOLD, "w").close()
check("a zero-byte hold reads as unowned, not absent", device_hold.read() == {})
device_hold.release()
check("release leaves an unowned zero-byte hold alone", device_hold.read() == {})

# 7b. acquire over it says what it does (it cannot ask the unowned hold's process) and then owns the phone.
said = io.StringIO()
with contextlib.redirect_stdout(said):
    device_hold.acquire("test_driver.py")
check("acquire over an unowned hold warns that it writes an owned hold over it",
      "writing an owned hold over it" in said.getvalue())
check("acquire over an unowned hold then owns it", device_hold.read().get("pid") == os.getpid())
device_hold.release()

# 8. A LIVE hold owned by another UID. `os.kill(pid, 0)` raises EPERM there, not
#    ESRCH — the process exists and is simply not ours. An implementation that
#    catches plain OSError reads that as "gone", clears the hold and takes the
#    phone. pid 1 (root) is the cheapest probe; processes of the same user never
#    raise EPERM, so only this case catches that implementation.
if os.getuid() != 0:
    import json
    with open(device_hold.HOLD, "w") as fh:
        json.dump({"pid": 1, "script": "root_owned_driver.py", "started": "x"}, fh)
    refused = False
    try:
        device_hold.acquire("test_driver.py")
    except SystemExit:
        refused = True
    check("a live hold owned by another uid blocks acquire (EPERM != gone)", refused)
    device_hold.release()
    cur = device_hold.read()
    check("release leaves another uid's live hold in place",
          bool(cur) and cur.get("pid") == 1)
else:
    print("skip  EPERM case (running as root — kill(1,0) would succeed)")

if os.path.exists(device_hold.HOLD):
    os.remove(device_hold.HOLD)

# 9. The label written into the hold names the phone of the file that arbitrates it, so a person reading the file
#    sees which phone it is: `.device_hold.<name>` (hold_path's name for another phone) -> <name>, the first phone's
#    `.device_hold` -> "phone", other names from the file's own name.
check("the default hold file is .device_hold next to device_hold.py",
      DEFAULT_HOLD == os.path.join(os.path.dirname(os.path.abspath(device_hold.__file__)), ".device_hold"))
for base, want in ((".device_hold", "phone"), (".device_hold.pixel8a", "pixel8a"), (".pixel8a_hold", "pixel8a"),
                   (".hold_pixel8a", "pixel8a"), (".bench_lock", "bench_lock")):
    check(f"the label of {base} is {want!r}", device_hold._device_label(os.path.join(tmp, base)) == want)
device_hold.acquire("test_driver.py")
check("a hold on the first phone's file is labelled 'phone'", device_hold.read().get("device") == "phone")
device_hold.release()
device_hold.acquire("test_driver.py", device="pixel8a")
check("a hold on hold_path('pixel8a') is labelled 'pixel8a'",
      (device_hold.read(device="pixel8a") or {}).get("device") == "pixel8a")
device_hold.release(device="pixel8a")
check("release removes that phone's hold", device_hold.read(device="pixel8a") is None)

os.rmdir(tmp)

print()
if fails:
    print(f"{len(fails)} FAILED: {fails}")
    sys.exit(1)
print("all device-hold cases pass")
