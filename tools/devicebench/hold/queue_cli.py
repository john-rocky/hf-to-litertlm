#!/usr/bin/env python3
"""A named waiting line for a shared phone, next to the owned hold (device_hold.py / hold_cli.py).

The hold file says who has the phone; the line says who is next, so every waiting keeper follows one rule instead of
racing for the phone at each release.

    python3 queue_cli.py enqueue <lockfile> <script-name> <contact> <minutes>     # append myself (idempotent by script name)
    python3 queue_cli.py list    <lockfile>                                          # the line, head first, plus the current holder
    python3 queue_cli.py dequeue <lockfile> <script-name>                            # remove myself (done, or giving up)
    python3 queue_cli.py front   <lockfile> <script-name> [--before <other-script>]  # reorder: move to the head (or before X)
    python3 queue_cli.py wait    <lockfile> <script-name> <owner-pid> [--timeout S] [--poll S]
        # block until (a) no entry ahead of mine has a keeper waiting and (b) the hold is free, then acquire the hold
        # with device_hold and dequeue myself; prints the acquire line; exit 0. Exit 3 on timeout (still queued), 2 on usage.
The queue file is <lockfile>.queue (JSON list, one object per entry: script, contact (who to ask about the entry),
minutes, enqueued, waiter). Entries whose script name matches the live hold are skipped as "running". No entry is ever
removed by another user of the line except through `front` (a visible reorder), so a stale entry needs its owner (or a
person) to `dequeue` it.

An entry is a place in the line, not a claim on the phone: `enqueue` writes it with no keeper (`waiter: null`), `wait`
writes its own pid there while it polls and clears it on timeout. An entry whose keeper is not waiting (enqueued before
its run is ready, or its `wait` died) does not stop the entries behind it; it keeps its place for when its keeper
starts. An entry without a `waiter` key (written by an older version of this tool) is treated as waiting until it
leaves the line. An entry without a `contact` key (written by another tool sharing the line) is listed with `?`.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import device_hold  # noqa: E402


def qpath(lock):
    return os.path.abspath(lock) + ".queue"


def load(lock):
    p = qpath(lock)
    if not os.path.exists(p):
        return []
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return []


def save(lock, q):
    p = qpath(lock)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(q, f, ensure_ascii=False, indent=0)
    os.replace(tmp, p)


def holder(lock):
    try:
        with open(os.path.abspath(lock), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def enqueue(lock, script, contact, minutes):
    q = load(lock)
    if any(e["script"] == script for e in q):
        return q
    q.append({"script": script, "contact": contact, "minutes": int(minutes),
              "enqueued": time.strftime("%Y-%m-%d %H:%M:%S"), "waiter": None})
    save(lock, q)
    return q


def keeper_waiting(e):
    """True when a `wait` process is polling for this entry. No `waiter` key = written by the previous version: waiting."""
    if "waiter" not in e:
        return True
    return e["waiter"] is not None and device_hold._alive(e["waiter"])


def mark(lock, script, waiter):
    """Write who is polling for `script` (a pid, or None); appends the entry when it is not queued. Saves only on change."""
    q = load(lock)
    mine = [e for e in q if e["script"] == script]
    if not mine:
        q.append({"script": script, "contact": "?", "minutes": 0,
                  "enqueued": time.strftime("%Y-%m-%d %H:%M:%S"), "waiter": waiter})
    elif mine[0].get("waiter", "absent") == waiter:
        return q
    else:
        mine[0]["waiter"] = waiter
    save(lock, q)
    return q


def dequeue(lock, script):
    q = [e for e in load(lock) if e["script"] != script]
    save(lock, q)
    return q


def front(lock, script, before=None):
    q = load(lock)
    me = [e for e in q if e["script"] == script]
    rest = [e for e in q if e["script"] != script]
    if not me:
        raise SystemExit(f"{script} is not queued")
    if before:
        idx = next((i for i, e in enumerate(rest) if e["script"] == before), 0)
    else:
        idx = 0
    q = rest[:idx] + me + rest[idx:]
    save(lock, q)
    return q


def fmt(lock):
    h = holder(lock)
    lines = ["holder: " + (f"{h.get('script')} (pid {h.get('pid')}, since {h.get('started')})" if h else "none")]
    for i, e in enumerate(load(lock), 1):
        running = " (running = the holder)" if h and h.get("script") == e["script"] else ""
        keeper = "" if running or "waiter" not in e else (" (keeper waiting)" if keeper_waiting(e) else " (no keeper waiting)")
        lines.append(f"{i}. {e['script']}  {e.get('contact', '?')}  ~{e['minutes']} min  queued {e['enqueued']}{running}{keeper}")
    return "\n".join(lines)


def wait(lock, script, pid, timeout, poll):
    device_hold.HOLD = os.path.abspath(lock)
    t0 = time.time()
    me = os.getpid()
    try:
        while True:
            q = mark(lock, script, me)  # (re)write my pid: a concurrent writer may have saved an older copy of the line
            h = holder(lock)
            waiting = [e for e in q if not (h and h.get("script") == e["script"])]
            ahead = []
            for e in waiting:
                if e["script"] == script:
                    break
                ahead.append(e)
            if h is None and not any(keeper_waiting(e) for e in ahead):
                try:
                    device_hold.acquire(script, pid=pid)
                    dequeue(lock, script)
                    print(f"hold acquired: {device_hold.HOLD} (owner pid {pid}) at {time.strftime('%H:%M:%S')}")
                    return 0
                except SystemExit:
                    pass  # someone took it in the gap; keep waiting
            if time.time() - t0 > timeout:
                print(f"timeout after {int(time.time() - t0)} s; still queued", file=sys.stderr)
                return 3
            time.sleep(poll)
    finally:
        # leaving without the hold (timeout, Ctrl-C): my entry stays in the line with no keeper
        if any(e["script"] == script for e in load(lock)):
            mark(lock, script, None)


def main(argv):
    if len(argv) < 3:
        raise SystemExit(__doc__)
    cmd, lock = argv[1], argv[2]
    if cmd == "enqueue":
        enqueue(lock, argv[3], argv[4], argv[5]); print(fmt(lock)); return 0
    if cmd == "list":
        print(fmt(lock)); return 0
    if cmd == "dequeue":
        dequeue(lock, argv[3]); print(fmt(lock)); return 0
    if cmd == "front":
        before = argv[argv.index("--before") + 1] if "--before" in argv else None
        front(lock, argv[3], before); print(fmt(lock)); return 0
    if cmd == "wait":
        script, pid = argv[3], argv[4]
        timeout = float(argv[argv.index("--timeout") + 1]) if "--timeout" in argv else 4 * 3600
        poll = float(argv[argv.index("--poll") + 1]) if "--poll" in argv else 2.0
        return wait(lock, script, pid, timeout, poll)
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
