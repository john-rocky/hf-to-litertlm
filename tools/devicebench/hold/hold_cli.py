#!/usr/bin/env python3
"""Bash-callable front end for the owned device hold (device_hold.py), for any phone: one hold file per phone.

    python3 hold_cli.py acquire <lockfile> <script-name> <owner-pid>   # exits 1 if held
    python3 hold_cli.py release <lockfile> <owner-pid>
    python3 hold_cli.py read    <lockfile>

A shell driver passes its own pid as the owner (the python process that writes the file exits at once). A pid in a
file says which process holds the phone; a check by process name cannot: it either matches every shell that merely
mentions the name or, once those are filtered out, misses a driver that itself runs as `sh -c`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import device_hold  # noqa: E402


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    action, path = sys.argv[1], os.path.abspath(sys.argv[2])
    device_hold.HOLD = path
    if action == "acquire":
        script = sys.argv[3] if len(sys.argv) > 3 else "unknown"
        owner = sys.argv[4] if len(sys.argv) > 4 else None
        try:
            device_hold.acquire(script, pid=owner)
        except SystemExit as exc:
            print(exc, file=sys.stderr)
            return 1
        print(f"hold acquired: {path} (owner pid {owner or os.getpid()})")
        return 0
    if action == "release":
        device_hold.release(pid=sys.argv[3] if len(sys.argv) > 3 else None)
        return 0
    if action == "read":
        print(device_hold.read())
        return 0
    raise SystemExit(f"unknown action {action}")


if __name__ == "__main__":
    sys.exit(main())
