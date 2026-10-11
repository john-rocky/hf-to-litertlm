#!/usr/bin/env python3
"""Regression test for queue_cli.py on a throwaway lock file (never the real .device_hold)."""
import json, os, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
CLI = os.path.join(HERE, "queue_cli.py")


def run(*a):
    return subprocess.run([sys.executable, CLI, *a], capture_output=True, text=True)


def main():
    d = tempfile.mkdtemp()
    lock = os.path.join(d, ".device_hold")
    assert run("enqueue", lock, "lane-A", "sup-A", "30").returncode == 0
    assert run("enqueue", lock, "lane-B", "sup-B", "25").returncode == 0
    assert run("enqueue", lock, "lane-A", "sup-A", "30").returncode == 0  # idempotent
    q = json.load(open(lock + ".queue"))
    assert [e["script"] for e in q] == ["lane-A", "lane-B"], q
    assert run("front", lock, "lane-B").returncode == 0
    assert [e["script"] for e in json.load(open(lock + ".queue"))] == ["lane-B", "lane-A"]
    # B waits and gets the hold (no holder): a long-lived owner pid is this test
    r = run("wait", lock, "lane-B", str(os.getpid()), "--timeout", "5", "--poll", "0.2")
    assert r.returncode == 0, r.stderr
    h = json.load(open(lock))
    assert h["script"] == "lane-B", h
    assert [e["script"] for e in json.load(open(lock + ".queue"))] == ["lane-A"]
    # A cannot take it while B holds: times out with exit 3, still queued
    r = run("wait", lock, "lane-A", str(os.getpid()), "--timeout", "1", "--poll", "0.2")
    assert r.returncode == 3, (r.returncode, r.stderr)
    assert [e["script"] for e in json.load(open(lock + ".queue"))] == ["lane-A"]
    # B releases, A gets it
    subprocess.run([sys.executable, os.path.join(HERE, "hold_cli.py"), "release", lock, str(os.getpid())], check=True, capture_output=True)
    r = run("wait", lock, "lane-A", str(os.getpid()), "--timeout", "5", "--poll", "0.2")
    assert r.returncode == 0, r.stderr
    assert json.load(open(lock))["script"] == "lane-A"
    assert json.load(open(lock + ".queue")) == []
    subprocess.run([sys.executable, os.path.join(HERE, "hold_cli.py"), "release", lock, str(os.getpid())], check=True, capture_output=True)

    def release():
        subprocess.run([sys.executable, os.path.join(HERE, "hold_cli.py"), "release", lock, str(os.getpid())], check=True, capture_output=True)

    def line():
        return json.load(open(lock + ".queue"))

    # an entry enqueued ahead of time has no keeper waiting: it must not block the entry behind it, and it keeps its place
    assert run("enqueue", lock, "lane-C", "sup-C", "15").returncode == 0
    assert run("enqueue", lock, "lane-D", "sup-D", "40").returncode == 0
    assert [e.get("waiter", "absent") for e in line()] == [None, None], line()
    assert "(no keeper waiting)" in run("list", lock).stdout
    r = run("wait", lock, "lane-D", str(os.getpid()), "--timeout", "5", "--poll", "0.2")
    assert r.returncode == 0, r.stderr
    assert json.load(open(lock))["script"] == "lane-D"
    assert [e["script"] for e in line()] == ["lane-C"], line()
    # C's keeper starts while D holds: it times out (exit 3), stays queued, and its mark is cleared on the way out
    r = run("wait", lock, "lane-C", str(os.getpid()), "--timeout", "1", "--poll", "0.2")
    assert r.returncode == 3, (r.returncode, r.stderr)
    assert [(e["script"], e["waiter"]) for e in line()] == [("lane-C", None)], line()
    release()
    # a LIVE keeper ahead blocks the entry behind it even with the hold free (the order of the line holds between keepers)
    sleeper = subprocess.Popen(["sleep", "30"])
    q = line(); q[0]["waiter"] = sleeper.pid
    json.dump(q, open(lock + ".queue", "w"))
    assert "(keeper waiting)" in run("list", lock).stdout
    r = run("wait", lock, "lane-E", str(os.getpid()), "--timeout", "1", "--poll", "0.2")
    assert r.returncode == 3, (r.returncode, r.stderr)
    assert [e["script"] for e in line()] == ["lane-C", "lane-E"], line()
    # that keeper dies: its entry no longer blocks, E takes the hold, C keeps its place
    sleeper.kill(); sleeper.wait()
    r = run("wait", lock, "lane-E", str(os.getpid()), "--timeout", "5", "--poll", "0.2")
    assert r.returncode == 0, r.stderr
    assert json.load(open(lock))["script"] == "lane-E"
    assert [e["script"] for e in line()] == ["lane-C"], line()
    release()
    # an entry written by the previous version (no "waiter" key) is treated as waiting: the old behaviour
    json.dump([{"script": "legacy-L", "contact": "sup-L", "minutes": 20, "enqueued": "2026-10-05 14:00:00"}], open(lock + ".queue", "w"))
    r = run("wait", lock, "lane-F", str(os.getpid()), "--timeout", "1", "--poll", "0.2")
    assert r.returncode == 3, (r.returncode, r.stderr)
    assert "legacy-L  sup-L  ~20 min  queued 2026-10-05 14:00:00\n" in run("list", lock).stdout + "\n"
    # an entry written by another tool sharing the line, with no "contact" key: listed with ?, and its keeper's wait
    # (which writes only the waiter) keeps the entry's own fields
    json.dump([{"script": "other-O", "owner": "x", "minutes": 5, "enqueued": "2026-10-10 13:33:39", "waiter": None}],
              open(lock + ".queue", "w"))
    r = run("list", lock)
    assert r.returncode == 0 and "1. other-O  ?  ~5 min  queued 2026-10-10 13:33:39 (no keeper waiting)" in r.stdout, (r.stdout, r.stderr)
    r = run("wait", lock, "other-O", str(os.getpid()), "--timeout", "5", "--poll", "0.2")
    assert r.returncode == 0, r.stderr
    assert json.load(open(lock))["script"] == "other-O" and line() == [], line()
    release()
    json.dump([{"script": "other-O", "owner": "x", "minutes": 5, "enqueued": "2026-10-10 13:33:39", "waiter": None}],
              open(lock + ".queue", "w"))
    sleeper = subprocess.Popen(["sleep", "30"])
    holder_cli = subprocess.run([sys.executable, os.path.join(HERE, "hold_cli.py"), "acquire", lock, "holder-X", str(sleeper.pid)],
                                capture_output=True, text=True)
    assert holder_cli.returncode == 0, holder_cli.stderr
    r = run("wait", lock, "other-O", str(os.getpid()), "--timeout", "1", "--poll", "0.2")
    assert r.returncode == 3, (r.returncode, r.stderr)
    assert line() == [{"script": "other-O", "owner": "x", "minutes": 5, "enqueued": "2026-10-10 13:33:39", "waiter": None}], line()
    subprocess.run([sys.executable, os.path.join(HERE, "hold_cli.py"), "release", lock, str(sleeper.pid)], check=True, capture_output=True)
    sleeper.kill(); sleeper.wait()
    json.dump([], open(lock + ".queue", "w"))
    # two keepers at once, the later entry started first: the earlier entry takes the hold, the later one after its release
    json.dump([], open(lock + ".queue", "w"))
    assert run("enqueue", lock, "lane-G", "sup-G", "10").returncode == 0
    assert run("enqueue", lock, "lane-H", "sup-H", "10").returncode == 0
    ph = subprocess.Popen([sys.executable, CLI, "wait", lock, "lane-H", str(os.getpid()), "--timeout", "20", "--poll", "0.2"], stdout=subprocess.PIPE, text=True)
    time.sleep(1.0)
    assert json.load(open(lock))["script"] == "lane-H"  # G has no keeper yet, so H does not wait for it
    pg = subprocess.Popen([sys.executable, CLI, "wait", lock, "lane-G", str(os.getpid()), "--timeout", "20", "--poll", "0.2"], stdout=subprocess.PIPE, text=True)
    assert ph.wait(timeout=10) == 0
    time.sleep(0.6)
    assert pg.poll() is None  # G waits while H holds
    release()
    assert pg.wait(timeout=10) == 0
    assert json.load(open(lock))["script"] == "lane-G"
    assert line() == [], line()
    release()
    # one hold file per phone: a keeper of the second phone (.device_hold.pixel8a, its own line) takes that phone while
    # the first is held, and each hold is labelled with the phone its file arbitrates
    lock2 = lock + ".pixel8a"
    r = run("wait", lock, "lane-I", str(os.getpid()), "--timeout", "5", "--poll", "0.2")
    assert r.returncode == 0, r.stderr
    r = run("wait", lock2, "lane-P", str(os.getpid()), "--timeout", "5", "--poll", "0.2")
    assert r.returncode == 0, r.stderr
    assert json.load(open(lock))["device"] == "phone", json.load(open(lock))
    assert json.load(open(lock2))["device"] == "pixel8a", json.load(open(lock2))
    subprocess.run([sys.executable, os.path.join(HERE, "hold_cli.py"), "release", lock2, str(os.getpid())], check=True, capture_output=True)
    release()
    print("test_queue_cli OK")


if __name__ == "__main__":
    main()
