#!/usr/bin/env python3
"""Self-test of the desk check against the shipped examples (no engine, no weights).

  python tools/chat_template_prefix/selftest.py

1. calibration: the renderer, set up as the runtime is, reproduces the two strings the runtime
   printed for the Nemotron-3-Nano-4B bundle's failing second message byte for byte;
2. the template that bundle carried until 2026-10-13 fails the three-turn table when the reply
   carries a thought and passes with thinking off;
3. its fix, templates/nemotron3nano_history.jinja, passes every cell.
Exit 0 = all three hold, 1 = a check failed (the failing check is named).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import desk_render as dr  # noqa: E402

EX = os.path.join(HERE, "examples")
ORIGINAL = os.path.join(EX, "nemotron3nano_original.jinja")
FIXED = os.path.join(HERE, "..", "..", "templates", "nemotron3nano_history.jinja")
CONV = os.path.join(EX, "conv_three_turns.json")
LOG_OLD = os.path.join(EX, "nemotron_old.txt")
LOG_NEW = os.path.join(EX, "nemotron_new.txt")
FORMS = ("rt", "i", "ii", "iii", "v0171")


def read(path, mode="r"):
  with open(path, mode, **({} if "b" in mode else {"encoding": "utf-8"})) as f:
    return f.read()


def engines():
  return ("jinja2", "minijinja") if dr.have_minijinja() else ("jinja2",)


def table_rows(src, conv, eng):
  r = dr.Renderer(src, eng)
  rows = dr.three_turn_table(r, conv["on"], FORMS)
  off = dr.three_turn_table(r, conv["off"], ["rt"])
  for row in off:
    row["form"] = "rt-off-run"
  return rows + off


def passes(row):
  return "error" not in row and row["step_check"]["pass"] and row["refill_check"]["pass"]


def main():
  conv = json.load(open(CONV, encoding="utf-8"))
  log_old, log_new = read(LOG_OLD, "rb"), read(LOG_NEW, "rb")
  original, fixed = read(ORIGINAL), read(FIXED)
  t1 = conv["on"][0]
  failures = []

  for eng in engines():
    # 1. calibration: runtime message form (rt), trim/lstrip/keep_trailing_newline on, edits on
    r = dr.Renderer(original, eng)
    hist = [dr.user_msg(t1["user"]), dr.assistant_msg(t1["content"], t1.get("thought"))]
    old = r.render(hist, False).encode("utf-8")
    new = r.render(hist + [dr.user_msg(conv["on"][1]["user"])], True).encode("utf-8")
    if old != log_old or new != log_new:
      failures.append(f"calibration ({eng}): old == log {old == log_old}, new == log {new == log_new}")

    # 2. the original fails with a thought in the history, passes with thinking off
    rows = table_rows(original, conv, eng)
    rt_fail = [row for row in rows if row["form"] == "rt" and not passes(row)]
    off_fail = [row for row in rows if row["form"] == "rt-off-run" and not passes(row)]
    if not rt_fail:
      failures.append(f"original ({eng}): no failing cell in form rt")
    if off_fail:
      failures.append(f"original ({eng}): thinking-off row failed: {dr.cell(off_fail[0])}")

    # 3. the fix passes every cell
    bad = [row for row in table_rows(fixed, conv, eng) if not passes(row)]
    if bad:
      failures.append(f"fix ({eng}): {len(bad)} failing cell(s), first = form {bad[0]['form']} "
                      f"enable_thinking {bad[0]['enable_thinking']} step {bad[0]['step']}")

  print(f"engines: {', '.join(engines())}")
  for f in failures:
    print("FAIL", f)
  if failures:
    sys.exit(1)
  print("PASS calibration, original fails, fix passes")


if __name__ == "__main__":
  main()
