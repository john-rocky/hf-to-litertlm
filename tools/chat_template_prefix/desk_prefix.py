#!/usr/bin/env python3
"""Desk prefix-contract checks (no engine): calibration against the runtime's log, and the
three-turn tables.

  desk_prefix.py calibrate JINJA LOG_OLD LOG_NEW CONV           -> markdown on stdout
  desk_prefix.py table JINJA CONV [--forms rt,ii,iii] [--json OUT] -> markdown on stdout

CONV = a conversation shaped like examples/conv_three_turns.json: "on" = the turns of a run with
thinking on, "off" = the turns of a run with thinking off, each turn {"user", "content", optional
"thought" (the reply's thought channel)}. table needs 3 turns in "on" and in "off"; calibrate
needs 2 in "on". Engines: jinja2 always, minijinja when minijinja-py is importable
(see desk_render.py).
"""
import argparse
import hashlib
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import desk_render as dr  # noqa: E402

CONFIGS = (
    ("F/F/F", dict(trim_blocks=False, lstrip_blocks=False, keep_trailing_newline=False)),
    ("T/T/T", dict(trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True)),
    ("F/F/T", dict(trim_blocks=False, lstrip_blocks=False, keep_trailing_newline=True)),
    ("T/T/F", dict(trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=False)),
)
FORMS = ("rt", "i", "ii", "iii", "v0171")


def engines():
  return ("jinja2", "minijinja") if dr.have_minijinja() else ("jinja2",)


def sha12(b):
  return hashlib.sha256(b).hexdigest()[:12]


def need_turns(cmd, path, doc, n, keys):
  """One line on stderr and exit 2 when a turn list of CONV is shorter than the check needs."""
  found = {k: len(doc.get(k) or []) for k in keys}
  if min(found.values()) < n:
    have = ", ".join(f'"{k}" {v}' for k, v in found.items())
    sys.stderr.write(f"desk_prefix.py {cmd}: {path} needs {n} turns in "
                     + " and ".join(f'"{k}"' for k in keys) + f"; found {have}\n")
    sys.exit(2)


def calibrate(args):
  doc = json.load(open(args.conv))
  need_turns("calibrate", args.conv, doc, 2, ("on",))
  conv = doc["on"]
  run_calibration(args.jinja, args.log_old, args.log_new, conv[0], conv[1])


def run_calibration(jinja, log_old_p, log_new_p, t1, t2):
  src = open(jinja, encoding="utf-8").read()
  log_old = open(log_old_p, "rb").read()
  log_new = open(log_new_p, "rb").read()
  print(f"log_old {len(log_old)} B sha {sha12(log_old)}; log_new {len(log_new)} B sha {sha12(log_new)}")
  print()
  print("| engine | edit | trim/lstrip/keep_nl | system msg | form | old == log_old | new == log_new |")
  print("|---|---|---|---|---|---|---|")
  hits = []
  for eng in engines():
    for edit in (True, False):
      for cname, cfg in CONFIGS:
        try:
          r = dr.Renderer(src, eng, edit=edit, **cfg)
        except Exception as e:  # noqa: BLE001
          print(f"| {eng} | {edit} | {cname} | - | - | parse error {type(e).__name__} | |")
          continue
        for sysmsg in (False, True):
          for form in FORMS:
            pre = [{"role": "system", "content": dr.parts("") if form != "v0171" else ""}] if sysmsg else []
            hist = pre + [dr.user_msg(t1["user"], form), dr.assistant_msg(t1["content"], t1.get("thought"), form)]
            try:
              old = r.render(hist, False).encode("utf-8")
              new = r.render(hist + [dr.user_msg(t2["user"], form)], True).encode("utf-8")
            except Exception as e:  # noqa: BLE001
              print(f"| {eng} | {edit} | {cname} | {sysmsg} | {form} | render error {type(e).__name__}: {str(e)[:60]} | |")
              continue
            mo, mn = old == log_old, new == log_new
            if mo and mn:
              hits.append((eng, edit, cname, sysmsg, form))
            def show(m, b, ref):
              if m:
                return "**byte-equal**"
              i = dr.divergence(b.decode("utf-8", "replace"), ref.decode("utf-8", "replace"))
              return f"differs at {i} ({len(b)} vs {len(ref)} B)"
            print(f"| {eng} | {edit} | {cname} | {sysmsg} | {form} | {show(mo, old, log_old)} | {show(mn, new, log_new)} |")
  print()
  print("byte-equal on both strings:", hits if hits else "none")


def table(args):
  src = open(args.jinja, encoding="utf-8").read()
  doc = json.load(open(args.conv))
  need_turns("table", args.conv, doc, 3, ("on", "off"))
  forms = args.forms.split(",")
  out = {"jinja": args.jinja, "jinja_sha256": hashlib.sha256(src.encode("utf-8")).hexdigest(),
         "engines": {}}
  for eng in engines():
    r = dr.Renderer(src, eng)
    rows = dr.three_turn_table(r, doc["on"], forms)
    off_rows = dr.three_turn_table(r, doc["off"], ["rt"])
    for row in off_rows:
      row["form"] = "rt-off-run"
    out["engines"][eng] = rows + off_rows
  if args.json:
    with open(args.json, "w") as f:
      json.dump(out, f, ensure_ascii=False, indent=1)
      f.write("\n")
  print(render_md(out, forms))


def render_md(out, forms):
  lines = []
  engs = list(out["engines"])
  lines.append(f"template sha256 {out['jinja_sha256'][:12]}; cell = step check / refill check "
               f"(PASS = old is a prefix of new); engines: {', '.join(engs)}")
  lines.append("")
  head = "| form | enable_thinking | " + " | ".join(f"{e} step 1 | {e} step 2" for e in engs) + " | runtime refills? | first split (step or refill, 40 chars each side) |"
  lines.append(head)
  lines.append("|" + "---|" * (head.count("|") - 1))
  base = out["engines"][engs[0]]
  for idx, row in enumerate(base):
    if row["step"] != 1:
      continue
    nxt = base[idx + 1]
    cells = []
    for e in engs:
      rr = out["engines"][e]
      cells += [dr.cell(rr[idx]), dr.cell(rr[idx + 1])]
    rf = "yes" if (row.get("refill_check", {}).get("runtime_runs_it")) else "no"
    split = ""
    for rw in (row, nxt):
      if "error" in rw:
        split = rw["error"][:120]
        break
      for name in ("step_check", "refill_check"):
        chk = rw[name]
        if not chk["pass"]:
          split = f"step {rw['step']} {name.split('_')[0]} at {chk['at']}: " + dr.fmt_split(chk)
          break
      if split:
        break
    lines.append(f"| {row['form']} | {row['enable_thinking']} | " + " | ".join(cells) + f" | {rf} | {split} |")
  return "\n".join(lines)


def main():
  ap = argparse.ArgumentParser()
  sub = ap.add_subparsers(dest="cmd", required=True)
  c = sub.add_parser("calibrate")
  c.add_argument("jinja")
  c.add_argument("log_old")
  c.add_argument("log_new")
  c.add_argument("conv")
  t = sub.add_parser("table")
  t.add_argument("jinja")
  t.add_argument("conv")
  t.add_argument("--forms", default="rt,i,ii,iii,v0171")
  t.add_argument("--json")
  args = ap.parse_args()
  {"calibrate": calibrate, "table": table}[args.cmd](args)


if __name__ == "__main__":
  main()
