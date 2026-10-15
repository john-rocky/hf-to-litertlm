#!/usr/bin/env python3
"""Tests of pool_parts.py on made-up parts (no phone, no lane files):

    python3 test_pool_parts.py

With --repro, it also re-pools whole-set files that were pooled before: each file's `parts` (paths read from --cwd)
pooled with the rules of --rules must give every field of that file back exactly, floats included; then the same
parts with the last one missing must give INCOMPLETE.

    python3 test_pool_parts.py --repro <rules file> --cwd <dir> <whole.json> [...]

Exit 0 when every case passes.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "pool_parts.py")
fails = []


def check(name, cond, got=None):
    print(f"{'ok  ' if cond else 'FAIL'} {name}" + ("" if cond or got is None else f": got {got!r}"))
    if not cond:
        fails.append(name)


def run(args, cwd=None):
    """-> (exit code, the whole-set JSON or None, stdout + stderr)"""
    out = os.path.join(tmp, "whole.json")
    if os.path.exists(out):
        os.remove(out)
    r = subprocess.run([sys.executable, TOOL, "--out", out] + args, cwd=cwd, capture_output=True, text=True)
    doc = json.load(open(out)) if os.path.exists(out) else None
    return r.returncode, doc, r.stdout + r.stderr


def part(name, **fields):
    p = os.path.join(tmp, name)
    with open(p, "w") as fh:
        json.dump(fields, fh)
    return p


def synthetic():
    rows_ok = [{"key": "r0", "mask_equal": True}, {"key": "r1"}]
    a = part("a.json", tag="A", verdict="FAIL", tokens=3, cos_mean=0.5, cos_min=0.2, max_abs=1.5, rows=2, rows_present=2,
             per_row=rows_ok, n={"below": 1, "list": [{"cos": 0.3}, {"cos": 0.1}], "silent": 1, "quiet_mean": 0.25,
                                 "quiet_min": 0.2})
    b = part("b.json", tag="B", verdict="PASS", tokens=1, cos_mean=1.0, cos_min=0.9, max_abs=0.5, rows=1, rows_present=1,
             per_row=rows_ok, n={"below": 0, "list": [{"cos": 0.05}], "silent": 0, "quiet_mean": 0.75, "quiet_min": 0.7})
    rule = ["--weight", "tokens", "--sum", "tokens", "--wmean", "cos_mean", "--min", "cos_min", "--max", "max_abs",
            "--sum", "n.below", "--concat", "n.list:cos:2", "--wmean", "n.quiet_mean:tokens-n.silent",
            "--per-part", "tag,verdict,tokens", "--complete", "verdict in (PASS, FAIL)", "--complete", "rows_present == @rows",
            "--complete", "per_row[].mask_equal != false"]

    # 1. the pooled numbers: token-weighted mean, min of the mins, max, sums, the joined list, a weight A-B
    rc, d, txt = run(rule + ["--bar", "cos_mean >= 0.6", "--bar", "cos_min >= 0.1", a, b])
    check("exit 0", rc == 0, txt)
    check("weighted mean = (0.5*3 + 1.0*1) / 4", d and d["cos_mean"] == 0.625, d and d["cos_mean"])
    check("min of the mins, max of the maxes", d and (d["cos_min"], d["max_abs"]) == (0.2, 1.5))
    check("sums, nested paths", d and (d["tokens"], d["n"]["below"]) == (4, 1))
    check("the joined list sorted by cos, first 2", d and d["n"]["list"] == [{"cos": 0.05}, {"cos": 0.1}], d and d["n"])
    check("a weight A-B: (0.25*2 + 0.75*1) / 3", d and d["n"]["quiet_mean"] == (0.25 * 2 + 0.75 * 1) / 3)
    check("parts as given, per_part, worst part = the smallest cos_min",
          d and d["parts"] == [a, b] and d["per_part"][0] == {"tag": "A", "verdict": "FAIL", "tokens": 3}
          and d["worst_part"] == "A")
    # 2. a part's own FAIL is a subset verdict: the pooled numbers decide
    check("PASS on the pooled bar although part A says FAIL", d and d["verdict"] == "PASS" and d["complete"] is True,
          d and d["verdict"])
    check("bar text = the bars joined", d and d["bar"] == "cos_mean >= 0.6 and cos_min >= 0.1")
    # 3. the pooled bar fails
    rc, d, _ = run(rule + ["--bar", "cos_min >= 0.5", a, b])
    check("FAIL when a pooled value misses the bar", d and d["verdict"] == "FAIL" and "incomplete" not in d)
    # 4. a listed part that is absent: INCOMPLETE, never PASS, and the line says which
    rc, d, txt = run(rule + ["--bar", "cos_min >= 0.1", a, b, os.path.join(tmp, "c.json")])
    check("an absent part -> INCOMPLETE", d and d["verdict"] == "INCOMPLETE" and d["complete"] is False, d and d["verdict"])
    check("... named in incomplete", d and d["incomplete"] == [{"part": os.path.join(tmp, "c.json"), "why": "absent"}])
    check("... and in the printed line", "INCOMPLETE 3 parts (2 read)" in txt and "c.json absent" in txt, txt)
    # 5. a part that fails --complete (rows missing, a mask that differs, a verdict that is not a score)
    c = part("c.json", tag="C", verdict="PASS", tokens=1, cos_mean=1.0, cos_min=0.95, max_abs=0.1, rows=3, rows_present=2,
             per_row=rows_ok, n={"below": 0, "list": [], "silent": 0, "quiet_mean": 1.0})
    rc, d, _ = run(rule + [a, b, c])
    check("rows_present != rows -> INCOMPLETE", d and d["verdict"] == "INCOMPLETE"
          and d["incomplete"] == [{"part": c, "why": "not rows_present == @rows"}], d and d.get("incomplete"))
    c = part("c.json", tag="C", verdict="PASS", tokens=1, cos_mean=1.0, cos_min=0.95, max_abs=0.1, rows=1, rows_present=1,
             per_row=[{"key": "r0", "mask_equal": False}], n={"below": 0, "list": [], "silent": 0, "quiet_mean": 1.0})
    rc, d, _ = run(rule + [a, b, c])
    check("a row whose mask differs -> INCOMPLETE", d and d["verdict"] == "INCOMPLETE")
    c = part("c.json", tag="C", verdict="ERROR", tokens=1, cos_mean=1.0, cos_min=0.95, max_abs=0.1, rows=1, rows_present=1,
             per_row=rows_ok, n={"below": 0, "list": [], "silent": 0, "quiet_mean": 1.0})
    rc, d, _ = run(rule + [a, b, c])
    check("verdict ERROR is not in (PASS, FAIL) -> INCOMPLETE", d and d["verdict"] == "INCOMPLETE")
    # 6. a part without a value a rule reads; a part whose weight is 0 may lack it
    c = part("c.json", tag="C", verdict="PASS", tokens=1, cos_mean=1.0, cos_min=0.95, rows=1, rows_present=1,
             per_row=rows_ok, n={"below": 0, "list": [], "silent": 0, "quiet_mean": 1.0})
    rc, d, _ = run(rule + [a, b, c])
    check("a part without max_abs -> INCOMPLETE", d and d["verdict"] == "INCOMPLETE"
          and {"part": c, "why": "max_abs is not a number"} in d["incomplete"], d and d.get("incomplete"))
    c = part("c.json", tag="C", verdict="PASS", tokens=2, cos_mean=1.0, cos_min=0.95, max_abs=0.1, rows=1, rows_present=1,
             per_row=rows_ok, n={"below": 0, "list": [], "silent": 2, "quiet_mean": None, "quiet_min": None})
    rc, d, _ = run(rule + ["--min", "n.quiet_min:tokens-n.silent", a, b, c])
    check("weight 0 (all tokens silent): its null values are skipped, the whole stays complete",
          d and d["complete"] is True and d["n"]["quiet_mean"] == (0.25 * 2 + 0.75 * 1) / 3 and d["n"]["quiet_min"] == 0.2,
          d and (d.get("incomplete"), d["n"]))
    # 7. a bar on a value that has no pooled number (sum of the weights 0)
    z = part("z.json", tag="Z", verdict="PASS", tokens=2, cos_mean=1.0, cos_min=0.95, max_abs=0.1, rows=1, rows_present=1,
             per_row=rows_ok, n={"below": 0, "list": [], "silent": 2, "quiet_mean": None})
    rc, d, _ = run(rule + ["--bar", "n.quiet_mean >= 0.5", z])
    check("no silent-free token at all: the mean is null and its bar gives INCOMPLETE",
          d and d["n"]["quiet_mean"] is None and d["verdict"] == "INCOMPLETE", d and d.get("incomplete"))
    # 8. the same rule from a file (shell quoting, comments), and conditions: in / not in, null never ordered
    rf = os.path.join(tmp, "rules.txt")
    with open(rf, "w") as fh:
        fh.write("# the rule of case 1\n" + " ".join(f"'{x}'" for x in rule) + "\n--bar 'cos_mean >= 0.6'  # the bar\n")
    rc, d2, _ = run(["--rules", rf, a, b])
    check("--rules FILE gives the same pooled values as the arguments", d2 and d2["cos_mean"] == 0.625
          and d2["n"] == {"below": 1, "list": [{"cos": 0.05}, {"cos": 0.1}], "quiet_mean": (0.25 * 2 + 0.75 * 1) / 3}
          and d2["verdict"] == "PASS")
    rc, d, _ = run(["--sum", "tokens", "--complete", "verdict not in (ERROR, NONFINITE)", "--complete", "note == null",
                    "--bar", "tokens >= 4", a, b])
    check("not in; an absent field == null", d and d["verdict"] == "PASS", d and (d["verdict"], d.get("incomplete")))
    rc, d, _ = run(["--sum", "tokens", "--complete", "cos_floor >= 0", a, b])
    check("an absent field is null, and null is never ordered (cos_floor >= 0 fails)", d and d["verdict"] == "INCOMPLETE")
    # 9. a part that is not JSON
    bad = os.path.join(tmp, "bad.json")
    open(bad, "w").write("not json")
    rc, d, _ = run(rule + [a, bad])
    check("an unreadable part -> INCOMPLETE", d and d["verdict"] == "INCOMPLETE"
          and d["incomplete"][0]["why"].startswith("unreadable"))
    # 10. usage errors: exit 2 and no file
    for args, what in ((["--sum", "tokens"], "no part"), (["--bar", "cos_mean => 1", a], "a condition with no operator"),
                       (["--wmean", "cos_mean", a], "--wmean with no weight"), (["--sum", "x", "--min", "x", a], "a field pooled twice"),
                       (["--complete", "verdict in PASS", a], "in without (...)"), (["--rules", rf + ".none", a], "no rules file")):
        rc, d, _ = run(args)
        check(f"usage error: {what} -> exit 2, no file", rc == 2 and d is None, rc)


def repro(rules, cwd, wholes):
    for w in wholes:
        ref = json.load(open(w))
        rc, d, txt = run(["--rules", rules] + ref["parts"], cwd=cwd)
        name = os.path.basename(w)
        if not d:
            check(f"{name}: pooled", False, txt)
            continue
        diff = [k for k in ref if d.get(k, "<absent>") != ref[k]]
        extra = sorted(set(d) - set(ref))
        check(f"{name}: {len(ref)} fields equal ({len(ref['parts'])} parts, {ref['verdict']}), no extra field",
              not diff and not extra, {"differ": {k: (d.get(k), ref[k]) for k in diff}, "extra": extra})
        cut = ref["parts"][:-1] + [ref["parts"][-1] + ".absent"]
        rc, d, txt = run(["--rules", rules] + cut, cwd=cwd)
        check(f"{name}: its last part missing -> INCOMPLETE", d and d["verdict"] == "INCOMPLETE", txt.strip())


tmp = tempfile.mkdtemp()
try:
    synthetic()
    if "--repro" in sys.argv:
        argv = sys.argv[1:]
        rules = os.path.abspath(argv[argv.index("--repro") + 1])
        cwd = argv[argv.index("--cwd") + 1] if "--cwd" in argv else os.getcwd()
        skip = {argv.index("--repro"), argv.index("--repro") + 1}
        if "--cwd" in argv:
            skip |= {argv.index("--cwd"), argv.index("--cwd") + 1}
        repro(rules, cwd, [os.path.abspath(a) for i, a in enumerate(argv) if i not in skip])
finally:
    shutil.rmtree(tmp)
print()
if fails:
    print(f"{len(fails)} FAILED: {fails}")
    sys.exit(1)
print("all pool_parts cases pass")
