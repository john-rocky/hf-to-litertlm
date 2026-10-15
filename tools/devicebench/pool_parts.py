#!/usr/bin/env python3
"""pool_parts.py: one exact verdict for a check set that ran as parts (one rows file and one leg per part), pooled from
the parts' scorer files with the pooling rule given as arguments. A part's own verdict is a verdict on a subset of the
set; the pooled numbers decide.

    python3 pool_parts.py --out <whole.json> [rules] <part.json> [<part.json> ...]

List every part of the set, present or not: a part whose file is absent, unreadable or not a JSON object makes the
whole INCOMPLETE (never PASS), so a leg that did not run cannot quietly shrink the set.

Rules (FIELD is a dotted path into a part's JSON, e.g. bar_tokens.n_below; every rule may be given more than once):
  --wmean F[:W]          sum(F * w) / sum(w) over the parts, w = the part's W: a field, a number, or A-B (the difference
                         of two fields). Default W: --weight. A part whose w is 0 adds nothing and may lack F; the
                         pooled value is null when sum(w) is 0.
  --weight W             the default W of --wmean
  --min F[:W], --max F[:W]   the smallest / largest F over the parts (a part whose W is 0 is skipped)
  --sum F                the sum over the parts
  --concat F[:KEY[:N]]   the parts' lists joined in part order, sorted by KEY (ascending, stable), the first N kept
  --complete COND        must hold for every part, else the whole is INCOMPLETE
  --bar COND             must hold on the pooled values for PASS, else FAIL
  --bar-text TEXT        the output's "bar" (default: the --bar conditions joined with " and ")
  --per-part F,F,...     the fields copied from each part into per_part
  --id F                 the part's name in worst_part (default tag; the file's name when the part has none)
  --worst F              worst_part = the part with the smallest F (default: the field of the first --min)
  --rules FILE           more arguments, read from FILE (shell quoting, any number per line, # comments)
A part that lacks a value a rule reads (absent, null, or not a number where a number is needed) makes the whole
INCOMPLETE, except where that part's weight is 0.

COND = FIELD OP VALUE with OP one of >= <= > < == != in, not in. VALUE: a number, true / false / null, a word or a
quoted string, (a, b, ...) for in / not in, or @FIELD (another field of the same document: the part for --complete,
the pooled values for --bar). A step written name[] goes through a list: the condition must hold for every element
(per_row[].mask_equal != false). An absent field reads as null, and null is never ordered (null >= 0.99 is false).

Output: parts (as given), the pooled fields at their paths, verdict (PASS / FAIL / INCOMPLETE), bar, complete,
worst_part, per_part (when --per-part is given) and, when the whole is not complete, incomplete (one reason per item).
Prints one line. Exit 0 whatever the verdict; 2 on a usage error. Standard library only.
"""
import argparse
import json
import os
import re
import shlex
import sys

MISSING = object()
COND = re.compile(r"^\s*(\S+)\s+(not\s+in|in|>=|<=|==|!=|>|<)\s+(.+?)\s*$")


def get(doc, path):
    """The value at a dotted path, or MISSING. A step ending in [] goes through a list: the result is the list of the
    values below it (MISSING when that step is not a list)."""
    vals, many = [doc], False
    for step in path.split("."):
        each = step.endswith("[]")
        key = step[:-2] if each else step
        out = []
        for x in vals:
            v = x.get(key, MISSING) if isinstance(x, dict) else MISSING
            if each:
                if not isinstance(v, list):
                    return MISSING
                out.extend(v)
            else:
                out.append(v)
        vals, many = out, many or each
    return vals if many else vals[0]


def null(v):
    return None if v is MISSING else v


def is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def literal(t):
    t = t.strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "'\"":
        return t[1:-1]
    if t.lower() in ("true", "false"):
        return t.lower() == "true"
    if t.lower() == "null":
        return None
    for conv in (int, float):
        try:
            return conv(t)
        except ValueError:
            pass
    return t


def same(a, b):
    """== without Python's bool / int mix-up (false == 0 is false here)"""
    return isinstance(a, bool) == isinstance(b, bool) and a == b


class Cond:
    def __init__(self, text):
        m = COND.match(text)
        if not m:
            raise ValueError(f"not a condition: {text!r} (FIELD OP VALUE)")
        self.text, self.path, self.op = text.strip(), m.group(1), " ".join(m.group(2).split())
        v = m.group(3).strip()
        self.ref = v[1:] if v.startswith("@") else None
        if self.ref is None and v.startswith("(") and v.endswith(")"):
            self.val = tuple(literal(x) for x in v[1:-1].split(",") if x.strip())
        else:
            self.val = None if self.ref else literal(v)
        if self.op in ("in", "not in") and not isinstance(self.val, tuple):
            raise ValueError(f"{text!r}: in / not in takes (a, b, ...)")

    def holds(self, doc):
        right = null(get(doc, self.ref)) if self.ref else self.val
        left = get(doc, self.path)
        if "[]" in self.path:
            return left is not MISSING and all(self.compare(null(x), right) for x in left)
        return self.compare(null(left), right)

    def compare(self, a, b):
        if self.op == "in":
            return any(same(a, x) for x in b)
        if self.op == "not in":
            return not any(same(a, x) for x in b)
        if self.op == "==":
            return same(a, b)
        if self.op == "!=":
            return not same(a, b)
        if a is None or b is None or isinstance(a, bool) or isinstance(b, bool):
            return False
        try:
            return {">=": a >= b, "<=": a <= b, ">": a > b, "<": a < b}[self.op]
        except TypeError:
            return False


def weight_fn(spec):
    """W = a field, a number, or A-B -> a function of a part: its weight, or MISSING"""
    terms = spec.split("-")
    if len(terms) > 2 or not all(t.strip() for t in terms):
        raise ValueError(f"not a weight: {spec!r} (a field, a number, or A-B)")

    def term(t):
        v = literal(t)
        return (lambda d: v) if is_num(v) else (lambda d: get(d, t.strip()))
    fs = [term(t) for t in terms]

    def w(doc):
        vals = [f(doc) for f in fs]
        if not all(is_num(v) for v in vals):
            return MISSING
        return vals[0] - vals[1] if len(vals) == 2 else vals[0]
    return w


class RuleAction(argparse.Action):
    """--wmean / --min / --max / --sum / --concat keep their order (the output's order) in one list"""
    def __call__(self, parser, ns, value, option_string=None):
        setattr(ns, "rules", list(getattr(ns, "rules", None) or []) + [(self.const, value)])


def expand_rules(argv, depth=0):
    """--rules FILE -> the arguments in FILE, in place"""
    out, i = [], 0
    while i < len(argv):
        a = argv[i]
        if a == "--rules" or a.startswith("--rules="):
            path = a.split("=", 1)[1] if "=" in a else (argv[i + 1] if i + 1 < len(argv) else "")
            i += 1 if "=" in a else 2
            if not path or depth > 4:
                raise ValueError("--rules needs a file (and files may nest 4 deep)")
            with open(path, encoding="utf-8") as fh:
                out.extend(expand_rules(shlex.split(fh.read(), comments=True), depth + 1))
            continue
        out.append(a)
        i += 1
    return out


def put(doc, path, value):
    *up, last = path.split(".")
    for k in up:
        doc = doc.setdefault(k, {})
        if not isinstance(doc, dict):
            raise ValueError(f"{path}: {k} is pooled as a value and as a parent")
    doc[last] = value


def pool(kind, spec, docs, default_w, why):
    """-> (output path, pooled value); a part that lacks what the rule reads is written to `why`"""
    if kind == "concat":
        field, *rest = spec.split(":")
        key = rest[0] if rest and rest[0] else None
        keep = int(rest[1]) if len(rest) > 1 and rest[1] else None
        items = []
        for name, d in docs:
            v = get(d, field)
            if not isinstance(v, list):
                why.append((name, f"{field} is not a list"))
                continue
            items.extend(v)
        if key:
            keyed = [(null(get(t, key)) if isinstance(t, dict) else None, t) for t in items]
            bad = sum(1 for k, _ in keyed if not is_num(k))
            if bad:
                why.append((None, f"{bad} items of {field} have no number {key}"))
            # a stable sort: items with equal keys keep the parts' order
            items = [t for _, t in sorted(keyed, key=lambda kt: (not is_num(kt[0]), kt[0] if is_num(kt[0]) else 0))]
        return field, items[:keep] if keep is not None else items
    field, _, wspec = spec.partition(":")
    if kind == "sum":
        vals = []
        for name, d in docs:
            v = get(d, field)
            if not is_num(v):
                why.append((name, f"{field} is not a number"))
                continue
            vals.append(v)
        return field, sum(vals)
    wspec = wspec or (default_w if kind == "wmean" else "")
    wf = weight_fn(wspec) if wspec else None
    terms, den, vals = [], [], []
    for name, d in docs:
        w = wf(d) if wf else 1
        if w is MISSING or w < 0:
            why.append((name, f"the weight {wspec} of {field} is not a number >= 0"))
            continue
        if w == 0:
            continue
        v = get(d, field)
        if not is_num(v):
            why.append((name, f"{field} is not a number"))
            continue
        terms.append(v * w)
        den.append(w)
        vals.append(v)
    if kind == "wmean":
        total = sum(den)
        return field, (sum(terms) / total if total else None)
    if kind == "min":
        return field, min(vals, default=None)
    return field, max(vals, default=None)


def main(argv):
    ap = argparse.ArgumentParser(prog="pool_parts.py", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    for kind, meta in (("wmean", "F[:W]"), ("min", "F[:W]"), ("max", "F[:W]"), ("sum", "F"), ("concat", "F[:KEY[:N]]")):
        ap.add_argument("--" + kind, action=RuleAction, const=kind, dest="rules", metavar=meta, default=[])
    ap.add_argument("--weight", default="")
    ap.add_argument("--complete", action="append", default=[], metavar="COND")
    ap.add_argument("--bar", action="append", default=[], metavar="COND")
    ap.add_argument("--bar-text", default=None)
    ap.add_argument("--per-part", default="", metavar="F,F,...")
    ap.add_argument("--id", default="tag", metavar="F")
    ap.add_argument("--worst", default=None, metavar="F")
    ap.add_argument("parts", nargs="+")
    try:
        args = ap.parse_intermixed_args(expand_rules(argv))
        completes = [Cond(c) for c in args.complete]
        bars = [Cond(c) for c in args.bar]
        for kind, spec in args.rules:
            if kind == "wmean" and ":" not in spec and not args.weight:
                raise ValueError(f"--wmean {spec}: no weight (F:W, or --weight W)")
            if kind in ("wmean", "min", "max") and ":" in spec:
                weight_fn(spec.split(":", 1)[1])
        paths = [spec.split(":")[0] for _, spec in args.rules]
        if len(set(paths)) != len(paths):
            raise ValueError("a field is pooled twice: " + ", ".join(sorted({p for p in paths if paths.count(p) > 1})))
    except (OSError, ValueError) as e:
        ap.error(str(e))

    why, docs = [], []
    for p in args.parts:
        if not os.path.exists(p):
            why.append((p, "absent"))
            continue
        try:
            with open(p, encoding="utf-8") as fh:
                d = json.load(fh)
        except (OSError, ValueError) as e:
            why.append((p, f"unreadable: {e}"))
            continue
        if not isinstance(d, dict):
            why.append((p, "not a JSON object"))
            continue
        docs.append((p, d))
    for p, d in docs:
        why.extend((p, f"not {c.text}") for c in completes if not c.holds(d))

    res = {"parts": list(args.parts)}
    try:
        for kind, spec in args.rules:
            path, value = pool(kind, spec, docs, args.weight, why)
            put(res, path, value)
    except ValueError as e:
        ap.error(str(e))
    for c in bars:
        if null(get(res, c.path)) is None:
            why.append((None, f"{c.path} has no pooled value"))
    complete = not why
    ok = [c.holds(res) for c in bars]
    res["verdict"] = "INCOMPLETE" if not complete else ("PASS" if all(ok) else "FAIL")
    res["bar"] = args.bar_text if args.bar_text is not None else " and ".join(c.text for c in bars)
    res["complete"] = complete
    worst_field = args.worst or next((spec.split(":")[0] for kind, spec in args.rules if kind == "min"), None)
    ranked = [(get(d, worst_field), n) for n, (p, d) in enumerate(docs)] if worst_field else []
    ranked = [(v, n) for v, n in ranked if is_num(v)]
    worst = docs[min(ranked)[1]] if ranked else None
    res["worst_part"] = (null(get(worst[1], args.id)) or os.path.basename(worst[0])) if worst else None
    if args.per_part:
        res["per_part"] = [{f: null(get(d, f)) for f in args.per_part.split(",") if f} for _, d in docs]
    if not complete:
        res["incomplete"] = [{"part": p, "why": w} for p, w in why]
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
        fh.write("\n")
    os.replace(tmp, args.out)
    line = [f"{res['verdict']} {len(args.parts)} parts" + (f" ({len(docs)} read)" if len(docs) != len(args.parts) else "")]
    line += [f"{c.text}: {null(get(res, c.path))} {'ok' if k else 'no'}" for c, k in zip(bars, ok)]
    if res["worst_part"] is not None:
        line.append(f"worst part {res['worst_part']} ({worst_field})")
    if not complete:
        line.append("incomplete: " + "; ".join(f"{os.path.basename(p) if p else 'pool'} {w}" for p, w in why[:5])
                    + (f"; +{len(why) - 5} more" if len(why) > 5 else ""))
    print("; ".join(line))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
