#!/usr/bin/env python3
"""Check every row of card/facts.tsv against the card and against the primary files (no code shared with anything that
wrote the card). A row: n, card_quote, printed, sources, op, fmt, note.
  1. card_quote is in card/README.md, and printed is in card_quote;
  2. sources = 'path#key' items joined by ' ; ' (path relative to the current directory, .gz read as text); key =
     j:a.b.0 (JSON path) | re:REGEX (every match of group 1) | re1:REGEX (the first match) | lit:TEXT (the file
     contains TEXT; the value is TEXT) | tsv:COL=VAL:OUT (the TSV row whose COL is VAL, its column OUT);
  3. op on the values: one | min | max | minmax | sum | inv | kb2gb | sumgb | diff (v0 - v1) | kb2gb_diff |
     mean | inv_mean (1 / the mean: a row averaged over runs) | ratio (v0 / v1) | k (one value / 1000: ms -> s) |
     rtf_mean / inv_rtf_mean (values = burst_wall_ms of n runs, then audio_s: mean(walls) / 1000 / audio_s, or 1 / it;
     a row's rtf_wall from the raw burst times, not from the runner's 6-decimal rtf_wall) | max_kb2gb (the largest
     kB as decimal GB) | pct_vs_first (values = n walls, n walls, then audio_s: (mean of the second n / mean of the
     first n - 1) x 100) |
     div16000 | minghz (min kHz -> GHz) | minmax_k (values / 1000) | minmax_of_max (each source's largest value) |
     minmax_of_max_gb (the same, kB as decimal GB) |
     maxdelta (values = dev, ref, dev, ref...; the largest dev - ref) | maxdelta_ceil2 (that, rounded up to 0.01) |
     ncsv (the number of comma-separated items) | word (an integer as an English word) | join | str |
     present (every source matched: printed = fmt) | equal (every source one value, all equal, and that value is the
     printed number, as text or as a number: printed = fmt) | agree (every source one value, all equal: printed = fmt;
     a wording the agreement carries, such as 'the same bytes');
  4. fmt = a Python format spec for the number(s) (join: 'SPEC|SEP'; minmax: 'SPEC|SEP'); a trailing '×' is appended
     as text. The formatted value must equal printed.
  python3 tools/card_facts/facts_check.py [card/README.md] [card/facts.tsv]      (run from the directory the source
  paths are relative to; exit 0 = every row PASS)
"""
import gzip
import json
import re
import sys
from pathlib import Path

K = Path.cwd()


def read_text(p):
    return gzip.open(p, "rt", errors="replace").read() if p.suffix == ".gz" else p.read_text(errors="replace")


def num(v):
    if isinstance(v, (int, float)):
        return v
    s = str(v).replace(",", "")
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return v


def values(item):
    path, key = item.split("#", 1)
    p = K / path.strip()
    if key.startswith("j:"):
        d = json.loads(read_text(p))
        for part in key[2:].split("."):
            d = d[int(part)] if isinstance(d, list) else d[part]
        return [num(d)]
    if key.startswith("re:") or key.startswith("re1:"):
        rx = key.split(":", 1)[1]
        ms = [m if isinstance(m, str) else m[0] for m in re.findall(rx, read_text(p), re.M)]
        return [num(m) for m in (ms[:1] if key.startswith("re1:") else ms)]
    if key.startswith("lit:"):
        t = key[4:]
        return [t] if t in read_text(p) else []
    if key.startswith("tsv:"):
        cond, out = key[4:].rsplit(":", 1)
        col, val = cond.split("=", 1)
        lines = read_text(p).splitlines()
        hdr = lines[0].split("\t")
        for ln in lines[1:]:
            c = ln.split("\t")
            if len(c) == len(hdr) and c[hdr.index(col)] == val:
                return [num(c[hdr.index(out)])]
        return []
    raise ValueError(f"key {key!r}")


def fmt1(x, spec):
    tail = ""
    if spec.endswith("×"):
        spec, tail = spec[:-1], "×"
    return format(x, spec) + tail


WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def apply(op, vals, fmt, per=None):
    if op in ("minmax_of_max", "minmax_of_max_gb"):
        spec, sep = fmt.split("|")
        tops = [max(v) * (1024 / 1e9 if op.endswith("_gb") else 1) for v in per]
        return fmt1(min(tops), spec) + sep + fmt1(max(tops), spec)
    if op == "present":  # every source matched: the claim's wording (fmt) stands
        return fmt if per and all(per) else "absent"
    if op == "equal":  # every source gave one value, all the same, and that value is what the card prints
        if not (per and all(len(v) == 1 for v in per) and len({v[0] for v in per}) == 1):
            return "differ"
        v = per[0][0]
        return fmt if str(v) == fmt or num(fmt) == v else f"agree on {v!r}, not on {fmt!r}"
    if op == "agree":  # every source gave one value and all are the same (a wording: printed = fmt)
        return fmt if per and all(len(v) == 1 for v in per) and len({v[0] for v in per}) == 1 else "differ"
    if op == "ncsv":
        assert len(vals) == 1, vals
        return fmt1(len([x for x in str(vals[0]).split(",") if x.strip()]), fmt)
    if op == "word":
        assert len(vals) == 1, vals
        return WORDS[int(vals[0])]
    if op == "diff":
        assert len(vals) == 2, vals
        return fmt1(vals[0] - vals[1], fmt)
    if op == "kb2gb_diff":
        assert len(vals) == 2, vals
        return fmt1((vals[0] - vals[1]) * 1024 / 1e9, fmt)
    if op == "k":  # one value / 1000 (ms -> s)
        assert len(vals) == 1, vals
        return fmt1(vals[0] / 1000, fmt)
    if op == "div16000":
        assert len(vals) == 1, vals
        return fmt1(vals[0] / 16000, fmt)
    if op == "minghz":
        return fmt1(min(vals) / 1e6, fmt)
    if op == "minmax_k":
        spec, sep = fmt.split("|")
        return fmt1(min(vals) / 1000, spec) + sep + fmt1(max(vals) / 1000, spec)
    if op == "maxdelta_ceil2":
        import math
        assert len(vals) % 2 == 0, vals
        return fmt1(math.ceil(round(max(vals[i] - vals[i + 1] for i in range(0, len(vals), 2)) * 100, 6)) / 100, fmt)
    if op == "str":
        assert len(vals) == 1, vals
        return str(vals[0])
    if op == "one":
        assert len(vals) == 1, vals
        return fmt1(vals[0], fmt)
    if op == "inv":
        assert len(vals) == 1, vals
        return fmt1(1 / vals[0], fmt)
    if op == "mean":
        return fmt1(sum(vals) / len(vals), fmt)
    if op == "inv_mean":
        return fmt1(len(vals) / sum(vals), fmt)
    if op == "ratio":
        assert len(vals) == 2, vals
        return fmt1(vals[0] / vals[1], fmt)
    if op == "max_kb2gb":
        return fmt1(max(vals) * 1024 / 1e9, fmt)
    if op == "pct_vs_first":
        w = vals[:-1]
        assert len(w) % 2 == 0, vals
        n = len(w) // 2
        return fmt1((sum(w[n:]) / sum(w[:n]) - 1) * 100, fmt)
    if op in ("rtf_mean", "inv_rtf_mean"):
        assert len(vals) >= 2, vals
        r = sum(vals[:-1]) / len(vals[:-1]) / 1000 / vals[-1]
        return fmt1(r if op == "rtf_mean" else 1 / r, fmt)
    if op == "kb2gb":
        assert len(vals) == 1, vals
        return fmt1(vals[0] * 1024 / 1e9, fmt)
    if op == "sumgb":
        return fmt1(sum(vals) / 1e9, fmt)
    if op in ("min", "max", "sum"):
        return fmt1({"min": min, "max": max, "sum": sum}[op](vals), fmt)
    if op == "minmax":
        spec, sep = fmt.split("|")
        return fmt1(min(vals), spec) + sep + fmt1(max(vals), spec)
    if op == "join":
        spec, sep = fmt.split("|")
        return sep.join(fmt1(v, spec) for v in vals)
    if op == "maxdelta":
        assert len(vals) % 2 == 0, vals
        return fmt1(max(vals[i] - vals[i + 1] for i in range(0, len(vals), 2)), fmt)
    raise ValueError(op)


def main():
    card = Path(sys.argv[1]) if len(sys.argv) > 1 else K / "card/README.md"
    facts = Path(sys.argv[2]) if len(sys.argv) > 2 else K / "card/facts.tsv"
    text = card.read_text()
    rows = [ln.split("\t") for ln in facts.read_text().splitlines()[1:] if ln.strip() and not ln.startswith("#")]
    bad = 0
    for r in rows:
        if len(r) < 6:
            print(f"FAIL short row {r}")
            bad += 1
            continue
        n, quote, printed, sources, op, fmt = r[:6]
        why = []
        if quote not in text:
            why.append("quote not in the card")
        if printed not in quote:
            why.append("printed value not in the quote")
        try:
            per = [values(s) for s in sources.split(" ; ")]
            vals = [v for vs in per for v in vs]
            got = apply(op, vals, fmt, per)
            if got != printed:
                why.append(f"primary gives {got!r}")
        except Exception as e:  # noqa: BLE001
            why.append(f"source error {type(e).__name__}: {e}")
        print(("PASS " if not why else "FAIL ") + f"{n} {printed!r}" + ("" if not why else " — " + "; ".join(why)))
        bad += bool(why)
    print(f"facts_check: {len(rows) - bad} / {len(rows)} PASS")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
