#!/usr/bin/env python3
"""Regenerate the "Converted models" table at the top of README.md from the published manifests.

Sources, all public:
  repos.txt     one Hugging Face repo per line; every repo must ship litertlm_manifest.json
  curated.json  per repo tail (without a -LiteRT suffix): task text, sort group, recipe link, and optionally
                `speed_skip`, files whose rows stay out of the speed columns (a file for a narrower use, such as
                a shorter context, whose speed the columns could not qualify; its card carries it)

Each row takes, per device class (phone / Mac), the measured row with the highest decode
tokens/s across every variant of the repo, and prints device, backend and decode speed. A
"lo-hi" spread stays a range. Nothing here is typed in by hand: the numbers come out of the
manifests, and the script refuses to write if a recipe link points at a missing card or a
REPRODUCE.md heading that no longer exists.

  python3 tools/readme_table/make_readme_table.py            # print the table
  python3 tools/readme_table/make_readme_table.py --write    # splice it into README.md, update the counts in AGENTS.md and llms.txt
  python3 tools/readme_table/make_readme_table.py --check    # exit 1 if any of the three is stale

Adding a model: append its repo to repos.txt, add its entry to curated.json, run --write.
"""
import argparse
import json
import os
import re
import sys
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
README = os.path.join(ROOT, "README.md")
AGENT_FILES = [os.path.join(ROOT, name) for name in ("AGENTS.md", "llms.txt")]
REPRODUCE = os.path.join(ROOT, "REPRODUCE.md")
START, END = "<!-- models-table:start -->", "<!-- models-table:end -->"
HEADER = ["| Model | Params | Task | Phone: decode tok/s | Mac: decode tok/s | Recipe |",
          "|---|---:|---|---|---|---|"]
MAC = re.compile(r"Apple M|macOS|\bMac\b")


def fetch(repo):
    url = f"https://huggingface.co/{repo}/resolve/main/litertlm_manifest.json"
    with urllib.request.urlopen(url, timeout=60) as r:
        return repo, json.load(r)


def lo(v):
    if isinstance(v, (int, float)):
        return float(v)
    m = re.match(r"\s*([\d.]+)", str(v))
    return float(m.group(1)) if m else -1.0


def fmt(v):
    if isinstance(v, (int, float)):
        return f"{v:.1f}"
    m = re.match(r"\s*([\d.]+)\s*[-–]\s*([\d.]+)", str(v))
    if m:
        return f"{float(m.group(1)):.1f}–{float(m.group(2)):.1f}"
    return str(v)


def device(d):
    d = re.sub(r"\s*\(.*?\)", "", d).strip()
    for p in ("Apple ", "Samsung ", "Mac "):
        d = d.replace(p, "")
    return d


def best_rows(manifest, skip=()):
    phone = mac = None
    for v in manifest["variants"]:
        if v.get("file") in skip:
            continue  # curated.json `speed_skip`
        rows = v.get("measured", [])
        is_mac = lambda r: bool(MAC.search(r.get("device", ""))) or str(r.get("os", "")).lower() == "macos"
        long_side = {side: any((r.get("decode_tokens") or 256) >= 64 and is_mac(r) == side for r in rows) for side in (True, False)}
        for r in rows:
            d = lo(r.get("decode_tps"))
            if d < 0:
                continue
            if long_side[is_mac(r)] and (r.get("decode_tokens") or 256) < 64:
                continue  # a short-prompt TTFT cell (16 prompt / 32 decode tokens) is not the decode-throughput row when a longer run exists on that side
            dev = r.get("device", "")
            cell = (d, f'{device(dev)} {r.get("backend", "?").upper()} {fmt(r["decode_tps"])}')
            if MAC.search(dev) or str(r.get("os", "")).lower() == "macos":
                mac = cell if mac is None or d > mac[0] else mac
            else:
                phone = cell if phone is None or d > phone[0] else phone
    return (phone[1] if phone else "—"), (mac[1] if mac else "—")


def slug(heading):
    h = re.sub(r"^#+\s*", "", heading).strip().lower()
    h = re.sub(r"[^\w\- ]", "", h)
    return h.replace(" ", "-")


def check_link(link, anchors):
    if link.startswith("cards/"):
        return os.path.exists(os.path.join(ROOT, link))
    if link.startswith("REPRODUCE.md#"):
        return link.split("#", 1)[1] in anchors
    return False


def build():
    repos = [l.strip() for l in open(os.path.join(HERE, "repos.txt")) if l.strip() and not l.startswith("#")]
    curated = json.load(open(os.path.join(HERE, "curated.json")))
    with open(REPRODUCE) as f:
        anchors = {slug(l) for l in f if l.startswith("#")}
    with ThreadPoolExecutor(max_workers=10) as ex:
        manifests = dict(ex.map(fetch, repos))
    rows, errors = [], []
    for repo in repos:
        tail = repo.split("/")[-1]
        name = tail[:-7] if tail.endswith("-LiteRT") else tail
        c = curated.get(name)
        if c is None:
            errors.append(f"{repo}: no curated.json entry for {name!r}")
            continue
        if not check_link(c["recipe"], anchors):
            errors.append(f"{repo}: recipe link not found: {c['recipe']}")
            continue
        m = manifests[repo]
        p = m["model"].get("parameters_b")
        phone, mac = best_rows(m, c.get("speed_skip", ()))
        label = "card" if c["recipe"].startswith("cards/") else "recipe"
        rows.append((c["group"], p or 0, name.lower(), f"| [{name}](https://huggingface.co/{repo}) | {p:.3g}B | {c['task']} | "
                                          f"{phone} | {mac} | [{label}]({c['recipe']}) |"))
    if errors:
        sys.exit("refusing to write:\n  " + "\n  ".join(errors))
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    return HEADER + [r[3] for r in rows], len(rows)


def splice(readme, table, n):
    if START not in readme or END not in readme:
        sys.exit(f"README.md has no {START} / {END} markers")
    head, rest = readme.split(START, 1)
    _, tail = rest.split(END, 1)
    head = re.sub(r"\d+ published conversions below", f"{n} published conversions below", head, count=1)
    return f"{head}{START}\n" + "\n".join(table) + f"\n{END}{tail}"


def agent_counts(path, text, table, n):
    """The counts AGENTS.md and llms.txt state, taken from the rows of the README's table."""
    orgs = Counter(re.search(r"huggingface\.co/([^/]+)/", row).group(1) for row in table[len(HEADER):])
    for pattern, value in ((r"\d+( published conversions)", n),
                           (r"\d+( in\s+\[?litert-community)", orgs["litert-community"]),
                           (r"\d+( under\s+\[?mlboydaisuke)", orgs["mlboydaisuke"])):
        text, found = re.subn(pattern, lambda m, v=value: f"{v}{m.group(1)}", text, count=1)
        if not found:
            sys.exit(f"{os.path.basename(path)} has no text matching {pattern}")
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--write", action="store_true", help="rewrite README.md in place and the counts in AGENTS.md and llms.txt")
    g.add_argument("--check", action="store_true", help="exit 1 if README.md, AGENTS.md or llms.txt differs from the generated text")
    args = ap.parse_args()
    table, n = build()
    if not (args.write or args.check):
        print("\n".join(table))
        return
    readme = open(README).read()
    new = splice(readme, table, n)
    agents = {path: open(path).read() for path in AGENT_FILES}
    counted = {path: agent_counts(path, text, table, n) for path, text in agents.items()}
    if args.check:
        if new != readme:
            sys.exit("README.md is stale: run make_readme_table.py --write")
        for path in AGENT_FILES:
            if counted[path] != agents[path]:
                sys.exit(f"{os.path.basename(path)} is stale: run make_readme_table.py --write")
        print(f"README.md, AGENTS.md and llms.txt are current ({n} rows)")
        return
    open(README, "w").write(new)
    for path, text in counted.items():
        open(path, "w").write(text)
    print(f"README.md, AGENTS.md and llms.txt updated ({n} rows)")
    import make_speed_chart  # the figure at the top of the README follows the table
    make_speed_chart.main()


if __name__ == "__main__":
    main()
