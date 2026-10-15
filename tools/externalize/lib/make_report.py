#!/usr/bin/env python3
"""REPORT.md for one externalize_repo.sh run, from the run's work directory.

  make_report.py --workdir W --out REPORT.md

Reads W/report/run.tsv (key TAB value, written by externalize_repo.sh as it goes; a later line wins),
W/report/sections.json (sections_check.py), W/report/breakdown.json (tflite_breakdown.py), W/report/smoke.tsv and
W/report/smoke.out (the smoke reply, verbatim), W/report/commands.txt, and the tool's print lines in
W/logs/externalize_<model_type>.log. Package versions are those of the interpreter running this script (the same one
that ran the tools). Missing pieces are written as "not run".
"""
import argparse
import glob
import json
import os
import re
import sys
from importlib import metadata

PRINT = re.compile(r"^(Writing weight file|Externalized tensors|Cleared buffers|Original model size|Processed model size|"
                   r"Processed total weight files size)")


def load_tsv(path):
    d = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            k, _, v = line.rstrip("\n").partition("\t")
            if k:
                d[k] = v
    return d


def load_json(path):
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None


def pkg(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def mib(n):
    return f"{int(n) / 1048576:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    W = a.workdir
    R = load_tsv(os.path.join(W, "report", "run.tsv"))
    sec = load_json(os.path.join(W, "report", "sections.json"))
    brk = load_json(os.path.join(W, "report", "breakdown.json"))
    smk = load_tsv(os.path.join(W, "report", "smoke.tsv"))
    g = R.get
    L = [f"# {g('repo', '?')} · {g('file', '?')} — weights moved to TFLiteWeights sections", ""]
    L += [f"- Result: **{g('result', 'not finished')}**. Checks: verify-peek {g('verify_peek', 'not run')}, sections "
          f"{g('sections_check', 'not run')}, tensors {g('breakdown', 'not run')}. Smoke: {smk.get('verdict', 'not run')}.",
          f"- Output: `{g('out_name', '?')}` {int(g('out_size', 0) or 0):,} B, sha256 `{g('out_sha256', '?')}` "
          f"(staged at `{g('staged_path', '—')}`).",
          "- The builder writes a new uuid and creation_timestamp into the container header at every pack, so the file's "
          "sha256 changes from build to build; the section table below says which sections equal the original's bytes.",
          ""]
    L += ["## Input", "",
          f"- Hub: `{g('repo')}` revision `{g('revision')}` (asked: `{g('revision_asked', 'main')}`), `{g('file')}`: size {g('hub_size')}, LFS size "
          f"{g('hub_lfs_size')}, LFS sha256 `{g('hub_sha256')}` (model info `{g('api_json')}`, read {g('api_read_at')}).",
          f"- Local: `{g('int_path')}` ({g('source_kind')}): {int(g('int_size', 0) or 0):,} B, sha256 `{g('int_sha256')}` → "
          f"{g('input_check', 'not checked')}.",
          f"- Options: keep inline at or under {g('keep_inline_under_mib', '0')} MiB; executor metadata "
          f"{g('executor_metadata', 'off')}; smoke {g('smoke_mode', '?')}.", ""]
    if sec:
        I, X = sec["orig"], sec["new"]
        orig_sha = {s["sha"]: s["i"] for s in I["secs"]}
        L += ["## Sections (original → externalized)", "",
              f"- Original: format {I['ver']}, {len(I['secs'])} sections, {I['size']:,} B. Externalized: format {X['ver']}, "
              f"{len(X['secs'])} sections, {X['size']:,} B ({X['size'] - I['size']:+,} B).", "",
              "| section | data type | model_type | bytes | raw sha256 | equals original section |",
              "|---:|---|---|---:|---|---|"]
        for s in X["secs"]:
            same = orig_sha.get(s["sha"])
            L.append(f"| {s['i']} | {s['type']} | {s['model_type'] or '—'} | {s['bytes']:,} | `{s['sha']}` | "
                     f"{'section ' + str(same) if same is not None else ('added' if s['i'] == sec.get('added') else 'new bytes')} |")
        L += ["", f"- sections_check: **{sec['verdict']}**" + (f" — {'; '.join(sec['reasons'])}" if sec["reasons"] else
              " (non-TFLite sections raw-identical, original order kept, TFLiteWeights at the end in results order, "
              "every TFLite section = the file its results row names)"), ""]
    if brk:
        L += ["## TFLite sections", "",
              "| model_type | form | original bytes | graph after (MiB) | weights section (MiB) | externalized tensors | "
              "tensor check | graph over the mark |", "|---|---|---:|---:|---:|---:|---|---|"]
        for r in brk["sections"]:
            chk = ("PASS" if not r["reasons"] else "**FAIL**") if r["form"] == "externalized" else "—"
            L.append(f"| {r['model_type']} | {r['form']} | {r['orig_bytes']:,} | {r['new_graph_bytes']:,} ({mib(r['new_graph_bytes'])}) "
                     f"| {r['blob_bytes']:,} ({mib(r['blob_bytes'])}) | {r['ext_tensors']:,} | {chk} | "
                     f"{'**yes**' if r['graph_over_mark'] else 'no'} |")
        L += ["", f"- Mark = {brk['graph_mark_bytes']:,} B (50 MiB). Tensor check = every externalized tensor's bytes in the "
              "weights section equal the original buffer, every other tensor keeps its original buffer, original payload = "
              "kept + cleared, group `tflite_weights`, 64-byte aligned, weights size = highest offset + length.", ""]
        L += ["### Externalized tensors by dtype", "",
              "| model_type | dtype | tensors | unique bytes | MiB | consumer op @ input |", "|---|---|---:|---:|---:|---|"]
        for r in brk["sections"]:
            for d, v in sorted(r["by_dtype"].items(), key=lambda kv: -kv[1]["unique_bytes"]):
                c = ", ".join(f"{k} ({n})" for k, n in v["consumers"].items())
                L.append(f"| {r['model_type']} | {d} | {v['tensors']:,} | {v['unique_bytes']:,} | {mib(v['unique_bytes'])} | {c} |")
        fl = [(r["model_type"], f) for r in brk["sections"] for f in r["float_ext"]]
        L += ["", f"- FLOAT_EXTERNAL={'yes' if brk['float_external'] else 'no'}"
              + (" — float tensors in a weights section (WebGPU's streamed loading takes INT8 / INT4 / INT2 external "
                 "weights only):" if fl else "")]
        agg = {}
        for mt, f in fl:
            k = (mt, f["dtype"], ", ".join(f["consumers"]))
            n, b = agg.get(k, (0, 0))
            agg[k] = (n + 1, b + f["bytes"])
        for (mt, d, c), (n, b) in agg.items():
            L.append(f"  - {mt}: {n} {d} tensors, {c}, {b:,} B (per-tensor sum)")
        L.append("")
    tp = sorted(glob.glob(os.path.join(W, "logs", "externalize_*.log")))
    if tp:
        L += ["### The tool's print, per section", ""]
        for p in tp:
            mt = os.path.basename(p)[len("externalize_"):-len(".log")]
            lines = [x.rstrip() for x in open(p, encoding="utf-8", errors="replace") if PRINT.match(x)]
            rss = [x.split()[0] for x in open(p, encoding="utf-8", errors="replace") if "maximum resident set size" in x]
            ex = [x.strip() for x in open(p, encoding="utf-8", errors="replace") if x.startswith("### exit")]
            L.append(f"- {mt}: " + "; ".join(lines + [f"peak RSS {int(rss[0]):,} B" if rss else ""] + ex).strip("; "))
        L.append("")
    L += ["## Smoke", ""]
    if smk:
        L += [f"- Command: `{smk.get('command', '?')}`", f"- Exit {smk.get('rc', '?')} after {smk.get('seconds', '?')} s; "
              f"verdict **{smk.get('verdict', '?')}** (a person reads the reply; empty, \"An error occurred\" or a non-zero "
              "exit is FAILED).", "- Reply (stdout, verbatim):", "", "```"]
        out = os.path.join(W, "report", "smoke.out")
        L += [open(out, encoding="utf-8", errors="replace").read().rstrip("\n") if os.path.exists(out) else "(none)", "```", ""]
    else:
        L += ["- not run (`--no-smoke`; run `externalize_repo.sh … --smoke-only` inside a quiet window).", ""]
    L += ["## Versions", "",
          f"- LiteRT tools: `{g('tools_path')}` at commit `{g('tools_commit')}`; externalize_tflite_flatbuffer.py sha256 "
          f"`{g('tool_sha_flatbuffer')}`, externalize_litertlm_sections.py sha256 `{g('tool_sha_sections')}` "
          f"(pinned: {g('tools_pinned')}).",
          f"- Python `{g('python')}` {sys.version.split()[0]}: " + ", ".join(
              f"{n} {v}" for n in ("ai-edge-litert", "ai-edge-litert-nightly", "flatbuffers", "numpy", "protobuf",
                                   "litert-lm-builder") if (v := pkg(n))),
          f"- litert-lm CLI `{g('cli')}`: {g('cli_version')}; builder `{g('builder')}`; peek `{g('peek')}`; hf `{g('hf')}`.",
          f"- Host: {g('host')}. Run {g('started')} → {g('finished', '?')}.", ""]
    cmds = os.path.join(W, "report", "commands.txt")
    L += ["## Commands (exact, in order)", "", "```"]
    L += [open(cmds, encoding="utf-8").read().rstrip("\n") if os.path.exists(cmds) else "(none)", "```", ""]
    L += [f"- The TOML the file was packed from is `{g('out_name', '<file>')}.model.toml` next to this report (its data "
          f"paths point into the work directory, cleaned after the run). Logs: `{os.path.join(W, 'logs')}`; checks: "
          f"`{os.path.join(W, 'report')}` (sections.md, breakdown.md, peek_orig.txt, peek_new.txt)."]
    with open(a.out, "w", encoding="utf-8") as fo:
        fo.write("\n".join(L) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
