#!/usr/bin/env python3
"""Section-level check of an externalized .litertlm against the file it was built from.

  sections_check.py --orig ORIG.litertlm --new NEW.litertlm --results results.tsv
                    [--added-executor-metadata EXEC.pbtext] --out sections.md [--json sections.json]

Reads both container headers with litert_lm_builder's header schema (the reader litertlm_peek uses), hashes every
section's raw bytes [begin, end) and checks:
  1. every non-TFLite section of ORIG is in NEW with the same raw bytes and items, in the same order; with
     --added-executor-metadata, NEW may carry one ExecutorMetadataProto section ORIG does not have, right after
     ORIG's LlmMetadataProto section (or first when there is none), whose bytes equal that pbtext serialized;
  2. section order: NEW = ORIG's sequence (plus the added section), then TFLiteWeights sections in the results
     order of the rows that carry a weights file;
  3. every TFLite section of NEW is byte-equal to the file its results row names: the tool's model.tflite and
     weights blob for an externalized row, ORIG's own section for a row kept inline (empty weights_path);
  4. byte accounting per model_type and for the whole file (written, not a check).
Prints and writes `VERDICT: PASS` or `VERDICT: FAIL (<reasons>)`. Exit 0 = PASS, 1 = FAIL, 2 = unreadable input.
Model types: the header items say `tf_lite_<x>`, the TOML / manifest / results say `<x>`; compared without the prefix.
"""
import argparse
import csv
import hashlib
import json
import os
import struct
import sys

from litert_lm_builder import litertlm_core, litertlm_peek
from litert_lm_builder import litertlm_header_schema_py_generated as H

TFLITE = ("TFLiteModel", "TFLiteWeights")
ADDED_TYPE = "ExecutorMetadataProto"


def sha_range(f, b, e):
    h = hashlib.sha256()
    f.seek(b)
    left = e - b
    while left:
        c = f.read(min(1 << 24, left))
        if not c:
            raise IOError("short read")
        h.update(c)
        left -= len(c)
    return h.hexdigest()


_SHA = {}


def sha_file(p):
    p = os.path.realpath(p)
    if p not in _SHA:
        with open(p, "rb") as f:
            _SHA[p] = sha_range(f, 0, os.path.getsize(p))
    return _SHA[p]


def mt_norm(mt):
    if mt and mt.startswith("tf_lite_"):
        return mt[len("tf_lite_"):]
    return mt


def read(path):
    with open(path, "rb") as f:
        magic = f.read(8)
        ver = struct.unpack("<III", f.read(12))
        f.seek(24)
        hend = struct.unpack("<Q", f.read(8))[0]
        f.seek(32)
        md = H.LiteRTLMMetaData.GetRootAs(f.read(hend - 32), 0)
        sysm = md.SystemMetadata()
        sysmeta = [litertlm_peek.kvp_to_dict(sysm.Entries(j)) for j in range(sysm.EntriesLength())] if sysm else []
        sm = md.SectionMetadata()
        secs = []
        for i in range(sm.ObjectsLength()):
            o = sm.Objects(i)
            items = [litertlm_peek.kvp_to_dict(o.Items(j)) for j in range(o.ItemsLength())]
            secs.append(dict(i=i, type=litertlm_core.any_section_data_type_to_string(o.DataType()),
                             model_type=litertlm_peek.get_model_type(o), begin=o.BeginOffset(), end=o.EndOffset(),
                             bytes=o.EndOffset() - o.BeginOffset(), sha=sha_range(f, o.BeginOffset(), o.EndOffset()),
                             items=items))
    return dict(path=os.path.abspath(path), magic=magic.decode(errors="replace"), ver=".".join(map(str, ver)),
                hend=hend, size=os.path.getsize(path), sysmeta=sysmeta, secs=secs, sha=sha_file(path))


def results(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def items_s(items):
    return "; ".join(f"{x['key']}={x['value']}" for x in items) or "—"


def num(v):
    return "—" if v is None else f"{v:,}"


def padding(X):
    secs = sorted(X["secs"], key=lambda s: s["begin"])
    first_gap = secs[0]["begin"] - X["hend"] if secs else 0
    gaps = sum(secs[k + 1]["begin"] - secs[k]["end"] for k in range(len(secs) - 1))
    tail = X["size"] - secs[-1]["end"] if secs else 0
    return first_gap, gaps, tail


def label(s):
    return f"{s['type']}[{mt_norm(s['model_type'])}]" if s["model_type"] else s["type"]


def expected_added_bytes(pbtext):
    from google.protobuf import text_format
    from litert_lm_builder.runtime.proto import executor_metadata_pb2
    msg = executor_metadata_pb2.ExecutorMetadata()
    text_format.Parse(open(pbtext, encoding="utf-8").read(), msg)
    return msg.SerializeToString()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--added-executor-metadata", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    try:
        I, X = read(a.orig), read(a.new)
        rows = results(a.results)
    except Exception as e:  # unreadable header or results file
        print(f"ERROR: {type(e).__name__}: {e}")
        return 2
    why = []
    rel = os.path.dirname(os.path.abspath(a.out))

    # 1. non-TFLite sections, with the one allowed addition
    In = [s for s in I["secs"] if s["type"] not in TFLITE]
    Xn = [s for s in X["secs"] if s["type"] not in TFLITE]
    added = None
    if a.added_executor_metadata:
        if any(s["type"] == ADDED_TYPE for s in In):
            why.append(f"--added-executor-metadata given but ORIG already has {ADDED_TYPE}")
        cand = [s for s in Xn if s["type"] == ADDED_TYPE]
        if len(cand) != 1:
            why.append(f"NEW has {len(cand)} {ADDED_TYPE} sections, expected 1 (added)")
        else:
            added = cand[0]
            want = hashlib.sha256(expected_added_bytes(a.added_executor_metadata)).hexdigest()
            if added["sha"] != want:
                why.append(f"added {ADDED_TYPE} bytes differ from the pbtext serialized")
            pos_in = next((k for k, s in enumerate(In) if s["type"] == "LlmMetadataProto"), -1) + 1
            if Xn.index(added) != pos_in:
                why.append(f"added {ADDED_TYPE} at non-TFLite position {Xn.index(added)}, expected {pos_in}")
            Xn = [s for s in Xn if s is not added]
    elif any(s["type"] == ADDED_TYPE for s in Xn) and not any(s["type"] == ADDED_TYPE for s in In):
        why.append(f"NEW has an {ADDED_TYPE} section that ORIG lacks (no --added-executor-metadata)")
    L = ["# Sections: original vs externalized (`sections_check.py`; raw section bytes from the headers)", ""]
    for lab, F in (("ORIG", I), ("NEW", X)):
        L.append(f"- {lab}: `{F['path']}` {F['size']:,} B, sha256 `{F['sha']}`, magic {F['magic']}, format version "
                 f"{F['ver']}, header end {F['hend']:,} B, sections {len(F['secs'])}")
    L += ["", "## Every section", "",
          "| file | section | data type | model_type | items | begin | end | bytes | raw sha256 |",
          "|---|---:|---|---|---|---:|---:|---:|---|"]
    for lab, F in (("ORIG", I), ("NEW", X)):
        for s in F["secs"]:
            L.append(f"| {lab} | {s['i']} | {s['type']} | {s['model_type'] or '—'} | {items_s(s['items'])} | "
                     f"{s['begin']:,} | {s['end']:,} | {s['bytes']:,} | `{s['sha']}` |")
    L += ["", "## Non-TFLite sections, paired in order", "",
          "| ORIG section | NEW section | data type | ORIG bytes | NEW bytes | raw sha256 equal | items equal |",
          "|---:|---:|---|---:|---:|---|---|"]
    n_ok = 0
    for k in range(max(len(In), len(Xn))):
        x = In[k] if k < len(In) else None
        y = Xn[k] if k < len(Xn) else None
        if x and y:
            eq = x["sha"] == y["sha"] and x["type"] == y["type"]
            ieq = x["items"] == y["items"]
            n_ok += eq and ieq
            L.append(f"| {x['i']} | {y['i']} | {x['type']}{'' if x['type'] == y['type'] else ' / ' + y['type']} "
                     f"| {x['bytes']:,} | {y['bytes']:,} | {'YES' if eq else '**NO**'} | {'YES' if ieq else '**NO**'} |")
        else:
            z = x or y
            L.append(f"| {x['i'] if x else '—'} | {y['i'] if y else '—'} | {z['type']} | {num(x['bytes'] if x else None)} "
                     f"| {num(y['bytes'] if y else None)} | **unpaired** | — |")
    if added:
        L.append(f"| — | {added['i']} | {ADDED_TYPE} (added) | — | {added['bytes']:,} | = `{os.path.relpath(a.added_executor_metadata, rel)}` serialized | — |")
    L += ["", f"- non-TFLite sections: ORIG {len(In)}, NEW {len(Xn)}{' + 1 added' if added else ''}; identical pairs {n_ok}"]
    if not (len(In) == len(Xn) == n_ok):
        why.append(f"non-TFLite sections identical {n_ok} of ORIG {len(In)} / NEW {len(Xn)}")

    # 2. order
    iseq = [(s["type"], mt_norm(s["model_type"])) for s in I["secs"]]
    xseq = [(s["type"], mt_norm(s["model_type"])) for s in X["secs"] if s is not added]
    want_w = [("TFLiteWeights", r["model_type"]) for r in rows if r["weights_path"]]
    prefix_ok = xseq[:len(iseq)] == iseq
    suffix_ok = xseq[len(iseq):] == want_w
    L += ["", f"- order ORIG: {' · '.join(label(s) for s in I['secs'])}",
          f"- order NEW: {' · '.join(label(s) + (' (added)' if s is added else '') for s in X['secs'])}",
          f"- NEW starts with ORIG's sequence: {'YES' if prefix_ok else '**NO**'}; the rest = TFLiteWeights in the results "
          f"order of the rows with a weights file ({', '.join(m for _, m in want_w) or 'none'}): "
          f"{'YES' if suffix_ok else '**NO**'}"]
    if any(t == "TFLiteWeights" for t, _ in iseq):
        why.append("ORIG already has TFLiteWeights sections")
    if not prefix_ok:
        why.append("order: ORIG's sequence is not NEW's prefix")
    if not suffix_ok:
        why.append("order: TFLiteWeights suffix differs from the results rows")

    # 3. identity of every TFLite section + 4. per model_type bytes
    im = {mt_norm(s["model_type"]): s for s in I["secs"] if s["type"] == "TFLiteModel"}
    xm = {mt_norm(s["model_type"]): s for s in X["secs"] if s["type"] == "TFLiteModel"}
    xw = {mt_norm(s["model_type"]): s for s in X["secs"] if s["type"] == "TFLiteWeights"}
    rmap = {r["model_type"]: r for r in rows}
    if set(rmap) != set(im):
        why.append(f"results rows {sorted(rmap)} != ORIG TFLiteModel sections {sorted(im)}")
    L += ["", "## TFLite sections: what each NEW section is, and bytes per model_type", "",
          "| model_type | ORIG TFLiteModel | NEW TFLiteModel | NEW TFLiteWeights | NEW model + weights | (model + weights) − ORIG "
          "| NEW TFLiteModel section is | NEW TFLiteWeights section is |",
          "|---|---:|---:|---:|---:|---:|---|---|"]
    tot = [0, 0, 0]
    per_mt = []
    for mt in sorted(set(im) | set(xm) | set(xw), key=lambda m: (im.get(m) or xm.get(m) or xw.get(m))["i"]):
        x, y, w = im.get(mt), xm.get(mt), xw.get(mt)
        xb, yb, wb = (x or {}).get("bytes", 0), (y or {}).get("bytes", 0), (w or {}).get("bytes", 0)
        tot[0] += xb; tot[1] += yb; tot[2] += wb
        r = rmap.get(mt)
        ext_row = bool(r and r["weights_path"])
        if y is None:
            what_m = "**missing**"; why.append(f"{mt}: no TFLiteModel section in NEW")
        elif not ext_row:
            if x is not None and y["sha"] == x["sha"]:
                what_m = "= ORIG's section (kept inline)"
            else:
                what_m = "**other bytes**"; why.append(f"{mt}: kept-inline row but the section differs from ORIG's")
        elif os.path.exists(r["model_path"]) and sha_file(r["model_path"]) == y["sha"]:
            what_m = f"= `{os.path.relpath(r['model_path'], rel)}`"
        else:
            what_m = "**other bytes**"; why.append(f"{mt}: model section differs from the results row's model_path")
        if w is None:
            what_w = "—"
            if ext_row:
                why.append(f"{mt}: results row has a weights file but NEW has no TFLiteWeights section")
        elif ext_row and os.path.exists(r["weights_path"]) and sha_file(r["weights_path"]) == w["sha"]:
            what_w = f"= `{os.path.relpath(r['weights_path'], rel)}`"
        else:
            what_w = "**other bytes**"; why.append(f"{mt}: weights section differs from the results row's weights_path")
        L.append(f"| {mt} | {num(xb)} | {num(yb)} | {num(wb) if w else '—'} | {num(yb + wb)} | {yb + wb - xb:+,} "
                 f"| {what_m} | {what_w} |")
        per_mt.append(dict(model_type=mt, orig_model=xb, new_model=yb, new_weights=wb if w else None,
                           kept_inline=not ext_row, model_is=what_m, weights_is=what_w))
    L.append(f"| **all TFLite** | {tot[0]:,} | {tot[1]:,} | {tot[2]:,} | {tot[1] + tot[2]:,} | "
             f"{tot[1] + tot[2] - tot[0]:+,} | | |")

    # whole-file accounting
    L += ["", "## Whole-file byte accounting", "", "| quantity | ORIG | NEW | NEW − ORIG |", "|---|---:|---:|---:|"]
    frows = [("TFLiteModel section bytes", tot[0], tot[1]), ("TFLiteWeights section bytes", 0, tot[2]),
             ("TFLiteModel + TFLiteWeights", tot[0], tot[1] + tot[2]),
             ("non-TFLite section bytes", sum(s["bytes"] for s in In), sum(s["bytes"] for s in Xn) + (added["bytes"] if added else 0)),
             ("header (0 .. header end)", I["hend"], X["hend"])]
    pi, px = padding(I), padding(X)
    frows += [("padding header end → first section", pi[0], px[0]),
              ("padding between sections", pi[1], px[1]),
              ("bytes after the last section", pi[2], px[2]), ("file size", I["size"], X["size"])]
    for name, x, y in frows:
        L.append(f"| {name} | {x:,} | {y:,} | {y - x:+,} |")
    chk_i = I["hend"] + sum(pi) + sum(s["bytes"] for s in I["secs"])
    chk_x = X["hend"] + sum(px) + sum(s["bytes"] for s in X["secs"])
    L.append(f"| check: header + padding + sections = file size | {'OK' if chk_i == I['size'] else chk_i} | "
             f"{'OK' if chk_x == X['size'] else chk_x} | |")
    if chk_x != X["size"]:
        why.append("NEW: header + padding + sections != file size")
    L += ["", "## System metadata", ""]
    for lab, F in (("ORIG", I), ("NEW", X)):
        L.append(f"- {lab}: " + "; ".join(f"{x['key']}={x['value']}" for x in F["sysmeta"]))
    verdict = "VERDICT: PASS" if not why else "VERDICT: FAIL (" + "; ".join(why) + ")"
    L += ["", verdict]
    text = "\n".join(L) + "\n"
    with open(a.out, "w", encoding="utf-8") as fo:
        fo.write(text)
    if a.json:
        def strip(F):
            return {k: v for k, v in F.items()}
        with open(a.json, "w", encoding="utf-8") as fo:
            json.dump(dict(orig=strip(I), new=strip(X), added=added["i"] if added else None, per_model_type=per_mt,
                           verdict="PASS" if not why else "FAIL", reasons=why), fo, indent=1)
    print(verdict)
    return 0 if not why else 1


if __name__ == "__main__":
    sys.exit(main())
