#!/usr/bin/env python3
"""The provider's own checks and sequences over the question set, without the weights: every record and control arm of
set/requests.json / set/red_arms.json goes through d1-2B's `SystemOne` question checks (`named`) and sequence builder
(`_sequences`, set packing, media processor) via seqs_only.py, and every gold read is parsed with the provider's own
parser. Writes set/token_counts.json: per record and arm the provider's sequence lengths (run_gate.py orders its
records by them, make_subsets.py picks the records of the 128-token graphs by them).

Run from the work directory with $REF:
    $REF conversion/check_set.py
Exit status 1 when a check fails (the failures are printed and kept in token_counts.json).

Checks:
- media files match media.json's SHA-256;
- kind refused: the provider refuses the request; every other record and arm: it accepts every question;
- kind text / image: no media / media; every sequence <= 4096 tokens, except wl_5k_state (state_len 5,000 +- 100) and
  kind long_refused (longer than GRAPH_MAX tokens);
- gold: option names in the criteria, a read's gold text inside the rendered state and parsed by the provider's
  parse_number / parse_date to the gold value, estimates inside their range, box edges ordered inside [0, 1], boxes in
  reading order and within the blocks read;
- control arms: the base record exists and the watched question is in both requests.
"""
from __future__ import annotations

import collections
import hashlib
import importlib
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from seqs_only import provider_engine, sequences  # noqa: E402  (sets the HF environment first)

LIMIT, GRAPH_MAX = 4096, 8192
WL5K, WL5K_LOW, WL5K_HIGH = "wl_5k_state", 4900, 5100


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def request_sha256(request):
    return hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def qtype(q):
    t = q.get("type", "choice")
    if t == "choice" and q.get("multiple"):
        return "multiple"
    if t == "number" and "range" in q:
        return "estimate"
    return t


def main():
    os.chdir(WORK)
    with open("set/requests.json", encoding="utf-8") as f:
        req_doc = json.load(f)
    with open("set/media.json", encoding="utf-8") as f:
        media = json.load(f)
    with open("set/red_arms.json", encoding="utf-8") as f:
        arms = json.load(f)
    engine = provider_engine()
    pkg = type(engine).__module__.rsplit(".", 1)[0]
    prompt = importlib.import_module(pkg + ".prompt")
    parsing = importlib.import_module(pkg + ".parsing")
    tok = engine.tokenizer
    errors = []

    media_sha = {}
    for mid, m in media.items():
        got = sha256_file(m["path"])
        media_sha[mid] = got
        if m.get("sha256") and got != m["sha256"]:
            errors.append(f"media {mid}: SHA-256 {got} != media.json")
        if m.get("bytes") is not None and os.path.getsize(m["path"]) != m["bytes"]:
            errors.append(f"media {mid}: bytes differ from media.json")

    def run(rid, req):
        paths = [media[m]["path"] for m in req.get("media") or []]
        try:
            seqs = sequences(engine, req.get("state"), req["questions"], paths)
            return seqs, None
        except Exception as e:  # a refusal is a result
            return None, f"{type(e).__name__}: {e}"

    def row_of(rid, rec_kind, req, seqs, err):
        row = {"request_sha256": request_sha256(req), "kind": rec_kind, "questions": len(req["questions"]),
               "error": err, "lengths": None, "longest": None, "state_lens": None, "grids": None}
        if seqs is not None:
            row.update(lengths=[len(s["ids"]) for s in seqs], longest=max(len(s["ids"]) for s in seqs),
                       state_lens=[s["state_len"] for s in seqs],
                       grids=[s["image_grid_thw"] for s in seqs if s["image_grid_thw"]][:1] or None)
        return row

    def check_limits(rid, rec_kind, req, row):
        if rec_kind == "refused":
            if row["error"] is None:
                errors.append(f"{rid}: kind refused but the provider accepts it")
            return
        if row["error"] is not None:
            errors.append(f"{rid}: the provider refuses it: {row['error']}")
            return
        if rec_kind in ("text", "image") and (rec_kind == "image") != bool(req.get("media")):
            errors.append(f"{rid}: kind {rec_kind} but media {req.get('media')}")
        if rid == WL5K:
            s = row["state_lens"][0]
            if not WL5K_LOW <= s <= WL5K_HIGH:
                errors.append(f"{rid}: state_len {s} outside {WL5K_LOW}..{WL5K_HIGH}")
            if row["longest"] > GRAPH_MAX:
                errors.append(f"{rid}: {row['longest']} tokens, above {GRAPH_MAX}")
        elif rec_kind == "long_refused":
            if row["longest"] <= GRAPH_MAX:
                errors.append(f"{rid}: kind long_refused but {row['longest']} tokens fit the {GRAPH_MAX} graph")
        elif row["longest"] > LIMIT:
            errors.append(f"{rid}: {row['longest']} tokens, above {LIMIT}")
        want_seqs = math.ceil(len(req["questions"]) / engine.set_size)
        if len(row["lengths"]) != want_seqs:
            errors.append(f"{rid}: {len(row['lengths'])} sequences, expected {want_seqs}")

    def check_gold(rec):
        rid, req, gold = rec["id"], rec["request"], rec.get("gold")
        if not isinstance(gold, dict) or rec["kind"] == "refused":
            return
        if set(gold) != set(req["questions"]):
            errors.append(f"{rid}: gold keys differ from the question ids")
        context = prompt.state_text(req["state"]) if req.get("state") is not None else ""
        for qid, q in req["questions"].items():
            g, t = gold.get(qid), qtype(q)
            if g is None:
                continue
            where = f"{rid}/{qid}"
            if t == "noul" and g not in ("true", "false"):
                errors.append(f"{where}: noul gold {g!r}")
            elif t == "choice" and g not in q["criteria"]:
                errors.append(f"{where}: choice gold {g!r} not an option")
            elif t == "score" and not (isinstance(g, str) and g.isdigit() and int(g) < len(q["criteria"])):
                errors.append(f"{where}: score gold {g!r}")
            elif t == "multiple":
                names = list(q["criteria"])
                if not set(g["choices"]) <= set(names) or g["choices"] != [n for n in names if n in g["choices"]]:
                    errors.append(f"{where}: multiple gold {g} not options in criteria order")
            elif t in ("text", "number", "date"):
                if g.get("found") is False:
                    continue
                span = g["text"]
                if span not in context:
                    errors.append(f"{where}: gold text {span!r} is not in the rendered state")
                    continue
                if t == "number":
                    got = parsing.parse_number(span, context, unit=q.get("unit"))
                    if got != g["number"] or type(got) is not type(g["number"]):
                        errors.append(f"{where}: parse_number({span!r}) = {got!r}, gold {g['number']!r}")
                elif t == "date":
                    got, ambiguous = parsing.parse_date(span, context)
                    if got != g["date"]:
                        errors.append(f"{where}: parse_date({span!r}) = {got!r} (ambiguous {ambiguous}), gold "
                                      f"{g['date']!r}")
            elif t == "estimate":
                low, high = q["range"]
                if not low <= g["number"] <= high:
                    errors.append(f"{where}: estimate gold outside the range")
            elif t == "box":
                b = g.get("box")
                if b is not None and not (0 <= b["top"] < b["bottom"] <= 1 and 0 <= b["left"] < b["right"] <= 1):
                    errors.append(f"{where}: box gold {b}")
            elif t == "boxes":
                blocks = q.get("max", 16)
                if g["count"] > blocks:
                    errors.append(f"{where}: boxes gold count above the blocks read")
                bs = g.get("boxes")
                if bs is not None:
                    if len(bs) != g["count"] or bs != sorted(bs, key=lambda b: (b["left"], b["top"])):
                        errors.append(f"{where}: boxes gold not {g['count']} boxes in reading order")
                    if g.get("at_least") and not g.get("instances", 0) > blocks:
                        errors.append(f"{where}: at_least without more instances than blocks")

    rows, by_kind_longest = {}, collections.defaultdict(lambda: (0, None))
    readouts = collections.Counter()
    for rec in req_doc["records"]:
        req = rec["request"]
        seqs, err = run(rec["id"], req)
        row = row_of(rec["id"], rec["kind"], req, seqs, err)
        rows[rec["id"]] = row
        check_limits(rec["id"], rec["kind"], req, row)
        check_gold(rec)
        if row["longest"] is not None and row["longest"] > by_kind_longest[rec["kind"]][0]:
            by_kind_longest[rec["kind"]] = (row["longest"], rec["id"])
        if err is None:
            for q in req["questions"].values():
                if qtype(q) == "estimate":
                    slot = prompt.question_slots(prompt.as_question(q), tok)[0]
                    readouts[f"{type(slot).__name__}:{slot.grid.size if hasattr(slot, 'grid') else len(slot.criteria)}"] += 1
    for name, want in (("own_long_15k_001", None), ("own_long_34k_001", None)):
        rec = next(r for r in req_doc["records"] if r["id"] == name)
        m = re.search(r"sequence for d1-2B is (\d+) tokens", rec["note"])
        if m and int(m[1]) != rows[name]["longest"]:
            errors.append(f"{name}: the note says {m[1]} tokens, the provider builds {rows[name]['longest']}")

    by_id = {r["id"]: r for r in req_doc["records"]}
    arm_rows = {}
    for a in arms:
        seqs, err = run(a["id"], a["request"])
        row = row_of(a["id"], a["kind"], a["request"], seqs, err)
        arm_rows[a["id"]] = row
        check_limits(a["id"], a["kind"], a["request"], row)
        base = by_id.get(a["base_id"])
        if base is None:
            errors.append(f"arm {a['id']}: base {a['base_id']} is not a record")
            continue
        qid = a["watch"]["qid"]
        if qid not in a["request"]["questions"] or qid not in base["request"]["questions"]:
            errors.append(f"arm {a['id']}: watched question {qid} missing")
        if a["request"] == base["request"]:
            errors.append(f"arm {a['id']}: the request equals its base")

    doc = {
        "what": "the provider's sequences for every record and red arm of set/ (check_set.py: seqs_only.py, "
                "d1-2B's SystemOne over a stand-in model, no weights): per record the request's SHA-256 (compact sorted "
                "JSON), each sequence's length and state_len, the first picture's grid, or the refusal",
        "engine": {"set_size": engine.set_size, "attention": engine.attention, "max_pixels": engine.max_pixels,
                   "d1_codes": tok.d1_codes, "d1_numbers": tok.d1_numbers, "d1_prompt": tok.d1_prompt,
                   "lead": engine.lead},
        "media_sha256": media_sha,
        "estimate_readouts": dict(sorted(readouts.items())),
        "longest_by_kind": {k: {"tokens": v[0], "record": v[1]} for k, v in sorted(by_kind_longest.items())},
        "errors": errors,
        "records": rows,
        "arms": arm_rows,
    }
    with open(os.path.join(WORK, "set", "token_counts.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    refused = {rid: r["error"] for rid, r in rows.items() if r["kind"] == "refused"}
    print(json.dumps({"records": len(rows), "arms": len(arm_rows), "longest_by_kind": doc["longest_by_kind"],
                      "arms_longest": max(r["longest"] or 0 for r in arm_rows.values()),
                      "estimate_readouts": doc["estimate_readouts"], "refused": refused,
                      "wl_5k_state": rows.get(WL5K, {}).get("state_lens"), "errors": len(errors)}, ensure_ascii=False,
                     indent=1))
    for e in errors:
        print("ERROR", e)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
