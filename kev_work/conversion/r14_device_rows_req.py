"""Phone inputs of the request comparison (the shared-state pair against the row graphs, per request): requests of 1, 2, 3
and 5 questions on the pair Ls128+Lq64 = tv4_000 (state 30 tokens), own_ticket_01_q2 (the first two questions of
own_ticket_01: state 109, question parts 49 / 37), own_ticket_01 and own_fiveq_09 (state 99); on the pair Ls256+Lq64 also
own_order_06 (2 questions, state 150) and own_email_03 (3 questions, state 167), whose states do not fit Ls128.

    python3 scripts/r14_device_rows_req.py        (after r14_device_rows.py --form C --pairs 128:64 256:64)

Writes into device/r14/ (never overwrites a file with other content):
  requests_Ls128_Lq64_<share|noshare>_req.json  the Ls128 pair's timing requests (tv4_000, own_ticket_01_q2,
      own_ticket_01, own_fiveq_09), copied from r14_device_rows.py's requests_Ls128_Lq64_<share|noshare>.json (same
      entries), with GPU constant tensor sharing on or off
  requests_Ls256_Lq64_<share|noshare>_req.json  the same for the Ls256 pair (tv4_000, own_order_06, own_email_03,
      own_ticket_01, own_fiveq_09); the timed legs ran its own_order_06 and own_email_03 entries
      (r18_req2.py writes that subset)
  timing_rows_r14_req.json  the same requests as rows, each row in its smallest bucket (64 / 128 / 256 ...): one
      "request" set per (request, bucket) = the request's rows of that bucket in order (the app runs the sets whose L
      is the graph's L; a request's row-graph time = the sum of its sets)
  inputs_r14_req.json  bytes and sha256 of these files and of their sources"""
import hashlib
import json
from pathlib import Path

K = Path(__file__).resolve().parents[1]
OUT = K / "device/r14"
ORACLE = K / "oracle/oracle_0.8b.json"
LS = (64, 128, 256, 512, 1024, 2048)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, doc):
    path = OUT / name
    text = json.dumps(doc, separators=(",", ":")) + "\n"
    if path.exists():
        assert path.read_text() == text, f"{path} exists with other content"
    else:
        path.write_text(text)
    return path


def main():
    oracle = json.loads(ORACLE.read_text())
    by = {f"{q['id']}/{q['qid']}": q for q in oracle["questions"]}
    out = {}
    pts = {128: ["tv4_000", "own_ticket_01_q2", "own_ticket_01", "own_fiveq_09"],
           256: ["tv4_000", "own_order_06", "own_email_03", "own_ticket_01", "own_fiveq_09"]}
    req_rows = {}
    for (Ls, want), mode in [(x, m) for x in pts.items() for m in ("share", "noshare")]:   # sharing on and off
        src = OUT / f"requests_Ls{Ls}_Lq64_{mode}.json"
        d = json.loads(src.read_text())
        reqs = {r["request"]: r for r in d["requests"]}
        sel = []
        for rid in want:
            if rid == "own_ticket_01_q2":
                base = reqs["own_ticket_01"]
                r = {**base, "request": rid, "questions": base["questions"][:2]}
            else:
                r = reqs[rid]
            assert all(len(q["ids"]) <= d["Lq"] for q in r["questions"]), rid
            assert len(r["state"]) <= Ls, (rid, len(r["state"]))
            sel.append(r)
            req_rows[rid] = [q["key"] for q in r["questions"]]
        doc = {**{k: v for k, v in d.items() if k not in ("requests", "timing_requests")},
               "timing_requests": want, "requests": sel,
               "note": f"request points (scripts/r14_device_rows_req.py) from {src.name}"}
        name = f"requests_Ls{Ls}_Lq64_{mode}_req.json"
        out[name] = (write(name, doc), len(sel), src)
    sets = []
    for rid, keys in req_rows.items():
        groups = {}
        for key in keys:
            q = by[key]
            L = next(L for L in LS if q["row_len"] <= L)
            assert len(q["row_ids"]) == q["row_len"], key
            groups.setdefault(L, []).append({"key": key, "ids": [int(x) for x in q["row_ids"]]})
        for L, rows in sorted(groups.items()):
            sets.append({"name": f"req_{rid}_L{L}", "L": L, "kind": "request", "request": rid, "rows": rows})
    protocol = ("5 warm-up requests, then 20 timed requests of the set's rows (each row = write + run + read-back of the "
                "full hidden output); a request's row-form time = the sum of its sets over the buckets")
    name = "timing_rows_r14_req.json"
    out[name] = (write(name, {"pad_id": json.loads((OUT / "rows_L128.json").read_text())["pad_id"],
                              "protocol": protocol, "sets": sets}), len(sets), ORACLE)
    doc = {"step": "request points (scripts/r14_device_rows_req.py)",
           "files": {n: {"entries": c, "bytes": p.stat().st_size, "sha256": sha256(p),
                         "source": str(s.relative_to(K)), "source_sha256": sha256(s)} for n, (p, c, s) in out.items()},
           "sets": [{"name": s["name"], "L": s["L"], "rows": [r["key"] for r in s["rows"]],
                     "row_len": [len(r["ids"]) for r in s["rows"]]} for s in sets]}
    (OUT / "inputs_r14_req.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(doc["sets"], indent=1))


if __name__ == "__main__":
    main()
