"""Rebuild the full fixture file (requests.json, 382 requests) from requests_public.json and the kev repository.

    python rebuild_requests.py --out requests.json
    python rebuild_requests.py --out requests.json --red-arms red_arms.json     (also the four control arms)
    python rebuild_requests.py --out requests.json --kev-repo path/to/kev       (an existing clone at tag kev-1.0)

requests_public.json lists the records taken from the kev repository's transfer-v4 development file by reference
(file, tag, line, SHA-256 of the line, _meta.id). This script reads that file from github.com/jaredpalmer/kev at the
commit of tag kev-1.0 (one 803,309-byte download from raw.githubusercontent.com unless --kev-repo is given), checks the
file's SHA-256 and every referenced line's SHA-256, turns each line into a fixture record exactly as the original
fixture builder did (`label` -> gold key, `label` / `src` / `_meta` moved out of the request into the provenance),
rebuilds red_arm_000 from tv4_000 with its one-word edit, and writes the full file. Every record's request must have the
request_sha256 recorded in requests_public.json (`work_request_sha256`), every rebuilt gold must equal the listed gold,
and the file must have the SHA-256 recorded there (`rebuild_target_sha256`); nothing is written otherwise.

--red-arms also restores the four control arms of red_arms_public.json: the two listed by rule are made from their base
records (an edit of the instructions, or the state of another record), each must have its recorded request_sha256, and
the file (the form the conversion scripts read as fixtures/red_arms.json) must have the recorded SHA-256.
Python 3.8+ standard library only.
"""
import argparse
import copy
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

DEV_FILE = "evals/v4/transfer-v4/development.jsonl"
DEV_SHA256 = "ff374c49c6c9f15f8a56fb274b4a4857d20497eb8dd1ac07ce01560e682a5f2e"
DEV_LINES = 764
KEV_TAG, KEV_COMMIT = "kev-1.0", "6b719c3c3f367295f6ef336f4f751cf5ff970abc"
RAW_URL = f"https://raw.githubusercontent.com/jaredpalmer/kev/{KEV_COMMIT}/{DEV_FILE}"
HERE = Path(__file__).resolve().parent


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def request_sha256(request):
    """sha256 of json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(',', ':')) in UTF-8."""
    return sha256_bytes(json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode())


def dump(doc):
    return json.dumps(doc, indent=1, ensure_ascii=False) + "\n"


def gold_key(qtype, label):
    if qtype == "noul":
        assert isinstance(label, bool), label
        return "true" if label else "false"
    if qtype == "score":
        assert isinstance(label, int) and not isinstance(label, bool), label
        return str(label)
    assert isinstance(label, str), label
    return label


def from_dev(rec_id, source, line_no, row, raw_line):
    """One development line -> one fixture record (the original builder's transform, unchanged)."""
    request = {"state": row["state"], "questions": {}}
    gold, src = {}, {}
    for qid, q in row["questions"].items():
        q = dict(q)
        label, src[qid] = q.pop("label"), q.pop("src")
        request["questions"][qid] = q
        gold[qid] = gold_key(q["type"], label)
    meta = row["_meta"]
    return {"id": rec_id, "source": source, "request": request, "gold": gold,
            "note": f"transfer-v4 development.jsonl line {line_no} ({meta['source']})",
            "provenance": {"file": DEV_FILE, "repo": "github.com/jaredpalmer/kev", "tag": KEV_TAG,
                           "line": line_no, "line_sha256": sha256_bytes(raw_line.encode()),
                           "meta_id": meta["id"], "meta_row_sha256": meta.get("row_sha256"), "src": src, "_meta": meta}}


def read_dev(kev_repo):
    """-> (file bytes, where they came from)."""
    if kev_repo is not None:
        try:
            head = subprocess.run(["git", "-C", str(kev_repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        except OSError:
            head = ""
        return (Path(kev_repo) / DEV_FILE).read_bytes(), f"{kev_repo} (HEAD {head or 'unknown'})"
    with urllib.request.urlopen(RAW_URL, timeout=60) as response:
        return response.read(), RAW_URL


def rebuild_requests(pub, lines, rows):
    """The 382 records in their order: the listed ones as they are, the referenced ones rebuilt from their lines."""
    records, rebuilt = [], {}
    for r in pub["records"]:
        if "request" in r:
            records.append(r)
            continue
        prov = r["provenance"]
        if "derived_from" in prov:
            records.append(r)   # filled in below, once its base record is rebuilt
            continue
        n = prov["line"]
        if sha256_bytes(lines[n].encode()) != prov["line_sha256"]:
            sys.exit(f"{r['id']}: line {n} of {DEV_FILE} does not match its SHA-256")
        full = from_dev(r["id"], r["source"], n, rows[n], lines[n])
        for k in ("file", "repo", "tag", "line", "line_sha256", "meta_id"):
            assert full["provenance"][k] == prov[k], (r["id"], k)
        assert full["gold"] == r["gold"] and full["note"] == r["note"], r["id"]
        rebuilt[r["id"]] = full
        records.append(full)

    out_records = []
    for r in records:
        prov = r.get("provenance", {})
        if "request" not in r and "derived_from" in prov:
            base = rebuilt[prov["derived_from"]]
            edit = prov["edit"]
            red = copy.deepcopy(base)
            qid = edit["field"].split(".")[1]
            q = red["request"]["questions"][qid]
            assert edit["field"] == f"questions.{qid}.instructions" and q["instructions"].count(edit["from"]) == 1, r["id"]
            q["instructions"] = q["instructions"].replace(edit["from"], edit["to"])
            red.update(id=r["id"], source=r["source"], note=r["note"])
            red["provenance"] = {**red["provenance"], "derived_from": prov["derived_from"], "edit": edit}
            assert red["gold"] == r["gold"], r["id"]
            out_records.append(red)
        else:
            out_records.append(r)
    want = pub["work_request_sha256"]
    assert [r["id"] for r in out_records] == list(want), "the records are not the listed ones in their order"
    bad = [r["id"] for r in out_records if request_sha256(r["request"]) != want[r["id"]]]
    if bad:
        sys.exit(f"{len(bad)} rebuilt requests differ from their recorded request_sha256, e.g. {bad[0]}")
    return out_records, len(rebuilt) + sum(1 for r in out_records if "derived_from" in r["provenance"])


def rebuild_red_arms(arms_pub, records):
    """The four control arms in the record form the conversion scripts read, from the rebuilt records."""
    by_id = {r["id"]: r for r in records}
    out = []
    for a in arms_pub["records"]:
        base = by_id[a["base_id"]]
        prov = a["provenance"]
        if request_sha256(base["request"]) != prov["base_request_sha256"]:
            sys.exit(f"{a['id']}: its base record {a['base_id']} differs from the recorded request")
        d = prov["derive"]
        if d["kind"] == "state_swap":
            request = {**base["request"], "state": by_id[d["state_of"]]["request"]["state"]}
        else:
            assert d["kind"] == "edit", d
            request = copy.deepcopy(base["request"])
            qid = d["field"].split(".")[1]
            q = request["questions"][qid]
            assert d["field"] == f"questions.{qid}.instructions" and q["instructions"].count(d["from"]) == 1, a["id"]
            q["instructions"] = q["instructions"].replace(d["from"], d["to"])
        if "request" in a and a["request"] != request:
            sys.exit(f"{a['id']}: the listed request differs from the one made from its base record")
        if request_sha256(request) != prov["request_sha256"]:
            sys.exit(f"{a['id']}: the rebuilt arm differs from its recorded request_sha256")
        rec = {}
        for k, v in a.items():
            rec[k] = v
            if k == "question":
                rec["request"] = request
        out.append(rec)
    return out


def write_checked(out, text, want_sha, what):
    digest = sha256_bytes(text.encode())
    if digest != want_sha:
        sys.exit(f"rebuilt {what} has SHA-256 {digest}, expected {want_sha}; nothing written")
    out.write_text(text, encoding="utf-8")
    return digest


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="where to write the rebuilt requests.json")
    ap.add_argument("--red-arms", default=None, help="also write the four control arms here (red_arms.json)")
    ap.add_argument("--public", default=str(HERE / "requests_public.json"))
    ap.add_argument("--public-arms", default=str(HERE / "red_arms_public.json"))
    ap.add_argument("--kev-repo", default=None, help="an existing clone of github.com/jaredpalmer/kev at tag kev-1.0")
    a = ap.parse_args()
    out = Path(a.out)
    arms_out = Path(a.red_arms) if a.red_arms else None
    for p in (out, arms_out):
        if p is not None and p.exists():
            sys.exit(f"{p} exists; pass another path")
    pub = json.loads(Path(a.public).read_text(encoding="utf-8"))
    raw, origin = read_dev(a.kev_repo)
    if sha256_bytes(raw) != DEV_SHA256:
        sys.exit(f"{DEV_FILE} has SHA-256 {sha256_bytes(raw)}, expected {DEV_SHA256}")
    lines = raw.decode().splitlines()
    rows = [json.loads(line) for line in lines]
    assert len(rows) == DEV_LINES, len(rows)

    records, n_rebuilt = rebuild_requests(pub, lines, rows)
    arms_text = None
    if arms_out is not None:
        arms_pub = json.loads(Path(a.public_arms).read_text(encoding="utf-8"))
        arms_text = dump({**arms_pub["header"], "records": rebuild_red_arms(arms_pub, records)})
        if sha256_bytes(arms_text.encode()) != arms_pub["rebuild_target_sha256"]:
            sys.exit(f"rebuilt red arms have SHA-256 {sha256_bytes(arms_text.encode())}, expected "
                     f"{arms_pub['rebuild_target_sha256']}; nothing written")
    digest = write_checked(out, dump({**pub["header"], "records": records}), pub["rebuild_target_sha256"], "requests.json")
    result = {"out": str(out), "records": len(records), "rebuilt_from_kev": n_rebuilt, "sha256": digest,
              "development_file": origin}
    if arms_text is not None:
        result.update(red_arms=str(arms_out), red_arms_sha256=write_checked(arms_out, arms_text, arms_pub["rebuild_target_sha256"],
                                                                             "red_arms.json"))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
