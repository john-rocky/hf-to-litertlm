"""Rebuild the full fixture file (requests.json, 377 requests) from requests_public.json and the kev repository.

    python rebuild_requests.py --out requests.json
    python rebuild_requests.py --out requests.json --kev-repo path/to/kev     (an existing clone at tag kev-1.0)

requests_public.json lists the records taken from the kev repository's transfer-v4 development file by reference
(file, tag, line, SHA-256 of the line, _meta.id). This script reads that file from github.com/jaredpalmer/kev at the
commit of tag kev-1.0 (one 803,309-byte download from raw.githubusercontent.com unless --kev-repo is given), checks the
file's SHA-256 and every referenced line's SHA-256, turns each line into a fixture record exactly as the original
fixture builder did (`label` -> gold key, `label` / `src` / `_meta` moved out of the request into the provenance),
rebuilds red_arm_000 from tv4_000 with its one-word edit, and writes the full file. The result must have the SHA-256
recorded in requests_public.json (asserted); nothing is written otherwise. Python 3.8+ standard library only.
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


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="where to write the rebuilt requests.json")
    ap.add_argument("--public", default=str(Path(__file__).resolve().parent / "requests_public.json"))
    ap.add_argument("--kev-repo", default=None, help="an existing clone of github.com/jaredpalmer/kev at tag kev-1.0")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        sys.exit(f"{out} exists; pass another --out")
    pub = json.loads(Path(a.public).read_text(encoding="utf-8"))
    raw, origin = read_dev(a.kev_repo)
    if sha256_bytes(raw) != DEV_SHA256:
        sys.exit(f"{DEV_FILE} has SHA-256 {sha256_bytes(raw)}, expected {DEV_SHA256}")
    lines = raw.decode().splitlines()
    rows = [json.loads(line) for line in lines]
    assert len(rows) == DEV_LINES, len(rows)

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

    doc = {**pub["header"], "records": out_records}
    text = json.dumps(doc, indent=1, ensure_ascii=False) + "\n"
    digest = sha256_bytes(text.encode())
    if digest != pub["full_requests_json_sha256"]:
        sys.exit(f"rebuilt file has SHA-256 {digest}, expected {pub['full_requests_json_sha256']}; nothing written")
    out.write_text(text, encoding="utf-8")
    print(json.dumps({"out": str(out), "records": len(out_records), "rebuilt_from_kev": len(rebuilt) + sum(
        1 for r in out_records if "derived_from" in r["provenance"]), "sha256": digest, "development_file": origin}))


if __name__ == "__main__":
    main()
