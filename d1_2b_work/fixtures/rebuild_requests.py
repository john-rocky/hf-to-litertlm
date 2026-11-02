"""Rebuild the full test requests (requests.json) from requests_public.json and the kev repository.

    python rebuild_requests.py --out requests.json
    python rebuild_requests.py --out requests.json --kev-repo path/to/kev     (an existing clone at tag kev-1.0)

requests_public.json lists every test record in the order the checks send them. The records made from the transfer-v4
development file of github.com/jaredpalmer/kev carry no request: a record taken from a line lists the line (repo, tag,
commit, file, 0-based line, SHA-256 of the line, the line's _meta.id), red_arm_000 lists its base record and its
one-word edit, and the two control arms made from transfer-v4 records list their rule (`derive`). This script reads
that file at the commit of tag kev-1.0 (one 803,309-byte download from raw.githubusercontent.com unless --kev-repo is
given), checks the file's SHA-256 and every listed line's SHA-256, and turns each line into a request as the d1-3B
fixture builder did (the line's state and questions, with `label` and `src` taken out of each question; the gold from
`label`). It then makes the form these checks send: `"media": []` on each of these requests (a transfer-v4 record
has no pictures or clips), red_arm_000 from tv4_000 with its edit, and each control arm from its base record (an edit of
the instructions, or the state of another record). Every request must have the request_sha256 listed for its record,
every gold and note made from a line must equal the listed ones, and the whole file must have the SHA-256 listed in
requests_public.json (rebuild_target_sha256); nothing is written otherwise. The records that carry their request are
copied as they are (their request_sha256 is checked too).
Standard library only.
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
KEV_REPO, KEV_TAG, KEV_COMMIT = "github.com/jaredpalmer/kev", "kev-1.0", "6b719c3c3f367295f6ef336f4f751cf5ff970abc"
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
    """One development line -> one record (the d1-3B fixture builder's transform, unchanged)."""
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
            "provenance": {"file": DEV_FILE, "repo": KEV_REPO, "tag": KEV_TAG,
                           "line": line_no, "line_sha256": sha256_bytes(raw_line.encode()),
                           "meta_id": meta["id"], "meta_row_sha256": meta.get("row_sha256"), "src": src, "_meta": meta}}


def with_media(request):
    """The form these checks send: the same request with an empty media list."""
    assert "media" not in request and "images" not in request, sorted(request)
    return {**request, "media": []}


def edit_instructions(request, edit, what):
    """A copy of the request with one string of one question's instructions replaced (it must occur exactly once)."""
    out = copy.deepcopy(request)
    parts = edit["field"].split(".")
    assert len(parts) == 3 and parts[0] == "questions" and parts[2] == "instructions", (what, edit["field"])
    q = out["questions"][parts[1]]
    if not isinstance(q["instructions"], str) or q["instructions"].count(edit["from"]) != 1:
        sys.exit(f"{what}: {edit['from']!r} does not occur exactly once in {edit['field']}")
    q["instructions"] = q["instructions"].replace(edit["from"], edit["to"])
    return out


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


def rebuild(pub, lines, rows):
    """-> {id: request} for every record listed without a request, in three passes: lines, edits, control arms."""
    listed = {r["id"]: r["request"] for r in pub["records"] if "request" in r}
    made, gold = {}, {}
    for r in pub["records"]:
        prov = r.get("provenance", {})
        if "request" in r or "line" not in prov:
            continue
        for k, want in (("repo", KEV_REPO), ("tag", KEV_TAG), ("commit", KEV_COMMIT), ("file", DEV_FILE)):
            if prov[k] != want:
                sys.exit(f"{r['id']}: provenance {k} is {prov[k]!r}; this script reads {want!r} only")
        n = prov["line"]
        if sha256_bytes(lines[n].encode()) != prov["line_sha256"]:
            sys.exit(f"{r['id']}: line {n} of {DEV_FILE} does not match its SHA-256")
        full = from_dev(r["id"], r["source"], n, rows[n], lines[n])
        if full["provenance"]["meta_id"] != prov["meta_id"]:
            sys.exit(f"{r['id']}: line {n} has _meta.id {full['provenance']['meta_id']}, listed {prov['meta_id']}")
        if full["gold"] != r["gold"] or full["note"] != r["note"]:
            sys.exit(f"{r['id']}: the gold or note made from line {n} differs from the listed one")
        made[r["id"]] = with_media(full["request"])
        gold[r["id"]] = full["gold"]
    for r in pub["records"]:
        prov = r.get("provenance", {})
        if "request" in r or "derived_from" not in prov:
            continue
        base = prov["derived_from"]
        if base not in made:
            sys.exit(f"{r['id']}: its base record {base} is not one made from a line")
        if r["gold"] != gold[base]:
            sys.exit(f"{r['id']}: the listed gold differs from its base record's")
        made[r["id"]] = edit_instructions(made[base], prov["edit"], r["id"])
    known = {**listed, **made}
    for r in pub["records"]:
        if "request" in r or "derive" not in r:
            continue
        if r["base_id"] not in known:
            sys.exit(f"{r['id']}: its base record {r['base_id']} is not in the file")
        d = r["derive"]
        if d["kind"] == "state_swap":
            made[r["id"]] = {**known[r["base_id"]], "state": known[d["state_of"]]["state"]}
        elif d["kind"] == "edit":
            made[r["id"]] = edit_instructions(known[r["base_id"]], d, r["id"])
        else:
            sys.exit(f"{r['id']}: unknown rule {d['kind']!r}")
    missing = [r["id"] for r in pub["records"] if "request" not in r and r["id"] not in made]
    if missing:
        sys.exit(f"{len(missing)} records have neither a request nor a reference, e.g. {missing[0]}")
    return made


def assemble(pub, made):
    """The records in their order, each with its request just before its request_sha256."""
    out = []
    for r in pub["records"]:
        if "request" in r:
            out.append(r)
            continue
        rec = {}
        for k, v in r.items():
            if k == "request_sha256":
                rec["request"] = made[r["id"]]
            rec[k] = v
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="where to write the rebuilt requests.json")
    ap.add_argument("--public", default=str(HERE / "requests_public.json"))
    ap.add_argument("--kev-repo", default=None, help="an existing clone of github.com/jaredpalmer/kev at tag kev-1.0")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        sys.exit(f"{out} exists; pass another path")
    pub = json.loads(Path(a.public).read_text(encoding="utf-8"))
    raw, origin = read_dev(a.kev_repo)
    if sha256_bytes(raw) != DEV_SHA256:
        sys.exit(f"{DEV_FILE} has SHA-256 {sha256_bytes(raw)}, expected {DEV_SHA256}")
    lines = raw.decode().splitlines()
    rows = [json.loads(line) for line in lines]
    assert len(rows) == DEV_LINES, len(rows)

    made = rebuild(pub, lines, rows)
    records = assemble(pub, made)
    bad = [r["id"] for r in records if request_sha256(r["request"]) != r["request_sha256"]]
    if bad:
        sys.exit(f"{len(bad)} requests differ from their listed request_sha256, e.g. {bad[0]}; nothing written")
    text = dump({**pub["header"], "records": records})
    digest = sha256_bytes(text.encode())
    if digest != pub["rebuild_target_sha256"]:
        sys.exit(f"the rebuilt file has SHA-256 {digest}, expected {pub['rebuild_target_sha256']}; nothing written")
    out.write_text(text, encoding="utf-8")
    print(json.dumps({"out": str(out), "records": len(records), "rebuilt": len(made),
                      "request_sha256_checked": len(records), "sha256": digest, "bytes": len(text.encode()),
                      "development_file": origin}))


if __name__ == "__main__":
    main()
