"""Fixtures -> fixtures/requests.json (+ fixtures/LICENSE-SemIf-MIT.txt, fixtures/token_lengths_precheck.json).

    python scripts/make_fixtures.py        (needs kev/ = the author's repository at tag kev-1.0, and
                                            src/semif/authored144.jsonl + src/semif/LICENSE = SemIf at ca3ba65f;
                                            KEV_SEMIF / KEV_SEMIF_LICENSE override those two paths)

Every record is {"id", "source", "request": {"state", "questions"}, "gold": {qid: key}, "note", "provenance"}; `request`
goes through kev.api.SystemOneRequest.model_validate unchanged (the dataset's `label` / `src` / `_meta` are moved out of
the request into `gold` / `provenance`). Gold keys are the ones kev.api.question_keys reports: the criteria key (choice),
"true" / "false" (noul), the level index as a string (score).

Slices (file order everywhere):
  tv4_000..059          the first 60 records of evals/v4/transfer-v4/development.jsonl @ kev-1.0 (all MMLU, 4-way choice)
  tv4x_<source>_<k:02d> the first 20 records of each of the 7 other sources (by _meta.source)
  tv4s_<k:02d>          the first 20 records whose question is of type score
  semif_<id>            SemIf authored144 (MIT, TheoLeeCJ/SemIf @ ca3ba65f) as 3-way choice
  own_*                 written for this port (own_records.py), invented names only
  red_arm_000           tv4_000 with one word of the instructions changed; a gate arm that must fail, not a fixture
A tokenizer-only pass (the base tokenizer, kev.api.to_record + kev.model.encode in the serving context, strict) checks that
every row (state + one question's branch) fits 2,048 tokens and prints the long own rows."""
import copy
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

K = Path(__file__).resolve().parents[1]
DEV = K / "kev/evals/v4/transfer-v4/development.jsonl"
DEV_SHA256 = "ff374c49c6c9f15f8a56fb274b4a4857d20497eb8dd1ac07ce01560e682a5f2e"
SEMIF = Path(os.environ.get("KEV_SEMIF", K / "src/semif/authored144.jsonl"))
SEMIF_SHA256 = "8162d1c73f925af64453f1ec05ef36d583b3815bf698e60f0d454bd11537e079"
SEMIF_REPO, SEMIF_PIN = "TheoLeeCJ/SemIf", "ca3ba65f"
TV4X_SOURCES = ("emotion", "tweet_offensive", "qnli", "paws", "sciq", "legacy_holdout", "composition_holdout")
RED_FROM, RED_TO = "correctly", "incorrectly"
BASE, BASE_REV = "Qwen/Qwen3.5-0.8B-Base", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"

SEMIF_LICENSE = Path(os.environ.get("KEV_SEMIF_LICENSE", K / "src/semif/LICENSE"))   # SemIf at ca3ba65f
SEMIF_LICENSE_SHA256 = "f765f2140f8507a8f0d81ec0fd2c4bd72fe6a066841ef27883ff876a76bf61be"


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
            "provenance": {"file": "evals/v4/transfer-v4/development.jsonl", "repo": "github.com/jaredpalmer/kev", "tag": "kev-1.0",
                           "line": line_no, "line_sha256": sha256_bytes(raw_line.encode()),
                           "meta_id": meta["id"], "meta_row_sha256": meta.get("row_sha256"), "src": src, "_meta": meta}}


def main():
    from kev.api import SystemOneRequest, question_keys, to_record
    from kev.model import SERVE_MAX_BRANCH, SERVE_MAX_STATE, MAX_BRANCH, MAX_STATE, encode, load_tokenizer, rows_of
    from own_records import OWN

    raw = DEV.read_bytes()
    assert sha256_bytes(raw) == DEV_SHA256
    lines = raw.decode().splitlines()
    rows = [json.loads(line) for line in lines]
    assert len(rows) == 764
    records = []
    # tv4: the first 60 records
    for i in range(60):
        records.append(from_dev(f"tv4_{i:03d}", "tv4", i, rows[i], lines[i]))
    # tv4x: first 20 of each other source
    for source in TV4X_SOURCES:
        idx = [i for i, r in enumerate(rows) if r["_meta"]["source"] == source][:20]
        assert len(idx) == 20, (source, len(idx))
        for k, i in enumerate(idx):
            records.append(from_dev(f"tv4x_{source}_{k:02d}", "tv4x", i, rows[i], lines[i]))
    # tv4s: first 20 score records
    idx = [i for i, r in enumerate(rows) if any(q["type"] == "score" for q in r["questions"].values())][:20]
    assert len(idx) == 20
    for k, i in enumerate(idx):
        records.append(from_dev(f"tv4s_{k:02d}", "tv4s", i, rows[i], lines[i]))
    meta_ids = [r["provenance"]["meta_id"] for r in records]
    assert len(meta_ids) == len(set(meta_ids)), "a development record is in two slices"
    # semif
    sraw = SEMIF.read_bytes()
    assert sha256_bytes(sraw) == SEMIF_SHA256
    slines = sraw.decode().splitlines()
    for n, line in enumerate(slines):
        s = json.loads(line)
        assert len(s["options"]) == 3 and isinstance(s["label"], int)
        criteria = {o["id"]: o["description"] for o in s["options"]}
        assert len(criteria) == len(s["options"])
        records.append({"id": f"semif_{s['id']}", "source": "semif",
                        "request": {"state": s["state"], "questions": {"answer": {"type": "choice", "instructions": s["question"], "criteria": criteria}}},
                        "gold": {"answer": s["options"][s["label"]]["id"]},
                        "note": f"SemIf authored144 line {n} ({s['family']})",
                        "provenance": {"file": "authored144.jsonl", "repo": SEMIF_REPO, "pin": SEMIF_PIN, "file_sha256": SEMIF_SHA256,
                                       "line": n, "line_sha256": sha256_bytes(line.encode()), "licence": "MIT",
                                       "semif_id": s["id"], "group_id": s["group_id"], "family": s["family"], "split": s["split"]}})
    # own
    for o in OWN:
        records.append({"id": o["id"], "source": "own", "request": copy.deepcopy(o["request"]), "gold": dict(o["gold"]),
                        "note": o["kind"], "provenance": {"file": "scripts/own_records.py", "licence": "written for this port; invented names only"}})
    # red arm
    base = next(r for r in records if r["id"] == "tv4_000")
    red = copy.deepcopy(base)
    (qid, q), = red["request"]["questions"].items()
    assert q["instructions"].count(RED_FROM) == 1, q["instructions"]
    q["instructions"] = q["instructions"].replace(RED_FROM, RED_TO)
    red.update(id="red_arm_000", source="red_arm",
               note=f"tv4_000 with '{RED_FROM}' -> '{RED_TO}' in the instructions; compared against tv4_000's oracle it must fail the gate (not a fixture)")
    red["provenance"] = {**red["provenance"], "derived_from": "tv4_000", "edit": {"field": f"questions.{qid}.instructions", "from": RED_FROM, "to": RED_TO}}
    records.append(red)

    # validation: the request is a SystemOneRequest as is, gold keys are question keys
    ids = [r["id"] for r in records]
    assert len(ids) == len(set(ids))
    for r in records:
        req = SystemOneRequest.model_validate(r["request"])
        assert set(r["gold"]) == set(req.questions), r["id"]
        for qid, q in req.questions.items():
            assert r["gold"][qid] in question_keys(q.type, q.criteria), (r["id"], qid, r["gold"][qid])

    # tokenizer-only length pass (same functions the oracle runs)
    tok = load_tokenizer(BASE, revision=BASE_REV)
    lengths = {}
    for r in records:
        rec, _ = to_record(SystemOneRequest.model_validate(r["request"]))
        enc = encode(tok, rec, max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH, strict=True)
        S, _, brs = rows_of(enc)
        lengths[r["id"]] = {"total": len(enc["ids"]), "state": len(S), "rows": [len(S) + len(b["ids"]) for b in brs]}
    too_long = {k: v for k, v in lengths.items() if max(v["rows"]) > 2048}
    assert not too_long, too_long
    for k, v in lengths.items():
        if k.startswith("own_"):
            print(k, v)

    counts = Counter(r["source"] for r in records)
    qtypes = Counter((r["source"], q["type"]) for r in records for q in r["request"]["questions"].values())
    out = {
        "version": 1,
        "created_by": "scripts/make_fixtures.py",
        "record_format": "{id, source, request: {state, questions}, gold: {qid: key}, note, provenance}; request -> kev.api.SystemOneRequest.model_validate",
        "gold_keys": "kev.api.question_keys: choice = criteria key, noul = 'true'/'false', score = level index as a string",
        "sources": {
            "tv4": {"what": "first 60 records (file order) of evals/v4/transfer-v4/development.jsonl", "repo": "github.com/jaredpalmer/kev",
                    "tag": "kev-1.0", "commit": "6b719c3c3f367295f6ef336f4f751cf5ff970abc", "file_sha256": DEV_SHA256},
            "tv4x": {"what": f"first 20 records (file order) of each of {list(TV4X_SOURCES)}", "file_sha256": DEV_SHA256},
            "tv4s": {"what": "first 20 records (file order) whose question type is score", "file_sha256": DEV_SHA256},
            "semif": {"what": "SemIf authored144 as 3-way choice: state -> state, question -> instructions, options -> criteria {id: description}, gold = options[label].id",
                      "repo": SEMIF_REPO, "pin": SEMIF_PIN, "file_sha256": SEMIF_SHA256, "licence": "MIT (LICENSE-SemIf-MIT.txt)"},
            "own": {"what": "written for this port (scripts/own_records.py); invented names only"},
            "red_arm": {"what": "tv4_000 with one word of the instructions changed; not a fixture"},
        },
        "counts": dict(counts), "question_types": {f"{s}/{t}": n for (s, t), n in sorted(qtypes.items())},
        "records": records,
    }
    path = K / "fixtures/requests.json"
    assert not path.exists(), f"refusing to overwrite {path}"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    lic = SEMIF_LICENSE.read_bytes()
    assert sha256_bytes(lic) == SEMIF_LICENSE_SHA256
    (K / "fixtures/LICENSE-SemIf-MIT.txt").write_bytes(lic)
    (K / "fixtures/token_lengths_precheck.json").write_text(json.dumps(lengths, indent=0) + "\n")
    print(json.dumps({"counts": dict(counts), "question_types": out["question_types"],
                      "sha256": sha256_bytes(path.read_bytes())}, indent=1))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
