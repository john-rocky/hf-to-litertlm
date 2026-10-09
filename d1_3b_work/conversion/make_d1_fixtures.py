"""Rounds 1-2: the d1-3B fixture set -> fixtures/requests.json and fixtures/LICENSE-SemIf-MIT.txt (README.md by hand).

    python3 scripts/make_d1_fixtures.py [--replace]      # standard library only, from W

Records (file order):
  the 377 records of Kev-0.8B LiteRT's requests.json, copied unchanged (id, source, request, gold, note, provenance with
      `_meta` and `_meta.id`; tv4 60, tv4x 140, tv4s 20, semif 144, own 12, red_arm_000 1 = tv4_000 with one word of the
      instructions changed, a gate arm that must fail, not a fixture)
  card_text_001      the d1-3B card's text example: its state and its three questions (refund / team / urgency)
  card_cats_001      the card's image example (the cats choice, state None); the image is chosen in round 5
  own_long_15k_001   round 2: the request of another fixture file's record `long_15k` (JSON object state, cold-room log)
  own_long_34k_001   round 2: the request of another fixture file's record `long_34k` (JSON array state, order ledger); both
                     copied from external/other_conversion/fixtures/records.json (sha256 D1B_SHA256, read-only) with their gold,
                     note and provenance, so that the two conversions' references can be compared record by record (key =
                     `request_sha256`); the round-1 records of these ids (scripts/own_long_records.py) left the fixture
  own_mid_08k_001    round 2: written for this port, a plain-text state of about 800 tokens (rows in the 513-1,024
                     bucket; scripts/own_mid_records.py)
The record format is Kev's: {id, source, request: {state, questions[, images]}, gold: {qid: key or null}, note, provenance}.
Gold keys: the criteria key (choice), "true" / "false" (noul), the level index as a string (score); null = no intended
answer (the card states none). The card records' strings are asserted to appear in hf_small/README.md as written.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from own_mid_records import OWN_MID  # noqa: E402

K = Path(__file__).resolve().parents[1]
REPO_ROOT = K.parent
KEV_FIXTURES = K / "external/kev/requests.json"
KEV_FIXTURES_SHA256 = "dfe55fb145df7a3967ed7213ae24d67b5d4ec42da0315fdb409d5e4b702e3a48"
SEMIF_LICENSE = K / "external/kev/LICENSE-SemIf-MIT.txt"
SEMIF_LICENSE_SHA256 = "f765f2140f8507a8f0d81ec0fd2c4bd72fe6a066841ef27883ff876a76bf61be"
CARD = K / "hf_small/README.md"
CARD_GIT_BLOB = "d931d05215c55c0163bddbf4da70561342e0a908"
REV = "da1fe36a861f24690f27f622dca1d8688503d113"
OUT = K / "fixtures/requests.json"
IMAGE_TBD = "<TBD: the image is chosen in round 5>"
D1B_FIXTURES = K / "external/other_conversion/fixtures/records.json"
D1B_SHA256 = "2284bb22e8cdc02a4d1c4a787e34981d2cee4044b32c3b4452abcd61136ef4dd"
FROM_D1B = {"own_long_15k_001": "long_15k", "own_long_34k_001": "long_34k"}
REQUEST_SHA256 = "sha256 of json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(',', ':')) in UTF-8"
CARD_IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"   # the card's example image (two cats on a sofa)

CARD_STATE = "I was charged twice this month, please refund one of them."
CARD_QUESTIONS = {
    "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
    "team": {"type": "choice", "instructions": "Which team should handle this?",
             "criteria": {"billing": "Charges, refunds, invoices", "technical": "App or site faults",
                          "fraud": "Suspected unauthorised use"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["Can wait", "Today", "Blocking the customer now"]},
}
CATS = {"type": "choice", "instructions": "How many cats are there?",
        "criteria": {"one": "One", "two": "Two", "more": "Three or more"}}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def git_blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def card_strings_present(card: str) -> None:
    """Every string of the two card records is in the card as written (Python literal form)."""
    def lit(s: str) -> str:
        return json.dumps(s, ensure_ascii=False)
    needles = [lit(CARD_STATE)]
    for qid, q in CARD_QUESTIONS.items():
        needles += [f"{lit(qid)}: {{", lit(q["instructions"])]
        crit = q.get("criteria")
        if isinstance(crit, dict):
            needles += [f"{lit(k)}: {lit(v)}" for k, v in crit.items()]
        elif crit:
            needles.append("[" + ", ".join(lit(v) for v in crit) + "]")
    needles += [lit(CATS["instructions"]), "{" + ", ".join(f"{lit(k)}: {lit(v)}" for k, v in CATS["criteria"].items()) + "}",
                f"model.system_one(None, {{{lit('cats')}: cats}}, images=[image])", CARD_IMAGE_URL]
    missing = [n for n in needles if n not in card]
    assert not missing, missing


def request_sha256(request: dict) -> str:
    return sha256_bytes(json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode())


def gold_ok(q: dict, key) -> bool:
    if key is None:
        return True
    kind = q.get("type", "choice")
    if kind == "noul":
        return key in ("true", "false")
    if kind == "score":
        return key in [str(i) for i in range(len(q["criteria"]))]
    return key in q["criteria"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replace", action="store_true", help="overwrite fixtures/requests.json")
    a = ap.parse_args()
    if OUT.exists() and not a.replace:
        raise SystemExit(f"refusing to overwrite {OUT} (pass --replace)")

    raw = KEV_FIXTURES.read_bytes()
    assert sha256_bytes(raw) == KEV_FIXTURES_SHA256
    kev = json.loads(raw)
    records = copy.deepcopy(kev["records"])
    assert len(records) == 377 and sum(len(r["request"]["questions"]) for r in records) == 402
    # the red arm is already among them
    red = next(r for r in records if r["id"] == "red_arm_000")
    base = next(r for r in records if r["id"] == "tv4_000")
    (qid, q), = red["request"]["questions"].items()
    assert base["request"]["questions"][qid]["instructions"].replace("correctly", "incorrectly") == q["instructions"]
    assert red["request"]["state"] == base["request"]["state"] and red["gold"] == base["gold"]

    card_bytes = CARD.read_bytes()
    assert git_blob_id(card_bytes) == CARD_GIT_BLOB
    card_strings_present(card_bytes.decode())
    card_prov = {"file": "README.md", "repo": "LiquidAI/d1-3B", "revision": REV, "git_blob": CARD_GIT_BLOB,
                 "licence": "LFM Open License v1.0 (the model card)"}
    records.append({
        "id": "card_text_001", "source": "card",
        "request": {"state": CARD_STATE, "questions": copy.deepcopy(CARD_QUESTIONS)},
        "gold": {"refund": "true", "team": "billing", "urgency": None},
        "note": "the d1-3B card's text example (`model.system_one(...)`), three questions over one state",
        "provenance": {**card_prov, "gold_basis": "the card states no answers; refund and team are this port's reading, urgency has none"},
    })
    records.append({
        "id": "card_cats_001", "source": "card",
        "request": {"state": None, "questions": {"cats": copy.deepcopy(CATS)}, "images": [IMAGE_TBD]},
        "gold": {"cats": None},
        "note": "the d1-3B card's image example (the photo is the whole state); the image is chosen in round 5",
        "provenance": {**card_prov, "card_image_url": CARD_IMAGE_URL,
                       "card_image_use": "COCO val2017: measurement only, never in a published file",
                       "gold_basis": "depends on the image chosen in round 5 (the card's photo shows two cats)"},
    })
    d1b_raw = D1B_FIXTURES.read_bytes()
    assert sha256_bytes(d1b_raw) == D1B_SHA256, "the other conversion's records.json changed"
    other = {r["id"]: r for r in json.loads(d1b_raw)["records"]}
    for ours, theirs in FROM_D1B.items():
        r = other[theirs]
        records.append({"id": ours, "source": "own", "request": copy.deepcopy(r["request"]), "gold": dict(r["gold"]),
                        "note": r["note"],
                        "provenance": {**copy.deepcopy(r["provenance"]), "written_for": "another conversion of d1-3B",
                                       "copied_from": {"file": f"another conversion's fixture file (records.json, sha256 {D1B_SHA256})",
                                                       "sha256": D1B_SHA256, "record": theirs,
                                                       "fields": "request, gold, note, provenance"},
                                       "request_sha256": request_sha256(r["request"])}})
    for o in OWN_MID:
        records.append({"id": o["id"], "source": "own", "request": copy.deepcopy(o["request"]), "gold": dict(o["gold"]),
                        "note": o["kind"],
                        "provenance": {"file": "conversion/own_mid_records.py",
                                       "licence": "written for this port; no person, organisation, product or place names",
                                       "request_sha256": request_sha256(o["request"])}})

    ids = [r["id"] for r in records]
    assert len(ids) == len(set(ids)), [i for i, n in Counter(ids).items() if n > 1]
    for r in records:
        qs = r["request"]["questions"]
        assert set(r["gold"]) == set(qs), r["id"]
        for qid, q in qs.items():
            assert gold_ok(q, r["gold"][qid]), (r["id"], qid, r["gold"][qid])

    counts = Counter(r["source"] for r in records)
    qtypes = Counter((r["source"], q.get("type", "choice")) for r in records for q in r["request"]["questions"].values())
    doc = {
        "version": 1,
        "created_by": "conversion/make_d1_fixtures.py",
        "request_sha256": REQUEST_SHA256,
        "record_format": "{id, source, request: {state, questions[, images]}, gold: {qid: key or null}, note, provenance} "
                         "(the Kev fixtures' format; `images` only on card_cats_001)",
        "gold_keys": "choice = criteria key, noul = 'true'/'false', score = level index as a string; null = no intended answer",
        "sources": {
            "kev": {"what": "the 377 records of the Kev-0.8B LiteRT conversion's fixture file (the records of "
                            "litert-community/Kev-0.8B-LiteRT's fixtures/), unchanged (tv4, tv4x, tv4s, semif, own, red_arm)",
                    "file_sha256": KEV_FIXTURES_SHA256, "file_sources": kev["sources"]},
            "card": {"what": "the d1-3B card's two examples", "repo": "LiquidAI/d1-3B", "revision": REV, "file": "README.md",
                     "git_blob": CARD_GIT_BLOB},
            "own_long": {"what": "own_long_15k_001 / own_long_34k_001 = the requests of the records long_15k / long_34k of "
                                 "another conversion's fixture file (gold, note and provenance copied too)",
                         "file": f"another conversion's fixture file (records.json, sha256 {D1B_SHA256})", "sha256": D1B_SHA256},
            "own_mid": {"what": "own_mid_08k_001, a state of about 800 tokens written for this port",
                        "file": "conversion/own_mid_records.py"},
        },
        "counts": dict(counts), "questions": sum(qtypes.values()),
        "question_types": {f"{s}/{t}": n for (s, t), n in sorted(qtypes.items())},
        "records": records,
    }
    OUT.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    lic = SEMIF_LICENSE.read_bytes()
    assert sha256_bytes(lic) == SEMIF_LICENSE_SHA256
    (K / "fixtures/LICENSE-SemIf-MIT.txt").write_bytes(lic)
    print(json.dumps({"records": len(records), "questions": doc["questions"], "counts": dict(counts),
                      "question_types": doc["question_types"], "sha256": sha256_bytes(OUT.read_bytes())}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
