#!/usr/bin/env python3
"""The red arms of round 1, written before the reference runs (pre-registration): three text records of
fixtures/records.json, each with ONE word of ONE question's instruction changed. A correct graph must move that
record's probabilities away from the BASE record's reference (> 0.02 on some option); a graph that ignored the
question text would not.

    venv-ref/bin/python scripts/make_red_arms.py        -> fixtures/red_arms.json (refuses to overwrite)

Arms (base record / question / word -> word; the expected direction is a reading of the text, not a measurement):
  semif_a3f18f3a63d45345942b / decision   "fitted"  -> "ordered"  (SemIf: the evidence says the lenses were ordered,
                                                                    not fitted: contradicted -> supported)
  own_t01 / wants_refund                  "refund"  -> "repair"   (the customer asks for a repair, not a refund)
  own_j01 / large                         "above"   -> "below"    (the invoice total 201.5 is below 500)
The arm records keep the fixture schema (id, source, request, gold, note, state_tokens) with source "red_arm" and
gold empty, so the reference script runs them as they are (--records-file fixtures/red_arms.json --tag _red)."""
import copy
import datetime
import hashlib
import json
import re
from pathlib import Path

K = Path(__file__).resolve().parents[1]
ARMS = (
    ("semif_a3f18f3a63d45345942b", "decision", "fitted", "ordered"),
    ("own_t01", "wants_refund", "refund", "repair"),
    ("own_j01", "large", "above", "below"),
)


def main() -> int:
    src = K / "fixtures/records.json"
    out = K / "fixtures/red_arms.json"
    assert not out.exists(), f"refusing to overwrite {out}"
    doc = json.loads(src.read_text())
    by_id = {r["id"]: r for r in doc["records"]}
    arms, records = [], []
    for base_id, qid, w_from, w_to in ARMS:
        base = by_id[base_id]
        assert not base.get("image_files"), base_id
        instr = base["request"]["questions"][qid]["instructions"]
        pat = re.compile(rf"\b{re.escape(w_from)}\b")
        assert len(pat.findall(instr)) == 1, (base_id, qid, instr)
        new = pat.sub(w_to, instr)
        req = copy.deepcopy(base["request"])
        req["questions"][qid]["instructions"] = new
        arm_id = f"red_{base_id}_{qid}"
        arms.append({"arm_id": arm_id, "base_id": base_id, "base_source": base["source"], "question_id": qid,
                     "word_from": w_from, "word_to": w_to, "instruction_base": instr, "instruction_arm": new,
                     "other_questions_unchanged": [q for q in req["questions"] if q != qid]})
        records.append({"id": arm_id, "source": "red_arm", "request": req, "gold": {},
                        "note": f"red arm of {base_id}: question {qid}, '{w_from}' -> '{w_to}'",
                        "state_tokens": base.get("state_tokens"), "base_id": base_id})
    res = {"what": "round 1 red arms: one word of one question's instruction changed in three text records "
                   "(pre-registered before the reference run)",
           "created_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "base_records_json": "fixtures/records.json",
           "base_records_json_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
           "rule": "each arm's probabilities (graph or reference) are compared with the BASE record's reference "
                   "probabilities, same question and option order; the arm moves when some option differs by > 0.02",
           "arms": arms, "records": records}
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(arms, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
