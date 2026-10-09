"""Round 8 close: fixtures/requests.json `card_cats_001` gets its picture (`request.images`) and its `note`; nothing
else changes. The picture is the d1-3B card's own example, COCO val2017 000000039769.jpg (chosen in
round 8): the record holds its URL, sha256, bytes and size, never the file (COCO's pictures are Flickr
photos under their own licences and are not redistributed here; a reader fetches it from the URL and checks the
sha256).

    $EXPORT scripts/d1_image_fixture.py [--write]

Checks: the file round-trips through make_d1_fixtures.py's serialisation (json.dumps(indent=1, ensure_ascii=False) +
"\n") before the change; the local copy (cache/realv/coco_cats.jpg, measurement only) has the sha256, bytes and size
written; after the change every top-level key, every other record and the record's other keys equal the original, and
the text diff's hunks lie inside the record's lines. Prints the old and new sha256 of requests.json.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image

K = Path(__file__).resolve().parents[1]
FIXTURES = K / "fixtures/requests.json"
RID = "card_cats_001"
LOCAL = K / "cache/realv/coco_cats.jpg"
IMAGE = {"url": "http://images.cocodataset.org/val2017/000000039769.jpg",
         "sha256": "dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e", "bytes": 173131, "wh": [640, 480],
         "note": "COCO val2017, not redistributed"}
NOTE = ("the d1-3B card's image example (the photo is the whole state): COCO val2017 000000039769.jpg, the card's own "
        "photo (two cats), chosen in round 8; the record keeps its URL, sha256, bytes and size, not the file (fetch it "
        "from the URL and check the sha256). The provider's float32 CPU answer on it is `two` (0.985, "
        "results/real_vision_e2e_ref.json)")


def dump(doc: dict) -> str:
    return json.dumps(doc, indent=1, ensure_ascii=False) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    raw = LOCAL.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == IMAGE["sha256"] and len(raw) == IMAGE["bytes"], "local copy differs"
    assert list(Image.open(LOCAL).size) == IMAGE["wh"]
    old_text = FIXTURES.read_text()
    old = json.loads(old_text)
    assert dump(old) == old_text, "requests.json does not round-trip through make_d1_fixtures.py's serialisation"
    new = json.loads(old_text)
    idx = [i for i, r in enumerate(new["records"]) if r["id"] == RID]
    assert len(idx) == 1, idx
    rec = new["records"][idx[0]]
    rec["request"]["images"] = [IMAGE]
    rec["note"] = NOTE
    for k in old:
        if k != "records":
            assert new[k] == old[k], k
    assert len(new["records"]) == len(old["records"])
    for i, (x, y) in enumerate(zip(old["records"], new["records"])):
        if i != idx[0]:
            assert x == y, x["id"]
    o, n = old["records"][idx[0]], rec
    assert {k: v for k, v in o.items() if k not in ("request", "note")} == {k: v for k, v in n.items() if k not in ("request", "note")}
    assert {k: v for k, v in o["request"].items() if k != "images"} == {k: v for k, v in n["request"].items() if k != "images"}
    new_text = dump(new)
    # the record's line span in the new text
    lines = new_text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip() == f'"id": "{RID}",') - 1
    depth, end = 0, start
    for end in range(start, len(lines)):
        depth += lines[end].count("{") - lines[end].count("}")
        if depth == 0:
            break
    diff = list(difflib.unified_diff(old_text.splitlines(), lines, n=0, lineterm=""))
    hunks = [ln for ln in diff if ln.startswith("@@")]
    inside = all(start + 1 <= int(h.split("+")[1].split(",")[0].split(" ")[0]) <= end + 1 for h in hunks)
    print(json.dumps({"old_sha256": hashlib.sha256(old_text.encode()).hexdigest(),
                      "new_sha256": hashlib.sha256(new_text.encode()).hexdigest(), "old_bytes": len(old_text.encode()),
                      "new_bytes": len(new_text.encode()), "record_lines_new": [start + 1, end + 1], "hunks": hunks,
                      "hunks_inside_record": inside, "write": a.write}, indent=1))
    print("\n".join(diff))
    assert inside, "a hunk lies outside the record"
    if a.write:
        FIXTURES.write_text(new_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
