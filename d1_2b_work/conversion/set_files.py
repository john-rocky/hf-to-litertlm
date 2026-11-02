"""Write the question set files that the check scripts of this folder read (set/ of the work directory) from this
repository's fixtures: the full requests.json that fixtures/rebuild_requests.py writes, split back into the files the
checks read (the same records in the same order), and the media tables in the form those scripts read: {media id:
{"kind", "path" (from the work directory), "sha256", ...}}, the files under fixtures/media/.

    python3 conversion/set_files.py --requests fixtures/requests.json        # from the work directory

The work directory holds a copy of this repository's fixtures/ folder, with the COCO photo fetched into
fixtures/media/ (and, for d1-Omni-2B, the card's recording written there), as fixtures/README.md says; a missing media
file is listed, and the scripts that open it stop on it. Writes, for d1-2B: set/requests.json (the 506 test records
before the control arms), set/red_arms.json (the 19 control arms, a list), set/extra_g96x72.json (the record added for
the 96 x 72 canvas), set/card.json (the five records of the d1-2B card's examples, with the COCO photo's entry),
set/media.json, set/media_extra.json and set/media_all.json; for d1-Omni-2B the first three of those, the two media
tables of the text and picture set, and set/requests_audio.json, set/red_arms_audio.json, set/card_omni.json (the
four records of the d1-Omni-2B card's examples, with their media entries) and set/media_audio.json. A media table
holds the media its files' requests name (in media.json's order); media_all.json is media.json and media_extra.json
together. Each file is checked against the layout below (count, first and last id, test records or control arms) and
every request against its request_sha256; nothing is written on a mismatch, and no file is overwritten.
Standard library only.
"""
import argparse
import hashlib
import json
import os
import sys

# per model: the check files in the order the checks send them: (file, form, records, first id, last id)
TEXT_PIC = [("requests.json", "records", 506, "tv4_000", "long_refused_9k")]
LAYOUT = {
    "LiquidAI/d1-2B": TEXT_PIC + [("red_arms.json", "arms", 19, "red_not_qnli_00", "arm_6_box_remote_to_cat"),
                                  ("extra_g96x72.json", "records", 1, "img_grid_1200x1600", "img_grid_1200x1600")],
    "LiquidAI/d1-Omni-2B": TEXT_PIC + [
        ("extra_g96x72.json", "records", 1, "img_grid_1200x1600", "img_grid_1200x1600"),
        ("red_arms.json", "arms", 19, "red_not_qnli_00", "arm_6_box_remote_to_cat"),
        ("requests_audio.json", "records", 55, "aud_cmd_lights", "spk_aud_supply_18q"),
        ("red_arms_audio.json", "arms", 12, "arm_g1_beeps_3to5", "arm_g2_cats_dogs"),
        ("card_omni.json", "records", 4, "card_omni_beeps", "card_omni_coco")],
}
# per model: media table -> the files whose requests' media it holds (minus the tables before it)
MEDIA = {
    "LiquidAI/d1-2B": [("media.json", ["requests.json", "red_arms.json"]), ("media_extra.json", ["extra_g96x72.json"])],
    "LiquidAI/d1-Omni-2B": [("media.json", ["requests.json", "red_arms.json"]),
                            ("media_extra.json", ["extra_g96x72.json"]),
                            ("media_audio.json", ["requests_audio.json", "red_arms_audio.json", "card_omni.json"])],
}
CARD_2B = ["card_ticket", "card_invoice", "card_coco", "card_batch_1", "card_batch_2"]
INLINE = {"card.json", "card_omni.json"}   # files that carry their records' media entries inline


def request_sha256(request):
    return hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def media_ids(records):
    """Every media id the records' requests name: their media, then their spoken questions' clips."""
    out = []
    for r in records:
        req = r["request"]
        ids = list(req.get("media") or [])
        for q in req["questions"].values():
            ins = q.get("instructions") if isinstance(q, dict) else None
            if isinstance(ins, dict) and ins.get("type") == "audio":
                ids.append(ins["media"])
        out.extend(m for m in ids if m not in out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--requests", required=True, help="the full requests.json (fixtures/rebuild_requests.py)")
    ap.add_argument("--media", default="fixtures/media.json")
    ap.add_argument("--out-dir", default="set")
    a = ap.parse_args()
    doc = json.load(open(a.requests, encoding="utf-8"))
    model = doc["model"].split("@")[0]
    if model not in LAYOUT:
        sys.exit(f"{a.requests}: model {doc['model']!r} is not one of {sorted(LAYOUT)}")
    recs = doc["records"]
    bad = [r["id"] for r in recs if request_sha256(r["request"]) != r["request_sha256"]]
    if bad:
        sys.exit(f"{len(bad)} requests differ from their request_sha256, e.g. {bad[0]}")
    files, pos = {}, 0
    for name, form, n, first, last in LAYOUT[model]:
        part = recs[pos:pos + n]
        pos += n
        ids = [r["id"] for r in part]
        if len(part) != n or ids[0] != first or ids[-1] != last:
            sys.exit(f"{name}: records {pos - n}..{pos - 1} are not {first} .. {last}")
        arms = [r["id"] for r in part if "base_id" in r]
        if (form == "arms") != (len(arms) == n) or (form == "records" and arms):
            sys.exit(f"{name}: expected {'only control arms' if form == 'arms' else 'no control arm'}")
        files[name] = part
    if pos != len(recs):
        sys.exit(f"{a.requests}: {len(recs)} records, the layout of {model} has {pos}")
    table = json.load(open(a.media, encoding="utf-8"))["media"]
    base = os.path.dirname(os.path.relpath(os.path.abspath(a.media)))

    def entry(mid):
        v = {k: x for k, x in table[mid].items() if k not in ("file", "stored", "made_by", "format")}
        return {"kind": v.pop("kind"), "path": os.path.join(base, table[mid]["file"]), **v}

    out = {}
    for name, form, *_ in LAYOUT[model]:
        out[name] = files[name] if form == "arms" else {"records": files[name]}
    if model == "LiquidAI/d1-2B":
        by_id = {r["id"]: r for r in files["requests.json"]}
        out["card.json"] = {"records": [by_id[i] for i in CARD_2B]}
    for name in INLINE & set(out):
        out[name]["media"] = {m: entry(m) for m in media_ids(out[name]["records"])}
    seen = []
    for name, srcs in MEDIA[model]:
        want = [m for f in srcs for m in media_ids(files[f])]
        out[name] = {m: entry(m) for m in table if m in want and m not in seen}
        seen.extend(out[name])
    if model == "LiquidAI/d1-2B":
        out["media_all.json"] = {**out["media.json"], **out["media_extra.json"]}
    unknown = sorted({m for f in files.values() for m in media_ids(f)} - set(table))
    if unknown:
        sys.exit(f"media ids not in {a.media}: {unknown}")
    there = [n for n in out if os.path.exists(os.path.join(a.out_dir, n))]
    if there:
        sys.exit(f"{a.out_dir}/ already holds {there}; pass another --out-dir")
    os.makedirs(a.out_dir, exist_ok=True)
    for name, d in out.items():
        with open(os.path.join(a.out_dir, name), "w", encoding="utf-8") as f:
            json.dump(d, f, indent=1, ensure_ascii=False)
            f.write("\n")
    missing = sorted({e["path"] for n, d in out.items() if n.startswith("media") for e in d.values()
                      if not os.path.exists(e["path"])})
    print(json.dumps({"model": model, "files": {n: (len(d) if isinstance(d, list) else len(d.get("records", d)))
                                                for n, d in out.items()},
                      "media_files_missing": missing}, indent=1))


if __name__ == "__main__":
    main()
