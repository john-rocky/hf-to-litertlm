#!/usr/bin/env python3
"""Round 5b: the public fixtures of the upload folder (staging/upload/fixtures) from the work directory's fixtures and references.

    venv-host/bin/python scripts/r5b_fixtures.py       -> staging/upload/fixtures/{requests_public.json, photos.json,
                                                          reference_probs.json, images/*.png (drawn), LICENSE-SemIf-MIT.txt}
                                                          results/r5b_fixtures.json (per-record license decision, checks)

Per record (memory public-fixtures-by-reference): written for the fixtures (own_*, ho_t* / ho_j* / ho_l*, and the
question text of every picture record) -> with its request; SemIf authored144 (MIT) -> with its request and the MIT
notice; drawn pictures (CC0-1.0, drawn with Pillow primitives) -> the PNG files; CC0 photographs from Wikimedia Commons
(CC0-1.0 in the fixture's metadata at fetch time and on Commons on 2026-10-21) -> the PNG the checks read (images/) and
the pinned JPEG thumbnail it was decoded from (photos/), with page, author, license and both sha256 in photos.json
(decided 2026-10-21: the thumbnail of photo_cat drifted on Commons, so a URL + sha256 reference no longer
restores the pixels; fixtures/fetch_photos.py only checks whether Commons still serves the pinned bytes); the four records whose invented names were found in use on 2026-10-04 (own_t02, own_t15, own_t16,
own_j01) and the red arm built on own_j01 -> id, gold, note and the sha256 of the request only (as the fixture's first
publication did). Every published request is asserted equal to the work directory's request (sha256 of the canonical JSON)."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

K = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K / "scripts"))
import r5b_sets as S  # noqa: E402

FX = K / "staging/upload/fixtures"
WITHHELD = {"own_t02", "own_t15", "own_t16", "own_j01"}
WITHHELD_SHA256 = {          # the fixture's first publication (make_fixtures.py WITHHELD_REQUEST_SHA256)
    "own_t02": "727341c27bc2ff3570eee6790dc40e572091414c4d587031377940b8c6e23111",
    "own_t15": "03ea2157d2721f94b568632995fe50ab5b1bc702c75e5074abc4845c46687cd9",
    "own_t16": "f86f4edabf30e7ffdd91b8480b886052baef417cfd804012e3d4c34c351eed0b",
    "own_j01": "d963c4013aa890a4f2411b2d66a122068a2a73eafb2828a0b2c87bb28fb595e0",
}
ZOO_FIXTURE = Path(os.environ.get("CLEF_ZOO_FIXTURE", "fixtures-clef-flash.json"))   # models/clef-flash/ of github.com/john-rocky/coreai-model-zoo
PHOTOS_SRC = [Path(d) for d in os.environ.get("CLEF_PHOTOS_SRC", "photos_src:heldout/photos_src").split(":")]   # the pinned thumbnails as fetched
COMMONS_META = K / "results/r5b_commons_meta.json"      # Commons imageinfo extmetadata of the six files, 2026-10-21


def canon_sha(request: dict) -> str:
    return hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rgb_sha(p: Path) -> str:
    from PIL import Image

    return hashlib.sha256(Image.open(p).convert("RGB").tobytes()).hexdigest()


def license_of(rec: dict, red_base: str | None = None) -> str:
    src = rec["source"]
    if src == "semif_authored144":
        return "MIT (SemIf authored144, LICENSE-SemIf-MIT.txt)"
    return "written for these fixtures; free to use (CC0-1.0)" if not red_base else \
        f"one word changed in {red_base}; as {red_base}"


def main() -> int:
    FX.mkdir(parents=True, exist_ok=True)
    (FX / "images").mkdir(exist_ok=True)
    fx = json.loads((K / "fixtures/records.json").read_text())
    ho = json.loads((K / "fixtures/heldout/records.json").read_text())
    red = json.loads((K / "fixtures/red_arms.json").read_text())
    meta = {x["name"]: (x, K / "fixtures") for x in json.loads((K / "fixtures/images_meta.json").read_text())["images"]}
    meta |= {x["name"]: (x, K / "fixtures/heldout") for x in
             json.loads((K / "fixtures/heldout/images_meta.json").read_text())["images"]}
    zoo = json.loads(ZOO_FIXTURE.read_text())
    semif = zoo["sources"]["semif_authored144"]
    decisions, out_records = [], []

    def add(rec: dict, set_name: str, base: str | None = None):
        rid = rec["id"]
        withheld = rid in WITHHELD or base in WITHHELD
        sha = canon_sha(rec["request"])
        if rid in WITHHELD_SHA256:
            assert sha == WITHHELD_SHA256[rid], (rid, "the work directory's request is not the one withheld in 2026-10")
        item = {"id": rid, "set": set_name, "source": rec["source"] if base is None else "red_arm",
                "request": {"withheld": "an invented name in it was found in use on 2026-10-04", "sha256": sha}
                if withheld else rec["request"], "request_sha256": sha}
        if rec.get("image_files"):
            item["image_files"] = rec["image_files"]
        for k in ("gold", "note"):
            if k in rec:
                item[k] = rec[k]
        if base is not None:
            item["base_id"] = base
        item["license"] = "not published (see request)" if withheld else license_of(rec, base)
        out_records.append(item)
        decisions.append({"id": rid, "set": set_name, "published": not withheld, "license": item["license"],
                          "request_sha256": sha})

    for rec in fx["records"]:
        add(rec, "fixture")
    for rec in ho["records"]:
        add(rec, "heldout")
    arms = {a["arm_id"]: a for a in red["arms"]}
    for rec in red["records"]:
        add(dict(rec, source=rec.get("source", "red_arm")), "red_arm", base=arms[rec["id"]]["base_id"])

    images, photos = [], []
    commons = {pg["title"]: pg["imageinfo"][0]["extmetadata"]
               for pg in json.loads(COMMONS_META.read_text())["query"]["pages"].values()}
    fetched = {}
    for d in PHOTOS_SRC:
        fetched |= {k: v | {"dir": d} for k, v in json.loads((d / "fetch_record.json").read_text()).items()}
    users = {}
    for r in out_records:
        for f in r.get("image_files", []):
            users.setdefault(f, []).append(r["id"])
    (FX / "photos").mkdir(exist_ok=True)
    for name, (m, base_dir) in sorted(meta.items()):
        src = base_dir / m["path"]
        assert sha256_file(src) == m["sha256"] and rgb_sha(src) == m["rgb_sha256"], name
        if "source" in m:            # a Commons photograph (CC0): the PNG the checks read + the pinned thumbnail
            src_jpg = fetched[name]["dir"] / f"{name}.jpg"
            assert sha256_file(src_jpg) == m["source"]["thumb_jpeg_sha256"], (name, "pinned thumbnail changed")
            assert rgb_sha(src_jpg) == m["rgb_sha256"], (name, "the PNG is not the decode of the pinned thumbnail")
            em = commons[m["source"]["commons_title"].replace("_", " ")]
            lic = {"fetch_record_2026-10-03": fetched[name]["license_short_name"], "images_meta": m["license"],
                   "commons_2026-10-21": em["LicenseShortName"]["value"]}
            assert lic["fetch_record_2026-10-03"] == "CC0" and lic["images_meta"] == "CC0-1.0" and \
                lic["commons_2026-10-21"] == "CC0", (name, lic)
            png, jpg = FX / "images" / Path(m["path"]).name, FX / "photos" / f"{name}.jpg"
            shutil.copyfile(src, png)
            shutil.copyfile(src_jpg, jpg)
            strip = lambda v: re.sub(r"<[^>]+>", "", v or "").strip()   # noqa: E731
            photos.append({"name": name, "records": users.get(Path(m["path"]).name, []),
                           "file": f"images/{png.name}", "bytes": png.stat().st_size, "sha256": m["sha256"],
                           "size": m["size"], "rgb_sha256": m["rgb_sha256"],
                           "thumbnail": {"file": f"photos/{jpg.name}", "bytes": jpg.stat().st_size,
                                         "sha256": m["source"]["thumb_jpeg_sha256"], "url": m["source"]["thumb_url"]},
                           "file_page": m["source"]["file_page"], "commons_title": m["source"]["commons_title"],
                           "author": strip(em.get("Artist", {}).get("value")),
                           "credit": strip(em.get("Credit", {}).get("value")),
                           "license": "CC0-1.0", "license_checks": lic,
                           "decode": "Pillow JPEG decode -> RGB -> PNG; the PNG's RGB = the thumbnail's decode"})
        else:
            dst = FX / "images" / Path(m["path"]).name
            shutil.copyfile(src, dst)
            images.append({"name": name, "file": f"images/{dst.name}", "size": m["size"], "png_sha256": m["sha256"],
                           "rgb_sha256": m["rgb_sha256"], "license": m["license"],
                           "drawn_by": f"{m['generator']['function']}() of make_fixtures.py"})
    doc = {"what": "the requests of the checks of the clef-flash LiteRT files: 186 fixture records, 30 held-out "
                   "records and 3 red arms, each a request body of the SystemOne-compatible request shape",
           "model": {"repo": "Cloudflare/clef-flash", "revision": "f0b7d3cd65296abfe459e6c228b618f5d8b26d2f"},
           "origin": "written for an earlier port of the same model (github.com/john-rocky/coreai-model-zoo, "
                     "conversion/clef_flash/make_fixtures.py and models/clef-flash/fixtures-clef-flash.json)",
           "pictures": "a record's image_files name its picture in images/ (drawn pictures and CC0 photographs; "
                       "photos.json gives each photograph's source, author and license)",
           "request_sha256": "sha256 of json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(',', ':')) "
                             "as UTF-8",
           "withheld": sorted(r["id"] for r in out_records if "withheld" in r["request"]),
           "semif_authored144": {k: semif[k] for k in ("upstream_repo", "upstream_path", "upstream_commit",
                                                       "file_sha256", "rows", "mapping", "license")},
           "drawn_images": images, "records": out_records}
    (FX / "requests_public.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    (FX / "photos.json").write_text(json.dumps({
        "what": "the six CC0 photographs of the picture records: images/<file> is the PNG the checks read (the pinned "
                "Commons thumbnail decoded by Pillow), photos/<name>.jpg the pinned thumbnail itself; fetch_photos.py "
                "checks whether Commons still serves the pinned thumbnails",
        "photos": photos}, indent=1, ensure_ascii=False) + "\n")
    (FX / "LICENSE-SemIf-MIT.txt").write_text(semif["license_notice"].rstrip("\n") + "\n")
    # reference probabilities: every run the checks compare with (no text, no token ids)
    runs = S.load_runs()
    refs = []
    for r in runs:
        refs.append({"key": r["key"], "set": r["set"], "id": r["id"], "arm": r["arm"], "tokens": r["tokens"],
                     "length": r["length"],
                     "questions": [{"question_id": q["question_id"], "type": q["type"], "option_ids": q["option_ids"],
                                    "probs": q["probs"], "argmax": q["argmax_id"], "near_tie": q["near_tie"]}
                                   for q in r["row"]["questions"]]})
    (FX / "reference_probs.json").write_text(json.dumps({
        "what": "the provider's own code (joint_schema_model.py systemone() at f0b7d3cd) in float32 on the CPU: "
                "per run (a record at an arm: text, or the picture at a 256 / 448 px tile) and question, the option "
                "ids in the head's order, the probabilities, the most likely option and whether the two most likely "
                "are 0.02 or less apart (near_tie). Python 3.12.11, torch 2.9.0, transformers 5.17.0, 10 threads.",
        "runs": refs}, indent=1) + "\n")
    checks = {"records": len(out_records), "published": sum("withheld" not in r["request"] for r in out_records),
              "withheld": doc["withheld"], "drawn_images": len(images), "photos_included": len(photos),
              "reference_runs": len(refs)}
    (K / "results/r5b_fixtures.json").write_text(json.dumps({"checks": checks, "decisions": decisions}, indent=1) + "\n")
    print(json.dumps(checks, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
