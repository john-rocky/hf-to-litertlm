#!/usr/bin/env python3
"""Pick the NASA fixture photos by a fixed rule and write fixtures/nasa_select.json.

Rule (2026-10-24): for each query term, search
`https://images-api.nasa.gov/search?q=<term>&media_type=image` and take, in result order,
the first image whose original is at least 1024 px wide and whose metadata carries no
credit other than NASA (description `Credit:` lines, photographer, secondary creator).
The chosen file is `~orig.jpg` (or `~large.jpg` when no original JPEG is listed).

Network only, standard library only. Usage: python3 -I select_nasa.py [out.json]
"""
import json
import re
import struct
import sys
import urllib.parse
import urllib.request

TERMS = ["astronauts group", "airplane runway", "launch pad trucks", "city aerial"]
MIN_WIDTH = 1024
# Organizations or phrases that mark a credit other than NASA (JPL / center names are NASA).
FOREIGN = re.compile(
    r"(?i)\b(esa|jaxa|csa|roscosmos|asi|dlr|cnes|isro|copernicus|google|esri|maxar|digitalglobe|"
    r"planet labs|getty|reuters|associated press|\bap\b|afp|spacex|boeing|lockheed|northrop|"
    r"united launch alliance|ula|blue origin|courtesy|copyright|©|all rights reserved)\b"
)
CREDIT = re.compile(r"(?i)(?:image|photo)?\s*credits?\s*:\s*([^\n]*)")


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.load(resp)


def foreign_credit(meta):
    texts = []
    desc = meta.get("description") or ""
    texts += CREDIT.findall(desc)
    for key in ("photographer", "secondary_creator"):
        if meta.get(key):
            texts.append(str(meta[key]))
    if re.search(r"(?i)courtesy|copyright|©", desc):
        return "description: courtesy/copyright"
    for t in texts:
        if FOREIGN.search(t):
            return t.strip()[:120]
    return None


def jpeg_size(url):
    """(width, height) from the SOF marker of the first 256 KB (metadata.json often has no size)."""
    req = urllib.request.Request(url, headers={"Range": "bytes=0-262143"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        b = resp.read()
    i = 2
    while i + 9 < len(b):
        if b[i] != 0xFF:
            i += 1
            continue
        marker = b[i + 1]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3):
            h, w = struct.unpack(">HH", b[i + 5:i + 9])
            return w, h
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        i += 2 + struct.unpack(">H", b[i + 2:i + 4])[0]
    return None, None


def image_width(nasa_id):
    q = urllib.parse.quote(nasa_id)
    files = [i["href"] for i in get_json(f"https://images-api.nasa.gov/asset/{q}")["collection"]["items"]]
    pick = next((f for f in files if f.lower().endswith("~orig.jpg")), None) or next(
        (f for f in files if f.lower().endswith("~large.jpg")), None)
    if not pick:
        return None, None, None
    pick = pick.replace("http://", "https://")
    width, height = jpeg_size(pick)
    return pick, width, height


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "nasa_select.json"
    chosen, log = [], []
    for term in TERMS:
        url = "https://images-api.nasa.gov/search?" + urllib.parse.urlencode({"q": term, "media_type": "image"})
        items = get_json(url)["collection"]["items"]
        for rank, it in enumerate(items):
            meta = it["data"][0]
            nid = meta["nasa_id"]
            why = foreign_credit(meta)
            if why:
                log.append({"term": term, "rank": rank, "nasa_id": nid, "skip": f"credit: {why}"})
                continue
            href, w, h = image_width(nid)
            if not href or not w or w < MIN_WIDTH:
                log.append({"term": term, "rank": rank, "nasa_id": nid, "skip": f"width {w} href {bool(href)}"})
                continue
            chosen.append({"term": term, "rank": rank, "nasa_id": nid, "title": meta.get("title"),
                           "date_created": meta.get("date_created"), "center": meta.get("center"),
                           "photographer": meta.get("photographer"), "description": meta.get("description"),
                           "url": href, "width": w, "height": h, "license": "NASA / public domain"})
            break
        else:
            log.append({"term": term, "skip": "no item passed"})
    json.dump({"rule": __doc__.split("\n\n")[1], "chosen": chosen, "skipped": log},
              open(out_path, "w"), indent=1, ensure_ascii=False)
    for c in chosen:
        print(c["term"], "->", c["nasa_id"], c["width"], "x", c["height"], c["url"])


if __name__ == "__main__":
    main()
