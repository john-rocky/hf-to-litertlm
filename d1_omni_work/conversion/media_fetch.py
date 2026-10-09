"""Round 2 step 1 (media): the fixture's image and copied audio files -> K/fixtures/{images,audio,media_measure_only}/,
K/fixtures/media_fetch.json

    cd K
    venv-ref/bin/python -I scripts/media_fetch.py --download-dir <new empty dir outside K>

1. Our three CC0 Commons photos (img_dogs_01, img_cat_02, img_bike_03): the original from the `url` of
   logs/commons_imageinfo.json (round 1's Commons API read) into --download-dir; its sha1 asserted equal to that
   file's `sha1` and to fx_define.IMAGES' `commons_sha1`; EXIF orientation applied (PIL ImageOps.exif_transpose, the
   step transformers.image_utils.load_image takes), RGB, then the long side resized to 384 px with
   Image.resize((w, h), Image.LANCZOS), the short side round(short * 384 / long); saved as PNG (no ICC profile
   written) to fixtures/images/<id>.png.
2. Copies from the Core AI lane d1d (~/code/coreai/_d1_omni, read only), each sha256 asserted against d1d's own
   manifest: audio/aud_01..03.wav -> fixtures/audio/, images/img_01..03.png -> fixtures/images/,
   images/coco_000000039769.jpg and audio/card_1.flac -> fixtures/media_measure_only/ (measurement only, gitignored).
The downloads are untrusted data: they live only in --download-dir (a new empty directory) and are only decoded by PIL.
"""
import argparse
import hashlib
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
import d1_src as S  # noqa: E402

K = S.K
D1D = Path.home() / "code/coreai/_d1_omni"
LONG_SIDE = 384
UA = "d1-omni-litert-port/0.1 (fixture fetch; one request per file)"


def sha1_file(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def commons_pages():
    doc = json.loads((K / "logs/commons_imageinfo.json").read_text())
    return {p["title"]: p["imageinfo"][0] for p in doc["query"]["pages"].values()}


def fetch(url, dst, tries=4):
    """One GET; on HTTP 429 wait Retry-After (default 30 s) and try again (upload.wikimedia.org rate-limits)."""
    import urllib.error

    req = urllib.request.Request(url, headers={"User-Agent": UA})
    t0 = time.time()
    for i in range(tries):
        try:
            tmp = dst.with_name(dst.name + ".part")
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            tmp.rename(dst)
            return round(time.time() - t0, 1)
        except urllib.error.HTTPError as e:
            if e.code != 429 or i + 1 == tries:
                raise
            wait = int(e.headers.get("Retry-After") or 30)
            print(f"HTTP 429 on {url[:80]}; retry in {wait} s", flush=True)
            time.sleep(min(wait, 120))


def our_images(download_dir, fx_images):
    from PIL import Image, ImageOps, features

    pages = commons_pages()
    out = {}
    (K / "fixtures/images").mkdir(parents=True, exist_ok=True)
    for im in fx_images:
        info = pages[im["title"]]
        assert info["sha1"] == im["commons_sha1"], (im["id"], info["sha1"], im["commons_sha1"])
        src = Path(download_dir) / f"{im['id']}.orig.jpg"
        seconds = None
        if not src.exists():
            seconds = fetch(info["url"], src)
        sha1 = sha1_file(src)
        assert sha1 == info["sha1"], (im["id"], sha1, info["sha1"])
        assert src.stat().st_size == info["size"], (im["id"], src.stat().st_size, info["size"])
        with Image.open(src) as raw:
            raw.load()
            fmt, mode0, size0 = raw.format, raw.mode, raw.size
            orientation = raw.getexif().get(0x0112)
            icc = raw.info.get("icc_profile")
            up = ImageOps.exif_transpose(raw)
            rgb = up.convert("RGB")
        w0, h0 = rgb.size
        if w0 >= h0:
            size = (LONG_SIDE, round(h0 * LONG_SIDE / w0))
        else:
            size = (round(w0 * LONG_SIDE / h0), LONG_SIDE)
        small = rgb.resize(size, Image.LANCZOS)
        dst = K / "fixtures/images" / f"{im['id']}.png"
        small.save(dst, format="PNG")
        out[im["id"]] = {
            "file": f"fixtures/images/{im['id']}.png", "sha256": S.sha256_file(dst), "bytes": dst.stat().st_size,
            "px": list(small.size), "mode": small.mode,
            "made": {"from_url": info["url"], "commons_page": info["descriptionurl"], "commons_sha1": sha1,
                     "original_bytes": src.stat().st_size, "original_px": list(size0), "original_format": fmt,
                     "original_mode": mode0, "exif_orientation": orientation, "upright_px": [w0, h0],
                     "icc_profile_in_original": None if icc is None else f"{len(icc)} B (not written to the PNG)",
                     "op": f"ImageOps.exif_transpose -> convert('RGB') -> resize {w0}x{h0} -> {size[0]}x{size[1]} "
                           f"(long side {LONG_SIDE}, short side round(short * {LONG_SIDE} / long))",
                     "call": "Image.resize(size, Image.LANCZOS); Image.save(path, format='PNG')",
                     "pillow": Image.__version__, "libjpeg_turbo": features.check_feature("libjpeg_turbo"),
                     "download_seconds": seconds},
        }
        print(im["id"], out[im["id"]]["px"], out[im["id"]]["sha256"][:12], flush=True)
    return out


def copy_checked(src, dst, sha):
    got = S.sha256_file(src)
    assert got == sha, (src, got, sha)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        assert S.sha256_file(dst) == sha, dst
    else:
        shutil.copyfile(src, dst)
    assert S.sha256_file(dst) == sha
    return {"file": str(dst.relative_to(K)), "from": str(src), "sha256": sha, "bytes": dst.stat().st_size}


def d1d_copies():
    amf = json.loads((D1D / "audio/manifest.json").read_text())
    imf = json.loads((D1D / "images/manifest.json").read_text())
    rec = json.loads((D1D / "fixtures/records.json").read_text())
    out = {"d1d_manifests": {"audio": {"path": str(D1D / "audio/manifest.json"),
                                       "sha256": S.sha256_file(D1D / "audio/manifest.json")},
                             "images": {"path": str(D1D / "images/manifest.json"),
                                        "sha256": S.sha256_file(D1D / "images/manifest.json")},
                             "records": {"path": str(D1D / "fixtures/records.json"),
                                         "sha256": S.sha256_file(D1D / "fixtures/records.json")}}}
    for cid in ("aud_01", "aud_02", "aud_03"):
        clip = amf["clips"][cid]
        out[cid] = copy_checked(D1D / clip["file"], K / "fixtures/audio" / f"{cid}.wav", clip["sha256"])
        out[cid].update({k: clip[k] for k in ("voice", "seed", "text", "seconds", "sample_rate", "channels",
                                              "sample_width_bytes", "frames")})
    by_id = {e["id"]: e for e in imf["images"]}
    for iid in ("img_01", "img_02", "img_03"):
        e = by_id[iid]
        out[iid] = copy_checked(D1D / e["file"], K / "fixtures/images" / f"{iid}.png", e["sha256"])
        out[iid]["manifest_entry"] = e
    card = rec["sources"]["card"]
    out["card_cats"] = copy_checked(D1D / card["image"]["path"],
                                    K / "fixtures/media_measure_only" / Path(card["image"]["path"]).name,
                                    card["image"]["sha256"])
    out["card_cats"].update({"url": card["image"]["url"], "license": card["image"]["license"]})
    out["card_topic"] = copy_checked(D1D / card["audio"]["path"],
                                     K / "fixtures/media_measure_only" / Path(card["audio"]["path"]).name,
                                     card["audio"]["sha256"])
    out["card_topic"].update({"url": card["audio"]["url"], "license": card["audio"]["license"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download-dir", required=True)
    a = ap.parse_args()
    dl = Path(a.download_dir).resolve()
    assert K not in dl.parents and dl != K, "keep downloads out of K"
    dl.mkdir(parents=True, exist_ok=True)
    import fx_define

    t0 = time.time()
    doc = {"written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "script": "scripts/media_fetch.py",
           "ours": our_images(dl, fx_define.IMAGES), "d1d": d1d_copies()}
    doc["seconds"] = round(time.time() - t0, 1)
    (K / "fixtures/media_fetch.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: (v if k != "d1d" else sorted(v)) for k, v in doc.items() if k != "ours"}, indent=1))


if __name__ == "__main__":
    main()
