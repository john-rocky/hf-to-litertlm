"""Step 3: the d1-omni fixture set -> fixtures/requests.json, fixtures/README.md, fixtures/LICENSE-SemIf-MIT.txt

    HF_HUB_DISABLE_XET=1 venv-ref/bin/python scripts/fx_define.py          # build (refuses to overwrite a different file)
    HF_HUB_DISABLE_XET=1 venv-ref/bin/python scripts/fx_define.py --tune   # print own_long_3400 state tokens vs N_FILLER
    HF_HUB_DISABLE_XET=1 venv-ref/bin/python scripts/fx_define.py --extend # round 2: v1 -> v2 (see extend())

Record = {id, source, publishable, request: {state, questions}, media, gold, note, provenance}; `request` is the
argument pair of the provider's system_one(state, questions, images, audio) (the media go in `media`, fetched or
synthesised in round 2: sha256 null here). The Kev records are copied with id / source / request / gold / note /
provenance unchanged; `publishable` and `media: null` are added.

--extend (round 2): requests.json is first copied to requests.v1.json (and fixture_summary.json to
fixture_summary.v1.json); then v2 = v1 with (a) the media of our records filled from fixtures/media_fetch.json
(scripts/media_fetch.py) and fixtures/audio/kokoro_manifest.json (scripts/kokoro_synth.py): `media.ref` becomes the
file's path relative to K, `media.sha256` that file's sha256; (b) nine records of the Core AI lane d1d
(~/code/coreai/_d1_omni/fixtures/records.json, read only) added under the same ids with request / gold / note copied:
aud_01..03, long_3400, card_batch_00 / 01, img_01..03; their files copied by media_fetch.py (sha256 = d1d's manifests).
publishable for the d1d images = the pool's licence is CC0 (a FLUX output is not CC0 -> false). Then README.md and
results/fixture_summary.json (round 1's keys + shared_with_d1d + the measured media prefix lengths).
"""
import argparse
import copy
import hashlib
import json
import shutil
import sys
import time

import d1_src as S
import own_long

KEV = S.K.parent / "kev_work/fixtures"
KEV_REQUESTS_SHA256 = "dfe55fb145df7a3967ed7213ae24d67b5d4ec42da0315fdb409d5e4b702e3a48"
SEMIF_LICENSE_SHA256 = "f765f2140f8507a8f0d81ec0fd2c4bd72fe6a066841ef27883ff876a76bf61be"
PUBLISHABLE_BY_SOURCE = {"tv4": True, "tv4x": False, "tv4s": True, "semif": True, "own": True, "red_arm": True}
README_LINES = {"text": [112, 133], "image": [135, 142], "audio": [144, 152]}  # README.md of the pinned revision

CARD_QUESTIONS = {  # README.md lines 113-132, verbatim
    "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
    "team": {"type": "choice", "instructions": "Which team should handle this?",
             "criteria": {"billing": "Charges, refunds, invoices", "technical": "App or site faults",
                          "fraud": "Suspected unauthorised use"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["Can wait", "Today", "Blocking the customer now"]},
}
CATS = {"type": "choice", "instructions": "How many cats are there?",
        "criteria": {"one": "One", "two": "Two", "more": "Three or more"}}  # README lines 137-141
TOPIC = {"type": "choice", "instructions": "What is the speaker talking about?",
         "criteria": {"food": "Food and meals", "travel": "Travel and transport", "weather": "The weather"}}  # 147-151

# CC0 images (Wikimedia Commons; licence, size and sha1 read from the Commons API on 2026-10-08, logs/commons_imageinfo.json)
IMAGES = [
    {"id": "img_dogs_01", "title": "File:Two French bulldogs swimming in life jackets.jpg",
     "url": "https://upload.wikimedia.org/wikipedia/commons/1/11/Two_French_bulldogs_swimming_in_life_jackets.jpg",
     "page": "https://commons.wikimedia.org/wiki/File:Two_French_bulldogs_swimming_in_life_jackets.jpg",
     "width": 3999, "height": 2249, "bytes": 5967525, "commons_sha1": "a07a019377c4043a22a683bba9d442c08e0e38d1",
     "author": "W.carter (Wikimedia Commons user)",
     "description": "Two French bulldogs swimming in life jackets in the water (Commons description; category '2 dogs')",
     "questions": {"count": {"type": "choice", "instructions": "How many dogs are in the photo?",
                             "criteria": {"one": "One", "two": "Two", "more": "Three or more"}},
                   "water": {"type": "noul", "instructions": "Are the animals in the water?"}},
     "gold": {"count": "two", "water": "true"}},
    {"id": "img_cat_02", "title": "File:Tabby cat with blue eyes-3336579.jpg",
     "url": "https://upload.wikimedia.org/wikipedia/commons/c/c7/Tabby_cat_with_blue_eyes-3336579.jpg",
     "page": "https://commons.wikimedia.org/wiki/File:Tabby_cat_with_blue_eyes-3336579.jpg",
     "width": 2877, "height": 3456, "bytes": 2114775, "commons_sha1": "181ba96777ca30bd43b6d49e1a794ebabc649d3a",
     "author": "AdinaVoicu (via Pixabay, CC0 on Commons)",
     "description": "Portrait of a black tabby and white cat with blue eyes (Commons description)",
     "questions": {"eyes": {"type": "choice", "instructions": "What colour are the animal's eyes?",
                            "criteria": {"blue": "Blue", "green": "Green", "brown": "Brown", "yellow": "Yellow"}},
                   "dog": {"type": "noul", "instructions": "Is there a dog in the photo?"}},
     "gold": {"eyes": "blue", "dog": "false"}},
    {"id": "img_bike_03", "title": "File:Bike on a snowy street in Quebec City at night.jpg",
     "url": "https://upload.wikimedia.org/wikipedia/commons/3/30/Bike_on_a_snowy_street_in_Quebec_City_at_night.jpg",
     "page": "https://commons.wikimedia.org/wiki/File:Bike_on_a_snowy_street_in_Quebec_City_at_night.jpg",
     "width": 7728, "height": 5162, "bytes": 17375849, "commons_sha1": "c10b8c7f290db323b0258a84ffedb11a121aa693",
     "author": "Wilfredor (Wikimedia Commons user)",
     "description": "A bicycle left chained to a post and buried by snowfall, photographed at night (Commons description)",
     "questions": {"vehicle": {"type": "choice", "instructions": "What vehicle is in the photo?",
                               "criteria": {"bicycle": "A bicycle", "car": "A car", "boat": "A boat", "bus": "A bus"}},
                   "daylight": {"type": "noul", "instructions": "Was the photo taken in daylight?"}},
     "gold": {"vehicle": "bicycle", "daylight": "false"}},
]
IMAGE_PREP = ("round 2: download `url`, check its sha1 against `commons_sha1`, convert to RGB, centre-crop to a square of "
              "side min(width, height), resize to 384x384 with PIL Image.LANCZOS, save as PNG; `sha256` is that PNG's")

# Kokoro-82M (hexgrad, Apache-2.0) scripts in the provider's audio domain (an English speaker asking an assistant)
KOKORO = {"repo": "hexgrad/Kokoro-82M", "revision": "f3ff3571791e39611d31c381e3a41a3af07b4987", "license": "apache-2.0"}
AUDIO_TOPIC = {"type": "choice", "instructions": "What is the speaker talking about?",
               "criteria": {"reservation": "Booking or changing a reservation", "weather": "The weather",
                            "food_order": "Ordering food", "music": "Playing music"}}
AUDIO_REQUEST = {"type": "noul", "instructions": "Is the speaker asking the assistant to do something?"}
AUDIO_URGENCY = {"type": "score", "instructions": "How urgent is the speaker's need?",
                 "criteria": ["Can wait", "Today", "Right now"]}
AUDIO = [
    {"id": "aud_reservation_01", "voice": "af_heart",
     "transcript": "Hi, could you move my dinner reservation for four from Friday to Saturday, same time? "
                   "No rush, any time this week is fine.",
     "gold": {"topic": "reservation", "request": "true", "urgency": "0"}},
    {"id": "aud_weather_02", "voice": "am_michael",
     "transcript": "Just so you know, the weather looks great tomorrow morning, so I'll ride my bike to work "
                   "instead of taking the bus.",
     "gold": {"topic": "weather", "request": "false", "urgency": "0"}},
    {"id": "aud_food_03", "voice": "af_sarah",
     "transcript": "I'd like to order two large pepperoni pizzas and a bottle of lemonade for delivery, please, "
                   "as fast as you can, we're starving.",
     "gold": {"topic": "food_order", "request": "true", "urgency": "2"}},
]
AUDIO_PREP = ("round 2: synthesise `transcript` with Kokoro-82M at `tts.revision`, voice `voice`, speed 1.0 (24 kHz float), "
              "resample to 16 kHz mono, write int16 PCM WAV; record the resampler (library + version) and the clip "
              "length; `sha256` is that WAV's")


def sha256_text(s):
    return hashlib.sha256(s.encode()).hexdigest()


def kev_records():
    path = KEV / "requests.json"
    assert S.sha256_file(path) == KEV_REQUESTS_SHA256, "kev_work/fixtures/requests.json changed"
    doc = json.loads(path.read_text())
    out = []
    for r in doc["records"]:
        rec = {"id": r["id"], "source": r["source"], "publishable": PUBLISHABLE_BY_SOURCE[r["source"]],
               "request": copy.deepcopy(r["request"]), "media": None, "gold": copy.deepcopy(r["gold"]),
               "note": r["note"], "provenance": copy.deepcopy(r["provenance"])}
        assert set(r) == {"id", "source", "request", "gold", "note", "provenance"}, sorted(r)
        out.append(rec)
    return doc, out


def card_records():
    src = {"file": "README.md", "repo": S.REPO, "revision": S.REV,
           "readme_sha256": S.sha256_file(S.SNAP / "README.md")}
    lic_card = "LFM Open License v1.0 (the provider's model card example; LICENSE shipped with any copy)"
    return [
        {"id": "card_text", "source": "card", "publishable": True,
         "request": {"state": "I was charged twice this month, please refund one of them.",
                     "questions": copy.deepcopy(CARD_QUESTIONS)},
         "media": None, "gold": {"refund": "true", "team": "billing", "urgency": None},
         "note": "the model card's text example, verbatim (3 questions in one call); gold: refund and team follow from "
                 "the text, urgency is not stated by the card or decided by the text (null)",
         "provenance": {**src, "lines": README_LINES["text"], "licence": lic_card}},
        {"id": "card_cats", "source": "card", "publishable": False,
         "request": {"state": None, "questions": {"cats": copy.deepcopy(CATS)}},
         "media": {"kind": "image", "ref": "http://images.cocodataset.org/val2017/000000039769.jpg",
                   "license": "COCO val2017 image (COCO terms: annotations CC BY 4.0, images under their Flickr "
                              "licences; this image's own licence id not read in round 1) - measurement only, never "
                              "in a published file",
                   "sha256": None, "prep": "round 2: download as the card does (transformers.image_utils.load_image), "
                                           "no resize: the provider's layout() decides the crops"},
         "gold": {"cats": "two"},
         "note": "the model card's image example (state None = the photo is the whole state); gold from the card's "
                 "comment 'two cats on a sofa'",
         "provenance": {**src, "lines": README_LINES["image"], "licence": "card example; the image itself is COCO"}},
        {"id": "card_topic", "source": "card", "publishable": False,
         "request": {"state": "Voice note from a user.", "questions": {"topic": copy.deepcopy(TOPIC)}},
         "media": {"kind": "audio", "ref": "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac",
                   "license": "LibriSpeech (CC BY 4.0) via the Narsil/asr_dummy dataset - measurement only, never in "
                              "a published file",
                   "sha256": None, "prep": "round 2: read as the card does (soundfile, dtype int16); the clip's "
                                           "sample rate is checked to be 16 kHz"},
         "gold": {"topic": None},
         "note": "the model card's audio example; no gold (the card prints the answer without stating it)",
         "provenance": {**src, "lines": README_LINES["audio"], "licence": "card example; the clip is LibriSpeech"}},
    ]


def own_long_record():
    st = own_long.state()
    return {"id": "own_long_3400", "source": "own", "publishable": True,
            "request": {"state": st, "questions": copy.deepcopy(own_long.QUESTIONS)}, "media": None,
            "gold": dict(own_long.GOLD),
            "note": "written for this port: an invented invoicing SaaS's incident review draft (timeline log, customer "
                    "ticket excerpts, follow-up actions, metrics JSON) sized to ~3,400 state tokens = the d1-3B card's "
                    "'3.4k token state' column; invented names only; gold = the answer the text was written to have",
            "provenance": {"script": "scripts/own_long.py", "n_filler": own_long.N_FILLER,
                           "state_sha256": sha256_text(st)}}


def image_records(vision):
    out = []
    for im in IMAGES:
        orig = vision.layout(im["width"], im["height"])
        sq = vision.layout(384, 384)
        tiles = orig["grid"][0] * orig["grid"][1] if orig["tiled"] else 0
        th, tw = orig["thumbnail"]
        out.append({
            "id": im["id"], "source": "img", "publishable": True,
            "request": {"state": None, "questions": copy.deepcopy(im["questions"])},
            "media": {"kind": "image", "ref": im["url"], "page": im["page"], "title": im["title"],
                      "license": "CC0 1.0 Universal (Public Domain Dedication), "
                                 "https://creativecommons.org/publicdomain/zero/1.0/ - Commons LicenseShortName 'CC0', "
                                 "AttributionRequired false",
                      "author": im["author"], "sha256": None, "commons_sha1": im["commons_sha1"],
                      "original": {"width": im["width"], "height": im["height"], "bytes": im["bytes"]},
                      "prep": IMAGE_PREP,
                      "layout_384_square": {**sq, "prefix_tokens": (384 // 32) * (384 // 32)},
                      "layout_original": {**orig, "prefix_tokens": tiles * 256 + (th // 32) * (tw // 32)}},
            "gold": dict(im["gold"]),
            "note": "CC0 photo; gold from the Commons description (to be checked by eye when the file is fetched in "
                    "round 2). The 384x384 form is 1 crop of 24x24 patches = 144 prefix tokens under the provider's "
                    "layout(); the original size would be tiled (see media.layout_original).",
            "provenance": {"commons_api": "logs/commons_imageinfo.json (2026-10-08)",
                           "description": im["description"]}})
    return out


def audio_records():
    out = []
    for a in AUDIO:
        tsha = sha256_text(a["transcript"])
        out.append({
            "id": a["id"], "source": "aud", "publishable": True,
            "request": {"state": None, "questions": {"topic": copy.deepcopy(AUDIO_TOPIC),
                                                     "request": copy.deepcopy(AUDIO_REQUEST),
                                                     "urgency": copy.deepcopy(AUDIO_URGENCY)}},
            "media": {"kind": "audio", "ref": f"kokoro:{a['voice']}:{tsha}", "transcript": a["transcript"],
                      "voice": a["voice"], "tts": dict(KOKORO),
                      "license": "synthesised for this port with Kokoro-82M (Apache-2.0) from a script written for "
                                 "this port",
                      "sha256": None, "format": "WAV, 16 kHz, mono, int16 (round 2)", "prep": AUDIO_PREP,
                      "words": len(a["transcript"].split())},
            "gold": dict(a["gold"]),
            "note": "own script in the provider's audio domain (an English speaker asking an assistant; state None -> "
                    "{} as the provider trained audio); gold = the answer the script was written to have",
            "provenance": {"transcript_sha256": tsha}})
    return out


def validate(records, prompt):
    """Every question through the provider's as_question(); a rejection is recorded, never patched."""
    rejected, n = [], 0
    for r in records:
        for qid, q in r["request"]["questions"].items():
            n += 1
            try:
                prompt.as_question(q)
            except ValueError as e:
                rejected.append({"id": r["id"], "qid": qid, "error": str(e)})
    return n, rejected


def tune():
    tok = S.tokenizer()
    P = S.provider_prompt()
    for n in range(0, 121, 4):
        st = own_long.state(n)
        k = len(tok(P.escape(st), add_special_tokens=False)["input_ids"])
        print(n, k)


# ---------------------------------------------------------------- round 2: v2 = v1 + media + the d1d records

D1D = S.K.parent.parent / "coreai/_d1_omni"
SHARED_WITH_D1D_NEW = ["aud_01", "aud_02", "aud_03", "long_3400", "card_batch_00", "card_batch_01",
                       "img_01", "img_02", "img_03"]
IMAGE_PREP_V2 = ("round 2 (scripts/media_fetch.py): the original from the Commons `source_url`, its sha1 checked "
                 "against `commons_sha1`; PIL ImageOps.exif_transpose, convert('RGB'), the long side resized to "
                 "384 px with Image.resize(size, Image.LANCZOS) (short side round(short * 384 / long)), saved as "
                 "PNG; `sha256` is that PNG's. The provider's layout() then reads it whole (one crop).")
AUDIO_PREP_V2 = ("round 2 (scripts/kokoro_synth.py, the Core AI lane d1d's recipe): KPipeline(lang_code 'a'), "
                 "speed 1.0, torch.manual_seed(0), 24 kHz float -> int16 WAV -> ffmpeg -ar 16000 -ac 1 -sample_fmt "
                 "s16; `sha256` is that WAV's")


GOLD_CHECK = {  # how each media record's gold was checked in round 2 (the files looked at / the scripts read)
    "img": "the PNG looked at by eye (Read) in round 2: the gold holds",
    "aud": "the script (transcript) read in round 2: the gold holds",
    "card_cats": "the jpg looked at by eye in round 2: two cats (the gold holds)",
    "card_topic": "no gold (the card prints the answer without stating it)",
    "d1d_img": "d1d's gold; the PNG looked at by eye in round 2: the gold holds",
    "d1d_aud": "d1d's gold; d1d's script (transcript) read in round 2: the gold holds",
    "d1d_long": "d1d's gold, copied (text record)",
    "d1d_card": "d1d's gold (null), copied (text record)",
}


def media_prefix(vision, kind, path):
    """The provider's prefix length for one file: layout() crops (image), n // 160 then three stride-2 steps (audio)."""
    if kind == "image":
        from PIL import Image

        with Image.open(path) as im:
            w, h = im.convert("RGB").size
        plan = vision.layout(w, h)
        n = 0
        if plan["tiled"]:
            n += plan["grid"][0] * plan["grid"][1] * (vision.TILE // 16) ** 2 // 4
        th, tw = plan["thumbnail"]
        return n + (th // 16) * (tw // 16) // 4, {**plan, "px": [w, h]}
    import soundfile as sf

    info = sf.info(str(path))
    n = max(min(info.frames, 30 * 16000), 8000)
    t = n // 160
    for _ in range(3):
        t = (t + 2 - 3) // 2 + 1
    return t, {"samples": info.frames, "sample_rate": info.samplerate, "channels": info.channels,
               "seconds": round(info.frames / info.samplerate, 4), "subtype": info.subtype}


def v2_media_ours(rec, mf, km, vision):
    """Fill our record's media from the media_fetch / kokoro manifests (v1 fields kept unless superseded)."""
    m = copy.deepcopy(rec["media"])
    rid = rec["id"]
    if rec["source"] == "img":
        e = mf["ours"][rid]
        src = m.pop("ref")
        m.pop("layout_384_square", None)
        p, lay = media_prefix(vision, "image", S.K / e["file"])
        m.update({"ref": e["file"], "source_url": src, "sha256": e["sha256"], "bytes": e["bytes"], "px": e["px"],
                  "prep": IMAGE_PREP_V2, "made": e["made"], "layout": lay, "prefix_tokens": p})
        note = (rec["note"].split(" The 384x384 form")[0].replace(
            "(to be checked by eye when the file is fetched in round 2)",
            "(checked by eye on the round-2 PNG: it matches)") +
                f" Round 2 form: long side 384 px ({e['px'][0]}x{e['px'][1]}), one crop, {p} prefix tokens under the "
                "provider's layout(); the original size would be tiled (media.layout_original).")
        return m, note
    if rec["source"] == "aud":
        c = km["clips"][rid]
        p, lay = media_prefix(vision, "audio", S.K / c["file"])
        m["tts"] = {**m["tts"], "revision_used": km["hub_revision_in_cache"]}
        m.update({"ref": c["file"], "sha256": c["sha256"], "bytes": c["bytes"], "prep": AUDIO_PREP_V2,
                  "format": "WAV, 16 kHz, mono, int16", "seconds": c["seconds"], "frames": c["frames"],
                  "synth": {**km["synth"], "kokoro_version": km["kokoro_version"], "torch": km["torch"],
                            "ffmpeg": km["ffmpeg"], "voice_file_sha256": c["voice_sha256"],
                            "voice_source": c["voice_source"], "phonemes": c["phonemes"]},
                  "audio_info": lay, "prefix_tokens": p})
        return m, rec["note"]
    if rid == "card_cats":
        e = mf["d1d"]["card_cats"]
        p, lay = media_prefix(vision, "image", S.K / e["file"])
        m.update({"ref": e["file"], "source_url": m["ref"], "sha256": e["sha256"], "bytes": e["bytes"],
                  "prep": "round 2: the card's URL as fetched by the Core AI lane d1d (images/coco_000000039769.jpg), "
                          "copied with its sha256 checked; read with transformers.image_utils.load_image as the card "
                          "does, no resize (the provider's layout() decides the crops)",
                  "copied_from": e["from"], "px": lay["px"], "layout": lay, "prefix_tokens": p})
        return m, rec["note"]
    if rid == "card_topic":
        e = mf["d1d"]["card_topic"]
        p, lay = media_prefix(vision, "audio", S.K / e["file"])
        m.update({"ref": e["file"], "source_url": m["ref"], "sha256": e["sha256"], "bytes": e["bytes"],
                  "prep": "round 2: the card's URL as fetched by the Core AI lane d1d (audio/card_1.flac), copied "
                          "with its sha256 checked; read with soundfile.read(path, dtype='int16') as the card does",
                  "copied_from": e["from"], "seconds": lay["seconds"], "audio_info": lay, "prefix_tokens": p})
        return m, rec["note"]
    raise KeyError(rid)


def v2_shared_records(mf, vision):
    """The nine d1d records in our format (request / gold / note copied; media = our copy of the file)."""
    d1d = json.loads((D1D / "fixtures/records.json").read_text())
    by = {r["id"]: r for r in d1d["records"]}
    ref = {"records_json": str(D1D / "fixtures/records.json"),
           "records_json_sha256": mf["d1d"]["d1d_manifests"]["records"]["sha256"]}
    out = []
    for rid in SHARED_WITH_D1D_NEW:
        r = by[rid]
        assert S.sha256_file(D1D / "fixtures/records.json") == ref["records_json_sha256"], "d1d records.json changed"
        prov = {"shared_with": "d1d (Core AI lane, ~/code/coreai/_d1_omni)", **ref, "d1d_source": r["source"],
                "d1d_public": r["public"], "d1d_media": r["media"], "d1d_provenance": copy.deepcopy(r["provenance"])}
        media, publishable = None, r["public"]
        if rid.startswith("aud_"):
            e = mf["d1d"][rid]
            p, lay = media_prefix(vision, "audio", S.K / e["file"])
            media = {"kind": "audio", "ref": e["file"], "sha256": e["sha256"], "bytes": e["bytes"],
                     "transcript": r["provenance"]["script"], "voice": r["provenance"]["voice"],
                     "tts": {"repo": "hexgrad/Kokoro-82M", "license": "apache-2.0"},
                     "synth": copy.deepcopy(r["provenance"]["synth"]),
                     "license": "synthesised by the Core AI lane d1d with Kokoro-82M (Apache-2.0) from d1d's own "
                                "script (no names)",
                     "format": "WAV, 16 kHz, mono, int16", "seconds": e["seconds"], "frames": e["frames"],
                     "copied_from": e["from"], "audio_info": lay, "prefix_tokens": p}
        elif rid.startswith("img_"):
            e = mf["d1d"][rid]
            me = e["manifest_entry"]
            p, lay = media_prefix(vision, "image", S.K / e["file"])
            assert p == me["prefix_positions"], (rid, p, me["prefix_positions"])
            publishable = me["pool"] == "cc0"
            media = {"kind": "image", "ref": e["file"], "sha256": e["sha256"], "bytes": e["bytes"], "px": me["px"],
                     "license": me["license"], "pool": me["pool"], "made": me["made"], "source": me["source"],
                     "shows": me["shows"], "checked": me["checked"], "copied_from": e["from"], "layout": lay,
                     "prefix_tokens": p}
            prov["publishable_rule"] = ("round 2 launch: publishable = the d1d pool's licence is CC0; pool "
                                        f"{me['pool']!r} -> {publishable}")
        prov["gold_check_round2"] = GOLD_CHECK["d1d_" + rid.split("_")[0]]
        out.append({"id": rid, "source": r["source"], "publishable": publishable,
                    "request": copy.deepcopy(r["request"]), "media": media, "gold": copy.deepcopy(r["gold"]),
                    "note": r["note"] + " (shared with the Core AI lane d1d: same id, same request)",
                    "provenance": prov})
    return out


def extend():
    t0 = time.time()
    fx = S.K / "fixtures/requests.json"
    v1p = S.K / "fixtures/requests.v1.json"
    if v1p.exists():
        v1_text = v1p.read_text()
    else:
        v1_text = fx.read_text()
        v1p.write_text(v1_text)
    v1 = json.loads(v1_text)
    assert v1["version"] == 1 and v1["n_records"] == 387, (v1["version"], v1["n_records"])
    fs, fs1 = S.K / "results/fixture_summary.json", S.K / "results/fixture_summary.v1.json"
    if not fs1.exists():
        shutil.copyfile(fs, fs1)
    mf = json.loads((S.K / "fixtures/media_fetch.json").read_text())
    km = json.loads((S.K / "fixtures/audio/kokoro_manifest.json").read_text())
    V = S.load_module("d1_provider_vision", S.SNAP / "vision.py")
    P = S.provider_prompt()
    records = []
    changed = []
    for r in v1["records"]:
        r = copy.deepcopy(r)
        if r["media"] is not None:
            r["media"], r["note"] = v2_media_ours(r, mf, km, V)
            r["provenance"] = {**r["provenance"], "round2": "media file made / copied in round 2 "
                                                            "(fixtures/media_fetch.json, fixtures/audio/kokoro_manifest.json)",
                               "gold_check_round2": GOLD_CHECK[r["source"] if r["source"] in GOLD_CHECK else r["id"]]}
            changed.append(r["id"])
        records.append(r)
    shared_new = v2_shared_records(mf, V)
    records += shared_new
    ids = [r["id"] for r in records]
    assert len(ids) == len(set(ids)) == 396, (len(ids), len(set(ids)))
    for r in records:
        if r["media"] is not None:
            assert S.sha256_file(S.K / r["media"]["ref"]) == r["media"]["sha256"], r["id"]
    n_questions, rejected = validate(records, P)
    # records whose request is the same as d1d's (same id); semif differs only in the question's name
    d1d = {r["id"]: r for r in json.loads((D1D / "fixtures/records.json").read_text())["records"]}
    same, same_but_qid = [], []
    for r in records:
        o = d1d.get(r["id"])
        if o is None:
            continue
        if json.dumps(o["request"], sort_keys=True) == json.dumps(r["request"], sort_keys=True):
            same.append(r["id"])
        elif json.dumps([o["request"]["state"], list(o["request"]["questions"].values())], sort_keys=True) == \
                json.dumps([r["request"]["state"], list(r["request"]["questions"].values())], sort_keys=True):
            same_but_qid.append(r["id"])
    # the same request under another id: text records only (media records with the same questions differ by their
    # file), plus a media pair whose file is the same bytes (our card_topic = d1d's card_audio)
    same_request_other_id = []
    d1d_media_sha = {"card_audio": mf["d1d"]["card_topic"]["sha256"], "card_cats": mf["d1d"]["card_cats"]["sha256"]}
    for r in records:
        for o in d1d.values():
            if o["id"] == r["id"] or json.dumps(o["request"], sort_keys=True) != json.dumps(r["request"], sort_keys=True):
                continue
            if r["media"] is None and o["media"] is None:
                same_request_other_id.append([r["id"], o["id"], "text"])
            elif r["media"] is not None and d1d_media_sha.get(o["id"]) == r["media"]["sha256"]:
                same_request_other_id.append([r["id"], o["id"], "same media file"])
    counts, qtypes = {}, {}
    for r in records:
        key = r["source"] if r["source"] not in ("own",) else ("own_long" if r["id"] == "own_long_3400" else "own")
        counts[key] = counts.get(key, 0) + 1
        for q in r["request"]["questions"].values():
            k = f"{r['source']}/{q['type']}"
            qtypes[k] = qtypes.get(k, 0) + 1
    sources = copy.deepcopy(v1["sources"])
    sources["img"]["what"] = "CC0 photos from Wikimedia Commons (licence read from the Commons API), long side 384 px"
    sources["aud"]["what"] = "own English scripts synthesised with Kokoro-82M (Apache-2.0) in round 2"
    d1d_src = json.loads((D1D / "fixtures/records.json").read_text())["sources"]
    sources["d1d"] = {"what": "nine records of the Core AI lane d1d added under the same ids (round 2)",
                      "records_json": str(D1D / "fixtures/records.json"),
                      "records_json_sha256": mf["d1d"]["d1d_manifests"]["records"]["sha256"],
                      "audio_manifest_sha256": mf["d1d"]["d1d_manifests"]["audio"]["sha256"],
                      "images_manifest_sha256": mf["d1d"]["d1d_manifests"]["images"]["sha256"],
                      "own_audio": d1d_src["audio"], "own_image": d1d_src["images"], "long_3400": d1d_src["long_3400"]}
    doc = {
        "version": 2,
        "created_by": "d1_omni_work/scripts/fx_define.py --extend (round 2; v1 = requests.v1.json)",
        "record_format": v1["record_format"] + "; v2: media.ref = the file's path relative to d1_omni_work, "
                                               "media.sha256 = that file's sha256",
        "gold_keys": v1["gold_keys"], "publishable_rule": v1["publishable_rule"], "sources": sources,
        "counts": counts, "question_types": dict(sorted(qtypes.items())), "n_records": len(records),
        "n_questions": n_questions, "d1_rejected_questions": rejected,
        "own_long_3400_state_tokens": v1["own_long_3400_state_tokens"],
        "v2": {"from_v1_sha256": S.sha256_file(v1p), "media_filled": changed, "added_from_d1d": SHARED_WITH_D1D_NEW,
               "shared_with_d1d_same_request": same, "shared_with_d1d_same_request_but_question_name": same_but_qid,
               "same_request_other_id_in_d1d": same_request_other_id,
               "media_fetch_json_sha256": S.sha256_file(S.K / "fixtures/media_fetch.json"),
               "kokoro_manifest_sha256": S.sha256_file(S.K / "fixtures/audio/kokoro_manifest.json")},
        "records": records,
    }
    text = json.dumps(doc, indent=1, ensure_ascii=False) + "\n"
    fx.write_text(text)
    summ = summary_v2(doc, V, P)
    fs.write_text(json.dumps(summ, indent=1, ensure_ascii=False) + "\n")
    write_readme_v2(doc, summ)
    gi = S.K / ".gitignore"
    if "fixtures/media_measure_only/" not in gi.read_text():
        gi.write_text(gi.read_text().rstrip("\n") + "\n# round 2: measurement-only media (COCO image, LibriSpeech "
                                                     "clip), never published\nfixtures/media_measure_only/\n")
    print(json.dumps({"n_records": len(records), "n_questions": n_questions, "counts": counts,
                      "rejected": rejected, "media_filled": changed, "added": SHARED_WITH_D1D_NEW,
                      "shared_same_request": len(same), "shared_same_request_but_qid": len(same_but_qid),
                      "same_request_other_id": same_request_other_id, "requests_sha256": S.sha256_file(fx),
                      "v1_sha256": S.sha256_file(v1p), "seconds": round(time.time() - t0, 1)}, indent=1))


def summary_v2(doc, vision, prompt):
    """results/fixture_summary.json for v2: round 1's keys (scripts/encode_rows.py's definitions), plus
    shared_with_d1d and the media prefix lengths measured from the files."""
    import encode_rows as E

    tok = S.tokenizer()
    cfg = S.config()
    temps = cfg["temperatures"]
    enc = lambda s: tok(prompt.escape(s), add_special_tokens=False)["input_ids"]  # noqa: E731
    rows, skipped, errors = [], [], []
    for r in doc["records"]:
        mode = S.mode_of(r)
        state = S.state_of(r, mode)
        P = 0 if r["media"] is None else r["media"]["prefix_tokens"]
        for qid, qd in r["request"]["questions"].items():
            try:
                q = prompt.as_question(qd)
            except ValueError as e:
                skipped.append({"id": r["id"], "qid": qid, "error": str(e)})
                continue
            max_len = min(mode["max_len"], S.MAX_LENGTH - P)
            ids, markers = prompt.encode(tok, state, q, max_len, mode["noul_default"], mode["audio"])
            opts = prompt.render_options(q, mode["noul_default"], mode["audio"])
            K_ = len(opts)
            budget = max(96, min(K_ * 24 + 32, max_len // 2))
            per = max(2, (budget - 3 * K_) // K_)
            instr = enc(q.instructions)
            opt_tok = [len(enc(" " + t)) for t in opts]
            question_len = min(1 + len(instr), max(16, budget)) + sum(3 + min(n, per) for n in opt_tok) + 1
            room = max(0, max_len - question_len - 2)
            state_tok = len(enc(prompt.serialize(state)))
            assert len(ids) == 1 + 1 + min(state_tok, room) + question_len, (r["id"], qid)
            texts = [prompt.serialize(state), q.instructions] + opts
            rows.append({"id": r["id"], "qid": qid, "source": r["source"], "mode": mode["mode"], "type": q.type,
                         "K": K_, "len": len(ids), "P": P, "positions": P + len(ids), "per": per,
                         "state_tokens": state_tok, "state_truncated": state_tok > room,
                         "instructions_truncated": 1 + len(instr) > max(16, budget),
                         "options_truncated": sum(n > per for n in opt_tok),
                         "escaped": any(prompt.escape(t) != t for t in texts),
                         "temperature_key": prompt.temperature_key(q) if mode["calibrate"] else None,
                         "temperature": (temps.get(prompt.temperature_key(q), temps.get(q.type, 1.0))
                                         if mode["calibrate"] else None),
                         "markers": markers, "ids": ids})
    from collections import Counter, defaultdict

    def src_key(x):
        return "own_long" if x["id"] == "own_long_3400" else x["source"]

    by_src, recs_by_src = defaultdict(list), Counter(
        "own_long" if r["id"] == "own_long_3400" else r["source"] for r in doc["records"])
    for x in rows:
        by_src[src_key(x)].append(x)
    shared = set(doc["v2"]["shared_with_d1d_same_request"]) | set(doc["v2"]["shared_with_d1d_same_request_but_question_name"])
    per_source = {}
    for src, xs in by_src.items():
        lens = [x["len"] for x in xs]
        per_source[src] = {"records": recs_by_src[src], "questions": len(xs),
                           "shared_with_d1d_records": len({x["id"] for x in xs if x["id"] in shared}),
                           "types": dict(Counter(x["type"] for x in xs)), "max_K": max(x["K"] for x in xs),
                           "len_p50": E.pct(lens, 50), "len_p99": E.pct(lens, 99), "len_max": max(lens),
                           "len_min": min(lens), "buckets_text_len": E.within(lens),
                           "buckets_positions": E.within([x["positions"] for x in xs])}
    text = [x["len"] for x in rows if x["mode"] == "text"]
    image = [x["len"] for x in rows if x["mode"] == "image"]
    audio = [x["len"] for x in rows if x["mode"] == "audio"]
    media_rows = [x for x in rows if x["mode"] != "text"]
    lengths = {
        "text": {"rows": len(text), "p50": E.pct(text, 50), "p99": E.pct(text, 99), "max": max(text),
                 "buckets": E.within(text)},
        "image_text_only": {"rows": len(image), "max": max(image), "lens": image},
        "image_plus_144_384px": {"max": max(image) + 144, "buckets": E.within([n + 144 for n in image])},
        "image_plus_256_512px": {"max": max(image) + 256, "buckets": E.within([n + 256 for n in image])},
        "audio_text_only": {"rows": len(audio), "max": max(audio), "lens": audio},
        "audio_plus_125_10s": {"max": max(audio) + 125, "buckets": E.within([n + 125 for n in audio])},
        "audio_plus_375_30s": {"max": max(audio) + 375, "buckets": E.within([n + 375 for n in audio])},
        "media_rows_with_measured_prefix": {
            "rows": len(media_rows), "positions_max": max(x["positions"] for x in media_rows),
            "buckets": E.within([x["positions"] for x in media_rows]),
            "by_smallest_bucket": dict(sorted(Counter(E.smallest_bucket(x["positions"]) for x in media_rows).items())),
            "per_row": [{"row": f"{x['id']}/{x['qid']}", "P": x["P"], "text": x["len"], "positions": x["positions"]}
                        for x in media_rows]},
    }
    pub = {r["id"] for r in doc["records"] if r["publishable"]}
    pub_text = [x["len"] for x in rows if x["mode"] == "text" and x["id"] in pub]
    all_ids = [[x["id"], x["qid"], x["ids"]] for x in rows]
    own_long = [x for x in rows if x["id"] == "own_long_3400"]
    return {
        "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "version": 2,
        "requests_sha256": S.sha256_file(S.K / "fixtures/requests.json"),
        "n_records": len(doc["records"]), "n_questions_in_fixture": sum(len(r["request"]["questions"])
                                                                        for r in doc["records"]),
        "n_rows_encoded": len(rows), "skipped_by_as_question": skipped, "encode_errors": errors,
        "by_source": per_source, "types_all": dict(Counter(x["type"] for x in rows)),
        "modes": dict(Counter(x["mode"] for x in rows)), "max_K": max(x["K"] for x in rows),
        "max_markers_per_row": max(len(x["markers"]) for x in rows),
        "K_histogram": dict(sorted(Counter(x["K"] for x in rows).items())), "lengths": lengths,
        "publishable_text_rows": {"rows": len(pub_text), "p50": E.pct(pub_text, 50), "p99": E.pct(pub_text, 99),
                                  "max": max(pub_text), "buckets": E.within(pub_text)},
        "longest_rows": [{"id": x["id"], "qid": x["qid"], "len": x["len"]} for x in sorted(rows, key=lambda x: -x["len"])[:8]],
        "state_truncated_rows": [f"{x['id']}/{x['qid']}" for x in rows if x["state_truncated"]],
        "instructions_truncated_rows": [f"{x['id']}/{x['qid']}" for x in rows if x["instructions_truncated"]],
        "options_truncated_rows": [{"row": f"{x['id']}/{x['qid']}", "n": x["options_truncated"], "per": x["per"]}
                                   for x in rows if x["options_truncated"]],
        "escape_effective_rows": [f"{x['id']}/{x['qid']}" for x in rows if x["escaped"]],
        "own_long_3400": {"state_tokens": own_long[0]["state_tokens"], "row_lens": [x["len"] for x in own_long]},
        "long_3400": {"state_tokens": next(x["state_tokens"] for x in rows if x["id"] == "long_3400"),
                      "row_lens": [x["len"] for x in rows if x["id"] == "long_3400"]},
        "temperatures_used": dict(Counter(f"{x['temperature_key']}={x['temperature']}" for x in rows
                                          if x["temperature_key"])),
        "bucket_proposal": E.bucket_proposal(text, image, audio),
        "media_prefix_tokens": {r["id"]: r["media"]["prefix_tokens"] for r in doc["records"] if r["media"]},
        "shared_with_d1d": sorted(shared),
        "shared_with_d1d_detail": {"same_id_same_request": len(doc["v2"]["shared_with_d1d_same_request"]),
                                   "same_id_question_name_differs": doc["v2"]["shared_with_d1d_same_request_but_question_name"][:3]
                                   + ["..."] if doc["v2"]["shared_with_d1d_same_request_but_question_name"] else [],
                                   "same_id_question_name_differs_n": len(doc["v2"]["shared_with_d1d_same_request_but_question_name"]),
                                   "same_request_other_id": doc["v2"]["same_request_other_id_in_d1d"]},
        "percentile_method": "numpy.percentile default (linear interpolation)",
        "ids_sha256": S.sha256_json(all_ids),
        "ids_sha256_note": "v2 rows in fixture order (v1's 424 rows keep their ids: their sha256 = "
                           "results/encoded_rows.json for the v1 subset, asserted)",
    } | {"ids_sha256_v1_subset_equals_encoded_rows": S.sha256_json(
        [a for a in all_ids if a[0] in {r["id"] for r in json.loads((S.K / "fixtures/requests.v1.json").read_text())["records"]}]) ==
        json.loads((S.K / "results/encoded_rows.json").read_text())["ids_sha256"]}


def write_readme_v2(doc, summ):
    v1_readme = (S.K / "fixtures/README.md").read_text()
    if not (S.K / "fixtures/README.v1.md").exists():
        (S.K / "fixtures/README.v1.md").write_text(v1_readme)
    v1_readme = (S.K / "fixtures/README.v1.md").read_text()
    recs = {r["id"]: r for r in doc["records"]}

    def med(rid):
        m = recs[rid]["media"]
        extra = (f"{m['px'][0]}x{m['px'][1]} px" if m["kind"] == "image" else f"{m['seconds']:.2f} s")
        return f"| `{rid}` | {m['kind']} | `{m['ref']}` | {extra} | {m['prefix_tokens']} | `{m['sha256']}` | " \
               f"{'yes' if recs[rid]['publishable'] else 'no'} |"

    media_ids = [r["id"] for r in doc["records"] if r["media"] is not None]
    bs = summ["by_source"]
    order = ["card", "tv4", "tv4x", "tv4s", "semif", "own", "red_arm", "own_long", "img", "aud", "own_audio",
             "own_long_extended", "own_image"]
    lines = [
        "# d1-omni-600M fixtures (v2, round 2)",
        "",
        f"`requests.json` holds {doc['n_records']} records with {doc['n_questions']} questions (v2). Built by "
        "`scripts/fx_define.py --extend` from v1 (`requests.v1.json`, round 1, 387 records / 425 questions, sha256 "
        f"`{doc['v2']['from_v1_sha256']}`; its README is `README.v1.md`). sha256 of `requests.json` v2: "
        f"`{summ['requests_sha256']}`.",
        "",
        "## What v2 changed",
        "",
        "- The media files exist now. `media.ref` is the file's path relative to `d1_omni_work/` and `media.sha256` "
        "is that file's sha256 (both were placeholders in v1). `media.prefix_tokens` is the provider's prefix length "
        "for that file.",
        "- Our three Commons photos keep their v1 questions and gold. The v1 plan (centre-crop to 384x384) is replaced "
        "by the round-2 form: the long side resized to 384 px with Pillow LANCZOS, aspect kept (`media.made`).",
        "- Our three audio scripts were synthesised with Kokoro-82M using the Core AI lane d1d's recipe "
        "(`audio/kokoro_manifest.json`).",
        "- Nine records of the Core AI lane d1d (`~/code/coreai/_d1_omni/fixtures/records.json`) were added under the "
        "same ids, with request, gold and note copied: `aud_01`..`aud_03`, `long_3400`, `card_batch_00`, "
        "`card_batch_01`, `img_01`..`img_03`. Their files are byte copies (sha256 checked against d1d's manifests).",
        "",
        "## Slices",
        "",
        "| source | records | questions | shared with d1d (records) | publishable |",
        "|---|---|---|---|---|",
    ]
    for src in order:
        if src not in bs:
            continue
        ids_src = [r["id"] for r in doc["records"] if (("own_long" if r["id"] == "own_long_3400" else r["source"]) == src)]
        pubs = sum(recs[i]["publishable"] for i in ids_src)
        lines.append(f"| {src} | {bs[src]['records']} | {bs[src]['questions']} | {bs[src]['shared_with_d1d_records']} "
                     f"| {pubs} of {len(ids_src)} |")
    lines += [f"| total | {doc['n_records']} | {summ['n_rows_encoded']} encoded (+{len(summ['skipped_by_as_question'])} "
              f"rejected) | {len(summ['shared_with_d1d'])} | {sum(r['publishable'] for r in doc['records'])} of "
              f"{doc['n_records']} |", "",
              f"Shared with d1d = same id and the same request ({summ['shared_with_d1d_detail']['same_id_same_request']} "
              "records) or the same request with another question name "
              f"({summ['shared_with_d1d_detail']['same_id_question_name_differs_n']} semif records: d1d names the "
              "question `decision`, we name it `answer`). `card_topic` has the same request as d1d's `card_audio` "
              "under another id.", "",
              "## Media files", "",
              "| record | kind | file | size | prefix tokens | sha256 | publishable |",
              "|---|---|---|---|---|---|---|"]
    lines += [med(i) for i in media_ids]
    lines += ["", "Licences: our photos are CC0 1.0 (Wikimedia Commons, read from the Commons API; `media.made` has "
              "the original URL and sha1). `img_02` and `img_03` are CC0 1.0 Commons photos from d1d's sieved pool. "
              "`img_01` is d1d's own FLUX.2 klein 4B output (Apache-2.0 model, no third-party image): it is not "
              "CC0, so under this round's rule (publishable only from a CC0 pool) it is `publishable: false`, while "
              "d1d marks it public. The audio clips are Kokoro-82M (Apache-2.0) synthesis of scripts written for "
              "this port or by d1d. `card_cats` (COCO) and `card_topic` (LibriSpeech, CC BY 4.0) live in "
              "`media_measure_only/` (gitignored) and are never published.", ""]
    # keep the v1 sections that still hold (record format, Kev copies, the rejected question)
    keep = v1_readme.split("## Record format", 1)[1].split("## Slices", 1)[0]
    kev = v1_readme.split("### Copied from the Kev fixtures", 1)[1].split("### New in this set", 1)[0]
    rej = v1_readme.split("## One question the provider's schema rejects", 1)[1].split("## Measurement only", 1)[0]
    lines += ["## Record format", keep.rstrip().replace(
        "The file is fetched or synthesised in round 2, so `sha256` is `null` here; `prep` says exactly how round 2 "
        "makes the file.", "In v2 `ref` is the local file and `sha256` its hash; `prep` says how the file was made."),
        "", "## Copied from the Kev fixtures", kev.rstrip(), "",
        "## One question the provider's schema rejects", rej.rstrip(), "",
        "## Measurement only: never put these in a published file", "",
        "- `card_cats`: a COCO image (Flickr licences).", "- `card_topic`: a LibriSpeech clip (CC BY 4.0, attribution "
        "required).", "- All 140 `tv4x_*` records (licences listed in `README.v1.md`).",
        "- `img_01`: not from a CC0 pool (see above).", "",
        "Everything else is `publishable: true`.", ""]
    (S.K / "fixtures/README.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--extend", action="store_true")
    a = ap.parse_args()
    if a.tune:
        return tune()
    if a.extend:
        return extend()
    P = S.provider_prompt()
    V = S.load_module("d1_provider_vision", S.SNAP / "vision.py")
    tok = S.tokenizer()

    kev_doc, kev = kev_records()
    records = card_records() + kev + [own_long_record()] + image_records(V) + audio_records()
    ids = [r["id"] for r in records]
    assert len(ids) == len(set(ids)), "duplicate ids"
    # the Kev copies: request / gold / provenance / note / source byte-equal to the source records
    src_by_id = {r["id"]: r for r in kev_doc["records"]}
    for r in kev:
        s = src_by_id[r["id"]]
        for k in ("source", "request", "gold", "note", "provenance"):
            assert json.dumps(r[k], sort_keys=True) == json.dumps(s[k], sort_keys=True), (r["id"], k)

    st_tokens = len(tok(P.escape(records[ids.index("own_long_3400")]["request"]["state"]),
                        add_special_tokens=False)["input_ids"])
    assert 3300 <= st_tokens <= 3500, f"own_long_3400 state is {st_tokens} tokens; retune N_FILLER"
    n_questions, rejected = validate(records, P)

    counts = {}
    for r in records:
        key = r["source"] if r["source"] not in ("own",) else ("own_long" if r["id"] == "own_long_3400" else "own")
        counts[key] = counts.get(key, 0) + 1
    qtypes = {}
    for r in records:
        for q in r["request"]["questions"].values():
            k = f"{r['source']}/{q['type']}"
            qtypes[k] = qtypes.get(k, 0) + 1
    doc = {
        "version": 1,
        "created_by": "d1_omni_work/scripts/fx_define.py (round 1)",
        "record_format": "{id, source, publishable, request: {state, questions}, media: null | {kind, ref, license, "
                         "sha256, ...}, gold: {qid: key}, note, provenance}; request = the (state, questions) of the "
                         "provider's system_one(state, questions, images, audio); media carries the image / audio",
        "gold_keys": "choice = criteria key, noul = 'true' / 'false', score = level index as a string, null = no gold",
        "publishable_rule": "true = may appear in a published file (card, repo, gist); false = measurement only",
        "sources": {
            "card": {"what": "the three examples of the d1-omni-600M model card", "repo": S.REPO, "revision": S.REV},
            "tv4": kev_doc["sources"]["tv4"], "tv4x": kev_doc["sources"]["tv4x"], "tv4s": kev_doc["sources"]["tv4s"],
            "semif": kev_doc["sources"]["semif"], "own": kev_doc["sources"]["own"],
            "red_arm": kev_doc["sources"]["red_arm"],
            "own_long": {"what": "own_long_3400, written for this port (scripts/own_long.py); invented names only"},
            "img": {"what": "CC0 photos from Wikimedia Commons (licence read from the Commons API)"},
            "aud": {"what": "own English scripts to be synthesised with Kokoro-82M (Apache-2.0)", "tts": KOKORO},
            "kev_requests_json_sha256": KEV_REQUESTS_SHA256,
        },
        "counts": counts,
        "question_types": dict(sorted(qtypes.items())),
        "n_records": len(records),
        "n_questions": n_questions,
        "d1_rejected_questions": rejected,
        "own_long_3400_state_tokens": st_tokens,
        "records": records,
    }
    text = json.dumps(doc, indent=1, ensure_ascii=False) + "\n"
    dst = S.K / "fixtures/requests.json"
    if dst.exists() and dst.read_text() != text:
        sys.exit(f"{dst} exists with other content; refusing to overwrite (move it aside first)")
    dst.write_text(text)
    lic = S.K / "fixtures/LICENSE-SemIf-MIT.txt"
    if not lic.exists():
        shutil.copyfile(KEV / "LICENSE-SemIf-MIT.txt", lic)
    assert S.sha256_file(lic) == SEMIF_LICENSE_SHA256
    print(json.dumps({"n_records": len(records), "n_questions": n_questions, "counts": counts,
                      "question_types": doc["question_types"], "rejected": rejected,
                      "own_long_3400_state_tokens": st_tokens, "requests_sha256": S.sha256_file(dst),
                      "written": time.strftime("%H:%M:%S")}, indent=1))


if __name__ == "__main__":
    main()
