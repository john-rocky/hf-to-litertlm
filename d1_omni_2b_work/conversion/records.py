"""The question set with audio: the d1-2B text and picture set (set/requests.json, set/extra_g96x72.json,
set/red_arms.json) + the audio requests, their control arms and the d1-Omni-2B card's examples (set/requests_audio.json,
set/red_arms_audio.json, set/card_omni.json), as the reference (ref_run.py) and the host checks read it.

    recs, media = load_records()                      # the five set files (not the card's), media tables merged
    state, questions, items = open_request(rec, media)  # what system_one takes: PIL pictures, {"type": "audio", "path"}
"""

from __future__ import annotations

import copy
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.normpath(os.path.join(HERE, ".."))

TEXTPIC_FILES = ["set/requests.json", "set/extra_g96x72.json", "set/red_arms.json"]
TEXTPIC_MEDIA = ["set/media.json", "set/media_extra.json"]
AUDIO_FILES = ["set/requests_audio.json", "set/red_arms_audio.json"]
AUDIO_MEDIA = ["set/media_audio.json"]
CARD_FILES = ["set/card_omni.json"]


def _path(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(WORK, p)


def default_files() -> tuple[list[str], list[str]]:
    """The full set: the text and picture files and the audio files, those that exist."""
    files = [p for p in TEXTPIC_FILES + AUDIO_FILES if os.path.exists(_path(p))]
    media = [p for p in TEXTPIC_MEDIA + AUDIO_MEDIA if os.path.exists(_path(p))]
    return files, media


def load_records(files: list[str] | None = None, media_files: list[str] | None = None) -> tuple[list[dict], dict]:
    """Records of `files` (requests form, or a red-arm list) and the merged media table (each file's inline `media`
    plus `media_files`). Ids must be unique across the files."""
    if files is None:
        files, default_media = default_files()
        media_files = default_media if media_files is None else media_files
    recs, media, seen = [], {}, set()
    for p in files:
        with open(_path(p)) as f:
            doc = json.load(f)
        if isinstance(doc, list):  # red arms: a full request each
            items = [{"id": a["id"], "source": "red_arm", "kind": a.get("kind"), "base_id": a.get("base_id"),
                      "frame": a.get("frame"), "watch": a.get("watch"), "request": a["request"]} for a in doc]
        else:
            items = doc["records"]
            media.update(doc.get("media", {}))
        for r in items:
            if r["id"] in seen:
                raise ValueError(f"record id {r['id']} appears twice")
            seen.add(r["id"])
            recs.append(r)
    for p in media_files or []:
        with open(_path(p)) as f:
            media.update(json.load(f))
    return recs, media


def _checked(m: dict, mid: str) -> str:
    p = _path(m["path"])
    if m.get("sha256"):
        with open(p, "rb") as f:
            if hashlib.sha256(f.read()).hexdigest() != m["sha256"]:
                raise ValueError(f"media {mid}: sha256 mismatch ({p})")
    return p


def media_item(media: dict, mid: str):
    """One media id as the provider takes it: a PIL picture, or {"type": "audio", "path": <absolute path>}."""
    m = media[mid]
    p = _checked(m, mid)
    if m.get("kind") == "audio":
        return {"type": "audio", "path": p}
    from PIL import Image

    return Image.open(p)


def open_request(rec: dict, media: dict) -> tuple:
    """(state, questions, media items) of a record: media ids opened, spoken instructions
    {"type": "audio", "media": id} turned into {"type": "audio", "path": ...}. The record is not changed."""
    req = rec["request"]
    questions = copy.deepcopy(req["questions"])
    for q in questions.values():
        ins = q.get("instructions") if isinstance(q, dict) else None
        if isinstance(ins, dict) and ins.get("type") == "audio" and "media" in ins:
            q["instructions"] = media_item(media, ins["media"])
    items = [media_item(media, mid) for mid in req.get("media") or []]
    return req.get("state"), questions, items


def audio_ids(rec: dict, media: dict) -> list[str]:
    """Every clip a record uses: its media's clips, then its spoken questions' clips (question order)."""
    req = rec["request"]
    ids = [m for m in req.get("media") or [] if media[m].get("kind") == "audio"]
    for q in req["questions"].values():
        ins = q.get("instructions") if isinstance(q, dict) else None
        if isinstance(ins, dict) and ins.get("type") == "audio":
            ids.append(ins["media"])
    return ids
