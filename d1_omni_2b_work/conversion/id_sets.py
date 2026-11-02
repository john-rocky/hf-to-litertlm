"""Split the records into the two rows of every gate run: the audio set (records that use a clip: audio media or a
spoken question) and the rest (the d1-2B text and picture set, the card's text and picture examples). Control-arm
candidates go with the row of their records; score.py scores the chosen arms of each row.

  $REF conversion/id_sets.py  ->  conversion/ids_textpic.json, conversion/ids_audio.json (+ the audio row's sub-rows
      ids_audio_state_only / ids_audio_spoken_only / ids_audio_both, instruments)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(WORK, "host"))
from records import TEXTPIC_FILES, TEXTPIC_MEDIA, AUDIO_FILES, AUDIO_MEDIA, CARD_FILES, audio_ids, load_records  # noqa: E402


def spoken(rec):
    return any(isinstance(q, dict) and isinstance(q.get("instructions"), dict)
               for q in rec["request"]["questions"].values())


def main():
    files = TEXTPIC_FILES + AUDIO_FILES + CARD_FILES
    recs, media = load_records(files, TEXTPIC_MEDIA + AUDIO_MEDIA)
    audio = sorted(r["id"] for r in recs if audio_ids(r, media))
    rest = sorted(r["id"] for r in recs if not audio_ids(r, media))
    # the audio row's sub-rows (instruments): audio state only, spoken questions only, both
    state = {r["id"] for r in recs if any(media[m].get("kind") == "audio" for m in r["request"].get("media") or [])}
    spk = {r["id"] for r in recs if spoken(r)}
    subs = {"ids_audio_state_only.json": sorted(state - spk), "ids_audio_spoken_only.json": sorted(spk - state),
            "ids_audio_both.json": sorted(state & spk)}
    for name, ids in (("ids_audio.json", audio), ("ids_textpic.json", rest), *subs.items()):
        json.dump(ids, open(os.path.join(HERE, name), "w"), indent=0)
    print(json.dumps({"files": files, "audio": len(audio), "textpic": len(rest),
                      **{k[4:-5]: len(v) for k, v in subs.items()}}))


if __name__ == "__main__":
    main()
