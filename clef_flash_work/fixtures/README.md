# Fixtures

The checks of these LiteRT files compare them with the provider's own code (`Cloudflare/clef-flash` at revision `f0b7d3cd65296abfe459e6c228b618f5d8b26d2f`, `joint_schema_model.py` `systemone()`, float32 on the CPU) on 243 runs: a run is a request read as text, or a request with a picture read at the 256 px or the 448 px tile. This folder holds the requests, the pictures (drawn ones and six CC0 photographs) and the provider's probabilities for every run.

| File | Content |
|---|---|
| `requests_public.json` | 219 records in their order: 186 fixture records, 30 held-out records and 3 red arms. 214 carry their request; 5 carry only the SHA-256 of their request (below). |
| `images/` | The pictures of the picture records as the checks read them (PNG): 18 drawn with Pillow primitives (CC0-1.0; no photo, font file or downloaded asset involved) and 6 photographs from Wikimedia Commons (CC0-1.0). |
| `photos/`, `photos.json` | The six photographs' 960 px Commons thumbnails as they were downloaded on 2026-10-03 (JPEG), and per photograph its file page, author, license, the thumbnail's URL, bytes and SHA-256, and the PNG's bytes and SHA-256 and the SHA-256 of its RGB pixels (the PNG is the thumbnail decoded by Pillow). |
| `fetch_photos.py` | Checks the photographs: the SHA-256 of both files, that Pillow decodes the thumbnail to the PNG's pixels, and whether Commons still serves the same thumbnail (`--local` skips the download). It writes nothing. |
| `reference_probs.json` | The provider's float32 probabilities for every run the checks compare with: per question the option ids in the head's order, the probabilities, the most likely option and `near_tie` (the two most likely options 0.02 or less apart). No text and no token ids. |
| `LICENSE-SemIf-MIT.txt` | The MIT license of SemIf. |

## Record format

Each record is `{id, set, source, request, request_sha256, image_files?, gold, note, license}`. `request` is a request body of the SystemOne-compatible request shape: `model`, `state` (a string or any JSON value), `questions` `{id: {type, instructions, criteria}}`. A picture record names its picture in `image_files`; a check adds the picture's path as `images: [path]` and reads it at the 256 px or the 448 px tile. Gold is the option a record was written to have (`"true"` / `"false"` for noul, the criteria key for choice, the level index for score) or null. `request_sha256` is the SHA-256 of `json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(",", ":"))` as UTF-8.

## Sources

Every record but the SemIf ones was written for these fixtures for an earlier port of the same model (`github.com/john-rocky/coreai-model-zoo`, `conversion/clef_flash/make_fixtures.py`, which also draws the pictures). People, organizations, products and places in them are invented.

- `own_t*` (16) and `own_j*` (12), held-out `ho_t*` (10), `ho_l*` (2) and `ho_j*` (8): text and JSON states: support tickets, an incident report, an email thread, meeting notes, a review, a chat log, scheduling, a recipe, requests, notices, an invoice, an order, sensor readings, a calendar, a profile, an error payload, an inventory, a config diff, a tracking history, a survey, room availability, a basket, and five long states of about 2,000 tokens.
- `img_*` (10) and held-out `ho_i*` (8): questions about the drawn pictures in `images/`.
- `photo_*` (4) and held-out `ho_p*` (2): questions about the six photographs (`photos.json`).
- `semif_<id>` (144): SemIf authored144 (`benchmarks/data/authored144.jsonl`, SHA-256 `8162d1c73f925af64453f1ec05ef36d583b3815bf698e60f0d454bd11537e079`, github.com/TheoLeeCJ/SemIf at commit `ca3ba65f142967030ecb453346e94d6f476a69df`), one choice question per record: `state` to the state, `question` to the instructions, `options` to the criteria `{id: description}`, gold = `options[label].id`. MIT, Copyright (c) 2026 TheoLeeCJ (`LICENSE-SemIf-MIT.txt`).
- `red_*` (3): a record with one word of one question's instructions changed (`base_id` names the record). Each must move some option of that question by more than 0.02 against its base record's probabilities: a check that ignored the changed word would not move.

## Records given by SHA-256 only

`own_t02`, `own_t15`, `own_t16` and `own_j01`, and the red arm built on `own_j01`: on 2026-10-04 a search found an invented name of each of the four records in use, so their requests are not published (as in the fixtures' first publication). Their ids, gold, notes and the SHA-256 of their requests stay; the checks ran on them, and `reference_probs.json` holds their probabilities.

## Photographs

The six photographs are Wikimedia Commons files whose license is CC0 (`LicenseShortName` CC0 when they were downloaded on 2026-10-03 and on 2026-10-21); `photos.json` names each one's page and author. They are included because the checks' probabilities belong to these exact pixels: on 2026-10-21 Commons served the same thumbnail for five of them, and for `photo_cat` (the record `photo_02`) other bytes (198,968 B, SHA-256 `0c1262d3593724d6c2bf0265938f5a8b234b165d056bd057427d807b6539dc52`) whose pixels differ from the ones the checks read.

## The held-out records

The 30 held-out records were written after the fixture records, to confirm a choice made on them; they were not used to choose anything in this conversion.
