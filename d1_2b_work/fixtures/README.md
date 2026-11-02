# Fixtures

The checks of these LiteRT files compare their answers with the provider's own code (`LiquidAI/d1-2B` at revision `10e7671d44e4b37700020b1eed3f594c5f2af0b4`, float32 on the CPU) on 526 requests: 507 test records and 19 control arms. This folder holds every request whose text may be redistributed, references to the others with a script that restores them, the provider's answers to all of them, and the test pictures.

| File | Content |
|---|---|
| `requests_public.json` | All 526 records in the order the checks send them, each with the SHA-256 of its request (`request_sha256`). 303 carry their request; 223 are references only (see below). |
| `rebuild_requests.py` | Restores those 223 requests from the kev repository at tag `kev-1.0` and writes the full `requests.json` (526 requests). It checks every request's SHA-256 and the file's. Standard library only. |
| `reference_answers.json` | The provider's answer to every record (521 answered, 5 refused), with its token counts. |
| `media.json` | Every picture the requests name: file, size, SHA-256, source, license. |
| `media/` | 23 files, 4.5 MB: 5 CC0 photos and 18 synthetic pictures. |
| `LICENSE-SemIf-MIT.txt` | The MIT license of SemIf. |

## Record format

A test record is `{id, source, kind, request, request_sha256, gold, note, provenance}`. `request` is a request in the form of the provider's `system_one()`: `state` (a string, any JSON value, or null when the media are the whole state), `questions` (`{qid: {type, instructions, ...}}`) and `media` (media ids, in order; see Media). Records per `kind`: `image` 36, `long_refused` 1, `refused` 5, `text` 484.

Gold: noul `"true"` / `"false"`, choice the option name, score the level index as a string, multiple `{choices}`, read `{found, text, number|date}` or `{found: false}`, estimate `{number}`, box `{found, box}`, boxes `{count, boxes[, at_least, instances]}` (edges as fractions, in reading order). Null means that no answer is intended. Photo boxes carry presence or count only. The five records of the d1-2B card's examples carry `card_prints` / `card_check` (the values the card prints) instead of gold, note and provenance.

A control arm is `{id, base_id, kind, frame, request, request_sha256, changes, watch}`: the request of the test record `base_id` with one change (`changes`). The watched value (`watch`) must move against the base record's: a check that ignored the changed input would not move. `frame` names the kind of change.

`request_sha256` is the SHA-256 of `json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(",", ":"))` in UTF-8.

## Sources

| Source | Records | With request | By reference | What |
|---|---:|---:|---:|---|
| `tv4` | 60 | 0 | 60 | lines 0 to 59 of the transfer-v4 development file: MMLU 4-way choice questions |
| `tv4x` | 140 | 0 | 140 | the 20 earliest records of each of the file's seven other sources (emotion, tweet_offensive, qnli, paws, sciq, legacy_holdout, composition_holdout) |
| `tv4s` | 20 | 0 | 20 | the file's 20 earliest records whose question type is score (legacy_holdout, lines 628 to 647) |
| `semif` | 144 | 144 | 0 | SemIf authored144 (github.com/TheoLeeCJ/SemIf at commit `ca3ba65f`), one 3-way choice per record |
| `own` | 15 | 15 | 0 | the d1-3B fixture's 15 records written for earlier conversions (a support ticket, an incident report, an email thread, meeting notes, a product review, JSON records, long logs and an order ledger) |
| `red_arm` | 1 | 0 | 1 | `red_arm_000`: `tv4_000` with one word of the instructions changed ("correctly" to "incorrectly") |
| `card` | 7 | 7 | 0 | the source cards' examples: the d1-2B card's four as five records (its batch example is two requests; with the values the card prints), the d1-3B card's two |
| `own_text` | 19 | 19 | 0 | texts written for this conversion (invoices, a receipt, a payslip, a purchase order, notices, logs, JSON records, a support chat; some in French, German, Spanish or Japanese) with typed reads (text, number, date; some with no answer in the state), estimates and multiple choice |
| `own_image` | 19 | 19 | 0 | synthetic pictures of shapes at known positions (200 x 150 up to 1,600 x 1,200 and 1,200 x 1,600 pixels): boxes, counts, estimates and choices |
| `commons_cc0` | 5 | 5 | 0 | questions over the five CC0 photographs (boxes and counts; box gold: presence and count only) |
| `coco_card` | 1 | 1 | 0 | more questions over the card's COCO photo |
| `workload` | 70 | 70 | 0 | the speed workloads (`wl_*`): one question, three questions, a state of about 5,000 tokens, a 384-pixel picture, a box and boxes over the COCO photo, and 64 states sent in one call |
| `refusal` | 5 | 5 | 0 | requests the provider refuses on purpose (kind `refused`) |
| `own_long` | 1 | 1 | 0 | `long_refused_9k`, a state longer than the largest decision graph (6,144 tokens): the provider answers it, the host refuses it |
| control arms | 19 | 17 | 2 | a test record's request with one change, which must move the watched answer: a "not", another record's state, a number or a word of the state, the picture, the box's object |
| all | 526 | 303 | 223 | |

In the records written for these conversions every person, organization, product and place is invented (in two long records of the d1-3B fixture people are roles).

## Why some requests are references only

The 223 records made from the transfer-v4 development file of github.com/jaredpalmer/kev carry no request. The 220 records taken from its lines (`tv4`, `tv4x`, `tv4s`) list `repo`, `tag`, `commit`, `file`, `line` (0-based), `line_sha256` and `meta_id` (the line's `_meta.id`); `red_arm_000` lists its base record `tv4_000` and its one-word edit; the control arms `red_not_qnli_00` and `tv4_001_with_state_of_tv4_000` list their rule (`derive`: "not" added to the instructions of `tv4x_qnli_00`; the question of `tv4_001` over the state of `tv4_000`). Their text comes from public datasets whose licenses differ (the tweet_offensive records quote social-media posts verbatim and their Hub card gives the license as unknown; SciQ is CC BY-NC 3.0), so this folder copies none of it, and the answers to them in `reference_answers.json` hold option keys and numbers only. `rebuild_requests.py` restores them:

```bash
python rebuild_requests.py --out requests.json                     # downloads one file (803,309 bytes) at the tag's commit
python rebuild_requests.py --out requests.json --kev-repo ../kev   # or reads it from a clone at tag kev-1.0
```

The script checks the file's SHA-256 (`ff374c49c6c9f15f8a56fb274b4a4857d20497eb8dd1ac07ce01560e682a5f2e`) and each listed line's SHA-256, turns each line into a request the way the d1-3B fixture builder did (the line's state and questions, with `label` and `src` taken out of each question, the gold from `label`), and gives each of these requests an empty `media` list. Every request must have its record's `request_sha256`, every gold and note made from a line must equal the listed one, and the script writes nothing unless the whole file has the SHA-256 listed in `requests_public.json` (`rebuild_target_sha256`, `9d264343c2dfadc71bfff71d989b0e963dec656b2820dab3c8326990f0956dce`). 

The checks read the same 526 records from their own files: the same requests (each has the `request_sha256` listed here), gold and order. This folder rewrote a few note and provenance strings that named files of the machine the conversion ran on or steps of the work (`rewritten` in `requests_public.json`).

The licenses of the transfer-v4 sources, as declared on the Hugging Face Hub at the revisions that the author's suite pins:

| Source | Dataset | License (Hub card) |
|---|---|---|
| mmlu | `cais/mmlu` @ `c30699e8` | mit |
| emotion | `dair-ai/emotion` @ `cab853a1` | other |
| tweet_offensive | `cardiffnlp/tweet_eval` @ `b3a375ba` | unknown |
| qnli | `nyu-mll/glue` @ `bcdcba79` | other |
| paws | `google-research-datasets/paws` @ `161ece95` | other |
| sciq | `allenai/sciq` @ `2c94ad3e` | cc-by-nc-3.0 |
| legacy_holdout, composition_holdout | generated by the author's suite code (no upstream dataset) | the kev repository's Apache-2.0 |

## Reference answers

`reference_answers.json` holds the provider's answer to every record, by id in the order above: its own code in float32 on the CPU (8 threads, PyTorch 2.14.1; the environment is `conversion/requirements-ref.txt`), one request per call. `answers` is what it returns per question (the picked option or value, the probabilities, spans, boxes; a score's `legend`, which repeats the request's criteria, is left out), `usage` its input and output token counts, and `error` its refusal text. It refuses the 5 records of kind `refused` (`answers` null). `long_refused_9k` is answered: the provider answers it, while the LiteRT host refuses it because no graph holds it.

## Media

`request.media` names media ids; `media.json` maps each id to its file under this folder (`media/<id>.<ext>`), its SHA-256 and size, its source and license.

- The COCO photo `coco_val2017_000000039769` (COCO val2017 000000039769.jpg, 640 x 480, two cats: the source card's example picture) is not in this folder (`stored: false`). It is listed by URL and SHA-256, as `examples/run_example.py` fetches it. To run the requests that name it, fetch it into `media/` (from the repository root) and check it:

  ```bash
  curl -o fixtures/media/coco_val2017_000000039769.jpg http://images.cocodataset.org/val2017/000000039769.jpg
  shasum -a 256 fixtures/media/coco_val2017_000000039769.jpg   # dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e
  ```

- 5 CC0 1.0 photographs from Wikimedia Commons, resized (page and author per photo in `media.json`).
- 18 synthetic pictures drawn for this set: shapes at known positions (a gold box is the shape's own drawn pixels), each with its scene (`generator.params`) and the SHA-256 of its pixels.

## Licenses

- SemIf records: MIT, Copyright (c) 2026 TheoLeeCJ (`LICENSE-SemIf-MIT.txt`).
- The source cards' examples: short quotes of the example requests of the d1-2B and d1-3B model cards (LFM Open License v1.0).
- The records written for these conversions and the synthetic pictures: part of this repository, under its license.
- The Wikimedia Commons photographs: CC0 1.0.
- The COCO photo and the transfer-v4 records: not redistributed here (see above).
