# Fixtures

The checks of these LiteRT files compare them with the provider's own code (`LiquidAI/d1-3B` at revision `da1fe36a861f24690f27f622dca1d8688503d113`, float32 on the CPU) on 382 test requests with 417 questions. This folder holds the requests whose text may be redistributed, references to the others with a script that restores them, the four control requests, the provider's probabilities for every question it answers, and tokenizer probes.

| File | Content |
|---|---|
| `requests_public.json` | All 382 records in their order. 161 carry their request: the 144 SemIf records, the source card's two examples and the 15 records written for these conversions. 221 are references only (see below). |
| `rebuild_requests.py` | Rebuilds the full `requests.json` (382 requests) from `requests_public.json` and the kev repository at tag `kev-1.0`, and with `--red-arms` the four control requests. It checks the SHA-256 of each result. Python 3.8+ standard library only. |
| `red_arms_public.json` | The four control requests: the two SemIf ones with their requests, the two made from transfer-v4 records as a rule over their base record. |
| `reference_probs.json` | The provider's float32 probabilities for every question it answers: the 415 text questions and the 3 requests with pictures, 418 in all, plus the one question it refuses. No text and no token ids. |
| `token_probes.json` | The ids of the option letters, digits, yes / no forms and special tokens under the d1-3B tokenizer, the provider's fallback codes, and the provider's option codes and read-out groups checked on every question it parses. |
| `LICENSE-SemIf-MIT.txt` | The MIT license of SemIf. |

## Record format

Each record is `{id, source, request: {state, questions[, images]}, gold: {qid: key or null}, note, provenance}`. `request` is a request in the form of the provider's `system_one()`: `state` (a string, any JSON value, or null when the pictures are the whole state), `questions` `{qid: {type, instructions, criteria}}`, and `images` on `card_cats_001` only (the photo's URL and SHA-256, not the file). Gold keys are the criteria key for choice, `"true"` / `"false"` for noul and the level index as a string for score; null means that the record has no intended answer.

## Slices

- `tv4_000`..`tv4_059` (60): lines 0 to 59 of `evals/v4/transfer-v4/development.jsonl` in github.com/jaredpalmer/kev at tag `kev-1.0` (commit `6b719c3c3f367295f6ef336f4f751cf5ff970abc`, file SHA-256 `ff374c49c6c9f15f8a56fb274b4a4857d20497eb8dd1ac07ce01560e682a5f2e`). All are MMLU 4-way choice questions, one per record.
- `tv4x_<source>_<k>` (140, k = 00..19): the 20 earliest records, in file order, of each of the seven other sources of the same file (emotion, tweet_offensive, qnli, paws, sciq, legacy_holdout, composition_holdout). 60 choice and 80 noul questions.
- `tv4s_<k>` (20): the 20 earliest records of the same file whose question type is score (legacy_holdout, lines 628 to 647).
- `semif_<id>` (144): SemIf authored144 (`authored144.jsonl`, SHA-256 `8162d1c73f925af64453f1ec05ef36d583b3815bf698e60f0d454bd11537e079`) from github.com/TheoLeeCJ/SemIf at commit `ca3ba65f`, one 3-way choice question per record: `state` to the state, `question` to the instructions, `options` to the criteria `{id: description}`, gold = `options[label].id`. MIT, Copyright (c) 2026 TheoLeeCJ (`LICENSE-SemIf-MIT.txt`).
- `own_*` (15 records, 48 questions): 12 records written for the Kev-0.8B LiteRT conversion (a support ticket, an incident report, an email thread, meeting notes, a product review, a JSON order, a JSON invoice, JSON sensor readings, a five-question request and three long states), in which every person, organization, product and place is invented; two long records written for another conversion of d1-3B, `own_long_15k_001` (a cold-room log as a JSON object) and `own_long_34k_001` (an order ledger as a JSON array), in which people are roles; and `own_mid_08k_001` (an overnight production log), which names no person, organization, product or place. Their gold is the answer each record was written to have.
- `card_text_001`, `card_cats_001` (2): the two examples of the source model card (`README.md` of `LiquidAI/d1-3B` at the revision above): one state with three questions, and the photo with one question. The card states no answers. The gold of `card_text_001` is this conversion's reading for `refund` (true) and `team` (billing) and null for `urgency`; the gold of `card_cats_001` is null.
- `red_arm_000` (1): `tv4_000` with one word of the instructions changed ("correctly" to "incorrectly"). On d1-3B this moves the float32 answer by 0.0069 only, so it is a record here, not a control.

One question has no instructions (`own_email_03` / `next_step`). The provider's parser refuses it (`KeyError: 'instructions'`), so the checks run `own_email_03` over its two other questions.

## Why some requests are references only

The 221 records from the transfer-v4 development file (tv4, tv4x, tv4s, and `red_arm_000`, derived from `tv4_000`) are listed by reference: id, source, gold, note, and the provenance fields `file`, `repo`, `tag`, `line` (0-based), `line_sha256` and `meta_id` (the record's `_meta.id`); `red_arm_000` lists its base record and its one-word edit. Their text comes from public datasets whose licenses differ (the tweet_offensive records quote social-media posts verbatim and their Hub card gives the license as unknown; SciQ is CC BY-NC 3.0), so this folder copies none of it. `rebuild_requests.py` restores them:

```bash
python rebuild_requests.py --out requests.json --red-arms red_arms.json                    # downloads one file (803,309 bytes) at the tag's commit
python rebuild_requests.py --out requests.json --red-arms red_arms.json --kev-repo ../kev  # or reads it from a clone at tag kev-1.0
```

The script checks the file's SHA-256 and each line's SHA-256, and turns each line into a record the way the fixture builder did. Every rebuilt request must have the SHA-256 recorded in `requests_public.json` (`work_request_sha256`), and every rebuilt gold must equal the listed one. It writes nothing unless the result has the SHA-256 recorded there (`rebuild_target_sha256`, `58f1d3ab3891dd5cbe5ee6a723a45f3c1483430882499cfaa5222d7d6ead89df`). With `--red-arms` it also writes the four control requests, in the form the conversion scripts read as `fixtures/red_arms.json`, and checks their SHA-256 (`c0760805a0a9fe7eb4a9c3fbbc1f82e24330f2a8adb487cde988c0579724e19f`). Both commands above gave these two SHA-256 values, the download with Python 3.9.6 and the local clone with Python 3.14.6.

The checks ran on the same 382 records. The conversion's own copy of `requests.json` (SHA-256 `05ecdeb17ce97a7e6fb0fcd1256c96179bb466e6bc820c97bfcfe456f0bc813e`, `work_fixtures_sha256`) holds the same requests, gold and notes in the same order. It differs from the rebuilt file only in the text of its header and of the provenance of three records (`own_long_15k_001`, `own_long_34k_001`, `own_mid_08k_001`), which named paths of the machine the conversion ran on.

The upstream licenses of the transfer-v4 sources, as declared on the Hugging Face Hub at the revisions that the author's suite pins:

| Source | Dataset | License (Hub card) |
|---|---|---|
| mmlu | `cais/mmlu` @ `c30699e8` | mit |
| emotion | `dair-ai/emotion` @ `cab853a1` | other |
| tweet_offensive | `cardiffnlp/tweet_eval` @ `b3a375ba` | unknown |
| qnli | `nyu-mll/glue` @ `bcdcba79` | other |
| paws | `google-research-datasets/paws` @ `161ece95` | other |
| sciq | `allenai/sciq` @ `2c94ad3e` | cc-by-nc-3.0 |
| legacy_holdout, composition_holdout | generated by the author's suite code (no upstream dataset) | the kev repository's Apache-2.0 |

## The rows

The checks also read `rows.json`, one row per question: the provider's rendered text, its token ids, the answer slot and the read-out token groups. It is not in this folder, because its text and ids spell the requests. `conversion/build_rows.py` writes it from `requests.json` with the provider's code and tokenizer (`conversion/README.md`).

## Control requests

Each control request changes the state or the instructions of a test request (`base_id`) and must move some option's probability by more than 0.02 against it: a check that ignored the changed input would not move. `red_not_qnli_00` adds "not" to the instructions of `tv4x_qnli_00`; `tv4_001_with_state_of_tv4_000` asks `tv4_001`'s question over the state of `tv4_000`; the two SemIf requests swap the evidence of two records that share a claim. `measured_before_adoption` holds each one's move on the provider's float32 CPU path when it was chosen.

## Reference probabilities

`reference_probs.json` holds the provider's code in float32 on the CPU, one row per question (text: Python 3.12.11, torch 2.14.1, transformers 5.14.1). `text` has the 415 text questions by `record/qid`: the option keys, the probabilities, `argmax` (the most likely key) and `near_tie` (the two most likely options 0.02 or less apart; one question, `tv4_013/answer`). `pictures` has the three requests with pictures by name: `card_cats_001` with the source card's photo, and the state and question of `tv4x_qnli_07` and `tv4s_00` with synthetic pictures that `conversion/d1_vision_host_check.py` draws (one split into 6 tiles plus a thumbnail, and two pictures). Each names its pictures by URL or file name, bytes and SHA-256. `refused` names the one question the provider's parser refuses.

## Tokenizer probes

`token_probes.json` holds the ids that the d1-3B tokenizer (`tokenizer/tokenizer.json`, as transformers 5.14.1 loads it for the provider's code) gives the option letters A..Z and " A".." Z", the digits, the yes / no forms, eight special tokens and the lower-case letters, which single-token codes the provider's fallback pool has, and the provider's option codes and read-out groups on every question of `requests.json` that the provider's parser accepts.
