# Fixtures

The checks of these LiteRT files compare them with the author's own code (the `kev` package at tag `kev-1.0`, PyTorch fp32 on the CPU) on 377 requests with 402 questions. This folder holds every number of that reference, the requests whose text may be redistributed, and a script that rebuilds the other requests from the author's repository.

| File | Content |
|---|---|
| `requests_public.json` | All 377 records in their original order. 156 carry their request (SemIf and the records written for this conversion); 221 are references only (see below) |
| `rebuild_requests.py` | Rebuilds the full `requests.json` (377 requests) from `requests_public.json` and the kev repository at tag `kev-1.0`; asserts the SHA-256 of the result |
| `oracle_probs.json` | The reference for all 402 questions: probabilities, logits before and after the temperature, argmax, top-2 gap, gold key, the answer object, row length and readout indices. No text and no token IDs |
| `tokenizer_probes.json` | 12 strings with the token IDs the author's tokenizer gives them, used to check the tokenizer file |
| `LICENSE-SemIf-MIT.txt` | The MIT license of SemIf |

## Record format

Each record is `{id, source, request: {state, questions}, gold: {qid: key}, note, provenance}`. `request` is a `/v1/systemone` request as the author's server takes it (`kev.api.SystemOneRequest`). Gold keys are the keys the answers are reported under: the criteria name for choice, `"true"` / `"false"` for noul, the level index as a string for score.

## Slices

- `tv4_000`..`tv4_059`: the first 60 records, in file order, of `evals/v4/transfer-v4/development.jsonl` in github.com/jaredpalmer/kev at tag `kev-1.0` (commit `6b719c3c3f367295f6ef336f4f751cf5ff970abc`, file SHA-256 `ff374c49c6c9f15f8a56fb274b4a4857d20497eb8dd1ac07ce01560e682a5f2e`). All are MMLU 4-way choice questions, one question per record.
- `tv4x_<source>_<k>` (k = 00..19): the first 20 records, in file order, of each of the seven other sources of the same file (emotion, tweet_offensive, qnli, paws, sciq, legacy_holdout, composition_holdout). 60 choice and 80 noul questions.
- `tv4s_<k>` (k = 00..19): the first 20 records of the same file whose question type is score (legacy_holdout, 0-based lines 628 to 647).
- `semif_<id>`: SemIf authored144 (`authored144.jsonl`, SHA-256 `8162d1c73f925af64453f1ec05ef36d583b3815bf698e60f0d454bd11537e079`) from GitHub `TheoLeeCJ/SemIf` at commit `ca3ba65f142967030ecb453346e94d6f476a69df`, one 3-way choice question per record: `state` to the state, `question` to the instructions, `options` to the criteria `{id: description}` in file order, gold = `options[label].id`. MIT, Copyright (c) 2026 TheoLeeCJ; the license text is in `LICENSE-SemIf-MIT.txt`.
- `own_*`: 12 records written for this conversion (37 questions): a support ticket, an incident report, an email thread, meeting notes, a product review, a JSON order, a JSON invoice, JSON sensor readings, a five-question request, and three long states (a service log, a contract excerpt, board minutes). Every person, organization, product and place in them is invented. Their gold is the answer each record was written to have; agreement with it is a figure on this subset only.
- `red_arm_000`: `tv4_000` with one word of the instructions changed ("correctly" to "incorrectly"). It is a control, not a fixture: compared with `tv4_000`'s reference, its probabilities must move by more than the 0.02 tolerance of the checks.

## Why some requests are references only

The records of the kev repository's transfer-v4 file (tv4, tv4x, tv4s, and the control derived from tv4_000) are listed by reference: id, source, gold, note, and the provenance fields `file`, `repo`, `tag`, `line` (0-based), `line_sha256` and `meta_id` (the record's `_meta.id`). Their text comes from public datasets whose licenses differ (table below; the tweet_offensive records quote social-media posts verbatim and their Hub card gives the license as unknown, SciQ is CC BY-NC 3.0), so this folder copies none of it. `rebuild_requests.py` restores them:

```bash
python rebuild_requests.py --out requests.json                    # downloads the one file (803,309 bytes) at the tag's commit
python rebuild_requests.py --out requests.json --kev-repo ../kev  # or reads it from a clone at tag kev-1.0
```

The script checks the file's SHA-256 and each line's SHA-256, turns each line into a record the way the original fixture builder did, and refuses to write unless the rebuilt file has the SHA-256 recorded in `requests_public.json` (`3bcc256671b49838ea781c2ff3388cf67745b69af6977652f7e90bc15723e2c7`). The checks ran on the same 377 records; the file they read differs from the rebuilt one only in the header's `created_by` text, and `oracle_probs.json` keeps that file's SHA-256 as `fixtures_sha256`. Standard library only.

Upstream licenses of the transfer-v4 sources, as declared on the Hugging Face Hub at the revisions the author's suite pins:

| source | dataset | license (Hub card) |
|---|---|---|
| mmlu | `cais/mmlu` @ `c30699e8` | mit |
| emotion | `dair-ai/emotion` @ `cab853a1` | other |
| tweet_offensive | `cardiffnlp/tweet_eval` @ `b3a375ba` | unknown |
| qnli | `nyu-mll/glue` @ `bcdcba79` | other |
| paws | `google-research-datasets/paws` @ `161ece95` | other |
| sciq | `allenai/sciq` @ `2c94ad3e` | cc-by-nc-3.0 |
| legacy_holdout, composition_holdout | generated by the author's suite code (no upstream dataset) | the kev repository's Apache-2.0 |

## The reference

`oracle_probs.json` comes from the author's code at tag `kev-1.0` on the CPU in fp32 (PyTorch 2.8.0, transformers 5.17.0, peft 0.21.0, the versions of the author's lock file), checkpoint `jaredpalmer/kev-0.8b` at tag `v1.0`: each request through `kev.api.to_record` and `kev.model.encode`, one causal row per question (the state, then that question's branch), the checkpoint's pointer head at its temperature (2.3510958125672174). `answer` is `kev.api.to_answers` for that question without its `legend` (the legend repeats the level texts). `near_tie` marks the 15 questions whose top two probabilities are within 0.02 of each other.

## Tokenizer probes

`tokenizer_probes.json` holds 12 strings and the token IDs that the tokenizer of the author's code gives them (transformers 5.17.0 `AutoTokenizer` for `Qwen/Qwen3.5-0.8B-Base` at `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`, class `Qwen2Tokenizer`), both as plain text (`ids`) and after the author's rewrite of `<|name|>` to `<¦name¦>` (`user_ids`). The `tokenizer.json` shipped here (the one in the `jaredpalmer/kev-0.8b` repository) gives the same IDs for all 12. The base repository's own `tokenizer.json`, read directly with the `tokenizers` library, differs on 4 of them (Devanagari text, `<think>`, `<tool_response>`, `<tts_pad>`), which is why it is not the file used here.
