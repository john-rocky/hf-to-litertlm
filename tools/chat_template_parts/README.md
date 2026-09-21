# Chat templates and the 0.18 content-parts form

From LiteRT-LM 0.18 the runtime hands a bundle's Jinja template every message `content` as a list of typed parts (`[{"type": "text", "text": "..."}]`), never as a plain string ([`models/README.md`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/models/README.md); the runtime change is `6c6b4582`, already in the `litert-lm-nightly` wheels). A template written for string content keeps working on 0.17.x and breaks on 0.18 in one of three ways, all with exit code 0 or a template error:

| failure mode | what the template does | what you see on 0.18 |
|---|---|---|
| DROP | `content = ''` when `content` is not a string | the user's question vanishes; a fluent answer to something else |
| RAISE | `'...' + message.content` | `Failed to apply template: tried to use + operator on unsupported types string and sequence` |
| LEAK | `{{ message.content }}` | the list repr is printed into the prompt; single turns still answer, a system turn usually raises, later turns degrade |

A dual-form template accepts both (LiteRT-LM `a8178d7d`, `models/qwen3/chat_template.jinja`): a `format_content` macro that returns a string as is and concatenates the `text` parts of a list, applied at every `content` read. Nothing else in the template changes, so the prompt rendered from string content is byte-identical to the original's.

## Check a bundle

```
pip install litert-lm==0.17.1            # one venv
pip install litert-lm-nightly            # another venv
litert-lm run model.litertlm --prompt="What is the capital of France?" < /dev/null
```

Read the answer, not the exit code: an answer about "the provided text" or a different subject is the DROP case. For a static check, render the template with [`minijinja`](https://pypi.org/project/minijinja/) once with `{"role": "user", "content": "..."}` and once with the parts form and compare the two prompts.

## Fix a published bundle without re-exporting

`swap_template.py` replaces only `LlmMetadata.jinja_prompt_template` (unpack → edit the pbtext through the protobuf text formatter → pack) and then proves the result: section list unchanged, TFLite bytes identical, tokenizer identical after decoding, executor metadata equal, LlmMetadata equal after clearing only the Jinja field, and the packed template's sha256 equal to the candidate's.

```
python -m venv .venv && .venv/bin/pip install litert-lm==0.17.1
.venv/bin/python swap_template.py in.litertlm out.litertlm --jinja dual/<sha12>.jinja
```

The bundle uuid and creation timestamp change with every pack; the weights do not. Measured on `LFM2.5-1.2B-Instruct_int4.litertlm` (2026-09-21): 735,049,088 B TFLite section byte-identical, swap 1.9 s, and the 0.17.1 answers of the swapped bundle byte-identical to the original's for one-turn, system+user and two-turn prompts.

## What is here

- `dual/<sha12>.jinja` — the dual-form version of every string-only template found in the bundles we published (32 families; `sha12` = the first 12 hex digits of the sha256 of the ORIGINAL template text as stored in the bundle, so you can match a bundle by hashing its `jinja_prompt_template`).
- `families.tsv` — which published file carried which family and its failure mode before the swap.
- `swap_template.py` — the swap and its verification (writes a JSON report and an evidence directory next to it).

Four templates keep a pre-existing role limitation on purpose: S1-mini and Shieldstral-vision drop a system message, CodeGemma raises on one, Falcon3 reads `eos_token` (the runtime binds it). The swap changes the content form only.
