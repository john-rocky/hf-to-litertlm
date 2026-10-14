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
- `add_tool_branch.py` — adds tool-call and tool-result rendering to the template a bundle carries (rules `lfm2`, `granite`, `falcon-h1`, `falcon3`; `lfm2` copies the tool rendering of LiteRT-LM's `models/lfm2/chat_template.jinja`). For the templates of the files re-packed in 2026-10 it checks the result against the template that was packed into them ([REPRODUCE.md](../../REPRODUCE.md), "Metadata re-packs (2026-10)").
- `granite-4.0-h_tools.jinja` — IBM's granite-4.0-h chat template (Apache-2.0) with `tool_response` content items rendered, as packed into the litert-community granite-4.0-h-350m and -1b files on 2026-10-14; `add_tool_branch.py granite` writes the same bytes from the bundles' previous template.
- `qwen3-0.6b-wo4_parts.jinja` — the Qwen3 chat template (Apache-2.0) of `mlboydaisuke/Qwen3-0.6B-wo4-float-repro-LiteRT` with a `format_content` macro at the places that read message content (its header says what changed), as packed into that file on 2026-10-14.

Four templates keep a pre-existing role limitation on purpose: S1-mini and Shieldstral-vision drop a system message, CodeGemma raises on one, Falcon3 reads `eos_token` (the runtime binds it). The swap changes the content form only.

The other template rule is about the history rather than the message, and litert-lm 0.17 and 0.18 both enforce it: every assistant turn must render the same way wherever it sits in the conversation, or the second message (or the third, after a reply that carried a thought) fails — see [../chat_template_prefix/](../chat_template_prefix/).

## A second turn: where the newline after `<|im_end|>` goes

LiteRT-LM prefills only the increment `render(history + message) − render(history)`, and the stop token the model emitted enters the cache as the pending token of the next prefill. A ChatML template that closes an assistant turn with `<|im_end|>\n` therefore never puts that `\n` in the cache: the second turn's context is the upstream render minus one token. End the assistant turn at `<|im_end|>` and write the newline at the head of whatever follows (the next message or the generation prompt); the full render is unchanged, the increment grows by one token and the cache equals the upstream ids.

`dual/a07bb77727e4.jinja` is such a template (MiniCPM-V-4.7-1B's text-only bundle; its two text+image bundles carry the same rule with image parts, `minicpmv47_work/chat_template_mv47_dual.jinja`). `dual/b3f69d650a63.jinja`, which the Qwen3.5 bundles carry, still ends the assistant turn with `<|im_end|>\n`: the one-token gap was measured on the runtime with MiniCPM-V-4.7's bundle before its fix (litert-lm 0.18.0, two turns in one process), and the Qwen3.5 files have not been re-checked.

```
python tools/chat_template_parts/two_turn_increment.py <template.jinja> <tokenizer.json> --stop "<|im_end|>"
```

renders two turns with the runtime's Jinja settings ([../chat_template_prefix/desk_render.py](../chat_template_prefix/desk_render.py)), prints the increment and compares the cache (turn-1 render + reply + stop token + increment) with the upstream two-turn ids: `dual/b3f69d650a63.jinja` → a gap of one token (`\n`), `dual/a07bb77727e4.jinja` → equal. The desk check assumes the reply tokenizes the same way it was generated; the runtime's own check is three turns on the engine.
