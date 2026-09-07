# constrained_space_3510 — constrained decoding on LiteRT-LM cannot emit a space (token table carries spellings, not bytes)

Re-run and verification for [google-ai-edge/LiteRT-LM#3510](https://github.com/google-ai-edge/LiteRT-LM/issues/3510): with `response_format`
(regex or JSON schema) a bundle can never produce a `0x20` byte, because `LlgConstraintProvider::Create` fills llguidance's token table from
`Tokenizer::GetTokens()`, and both tokenizers return the vocab **spelling** of each token — SentencePiece `▁dog` / `<0x20>`, byte-level BPE `Ġdog` /
`Ċ` — while only the text-output path converts those spellings back (`runtime/core/tasks.cc`, `▁` → space).

Measured 2026-09-07 with the litert-lm **0.16.1** pip wheel on macOS (Mac Studio), `litert-lm serve --api openai`, temperature 0, reporter's prompt
("two lowercase words separated by one space: the pet that meows and the pet that barks").

| bundle (litert-community) / backend / ctx | response_format | reply | completion tokens |
|---|---|---|---:|
| gemma-4-E2B-it (SentencePiece) / gpu / 8192 | none | `cat dog` | 3 |
| 〃 | regex `[a-z]+[ \n][a-z]+` | `cat\ndog` | 4 |
| 〃 | json_schema `{meows, barks}` | `{"meows"` then `\n\t` repeated | 8154 (= 8192 − 38) |
| 〃 | regex `[a-z]+ [a-z]+` | `catdogssnooze…` until the context limit | 8154 |
| 〃 | regex `[a-z]+▁[a-z]+` (U+2581 literal) | **`cat dog`** with a real space | 3 |
| 〃 / cpu / 1024 | regex `[a-z]+ [a-z]+` / `[a-z]+▁[a-z]+` | letters to the limit / **`cat dog`** | 986 / 3 |
| 〃 / gpu / 1024 | regex `cat dog` (only a space fits after `cat`) | fails at token 2: llguidance `token "▁dog" doesn't satisfy the grammar; forced bytes: got ' '; applying 'â'` | — |
| LFM2.5-1.2B-Instruct_int4 (HF tokenizer.json) / cpu / 1024 | regex `[a-z]+ [a-z]+` | letters to the limit | 985 (= 1024 − 39) |
| 〃 | regex `[a-z]+[ \n][a-z]+` | letters to the limit (newline is spelled `Ċ`) | 985 |
| 〃 | regex `[a-z]+Ġ[a-z]+` (U+0120 literal) | **`cat dog`** | 3 |

Offline audit of the gemma-4-E2B-it SP section (`sp_table_audit.py`): 262,144 pieces, **0** contain byte 0x20, 137,542 contain U+2581, the byte
piece for 0x20 is the six-character string `<0x20>` (id 270), and the only pieces made of ASCII whitespace are 62 newline/tab pieces.

Two more things the runs show: the "request never returns" rows are generation running to `max_num_tokens` (letters stay allowed; 82 s on this Mac
for 8154 tokens), and `max_tokens` is ignored on every path because the 0.16.1 OpenAI handler reads only `max_completion_tokens` (with
`max_completion_tokens: 3` the constrained run stops at 3 tokens). A grammar with a genuine dead end errors out; it does not hang. The server then
drops the connection instead of delivering that 500, because `send_error` puts the error text (`▁` included) into the Latin-1 status line.

## Files

- `probe_3510.py` — the reporter's six cases plus the literal-spelling grammars and the `max_tokens` / `max_completion_tokens` checks, one JSON
  line per request (`content`, `finish_reason`, `usage`, wall time). `--model "path.litertlm,gpu,8192"`, `--cases a,b`, `--max-tokens-field`.
- `sp_table_audit.py` — reads the `SP_Tokenizer` section straight out of a `.litertlm` and reports the piece-table facts above (needs
  `litert-lm-builder` and `sentencepiece`).
- `probe_*.jsonl` — the runs behind the table; `serve_deadend_grammar_excerpt.log` — the server side of the `cat dog` row; `sp_table_audit.log`.

## Reproduce

```sh
python3 -m venv venv && venv/bin/pip install litert-lm==0.16.1 requests sentencepiece protobuf
venv/bin/litert-lm serve --api openai --host 127.0.0.1 --port 8093 &
venv/bin/python probe_3510.py --model "$HOME/models/gemma-4-E2B-it.litertlm,gpu,8192" --cases none,regex_nl_or_space,json_schema,regex_space,regex_u2581
venv/bin/python sp_table_audit.py "$HOME/models/gemma-4-E2B-it.litertlm"
```

Until the table is built from bytes, no grammar can require a space; compact JSON (`{"a":["x"]}`) still works, which is how this stayed hidden.
