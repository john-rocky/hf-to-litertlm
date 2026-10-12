# Qwen3.5 on LiteRT-LM 0.18.0: cross-conversation state on a shared Engine, and tool-call parsing (2026-10-12)

Probes run on a Mac (Apple M4 Max, macOS 27.0) with the PyPI wheel `litert-lm==0.18.0` (Python API, CPU backend, greedy
`SamplerConfig(top_k=1)`, `max_num_tokens=1024`), one fresh process per run, against the current litert-community files:
`Qwen3.5-2B_int8.litertlm` (sha256 e7473f57…), `Qwen3.5-0.8B_int8.litertlm` (64ed396f…), `Qwen3.5-4B_mixed_int4.litertlm`
(37209c42…) and, for the digit test, `Qwen3.5-4B_int8.litertlm` (a360b83d…). Upstream thread for the state carry-over:
https://github.com/google-ai-edge/LiteRT-LM/issues/3165.

## Run
```bash
uv run --no-project --with litert-lm==0.18.0 python -I tools/leak_probe.py sysswitch --model <bundle> --label 2B --mode shared --run 1 --out results/leak/2B_sysswitch_shared_r1.json
uv run --no-project --with litert-lm==0.18.0 python -I tools/leak_probe.py qset --model <bundle> --label 2B --mode fresh --question 0 --run 1 --out results/leak/2B_qset_fresh_q00.json
uv run --no-project --with litert-lm==0.18.0 python -I tools/tc_probe.py --model <bundle> --label 2B --arm bundle --automatic 1 --run 1 --out results/tc/2B_bundle_auto_r1.json
uv run --no-project --with litert-lm==0.18.0 python -I tools/digit_probe.py --model <bundle> --label 2B_int8 --run 1 --out results/digit/2B_int8_r1.json
```
`tools/run_phase1.sh` … `run_phase3.sh` are the drivers that produced `results/` (87 state-probe files,
24 tool-call files, 9 digit-test files). `--arm canonical` passes the LiteRT-LM
`models/qwen3_5/chat_template.jinja` (main 03d3979f) as the conversation's `chat_template`; `--arm workaround` runs a
metadata-only re-pack whose `LlmMetadataProto.pbtext` is in `workaround/` (llm_model_type `qwen3` + the `models/qwen3`
template; built with `tools/make_workaround_pbtext.py`; the re-pack reports are the `*_repack_report.json`).

## State carry-over across conversations on one Engine (results/leak/)
`sysswitch`: conversation 1 (system prompt "Your name is Alpha. Begin every reply with the word Alpha.", 3 turns), then
conversation 2 on the same Engine (system prompt "… Beta …", the same 3 turns); control = conversation 2 alone on a fresh Engine.

| file | shared Engine: conversation 2 opens with "Beta" | fresh Engine (control) |
|---|---|---|
| 2B int8 | 0 of 9 turns (3 runs); answers otherwise right | 9 of 9 |
| 0.8B int8 | 0 of 9; "What is your name?" → "I am Qwen3.5." | 9 of 9 |
| 4B mixed_int4 | the whole reply is the single word "Beta", 9 of 9 turns | 9 of 9, with the answers |

`qset`: 20 short questions, a new Conversation per question on one Engine (round 1, then the same 20 as round 2) vs one
process per question; right answers of 20, three runs:

| file | shared, round 1 | shared, round 2 | one process per question |
|---|---|---|---|
| 2B int8 | 20, 20, 19 | 19, 19, 19 (repetition loops in 5, 1, 0 answers) | 20 |
| 0.8B int8 | 18, 15, 17 | 18, 16, 17 | 19 |
| 4B mixed_int4 | 20, 19, 19 | 19, 18, 19 (wrong arithmetic) | 19 (one derivation cut by the 64-token cap) |

## Tool calls (results/tc/)
With the published files the tool list never enters the prompt (their template has no `tools` branch) and the model answers
in plain text. With the official `models/qwen3_5` template the model emits `<tool_call>\n<function=get_weather>\n<parameter=location>\nParis…`
and the runtime returns it as text (`generic_model`: no parser). With `llm_model_type qwen3` and the `models/qwen3` template
(JSON inside `<tool_call>`), the runtime parses the call and the Python tool runs, 3 of 3 on the 2B and the 0.8B.

## Digit test (results/digit/)
"What is 17 times 23?" at the default rendering, 3 fresh CPU processes per file: 2B int8 391 3/3; 4B mixed_int4 391 3/3;
4B int8 391 3/3, one run then repeated its check line to the 400-token cap.
