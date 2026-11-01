# Multi-turn gate for chat bundles

`multiturn_gate.py` runs one three-turn conversation on a `.litertlm` bundle through the LiteRT-LM Python API: plant a fact, ask a short sum, recall the fact. Decoding is greedy, and every run gets a fresh process. A turn fails when its text misses the expected answer, is empty, or carries a template marker (a thought tag or a role marker). One-turn checks such as `scripts/verify_quality.py` do not see what this gate catches: the runtime's prefix contract (a hard error "does not start with the previous rendered template string", or a silent rewind), state damage that shows only in a later turn, and a thinking model that answers the first questions and then ends the third turn empty.

```sh
pip install litert-lm==0.19.0
python tools/gates/multiturn_gate.py model.litertlm --backend gpu --thinking on --json-out mt_gpu.json
python tools/gates/multiturn_gate.py model.litertlm --backend cpu --thinking on --json-out mt_cpu.json
```

Exit 0 = PASS (every run of every mode passed), 1 = FAIL, 2 = usage error, 3 = UNDECIDED (no run failed, and a run was still void after its retries). The verdict covers one backend, so run the gate once for each backend the bundle is meant for.

## History modes

`--modes` takes a comma-separated list of `filter`, `keep` and `default` (the default list is `filter,keep`):

- `filter` passes `filter_channel_content_from_kv_cache=True`: the channel content of a reply (the thought) leaves the KV cache before the next turn.
- `keep` passes `False`: the thought stays in the KV cache.
- `default` does not pass the argument. LiteRT-LM then filters: the Python and C APIs pass the setting only when it is given, and `ConversationConfig::Builder` starts it at true (`runtime/conversation/conversation.h`, LiteRT-LM 0.15.0 to 0.19.0; 0.10.2 to 0.14.0 start it at false; earlier releases have no such setting). So on those versions `default` runs the same setting as `filter`.

The report keeps the conversation's token count after each turn, and the count tells the settings apart. On `Spark-X2.6-1.7B_int8.litertlm` (Mac GPU, LiteRT-LM 0.19.0, thinking on), `default` and `filter` both counted 87 / 109 / 133 tokens after the three turns, with the same thoughts. `keep` counted 260 / 359 / 434. The Spark-X2.5 gate in `spark_work/multiturn_gate.py`, written for LiteRT-LM 0.17, runs one conversation without the argument and one with `True`, which is the same setting twice.

## Runs, void runs and processes

- `--processes` sets the fresh processes per mode (default 3). The CPU path can answer differently from one process to the next. On the Mac CPU, the token count of `Spark-X2.6-1.7B_int8.litertlm` after the third turn was 133, 133 and 136 in three processes of `filter`, and 445, 427 and 553 in three of `keep`. On the same Mac CPU, `Spark-X2.6-4B_int4.litertlm` ended the third turn empty right after the thought in 3 of 8 conversations, while the GPU passed (measured with the earlier gate: two conversations in each of 4 processes on one engine, both filtering the thought). One CPU pass is not a CPU verdict.
- In LiteRT-LM 0.18.x and 0.19.0, a blocking `send_message` that is still generating after 10 minutes fails with `DEADLINE_EXCEEDED`. The timeout is fixed and has no setting, and its clock keeps running while the process is stopped. Such a run is void, not an answer of the model, and it runs again in a new process, up to `--void-retries` times (default 2). In 0.19.0 the generation that the deadline leaves running can end the process soon after, so the gate writes the void result as soon as the call fails, and it also counts as void a process that ends without a result after `DEADLINE_EXCEEDED` reached its stderr.
- Any other run whose process ends without a result is a failure: a crash, or a process killed at `--run-timeout` (default 3600 s).

## Options

- `--thinking template|on|off` (default `template`): `template` passes no `ThinkingConfig`, so `enable_thinking` stays out of the template context and the chat template's own default decides. `on` and `off` pass `ThinkingConfig(enable_thinking=...)`.
- `--max-output-tokens` (per turn, default 3072): a thinking model needs the room to close its thought. If the cap hits inside the thought, the result looks like state damage.
- `--max-num-tokens` (the engine's context, default 4096).
- `--cache-dir` (created if missing; run one process at a time per directory).
- `--turns TURNS.json`: a JSON list of `[message, regex or null]` that replaces the three built-in turns.
- `--leak STR`: one more marker that a reply must not contain (repeatable).
- `--json-out REPORT.json`.

## Checked

On `Spark-X2.6-1.7B_int8.litertlm` ([litert-community/Spark-X2.6-1.7B](https://huggingface.co/litert-community/Spark-X2.6-1.7B), sha256 e81b8821…), Apple M4 Max, LiteRT-LM 0.19.0 from PyPI, thinking on, 2026-11-01:

- GPU: `filter` 3 / 3 and `keep` 3 / 3.
- CPU: `filter` 3 / 3 and `keep` 3 / 3.
- A real deadline, on the CPU: the first attempt's process was stopped (SIGSTOP) for 620 s inside the second turn. When it went on, its `send_message` failed with `DEADLINE_EXCEEDED: Session 0 did not complete within the timeout of 10m.`, the gate wrote the void result, and the process then ended with SIGSEGV. The second attempt, in a new process, passed: exit 0.
- Failure paths, on the same file:
  - an expected answer the model cannot give: exit 1;
  - a marker the reply carries (`--leak Alfred`): exit 1;
  - a context the conversation does not fit in (`--max-num-tokens 128`): exit 1;
  - a void run followed by a pass in a new process: exit 0, also when the void run's process aborts right after the failed call;
  - every attempt void: exit 3;
  - a file that is not a bundle: exit 1;
  - a process killed at `--run-timeout 5`: exit 1;
  - usage errors (an unknown mode, a missing file, a `--turns` file that is not a list of `[message, regex or null]`, a `--turns` regex that does not compile, `--processes 0`): exit 2.

The other void runs use the gate's own test switches: `MULTITURN_GATE_TEST_VOID=N` makes attempts 1 to N fail as if the deadline had hit, and `MULTITURN_GATE_TEST_ABORT=1` then aborts that process, as 0.19.0 can after a real deadline.
