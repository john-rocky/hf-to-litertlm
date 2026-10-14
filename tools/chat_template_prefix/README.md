# chat_template_prefix — check a bundle's chat template against LiteRT-LM's multi-turn rule before shipping it

Before every message, LiteRT-LM renders the conversation so far without a generation prompt (old) and the conversation plus the new user message with one (new), requires new to start with old, and prefills only the difference. When new does not start with old, the message fails:

```
INTERNAL: The new rendered template string does not start with the previous rendered template string.
old_string: ...
new_string: ...
```

The same check runs a second time inside the refill: when a reply carried a channel (a thought), the next message rewinds the KV cache and prefills the history render again, comparing it with the render up to the rewind point. So a template passes only if every assistant turn in the history renders the same way wherever it sits, and a template can pass message 2 and fail message 3.

From litert-lm 0.18.0 the runtime hands the template each reply's reasoning as `reasoning_content` (and `channels`) next to its content parts, for every model type. litert-lm 0.17.1 did so only for bundles of the `lfm2` model type and, on the generic model type, for templates that read `content` only as parts; otherwise it passed a one-part reply as a plain string without them. So a template whose think handling depends on the turn's position passes on 0.17.1 and fails from 0.18.0 on, except in those two cases, where it fails on 0.17 already (the LFM2.5-Thinking row below is a litert-lm 0.17.0 run).

## Template shapes that fail

| shape | example | fails |
|---|---|---|
| the latest assistant turn keeps its think block, earlier turns get an empty one | `truncate_history_thinking` with `loop.index0 < ns.last_user_idx` (Nemotron-3-Nano) | message 2 when the reply carried a thought; thinking off passes |
| the turn after the last user message gets a think block even when the reasoning is empty | the Hugging Face Qwen3 template, `last_query_index` | message 2, thinking on or off, 0.17.1 too |
| think blocks kept only after the last user message, opt-in | `preserve_thinking` with `loop.index0 > ns.last_user_index` (LFM2.5) | message 2 when the reply carried a thought |
| the think block kept on the last assistant turn only | `loop.index0 == ns.last_assistant_index` (LFM2.5-Thinking) | message 3, in the refill (measured on litert-lm 0.17.0); message 2 passes because that turn is still the last assistant turn |
| the non-generation render ends with a token the generation render does not have | `{% if add_generation_prompt %}<assistant>{% else %}<placeholder>{% endif %}` (Hy-MT2) | message 2, every setting, 0.17.1 too |

A keyword search does not decide it: some templates compute `last_query_index` and never use it, and some render the reasoning on every turn, which is position-independent and passes. Render the template.

## Check a bundle on the desk

Install the renderers once: `pip install jinja2 minijinja==2.14.0` (checked with Python 3.14.6, Jinja2 3.1.6 and minijinja 2.14.0). The commands run from the repository root.

1. Get the template the bundle carries, without downloading weights: `python3 tools/vlm_jinja_history/fetch_template.py <org>/<repo>/<file>.litertlm` range-reads the header and LlmMetadata and writes the template to `tools/vlm_jinja_history/templates/<org>__<repo>__<file>.litertlm.jinja`.
2. Run the three-turn table:

   ```
   python3 tools/chat_template_prefix/desk_prefix.py table <template.jinja> tools/chat_template_prefix/examples/conv_three_turns.json
   ```

   Every cell must read PASS: message form × `enable_thinking` unset / true / false × message 2 and 3, each cell the check before the message and the refill check. A failing cell shows 40 characters on each side of the first differing character. The command exits 0 whether cells pass or fail (read the table), 2 when the conversation file does not hold three turns. `examples/nemotron3nano_original.jinja` is the template the published Nemotron-3-Nano-4B bundle carried until 2026-10-13 and gives a failing table (36 of 72 cells: every form that carries the thought, at message 2 and 3); `templates/nemotron3nano_history.jinja` is its fix and gives a passing one (72 of 72).
3. To confirm that the renderer matches your runtime build, calibrate it once against the runtime's own strings. The INTERNAL error above carries both; save `old_string` and `new_string` from one failing second message to two files, write that exchange as a conversation file in the shape of `examples/conv_three_turns.json` (the first user message, the reply's text and thought as the runtime returned them, the second user message; the calibration renders exactly this exchange, so a different conversation never matches), and run

   ```
   python3 tools/chat_template_prefix/desk_prefix.py calibrate <template.jinja> old.txt new.txt <conversation.json>
   ```

   It prints which renderer settings and message forms reproduce the two strings byte for byte, or `none`. The runtime's are minijinja with `trim_blocks`, `lstrip_blocks` and `keep_trailing_newline` on, its regex rewrites of Python idioms (`.split(x)[-1]` to `| split(x) | last` and the like), and replies as content parts with `channels` and `reasoning_content`. `examples/nemotron_old.txt` and `examples/nemotron_new.txt` are the two strings from one such failure of the Nemotron-3-Nano-4B bundle on litert-lm 0.18.0, and `examples/conv_three_turns.json` is the conversation that produced them (its replies and thoughts are the runtime's own output), so running the command on the three example files reproduces them with both engines.

`desk_render.py` renders with jinja2 (always) and with minijinja-py 2.14.0, the runtime's crate version, when it is installed; the first line of every output names the engines used. The two engines gave the same verdict on all 62 Jinja templates found in 153 published bundles. A template that raises at render time fills its cells with ERROR and the exception's message. `selftest.py` re-runs the three checks above on the example files (calibration, the failing original, the passing fix) and exits 0 when all three hold. The conversation file holds `on` (three turns with thoughts) and `off` (three turns without), each turn `user`, `content` and an optional `thought`.

## Fix

Give every assistant turn in the history one form, independent of its position. For a thinking model that is the form the original template already used once a newer user message followed, an empty think block and the answer. Keeping the reasoning on every history turn also passes, but then the runtime re-prefills the previous reasoning on every message (619 characters instead of 292 in the two refills of the example conversation). Re-pack the metadata only:

```
python3 scripts/set_prompt_template.py <old>.litertlm <new>.litertlm <fixed>.jinja --litert-lm litert-lm
python3 scripts/section_identity.py <old>.litertlm <new>.litertlm      # metadata-only: True
```

Then run three turns on the runtime (litert-lm 0.18.0, thinking on and off). Worked example: `templates/nemotron3nano_history.jinja` and the 2026-10-13 paragraph on Nemotron-3-Nano-4B in REPRODUCE.md.

Content parts are the other 0.18 rule, about the message rather than the history: a template that reads `message.content` only as a string renders an empty turn ([../chat_template_parts/](../chat_template_parts/)).
