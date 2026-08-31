# vlm_jinja_history — role 'assistant' history dropped by 'model'-only templates

Companion scripts for the LiteRT-LM issue on the FastVLM-0.5B bundle's jinja
template (its role conditions are `'user'` / `'model'` / `'system'`, so a
conversation created through `create_conversation(messages=[...])` with the
standard `assistant` role silently loses every assistant turn from the rendered
prompt). Several of this repo's own published VLM conversions copied that
template and carry the same defect; they are being re-packed.

| file | what it does |
|---|---|
| `fetch_template.py` | range-reads the header + LlmMetadata of a published `.litertlm` from the Hub (no weights) and extracts its `jinja_prompt_template` to `templates/*.jinja`, recording which roles the conditions match. |
| `repro_history_roles.py` | the two-arm engine repro: injects the history `[user "one", <role> "ok"]` and renders the next user turn, once with `role='assistant'` and once with `role='model'`. The only variable between the arms is the role string. On the affected bundles the assistant arm renders two consecutive user turns. |
| `fix_template.py` | widens every `message.role == 'model'` condition in the template to `... or message.role == 'assistant'` and repacks metadata-only; every other section stays byte-identical. Re-run `repro_history_roles.py` on the output. |
| `repack_lib.py` | the metadata-only repack helper (`litert-lm unpack` → proto edit → `litert-lm pack`, plus per-section sha256 comparison). |

## Run

```
pip install litert-lm-builder litert-lm-api
export HF_HUB_DISABLE_XET=1
python3 fetch_template.py                       # defaults to litert-community/FastVLM-0.5B
python3 repro_history_roles.py /path/to/FastVLM-0.5B.litertlm      # defect_reproduced=True
python3 fix_template.py /path/to/FastVLM-0.5B.litertlm             # conditions widened: 3
python3 repro_history_roles.py FastVLM-0.5B.assistant_fix.litertlm # defect_reproduced=False
```

Measured with litert-lm-api 0.16.1 on the published `FastVLM-0.5B.litertlm`
(sha256 `ccba1e8b…`): the assistant arm drops the history turn, the model arm
renders it; after the 3-condition widening both arms render identically. The
engine's own `send_message` flow is unaffected either way — the incremental
render after a reply returns only the new user turn, so only conversations
constructed or restored through `messages=[...]` hit the drop.
