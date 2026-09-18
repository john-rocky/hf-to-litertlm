# Re-run on litert-lm 0.17.1 — 2026-09-19

Same two-arm script (`repro_history_roles.py`, unchanged), same published file, newer runtime.

| item | value |
|---|---|
| runtime | `pip install litert-lm==0.17.1` (litert-lm-api 0.17.1, litert-lm-builder 0.17.1), fresh venv, Python 3.12, macOS arm64 |
| published file | `litert-community/FastVLM-0.5B/FastVLM-0.5B.litertlm`, sha256 `ccba1e8bfa0bab78345f5d009fdffd20bd8c907cf39b4bc632e391b0a96f3b18`, 1,156,342,768 bytes (Hub commit `4600132463`, 2026-06-24 — unchanged since the report) |
| control | the `fix_template.py` output of the report (three `'model'` conditions widened to `'model' or 'assistant'`), sha256 `c0c6305a3dcb0719233593ceda8b9143bc6c6c691d327d6771172a840eca15ca` |

```
python3 repro_history_roles.py FastVLM-0.5B.litertlm                 # defect_reproduced=True   (v1_repro_FastVLM-0.5B.json)
python3 repro_history_roles.py FastVLM-0.5B.assistant_fix.litertlm   # defect_reproduced=False  (v1_repro_FastVLM-0.5B.assistant_fix.json)
```

| bundle | injected history role | history turn in the render |
|---|---|---|
| published | `assistant` | no — two consecutive user turns |
| published | `model` | yes |
| control (widened template) | `assistant` | yes |
| control (widened template) | `model` | yes |

The published file's renders are byte-identical to the 0.16.1 run recorded in the issue. `section_diff.json`: the control differs from the published file in the LlmMetadata section only; the SP_Tokenizer and four TFLiteModel sections are byte-identical.

On this wheel `litert_lm.Role.MODEL.value` is still `"model"`; the engine's own `send_message` flow returns the reply as `{"role": "assistant", ...}` and renders only the new user turn afterwards, as in the report.
