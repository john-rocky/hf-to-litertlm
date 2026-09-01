# think_prefix — the structured-template path never renders the generation prompt

`litert-torch`'s `parse_chat_template` (`litert_torch/generative/export_hf/core/litert_lm_builder.py`) derives `prompt_templates.model.prefix` from an assistant **history** turn (`add_generation_prompt=False`, `enable_thinking=False`). LiteRT-LM sends `model.prefix` as the generation prompt (`runtime/core/session_utils.cc`, `runtime/util/model_type_utils.cc`). For hybrid Qwen3 the two strings coincide (the no-think scaffold), so nobody noticed. For thinking-only models they differ, and the bundle prompts the model with a scaffold the vendor template never produces at generation time — reasoning switches off, silently. Reported upstream as google-ai-edge/litert-torch#TBD.

| tokenizer / template | vendor generation prompt | extracted `model.prefix` |
|---|---|---|
| Qwen/Qwen3-4B-Thinking-2507 | `<\|im_start\|>assistant\n<think>\n` | `<\|im_start\|>assistant\n<think>\n\n</think>\n\n` |
| ibm-granite/granite-4.2-3b | `<\|im_start\|>assistant\n<think>\n` | `<\|im_start\|>assistant\n<think></think>` |
| ChatML, `<think>\n` only in the generation branch | `<\|im_start\|>assistant\n<think>\n` | `<\|im_start\|>assistant\n` |
| ChatML, `<think>\n` in the assistant history branch too | `<\|im_start\|>assistant\n<think>\n` | `<\|im_start\|>assistant\n<think>\n` |

Measured with litert-torch 0.9.3 and 0.9.4; the function is identical on main.

## Files

- `repro_think_prefix.py` — prints the vendor generation prompt next to what `parse_chat_template` extracts, for any tokenizer (optionally with `--template file.jinja`). No export needed.
- `set_model_prefix.py` — sets `prompt_templates.model.prefix` on a built `.litertlm`, metadata only. Refuses unless every tflite section stays byte-identical and the metadata differs in nothing else. Needs `litert-lm-builder`, the `litert-lm` CLI, and `../add_thought_channel.py`.

## Workarounds until the exporter renders the generation prompt

1. **Template shape**: put the think opener in the assistant history branch too (the last table row; this repo's `templates/chatml_think.jinja`). The runtime renders history turns from the same prefix, so a past turn re-renders as prefix + generated text + suffix.
2. **Built bundles**: `set_model_prefix.py in.litertlm out.litertlm --prefix '<|im_start|>assistant\n<think>\n'` — the new prefix must end with the bundle's declared thought-channel start, which is what makes LiteRT-LM ≥ 0.16 pre-open the channel.

After every thinking-model export, read `prompt_templates.model.prefix` out of the bundle before trusting it.
