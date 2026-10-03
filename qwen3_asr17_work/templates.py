"""The two prompt templates a Qwen3-ASR-1.7B bundle can carry (pure python: imported by export_bundle.py and by the
released-runtime scripts, whose venv has no torch)."""

# Both templates render message by message (so the render of a shorter history is always a prefix of a longer one, and
# an empty history renders '') and put the text items of the last user message after the generation prompt (where
# qwen-asr puts a forced 'language X<asr_text>' and its streaming loop the committed prefix).
# JINJA_OFFICIAL = the checkpoint's chat_template.jinja for [system?, user(audio)] + generation prompt:
# '<|im_start|>system\n{system text}<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n'
# '<|im_start|>assistant\n' (an empty system turn when there is no system message; transformers'
# apply_transcription_request(language=X) passes X as the system text).
# JINJA_LITERT = the litert-torch Qwen3AsrProcessor._PROMPT string, the prompt litert-torch's export bakes into its
# encoder output: '<|im_start|>user<|audio_start|><|audio_pad|><|audio_end|><|im_end|><|im_start|>assistant\n' (no
# system turn, no newline after the role names or <|im_end|>); system messages render nothing.
JINJA_LITERT = (
    "{%- set ns = namespace(prefix='') -%}"
    "{%- for m in messages -%}"
    "{%- if m.role == 'user' -%}"
    "{{- '<|im_start|>user' -}}"
    "{%- set ns.prefix = '' -%}"
    "{%- if m.content is string -%}{%- set ns.prefix = m.content -%}{%- else -%}"
    "{%- for c in m.content -%}"
    "{%- if c.type == 'audio' -%}{{- '<|audio_start|><|audio_pad|><|audio_end|>' -}}"
    "{%- elif c.type == 'text' -%}{%- set ns.prefix = ns.prefix + c.text -%}{%- endif -%}"
    "{%- endfor -%}"
    "{%- endif -%}"
    "{{- '<|im_end|>' -}}"
    "{%- elif m.role == 'assistant' -%}"
    "{{- '<|im_start|>assistant\\n' -}}"
    "{%- if m.content is string -%}{{- m.content -}}{%- else -%}"
    "{%- for c in m.content -%}{%- if c.type == 'text' -%}{{- c.text -}}{%- endif -%}{%- endfor -%}"
    "{%- endif -%}"
    "{{- '<|im_end|>' -}}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' + ns.prefix -}}{%- endif -%}"
)

JINJA_OFFICIAL = (
    "{%- set ns = namespace(has_system=false, prefix='') -%}"
    "{%- for m in messages -%}"
    "{%- if m.role == 'system' -%}"
    "{%- set ns.has_system = true -%}"
    "{{- '<|im_start|>system\\n' -}}"
    "{%- if m.content is string -%}{{- m.content -}}{%- else -%}"
    "{%- for c in m.content -%}{%- if c.type == 'text' -%}{{- c.text -}}{%- endif -%}{%- endfor -%}"
    "{%- endif -%}"
    "{{- '<|im_end|>\\n' -}}"
    "{%- elif m.role == 'user' -%}"
    "{%- if not ns.has_system -%}{{- '<|im_start|>system\\n<|im_end|>\\n' -}}{%- set ns.has_system = true -%}"
    "{%- endif -%}"
    "{{- '<|im_start|>user\\n' -}}"
    "{%- set ns.prefix = '' -%}"
    "{%- if m.content is string -%}{%- set ns.prefix = m.content -%}{%- else -%}"
    "{%- for c in m.content -%}"
    "{%- if c.type == 'audio' -%}{{- '<|audio_start|><|audio_pad|><|audio_end|>' -}}"
    "{%- elif c.type == 'text' -%}{%- set ns.prefix = ns.prefix + c.text -%}{%- endif -%}"
    "{%- endfor -%}"
    "{%- endif -%}"
    "{{- '<|im_end|>\\n' -}}"
    "{%- elif m.role == 'assistant' -%}"
    "{{- '<|im_start|>assistant\\n' -}}"
    "{%- if m.content is string -%}{{- m.content -}}{%- else -%}"
    "{%- for c in m.content -%}{%- if c.type == 'text' -%}{{- c.text -}}{%- endif -%}{%- endfor -%}"
    "{%- endif -%}"
    "{{- '<|im_end|>\\n' -}}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' + ns.prefix -}}{%- endif -%}"
)


JINJAS = {"litert": JINJA_LITERT, "official": JINJA_OFFICIAL}
