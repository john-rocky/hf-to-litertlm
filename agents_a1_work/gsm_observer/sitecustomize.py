"""Opt-in CLI observation only; no model settings, outputs or timing changed."""
import os
import sys

_out = os.environ.get("AGENTS_A1_GSM_OBSERVATION")
if _out:
    import functools
    import hashlib
    import json
    from pathlib import Path
    from litert_lm_cli.commands import run as cli_run

    _path = Path(_out).resolve(); _path.parent.mkdir(parents=True,exist_ok=True)
    _original = cli_run._execute_prompt

    @functools.wraps(_original)
    def observed_execute_prompt(state, conversation, prompt, attachments=()):
        data = {"method":"native Conversation.token_count (KV occupancy), sampled around unchanged CLI _execute_prompt",
                "prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest(),
                "max_num_tokens":4096, "timing_collected":False,
                "native_kv_tokens_before":conversation.token_count}
        try:
            return _original(state, conversation, prompt, attachments=attachments)
        finally:
            try:
                data["native_kv_tokens_after"] = conversation.token_count
                data["context_limit_reached"] = data["native_kv_tokens_after"] >= 4096
            except Exception as exc:
                data["observation_error"] = repr(exc)
                data["context_limit_reached"] = None
            _path.write_text(json.dumps(data,indent=2)+"\n")

    cli_run._execute_prompt = observed_execute_prompt
