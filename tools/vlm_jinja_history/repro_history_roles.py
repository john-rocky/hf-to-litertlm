#!/usr/bin/env python3
"""v1: minimal engine reproduction of the assistant-history drop.

Runs on litert-lm-api (pip install litert-lm-api).  For the given .litertlm bundle:
  arm A: create_conversation(messages=[{user,'one'},{role='assistant','ok'}]),
         then render the next user turn 'two' to a string.
  arm B: same with role='model'.
Exactly one variable moves between the arms (the role string of the injected
history turn).  The defect: arm A's render contains no trace of 'ok' — the
assistant turn is dropped from the prompt; arm B renders it.

Usage: python3 repro_history_roles.py /path/to/bundle.litertlm
Output: v1_repro_<basename>.json + stdout transcript.
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).parent


def main():
    path = sys.argv[1]
    from litert_lm.engine import Engine
    eng = Engine(path)
    rec = {"bundle": path, "arms": {}}
    try:
        for role in ("assistant", "model"):
            c = eng.create_conversation(messages=[
                {"role": "user", "content": "one"},
                {"role": role, "content": "ok"},
            ])
            render = c.render_message_to_string({"role": "user", "content": "two"})
            c.close()
            rec["arms"][role] = {
                "render": render,
                "history_turn_present": "ok" in render,
            }
            print(f"--- injected history role = {role!r}")
            print(render)
    finally:
        eng.close()
    rec["defect_reproduced"] = (
        rec["arms"]["assistant"]["history_turn_present"] is False
        and rec["arms"]["model"]["history_turn_present"] is True
    )
    out = HERE / f"v1_repro_{Path(path).stem}.json"
    out.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    print(f"defect_reproduced={rec['defect_reproduced']}  -> {out.name}")


if __name__ == "__main__":
    main()
