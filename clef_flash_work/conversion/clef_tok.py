"""The token path of the checkpoint's own `encode_record()` (joint_schema_model.py, sha256 0e304cf7...) written out on a
`tokenizers.Tokenizer` alone (no torch, no transformers): the host form of round 5, checked in round 1 against the
author's function (scripts/tok_compare.py).

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file("<snapshot>/tokenizer.json")
    enc = encode_record(tok, request)          # {"ids", "questions": [...], "prefix_len"}

Each piece is tokenized on its own without special tokens and the pieces are concatenated (the author's `_tokens`):
    prefix   "<|im_start|>system\\n{SYSTEM_PROMPT}<|im_end|>\\n<|im_start|>user\\nSTATE:\\n"
    state    render(state): a string as is, anything else compact JSON (sorted keys, ensure_ascii off)
    schema   "\\n\\nSCHEMA FIELDS:\\n", then per question
             "\\nFIELD {i}\\nID: {id}\\nTYPE: {type}\\nINSTRUCTION: " + [render(instructions or id)]
             + "\\nALLOWED OPTIONS:\\n" + per option "OPTION {j}: " + [render({"option_id", "description"})] + "\\n",
             and "END FIELD\\n"                              ([..] = the head's spans, [start, end))
    suffix   "\\n<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\nJOINT SCHEMA DECISIONS:"
Option order: noul = (true, false) with the author's default criteria overridable; choice = option ids sorted as
strings; score = levels 0..n-1. The state is cut to fit max_length exactly as encode_record cuts it. Media (images)
are not handled here: the author's media block sits right after the prefix (round 3)."""
from __future__ import annotations

import json
from typing import Any

SYSTEM_PROMPT = (
    "Read the complete state and schema. Decide every field jointly. Each answer "
    "must be exactly one of that field's allowed options."
)
PREFIX = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"
SUFFIX = "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"
SCHEMA_HEADER = "\n\nSCHEMA FIELDS:\n"
QUESTION_TYPES = {"noul": 0, "choice": 1, "score": 2}
NOUL_CRITERIA = {
    "true": "The proposition is true or the answer is yes.",
    "false": "The proposition is false or the answer is no.",
}
MAX_LENGTH = 16384


def render(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def question_options(question: dict) -> list[tuple[str, Any]]:
    kind = str(question["type"])
    if kind == "noul":
        criteria = dict(NOUL_CRITERIA)
        criteria.update(question.get("criteria") or {})
        return [(key, criteria[key]) for key in ("true", "false")]
    if kind == "choice":
        return sorted((str(key), value) for key, value in question["criteria"].items())
    return [(str(index), value) for index, value in enumerate(question["criteria"])]


def tokens(tok, text: str) -> list[int]:
    return list(tok.encode(text, add_special_tokens=False).ids)


def encode_record(tok, record: dict, max_length: int = MAX_LENGTH, max_state_tokens: int | None = None) -> dict:
    schema = tokens(tok, SCHEMA_HEADER)
    questions = []
    for qi, (qid, q) in enumerate(record["questions"].items()):
        schema += tokens(tok, f"\nFIELD {qi + 1}\nID: {qid}\nTYPE: {q['type']}\nINSTRUCTION: ")
        q0 = len(schema)
        instructions = q.get("instructions")
        if instructions is None or instructions == "":
            instructions = str(qid)
        schema += tokens(tok, render(instructions))
        q1 = len(schema)
        schema += tokens(tok, "\nALLOWED OPTIONS:\n")
        spans, oids = [], []
        for oi, (oid, desc) in enumerate(question_options(q)):
            schema += tokens(tok, f"OPTION {oi + 1}: ")
            o0 = len(schema)
            sem = {"option_id": oid}
            if desc is not None:
                sem["description"] = desc
            schema += tokens(tok, render(sem))
            spans.append([o0, len(schema)])
            oids.append(oid)
            schema += tokens(tok, "\n")
        schema += tokens(tok, "END FIELD\n")
        questions.append({"question_id": str(qid), "type_id": QUESTION_TYPES[str(q["type"])],
                          "question_span": [q0, q1], "option_spans": spans, "option_ids": oids})
    prefix = tokens(tok, PREFIX)
    suffix = tokens(tok, SUFFIX)
    state = tokens(tok, render(record["state"]))
    if max_state_tokens is not None:
        state = state[:max_state_tokens]
    fixed = len(prefix) + len(schema) + len(suffix)
    if fixed > max_length:
        raise ValueError(f"schema requires {fixed} tokens before state; maximum is {max_length}")
    state = state[: max_length - fixed]
    shift = len(prefix) + len(state)
    for q in questions:
        q["question_span"] = [q["question_span"][0] + shift, q["question_span"][1] + shift]
        q["option_spans"] = [[a + shift, b + shift] for a, b in q["option_spans"]]
    ids = prefix + state + schema + suffix
    if not ids or not questions:
        raise ValueError("record produced no model input or questions")
    return {"ids": ids, "questions": questions, "prefix_len": len(prefix)}
