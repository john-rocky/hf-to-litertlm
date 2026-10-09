# Source: prompt.py from https://huggingface.co/LiquidAI/d1-omni-600M at revision
# 414f8d6438174f5b2133a9c21a478fc42625e308 (file sha256 a6b29a55ec8345f1fc1fdbfcc4b64d80d473dc4316f095a62c51cb6cce1194cf).
# Licensor: Liquid AI, Inc. Licensed under the LFM Open License v1.0 (the LICENSE file of that repository).
# Changed by the d1-omni LiteRT port: these five comment lines were added; everything below them is the
# original file, byte for byte.
"""Questions in, token sequences out, answers back.

A question is a dict in the Decision Index schema: `type` (noul, choice or score), `instructions`, and
`criteria`. Each question is rendered against the state as one sequence:

    <bos> <state> state <q> instructions <opt> <mask> option_0 </opt> <opt> <mask> option_1 </opt> ... <decide>

The model scores the hidden state at every `<mask>` and softmaxes over the question's options. Delimiters come
from the tokenizer's reserved block, and `<|...|>` in caller text is rewritten so a state can never forge one.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

QTYPES = {"choice": 0, "score": 1, "noul": 2}
DELIM = {"state": "<|reserved_7|>", "q": "<|reserved_8|>", "opt": "<|reserved_9|>", "opt_end": "<|reserved_10|>",
         "decide": "<|reserved_11|>"}
MARKER = "<|mask|>"
_SPECIAL = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


@dataclass
class Question:
    type: str
    instructions: str
    criteria: Any = None

    def __post_init__(self):
        if self.type not in QTYPES:
            raise ValueError(f"question type must be one of {sorted(QTYPES)}, got {self.type!r}")
        if self.type == "choice" and (not isinstance(self.criteria, dict) or len(self.criteria) < 2):
            raise ValueError("a choice needs criteria {name: description} with at least two options")
        if self.type == "score" and (not isinstance(self.criteria, (list, tuple)) or not 2 <= len(self.criteria) <= 10):
            raise ValueError("a score needs criteria: a list of 2 to 10 level descriptions, lowest first")
        if self.type == "noul" and self.criteria is not None and not isinstance(self.criteria, dict):
            raise ValueError('noul criteria are optional: {"true": "...", "false": "..."} (or "yes", "no")')

    @property
    def options(self) -> int:
        return 2 if self.type == "noul" else len(self.criteria)


def as_question(q: Any) -> Question:
    if isinstance(q, Question):
        return q
    if not isinstance(q, dict) or "type" not in q or "instructions" not in q:
        raise ValueError("a question is a dict with `type`, `instructions` and, for choice and score, `criteria`")
    return Question(q["type"], str(q["instructions"]), q.get("criteria"))


def escape(text: str) -> str:
    """`<|name|>` -> `<¦name¦>`, so caller text cannot emit a delimiter or marker token."""
    return _SPECIAL.sub(r"<¦\1¦>", text)


def serialize(state: Any) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def _criterion(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(", ", ": "),
                                                           default=str)


def render_options(q: Question, noul_default: dict | None = None, audio: bool = False) -> list[str]:
    """Option texts in the model's order. A noul is read as [false, true]. After an audio prefix, options are
    written as the audio questions were trained: `option_000: text`, and a noul as `false: no`, `true: yes`."""
    if q.type == "choice":
        if audio:
            return [f"option_{i:03d}: {_criterion(k if v is None or v == '' else v)}"
                    for i, (k, v) in enumerate(q.criteria.items())]
        return [k if v is None or v == "" else f"{k}: {_criterion(v)}" for k, v in q.criteria.items()]
    if q.type == "score":
        return [f"level {i}: {_criterion(c)}" for i, c in enumerate(q.criteria)]
    if audio:
        return ["false: no", "true: yes"]
    crit = q.criteria or noul_default or {}
    false, true = crit.get("false", crit.get("no")), crit.get("true", crit.get("yes"))
    return ["false: " + (_criterion(false) if false not in (None, "") else "no, the statement does not hold"),
            "true: " + (_criterion(true) if true not in (None, "") else "yes, the statement holds")]


def encode(tok, state: Any, q: Question, max_len: int, noul_default: dict | None = None, audio: bool = False,
           per_option: int = 24) -> tuple[list[int], list[int]]:
    """Token ids of one question over one state, and the position of each option's marker.

    The option block gets max(96, min(24k + 32, max_len / 2)) tokens, shared out evenly; the state is
    truncated on the right to the room that is left.
    """
    ids_of = tok.convert_tokens_to_ids
    enc = lambda s: tok(escape(s), add_special_tokens=False)["input_ids"]  # noqa: E731
    opts = render_options(q, noul_default, audio)
    budget = max(96, min(len(opts) * per_option + 32, max_len // 2))
    per = max(2, (budget - 3 * len(opts)) // len(opts))
    question = ([ids_of(DELIM["q"])] + enc(q.instructions))[: max(16, budget)]
    markers = []
    for text in opts:
        markers.append(len(question) + 1)
        question += [ids_of(DELIM["opt"]), ids_of(MARKER)] + enc(" " + text)[:per] + [ids_of(DELIM["opt_end"])]
    question.append(ids_of(DELIM["decide"]))
    room = max(0, max_len - len(question) - 2)
    state_ids = [ids_of(DELIM["state"])] + enc(serialize(state))[:room]
    ids = ([tok.bos_token_id] + state_ids + question)[:max_len]
    markers = [m + 1 + len(state_ids) for m in markers]
    if markers[-1] >= max_len:
        raise ValueError("the options do not fit in the context")
    return ids, markers


def answer(q: Question, probs: list[float]) -> dict:
    """A noul's P(yes) (its probabilities are [yes, no]); a choice's pick and its probabilities; a score's
    expected level."""
    if q.type == "noul":
        return {"type": "noul", "noul": probs[0]}
    best = max(range(len(probs)), key=probs.__getitem__)
    if q.type == "choice":
        names = list(q.criteria)
        return {"type": "choice", "choice": names[best], "confidence": probs[best],
                "probabilities": dict(zip(names, probs))}
    return {"type": "score", "score": sum(i * p for i, p in enumerate(probs)), "confidence": probs[best],
            "probabilities": {str(i): p for i, p in enumerate(probs)},
            "legend": {str(i): _criterion(text) for i, text in enumerate(q.criteria)}}


def temperature_key(q: Question) -> str:
    k = q.options
    return f"{q.type}:" + ("2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+")
