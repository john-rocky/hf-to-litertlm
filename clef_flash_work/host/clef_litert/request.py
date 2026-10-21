"""Request -> ids and spans: the checks of the checkpoint's own `systemone()` and its `encode_record()`
(joint_schema_model.py at Cloudflare/clef-flash f0b7d3cd, sha256 0e304cf7...), written out on a `tokenizers.Tokenizer`
built from the repository's own tokenizer.json (sha256 06b95093...).

Each piece is tokenized on its own without special tokens and the pieces are concatenated (the author's `_tokens`):

    prefix   "<|im_start|>system\\n{SYSTEM_PROMPT}<|im_end|>\\n<|im_start|>user\\nSTATE:\\n"                    36 ids
    [image]  "<|vision_start|>" + N x "<|image_pad|>" + "<|vision_end|>\\n"                                 N + 3 ids
    state    render(state): a string as is, anything else compact JSON (sorted keys, ensure_ascii off)
    schema   "\\n\\nSCHEMA FIELDS:\\n", then per question
             "\\nFIELD {i}\\nID: {id}\\nTYPE: {type}\\nINSTRUCTION: " + [render(instructions or id)]
             + "\\nALLOWED OPTIONS:\\n" + per option "OPTION {j}: " + [render({"option_id", "description"})] + "\\n",
             and "END FIELD\\n"                                         [..] = the head's spans, [start, end)
    suffix   "\\n<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\nJOINT SCHEMA DECISIONS:"         18 ids

Option order (the head scores options in this order): noul = (true, false), the author's default descriptions
overridable by `criteria`; choice = option ids sorted as strings; score = levels 0..n-1. The image block (one picture,
N = H x W merged tokens) sits right after the prefix, where the processor's media ids go; its index is the
`token_offset`. The state is cut to fit `max_length` (the author's 16384) exactly as encode_record cuts it; the row
graphs' own limit (4,096) is the host's check (host.py), never a cut.

Malformed requests: the author's ValueError conditions raise the named subclasses of errors.py with the author's
messages; anything the author's code would stop on otherwise (a question that is not an object, choice criteria that
are not an object, unsortable state keys) stops here the same way (AttributeError, TypeError, ...)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .errors import BadQuestionType, EmptyCriteria, EmptyRecord, MissingModelOrState, NoQuestions, SchemaTooLong

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
MAX_LENGTH = 16384                    # encode_record / systemone default
VISION_START, VISION_END, IMAGE_PAD = 248053, 248054, 248056
PAD_ID = 248044                       # <|endoftext|>, the processor's pad (never written into a row: pads are zeros)


def render(value: Any) -> str:
    """A string as is; anything else compact JSON (the author's `render`)."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def question_options(question: dict) -> list[tuple[str, Any]]:
    """(option id, description) in the order the head scores them (the author's `question_options`)."""
    question_type = str(question["type"])
    if question_type == "noul":
        criteria = dict(NOUL_CRITERIA)
        criteria.update(question.get("criteria") or {})
        return [(key, criteria[key]) for key in ("true", "false")]
    if question_type == "choice":
        return sorted((str(key), value) for key, value in question["criteria"].items())
    return [(str(index), value) for index, value in enumerate(question["criteria"])]


def validate_request(request: dict) -> None:
    """The checks `systemone()` makes before it encodes, in its order and with its messages."""
    questions = request.get("questions")
    if not isinstance(request.get("model"), str) or "state" not in request:
        raise MissingModelOrState("model and state are required")
    if not isinstance(questions, dict) or not questions:
        raise NoQuestions("at least one question is required")
    for question_id, question in questions.items():
        if question.get("type") not in QUESTION_TYPES:
            raise BadQuestionType(f"{question_id}: type must be noul, choice, or score")
        if question["type"] != "noul" and not question.get("criteria"):
            raise EmptyCriteria(f"{question_id}: criteria must not be empty")


def tokens(tok, text: str) -> list[int]:
    """`text` -> ids without special tokens (`tokenizers.Tokenizer`)."""
    return list(tok.encode(text, add_special_tokens=False).ids)


def image_block(tok, n: int) -> list[int]:
    """The processor's media ids of one picture of n merged tokens: <|vision_start|>, n pads, <|vision_end|>, "\\n"."""
    ids = tokens(tok, "<|vision_start|>" + "<|image_pad|>" * n + "<|vision_end|>\n")
    if not (ids[:1] == [VISION_START] and ids[1:n + 1] == [IMAGE_PAD] * n and ids[n + 1:n + 2] == [VISION_END]
            and len(ids) == n + 3):
        raise ValueError("the tokenizer does not keep the image placeholder tokens whole")
    return ids


@dataclass(frozen=True)
class EncodedQuestion:
    question_id: str
    type: str                                   # noul | choice | score
    type_id: int                                # 0 | 1 | 2 (the head's type embedding)
    question_span: tuple[int, int]              # [start, end) in ids
    option_spans: tuple[tuple[int, int], ...]
    option_ids: tuple[str, ...]


@dataclass(frozen=True)
class EncodedRecord:
    ids: tuple[int, ...]                        # the processor-form ids (N x <|image_pad|> for a picture)
    questions: tuple[EncodedQuestion, ...]
    token_offset: int | None = None             # index of <|vision_start|> (= len(prefix)), None without a picture
    image_grid: tuple[int, int] | None = None   # merged (H, W) of the picture, None without one

    @property
    def tokens(self) -> int:
        return len(self.ids)


def encode_record(tok, record: dict, max_length: int = MAX_LENGTH, max_state_tokens: int | None = None,
                  image_grid: tuple[int, int] | None = None) -> EncodedRecord:
    """The author's encode_record. `image_grid` = the merged (H, W) of one picture (its block goes right after the
    prefix), None for a record without one."""
    schema_ids = tokens(tok, SCHEMA_HEADER)
    questions = []
    for question_index, (question_id, question) in enumerate(record["questions"].items()):
        schema_ids += tokens(tok, f"\nFIELD {question_index + 1}\nID: {question_id}\nTYPE: {question['type']}"
                                  f"\nINSTRUCTION: ")
        question_start = len(schema_ids)
        instructions = question.get("instructions")
        if instructions is None or instructions == "":
            instructions = str(question_id)
        schema_ids += tokens(tok, render(instructions))
        question_end = len(schema_ids)
        schema_ids += tokens(tok, "\nALLOWED OPTIONS:\n")
        option_spans, option_ids = [], []
        for option_index, (option_id, description) in enumerate(question_options(question)):
            schema_ids += tokens(tok, f"OPTION {option_index + 1}: ")
            option_start = len(schema_ids)
            semantics = {"option_id": option_id}
            if description is not None:
                semantics["description"] = description
            schema_ids += tokens(tok, render(semantics))
            option_spans.append((option_start, len(schema_ids)))
            option_ids.append(option_id)
            schema_ids += tokens(tok, "\n")
        schema_ids += tokens(tok, "END FIELD\n")
        questions.append((str(question_id), str(question["type"]), QUESTION_TYPES[str(question["type"])],
                          (question_start, question_end), tuple(option_spans), tuple(option_ids)))

    prefix_ids = tokens(tok, PREFIX)
    suffix_ids = tokens(tok, SUFFIX)
    token_offset = None
    if image_grid is not None:
        h, w = (int(v) for v in image_grid)
        token_offset = len(prefix_ids)
        prefix_ids = prefix_ids + image_block(tok, h * w)
        image_grid = (h, w)
    state_ids = tokens(tok, render(record["state"]))
    if max_state_tokens is not None:
        state_ids = state_ids[:max_state_tokens]
    fixed_length = len(prefix_ids) + len(schema_ids) + len(suffix_ids)
    if fixed_length > max_length:
        raise SchemaTooLong(f"schema requires {fixed_length} tokens before state; maximum is {max_length}")
    state_ids = state_ids[: max_length - fixed_length]
    shift = len(prefix_ids) + len(state_ids)
    shifted = tuple(
        EncodedQuestion(question_id=qid, type=qtype, type_id=type_id,
                        question_span=(span[0] + shift, span[1] + shift),
                        option_spans=tuple((a + shift, b + shift) for a, b in option_spans),
                        option_ids=option_ids)
        for qid, qtype, type_id, span, option_spans, option_ids in questions)
    ids = tuple(prefix_ids + state_ids + schema_ids + suffix_ids)
    if not ids or not shifted:
        raise EmptyRecord("record produced no model input or questions")
    return EncodedRecord(ids=ids, questions=shifted, token_offset=token_offset, image_grid=image_grid)
