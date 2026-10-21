"""Named refusals of the clef-flash host.

Every refusal is a `RequestError`, itself a `ValueError`: code that catches the ValueError the checkpoint's own
`systemone()` raises catches these too, and the first six carry the author's message word for word. The last two are
the host's own: it reads one picture at a fixed tile and runs row graphs of at most 4,096 tokens."""
from __future__ import annotations


class RequestError(ValueError):
    """The request cannot be answered as given."""


class MissingModelOrState(RequestError):
    """`model` is not a string or `state` is absent ("model and state are required")."""


class NoQuestions(RequestError):
    """`questions` is not a non-empty object ("at least one question is required")."""


class BadQuestionType(RequestError):
    """A question's `type` is not noul, choice or score ("<id>: type must be noul, choice, or score")."""


class EmptyCriteria(RequestError):
    """A choice or score question without criteria ("<id>: criteria must not be empty")."""


class SchemaTooLong(RequestError):
    """The prefix, picture, schema and suffix alone exceed max_length (encode_record: "schema requires N tokens
    before state; maximum is M")."""


class EmptyRecord(RequestError):
    """The record gives no ids or no questions (encode_record: "record produced no model input or questions")."""


class UnsupportedMedia(RequestError):
    """Host only: more than one picture, `videos`, `media_kwargs`, a picture the host cannot open, or a tile other
    than 256 / 448 (the two tower graphs)."""


class RequestTooLong(RequestError):
    """Host only: the record is longer than the largest row graph (4,096 tokens). It is refused, never truncated."""
