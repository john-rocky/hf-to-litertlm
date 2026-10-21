"""ClefHost: one request through the host steps around the graphs (the graphs themselves are the caller's callables).

    host = ClefHost.from_dir("<folder>")       # tokenizer.json, embed_tokens.safetensors, lm_head.safetensors,
                                               # joint_head.safetensors, joint_head_config.json
    enc = host.encode(request, tile=448)       # checks, ids, spans, bucket L; a picture -> enc.patches
    rows = tower(enc.tile, enc.patches)        # the tower graph: [G^2, 4096] picture rows (requests with a picture)
    x = host.row_inputs(enc, rows)             # {"embeds", "valid", "cos", "sin"} at length enc.length
    hidden = row_graph(enc.length, x)          # the row graph of that length: [1, L, 4096] after the final norm
    logits = host.scores(enc, hidden)          # the joint head, NumPy float32: one array per question
    body = host.respond(request, enc, logits)  # the SystemOne-compatible response

or `host.decide(request, row_graph, tower)` for all of it. Refusals: errors.py (the author's ValueError conditions
by name, plus the host's own: one picture at a 256 / 448 tile, at most 4,096 tokens)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .errors import UnsupportedMedia
from .head import JointHead
from .pixels import tile_grid, tile_patches
from .positions import BUCKETS, mrope_tables, padded_positions, pick_length, record_positions
from .request import MAX_LENGTH, EncodedRecord, encode_record, validate_request
from .response import question_probs, response
from .tables import Bf16Table

HIDDEN = 4096
FILES = {"tokenizer": "tokenizer.json", "embed_tokens": "embed_tokens.safetensors", "lm_head": "lm_head.safetensors",
         "joint_head": "joint_head.safetensors", "joint_head_config": "joint_head_config.json"}


@dataclass(frozen=True)
class Encoded:
    record: EncodedRecord
    length: int                          # the row graph length: the smallest bucket >= record.tokens
    tile: int | None = None              # 256 | 448 for a request with a picture
    patches: np.ndarray | None = None    # the tower graph's input [4 G^2, 1536] float32


def media_of(request: dict) -> list:
    """The request's one picture as a list ([] or [picture]); UnsupportedMedia for what the host does not read."""
    if request.get("videos"):
        raise UnsupportedMedia("videos are not read by this host")
    if request.get("media_kwargs"):
        raise UnsupportedMedia("media_kwargs are not read by this host (one picture at a fixed tile)")
    images = list(request.get("images") or [])
    if len(images) > 1:
        raise UnsupportedMedia(f"{len(images)} pictures: this host reads one picture per request")
    return images


class ClefHost:
    def __init__(self, tokenizer, embed_tokens: Bf16Table, lm_head: Bf16Table, head: JointHead,
                 buckets=BUCKETS, max_length: int = MAX_LENGTH):
        if embed_tokens.shape[1] != HIDDEN or lm_head.shape[1] != HIDDEN:
            raise ValueError(f"tables {embed_tokens.shape} / {lm_head.shape}: expected [V, {HIDDEN}]")
        self.tokenizer, self.embed_tokens, self.lm_head, self.head = tokenizer, embed_tokens, lm_head, head
        self.buckets, self.max_length = tuple(sorted(buckets)), max_length

    @classmethod
    def from_files(cls, tokenizer: str | Path, embed_tokens: str | Path, lm_head: str | Path, joint_head: str | Path,
                   joint_head_config: str | Path, **kw) -> "ClefHost":
        from tokenizers import Tokenizer

        return cls(Tokenizer.from_file(str(tokenizer)), Bf16Table(embed_tokens), Bf16Table(lm_head),
                   JointHead.from_files(joint_head, joint_head_config), **kw)

    @classmethod
    def from_dir(cls, folder: str | Path, **kw) -> "ClefHost":
        folder = Path(folder)
        return cls.from_files(**{k: folder / v for k, v in FILES.items()}, **kw)

    def encode(self, request: dict, tile: int = 448) -> Encoded:
        """The author's checks, the host's media checks, encode_record, the bucket (RequestTooLong past 4,096)."""
        validate_request(request)
        images = media_of(request)
        grid = None
        if images:
            g = tile_grid(tile)
            grid = (g, g)
        record = encode_record(self.tokenizer, request, max_length=self.max_length, image_grid=grid)
        length = pick_length(record.tokens, self.buckets)
        if not images:
            return Encoded(record, length)
        return Encoded(record, length, tile, tile_patches(images[0], tile))

    def row_inputs(self, enc: Encoded, image_rows: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """The row graph's inputs at enc.length: embeds = the float32 rows of the bf16 table at the ids, the picture's
        rows (the tower's output, [H * W, 4096]) at the <|image_pad|> positions; valid = 1 on the record; the pad
        positions after it: embeds 0, valid 0; cos / sin of the record's planes (positions.py)."""
        rec, length = enc.record, enc.length
        t = rec.tokens
        embeds = np.zeros((1, length, HIDDEN), np.float32)
        embeds[0, :t] = self.embed_tokens.rows(rec.ids)
        if rec.image_grid is not None:
            h, w = rec.image_grid
            if image_rows is None:
                raise ValueError("a request with a picture needs the tower's rows")
            image_rows = np.asarray(image_rows, dtype=np.float32)
            if image_rows.shape != (h * w, HIDDEN):
                raise ValueError(f"picture rows {image_rows.shape}, expected {(h * w, HIDDEN)}")
            p = rec.token_offset
            embeds[0, p + 1:p + 1 + h * w] = image_rows
        elif image_rows is not None:
            raise ValueError("picture rows for a request without a picture")
        valid = np.zeros((1, length), np.float32)
        valid[0, :t] = 1.0
        cos, sin = mrope_tables(padded_positions(record_positions(t, rec.token_offset, rec.image_grid), length))
        return {"embeds": embeds, "valid": valid, "cos": cos, "sin": sin}

    def scores(self, enc: Encoded, hidden: np.ndarray, lexical: list | None = None) -> list[np.ndarray]:
        """The row graph's hidden ([1, L, 4096] or [T, 4096]) -> the head's logits, one float32 array per question."""
        hidden = np.asarray(hidden, dtype=np.float32)
        if hidden.ndim == 3:
            hidden = hidden[0]
        t = enc.record.tokens
        return self.head.scores(hidden[:t], enc.record.ids, enc.record.questions, self.lm_head, lexical=lexical)

    def respond(self, request: dict, enc: Encoded, logits: list[np.ndarray]) -> dict:
        return response(request, enc.record.questions, [question_probs(z) for z in logits], enc.record.tokens)

    def decide(self, request: dict, row_graph: Callable[[int, dict], np.ndarray],
               tower: Callable[[int, np.ndarray], np.ndarray] | None = None, tile: int = 448) -> dict:
        enc = self.encode(request, tile=tile)
        rows = None
        if enc.patches is not None:
            if tower is None:
                raise UnsupportedMedia("a request with a picture needs a tower")
            rows = tower(enc.tile, enc.patches)
        hidden = row_graph(enc.length, self.row_inputs(enc, rows))
        return self.respond(request, enc, self.scores(enc, hidden))
