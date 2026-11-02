"""d1-Omni-2B's (and d1-2B's) System One API with no PyTorch: the provider's `model.system_one(state, questions,
media=None, set_size=None)` around three backends, the language model (`decide`), the picture tower (`tower`) and the
audio encoder (`audio`).

Source of truth: LiquidAI/d1-Omni-2B @ 431f1d66 (the snapshot directory passed in; d1-2B @ 10e7671d serves too, with no
audio). What the host imports and what it ports:

- Imported from the snapshot, unchanged (they need no torch): `prompt` (render, question classes, slot layouts,
  readout tables), `api` (`SystemOneApi`: question checks, `system_one_batch`, `answer`), `box`, `grid`,
  `calibration`, `parsing`, `json_spans`, `media` (`media_list`, `cap_pixels`, `spoken`, `SPOKEN`, `read_audio` for
  WAV and .npy clips). They are loaded as a package of their own (`provider_modules`: a module object whose `__path__`
  is the snapshot, so their relative imports resolve; no bytecode is written into the snapshot). The lead `lfm3.py`
  registers on import (`LEADS`: "<think></think>" for lfm3 / lfm3_vl / lfm3_omni) is read from its source with `ast`
  (lfm3.py imports torch).
- Ported (their modules import torch): `runner.SlotReader._serve / _prefix / _suffixes / _sequences / _padded`,
  `runner.group_size`, `SystemOne._readouts` (= `prompt.readouts` in numpy), `attention.pack_set / state_length`,
  `pointer.StatePositions / state_positions / query_plan + read_logits (one row) / spans`, the media reader
  (`media.Reader._encode / place`): pictures through `Lfm3VlImageProcessor` (d1_pictures.py), clips through
  `media.resampled` + `Lfm3OmniFeatureExtractor` (d1_audio_features.py), each item's markup as `Lfm3OmniProcessor`
  expands it; and the tokenizer (d1_tokenizer.py).

Engine settings, as `SystemOne.__init__` and `_serve` read them: config.json's `d1_codes` (2), `d1_numbers` (true),
`d1_box_edges` (absent: "median"), `d1_prompt` (absent: 1), `d1_vision_max_pixels` (1,638,400; the environment's
D1_VISION_MAX_PIXELS overrides it, as in the provider), `d1_attention` ("set"), `d1_set_size` (16); the tokenizer's BOS
string; the lead of the model type; state style "json_only", no system turn, option style "desc"; calibration.json
(d1-Omni-2B: temperature 3.383 for option answers; typed tables).

Backends (the host never pads; buckets are the backend's business):
- `decide(embeds float32 [T, 2048], state_len, question list[int] of length T) -> float32 [T, 2048]`: the language
  model's final-norm hidden state (`embedding_norm`) at every position under the provider's `set` attention (`question`:
  -1 on the state, k on question k's tokens). A backend may carry `max_len`: a longer sequence is refused (ValueError)
  before any backend runs.
- `tower(pixel_values float32 [N, 1536], (gh, gw)) -> float32 [N // 4, 2048]`: vision tower + projector, the tokens in
  the provider's order (`get_image_features`). One call per picture of a request.
- `audio(input_features float32 [128, F]) -> float32 [ceil(F / 8), 2048]`: audio encoder + projector on one clip, its
  tokens in order (`get_audio_features` with that clip alone, every frame valid). A backend may carry `max_frames`: a
  longer clip is refused (ValueError) before any backend runs. One call per clip of a request.

Embeddings: the tied bf16 table `model.language_model.embed_tokens.weight` [128000, 2048] read as its raw bits and
widened to float32 (bits << 16, exact = the provider's float32 table); the picture tokens replace the `<|image|>` rows
and the clips' tokens the `<|audio|>` rows, each in the order of the sequence's media (the state's items, then each
packed question's spoken clip), as the provider's two `masked_scatter`s fill them.

Readout: the provider projects each code slot's hidden state on the whole vocabulary in float32, subtracts the
log-sum-exp and reads the option tokens (`prompt.readouts`: the best form of each option, float64, divided by the
temperature, a softmax). That softmax is invariant to a shift common to the slot's logits, and the log-sum-exp is one,
so the host computes only the logits the readout table names, hidden · rowᵀ in float32 with the table's float32 rows,
and runs the same numpy steps. What is left of the shift is the rounding of the provider's float32 subtraction
(measured on the reference's stored slots, 584 batches: at most 7.1e-8 on the probabilities; without it the host's
readout equals the provider's bit for bit). The sums of a float32 product depend on its shape
(on Accelerate, which numpy and torch both use here: one row takes a matrix-vector path that sums in another order than
two or more rows; a product with few columns, under 16 for one row, a handful for two to five rows, takes another path
again; with 16 or more columns every element is the full product's): so the host groups the sequences into the
provider's batches (`_batches`: the backends still run one sequence a call), takes a batch's code-slot logits from all
its code slots at once, as the provider's LM head reads them, and pads each readout table with zero rows to a multiple
of 16. A read slot is scored against its row's state positions in the provider's shapes (`pointer.query_plan` +
`read_logits`: float32, / sqrt(d), positions off the mask -inf); the span search (`pointer.spans`) runs in float64.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import math
import os
import struct
import sys
import time
import types
from bisect import bisect_right
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from d1_audio_features import FeatureExtractor, read_audio, resampled
from d1_pictures import PictureProcessor
from d1_tokenizer import TokenizerAdapter

PROVIDER_MODULES = ("media", "grid", "json_spans", "prompt", "calibration", "box", "parsing", "api")
EMBED_KEY = "model.language_model.embed_tokens.weight"
IMAGE_MARKUP = "<|vision_start|><|image|><|vision_end|>"  # chat_template.jinja's picture item (checked at load)


# --------------------------------------------------------------------------- #
# the provider's torch-free modules, and constants of the ones that import torch
# --------------------------------------------------------------------------- #


def _assigned(snap: str, file: str, name: str) -> ast.AST:
    """The value node of `name = ...` at the top of the snapshot's `file`."""
    with open(os.path.join(snap, file)) as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if any(isinstance(t, ast.Name) and t.id == name for t in targets) and node.value is not None:
            return node.value
    raise RuntimeError(f"{file} no longer assigns {name}: read it again")


def constant(snap: str, file: str, name: str) -> Any:
    """A literal constant of a provider module the host cannot import (it imports torch)."""
    return ast.literal_eval(_assigned(snap, file, name))


def default_argument(snap: str, file: str, cls: str, method: str, name: str) -> Any:
    """A literal default of a provider method the host cannot import (SystemOne.__init__'s token_budget)."""
    with open(os.path.join(snap, file)) as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method:
                    args = item.args.args[len(item.args.args) - len(item.args.defaults):]
                    for arg, default in zip(args, item.args.defaults):
                        if arg.arg == name:
                            return ast.literal_eval(default)
    raise RuntimeError(f"{file} no longer gives {cls}.{method} a default {name}: read it again")


def family_leads(snap: str) -> dict[str, str]:
    """What lfm3.py adds to `prompt.LEADS` on import: `LEADS.update(dict.fromkeys(MODULES, lead))`."""
    modules = constant(snap, "lfm3.py", "MODULES")
    with open(os.path.join(snap, "lfm3.py")) as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        call = node.value if isinstance(node, ast.Expr) else None
        if (isinstance(call, ast.Call) and ast.unparse(call.func) == "LEADS.update" and len(call.args) == 1
                and isinstance(call.args[0], ast.Call) and ast.unparse(call.args[0].func) == "dict.fromkeys"
                and ast.unparse(call.args[0].args[0]) == "MODULES"):
            return dict.fromkeys(modules, ast.literal_eval(call.args[0].args[1]))
    raise RuntimeError("lfm3.py no longer registers its lead as LEADS.update(dict.fromkeys(MODULES, lead))")


def provider_modules(snapshot_dir: str) -> types.SimpleNamespace:
    """The snapshot's torch-free modules, imported as one package of their own (once a process and snapshot)."""
    snap = os.path.realpath(snapshot_dir)
    name = "d1_provider_" + hashlib.sha1(snap.encode()).hexdigest()[:12]
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [snap]
        package.__package__ = name
        sys.modules[name] = package
        quiet = sys.dont_write_bytecode
        sys.dont_write_bytecode = True  # never a __pycache__ in the snapshot
        try:
            for module in PROVIDER_MODULES:
                importlib.import_module(f"{name}.{module}")
            sys.modules[f"{name}.prompt"].LEADS.update(family_leads(snap))
        except BaseException:
            for key in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
                del sys.modules[key]
            raise
        finally:
            sys.dont_write_bytecode = quiet
    return types.SimpleNamespace(package=name, snapshot=snap,
                                 **{m: sys.modules[f"{name}.{m}"] for m in PROVIDER_MODULES})


def group_size(n: int | None) -> int | None:
    """runner.group_size."""
    if n is None:
        return None
    if isinstance(n, bool) or not isinstance(n, int):
        raise ValueError(f"set_size must be a whole number, not {n!r}")
    if n < 1:
        raise ValueError(f"set_size must be at least 1, not {n}")
    return n


# --------------------------------------------------------------------------- #
# the embedding table
# --------------------------------------------------------------------------- #


def safetensors_bf16(path: str, key: str = EMBED_KEY) -> np.ndarray:
    """A BF16 tensor of a safetensors file as its raw bits (uint16, memory-mapped), from the file's own header."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    entry = header.get(key)
    if entry is None or entry["dtype"] != "BF16":
        raise ValueError(f"{path}: no BF16 tensor {key}")
    shape = tuple(int(s) for s in entry["shape"])
    begin, end = entry["data_offsets"]
    if end - begin != 2 * int(np.prod(shape)):
        raise ValueError(f"{path}: {key} holds {end - begin} bytes for shape {shape}")
    return np.memmap(path, dtype="<u2", mode="r", offset=8 + n + begin, shape=shape)


def write_embed_table(snapshot_dir: str, out_path: str) -> dict:
    """The tied table's bf16 bits as an .npy of uint16 [vocab, hidden], with a JSON note beside it."""
    source = os.path.join(snapshot_dir, "model.safetensors")
    bits = safetensors_bf16(source)
    digest = hashlib.sha256()
    for start in range(0, bits.shape[0], 8192):
        digest.update(np.ascontiguousarray(bits[start:start + 8192]).tobytes())
    tmp = out_path + ".tmp.npy"
    np.save(tmp, bits)
    os.replace(tmp, out_path)
    note = {"source": source, "tensor": EMBED_KEY, "stored_as": "uint16 = the bf16 bits; float32 = bits << 16",
            "shape": list(bits.shape), "sha256_of_bits": digest.hexdigest(), "npy_bytes": os.path.getsize(out_path)}
    with open(os.path.splitext(out_path)[0] + ".json", "w") as f:
        json.dump(note, f, indent=1)
    return note


class EmbedTable:
    """The tied bf16 table as uint16 bits: an .npy written by `write_embed_table`, the snapshot's model.safetensors,
    or an array. `rows(ids)` widens the rows to float32 (bits << 16): exactly the provider's float32 table."""

    def __init__(self, source: str | np.ndarray):
        if isinstance(source, np.ndarray):
            bits = source
        elif str(source).endswith(".npy"):
            bits = np.load(source, mmap_mode="r")
        else:
            bits = safetensors_bf16(str(source))
        if bits.dtype != np.uint16 or bits.ndim != 2:
            raise ValueError(f"an embedding table is uint16 [vocab, hidden] bf16 bits, not {bits.dtype} {bits.shape}")
        self.bits = bits
        self.vocab, self.hidden = bits.shape

    def rows(self, ids: Sequence[int] | np.ndarray) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= self.vocab):
            raise ValueError(f"token id outside the table ({int(ids.min())}..{int(ids.max())}, vocab {self.vocab})")
        return (np.asarray(self.bits[ids]).astype(np.uint32) << 16).view(np.float32)


# --------------------------------------------------------------------------- #
# pointer.py and attention.py, ported
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StatePositions:
    """pointer.StatePositions: token positions mapped to the caller's rendered state; None = not a text position."""

    text: str
    offsets: tuple
    paths: tuple = ()
    json_text: Callable[[str], str] | None = None

    @property
    def valid(self) -> list[bool]:
        return [i == 0 or span is not None for i, span in enumerate(self.offsets)]

    @property
    def pointer_mask(self) -> list[bool]:
        valid = self.valid
        return valid[:max(i for i, v in enumerate(valid) if v) + 1]

    def read(self, first: int, last: int, max_tokens: int = 512) -> dict:
        if first == last == 0:
            return {"text": None, "span": None}
        if not 0 < first <= last < len(self.offsets) or last - first >= max_tokens:
            raise ValueError("invalid pointer endpoints")
        a, b = self.offsets[first], self.offsets[last]
        if a is None or b is None:
            raise ValueError("pointer leaves the state text")
        span = [a[0], b[1]]
        while span[0] < span[1] and self.text[span[0]].isspace():
            span[0] += 1
        while span[1] > span[0] and self.text[span[1] - 1].isspace():
            span[1] -= 1
        held = [(start, end, p) for start, end, p in self.paths if start < span[1] and span[0] < end]
        if len(held) == 1:
            start, end, path = held[0]
            span = [max(span[0], start), min(span[1], end)]
            return {"text": self.json_text(self.text[slice(*span)]), "span": span, "path": path}
        return {"text": self.text[slice(*span)], "span": span}


def state_positions(p: types.SimpleNamespace, tokenizer, state, prefix: str, full_text: str,
                    style: str = "json_only", *, encoded=None) -> StatePositions:
    """pointer.state_positions, with the provider's prompt and json_spans modules `p`."""
    prompt = p.prompt
    if not full_text.startswith(prefix):
        raise ValueError("the tokenized prompt must start with the state prefix")
    if state is None:
        body, escaped, boundaries = "", "", [0]
        start = len(prefix)
    else:
        body = prompt.state_block(state, style)
        escaped, boundaries = prompt.plain_offsets(tokenizer, body)
        tail = escaped + "\n" + prompt.question_header(tokenizer, state)
        if not prefix.endswith(tail):
            raise ValueError("state rendering differs from the prompt prefix")
        start = len(prefix) - len(tail)
    encoded = (
        tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True) if encoded is None else encoded
    )
    text = prompt.state_text(state, style) if state is not None else ""
    stop = start + bisect_right(boundaries, len(text)) - 1
    offsets = []
    for a, b in encoded["offset_mapping"]:
        span = None
        lo, hi = max(a, start), min(b, stop)
        if lo < hi and not full_text[a:lo].strip() and not full_text[hi:b].strip():
            left, right = boundaries[lo - start], boundaries[hi - start]
            if left < right <= len(text) and text[left:right].strip():
                span = (left, right)
        offsets.append(span)
    if not offsets or offsets[0] is not None:
        raise ValueError("position 0 must be a template token for the nothing readout")
    paths = (
        p.json_spans.string_paths(text)
        if state is not None and not isinstance(state, str) and style.endswith("_only") and style.startswith("json")
        else []
    )
    return StatePositions(text, tuple(offsets), tuple(paths), p.json_spans.json_text)


def spans(logits: np.ndarray, max_tokens: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """pointer.spans on [R, 2, T] start / end scores, in float64: found when P(found) >= 0.5, then the likeliest span
    within `max_tokens` (max_pool1d's and argmax's first maximum); (0, 0) for nothing."""
    x = np.asarray(logits, dtype=np.float64)
    top = x.max(-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = x - top - np.log(np.exp(x - top).sum(-1, keepdims=True))
    start, end = logp[:, 0], logp[:, 1]
    found = -np.expm1(start[:, 0])
    count = x.shape[0]
    if x.shape[2] == 1:
        return np.zeros((count, 2), dtype=np.int64), found
    cap = min(max_tokens, x.shape[2] - 1)
    padded = np.concatenate([end[:, 1:], np.full((count, cap - 1), -np.inf)], axis=1)
    windows = np.lib.stride_tricks.sliding_window_view(padded, cap, axis=1)  # [R, T - 1, cap]
    within = windows.argmax(-1)
    best = np.take_along_axis(windows, within[..., None], -1)[..., 0]
    scores = start[:, 1:] + best
    index = scores.argmax(-1)
    rows = np.arange(count)
    at = np.stack((index + 1, index + within[rows, index] + 1), -1).astype(np.int64)
    at[found < 0.5] = 0
    return at, found


def state_length(tokenizer, ids: list[int], suffix: str) -> int:
    """attention.state_length."""
    return len(ids) - len(tokenizer.encode(suffix, add_special_tokens=False))


def pack_set(tokenizer, prefix: str, suffixes: list[str], header: str = "",
             slot_offsets: list[list[int]] | None = None, first_ids: list[int] | None = None,
             ) -> tuple[list[int], int, list[int], list[list[int]]]:
    """attention.pack_set."""
    if not suffixes:
        raise ValueError("a question set cannot be empty")
    offsets = [[0] for _ in suffixes] if slot_offsets is None else slot_offsets
    if len(offsets) != len(suffixes):
        raise ValueError("one slot layout is required per question")

    def positions(k: int, length: int, end: int) -> list[int]:
        own = offsets[k]
        if (not own or own != sorted(set(own)) or own[-1] != 0
                or any(isinstance(o, bool) or not isinstance(o, int) or not -length < o <= 0 for o in own)):
            raise ValueError("slots must be increasing positions inside their question, ending at its last token")
        return [end + o for o in own]

    ids = tokenizer.encode(prefix + suffixes[0], add_special_tokens=False) if first_ids is None else list(first_ids)
    state = state_length(tokenizer, ids, suffixes[0])
    question = [-1] * state + [1] * (len(ids) - state)
    ends = [positions(0, len(ids) - state, len(ids) - 1)]
    for k, suffix in enumerate(suffixes[1:], 2):
        own = tokenizer.encode(header + suffix, add_special_tokens=False)
        ids, question = ids + own, question + [k] * len(own)
        ends.append(positions(k - 1, len(own), len(ids) - 1))
    return ids, state, question, ends


# --------------------------------------------------------------------------- #
# the host
# --------------------------------------------------------------------------- #


@dataclass(eq=False)
class Picture:
    """One picture's processor outputs: pixel_values [gh * gw, 1536] and its patch grid."""

    pixel_values: np.ndarray
    grid: tuple[int, int]


@dataclass(eq=False)
class Clip:
    """One clip's feature extractor output: input_features [128, frames] (every frame valid), the encoder tokens it
    makes (ceil(frames / 8)) and its samples at 16 kHz."""

    input_features: np.ndarray
    tokens: int
    samples: int

    @property
    def frames(self) -> int:
        return int(self.input_features.shape[1])


@dataclass(frozen=True)
class AudioMarkup:
    """How Lfm3OmniProcessor writes a clip (processor_config.json): `_strip_spans` takes the template's span pair off
    the placeholder, `replace_audio_token` puts back `_wrap(audio_token * tokens, audio_markers)`."""

    token: str
    start: str
    end: str
    markers: bool

    @property
    def template(self) -> str:  # chat_template.jinja's audio item
        return self.start + self.token + self.end

    def __call__(self, tokens: int) -> str:
        placeholders = self.token * tokens
        return self.start + placeholders + self.end if self.markers else placeholders


class D1Host:
    """The provider's engine (`SystemOne`) and API (`SystemOneApi`) over `decide`, `tower` and `audio` backends."""

    serves_typed = True  # api.SystemOneApi reads it

    def __init__(self, snapshot_dir: str | None, decide: Callable | None, tower: Callable | None = None,
                 embed_table: str | np.ndarray | None = None, audio: Callable | None = None, *,
                 lead: str | None = None):
        self.snapshot_dir = os.path.realpath(snapshot_default() if snapshot_dir is None else snapshot_dir)
        self.p = p = provider_modules(self.snapshot_dir)
        with open(os.path.join(self.snapshot_dir, "config.json")) as f:
            self.config = json.load(f)
        self.tokenizer = TokenizerAdapter(self.snapshot_dir)
        bos = self.tokenizer.bos_token
        self.bos = bos if isinstance(bos, str) else ""
        self.lead = p.prompt.default_lead(self.config.get("model_type"), self.tokenizer) if lead is None else lead
        self.state_style, self.system, self.option_style = p.prompt.DEFAULT_STATE_STYLE, p.prompt.DEFAULT_SYSTEM, "desc"
        found = os.path.join(self.snapshot_dir, "calibration.json")
        self.calibration = p.calibration.Calibration.load(found) if os.path.exists(found) else None
        self.span_caps = constant(self.snapshot_dir, "pointer.py", "SPAN_CAPS")
        self.token_budget = default_argument(self.snapshot_dir, "runner.py", "SystemOne", "__init__", "token_budget")
        self._serve(self.config)
        self.decide, self.tower, self.audio = decide, tower, audio
        self.embed = EmbedTable(os.path.join(self.snapshot_dir, "model.safetensors") if embed_table is None
                                else embed_table)
        hidden = self.config.get("text_config", {}).get("hidden_size")
        if hidden is not None and hidden != self.embed.hidden:
            raise ValueError(f"the table's rows hold {self.embed.hidden} values, the model's hidden size is {hidden}")
        with open(os.path.join(self.snapshot_dir, "processor_config.json")) as f:
            self.processor_config = json.load(f)
        self.takes_audio = self.processor_config.get("feature_extractor") is not None  # media.PROCESSORS["audio"]
        self._pictures: PictureProcessor | None = None
        self._features: FeatureExtractor | None = None
        self._markups: dict[tuple[int, ...], str] = {}
        self._readout_rows: dict[tuple[int, ...], np.ndarray] = {}
        self._depth = 0
        self.last: dict = self._new_last()

    # ----------------------------------------------------------------- instruments

    @staticmethod
    def _new_last() -> dict:
        """What the last `run` / `system_one_batch` call did: its sequences (as `sequences()` gives them, with their
        request index), the hidden states at each sequence's slots [S, d] in slot order, the read slots' position
        logits of each batch that has reads (as the provider's `_slot_logits` returns them), the audio backend's output
        for each clip it ran (in call order) and for each sequence its clips' outputs in the sequence's media order
        (`seq_audio`), the per-question values (`run`'s result), and where the time went (prep
        = questions, render, tokenizer, pictures' and clips' preprocessing, embeddings; feature = the clips' reading,
        resampling and log-mel, a part of prep; readout = slot logits, readouts, pointer, answers; host = everything
        but the backends)."""
        return {"seqs": [], "slot_hidden": [], "read_logits": [], "audio_features": [], "seq_audio": [], "probs": None,
                "timing": {"prep_s": 0.0, "feature_s": 0.0, "tower_s": 0.0, "audio_s": 0.0, "decide_s": 0.0,
                           "readout_s": 0.0, "tower_calls": 0, "audio_calls": 0, "decide_calls": 0, "tokens": 0}}

    def _tick(self, phase: str, since: float) -> float:
        now = time.perf_counter()
        self.last["timing"][phase] += now - since
        return now

    def _finish(self, since: float) -> None:
        timing = self.last["timing"]
        timing["wall_s"] = time.perf_counter() - since
        timing["host_s"] = timing["wall_s"] - timing["tower_s"] - timing["audio_s"] - timing["decide_s"]

    # ----------------------------------------------------------------- settings

    def _serve(self, recorded: Mapping) -> None:
        """SlotReader._serve with the checkpoint's config and no override."""
        tok, p = self.tokenizer, self.p
        tok.d1_codes = recorded.get("d1_codes", 1)
        tok.d1_numbers = bool(recorded.get("d1_numbers", False))
        tok.d1_box_edges = recorded.get("d1_box_edges") or "median"
        if tok.d1_box_edges not in p.box.EDGE_READS:
            raise ValueError(f"d1_box_edges is one of {p.box.EDGE_READS}, not {tok.d1_box_edges!r}")
        tok.d1_prompt = recorded.get("d1_prompt", 1)
        if tok.d1_prompt not in p.prompt.PROMPTS:
            raise ValueError(f"d1_prompt is one of {p.prompt.PROMPTS}, not {tok.d1_prompt!r}")
        pixels_set = os.environ.get("D1_VISION_MAX_PIXELS")  # runner.PIXELS_SET
        vision_max_pixels = int(pixels_set or 1024 * 1024)   # runner.VISION_MAX_PIXELS
        self.max_pixels = (vision_max_pixels if pixels_set else
                           recorded.get("d1_vision_max_pixels") or vision_max_pixels)
        self.attention = recorded.get("d1_attention") or "causal"
        trained = constant(self.snapshot_dir, "attention.py", "TRAINED_SET_SIZE")
        self.set_size = group_size(recorded.get("d1_set_size", constant(self.snapshot_dir, "attention.py", "SET_SIZE")))
        if not 2 <= self.set_size <= trained:
            raise ValueError(f"d1_set_size must be between 2 and {trained}")
        if self.attention != "set":
            raise NotImplementedError(f"the host serves `set` attention (d1-2B's), not {self.attention!r}")

    # ----------------------------------------------------------------- the API (api.SystemOneApi, unchanged)

    def ask(self, q):
        return self.p.api.SystemOneApi.ask(self, q)

    def named(self, questions):
        t0 = time.perf_counter()
        try:
            return self.p.api.SystemOneApi.named(self, questions)
        finally:
            if self._depth:
                self._tick("prep_s", t0)

    def answer(self, state, q, probs, score_readout: str | None = None, abstain_below: float | None = None) -> dict:
        score_readout = self.p.api.SCORE_READOUT if score_readout is None else score_readout
        t0 = time.perf_counter()
        try:
            return self.p.api.SystemOneApi.answer(self, state, q, probs, score_readout, abstain_below)
        finally:
            if self._depth:
                self._tick("readout_s", t0)

    def probabilities(self, state, questions, media=None):
        return self.p.api.SystemOneApi.probabilities(self, state, questions, media)

    def system_one(self, state: Any, questions: Mapping[str, Any], media: Sequence | None = None,
                   score_readout: str | None = None, set_size: int | None = None, *,
                   abstain_below: float | None = None) -> dict:
        """model.system_one: `{"answers": {name: answer}, "usage": {"input_tokens": n, "output_tokens": 0}}`."""
        return self.system_one_batch([(state, questions, media)], score_readout, set_size,
                                     abstain_below=abstain_below)[0]

    def system_one_batch(self, requests: Sequence[tuple], score_readout: str | None = None,
                         set_size: int | None = None, *, abstain_below: float | None = None) -> list[dict]:
        """model.system_one_batch: `(state, questions)` or `(state, questions, media)` requests, one response each,
        composed by the provider's own `SystemOneApi.system_one_batch` (named -> run -> answer)."""
        score_readout = self.p.api.SCORE_READOUT if score_readout is None else score_readout
        t0 = time.perf_counter()
        self.last = self._new_last()
        self._depth += 1
        try:
            return self.p.api.SystemOneApi.system_one_batch(self, requests, score_readout, abstain_below, set_size)
        finally:
            self._depth -= 1
            self._finish(t0)

    def run(self, requests: Sequence[tuple], set_size: int | None = None) -> list[tuple[list, int]]:
        """SystemOne.run: for each `(state, [Question], media)` request, the per-question values `_padded` produces
        (the input of `api.answer`) and the tokens it read."""
        t0 = time.perf_counter()
        outer = self._depth > 0
        if not outer:
            self.last = self._new_last()
        self._depth += 1
        try:
            set_size = group_size(set_size)
            requests = [(state, qs, self.p.media.media_list(media)) for state, qs, media in requests]
            return self._padded(requests, set_size)
        finally:
            self._depth -= 1
            if not outer:
                self._finish(t0)

    def sequences(self, state: Any, questions: Mapping[str, Any], media: Sequence | None = None,
                  set_size: int | None = None) -> list[dict]:
        """The request's sequences, as the provider's runner builds them (plus the position map, the pixel
        values and the clips' input features, joined as `media.join` joins them)."""
        named = self.named(questions)
        reqs = [(state, list(named.values()), self.p.media.media_list(media))]
        maps: dict = {}
        seqs = self._sequences(reqs, position_maps=maps, set_size=set_size)
        return [self._seq_record(n, seq, maps, with_request=False) for n, seq in enumerate(seqs)]

    @staticmethod
    def _seq_record(n: int, seq: tuple, maps: dict, with_request: bool = True) -> dict:
        i, k, ids, state_len, question, ends, parts = seq
        pics = [p for p in parts if isinstance(p, Picture)]
        clips = [p for p in parts if isinstance(p, Clip)]
        features = None
        if clips:  # media.join: the clips' [1, 128, F] stacked, zero-padded to the longest
            features = np.zeros((len(clips), clips[0].input_features.shape[0], max(c.frames for c in clips)),
                                dtype=np.float32)
            for j, c in enumerate(clips):
                features[j, :, :c.frames] = c.input_features
        record = {"first_question": k, "ids": list(ids), "state_len": int(state_len),
                  "question": None if question is None else list(question), "ends": [list(e) for e in ends],
                  "image_grid_thw": [[1, *pic.grid] for pic in pics] or None,
                  "pixel_values_shape": None if not pics else
                  [sum(pic.pixel_values.shape[0] for pic in pics), pics[0].pixel_values.shape[1]],
                  "audio": None if not clips else {"input_features_shape": list(features.shape),
                                                   "frames": [c.frames for c in clips]},
                  "clips": [{"frames": c.frames, "tokens": c.tokens, "samples": c.samples} for c in clips],
                  "has_position_map": n in maps, "position_map": maps.get(n),
                  "pixel_values": None if not pics else pics[0].pixel_values if len(pics) == 1
                  else np.concatenate([pic.pixel_values for pic in pics]),
                  "input_features": features}
        if with_request:
            record["request"] = i
        return record

    # ----------------------------------------------------------------- media (media.Reader: pictures and clips)

    @property
    def pictures(self) -> PictureProcessor:
        if self._pictures is None:
            self._pictures = PictureProcessor(self.snapshot_dir)
            with open(os.path.join(self.snapshot_dir, "chat_template.jinja")) as f:
                if f'"{IMAGE_MARKUP}"' not in f.read():
                    raise RuntimeError(f"chat_template.jinja no longer writes {IMAGE_MARKUP} for a picture")
            self.image_token_id = self.config["image_token_id"]
            if self.tokenizer.convert_tokens_to_ids(self.tokenizer.image_token) != self.image_token_id:
                raise RuntimeError("the tokenizer's image token is not config.json's image_token_id")
        return self._pictures

    @property
    def features(self) -> FeatureExtractor:
        """The clips' feature extractor and markup (Lfm3OmniProcessor's), set up on first use."""
        if self._features is None:
            if not self.takes_audio:
                raise ValueError("this checkpoint takes no audio")
            c = self.processor_config
            if c.get("processor_class") != "Lfm3OmniProcessor":
                raise ValueError(f"the host writes clips as Lfm3OmniProcessor does, not {c.get('processor_class')}")
            tok = self.tokenizer
            markup = AudioMarkup(c.get("audio_token", "<|audio|>"), c.get("vision_start_token", "<|media_start|>"),
                                 c.get("vision_end_token", "<|media_end|>"), bool(c.get("audio_markers", True)))
            for name, token in (("audio", markup.token), ("span start", markup.start), ("span end", markup.end)):
                if tok.convert_tokens_to_ids(token) is None:  # the processor would add or rename it: not ported
                    raise RuntimeError(f"the tokenizer has no {name} token {token!r}")
            with open(os.path.join(self.snapshot_dir, "chat_template.jinja")) as f:
                if f'"{markup.template}"' not in f.read():
                    raise RuntimeError(f"chat_template.jinja no longer writes {markup.template} for a clip")
            self.audio_token_id = self.config["audio_token_id"]
            if tok.convert_tokens_to_ids(markup.token) != self.audio_token_id:
                raise RuntimeError("the processor's audio token is not config.json's audio_token_id")
            self.audio_markup = markup
            self._features = FeatureExtractor(self.snapshot_dir)
        return self._features

    def encode_media(self, items: Sequence) -> tuple[str, list[Picture | Clip]]:
        """media.Reader.encode: the items' markup back to back, and each item's processor outputs."""
        encoded = [self._encode_item(m) for m in items]
        return "".join(text for text, _ in encoded), [out for _, out in encoded]

    def _markup_text(self, markup: str, kind: str) -> str:
        """An item's expanded markup as text that encodes back to the processor's ids (checked once an expansion, as
        Reader._encode keeps it)."""
        ids = tuple(self.tokenizer.encode(markup, add_special_tokens=False))
        if ids not in self._markups:
            text = self.tokenizer.decode(ids)
            if tuple(self.tokenizer.encode(text, add_special_tokens=False)) != ids:
                raise ValueError(f"{self.processor_config.get('processor_class')}'s {kind} markup does not read back "
                                 f"as text")
            if len(self._markups) >= 4096:
                self._markups.clear()
            self._markups[ids] = text
        return self._markups[ids]

    def _encode_item(self, item) -> tuple[str, Picture | Clip]:
        if item.type == "audio":
            return self._encode_clip(item)
        if item.type != "image":
            raise NotImplementedError(f"the host reads pictures and audio clips, not {item.type}")
        processor = self.pictures
        from PIL import Image

        image = Image.open(item.value) if isinstance(item.value, str) else item.value
        image = self.p.media.cap_pixels(image, self.max_pixels)
        pixel_values, grid = processor(image)
        count = grid[0] * grid[1] // processor.merge ** 2  # replace_image_token
        token = self.tokenizer.image_token
        c = self.processor_config
        if c.get("processor_class") == "Lfm3OmniProcessor" and not c.get("vision_markers", True):
            markup = token * count  # _strip_spans, then _wrap without the span tokens
        else:  # Lfm3VlProcessor expands the template's placeholder in place; the omni's _wrap puts the pair back
            markup = IMAGE_MARKUP.replace(token, token * count)
        return self._markup_text(markup, item.type), Picture(pixel_values, grid)

    def _encode_clip(self, item) -> tuple[str, Clip]:
        """Reader._decoded for a clip (media.read_audio, then media.resampled to the extractor's rate), its log-mel
        (Lfm3OmniFeatureExtractor) and markup (`replace_audio_token`: `<|audio|>` once per encoder token, between the
        span tokens)."""
        extractor = self.features
        t0 = time.perf_counter()
        rate = extractor.sampling_rate
        if isinstance(item.value, str):
            wave, given = read_audio(self.p.media, item.value, rate)
        else:
            wave, given = item.value, item.sampling_rate
        if given is None:
            raise ValueError("a waveform needs its sampling_rate")
        wave = resampled(wave, given, rate)
        features = extractor(wave)
        tokens = extractor.num_encoder_tokens(features.shape[1])
        if self._depth:  # inside an API call (`sequences()` alone leaves the last call's record as it was)
            self.last["timing"]["feature_s"] += time.perf_counter() - t0
        return self._markup_text(self.audio_markup(tokens), item.type), Clip(features, tokens, int(wave.shape[0]))

    # ----------------------------------------------------------------- sequences (runner.SlotReader)

    def _prefix(self, state: Any, media: Sequence) -> tuple[str, list[Picture]]:
        markup, parts = self.encode_media(media) if media else ("", [])
        return self.p.prompt.prefix_text(self.tokenizer, state, self.bos, self.state_style, self.system, markup), parts

    def _suffixes(self, qs: Sequence) -> tuple[list[str], list[list[int]], list[list]]:
        """Each question's suffix, a spoken one's clip in place of `media.SPOKEN` (Reader.place), its slots' offsets
        and its clip's outputs."""
        media = self.p.media
        layouts = [self.p.prompt.suffix_layout(self.tokenizer, q, self.lead, self.option_style) for q in qs]
        placed = []
        for (text, _), q in zip(layouts, qs):
            voice = media.spoken(q)
            if voice is None:
                placed.append((text, []))
            else:
                markup, parts = self.encode_media([voice])
                placed.append((text.replace(media.SPOKEN, markup), parts))
        return [text for text, _ in placed], [ends for _, ends in layouts], [parts for _, parts in placed]

    def _sequences(self, requests: Sequence[tuple], *, position_maps=None, set_size: int | None = None) -> list[tuple]:
        """runner.SlotReader._sequences under `set`: `(request, first question, ids, state length, each token's
        question, answer slots, pictures)`."""
        prompt, media_list = self.p.prompt, self.p.media.media_list
        seqs = []
        for i, (state, qs, media) in enumerate(requests):
            media = media_list(media)
            if any(isinstance(q, (prompt.Box, prompt.Boxes)) for q in qs) and (
                    n := sum(m.type == "image" for m in media)) != 1:
                raise ValueError(f"a box or boxes question reads exactly one picture in `media`; this request has {n}")
            prefix, state_parts = self._prefix(state, media)
            suffixes, offsets, voices = self._suffixes(qs)
            step = self.set_size if set_size is None else set_size
            for k in range(0, len(qs), step):
                text = prefix + suffixes[k]
                encoded = None
                if position_maps is not None and any(isinstance(q, prompt.Read) for q in qs[k:k + step]):
                    encoded = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
                ids = self.tokenizer.encode(text, add_special_tokens=False) if encoded is None else encoded["input_ids"]
                ids, state_len, question, ends = pack_set(self.tokenizer, prefix, suffixes[k:k + step],
                                                          prompt.question_header(self.tokenizer, state),
                                                          offsets[k:k + step], first_ids=ids)
                parts = state_parts + [part for own in voices[k:k + step] for part in own]
                if encoded is not None:
                    position_maps[len(seqs)] = state_positions(self.p, self.tokenizer, state, prefix, text,
                                                               self.state_style, encoded=encoded)
                seqs.append((i, k, ids, state_len, question, ends, parts))
        return seqs

    # ----------------------------------------------------------------- the forward and the read (SlotReader._padded)

    def _batches(self, seqs: list[tuple]) -> list[list[int]]:
        """SystemOne._batches: the sequences in the provider's batches (a request's sequences together unless they
        pass the token budget, shortest first). The host runs the backends one sequence at a time but reads the slots
        a batch at once, as the provider's one forward did (see `_readouts`)."""
        own: dict[int, list[int]] = {}
        for j in sorted(range(len(seqs)), key=lambda j: len(seqs[j][2])):
            own.setdefault(seqs[j][0], []).append(j)
        batches: list[list[int]] = [[]]
        for rows in sorted(own.values(), key=lambda rows: len(seqs[rows[-1]][2])):
            if batches[-1] and len(seqs[rows[-1]][2]) * (len(batches[-1]) + len(rows)) > self.token_budget:
                batches.append([])
            for j in rows:
                if batches[-1] and len(seqs[j][2]) * (len(batches[-1]) + 1) > self.token_budget:
                    batches.append([])
                batches[-1].append(j)
        return list(filter(None, batches))

    def _padded(self, requests: Sequence[tuple], set_size: int | None = None) -> list[tuple[list, int]]:
        """runner.SlotReader._padded: each sequence through the backends (one call each, never padded), each of the
        provider's batches read at its slots: option codes through the readout, reads through the pointer."""
        prompt = self.p.prompt
        t0 = time.perf_counter()
        position_maps: dict = {}
        seqs = self._sequences(requests, position_maps=position_maps, set_size=set_size)
        self.last["seqs"] = [self._seq_record(n, seq, position_maps) for n, seq in enumerate(seqs)]
        self.last["slot_hidden"] = [None] * len(seqs)
        self.last["seq_audio"] = [[] for _ in seqs]
        self.last["read_logits"] = []
        read = [0] * len(requests)
        for i, _, ids, *_ in seqs:
            read[i] += len(ids)
        limit = getattr(self.decide, "max_len", None)
        for i, k, ids, *_ in seqs:
            if limit is not None and len(ids) > limit:
                raise ValueError(f"request {i}: a sequence of {len(ids)} tokens (questions {k + 1}..) is longer than "
                                 f"the largest the decision backend takes ({limit}); the host never truncates: shorten "
                                 f"the state, or read fewer questions together (set_size)")
        frames_limit = getattr(self.audio, "max_frames", None)
        for i, k, *_, parts in seqs:
            clips = [p for p in parts if isinstance(p, Clip)]
            if clips and self.audio is None:
                raise ValueError("a request with audio clips needs an audio backend (D1Host(..., audio=...))")
            if any(isinstance(p, Picture) for p in parts) and self.tower is None:
                raise ValueError("a request with pictures needs a tower backend (D1Host(..., tower=...))")
            for c in clips:
                if frames_limit is not None and c.frames > frames_limit:
                    hop, rate = self.features.settings.hop_length, self.features.sampling_rate
                    raise ValueError(f"request {i}: a clip of {c.frames} frames ({c.samples / rate:.2f} s, questions "
                                     f"{k + 1}..) is longer than the largest the audio backend takes ({frames_limit} "
                                     f"frames = {((frames_limit - 1) * hop + self.features.settings.n_fft) / rate:.2f}"
                                     f" s); the host never truncates: shorten the clip")

        def asked(key: tuple[int, int]):
            return requests[key[0]][1][key[1]]

        probs: dict = {}
        features: dict[int, np.ndarray] = {}
        t0 = self._tick("prep_s", t0)
        for batch in self._batches(seqs):
            slots = []  # (row in the batch, (request, question), the slot's readout, its position), as _padded has them
            at_rows, hiddens = [], []
            for r, b in enumerate(batch):
                i, first, ids, state_len, question, ends, parts = seqs[b]
                own_slots = []
                for k, own in enumerate(ends):
                    key = (i, first + k)
                    own_slots.extend((r, key, q, end)
                                     for q, end in zip(prompt.question_slots(asked(key), self.tokenizer), own))
                hidden = self._hidden(ids, state_len, question, parts, features)
                self.last["seq_audio"][b] = [features[id(c)] for c in parts if isinstance(c, Clip)]
                at = hidden[[s[3] for s in own_slots]]
                self.last["slot_hidden"][b] = at
                slots += own_slots
                at_rows.append(at)
                hiddens.append(hidden if b in position_maps else None)  # the pointer reads a row's state positions
            t0 = time.perf_counter()
            at = at_rows[0] if len(at_rows) == 1 else np.concatenate(at_rows)
            reads = [j for j, s in enumerate(slots) if isinstance(s[2], prompt.Position)]
            if reads:
                maps = {r: position_maps[b] for r, b in enumerate(batch) if b in position_maps}  # rows with reads
                positions = self._read_logits(slots, reads, at, hiddens, maps)
                self.last["read_logits"].append(positions)
                logits = positions.reshape(-1, 2, positions.shape[-1])
                capped: dict[int, list] = {}  # the reads of each span cap
                for index, j in enumerate(reads[::2]):
                    capped.setdefault(self.span_caps[asked(slots[j][1]).type], []).append((index, j))
                for cap, own in capped.items():
                    found_at, found = spans(logits[[index for index, _ in own]], max_tokens=cap)
                    for (_, j), span, p in zip(own, found_at.tolist(), found.tolist()):
                        r, key = slots[j][:2]
                        probs[key] = {**maps[r].read(*span, max_tokens=cap), "found": p}
            reading = set(reads)
            codes = [j for j in range(len(slots)) if j not in reading]
            scored = self._readouts([slots[j][2] for j in codes], at[codes],
                                    [not prompt.typed(asked(slots[j][1])) for j in codes])
            for j, p in zip(codes, scored):
                key = slots[j][1]
                q = asked(key)
                if isinstance(q, prompt.Choice) and q.multiple:
                    probs.setdefault(key, []).append(p[0])
                elif isinstance(q, (prompt.Box, prompt.Boxes)):
                    probs.setdefault(key, []).append(p)
                else:
                    probs[key] = p
            self._tick("readout_s", t0)
        out = [([probs[(i, k)] for k in range(len(qs))], read[i]) for i, (_, qs, _) in enumerate(requests)]
        self.last["probs"] = out
        return out

    def _hidden(self, ids: list[int], state_len: int, question: list[int], parts: list[Picture | Clip],
                features: dict[int, np.ndarray]) -> np.ndarray:
        """The sequence's embeddings (table rows; picture tokens at the `<|image|>` positions, the clips' tokens at the
        `<|audio|>` positions, each in the sequence's media order) through `decide`."""
        timing = self.last["timing"]
        t0 = time.perf_counter()
        embeds = self.embed.rows(ids)
        pics = [p for p in parts if isinstance(p, Picture)]
        clips = [p for p in parts if isinstance(p, Clip)]
        if pics:
            merge = self.pictures.merge
            tokens = []
            for pic in pics:
                if id(pic) not in features:  # once a picture: a request's sequences share its pictures
                    if self.tower is None:
                        raise ValueError("a request with pictures needs a tower backend (D1Host(..., tower=...))")
                    t0 = self._tick("prep_s", t0)
                    out = np.asarray(self.tower(pic.pixel_values, pic.grid), dtype=np.float32)
                    t0 = self._tick("tower_s", t0)
                    timing["tower_calls"] += 1
                    want = (pic.grid[0] * pic.grid[1] // merge ** 2, self.embed.hidden)
                    if out.shape != want:
                        raise ValueError(f"the tower returned {out.shape} for a {pic.grid} grid, expected {want}")
                    features[id(pic)] = out
                tokens.append(features[id(pic)])
            tokens = np.concatenate(tokens)
            mask = np.asarray(ids) == self.image_token_id
            if int(mask.sum()) != tokens.shape[0]:
                raise ValueError(f"Image features and image tokens do not match, tokens: {int(mask.sum())}, "
                                 f"features: {tokens.shape[0]}")
            embeds[mask] = tokens
        if clips:
            tokens = []
            for clip in clips:
                if id(clip) not in features:  # once a clip: a request's sequences share the state's clips
                    if self.audio is None:
                        raise ValueError("a request with audio clips needs an audio backend (D1Host(..., audio=...))")
                    t0 = self._tick("prep_s", t0)
                    out = np.asarray(self.audio(clip.input_features), dtype=np.float32)
                    t0 = self._tick("audio_s", t0)
                    timing["audio_calls"] += 1
                    want = (clip.tokens, self.embed.hidden)
                    if out.shape != want:
                        raise ValueError(f"the audio backend returned {out.shape} for a clip of {clip.frames} frames, "
                                         f"expected {want}")
                    features[id(clip)] = out
                    self.last["audio_features"].append(out)
                tokens.append(features[id(clip)])
            tokens = np.concatenate(tokens)
            mask = np.asarray(ids) == self.audio_token_id
            if int(mask.sum()) != tokens.shape[0]:
                raise ValueError(f"Placeholder tokens and features do not match: {int(mask.sum())} tokens "
                                 f"{self.audio_token_id}, {tokens.shape[0]} features")
            embeds[mask] = tokens
        if self.decide is None:
            raise ValueError("no decision backend: D1Host(snapshot, decide=...)")
        t0 = self._tick("prep_s", t0)
        hidden = np.asarray(self.decide(embeds, int(state_len), list(question)), dtype=np.float32)
        self._tick("decide_s", t0)
        timing["decide_calls"] += 1
        timing["tokens"] += len(ids)
        if hidden.shape != embeds.shape:
            raise ValueError(f"the decision backend returned {hidden.shape}, expected {embeds.shape}")
        return hidden

    @staticmethod
    def _read_logits(slots: list[tuple], reads: list[int], at: np.ndarray, hiddens: list[np.ndarray],
                     maps: Mapping[int, StatePositions]) -> np.ndarray:
        """pointer.query_plan + read_logits for a batch, in the provider's shapes: the rows that hold reads, each row's
        reads as its queries (padded to the largest group) against the row's hidden states up to the widest mask
        ([rows, groups, d] x [rows, width, d], one product a row as torch.bmm makes them; float32, / sqrt(d)), -inf
        where the row's mask is off; [reads, width] in the order of `reads`. The padding never reaches a kept value:
        each element of a product is its own row's and column's sum, and the shapes are the provider's."""
        rows = sorted({slots[j][0] for j in reads})
        index = {r: n for n, r in enumerate(rows)}
        width = max(len(maps[r].pointer_mask) for r in rows)
        groups: list[list[int]] = [[] for _ in rows]
        where = []
        for n, j in enumerate(reads):
            group = groups[index[slots[j][0]]]
            where.append((index[slots[j][0]], len(group)))
            group.append(reads[n])
        d = at.shape[-1]
        queries = np.zeros((len(rows), max(map(len, groups)), d), dtype=np.float32)
        states = np.zeros((len(rows), width, d), dtype=np.float32)
        valid = np.zeros((len(rows), width), dtype=bool)
        for i, (r, group) in enumerate(zip(rows, groups)):
            queries[i, :len(group)] = at[group]
            own = hiddens[r][:width]
            states[i, :len(own)] = own
            mask = np.asarray(maps[r].pointer_mask, dtype=bool)
            valid[i, :len(mask)] = mask
        logits = np.matmul(queries, states.transpose(0, 2, 1)) / np.float32(math.sqrt(d))
        logits = np.where(valid[:, None, :], logits, np.float32(-np.inf))
        return np.stack([logits[i, c] for i, c in where])

    def _readouts(self, qs: Sequence, rows: np.ndarray, tempered: Sequence[bool]) -> list[list[float]]:
        """prompt.readouts on a batch's code slots' hidden states `rows` [n, d]: each shared readout table's logits
        (float32), its options' best forms, float64, / temperature, softmax.

        Every table's logits are taken for all n rows at once and its members' rows kept, against the table's rows
        padded with zero rows to a multiple of 16: the provider's LM head reads the n code slots as one [n, d] input
        against all 128,000 rows, and a float32 product's sums depend on its shape (on Accelerate, measured: one row
        takes a matrix-vector path that sums in another order than two or more; and a product with few columns, under
        16 for one row, a handful for two to five, takes another path again; with 16 or more columns every element is
        the full product's, whatever the columns), so this is the provider's logit for the provider's row count."""
        prompt = self.p.prompt
        tables = [prompt.readout_table(self.tokenizer, q) for q in qs]
        shared: dict[int, list[int]] = {}
        for j, table in enumerate(tables):
            shared.setdefault(id(table), []).append(j)
        out: list = [None] * len(qs)
        for members in shared.values():
            tokens, forms, _ = tables[members[0]]
            values = (rows @ self._table_rows(tokens).T)[members, :len(tokens)]  # float32 [m, tokens]
            scores = values[:, forms].max(-1).astype(np.float64)
            scores /= np.array([1.0 if not tempered[j] or self.calibration is None
                                else self.calibration.temperature(qs[j]) for j in members])[:, None]
            exps = np.exp(scores - scores.max(-1, keepdims=True))
            for j, p in zip(members, (exps / exps.sum(-1, keepdims=True)).tolist()):
                out[j] = p
        return out

    def _table_rows(self, tokens: tuple[int, ...]) -> np.ndarray:
        """A readout table's float32 rows [len(tokens), d], then zero rows up to a multiple of 16 (see `_readouts`)."""
        rows = self._readout_rows.get(tokens)
        if rows is None:
            if len(self._readout_rows) >= 256:
                self._readout_rows.clear()
            rows = np.zeros((-(-len(tokens) // 16) * 16, self.embed.hidden), dtype=np.float32)
            rows[:len(tokens)] = self.embed.rows(tokens)
            self._readout_rows[tokens] = rows
        return rows


def snapshot_default() -> str:
    """The repository's provider/ folder (the files of LiquidAI/d1-Omni-2B at revision 431f1d66 that the host
    reads)."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "provider"))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="write the tied embedding table's bf16 bits as .npy (uint16)")
    ap.add_argument("--snapshot", default=snapshot_default())
    ap.add_argument("--table-out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.table_out)), exist_ok=True)
    print(json.dumps(write_embed_table(a.snapshot, a.table_out), indent=1))
