"""d1-3B on LiteRT: a reference host in plain Python: the render, the tokenizer, the graph calls through the
CompiledModel API, the read-out and the answers, for requests with text and with pictures.

Dependencies: `tokenizers` and `numpy` (and `safetensors` for the read-out table, `ai-edge-litert` for the graphs). No
PyTorch, no transformers and none of the provider's code at run time.

The host answers a System One request the way the provider's code in LiquidAI/d1-3B (revision da1fe36a) does with its
defaults (`model.system_one(state, questions)`: `prompt.render`, `prompt.readout_ids`, `prompt.readout`, `api.answer`),
around one kind of LiteRT graph: a row graph per length bucket, one causal row per question (the state followed by that
question, from position 0).

1. Request: `{"state": str | JSON | None, "questions": {name: {"type": "noul" | "choice" | "score",
   "instructions": ..., "criteria": ...}}}`. `type` defaults to "choice". `instructions` is required (the provider's
   parser raises KeyError without it; so does this host). Choice criteria are an object {name: description or null},
   noul criteria an optional object with "true" / "false" descriptions, score criteria a list of levels, low to high
   (2 to 10 levels: the read-out needs a single-token digit per level). Pictures: `images`, a list of pictures (PIL
   images, paths or the bytes of image files); see 9.
2. Render (`render`), with the defaults `D1Model.engine` uses (no system turn, state style `json_only`, option style
   `desc`, no lead): "<|startoftext|><|im_start|>user\n" + state block + "\nQUESTION:\n" + question block +
   "<|im_end|>\n<|im_start|>assistant\n". The state block is a string state as it is, any other JSON value as
   `json.dumps(value, ensure_ascii=False, indent=2)`, then "\n\n"; a null state writes neither the block nor the
   "QUESTION:" line. Question blocks:
   - choice: instructions + "\n\nOptions:\n" + one line "<code> <text>" per option (text = the description, or the
     name with "_" as spaces when the description is empty) + "\n\nReply with the option code only.";
   - noul: instructions (+ "\nYes: <true>\nNo: <false>" when criteria are given) + "\n\nReply with yes or no only.";
   - score: instructions + "\n\n" + one line "<i> <level>" per level + "\n\nReply with a single digit 0-<K-1> only.".
3. Option codes (`option_codes`, `aliases`): the names themselves when every name is one letter, else A, B, C, ... up to
   26 options, else 00, 01, ...; a code that is not one token, or whose token another option already took, is replaced
   by the next free single-token entry of the pool A..Z, 00..99, a..z, #0..#199, AA..ZZ.
4. Tokenize with the repository's `tokenizer.json` (the `tokenizers` library), no special tokens added: the BOS is in
   the text. On every row of the test requests this gives the provider's ids (transformers 5.14.1 AutoTokenizer,
   class TokenizersBackend).
5. Read-out groups (`readout_ids`): a choice option scores [id(code)] plus id(" " + code) when that is one token and a
   different one; noul yes = the single-token forms of yes / Yes / YES, no = those of no / No / NO (with this
   tokenizer [11683, 12447] and [2243, 4547, 19598]: "YES" is two tokens); score level i = [id(str(i))].
6. Run. A row graph maps `ids` int32 [1,L] and `valid` float32 [1,L] (1.0 on real tokens, 0.0 on padding) to
   `hidden` float32 [1,L,d], the hidden states after the final RMSNorm (`embedding_norm`); d is read from the graph's
   output shape (2048 on d1-3B). A row is right-padded with the contract's pad id (`<|pad|>` = 124893) to the smallest
   bucket that holds it (`pick_L` over the graphs present); a longer row is refused (`RowTooLong`), never truncated.
   The answer slot is the row's last real token. `LiteRTRowGraph` runs one `.tflite` through
   `ai_edge_litert.compiled_model` (signature `serving_default`, CPU with N threads or the GPU with float32 or default
   precision).
7. Read out on the host (`readout`): for every id of the question's groups, logit = h · E[id] in float32, with h the
   hidden state at the answer slot and E the tied embedding table (the checkpoint's bfloat16 rows as float32, shipped as
   a small table of the rows the groups can use); an option scores its group's maximum; probs = softmax over the
   options, in float64. This equals the provider's `readout(log_softmax(lm_head(h)))`: the log-softmax subtracts one
   number from every logit, which neither the maximum nor the softmax sees, so the other 127,9xx logits are not needed.
8. Answer like `api.answer`: noul {"type": "noul", "noul": P(yes)}; choice {"type": "choice", "choice": the most likely
   name, "confidence", "probabilities"}; score {"type": "score", "score": the expected level, "confidence",
   "probabilities", "legend"}. The response is {"answers": {name: answer}, "usage": {"input_tokens", "output_tokens": 0}};
   input_tokens counts what the provider's runner reads: the row for a request of one question, the state's tokens
   once plus every question's own tokens for a request of several (with pictures, the state's tokens include the
   pictures' ids).
9. Pictures (the steps: host/d1_vision.py). A request with `images` renders one
   `<image>` per picture at the head of the user turn (`prefix_text(state, image_markup(n))`); a row's ids are the
   processor's expansion of that text (`d1_vision.row_ids`); the pictures' tokens come from the tower and projector
   graphs once per request (`d1_vision.VisionPath`). The row runs on the embeds variant of the row graph
   (`LiteRTEmbedsGraph`: `embeds` float32 [1,L,d] + `valid` -> `hidden`): the text positions take the float32 rows of
   the bfloat16 table (`EmbedTable`, embed_table.safetensors), the `<image>` positions the picture tokens in order
   (`d1_vision.insert_embeddings`), the padding the pad id's row; the bucket is the smallest embeds graph that holds
   the row (`pick_L`). The answer slot, the read-out and the answer are the text ones. Several questions with pictures
   are one row each, the pictures' tokens computed once.
10. The text path: text rows run on the embeds row graphs too (the float32 rows of the bfloat16 table, the pad id's
   row on the pads), whenever embeds graphs and the table are loaded (`prefer_embeds`, default True); graphs that
   take `ids` (an earlier form with an int8 table inside the graph, not in this repository) run a text row only
   when no embeds graph is loaded or prefer_embeds is False. `from_dir` loads the embeds buckets of the contract
   (`embeds_graph.buckets` + its table) unless ids_graphs=True (the ids graphs of `graphs`). Buckets: L128 .. L4096
   (`BUCKETS`; a graph computes all L positions, so the smallest that holds the row).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

BOS = "<|startoftext|>"
IM_START, IM_END = "<|im_start|>", "<|im_end|>"
PAD_ID = 124893
HIDDEN = 2048             # d1-3B's d; the host reads d from each graph's output shape
BUCKETS = (128, 256, 512, 1024, 2048, 4096)     # 311 of the 415 text rows of the test requests are <= 128 tokens
SIGNATURE = "serving_default"
INPUT_NAMES = ("ids", "valid")
EMBEDS_INPUT_NAMES = ("embeds", "valid")
OUTPUT_NAME = "hidden"
EMBED_KEY = "embed_tokens.weight"
CONTRACT_FILE = "contract.json"
YES_FORMS = ("yes", "Yes", "YES")
NO_FORMS = ("no", "No", "NO")
FALLBACK_POOL = tuple(
    [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    + [f"{i:02d}" for i in range(100)]
    + [chr(c) for c in range(ord("a"), ord("z") + 1)]
    + [f"#{i}" for i in range(200)]
    + [chr(a) + chr(b) for a in range(ord("A"), ord("Z") + 1) for b in range(ord("A"), ord("Z") + 1)]
)
TOKENIZER_FILE = "tokenizer.json"


class RequestError(ValueError):
    """The request cannot be answered as given."""


class RowTooLong(RequestError):
    """A question's row is longer than the largest graph."""


# --------------------------------------------------------------------------- #
# request parsing
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Question:
    type: str                 # "choice" | "noul" | "score"
    instructions: Any
    criteria: Any             # dict for choice, dict or None for noul, list for score


def as_question(q: Mapping) -> Question:
    """The provider's `prompt.as_question` on a JSON question (KeyError without instructions, as there)."""
    kind = q.get("type", "choice")
    if kind == "noul":
        return Question("noul", q["instructions"], q.get("criteria"))
    if kind == "score":
        return Question("score", q["instructions"], list(q["criteria"]))
    return Question("choice", q["instructions"], q["criteria"])


# --------------------------------------------------------------------------- #
# tokenizer
# --------------------------------------------------------------------------- #


class D1Tokenizer:
    """`tokenizers.Tokenizer` on the repository's tokenizer.json; encode never adds special tokens."""

    def __init__(self, path: str | Path):
        from tokenizers import Tokenizer

        self._tok = Tokenizer.from_file(str(path))
        self._single: Dict[str, Optional[int]] = {}

    def encode(self, text: str) -> List[int]:
        return self._tok.encode(text, add_special_tokens=False).ids

    def single(self, text: str) -> Optional[int]:
        """The id when `text` is one token, else None (memoised)."""
        if text not in self._single:
            ids = self.encode(text)
            self._single[text] = ids[0] if len(ids) == 1 else None
        return self._single[text]


# --------------------------------------------------------------------------- #
# verbalizer (prompt.option_codes / aliases / readout_ids)
# --------------------------------------------------------------------------- #


def option_codes(labels: Sequence[str]) -> List[str]:
    labs = [str(x).strip() for x in labels]
    if labs and all(len(k) == 1 and k.isalpha() for k in labs):
        return labs
    if len(labs) <= 26:
        return [chr(ord("A") + i) for i in range(len(labs))]
    return [f"{i:02d}" for i in range(len(labs))]


def aliases(tok: D1Tokenizer, labels: Sequence[str]) -> List[Tuple[str, int]]:
    """Every label's distinct single-token code: [(code, token id)]."""
    used: set = set()
    out: List[Tuple[str, int]] = []

    def take(raw: str) -> bool:
        tid = tok.single(raw)
        if tid is None or tid in used:
            return False
        out.append((raw, tid))
        used.add(tid)
        return True

    codes = option_codes(labels)
    for code in codes:
        if take(code):
            continue
        if not any(take(raw) for raw in FALLBACK_POOL):
            raise RequestError(f"no single-token alias left for {len(codes)} options")
    return out


def _single_ids(tok: D1Tokenizer, texts: Sequence[str]) -> List[int]:
    out: List[int] = []
    for t in texts:
        tid = tok.single(t)
        if tid is not None and tid not in out:
            out.append(tid)
    return out


def readout_ids(tok: D1Tokenizer, q: Question) -> List[List[int]]:
    if q.type == "noul":
        yes, no = _single_ids(tok, YES_FORMS), _single_ids(tok, NO_FORMS)
        if not yes or not no:
            raise RequestError("tokenizer has no single-token yes/no")
        return [yes, no]
    if q.type == "score":
        groups = [_single_ids(tok, [str(i)]) for i in range(len(q.criteria))]
        if any(not g for g in groups):
            raise RequestError(f"score with {len(q.criteria)} levels needs single-token digits (2 to 10 levels)")
        return groups
    groups = []
    for code, tid in aliases(tok, list(q.criteria.keys())):
        extra = _single_ids(tok, [f" {code}"])
        groups.append([tid] + [i for i in extra if i != tid])
    if not groups:
        raise RequestError("choice with no options")
    return groups


def option_keys(q: Question) -> List[str]:
    """The fixtures' gold keys in read-out order (a noul reads out [yes, no] = keys ["true", "false"])."""
    if q.type == "choice":
        return list(q.criteria.keys())
    if q.type == "noul":
        return ["true", "false"]
    return [str(i) for i in range(len(q.criteria))]


# --------------------------------------------------------------------------- #
# render (prompt.state_block / question_block / prefix_text / suffix_text)
# --------------------------------------------------------------------------- #


def state_block(state: Any) -> str:
    if isinstance(state, str):
        return f"{state}\n\n"
    return json.dumps(state, ensure_ascii=False, indent=2) + "\n\n"


def question_block(tok: D1Tokenizer, q: Question) -> str:
    if q.type == "choice":
        labels = list(q.criteria.keys())
        codes = aliases(tok, labels)
        lines = "\n".join(f"{codes[i][0]} {q.criteria[lab] or lab.replace('_', ' ')}" for i, lab in enumerate(labels))
        return f"{q.instructions}\n\nOptions:\n{lines}\n\nReply with the option code only."
    if q.type == "noul":
        extra = ""
        if q.criteria:
            extra = f"\nYes: {q.criteria.get('true')}\nNo: {q.criteria.get('false')}"
        return f"{q.instructions}{extra}\n\nReply with yes or no only."
    legend = "\n".join(f"{i} {name}" for i, name in enumerate(q.criteria))
    return f"{q.instructions}\n\n{legend}\n\nReply with a single digit 0-{len(q.criteria) - 1} only."


def prefix_text(state: Any, images: str = "") -> str:
    body = "" if state is None else f"{state_block(state)}\nQUESTION:\n"
    return f"{BOS}{IM_START}user\n{images}{body}"


def suffix_text(tok: D1Tokenizer, q: Question) -> str:
    return f"{question_block(tok, q)}{IM_END}\n{IM_START}assistant\n"


def render(tok: D1Tokenizer, state: Any, q: Question) -> str:
    return prefix_text(state) + suffix_text(tok, q)


# --------------------------------------------------------------------------- #
# rows, graphs, read-out
# --------------------------------------------------------------------------- #


@dataclass
class Row:
    name: str
    question: Question
    text: str
    ids: List[int]
    groups: List[List[int]]
    state_len: int
    pictures: Optional[list] = None     # d1_vision.Picture per picture of the request (item 9), None for text

    @property
    def answer_slot(self) -> int:
        return len(self.ids) - 1


def pick_L(n: int, buckets: Sequence[int] = BUCKETS) -> int:
    for L in sorted(buckets):
        if n <= L:
            return L
    raise RowTooLong(f"a row of {n} tokens is longer than the largest graph ({max(buckets)})")


def pad_row(ids: Sequence[int], L: int, pad_id: int = PAD_ID) -> Tuple[np.ndarray, np.ndarray]:
    """`ids` int32 [1,L] right-padded with `pad_id`, `valid` float32 [1,L]."""
    n = len(ids)
    if n > L:
        raise RowTooLong(f"{n} > {L}")
    out = np.full((1, L), pad_id, dtype=np.int32)
    out[0, :n] = ids
    valid = np.zeros((1, L), dtype=np.float32)
    valid[0, :n] = 1.0
    return out, valid


RowGraph = Callable[[np.ndarray, np.ndarray], np.ndarray]   # (ids [1,L] int32, valid [1,L] f32) -> hidden [1,L,d] f32
EmbedsGraph = Callable[[np.ndarray, np.ndarray], np.ndarray]  # (embeds [1,L,d] f32, valid [1,L] f32) -> hidden [1,L,d]


def compiled_options(accelerator: str, precision: str, threads: int):
    """CompiledModel options: "cpu" (XNNPACK, `threads`) or "gpu" (precision "fp32" = GpuOptions(enforce_f32=True),
    "default" = the delegate's default precision)."""
    from ai_edge_litert.compiled_model import CpuOptions, GpuOptions, HardwareAccelerator, Options

    if accelerator == "gpu":
        if precision not in ("fp32", "default"):
            raise ValueError("precision must be 'fp32' or 'default'")
        return Options(hardware_accelerators=HardwareAccelerator.GPU, gpu_options=GpuOptions(enforce_f32=precision == "fp32"))
    if accelerator == "cpu":
        return Options(hardware_accelerators=HardwareAccelerator.CPU, cpu_options=CpuOptions(num_threads=threads))
    raise ValueError("accelerator must be 'cpu' or 'gpu'")


class LiteRTRowGraph:
    """One row graph on the CompiledModel API: ids int32 [1,L] + valid float32 [1,L] -> hidden float32 [1,L,d].

    accelerator "cpu" (XNNPACK, `threads`) or "gpu" (precision "fp32" = GpuOptions(enforce_f32=True), "default" = the
    delegate's default precision). L and d come from the graph's signature."""

    def __init__(self, path: str | Path, accelerator: str = "cpu", precision: str = "fp32", threads: int = 4):
        self.path, self.accelerator, self.precision = Path(path), accelerator, precision
        if not self.path.is_file():
            raise FileNotFoundError(f"no row graph at {self.path}")
        from ai_edge_litert.compiled_model import CompiledModel

        options = compiled_options(accelerator, precision, threads)
        self.model = CompiledModel.from_file(str(self.path), options=options)
        sig = self.model.get_signature_list().get(SIGNATURE, {})
        if sorted(sig.get("inputs", [])) != sorted(INPUT_NAMES) or list(sig.get("outputs", [])) != [OUTPUT_NAME]:
            self.model.close()
            raise ValueError(f"unexpected signature in {self.path.name}: {sig}")
        ids = self.model.get_input_tensor_details(SIGNATURE)["ids"]
        hidden = self.model.get_output_tensor_details(SIGNATURE)[OUTPUT_NAME]
        self.L = int(list(ids["shape"])[1])
        self.hidden_size = int(list(hidden["shape"])[2])
        assert list(hidden["shape"]) == [1, self.L, self.hidden_size], hidden["shape"]
        self.inputs = {n: self.model.create_input_buffer_by_name(SIGNATURE, n) for n in INPUT_NAMES}
        self.outputs = {OUTPUT_NAME: self.model.create_output_buffer_by_name(SIGNATURE, OUTPUT_NAME)}
        try:
            self.fully_accelerated = bool(self.model.is_fully_accelerated())
        except Exception:   # informational only
            self.fully_accelerated = None

    def __call__(self, ids: np.ndarray, valid: np.ndarray) -> np.ndarray:
        if ids.shape != (1, self.L) or valid.shape != (1, self.L):
            raise ValueError(f"{self.path.name} takes [1,{self.L}] inputs, got {ids.shape} / {valid.shape}")
        self.inputs["ids"].write(np.ascontiguousarray(ids, dtype=np.int32))
        self.inputs["valid"].write(np.ascontiguousarray(valid, dtype=np.float32))
        self.model.run_by_name(SIGNATURE, self.inputs, self.outputs)
        size = self.L * self.hidden_size
        out = self.outputs[OUTPUT_NAME].read(size, np.float32)
        return np.asarray(out, dtype=np.float32).reshape(1, self.L, self.hidden_size)

    def close(self) -> None:
        for buffer in list(self.inputs.values()) + list(self.outputs.values()):
            try:
                buffer.destroy()
            except Exception:
                pass
        self.inputs, self.outputs = {}, {}
        self.model.close()


class LiteRTEmbedsGraph:
    """The embeds variant of a row graph (item 9; no table in the graph): embeds float32 [1,L,d] + valid float32 [1,L] ->
    hidden float32 [1,L,d], on the CompiledModel API with the options of `LiteRTRowGraph`."""

    def __init__(self, path: str | Path, accelerator: str = "cpu", precision: str = "fp32", threads: int = 4):
        self.path, self.accelerator, self.precision = Path(path), accelerator, precision
        if not self.path.is_file():
            raise FileNotFoundError(f"no row graph at {self.path}")
        from ai_edge_litert.compiled_model import CompiledModel

        self.model = CompiledModel.from_file(str(self.path), options=compiled_options(accelerator, precision, threads))
        sig = self.model.get_signature_list().get(SIGNATURE, {})
        if sorted(sig.get("inputs", [])) != sorted(EMBEDS_INPUT_NAMES) or list(sig.get("outputs", [])) != [OUTPUT_NAME]:
            self.model.close()
            raise ValueError(f"unexpected signature in {self.path.name}: {sig}")
        embeds = list(self.model.get_input_tensor_details(SIGNATURE)["embeds"]["shape"])
        hidden = list(self.model.get_output_tensor_details(SIGNATURE)[OUTPUT_NAME]["shape"])
        self.L, self.hidden_size = int(embeds[1]), int(embeds[2])
        assert hidden == [1, self.L, self.hidden_size], hidden
        self.inputs = {n: self.model.create_input_buffer_by_name(SIGNATURE, n) for n in EMBEDS_INPUT_NAMES}
        self.outputs = {OUTPUT_NAME: self.model.create_output_buffer_by_name(SIGNATURE, OUTPUT_NAME)}
        try:
            self.fully_accelerated = bool(self.model.is_fully_accelerated())
        except Exception:   # informational only
            self.fully_accelerated = None

    def __call__(self, embeds: np.ndarray, valid: np.ndarray) -> np.ndarray:
        if embeds.shape != (1, self.L, self.hidden_size) or valid.shape != (1, self.L):
            raise ValueError(f"{self.path.name} takes [1,{self.L},{self.hidden_size}] / [1,{self.L}], got "
                             f"{embeds.shape} / {valid.shape}")
        self.inputs["embeds"].write(np.ascontiguousarray(embeds, dtype=np.float32))
        self.inputs["valid"].write(np.ascontiguousarray(valid, dtype=np.float32))
        self.model.run_by_name(SIGNATURE, self.inputs, self.outputs)
        out = self.outputs[OUTPUT_NAME].read(self.L * self.hidden_size, np.float32)
        return np.asarray(out, dtype=np.float32).reshape(1, self.L, self.hidden_size)

    close = LiteRTRowGraph.close


class EmbedTable:
    """The whole tied table for the embeds graphs (item 9): embed_table.safetensors, one BF16 tensor [V, d]
    (`embed_tokens.weight`), memory-mapped; rows(ids) = those rows widened to float32 (the bfloat16 bits become the high
    half of each float32 = the checkpoint's table loaded in float32)."""

    def __init__(self, path: str | Path):
        import struct

        self.path = Path(path)
        with open(self.path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        e = header.get(EMBED_KEY)
        if e is None or e["dtype"] != "BF16" or len(e["shape"]) != 2:
            raise ValueError(f"{self.path.name}: no BF16 [V, d] tensor {EMBED_KEY}")
        self.vocab, self.hidden = (int(x) for x in e["shape"])
        begin, end = e["data_offsets"]
        if end - begin != self.vocab * self.hidden * 2:
            raise ValueError(f"{self.path.name}: {EMBED_KEY} holds {end - begin} bytes")
        self._raw = np.memmap(self.path, dtype="<u2", mode="r", offset=8 + n + begin, shape=(self.vocab, self.hidden))

    def rows(self, ids: Sequence[int]) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= self.vocab):
            raise RequestError(f"token id outside the table ({int(ids.min())}..{int(ids.max())}, vocab {self.vocab})")
        return (np.asarray(self._raw[ids]).astype(np.uint32) << 16).view(np.float32)


class ReadoutTable:
    """Rows of the tied embedding table (float32 copies of the checkpoint's bfloat16 rows) for the ids a read-out uses."""

    def __init__(self, ids: Sequence[int], rows: np.ndarray):
        rows = np.asarray(rows, dtype=np.float32)
        assert rows.ndim == 2 and rows.shape == (len(ids), rows.shape[1]), rows.shape
        self.index = {int(i): k for k, i in enumerate(ids)}
        self.rows = rows

    @classmethod
    def from_file(cls, path: str | Path) -> "ReadoutTable":
        from safetensors.numpy import load_file

        t = load_file(str(path))
        return cls(t["ids"].tolist(), t["rows"])

    def logits(self, h: np.ndarray, ids: Sequence[int]) -> np.ndarray:
        missing = [i for i in ids if i not in self.index]
        if missing:
            raise RequestError(f"read-out table has no row for ids {missing[:8]}")
        return self.rows[[self.index[i] for i in ids]] @ np.asarray(h, dtype=np.float32)


def readout(h: np.ndarray, table: ReadoutTable, groups: Sequence[Sequence[int]]) -> List[float]:
    """Option probabilities from the hidden state at the answer slot (step 7)."""
    flat = [i for g in groups for i in g]
    z = dict(zip(flat, table.logits(h, flat).tolist()))
    scores = [max(float(z[i]) for i in g) for g in groups]
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def answer(q: Question, probs: Sequence[float]) -> dict:
    """`api.answer`."""
    if q.type == "noul":
        return {"type": "noul", "noul": probs[0]}
    best = max(range(len(probs)), key=probs.__getitem__)
    if q.type == "choice":
        names = list(q.criteria)
        return {"type": "choice", "choice": names[best], "confidence": probs[best],
                "probabilities": dict(zip(names, probs))}
    return {"type": "score", "score": sum(i * p for i, p in enumerate(probs)), "confidence": probs[best],
            "probabilities": {str(i): p for i, p in enumerate(probs)},
            "legend": {str(i): text for i, text in enumerate(q.criteria)}}


class D1Host:
    """Render, tokenize, run one row per question, read out and answer. Pictures (item 9) need `embeds_graphs`
    ({L: EmbedsGraph}), `embed_table` (EmbedTable) and `vision` (d1_vision.VisionPath); text rows take the embeds graphs
    too when they are loaded (item 10, prefer_embeds)."""

    def __init__(self, tokenizer: D1Tokenizer, graphs: Optional[Mapping[int, RowGraph]] = None,
                 table: Optional[ReadoutTable] = None, pad_id: int = PAD_ID,
                 embeds_graphs: Optional[Mapping[int, EmbedsGraph]] = None, embed_table: Optional[EmbedTable] = None,
                 vision=None, prefer_embeds: bool = True):
        self.tok, self.graphs, self.table, self.pad_id = tokenizer, dict(graphs or {}), table, pad_id
        self.embeds_graphs, self.embed_table, self.vision = dict(embeds_graphs or {}), embed_table, vision
        self.prefer_embeds = prefer_embeds

    @classmethod
    def from_dir(cls, path: str | Path, accelerator: str = "cpu", precision: str = "fp32", threads: int = 4,
                 tokenizer: Optional[D1Tokenizer] = None, ids_graphs: bool = False) -> "D1Host":
        """A bundle directory: contract.json (`embeds_graph.buckets` + `embeds_graph.table`, or `graphs`: [{"L",
        "file"}]; `readout_table.file`, `token_ids.pad`), the tokenizer at `tokenizer.file` (or `tokenizer`), the row graphs and the tables
        it names. The embeds row graphs when the contract names them (item 10, the shipped form), unless ids_graphs."""
        path = Path(path)
        contract = json.loads((path / CONTRACT_FILE).read_text())
        table = ReadoutTable.from_file(path / contract["readout_table"]["file"])
        tok = tokenizer or D1Tokenizer(path / contract["tokenizer"]["file"])
        pad = int(contract["token_ids"]["pad"])
        buckets = (contract.get("embeds_graph") or {}).get("buckets")
        if buckets and not ids_graphs:
            embeds = {int(g["L"]): LiteRTEmbedsGraph(path / g["file"], accelerator, precision, threads) for g in buckets}
            for L, g in embeds.items():
                if g.L != L:
                    raise ValueError(f"{g.path.name} is a {g.L}-token graph, the contract says {L}")
            return cls(tok, {}, table, pad, embeds_graphs=embeds,
                       embed_table=EmbedTable(path / contract["embeds_graph"]["table"]))
        graphs = {int(g["L"]): LiteRTRowGraph(path / g["file"], accelerator, precision, threads)
                  for g in contract["graphs"]}
        for L, g in graphs.items():
            if g.L != L:
                raise ValueError(f"{g.path.name} is a {g.L}-token graph, the contract says {L}")
        return cls(tok, graphs, table, pad, prefer_embeds=False)

    def text_on_embeds(self) -> bool:
        """Item 10: a text row runs on the embeds graphs (rather than the ids graphs)."""
        return bool(self.embeds_graphs) and self.embed_table is not None and (self.prefer_embeds or not self.graphs)

    def row_buckets(self) -> tuple:
        """The buckets a text row can take (the loaded graphs of the text path)."""
        return tuple(self.embeds_graphs) if self.text_on_embeds() else tuple(self.graphs)

    def rows(self, request: Mapping) -> List[Row]:
        state = request.get("state")
        pictures, markup, encode = None, "", self.tok.encode
        if request.get("images"):
            if self.vision is None or self.embed_table is None or not self.embeds_graphs:
                raise RequestError("pictures need the picture path, the embed table and an embeds graph (item 9)")
            from d1_vision import image_markup, row_ids

            pictures = self.vision.pictures(list(request["images"]))
            markup = image_markup(len(pictures))

            def encode(text: str) -> List[int]:
                try:
                    return row_ids(self.tok.encode, text, pictures, self.vision.ids, self.vision.s)
                except ValueError as e:   # e.g. a literal "<image>" in the state
                    raise RequestError(str(e)) from e
        prefix = prefix_text(state, markup)
        state_len = len(encode(prefix))
        out = []
        for name, qd in request["questions"].items():
            q = as_question(qd)
            suffix = suffix_text(self.tok, q)
            out.append(Row(name, q, prefix + suffix, encode(prefix + suffix), readout_ids(self.tok, q), state_len, pictures))
        return out

    def input_tokens(self, request: Mapping, rows: Sequence[Row]) -> int:
        """What the provider's runner reports: the row (one question), or trunk + every branch (several)."""
        if len(rows) == 1:
            return len(rows[0].ids)
        return rows[0].state_len + sum(len(self.tok.encode(suffix_text(self.tok, r.question))) for r in rows)

    def picture_tokens(self, row: Row) -> Optional[np.ndarray]:
        """The tokens of the row's pictures [n, d] (tower and projector graphs), None for a text row."""
        return None if row.pictures is None else self.vision.tokens(row.pictures)

    def embeddings(self, row: Row, picture_tokens: Optional[np.ndarray] = None) -> np.ndarray:
        """The row's input to an embeds graph [n, d] before padding: the table's float32 rows, and the picture tokens
        (computed here when not given) at the `<image>` positions."""
        if self.embed_table is None:
            raise RequestError("no embed table loaded")
        if row.pictures is None:
            return self.embed_table.rows(row.ids)
        from d1_vision import insert_embeddings

        if picture_tokens is None:
            picture_tokens = self.picture_tokens(row)
        return insert_embeddings(row.ids, self.embed_table.rows, picture_tokens, self.vision.ids["image"])

    def hidden_at_slot(self, row: Row, picture_tokens: Optional[np.ndarray] = None) -> np.ndarray:
        if row.pictures is not None or not self.graphs or self.text_on_embeds():
            if not self.embeds_graphs:
                raise RequestError("no row graph loaded")
            L = pick_L(len(row.ids), tuple(self.embeds_graphs))
            emb = self.embeddings(row, picture_tokens)
            n = emb.shape[0]
            x = np.empty((1, L, emb.shape[1]), dtype=np.float32)
            x[0, :n] = emb
            x[0, n:] = self.embed_table.rows([self.pad_id])
            valid = np.zeros((1, L), dtype=np.float32)
            valid[0, :n] = 1.0
            hidden = np.asarray(self.embeds_graphs[L](x, valid))
        else:
            L = pick_L(len(row.ids), tuple(self.graphs))
            ids, valid = pad_row(row.ids, L, self.pad_id)
            hidden = np.asarray(self.graphs[L](ids, valid))
        if hidden.ndim != 3 or hidden.shape[:2] != (1, L):
            raise ValueError(f"the L{L} graph returned {hidden.shape}, expected [1,{L},d]")
        if self.table is not None and self.table.rows.shape[1] != hidden.shape[2]:
            raise ValueError(f"graph d = {hidden.shape[2]}, read-out table d = {self.table.rows.shape[1]}")
        h = hidden[0, row.answer_slot]
        if not np.all(np.isfinite(h)):
            raise FloatingPointError("non-finite hidden state at the answer slot")
        return h

    def decide(self, request: Mapping) -> dict:
        if self.table is None:
            raise RequestError("no read-out table loaded")
        rows = self.rows(request)
        tokens = self.picture_tokens(rows[0]) if rows else None     # once per request: every row has the same pictures
        answers = {r.name: answer(r.question, readout(self.hidden_at_slot(r, tokens), self.table, r.groups)) for r in rows}
        return {"answers": answers, "usage": {"input_tokens": self.input_tokens(request, rows), "output_tokens": 0}}
