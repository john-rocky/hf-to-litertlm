"""Kev on LiteRT (Kev-0.8B and Kev-4B): a reference host in plain Python.

Dependencies: `tokenizers`, `numpy`, `safetensors` and `ai-edge-litert` (the CompiledModel API). No PyTorch and no
`kev` package at run time.

The host answers a `/v1/systemone` request (the TypeSafe request shape the author's server takes) the way the author's
code does at tag kev-1.0 (`kev.api.to_record`, `kev.model.encode` in the row form, `kev.model.PointerHead`,
`kev.api.to_answers`), around two kinds of LiteRT graph:

- row graphs, `*_rowprefill_L{L}_fp16fc_i8emb.tflite` (one signature, `serving_default`): one causal row per question,
  the state followed by that question, from position 0;
- shared-state pairs, `*_sharedstate_Ls{Ls}_Lq{Lq}_fp16fc_i8emb.tflite` (two signatures in one file that share the
  weights): `state_prefill_<Ls>` runs the state once and returns the model state after it (the Gated DeltaNet recurrent
  states and conv tails, the attention keys and values), then `question_step_<Ls>_<Lq>` runs one question on its own
  tokens from that state, its positions continuing after the state. The hidden states are the row form's up to float
  rounding.

1. Request: `{"state": str | JSON, "model"?: str, "questions": {id: {"type": "noul" | "choice" | "score",
   "instructions"?: str | JSON, "criteria": ...}}}`. Noul criteria are an optional object with "false" / "true"
   descriptions, choice criteria an object {name: description} (1 to 255 options), score criteria a list of levels
   (1 to 255).
2. Render (`to_record`): JSON states, instructions and levels become text with `render`; the options of a question are
   noul ["no: <false>", "yes: <true>"] (keys "false", "true"), choice ["name: description", ...] (keys = the criteria
   names), score [rendered level, ...] (keys "0".."n-1"). An empty description leaves the bare name.
3. Tokenize with the Kev repositories' `tokenizer.json` (the same file in jaredpalmer/kev-0.8b and jaredpalmer/kev-4b at
   v1.0), no special tokens added. That file holds the pipeline the author's code gets from transformers
   (`Qwen2Tokenizer` for Qwen/Qwen3.5-0.8B-Base at dc7cdfe2: 248,077 entries, 33 added tokens). The base repository's
   own `tokenizer.json` read directly gives other ids for some text (Devanagari, `<think>`, `<tool_response>`,
   `<tts_pad>`), so the host refuses it. In caller text `<|name|>` is rewritten to `<¦name¦>` first, so no caller text
   can produce a delimiter token.
4. One causal row per question (`encode_rows`): [248060] + state + [248061] + instructions + for each option
   [248049] + option + [248050], then [248062]. Every row starts at position 0. The readout positions are the
   last token (248062, "decide") and each option's closing 248050. The state part of every row is [248060] + state;
   the rest of a row is that question's branch.
5. Route each request (`route`, `mode`):
   - "row": every question through the smallest row graph that holds its row (`pick_L`: 64, 128, 256, 512, 1,024 or
     2,048 tokens). A longer row is refused (`RowTooLong`), never truncated.
   - "pair": the state through the smallest loaded pair whose Ls holds it, every question through that pair's question
     step. A request whose state or one of whose branches does not fit raises `PairDoesNotFit` (a `RowTooLong`).
   - "auto" (the default): take the smallest loaded pair whose Ls holds the state and the n questions whose branches fit
     its Lq. They take the pair when their row graphs would compute more than `pair_ratio` (default 1.5) times as many
     positions as the pair: R > pair_ratio * P, with R = the sum of the lengths of the row graphs they would use
     (`pick_L` of each row; a row that no loaded row graph holds counts as infinite) and P = Ls + n * Lq. The other
     questions, and all of them when the condition fails, take their row graphs.
   Inputs are right-padded: `ids` with 248044 (`<|endoftext|>`), `valid` 1.0 on real tokens and 0.0 on padding.
6. Run. A row graph (`KevGraph`) maps `ids` int32 [1,L] and `valid` float32 [1,L] to `hidden` float32 [1,L,d], the
   hidden states after the final RMSNorm (d = 1024 for Kev-0.8B, 2560 for Kev-4B). A pair (`KevPairGraph`):
   `state_prefill_<Ls>` maps the state's `ids` / `valid` [1,Ls] to the state tensors; `question_step_<Ls>_<Lq>` maps a
   branch's `ids` / `valid` [1,Lq], the state's `valid` as `state_valid` [1,Ls] and the state tensors to `hidden`
   float32 [1,Lq,d]. `handover` moves the state between the two: "direct" (the default) gives state_prefill's output
   buffers to question_step as its inputs, "host" reads them back to numpy and writes them into question_step's input
   buffers (23,347,200 bytes per request for Kev-0.8B at Ls 128). Both give the same bits.
7. Read out on the host in float32 (`PointerHead`): q = h[decide] @ Wq^T + bq, k = h[options] @ Wk^T + bk,
   z = (k @ q) / sqrt(256) / T, probs = softmax(z). T is the checkpoint's temperature, read from the head's JSON
   (Kev-0.8B: 2.3510958125672174, Kev-4B: 2.406050072164233). On the pair path the readout positions are taken in the
   branch (row index minus the state length).
8. Answer like `kev.api.to_answers`: noul {"noul": p(true)}, choice {"choice": the most likely name, "confidence",
   "probabilities"}, score {"score": the expected level, "legend", "probabilities", "confidence"}; every number is
   rounded to 4 decimals. The response is {"model", "answers", "usage": {"input_tokens", "output_tokens"},
   "latency_ms"} as in the author's server: input_tokens counts the state once plus every question's branch,
   output_tokens is the token count of the serialized answers, latency_ms is the model time (graphs and readout).

GPU precision: `accelerator="gpu"` runs at float32 precision (`precision="fp32"`, the default:
`GpuOptions(enforce_f32=True)`). On a Mac (Metal, Kev-0.8B, the L128 and L2048 files), the default GPU precision
(float16 activations, `precision="fp16"`) gave finite hidden states on every test row, but it moved the probabilities
by up to 0.0332 (L128) and 0.0352 (L2048) from the author's fp32 code, mean 3.6e-3 and 4.1e-3, outside the test
tolerance (max 0.02, mean 0.002); at float32 the same files stay within 0.0104 (mean 9.7e-4 and 8.9e-4). Python's
`GpuOptions` offers only these two precisions. The Kotlin and C APIs also have `FP16_WITH_FP32_ACCUM` (float16 storage,
float32 accumulation), a third mode that this host does not select. A non-finite readout raises `NonFiniteOutput`
instead of returning an answer.

Pairs on the GPU: `constant_tensor_sharing=True` (the default) keeps one copy of the weights on the GPU for the two
signatures. Without it the GPU holds them once per signature: on a Mac (Metal, float32, Kev-0.8B Ls128 pair, a new
process per run) the process footprint after one request was 6.3 GB without sharing and 3.0 GB with it, and its peak
during the compile 9.3 to 9.5 GB and 3.3 GB. Sharing costs time on Metal (two timing passes): one question took 75.3 to
75.7 ms through the Ls128 pair with sharing and 62.3 to 62.5 ms without, against 33.9 ms through a row graph; five
questions took 199.4 to 200.1 ms and 163.6 to 164.0 ms, against 230.0 to 231.8 ms through the row graphs (rows of 94 to
142 tokens, on the 128- and 256-token graphs). A pair computes Ls positions once and Lq per question, a row graph its
whole length per question; hence the rule of "auto": the pair takes a request's questions only when the row graphs
would compute more than `pair_ratio` times as many positions.

Example:
    from kev_litert import KevLiteRT

    with KevLiteRT.from_dir("Kev-0.8B-LiteRT") as kev:   # or a Kev-4B-LiteRT directory
        kev.decide({"state": "Order #1182 arrived with a cracked screen.",
                    "questions": {"refund": {"type": "noul", "instructions": "Should we offer a refund?"}}})

Command line:
    python kev_litert.py --graph kev-0.8b_rowprefill_L512_fp16fc_i8emb.tflite \\
        --head head/kev_0.8b_pointer_head.safetensors --tokenizer tokenizer/tokenizer.json --request request.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from tokenizers import Tokenizer

STATE_ID, QUESTION_ID, OPTION_START_ID, OPTION_END_ID, DECIDE_ID = 248060, 248061, 248049, 248050, 248062
PAD_ID = 248044
DELIMITER_TOKENS = {"<|fim_prefix|>": STATE_ID, "<|fim_middle|>": QUESTION_ID, "<|box_start|>": OPTION_START_ID,
                    "<|box_end|>": OPTION_END_ID, "<|fim_suffix|>": DECIDE_ID, "<|endoftext|>": PAD_ID}
PIPELINE_TOKENS = {"<think>": 248068, "<tool_response>": 248066, "<tts_pad>": 248072}   # added by transformers, absent from the base repo's file
VOCAB_WITH_ADDED = 248077
LENGTHS = (64, 128, 256, 512, 1024, 2048)
PAIR_RATIO = 1.5                # mode "auto": questions take the pair when the row graphs' positions exceed this x the pair's
MODES = ("auto", "row", "pair")
HANDOVERS = ("direct", "host")
MAX_OPTIONS = 255
DEFAULT_MODEL = "kev-latest"
SIGNATURE = "serving_default"
INPUT_NAMES = ("ids", "valid")
OUTPUT_NAME = "hidden"
GRAPH_GLOB = "*_rowprefill_L{length}_fp16fc_i8emb.tflite"   # kev-0.8b_... or kev-4b_... at the top level
PAIR_GLOB = "*_sharedstate_Ls*_Lq*_fp16fc_i8emb.tflite"
HEAD_GLOB = "head/*_pointer_head.safetensors"
TOKENIZER_FILE = "tokenizer/tokenizer.json"

JSONContent = Union[str, dict, list, int, float, bool, None]
PairShape = Tuple[int, int]


class RequestError(ValueError):
    """The request cannot be answered as given (the author's server answers 422)."""

    status = 422


class RowTooLong(RequestError):
    """A question's row (state + its branch) is longer than the largest graph."""


class PairDoesNotFit(RowTooLong):
    """mode "pair": no loaded shared-state pair holds the request's state, or a question's branch is longer than its Lq."""


class NonFiniteOutput(RuntimeError):
    """The graph returned a non-finite hidden state at a readout position."""


# --- the author's request rendering (kev.api at kev-1.0) -----------------------------------------------------------

def render(v: JSONContent, indent: int = 0) -> str:
    """Flatten str | object | array into the text the model sees; field names are kept as labels (kev.api.render)."""
    pad = "  " * indent
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    if isinstance(v, list):
        return "\n".join(f"{pad}- {render(x, indent + 1).lstrip()}" for x in v)
    return "\n".join(f"{pad}{k}:\n{render(x, indent + 1)}" if isinstance(x, (dict, list)) else f"{pad}{k}: {render(x)}"
                     for k, x in v.items())


def option_text(name: str, desc: JSONContent) -> str:
    """'name' or 'name: description' (kev.api.option_text)."""
    return name if desc is None or desc == "" else f"{name}: {render(desc)}"


def question_keys(qtype: str, criteria) -> List[str]:
    """The keys a question's probabilities are reported under, in option order (kev.api.question_keys)."""
    if qtype == "choice":
        return list(criteria)
    if qtype == "noul":
        return ["false", "true"]
    return [str(i) for i in range(len(criteria))]


def _check_json(value, where: str) -> None:
    if not isinstance(value, (str, dict, list, int, float, bool, type(None))):
        raise RequestError(f"{where}: must be a string, number, boolean, null, object or array")


def validate(request: Mapping[str, Any]) -> Tuple[JSONContent, str, Dict[str, Dict[str, Any]]]:
    """The checks kev.api.SystemOneRequest applies -> (state, model, {qid: {type, instructions, criteria}}).
    Unknown fields are ignored, as there."""
    if not isinstance(request, Mapping):
        raise RequestError("the request must be a JSON object")
    if "state" not in request:
        raise RequestError("state: field required")
    _check_json(request["state"], "state")
    model = request.get("model", DEFAULT_MODEL)
    if not isinstance(model, str):
        raise RequestError("model: must be a string")
    questions = request.get("questions")
    if not isinstance(questions, Mapping) or not questions:
        raise RequestError("questions: must be an object with at least one question")
    out = {}
    for qid, q in questions.items():
        where = f"questions.{qid}"
        if not isinstance(q, Mapping):
            raise RequestError(f"{where}: must be an object")
        qtype = q.get("type")
        if qtype not in ("noul", "choice", "score"):
            raise RequestError(f"{where}.type: must be 'noul', 'choice' or 'score'")
        instructions = q.get("instructions")
        _check_json(instructions, f"{where}.instructions")
        criteria = q.get("criteria")
        if qtype == "noul":
            if criteria is not None and not isinstance(criteria, Mapping):
                raise RequestError(f"{where}.criteria: a noul question takes an object with 'false' / 'true' or null")
        elif qtype == "choice":
            if not isinstance(criteria, Mapping) or not 1 <= len(criteria) <= MAX_OPTIONS:
                raise RequestError(f"{where}.criteria: must be an object with 1..{MAX_OPTIONS} options")
        elif not isinstance(criteria, list) or not 1 <= len(criteria) <= MAX_OPTIONS:
            raise RequestError(f"{where}.criteria: must be a list of 1..{MAX_OPTIONS} levels")
        out[qid] = {"type": qtype, "instructions": instructions, "criteria": criteria}
    return request["state"], model, out


def to_record(request: Mapping[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """-> (internal record {"state", "questions": [{"instr", "options", "label"}]}, per-question metadata
    {"id", "type", "keys", "legend" for score}) as kev.api.to_record builds them."""
    state, _, questions = validate(request)
    return _record(state, questions)


def _record(state: JSONContent, questions: Mapping[str, Mapping[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    qs, meta = [], []
    for qid, q in questions.items():
        m = {"id": qid, "type": q["type"], "keys": question_keys(q["type"], q["criteria"])}
        if q["type"] == "noul":
            c = q["criteria"] or {}
            opts = [option_text("no", c.get("false")), option_text("yes", c.get("true"))]
        elif q["type"] == "choice":
            opts = [option_text(k, v) for k, v in q["criteria"].items()]
        else:
            opts = [render(x) for x in q["criteria"]]
            m["legend"] = dict(zip(m["keys"], opts))
        qs.append({"instr": render(q["instructions"]), "options": opts, "label": 0})
        meta.append(m)
    return {"state": render(state), "questions": qs}, meta


# --- the author's answer shaping (kev.api at kev-1.0) ---------------------------------------------------------------

def _normalize(p: List[float]) -> List[float]:
    t = sum(p)
    return [1 / len(p)] * len(p) if t == 0 else [x / t for x in p]


def choice_confidence(p: List[float]) -> float:
    """(p_max - 1/K) / (1 - 1/K): 0 at uniform, 1 at certainty."""
    K = len(p)
    return 1.0 if K == 1 else (max(_normalize(p)) - 1 / K) / (1 - 1 / K)


def score_confidence(p: List[float]) -> float:
    """max(0, 1 - E|level - mode| / D), D = the mean absolute deviation of a uniform distribution over the levels."""
    L = len(p)
    if L == 1:
        return 1.0
    p = _normalize(p)
    mode = max(range(L), key=p.__getitem__)
    D = sum(abs(i - (L - 1) / 2) for i in range(L)) / L
    return max(0.0, 1.0 - sum(pi * abs(i - mode) for i, pi in enumerate(p)) / D)


def round_prob(x: float) -> float:
    return round(float(x), 4)


def to_answers(probs: List[List[float]], meta: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-question probabilities (Python floats, option order) -> the response's "answers" (kev.api.to_answers)."""
    out = {}
    for p, m in zip(probs, meta):
        if m["type"] == "noul":
            out[m["id"]] = {"type": "noul", "noul": round_prob(p[1])}
        elif m["type"] == "choice":
            dist = {k: round_prob(v) for k, v in zip(m["keys"], p)}
            out[m["id"]] = {"type": "choice", "choice": m["keys"][max(range(len(p)), key=lambda i: p[i])],
                            "confidence": round_prob(choice_confidence(p)), "probabilities": dist}
        else:
            score = sum(i * pi for i, pi in enumerate(p))
            out[m["id"]] = {"type": "score", "score": round_prob(score), "legend": m["legend"],
                            "probabilities": {str(i): round_prob(v) for i, v in enumerate(p)},
                            "confidence": round_prob(score_confidence(p))}
    return out


# --- tokenizer, rows, graph, head -----------------------------------------------------------------------------------

_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


class KevTokenizer:
    """The Kev repository's tokenizer.json through `tokenizers`, as kev.model.user_tokens uses the tokenizer."""

    def __init__(self, path: Union[str, Path]):
        self._tokenizer = Tokenizer.from_file(str(path))
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()
        for token, token_id in DELIMITER_TOKENS.items():
            if self._tokenizer.token_to_id(token) != token_id:
                raise ValueError(f"{path} maps {token} to {self._tokenizer.token_to_id(token)}, not {token_id}")
        size = self._tokenizer.get_vocab_size(with_added_tokens=True)
        if size != VOCAB_WITH_ADDED or any(self._tokenizer.token_to_id(t) != i for t, i in PIPELINE_TOKENS.items()):
            raise ValueError(f"{path} is not the Kev repository's tokenizer.json ({size} entries, expected "
                             f"{VOCAB_WITH_ADDED} with <think> = 248068); the base repository's file tokenizes some "
                             "text differently from the author's code")

    def user_tokens(self, text: str) -> List[int]:
        """Token ids of caller text; `<|name|>` becomes `<¦name¦>` first, so no delimiter can be forged."""
        return self._tokenizer.encode(_SPECIAL_RE.sub(r"<¦\1¦>", text), add_special_tokens=False).ids

    def count(self, text: str) -> int:
        """Plain token count (kev.api.output_tokens counts the serialized answers this way)."""
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)


def pick_L(row_len: int, lengths: Sequence[int] = LENGTHS) -> int:
    """The smallest graph length that holds a row of `row_len` tokens; RowTooLong when none does."""
    for length in sorted(lengths):
        if row_len <= length:
            return length
    if not lengths:
        raise RowTooLong(f"a question row is {row_len:,} tokens and no row graph is loaded")
    raise RowTooLong(f"a question row is {row_len:,} tokens; the largest graph holds {max(lengths):,}. "
                     "Shorten the state or the question.")


def encode_rows(tokenizer: KevTokenizer, request: Mapping[str, Any]) -> Dict[str, Any]:
    """-> {"model", "input_tokens", "state_tokens", "questions": [{"id", "type", "keys", "legend"?, "row_ids",
    "decide_idx", "opt_idx"}]}. One causal row per question = the state + that question's branch (kev.model.encode
    followed by kev.model.rows_of); decide_idx is the row's last token, opt_idx each option's closing token."""
    state_value, model, validated = validate(request)
    rec, meta = _record(state_value, validated)
    state = [STATE_ID] + tokenizer.user_tokens(rec["state"])
    questions, input_tokens = [], len(state)
    for q, m in zip(rec["questions"], meta):
        branch = [QUESTION_ID] + tokenizer.user_tokens(q["instr"])
        opt_idx = []
        for option in q["options"]:
            branch += [OPTION_START_ID] + tokenizer.user_tokens(option) + [OPTION_END_ID]
            opt_idx.append(len(state) + len(branch) - 1)
        branch.append(DECIDE_ID)
        row = state + branch
        questions.append({**m, "row_ids": row, "decide_idx": len(row) - 1, "opt_idx": opt_idx})
        input_tokens += len(branch)
    return {"model": model, "input_tokens": input_tokens, "state_tokens": len(state), "questions": questions}


def graph_length(path: Union[str, Path]) -> Optional[int]:
    """The row length written in a graph's file name (`..._L1024_...`), or None."""
    match = re.search(r"_L(\d+)[_.]", Path(path).name)
    return int(match.group(1)) if match else None


def pair_shape(path: Union[str, Path]) -> Optional[PairShape]:
    """(Ls, Lq) written in a shared-state pair's file name (`..._sharedstate_Ls128_Lq64_...`), or None."""
    match = re.search(r"_sharedstate_Ls(\d+)_Lq(\d+)[_.]", Path(path).name)
    return (int(match.group(1)), int(match.group(2))) if match else None


def _padded(ids: Sequence[int], length: int) -> Tuple[np.ndarray, np.ndarray]:
    """(ids int32 [1, length] right-padded with PAD_ID, valid float32 [1, length])."""
    a = np.full((1, length), PAD_ID, dtype=np.int32)
    a[0, :len(ids)] = np.asarray(ids, dtype=np.int32)
    v = np.zeros((1, length), dtype=np.float32)
    v[0, :len(ids)] = 1.0
    return a, v


class KevGraph:
    """One row-prefill graph on the CompiledModel API: ids int32 [1,L] + valid float32 [1,L] -> hidden [1,L,d]."""

    def __init__(self, path: Union[str, Path], accelerator: str = "cpu", precision: str = "fp32", threads: int = 4):
        from ai_edge_litert.compiled_model import (CompiledModel, CpuOptions, GpuOptions, HardwareAccelerator,
                                                   Options)
        if accelerator == "gpu":
            if precision not in ("fp32", "fp16"):
                raise ValueError("precision must be 'fp32' or 'fp16'")
            options = Options(hardware_accelerators=HardwareAccelerator.GPU,
                              gpu_options=GpuOptions(enforce_f32=precision == "fp32"))
        elif accelerator == "cpu":
            options = Options(hardware_accelerators=HardwareAccelerator.CPU,
                              cpu_options=CpuOptions(num_threads=threads))
        else:
            raise ValueError("accelerator must be 'cpu' or 'gpu'")
        self.path, self.accelerator, self.precision = Path(path), accelerator, precision
        started = time.perf_counter()
        self.model = CompiledModel.from_file(str(path), options=options)
        self.compile_seconds = time.perf_counter() - started
        signature = self.model.get_signature_list().get(SIGNATURE, {})
        if sorted(signature.get("inputs", [])) != sorted(INPUT_NAMES) or list(signature.get("outputs", [])) != [OUTPUT_NAME]:
            raise ValueError(f"unexpected signature in {self.path.name}: {signature}")
        ids = self.model.get_input_tensor_details(SIGNATURE)["ids"]
        hidden = self.model.get_output_tensor_details(SIGNATURE)[OUTPUT_NAME]
        self.length = int(list(ids["shape"])[1])
        self.hidden_size = int(list(hidden["shape"])[2])
        self.inputs = {name: self.model.create_input_buffer_by_name(SIGNATURE, name) for name in INPUT_NAMES}
        self.outputs = {OUTPUT_NAME: self.model.create_output_buffer_by_name(SIGNATURE, OUTPUT_NAME)}
        try:
            self.fully_accelerated = bool(self.model.is_fully_accelerated())
        except Exception:  # informational only
            self.fully_accelerated = None

    def run(self, row_ids: Sequence[int]) -> np.ndarray:
        """Hidden states [L, d] (float32) of one right-padded row."""
        n, length = len(row_ids), self.length
        if n > length:
            raise RowTooLong(f"a row of {n} tokens does not fit this {length}-token graph")
        ids = np.full((1, length), PAD_ID, dtype=np.int32)
        ids[0, :n] = np.asarray(row_ids, dtype=np.int32)
        valid = np.zeros((1, length), dtype=np.float32)
        valid[0, :n] = 1.0
        self.inputs["ids"].write(ids)
        self.inputs["valid"].write(valid)
        self.model.run_by_name(SIGNATURE, self.inputs, self.outputs)
        size = length * self.hidden_size
        return np.asarray(self.outputs[OUTPUT_NAME].read(size, np.float32), dtype=np.float32).reshape(length, self.hidden_size)

    def close(self) -> None:
        for buffer in list(self.inputs.values()) + list(self.outputs.values()):
            try:
                buffer.destroy()
            except Exception:
                pass
        self.inputs, self.outputs = {}, {}
        self.model.close()


class KevPairGraph:
    """One shared-state pair on the CompiledModel API: state_prefill_<Ls> (the state's ids / valid -> the state tensors)
    and question_step_<Ls>_<Lq> (a branch's ids / valid, state_valid and the state tensors -> hidden [1,Lq,d]).
    run_state() once per request, then run_question() per question."""

    def __init__(self, path: Union[str, Path], accelerator: str = "cpu", precision: str = "fp32", threads: int = 4,
                 handover: str = "direct", constant_tensor_sharing: bool = True):
        from ai_edge_litert.compiled_model import (CompiledModel, CpuOptions, GpuOptions, HardwareAccelerator,
                                                   Options)
        if handover not in HANDOVERS:
            raise ValueError(f"handover must be one of {HANDOVERS}")
        if accelerator == "gpu":
            if precision not in ("fp32", "fp16"):
                raise ValueError("precision must be 'fp32' or 'fp16'")
            options = Options(hardware_accelerators=HardwareAccelerator.GPU,
                              gpu_options=GpuOptions(enforce_f32=precision == "fp32",
                                                     constant_tensor_sharing=constant_tensor_sharing))
        elif accelerator == "cpu":
            options = Options(hardware_accelerators=HardwareAccelerator.CPU,
                              cpu_options=CpuOptions(num_threads=threads))
        else:
            raise ValueError("accelerator must be 'cpu' or 'gpu'")
        self.path, self.accelerator, self.precision, self.handover = Path(path), accelerator, precision, handover
        self.constant_tensor_sharing = bool(constant_tensor_sharing) if accelerator == "gpu" else None
        started = time.perf_counter()
        self.model = CompiledModel.from_file(str(path), options=options)
        self.compile_seconds = time.perf_counter() - started
        sigs = self.model.get_signature_list()
        state_sig = [k for k in sigs if re.fullmatch(r"state_prefill_\d+", k)]
        question_sig = [k for k in sigs if re.fullmatch(r"question_step_\d+_\d+", k)]
        if len(state_sig) != 1 or len(question_sig) != 1:
            raise ValueError(f"{self.path.name}: expected state_prefill_<Ls> and question_step_<Ls>_<Lq>, got {list(sigs)}")
        self.sig_state, self.sig_question = state_sig[0], question_sig[0]
        self.Ls, self.Lq = (int(x) for x in self.sig_question.split("_")[2:])
        if int(self.sig_state.split("_")[2]) != self.Ls:
            raise ValueError(f"{self.path.name}: state_prefill and question_step disagree on Ls")
        self.state_names = list(sigs[self.sig_state]["outputs"])
        if (sorted(sigs[self.sig_state]["inputs"]) != sorted(INPUT_NAMES)
                or sorted(sigs[self.sig_question]["inputs"]) != sorted([*INPUT_NAMES, "state_valid", *self.state_names])
                or list(sigs[self.sig_question]["outputs"]) != [OUTPUT_NAME]):
            raise ValueError(f"{self.path.name}: question_step does not take state_prefill's outputs")
        det = self.model.get_output_tensor_details(self.sig_state)
        self.numel = {n: int(np.prod(list(det[n]["shape"]))) for n in self.state_names}
        self.state_bytes = 4 * sum(self.numel.values())
        self.hidden_size = int(list(self.model.get_output_tensor_details(self.sig_question)[OUTPUT_NAME]["shape"])[2])
        m = self.model
        self.in_state = {n: m.create_input_buffer_by_name(self.sig_state, n) for n in INPUT_NAMES}
        self.out_state = {n: m.create_output_buffer_by_name(self.sig_state, n) for n in self.state_names}
        self.in_question = {n: m.create_input_buffer_by_name(self.sig_question, n) for n in sigs[self.sig_question]["inputs"]}
        self.out_question = {OUTPUT_NAME: m.create_output_buffer_by_name(self.sig_question, OUTPUT_NAME)}
        # "direct": question_step reads the state from state_prefill's own output buffers
        self.direct = {**{n: self.in_question[n] for n in (*INPUT_NAMES, "state_valid")},
                       **{n: self.out_state[n] for n in self.state_names}}
        try:
            self.fully_accelerated = bool(self.model.is_fully_accelerated())
        except Exception:  # informational only
            self.fully_accelerated = None

    def run_state(self, state_ids: Sequence[int]) -> None:
        """Runs the state ([248060] + state tokens) and hands its state tensors to the question step."""
        if len(state_ids) > self.Ls:
            raise PairDoesNotFit(f"the state is {len(state_ids):,} tokens; this pair holds {self.Ls}")
        ids, valid = _padded(state_ids, self.Ls)
        self.in_state["ids"].write(ids)
        self.in_state["valid"].write(valid)
        self.model.run_by_name(self.sig_state, self.in_state, self.out_state)
        if self.handover == "host":
            for n in self.state_names:
                self.in_question[n].write(np.asarray(self.out_state[n].read(self.numel[n], np.float32), dtype=np.float32))
        self.in_question["state_valid"].write(valid)

    def run_question(self, branch_ids: Sequence[int]) -> np.ndarray:
        """Hidden states [Lq, d] (float32) of one branch after the last run_state()."""
        if len(branch_ids) > self.Lq:
            raise PairDoesNotFit(f"a question is {len(branch_ids):,} tokens; this pair holds {self.Lq}")
        ids, valid = _padded(branch_ids, self.Lq)
        self.in_question["ids"].write(ids)
        self.in_question["valid"].write(valid)
        inputs = self.direct if self.handover == "direct" else self.in_question
        self.model.run_by_name(self.sig_question, inputs, self.out_question)
        size = self.Lq * self.hidden_size
        return np.asarray(self.out_question[OUTPUT_NAME].read(size, np.float32), dtype=np.float32).reshape(
            self.Lq, self.hidden_size)

    def close(self) -> None:
        for buffer in [*self.in_state.values(), *self.out_state.values(), *self.in_question.values(),
                       *self.out_question.values()]:
            try:
                buffer.destroy()
            except Exception:
                pass
        self.in_state, self.out_state, self.in_question, self.out_question, self.direct = {}, {}, {}, {}, {}
        self.model.close()


class PointerHead:
    """The checkpoint's pointer head in numpy float32: logits over a question's options from two hidden states."""

    def __init__(self, path: Union[str, Path], temperature: Optional[float] = None):
        from safetensors.numpy import load_file
        tensors = load_file(str(path))
        self.Wq = tensors["q.weight"].astype(np.float32)
        self.bq = tensors["q.bias"].astype(np.float32)
        self.Wk = tensors["k.weight"].astype(np.float32)
        self.bk = tensors["k.bias"].astype(np.float32)
        if temperature is None:
            contract = Path(path).with_suffix(".json")
            if not contract.is_file():
                raise ValueError(f"no temperature given and no {contract.name} next to {Path(path).name}")
            doc = json.loads(contract.read_text())
            temperature = doc["temperature"]
            ids = doc.get("delimiter_token_ids", {})
            expected = {"state": STATE_ID, "question": QUESTION_ID, "option_start": OPTION_START_ID,
                        "option_end": OPTION_END_ID, "decide": DECIDE_ID}
            if (ids and ids != expected) or doc.get("pad_token_id", PAD_ID) != PAD_ID:
                raise ValueError(f"{contract.name} declares other delimiter / pad ids than this host")
        self.temperature = float(temperature)
        self.head_dim, self.hidden_size = self.Wq.shape
        if self.Wk.shape != self.Wq.shape or self.bq.shape != (self.head_dim,) or self.bk.shape != (self.head_dim,):
            raise ValueError(f"unexpected head shapes in {Path(path).name}")
        self.scale = np.float32(1.0 / math.sqrt(self.head_dim))

    def __call__(self, h_sel: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """h_sel [1+K, d] in the order [decide, *options] -> (z before the temperature, z after it, probs), float32."""
        h = np.asarray(h_sel, dtype=np.float32)
        q = h[0] @ self.Wq.T + self.bq
        k = h[1:] @ self.Wk.T + self.bk
        z_pre = (k @ q) * self.scale
        z_post = (z_pre / np.float32(self.temperature)).astype(np.float32)
        e = np.exp(z_post - z_post.max())
        return z_pre.astype(np.float32), z_post, (e / e.sum()).astype(np.float32)


def select(hidden: np.ndarray, question: Mapping[str, Any]) -> np.ndarray:
    """hidden [L, d] of one row -> [1+K, d]: the decide row, then each option's closing token."""
    return np.concatenate([hidden[question["decide_idx"]][None], hidden[np.asarray(question["opt_idx"])]], 0)


# --- the host -------------------------------------------------------------------------------------------------------

class KevLiteRT:
    """Kev on LiteRT: /v1/systemone requests in, answers out. Graphs are compiled on first use."""

    def __init__(self, graphs: Union[Mapping[int, Union[str, Path]], Sequence[Union[str, Path]]],
                 head: Union[str, Path], tokenizer: Union[str, Path], accelerator: str = "cpu",
                 precision: str = "fp32", threads: int = 4, temperature: Optional[float] = None, *,
                 pairs: Union[Mapping[PairShape, Union[str, Path]], Sequence[Union[str, Path]], None] = None,
                 mode: str = "auto", pair_ratio: float = PAIR_RATIO, handover: str = "direct",
                 constant_tensor_sharing: bool = True):
        """`graphs`: {row length: tflite path} or a list of paths whose names carry the length (`_L1024_`); a path
        without one is compiled here to read it; a list may also hold shared-state pairs (`_sharedstate_Ls128_Lq64_`).
        `pairs`: {(Ls, Lq): path} or a list of pair paths. `accelerator`: "cpu" or "gpu"; `precision` ("fp32" or
        "fp16") applies to the GPU; `threads` to the CPU. `temperature`: default = the value in the head's JSON.
        `mode`: "auto", "row" or "pair" (module docstring, step 5); `pair_ratio`: in mode "auto", how many times the
        pair's positions the row graphs must exceed before a request's questions take the pair (step 5);
        `handover`: "direct" or "host" (step 6); `constant_tensor_sharing`:
        one copy of a pair's weights on the GPU for its two signatures (module docstring, "Pairs on the GPU")."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if handover not in HANDOVERS:
            raise ValueError(f"handover must be one of {HANDOVERS}")
        if not pair_ratio > 0:
            raise ValueError("pair_ratio must be above 0")
        self.accelerator, self.precision, self.threads = accelerator, precision, threads
        self.mode, self.pair_ratio, self.handover = mode, float(pair_ratio), handover
        self.constant_tensor_sharing = bool(constant_tensor_sharing)
        self.tokenizer = KevTokenizer(tokenizer)
        self.head = PointerHead(head, temperature)
        self._graphs: Dict[int, KevGraph] = {}
        self._paths: Dict[int, Path] = {}
        self._pairs: Dict[PairShape, KevPairGraph] = {}
        self._pair_paths: Dict[PairShape, Path] = {}
        pair_items = list(pairs.items()) if isinstance(pairs, Mapping) else [(pair_shape(p), p) for p in pairs or []]
        if isinstance(graphs, Mapping):
            items = list(graphs.items())
        else:
            items = []
            for p in graphs:
                shape = pair_shape(p)
                if shape is not None:
                    pair_items.append((shape, p))
                else:
                    items.append((graph_length(p), p))
        for length, path in items:
            path = Path(path)
            if not path.is_file():
                raise FileNotFoundError(path)
            if length is None:
                graph = self._compile(path)
                length = graph.length
                self._graphs[length] = graph
            if length in self._paths:
                raise ValueError(f"two graphs for length {length}")
            self._paths[int(length)] = path
        for shape, path in pair_items:
            path = Path(path)
            if not path.is_file():
                raise FileNotFoundError(path)
            if shape is None:
                raise ValueError(f"{path.name}: the file name does not carry _sharedstate_Ls<Ls>_Lq<Lq>_")
            shape = (int(shape[0]), int(shape[1]))
            if shape in self._pair_paths:
                raise ValueError(f"two pairs for Ls {shape[0]} / Lq {shape[1]}")
            self._pair_paths[shape] = path
        if not self._paths and not self._pair_paths:
            raise ValueError("no graph given")

    @classmethod
    def from_dir(cls, root: Union[str, Path], lengths: Sequence[int] = LENGTHS,
                 pair_shapes: Optional[Sequence[PairShape]] = None, **kwargs) -> "KevLiteRT":
        """The shipped layout: the graphs at the top level, head/ and tokenizer/ beside them (one model per folder).
        Loads the row graphs of `lengths` and the shared-state pairs that are present (`pair_shapes`: None = every pair
        present, or a list of (Ls, Lq); () = none); at least one must be present. A row longer than the longest row
        graph present is refused (`RowTooLong`) unless a pair takes its question, so a download of the L512 file alone
        serves rows of up to 512 tokens, and the shorter files serve short rows faster."""
        root = Path(root)

        def one(pattern: str) -> Path:
            found = sorted(root.glob(pattern))
            if len(found) != 1:
                raise FileNotFoundError(f"{root}: expected one {pattern}, found {[p.name for p in found]}")
            return found[0]

        graphs = {}
        for length in lengths:
            pattern = GRAPH_GLOB.format(length=length)
            found = sorted(root.glob(pattern))
            if len(found) > 1:
                raise FileNotFoundError(f"{root}: expected at most one {pattern}, found {[p.name for p in found]}")
            if found:
                graphs[length] = found[0]
        pairs: Dict[PairShape, Path] = {}
        for path in sorted(root.glob(PAIR_GLOB)):
            shape = pair_shape(path)
            if shape is None or (pair_shapes is not None and shape not in {tuple(s) for s in pair_shapes}):
                continue
            if shape in pairs:
                raise FileNotFoundError(f"{root}: two pairs for Ls {shape[0]} / Lq {shape[1]}: {pairs[shape].name}, {path.name}")
            pairs[shape] = path
        missing = [tuple(s) for s in (pair_shapes or ()) if tuple(s) not in pairs]
        if missing:
            raise FileNotFoundError(f"{root}: no shared-state pair for (Ls, Lq) {missing}")
        if not graphs and not pairs:
            raise FileNotFoundError(f"{root}: no row-prefill graph ({GRAPH_GLOB.format(length='*')}) and no "
                                    f"shared-state pair ({PAIR_GLOB}) found")
        return cls(graphs, one(HEAD_GLOB), root / TOKENIZER_FILE, pairs=pairs, **kwargs)

    @property
    def lengths(self) -> List[int]:
        """The row lengths of the loaded row graphs."""
        return sorted(self._paths)

    @property
    def pair_shapes(self) -> List[PairShape]:
        """(Ls, Lq) of the loaded shared-state pairs."""
        return sorted(self._pair_paths)

    def _compile(self, path: Path) -> KevGraph:
        graph = KevGraph(path, self.accelerator, self.precision, self.threads)
        if graph.hidden_size != self.head.hidden_size:
            graph.close()
            raise ValueError(f"{path.name} returns d={graph.hidden_size}; the head reads d={self.head.hidden_size}")
        return graph

    def graph(self, length: int) -> KevGraph:
        """The compiled graph for one row length (compiled on first use)."""
        if length not in self._graphs:
            if length not in self._paths:
                raise RowTooLong(f"no {length}-token graph loaded (have {self.lengths})")
            graph = self._compile(self._paths[length])
            if graph.length != length:
                graph.close()
                raise ValueError(f"{self._paths[length].name} holds {graph.length} tokens, not {length}")
            self._graphs[length] = graph
        return self._graphs[length]

    def close_graph(self, length: int) -> None:
        graph = self._graphs.pop(length, None)
        if graph is not None:
            graph.close()

    def pair(self, shape: PairShape) -> KevPairGraph:
        """The compiled shared-state pair for (Ls, Lq) (compiled on first use)."""
        shape = (int(shape[0]), int(shape[1]))
        if shape not in self._pairs:
            if shape not in self._pair_paths:
                raise PairDoesNotFit(f"no pair for Ls {shape[0]} / Lq {shape[1]} loaded (have {self.pair_shapes})")
            path = self._pair_paths[shape]
            pair = KevPairGraph(path, self.accelerator, self.precision, self.threads, self.handover,
                                self.constant_tensor_sharing)
            if pair.hidden_size != self.head.hidden_size or (pair.Ls, pair.Lq) != shape:
                pair.close()
                raise ValueError(f"{path.name}: Ls {pair.Ls} / Lq {pair.Lq} / d={pair.hidden_size}; expected "
                                 f"{shape} and the head's d={self.head.hidden_size}")
            self._pairs[shape] = pair
        return self._pairs[shape]

    def close_pair(self, shape: PairShape) -> None:
        pair = self._pairs.pop((int(shape[0]), int(shape[1])), None)
        if pair is not None:
            pair.close()

    def encode_rows(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        return encode_rows(self.tokenizer, request)

    def pick_L(self, row_len: int) -> int:
        return pick_L(row_len, self.lengths)

    def pick_pair(self, state_len: int, branch_lens: Sequence[int] = ()) -> Optional[PairShape]:
        """The pair for a request: the smallest loaded Ls that holds the state; among the pairs of that Ls, the smallest
        Lq that holds every branch, else the largest Lq. None when no loaded pair holds the state."""
        fitting = [s for s in self._pair_paths if s[0] >= state_len]
        if not fitting:
            return None
        Ls = min(s[0] for s in fitting)
        lqs = sorted(s[1] for s in fitting if s[0] == Ls)
        longest = max(branch_lens, default=0)
        return Ls, next((lq for lq in lqs if lq >= longest), lqs[-1])

    def route(self, enc: Dict[str, Any], mode: Optional[str] = None, length: Optional[int] = None) -> Dict[str, Any]:
        """Chooses each question's path (module docstring, step 5) and writes it into the encoded request:
        q["path"] = "row" or "pair", q["length"] = the row graph's length (None on the pair path), q["pair"] = (Ls, Lq)
        or None; enc["route"] = {"mode", "pair", "pair_questions", "row_questions", "row_positions", "pair_positions"}
        (the two positions are the R and P of mode "auto", None when no pair applies). `length` forces one row graph for
        every question (mode "row")."""
        mode = self.mode if mode is None else mode
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if length is not None:
            mode = "row"
        qs, n = enc["questions"], enc["state_tokens"]
        branch_lens = [len(q["row_ids"]) - n for q in qs]
        shape = self.pick_pair(n, branch_lens) if mode != "row" else None
        fits = [shape is not None and b <= shape[1] for b in branch_lens]
        row_positions = pair_positions = None
        if mode == "pair":
            if shape is None:
                raise PairDoesNotFit(f"the state is {n:,} tokens and no loaded pair holds it (have {self.pair_shapes})")
            if not all(fits):
                raise PairDoesNotFit(f"a question is {max(branch_lens):,} tokens; the pair Ls {shape[0]} holds "
                                     f"{shape[1]} per question")
            use_pair = fits
        elif mode == "auto" and any(fits):
            # R = the row graphs' lengths for the questions that fit the pair (a row no loaded graph holds = infinite),
            # P = the pair's positions for them; the pair takes them when R > pair_ratio * P
            lengths = sorted(self._paths)
            row_positions = 0.0
            for q, f in zip(qs, fits):
                if f:
                    row_positions += next((L for L in lengths if L >= len(q["row_ids"])), math.inf)
            pair_positions = shape[0] + sum(fits) * shape[1]
            use_pair = fits if row_positions > self.pair_ratio * pair_positions else [False] * len(qs)
        else:
            use_pair = [False] * len(qs)
        for q, on_pair in zip(qs, use_pair):
            if on_pair:
                q["path"], q["length"], q["pair"] = "pair", None, shape
                continue
            row_len = len(q["row_ids"])
            if length is None:
                q["length"] = self.pick_L(row_len)
            elif row_len > length:
                raise RowTooLong(f"a question row is {row_len:,} tokens; the {length}-token graph cannot hold it")
            else:
                q["length"] = length
            q["path"], q["pair"] = "row", None
        n_pair = sum(use_pair)
        enc["route"] = {"mode": mode, "pair": shape if n_pair else None, "pair_questions": n_pair,
                        "row_questions": len(qs) - n_pair, "row_positions": row_positions,
                        "pair_positions": pair_positions}
        return enc["route"]

    def score(self, request: Mapping[str, Any], length: Optional[int] = None, mode: Optional[str] = None) -> Dict[str, Any]:
        """encode_rows plus the route (`route`) and, per question, the readout: "z_pre", "z_post", "probs" (float32
        arrays) and "ms" (write, run, read back and readout of that question; on the pair path the state's run is
        enc["pair_state_ms"]). `length` forces one row graph for every question; `mode` overrides the host's mode."""
        enc = self.encode_rows(request)
        self.route(enc, mode=mode, length=length)
        for needed in sorted({q["length"] for q in enc["questions"] if q["path"] == "row"}):
            self.graph(needed)   # compile before the clock starts
        shape = enc["route"]["pair"]
        if shape is not None:
            self.pair(shape)
        started = time.perf_counter()
        if shape is not None:
            self.readout_pair(enc)
        for q in enc["questions"]:
            if q["path"] == "row":
                self.readout(q)
        enc["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return enc

    def readout(self, question: Dict[str, Any]) -> Dict[str, Any]:
        """Runs one encoded question's row through its graph (question["length"]) and the head; adds "z_pre",
        "z_post", "probs" (float32 arrays) and "ms" to it."""
        started = time.perf_counter()
        h_sel = select(self.graph(question["length"]).run(question["row_ids"]), question)
        if not np.isfinite(h_sel).all():
            raise NonFiniteOutput(f"question {question['id']!r}: non-finite hidden state from the {question['length']}-token "
                                  f"graph ({self.accelerator}, {self.precision}); on a GPU use precision fp32")
        question["z_pre"], question["z_post"], question["probs"] = self.head(h_sel)
        question["ms"] = (time.perf_counter() - started) * 1000
        return question

    def readout_pair(self, enc: Dict[str, Any]) -> Dict[str, Any]:
        """Runs the request's state once through its pair (enc["route"]["pair"]), then each question routed there
        through the question step and the head; adds "z_pre", "z_post", "probs" and "ms" to those questions and
        "pair_state_ms" to enc."""
        pair = self.pair(enc["route"]["pair"])
        n = enc["state_tokens"]
        qs = [q for q in enc["questions"] if q["path"] == "pair"]
        state = qs[0]["row_ids"][:n]
        started = time.perf_counter()
        pair.run_state(state)
        enc["pair_state_ms"] = (time.perf_counter() - started) * 1000
        for q in qs:
            started = time.perf_counter()
            if q["row_ids"][:n] != state:
                raise AssertionError("the rows of one request do not share the state")
            branch = {"decide_idx": q["decide_idx"] - n, "opt_idx": [o - n for o in q["opt_idx"]]}
            h_sel = select(pair.run_question(q["row_ids"][n:]), branch)
            if not np.isfinite(h_sel).all():
                raise NonFiniteOutput(f"question {q['id']!r}: non-finite hidden state from the pair Ls {pair.Ls} / Lq "
                                      f"{pair.Lq} ({self.accelerator}, {self.precision}); on a GPU use precision fp32")
            q["z_pre"], q["z_post"], q["probs"] = self.head(h_sel)
            q["ms"] = (time.perf_counter() - started) * 1000
        return enc

    def respond(self, scored: Mapping[str, Any]) -> Dict[str, Any]:
        """The /v1/systemone response body from a score() result."""
        answers = to_answers([q["probs"].tolist() for q in scored["questions"]], scored["questions"])
        return {"model": scored["model"], "answers": answers,
                "usage": {"input_tokens": scored["input_tokens"], "output_tokens": self.tokenizer.count(json.dumps(answers))},
                "latency_ms": scored["latency_ms"]}

    def decide(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        """The /v1/systemone response for one request: {"model", "answers", "usage", "latency_ms"}."""
        return self.respond(self.score(request))

    def close(self) -> None:
        for length in list(self._graphs):
            self.close_graph(length)
        for shape in list(self._pairs):
            self.close_pair(shape)

    def __enter__(self) -> "KevLiteRT":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _load_request(path: str, record_id: Optional[str]) -> Dict[str, Any]:
    """A request file: a bare request, a fixture record ({"id", "request", ...}) or a fixture file ({"records": [...]},
    pick one with --id)."""
    doc = json.loads(sys.stdin.read() if path == "-" else Path(path).read_text())
    if isinstance(doc, Mapping) and "records" in doc:
        if record_id is None:
            raise SystemExit("this file holds several records; pass --id")
        doc = next((r for r in doc["records"] if r.get("id") == record_id), None)
        if doc is None:
            raise SystemExit(f"no record {record_id!r}")
        if "request" not in doc:
            raise SystemExit(f"record {record_id!r} is listed by reference only; rebuild the full file with "
                             "fixtures/rebuild_requests.py and pass that file")
    if isinstance(doc, Mapping) and "request" in doc and "questions" not in doc:
        doc = doc["request"]
    if not isinstance(doc, Mapping) or "questions" not in doc:
        raise SystemExit("not a request: expected an object with 'state' and 'questions'")
    return doc


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Answer one /v1/systemone request with Kev on LiteRT.")
    ap.add_argument("--graph", nargs="+", required=True,
                    help="one or more .tflite files: row graphs (*_rowprefill_L<L>_*) and shared-state pairs "
                         "(*_sharedstate_Ls<Ls>_Lq<Lq>_*)")
    ap.add_argument("--head", required=True, help="head/kev_*_pointer_head.safetensors (its .json beside it gives T)")
    ap.add_argument("--tokenizer", required=True, help="the Kev repository's tokenizer.json (tokenizer/tokenizer.json)")
    ap.add_argument("--request", required=True, help="request JSON file ('-' = stdin)")
    ap.add_argument("--id", default=None, help="record id when --request is a fixture file")
    ap.add_argument("--accel", choices=("cpu", "gpu"), default="cpu")
    ap.add_argument("--precision", choices=("fp32", "fp16"), default="fp32",
                    help="GPU precision (fp32 keeps the probabilities within the test tolerance)")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads")
    ap.add_argument("--mode", choices=MODES, default="auto", help="route through the row graphs, the pairs, or both")
    ap.add_argument("--pair-ratio", type=float, default=PAIR_RATIO,
                    help="mode auto: a request's questions take a pair when their row graphs compute more than this "
                         "many times the pair's positions")
    ap.add_argument("--handover", choices=HANDOVERS, default="direct", help="how a pair passes its state")
    ap.add_argument("--no-constant-tensor-sharing", action="store_true",
                    help="GPU: hold a pair's weights once per signature (faster on a Mac, about twice the GPU memory)")
    a = ap.parse_args(argv)
    request = _load_request(a.request, a.id)
    with KevLiteRT(a.graph, a.head, a.tokenizer, accelerator=a.accel, precision=a.precision, threads=a.threads,
                   mode=a.mode, pair_ratio=a.pair_ratio, handover=a.handover,
                   constant_tensor_sharing=not a.no_constant_tensor_sharing) as kev:
        try:
            response = kev.decide(request)
        except RequestError as e:
            print(json.dumps({"status": e.status, "detail": str(e)}, ensure_ascii=False), file=sys.stderr)
            raise SystemExit(2)
        except NonFiniteOutput as e:
            print(json.dumps({"status": 500, "detail": str(e)}, ensure_ascii=False), file=sys.stderr)
            raise SystemExit(3)
    print(json.dumps(response, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
