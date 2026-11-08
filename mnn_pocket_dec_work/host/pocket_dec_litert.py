"""MNN-Pocket-0.3B-Dec on LiteRT: a Python host with the API of the original inference.py.

    from pocket_dec_litert import PocketDecisionLiteRT

    model = PocketDecisionLiteRT.from_pretrained(".", backend="cpu")    # or backend="gpu"
    result = model.predict(state, questions)                            # {"answers", "input_tokens", "temperature"}

The graphs this host runs, mnn-pocket-0.3b-dec_L{64,96,128,256,512,768,1024,2048}_fp16w.tflite, are the decoder of
taobao-mnn/MNN-Pocket-0.3B-Dec with its LoRA adapter folded in (the files named _fp16w_npu, _fp16w_npu_SM8850 and _fp16w_tensor
are for phone NPUs, and this host does not load them):

    embeds f32 [1, L, 896], cos f32 [1, L, 64], sin f32 [1, L, 64], mask f32 [1, 1, L, L] -> hidden f32 [1, L, 896]

where hidden is the output of the final RMSNorm at every position. This module does what the original inference.py
does around its decoder, in the same order:
  encode()   the request as inference.py serialises it (no chat template, the same token ids, positions and segments);
  inputs     the rows of the checkpoint's bf16 embedding table (embedding.safetensors), the rotary cos / sin of the
             original positions (every question branch restarts at the state length), and the additive attention mask
             (a key is visible when it is not later than the query and is in the query's question or in the state,
             and every position sees itself). Positions after the request, up to the graph length L, take the
             <|endoftext|> row, position 0 and only their own diagonal, so no request position reads them;
  graph      the smallest graph whose L holds the request, one CompiledModel per L, created on first use and kept;
  head       the pointer head in float32 (decision_head.safetensors): q = h_decide Wq^T + bq, k = h_option Wk^T + bk,
             logit = sum(q * k) / 16, read at each question's decide token and at the closing token of each option;
  answers    softmax(logit / T) in Python floats with T from temperature.json, winner = the first maximal logit.

Requests longer than the largest graph (2,048 tokens) raise ValueError; nothing is truncated. The GPU backend runs in
float32 (GpuOptions(enforce_f32=True)): the GPU's default 16-bit activations move the probabilities by more than 0.02.

Command line (the same arguments as inference.py, with --backend in place of --device / --dtype, and --threads for the CPU):
    python host/pocket_dec_litert.py [--model-dir DIR] [--input request.json | -] [--backend cpu|gpu] [--threads N] [--temperature T]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections.abc import Mapping
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np

CONTROL_TOKEN = re.compile(r"<\|([A-Za-z0-9_]+)\|>")
SPECIAL_TOKENS = ("<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>")
SPECIAL_IDS = (65526, 65527, 65515, 65516, 65528)      # decision_config.json of the original repository
PAD_TOKEN, PAD_ID = "<|endoftext|>", 65510
HIDDEN, HEAD_DIM, VOCAB = 896, 64, 65536
MASK_NEG = -30000.0                                     # additive mask value for a hidden key (exp underflows to 0)
GRAPH_FILE = re.compile(r"^mnn-pocket-0\.3b-dec_L(\d+)_fp16w\.tflite$")
TABLE_KEY = "model.embed_tokens.weight"
# Qwen3 rotary inverse frequencies of the original configuration (rope_theta 1e6, head_dim 64), the float32 values
# transformers computes, written as hex so that the host gives the same cos / sin inputs on every platform.
INV_FREQ_HEX = (
    "0x1.0000000000000p+0", "0x1.4c7bc00000000p-1", "0x1.afd1360000000p-2", "0x1.186a080000000p-2",
    "0x1.6c310e0000000p-3", "0x1.d8ffac0000000p-4", "0x1.33281a0000000p-4", "0x1.8eec7e0000000p-5",
    "0x1.030dc60000000p-5", "0x1.5073160000000p-6", "0x1.b4f7e40000000p-7", "0x1.1bc25a0000000p-7",
    "0x1.7089380000000p-8", "0x1.dea41c0000000p-9", "0x1.36d21a0000000p-9", "0x1.93aeb80000000p-10",
    "0x1.0624de0000000p-10", "0x1.5476880000000p-11", "0x1.ba2e4a0000000p-12", "0x1.1f24e40000000p-12",
    "0x1.74eea60000000p-13", "0x1.e459c40000000p-14", "0x1.3a87480000000p-14", "0x1.987f7c0000000p-15",
    "0x1.0945640000000p-15", "0x1.58863a0000000p-16", "0x1.bf749e0000000p-17", "0x1.2291c60000000p-17",
    "0x1.7961820000000p-18", "0x1.ea20e00000000p-19", "0x1.3e47c80000000p-19", "0x1.9d5ef20000000p-20",
)


# -- request handling: render, _json_value and _question are inference.py's functions, unchanged --

def render(value: Any, indent: int = 0) -> str:
    """Preserve the training serializer, including insertion order and booleans."""
    pad = "  " * indent
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(f"{pad}- {render(item, indent + 1).lstrip()}" for item in value)
    if isinstance(value, dict):
        return "\n".join(
            f"{pad}{key}:\n{render(item, indent + 1)}"
            if isinstance(item, (dict, list)) else f"{pad}{key}: {render(item)}"
            for key, item in value.items()
        )
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _json_value(value: Any) -> None:
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        for item in value.values():
            _json_value(item)
    elif isinstance(value, list):
        for item in value:
            _json_value(item)
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise TypeError(f"unsupported JSON value: {type(value).__name__}")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON numbers must be finite")


def _question(question_id: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"question {question_id!r} must be an object")
    kind, criteria = value.get("type"), value.get("criteria")
    instructions = value.get("instructions")
    _json_value(instructions)
    _json_value(criteria)

    def option(key: str, description: Any) -> str:
        return key if description is None or description == "" else key + ": " + render(description)

    if kind == "choice":
        if not isinstance(criteria, dict) or not criteria:
            raise ValueError(f"question {question_id!r}: choice criteria must be a nonempty object")
        keys = list(criteria)
        options = [option(key, description) for key, description in criteria.items()]
    elif kind == "noul":
        if criteria is not None and (
            not isinstance(criteria, dict) or not set(criteria).issubset({"false", "true"})
        ):
            raise ValueError(f"question {question_id!r}: noul criteria may contain only false and true")
        criteria = criteria or {}
        keys = ["false", "true"]
        options = [option("no", criteria.get("false")), option("yes", criteria.get("true"))]
    elif kind == "score":
        if not isinstance(criteria, list) or not criteria:
            raise ValueError(f"question {question_id!r}: score criteria must be a nonempty ordered list")
        keys = [str(index) for index in range(len(criteria))]
        options = [render(description) for description in criteria]
    else:
        raise ValueError(f"question {question_id!r}: type must be choice, noul or score")
    if not 1 <= len(keys) <= 255:
        raise ValueError(f"question {question_id!r}: expected 1..255 candidates")
    return {"id": question_id, "type": kind, "keys": keys, "options": options,
            "instructions": render(instructions)}


def _temperature(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("temperature must be a finite positive number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("temperature must be a finite positive number")
    return value


class _Graph:
    """One graph file on one backend: the CompiledModel and its input / output buffers, made once."""

    def __init__(self, path: str, L: int, backend: str, num_threads: int):
        from ai_edge_litert.compiled_model import CompiledModel, CpuOptions, GpuOptions, HardwareAccelerator, Options
        if backend == "cpu":
            options = Options(hardware_accelerators=HardwareAccelerator.CPU,
                              cpu_options=CpuOptions(num_threads=num_threads))
        else:
            options = Options(hardware_accelerators=HardwareAccelerator.GPU,
                              gpu_options=GpuOptions(enforce_f32=True))
        self.L = L
        self.model = CompiledModel.from_file(path, options=options)
        signatures = self.model.get_signature_list()
        if len(signatures) != 1:
            raise ValueError(f"{os.path.basename(path)}: expected one signature, found {len(signatures)}")
        self.key = next(iter(signatures))
        inputs = self.model.get_input_tensor_details(self.key)
        outputs = self.model.get_output_tensor_details(self.key)
        if sorted(inputs) != ["cos", "embeds", "mask", "sin"] or list(outputs) != ["hidden"]:
            raise ValueError(f"{os.path.basename(path)}: unexpected signature {sorted(inputs)} -> {list(outputs)}")
        self.inputs = {name: self.model.create_input_buffer_by_name(self.key, name) for name in inputs}
        self.outputs = {name: self.model.create_output_buffer_by_name(self.key, name) for name in outputs}

    def run(self, feeds: dict[str, np.ndarray]) -> np.ndarray:
        for name, array in feeds.items():
            self.inputs[name].write(np.ascontiguousarray(array, dtype=np.float32))
        self.model.run_by_name(self.key, self.inputs, self.outputs)
        hidden = self.outputs["hidden"].read(self.L * HIDDEN, np.float32)
        return np.asarray(hidden, dtype=np.float32).reshape(self.L, HIDDEN)


class PocketDecisionLiteRT:
    """The decision model on LiteRT graphs, with the request handling and answers of the original inference.py."""

    def __init__(self, root: str, backend: str = "cpu", num_threads: int = 4, max_length: int | None = None):
        """from_pretrained() is the normal entry. max_length is for conversion/export_dec.py, which encodes requests
        before any graph file exists."""
        import ml_dtypes  # noqa: F401  (registers numpy's bfloat16 dtype; safetensors needs it for the embedding table)
        from safetensors.numpy import load_file
        from tokenizers import Tokenizer

        if backend not in ("cpu", "gpu"):
            raise ValueError("backend must be cpu or gpu")
        if isinstance(num_threads, bool) or not isinstance(num_threads, int) or num_threads < 0:
            raise ValueError("num_threads must be a non-negative integer (0 = one thread, the runtime's default)")
        self.root, self.backend, self.num_threads = root, backend, num_threads

        def local_path(name: str) -> str:
            path = os.path.join(root, name)
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            return path

        self.tokenizer = Tokenizer.from_file(local_path("tokenizer.json"))
        if [self.tokenizer.token_to_id(token) for token in SPECIAL_TOKENS] != list(SPECIAL_IDS) or \
                self.tokenizer.token_to_id(PAD_TOKEN) != PAD_ID:
            raise ValueError("tokenizer control-token IDs do not match the checkpoint")
        table = load_file(local_path("embedding.safetensors"))[TABLE_KEY]
        if table.shape != (VOCAB, HIDDEN) or table.dtype.name != "bfloat16":
            raise ValueError(f"embedding.safetensors: expected bfloat16 {(VOCAB, HIDDEN)}, got {table.dtype} {table.shape}")
        self.table = table.astype(np.float32)               # exact: every bfloat16 value is a float32 value
        head = load_file(local_path("decision_head.safetensors"))
        self.wq, self.bq = head["q.weight"].astype(np.float32), head["q.bias"].astype(np.float32)
        self.wk, self.bk = head["k.weight"].astype(np.float32), head["k.bias"].astype(np.float32)
        if self.wq.shape[1] != HIDDEN or self.wk.shape != self.wq.shape:
            raise ValueError("decision_head.safetensors does not match the hidden size")
        self.scale = 1 / math.sqrt(self.wq.shape[0])
        with open(local_path("temperature.json"), encoding="utf-8") as f:
            self.temperature = _temperature(json.load(f)["temperature"])
        self.inv_freq = np.array([float.fromhex(x) for x in INV_FREQ_HEX], dtype=np.float32)
        self.graph_files = {}
        for name in sorted(os.listdir(root)):
            match = GRAPH_FILE.match(name)
            if match:
                self.graph_files[int(match.group(1))] = os.path.join(root, name)
        if not self.graph_files and max_length is None:
            raise FileNotFoundError(f"no mnn-pocket-0.3b-dec_L<L>_fp16w.tflite graph in {root}")
        self.max_length = max(self.graph_files) if max_length is None else max_length
        self.graphs: dict[int, _Graph] = {}

    @classmethod
    def from_pretrained(cls, model_dir: str | Path = ".", backend: str = "cpu", num_threads: int = 4):
        """Load the files of a local copy of this repository; no Hub lookup. backend: "cpu" (XNNPACK, num_threads
        threads) or "gpu" (float32). Graphs are compiled on first use, one per length."""
        root = os.path.abspath(os.path.expanduser(str(model_dir)))
        return cls(root, backend, num_threads)

    # -- encode: inference.py's PocketDecisionModel.encode, with this host's length limit --
    def _tokens(self, text: str) -> list[int]:
        escaped = CONTROL_TOKEN.sub(r"<¦\1¦>", text)
        return self.tokenizer.encode(escaped, add_special_tokens=False).ids

    def encode(self, state: Any, questions: Mapping[str, Any]) -> dict[str, Any]:
        """Encode state and independent question branches without a chat template."""
        _json_value(state)
        if not isinstance(questions, dict) or not questions:
            raise ValueError("questions must be a nonempty object")
        if any(not isinstance(key, str) or not key for key in questions):
            raise ValueError("question IDs must be nonempty strings")
        state_id, question_id, option_id, close_id, decide_id = SPECIAL_IDS
        ids = [state_id] + self._tokens(render(state))
        segments = [0] * len(ids)
        positions = list(range(len(ids)))
        state_length = len(ids)
        encoded_questions = []
        decide_positions, option_positions = [], []
        for segment, (key, value) in enumerate(questions.items(), 1):
            question = _question(key, value)
            start = len(ids)
            branch = [question_id] + self._tokens(question["instructions"])
            ends = []
            for option in question["options"]:
                branch += [option_id] + self._tokens(option) + [close_id]
                ends.append(start + len(branch) - 1)
            branch.append(decide_id)
            decide_positions.append(start + len(branch) - 1)
            option_positions.append(ends)
            ids += branch
            segments += [segment] * len(branch)
            positions += list(range(state_length, state_length + len(branch)))
            encoded_questions.append(question)
        if len(ids) > self.max_length:
            raise ValueError(f"input has {len(ids)} tokens; maximum is {self.max_length}; no truncation")
        return {"ids": ids, "seg": segments, "pos": positions, "decide_idx": decide_positions,
                "opt_idx": option_positions, "questions": encoded_questions}

    # -- graph inputs --
    def _feeds(self, encoded: dict[str, Any], L: int) -> dict[str, np.ndarray]:
        n = len(encoded["ids"])
        ids = np.full(L, PAD_ID, dtype=np.int64)
        ids[:n] = encoded["ids"]
        pos = np.zeros(L, dtype=np.int64)
        pos[:n] = encoded["pos"]
        freqs = pos.astype(np.float32)[:, None] * self.inv_freq[None, :]            # one float32 product each
        angles = np.concatenate([freqs, freqs], axis=-1).astype(np.float64)
        seg = np.asarray(encoded["seg"])
        index = np.arange(n)
        allow = (index[None, :] <= index[:, None]) & ((seg[None, :] == seg[:, None]) | (seg[None, :] == 0))
        allow |= np.eye(n, dtype=bool)
        mask = np.full((L, L), MASK_NEG, dtype=np.float32)
        mask[:n, :n] = np.where(allow, np.float32(0.0), np.float32(MASK_NEG))
        pad = np.arange(n, L)
        mask[pad, pad] = 0.0
        return {"embeds": self.table[ids][None], "cos": np.cos(angles).astype(np.float32)[None],
                "sin": np.sin(angles).astype(np.float32)[None], "mask": mask[None, None]}

    def _graph(self, n: int) -> _Graph:
        L = min(length for length in self.graph_files if length >= n)
        if L not in self.graphs:
            self.graphs[L] = _Graph(self.graph_files[L], L, self.backend, self.num_threads)
        return self.graphs[L]

    def _head(self, hidden: np.ndarray, encoded: dict[str, Any]) -> list[list[float]]:
        logits = []
        for decide, ends in zip(encoded["decide_idx"], encoded["opt_idx"]):
            q = hidden[decide] @ self.wq.T + self.bq
            k = hidden[ends] @ self.wk.T + self.bk
            logits.append(((k * q[None, :]).sum(-1) * np.float32(self.scale)).astype(np.float32).tolist())
        return logits

    def _answers(self, encoded: dict[str, Any], logits: list[list[float]], temperature: float) -> dict[str, Any]:
        answers = {}
        for question, scores in zip(encoded["questions"], logits):
            if not all(math.isfinite(value) for value in scores):
                raise RuntimeError("non-finite decision logits; check the selected backend")
            maximum = max(scores)
            weights = [math.exp((value - maximum) / temperature) for value in scores]
            total = sum(weights)
            probabilities = [value / total for value in weights]
            winner = max(range(len(scores)), key=scores.__getitem__)
            answer = {"type": question["type"], "probabilities": dict(zip(question["keys"], probabilities))}
            if question["type"] == "choice":
                answer["choice"] = question["keys"][winner]
            elif question["type"] == "noul":
                answer.update(noul=probabilities[1], decision=bool(winner))
            else:
                answer.update(score=sum(index * value for index, value in enumerate(probabilities)),
                              legend=dict(zip(question["keys"], question["options"])))
            answers[question["id"]] = answer
        return answers

    def predict(self, state: Any, questions: Mapping[str, Any], temperature=None) -> dict[str, Any]:
        temperature = self.temperature if temperature is None else _temperature(temperature)
        encoded = self.encode(state, questions)
        graph = self._graph(len(encoded["ids"]))
        hidden = graph.run(self._feeds(encoded, graph.L))
        answers = self._answers(encoded, self._head(hidden, encoded), temperature)
        return {"answers": answers, "input_tokens": len(encoded["ids"]), "temperature": temperature}


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="MNN-Pocket-0.3B-Dec on LiteRT (the request JSON of inference.py)")
    parser.add_argument("--model-dir", default=root, help="a local copy of this repository (default: the one holding "
                                                          "this file)")
    parser.add_argument("--input", default=None, help="request JSON file, or - for stdin; omit to run "
                                                      "examples/default_request.json")
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads (default 4; 0 = one thread, the runtime's default)")
    parser.add_argument("--temperature", type=float, default=None)
    args = parser.parse_args()
    try:
        if args.input is None:
            with open(os.path.join(root, "examples", "default_request.json"), encoding="utf-8") as f:
                request = json.load(f)
        elif args.input == "-":
            request = json.load(sys.stdin)
        else:
            with open(args.input, encoding="utf-8") as f:
                request = json.load(f)
        if not isinstance(request, dict) or "state" not in request or "questions" not in request:
            raise ValueError("request must contain state and questions")
        with redirect_stdout(sys.stderr):
            model = PocketDecisionLiteRT.from_pretrained(args.model_dir, args.backend, args.threads)
            result = model.predict(request["state"], request["questions"], args.temperature)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, TypeError, OSError, RuntimeError, ImportError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
