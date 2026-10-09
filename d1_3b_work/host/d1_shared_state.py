"""d1-3B on LiteRT with the shared-state pair: the host of host/d1_litert.py (imported, not changed) with a second
kind of graph, for requests of several questions.

One file holds two signatures that share the weights:

    state_prefill_<Ls>(embeds float32 [1, Ls, 2048], valid float32 [1, Ls]) -> 38 state tensors
        conv_tail_<l> float32 [1, 2048, 2]  the 22 ShortConv layers' conv input at the last 2 real state tokens
        k_<l>, v_<l>  float32 [1, 8, Ls, 64]  the 8 attention layers' keys (after k_layernorm and RoPE) and values
    question_step_<Ls>_<Lq>(embeds float32 [1, Lq, 2048], valid float32 [1, Lq], state_valid float32 [1, Ls],
                            the 38 tensors) -> hidden float32 [1, Lq, 2048]   (after the final RMSNorm, at the
                            question's positions)
(an earlier form of the pair took `ids` int32 in place of `embeds`; SharedStatePair reads the kind from the file)

A request runs its state once, then one question_step per question; positions continue after the state inside the
graph, so the hidden states are the row form's (state + question as one causal row) up to float rounding.

The request contract, render, tokenizer, read-out and answers are host/d1_litert.py's (`D1Host`). The rows of a request
are D1Host.rows (each question's whole row, encode(prefix + suffix)); the state is the leading `state_len` tokens
(= encode(prefix)) and must be the same in every row; a question's own tokens are the rest of its row; its answer slot
is its last token. On the 415 fixture rows encode(prefix) + encode(suffix) equals the row, so these are also the
provider's trunk and branch ids (its multi-question path).

pick (which graph a request runs on), `D1SharedHost.route`:
- pictures -> the row form (D1Host);
- `call_ms` (the rule of contract.json `shared_state.pick.call_ms`, measured on the Mac Metal fp32):
  the form with the smallest expected time, the rows at the sum of their row graphs' call times (each row in the
  smallest loaded bucket that holds it) against each pair that holds the request (state <= Ls, every question's own
  tokens <= Lq) at one state call + one question call per question; ties go to the rows. With the L128 row graph the
  rows' cost depends on their bucket, which a question count alone does not see;
- `min_questions` (without call_ms): the smallest pair (by Ls, then Lq) whose Ls holds the state, whose Lq holds
  every question's own tokens and whose `min_questions` the request reaches (one number for every pair, or {(Ls, Lq):
  number} per pair);
- no pair holds it, or the rows do not share the state's tokens -> the row form.
The pairs of this repository are embeds pairs (both signatures take `embeds` = the float32 rows of
tables/embed_table.safetensors at the padded ids, no table in the file), at Ls64+Lq64, Ls128+Lq64 and Ls256+Lq128;
give SharedStatePair the host's EmbedTable (`embed_table`).

Hand-over of the state between the two signatures: "direct" passes state_prefill's output TensorBuffers as
question_step's inputs (no copy through the host); "host" reads them back to numpy and writes them into question_step's
input buffers (both give the same bytes on the Mac CPU and Metal). On the GPU,
GpuOptions(constant_tensor_sharing=True) lets the two signatures use one copy of the weights; without it the GPU holds
them once per signature. Measured on the Mac Metal fp32 with the earlier pairs (an int8 table inside the graph):
sharing made the calls 1.35-2.0x slower (Ls128+Lq64: state 97.4 vs 60.4 ms, question 77.0 vs 38.8 ms) and halved the
memory (11.2 vs 23.7 GB phys_footprint); both passed the tolerance. The default here is no sharing; share=True when
memory matters.

    # examples/run_example.py load_host() builds D1SharedHost on this repository's files from contract.json
    host.decide({"state": ..., "questions": {...}})
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from d1_litert import (OUTPUT_NAME, PAD_ID, D1Host, RequestError, Row, RowTooLong, answer, readout)

HANDOVERS = ("direct", "host", "both")


class SharedStatePair:
    """The pair file on the CompiledModel API. handover "direct" / "host" (above) or "both" (the state is read back
    and written for the host path as well; run_question then takes the hand-over per call).

    The embeds pair (the form of this repository's files): both signatures take `embeds` float32 [1, L, d] in place of
    `ids`
    (no table in the file); `embed_table` (d1_litert.EmbedTable on embed_table.safetensors) gives the float32 rows of
    the padded ids, the pad id's row on the pads, inside run_state / run_question. The pair's kind is read from the
    file's signatures (`input` = "ids" | "embeds")."""

    def __init__(self, path: str | Path, accelerator: str = "cpu", precision: str = "fp32", threads: int = 4,
                 handover: str = "direct", share: bool = False, pad_id: int = PAD_ID, embed_table=None):
        from ai_edge_litert.compiled_model import (CompiledModel, CpuOptions, GpuOptions, HardwareAccelerator,
                                                   Options)

        if handover not in HANDOVERS:
            raise ValueError(f"handover must be one of {HANDOVERS}")
        self.path, self.handover, self.pad_id = Path(path), handover, pad_id
        if not self.path.is_file():
            raise FileNotFoundError(f"no pair file at {self.path}")
        if accelerator == "gpu":
            if precision not in ("fp32", "default"):
                raise ValueError("precision must be 'fp32' or 'default'")
            gpu = GpuOptions(enforce_f32=precision == "fp32", constant_tensor_sharing=share)
            options = Options(hardware_accelerators=HardwareAccelerator.GPU, gpu_options=gpu)
            self.options_desc = {"accelerator": "GPU (Metal)", "gpu_options": gpu._as_flat_kwargs()}
        elif accelerator == "cpu":
            options = Options(hardware_accelerators=HardwareAccelerator.CPU, cpu_options=CpuOptions(num_threads=threads))
            self.options_desc = {"accelerator": "CPU", "threads": threads}
        else:
            raise ValueError("accelerator must be 'cpu' or 'gpu'")
        started = time.perf_counter()
        self.model = CompiledModel.from_file(str(self.path), options=options)
        self.compile_seconds = time.perf_counter() - started
        sigs = self.model.get_signature_list()
        state = [k for k in sigs if re.fullmatch(r"state_prefill_\d+", k)]
        question = [k for k in sigs if re.fullmatch(r"question_step_\d+_\d+", k)]
        if len(state) != 1 or len(question) != 1 or len(sigs) != 2:
            self.model.close()
            raise ValueError(f"{self.path.name}: expected state_prefill_<Ls> and question_step_<Ls>_<Lq>, got {list(sigs)}")
        self.sig_state, self.sig_question = state[0], question[0]
        self.Ls, self.Lq = (int(x) for x in self.sig_question.split("_")[2:])
        if int(self.sig_state.split("_")[2]) != self.Ls:
            self.model.close()
            raise ValueError("state_prefill and question_step disagree on Ls")
        self.names = list(sigs[self.sig_state]["outputs"])
        self.input = "embeds" if "embeds" in sigs[self.sig_state]["inputs"] else "ids"
        if sorted(sigs[self.sig_state]["inputs"]) != sorted([self.input, "valid"]) or \
                sorted(sigs[self.sig_question]["inputs"]) != sorted([self.input, "valid", "state_valid"] + self.names):
            self.model.close()
            raise ValueError("question_step does not take state_prefill's outputs")
        if self.input == "embeds" and embed_table is None:
            self.model.close()
            raise ValueError(f"{self.path.name} is an embeds pair: it needs the embed table")
        self.embed_table = embed_table if self.input == "embeds" else None
        det = self.model.get_output_tensor_details(self.sig_state)
        self.numel = {n: int(np.prod(list(det[n]["shape"]))) for n in self.names}
        self.shapes = {n: [int(x) for x in det[n]["shape"]] for n in self.names}
        self.state_bytes = 4 * sum(self.numel.values())
        hidden = list(self.model.get_output_tensor_details(self.sig_question)[OUTPUT_NAME]["shape"])
        self.hidden_size = int(hidden[2])
        if hidden != [1, self.Lq, self.hidden_size]:
            self.model.close()
            raise ValueError(f"question_step returns {hidden}")
        m = self.model
        self.in_state = {n: m.create_input_buffer_by_name(self.sig_state, n) for n in (self.input, "valid")}
        self.out_state = {n: m.create_output_buffer_by_name(self.sig_state, n) for n in self.names}
        self.in_question = {n: m.create_input_buffer_by_name(self.sig_question, n) for n in sigs[self.sig_question]["inputs"]}
        self.out_question = {OUTPUT_NAME: m.create_output_buffer_by_name(self.sig_question, OUTPUT_NAME)}
        self.direct = {**{n: self.in_question[n] for n in (self.input, "valid", "state_valid")},
                       **{n: self.out_state[n] for n in self.names}}
        try:
            self.fully_accelerated = bool(m.is_fully_accelerated())
        except Exception:   # informational only
            self.fully_accelerated = None

    def _padded(self, ids: Sequence[int], length: int):
        a = np.full((1, length), self.pad_id, dtype=np.int32)
        v = np.zeros((1, length), dtype=np.float32)
        if len(ids):
            a[0, :len(ids)] = np.asarray(ids, dtype=np.int32)
            v[0, :len(ids)] = 1.0
        return a, v

    def _first(self, ids: np.ndarray) -> np.ndarray:
        """The input for padded ids [1, L]: the ids, or (embeds pair) the table's float32 rows [1, L, d]."""
        if self.embed_table is None:
            return ids
        return np.ascontiguousarray(self.embed_table.rows(ids[0])[None], dtype=np.float32)

    def run_state(self, state_ids: Sequence[int], read_back: Optional[bool] = None) -> Optional[Dict[str, np.ndarray]]:
        """Run state_prefill on the state's tokens. read_back (default: handover host / both) reads the state back and
        writes it into question_step's own inputs (the host hand-over) and returns it; else None."""
        if len(state_ids) > self.Ls:
            raise RowTooLong(f"the state is {len(state_ids):,} tokens; this pair holds {self.Ls}")
        if read_back is None:
            read_back = self.handover in ("host", "both")
        ids, valid = self._padded(state_ids, self.Ls)
        self.in_state[self.input].write(self._first(ids))
        self.in_state["valid"].write(valid)
        self.model.run_by_name(self.sig_state, self.in_state, self.out_state)
        arrays = None
        if read_back:
            arrays = {n: np.asarray(self.out_state[n].read(self.numel[n], np.float32), dtype=np.float32)
                      .reshape(self.shapes[n]) for n in self.names}
            for n in self.names:
                self.in_question[n].write(np.ascontiguousarray(arrays[n]))
        self.in_question["state_valid"].write(valid)
        self._host_ready = bool(read_back)
        return arrays

    def run_question(self, own_ids: Sequence[int], handover: Optional[str] = None) -> np.ndarray:
        """Hidden states [Lq, d] of one question's own tokens after the last run_state."""
        mode = handover or self.handover
        if mode == "both" or (mode == "host" and not getattr(self, "_host_ready", False)):
            raise ValueError(f"hand-over {mode!r}: the last run_state did not hand the state over through the host")
        if not 0 < len(own_ids) <= self.Lq:
            raise RowTooLong(f"a question is {len(own_ids):,} tokens; this pair holds 1..{self.Lq}")
        ids, valid = self._padded(own_ids, self.Lq)
        self.in_question[self.input].write(self._first(ids))
        self.in_question["valid"].write(valid)
        self.model.run_by_name(self.sig_question, self.direct if mode == "direct" else self.in_question,
                               self.out_question)
        out = self.out_question[OUTPUT_NAME].read(self.Lq * self.hidden_size, np.float32)
        return np.asarray(out, dtype=np.float32).reshape(self.Lq, self.hidden_size)

    def close(self) -> None:
        for b in [*self.in_state.values(), *self.out_state.values(), *self.in_question.values(),
                  *self.out_question.values()]:
            try:
                b.destroy()
            except Exception:
                pass
        self.in_state, self.out_state, self.in_question, self.out_question, self.direct = {}, {}, {}, {}, {}
        self.model.close()


class D1SharedHost:
    """D1Host's answers, with the requests that a pair holds run on that pair (module docstring, pick).

    Two pick rules: `call_ms` (the rule of contract.json) = {"row": {L: ms per row call}, "pair": {(Ls, Lq): (ms of
    the state call, ms per question call)}}, measured on the target accelerator (contract.json `shared_state.pick`
    holds the Mac Metal fp32 medians): a request takes the form with the smallest expected time, the rows at
    sum(row[L(row)]) over its questions (L = the smallest loaded bucket that holds the row) against each pair that holds
    it at state + n x question, ties to the rows; a bucket or pair missing from the table is not chosen. Else
    `min_questions`: the smallest holding pair whose question count reaches its min_questions."""

    def __init__(self, host: D1Host, pairs: Sequence[SharedStatePair] = (),
                 min_questions: int | Mapping[tuple, int] = 2, call_ms: Optional[Mapping] = None):
        self.host, self.min_questions, self.call_ms = host, min_questions, call_ms
        self.pairs = sorted(pairs, key=lambda p: (p.Ls, p.Lq))
        if call_ms is not None:
            missing = [(p.Ls, p.Lq) for p in self.pairs if (p.Ls, p.Lq) not in call_ms["pair"]]
            if missing:
                raise ValueError(f"call_ms has no entry for the pairs {missing}")
        elif isinstance(min_questions, Mapping):
            missing = [(p.Ls, p.Lq) for p in self.pairs if (p.Ls, p.Lq) not in min_questions]
            if missing:
                raise ValueError(f"min_questions has no entry for the pairs {missing}")
        for p in self.pairs:
            if host.table is not None and host.table.rows.shape[1] != p.hidden_size:
                raise ValueError(f"{p.path.name} returns d={p.hidden_size}; the read-out table has "
                                 f"d={host.table.rows.shape[1]}")
            if p.handover == "both":
                raise ValueError("a host pair hands over 'direct' or 'host'")
            if p.pad_id != host.pad_id:
                raise ValueError(f"{p.path.name} pads with {p.pad_id}, the host with {host.pad_id}")

    def min_questions_for(self, pair: SharedStatePair) -> int:
        m = self.min_questions
        return int(m[(pair.Ls, pair.Lq)]) if isinstance(m, Mapping) else int(m)

    def _pick(self, rows: Sequence[Row]) -> tuple[Optional[SharedStatePair], str]:
        if not rows:
            return None, "no questions"
        if rows[0].pictures is not None:
            return None, "pictures: the embeds row graphs (d1_litert item 9)"
        n = rows[0].state_len
        state = rows[0].ids[:n]
        if any(r.state_len != n or r.ids[:n] != state or len(r.ids) <= n for r in rows):
            return None, "the rows do not share the state's tokens"
        longest = max(len(r.ids) - n for r in rows)
        holding = [p for p in self.pairs if n <= p.Ls and longest <= p.Lq]
        if not holding:
            return None, f"no pair holds a state of {n} tokens and questions up to {longest} tokens"
        if self.call_ms is not None:
            return self._pick_by_time(rows, holding, n, longest)
        pair = next((p for p in holding if len(rows) >= self.min_questions_for(p)), None)
        if pair is None:
            p = holding[0]
            return None, (f"{len(rows)} question(s): fewer than {self.min_questions_for(p)} for the Ls{p.Ls}+Lq{p.Lq} "
                          "pair")
        return pair, (f"state {n} <= Ls {pair.Ls}, questions <= {longest} <= Lq {pair.Lq}, {len(rows)} question(s) >= "
                      f"{self.min_questions_for(pair)}")

    def _pick_by_time(self, rows, holding, n, longest) -> tuple[Optional[SharedStatePair], str]:
        """call_ms rule (class docstring)."""
        from d1_litert import pick_L

        row_ms, k = self.call_ms["row"], len(rows)
        try:
            Ls_rows = [pick_L(len(r.ids), self.host.row_buckets()) for r in rows]
            t_rows = sum(float(row_ms[L]) for L in Ls_rows)
        except (KeyError, RowTooLong):
            t_rows = float("inf")
        cand = [(float(self.call_ms["pair"][(p.Ls, p.Lq)][0]) + k * float(self.call_ms["pair"][(p.Ls, p.Lq)][1]), p)
                for p in holding]
        t_pair, pair = min(cand, key=lambda x: (x[0], x[1].Ls, x[1].Lq))
        rows_txt = "rows " + "+".join(f"L{L}" for L in Ls_rows) if t_rows < float("inf") else "rows (no timed bucket)"
        if t_pair < t_rows:
            return pair, (f"{k} question(s), state {n}, questions <= {longest}: pair Ls{pair.Ls}+Lq{pair.Lq} "
                          f"{t_pair:.1f} ms < {rows_txt} {t_rows:.1f} ms")
        return None, (f"{k} question(s), state {n}, questions <= {longest}: {rows_txt} {t_rows:.1f} ms <= pair "
                      f"Ls{pair.Ls}+Lq{pair.Lq} {t_pair:.1f} ms")

    def route(self, request: Mapping) -> Dict[str, Any]:
        """{"route": "pair" | "row", "Ls", "Lq" (pair), "why"} for a request (renders it; runs nothing)."""
        pair, why = self._pick(self.host.rows(request))
        if pair is None:
            return {"route": "row", "why": why}
        return {"route": "pair", "Ls": pair.Ls, "Lq": pair.Lq, "file": pair.path.name, "why": why}

    def decide(self, request: Mapping) -> dict:
        if self.host.table is None:
            raise RequestError("no read-out table loaded")
        rows = self.host.rows(request)
        pair, _ = self._pick(rows)
        if pair is None:
            return self.host.decide(request)
        n = rows[0].state_len
        pair.run_state(rows[0].ids[:n])
        answers = {}
        for r in rows:
            own = r.ids[n:]
            h = pair.run_question(own)[len(own) - 1]
            if not np.all(np.isfinite(h)):
                raise FloatingPointError("non-finite hidden state at the answer slot")
            answers[r.name] = answer(r.question, readout(h, self.host.table, r.groups))
        return {"answers": answers, "usage": {"input_tokens": self.host.input_tokens(request, rows), "output_tokens": 0}}

    def close(self) -> None:
        for p in self.pairs:
            p.close()
