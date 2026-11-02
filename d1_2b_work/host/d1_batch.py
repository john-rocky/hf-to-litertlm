"""The batch path of the PyTorch-free host: several requests' sequences through ONE decide call of a batch graph
(signature decide_b<B>_L<L>: embeds / sees_q / sees_k [B, L, ...] -> hidden [B, L, 2048]), as the provider's own
engine runs a batch of sequences in one forward.

  decide = LiteRTDecideBatch("d1-2b_decide_b8_L128_fp16fc.tflite", accelerator="gpu")
  host = D1HostBatched(SNAP, decide, tower, embed_table=...)
  host.system_one_batch([(state, questions), ...])      # 64 requests -> their sequences -> 8 calls of 8 rows

D1HostBatched = d1_host.D1Host with `_padded` computing the hidden states of a provider batch (`_batches`: the
sequences of the requests together, under the token budget) through `decide.many([(embeds, state_len, question), ...])`
instead of one `decide(embeds, state_len, question)` per sequence; the slot reading after it is D1Host's code line for
line (the provider's batch read at once). The batch backend takes up to B sequences a call, each right-padded to L;
the slots past the last sequence are padding rows (embeds 0, sees_q [0, 1, 0..], sees_k 0) whose outputs are never
read. A plain `decide(embeds, state_len, question)` call (one sequence) also works (a batch of one + padding rows), so
the code paths that call it stay valid. numpy only (no torch).
"""
from __future__ import annotations

import os
import sys
import time
from typing import Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from d1_host import D1Host, spans  # noqa: E402


class LiteRTDecideBatch:
    """One batch graph file (signature decide_b<B>_L<L>) on one CompiledModel, buffers made once."""

    def __init__(self, path: str, accelerator: str = "gpu", precision: str = "fp32", threads: int = 8):
        from lrt_backend import compiled_options   # the CPU threads / GPU float32 options
        from ai_edge_litert.compiled_model import CompiledModel

        self.path, self.accelerator, self.precision = path, accelerator, precision
        t0 = time.time()
        self.model = CompiledModel.from_file(path, options=compiled_options(accelerator, precision, threads))
        sigs = self.model.get_signature_list()
        assert len(sigs) == 1, sigs
        self.signature = next(iter(sigs))
        s = sigs[self.signature]
        self.in_names, self.out_names = list(s["inputs"]), list(s["outputs"])
        self.in_shapes = {n: list(self.model.get_input_tensor_details(self.signature)[n]["shape"]) for n in self.in_names}
        self.out_shapes = {n: list(self.model.get_output_tensor_details(self.signature)[n]["shape"])
                           for n in self.out_names}
        self.B, self.L, self.hidden = self.in_shapes["embeds"]
        self.inputs = {n: self.model.create_input_buffer_by_name(self.signature, n) for n in self.in_names}
        self.outputs = {n: self.model.create_output_buffer_by_name(self.signature, n) for n in self.out_names}
        self.load_s = time.time() - t0
        try:
            self.fully_accelerated = bool(self.model.is_fully_accelerated())
        except Exception:  # noqa: BLE001
            self.fully_accelerated = None
        self.max_len = self.L
        self.loads = [{"file": os.path.basename(path), "load_s": round(self.load_s, 2),
                       "fully_accelerated": self.fully_accelerated, "B": self.B, "L": self.L}]
        self.calls = []
        self.last = {}
        from d1_layout_b import make_sees_batch   # numpy only
        self._sees = make_sees_batch

    def _call(self, chunk):
        e = np.zeros((self.B, self.L, self.hidden), np.float32)
        for b, (embeds, _, _) in enumerate(chunk):
            e[b, : embeds.shape[0]] = embeds
        q, k = self._sees([(s, qq) for _, s, qq in chunk], self.B, self.L)
        t0 = time.perf_counter()
        for n, a in (("embeds", e), ("sees_q", q), ("sees_k", k)):
            self.inputs[n].write(np.ascontiguousarray(a))
        self.model.run_by_name(self.signature, self.inputs, self.outputs)
        shape = self.out_shapes["hidden"]
        h = np.asarray(self.outputs["hidden"].read(int(np.prod(shape)), np.float32), np.float32).reshape(shape)
        dt = time.perf_counter() - t0
        self.calls.append({"rows": len(chunk), "run_s": dt})
        return [h[b, : chunk[b][0].shape[0]] for b in range(len(chunk))]

    def many(self, seqs: Sequence[tuple]) -> list[np.ndarray]:
        """[(embeds [T, d], state_len, question tags)] -> [hidden [T, d]], B sequences a call."""
        for embeds, _, question in seqs:
            if embeds.shape[0] > self.L or len(question) != embeds.shape[0]:
                raise ValueError(f"a sequence of {embeds.shape[0]} tokens does not fit the batch graph's L {self.L}")
        out = []
        for i in range(0, len(seqs), self.B):
            out.extend(self._call(list(seqs[i:i + self.B])))
        self.last = {"L": self.L, "B": self.B, "sequences": len(seqs), "calls": -(-len(seqs) // self.B)}
        return out

    def __call__(self, embeds, state_len, question):
        return self.many([(np.asarray(embeds, np.float32), int(state_len), list(question))])[0]

    def close(self):
        for b in list(self.inputs.values()) + list(self.outputs.values()):
            try:
                b.destroy()
            except Exception:  # noqa: BLE001
                pass
        self.inputs, self.outputs = {}, {}
        self.model.close()


class D1HostBatched(D1Host):
    """D1Host whose `_padded` runs each provider batch's sequences through `self.decide.many` (one or a few batch-graph
    calls) and then reads the slots exactly as D1Host does."""

    def _embeds(self, ids, parts, features):
        """The first half of D1Host._hidden: table rows, picture tokens at the `<|image|>` positions (tower once a
        picture)."""
        timing = self.last["timing"]
        t0 = time.perf_counter()
        embeds = self.embed.rows(ids)
        if parts:
            merge = self.pictures.merge
            tokens = []
            for pic in parts:
                if id(pic) not in features:
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
        self._tick("prep_s", t0)
        return embeds

    def _padded(self, requests, set_size=None):
        """D1Host._padded with the hidden states of each provider batch from one `decide.many` (D1Host's code otherwise)."""
        prompt = self.p.prompt
        t0 = time.perf_counter()
        position_maps: dict = {}
        seqs = self._sequences(requests, position_maps=position_maps, set_size=set_size)
        self.last["seqs"] = [self._seq_record(n, seq, position_maps) for n, seq in enumerate(seqs)]
        self.last["slot_hidden"] = [None] * len(seqs)
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

        def asked(key):
            return requests[key[0]][1][key[1]]

        probs: dict = {}
        features: dict = {}
        t0 = self._tick("prep_s", t0)
        timing = self.last["timing"]
        for batch in self._batches(seqs):
            embeds = [self._embeds(seqs[b][2], seqs[b][6], features) for b in batch]
            t1 = time.perf_counter()
            many = getattr(self.decide, "many", None)
            if many is not None:
                hiddens_all = many([(e, int(seqs[b][3]), list(seqs[b][4])) for e, b in zip(embeds, batch)])
            else:   # a one-sequence backend (LiteRTDecide): one call a sequence, as D1Host
                hiddens_all = [np.asarray(self.decide(e, int(seqs[b][3]), list(seqs[b][4])), np.float32)
                               for e, b in zip(embeds, batch)]
            self._tick("decide_s", t1)
            timing["decide_calls"] += getattr(self.decide, "last", {}).get("calls", len(batch)) if many else len(batch)
            slots = []
            at_rows, hiddens = [], []
            for r, b in enumerate(batch):
                i, first, ids, state_len, question, ends, parts = seqs[b]
                timing["tokens"] += len(ids)
                hidden = np.asarray(hiddens_all[r], dtype=np.float32)
                if hidden.shape != embeds[r].shape:
                    raise ValueError(f"the decision backend returned {hidden.shape}, expected {embeds[r].shape}")
                own_slots = []
                for k, own in enumerate(ends):
                    key = (i, first + k)
                    own_slots.extend((r, key, q, end)
                                     for q, end in zip(prompt.question_slots(asked(key), self.tokenizer), own))
                at = hidden[[s[3] for s in own_slots]]
                self.last["slot_hidden"][b] = at
                slots += own_slots
                at_rows.append(at)
                hiddens.append(hidden if b in position_maps else None)
            t0 = time.perf_counter()
            at = at_rows[0] if len(at_rows) == 1 else np.concatenate(at_rows)
            reads = [j for j, s in enumerate(slots) if isinstance(s[2], prompt.Position)]
            if reads:
                maps = {r: position_maps[b] for r, b in enumerate(batch) if b in position_maps}
                positions = self._read_logits(slots, reads, at, hiddens, maps)
                self.last["read_logits"].append(positions)
                logits = positions.reshape(-1, 2, positions.shape[-1])
                capped: dict = {}
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
