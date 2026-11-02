"""The batch path of the d1-Omni-2B PyTorch-free host: d1_host.D1Host (pictures and audio clips) with `_padded`
running a provider batch's sequences through ONE `decide.many([(embeds, state_len, question), ...])` (multi8_omni.
Multi8Decide: decide_b8_L128 eight a call, or the smallest bucket per sequence; b8_single_omni.B8SingleDecide: the B = 8
single file) instead of one `decide(...)` per sequence; everything else is D1Host's `_padded` line for line (slot
reading of the batch at once, the pointer reads, `seq_audio`, the timing keys).

  host = D1HostBatchedOmni(SNAP, decide, tower, embed_table=TABLE, audio=audio)
  host.system_one_batch([(state, questions, media), ...])

`_embeds` = the first half of `D1Host._hidden` (table rows; the tower once a picture, the audio backend once a clip;
the tokens at the `<|image|>` / `<|audio|>` positions), copied; the decide call is the batch's `many`. A backend
without `many` (lrt_backend.LiteRTDecide) gets one call a sequence, as D1Host. numpy only (no torch).

Import rule: the `d1_host` module is the one in this folder (asserted).
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import d1_host as _h  # noqa: E402
from d1_host import Clip, D1Host, Picture, spans  # noqa: E402

assert os.path.samefile(_h.__file__, os.path.join(HERE, "d1_host.py")), _h.__file__


class D1HostBatchedOmni(D1Host):
    def _embeds(self, ids, parts, features):
        """D1Host._hidden without the decide call (copied): the sequence's embeddings."""
        timing = self.last["timing"]
        t0 = time.perf_counter()
        embeds = self.embed.rows(ids)
        pics = [p for p in parts if isinstance(p, Picture)]
        clips = [p for p in parts if isinstance(p, Clip)]
        if pics:
            merge = self.pictures.merge
            tokens = []
            for pic in pics:
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
        if clips:
            tokens = []
            for clip in clips:
                if id(clip) not in features:
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
        self._tick("prep_s", t0)
        return embeds

    def _padded(self, requests, set_size=None):
        """D1Host._padded with the hidden states of each provider batch from one `decide.many`."""
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
                timing["decide_calls"] += getattr(self.decide, "last", {}).get("calls", len(batch))
            else:   # a one-sequence backend (LiteRTDecide): one call a sequence, as D1Host
                hiddens_all = [np.asarray(self.decide(e, int(seqs[b][3]), list(seqs[b][4])), np.float32)
                               for e, b in zip(embeds, batch)]
                timing["decide_calls"] += len(batch)
            self._tick("decide_s", t1)
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
                self.last["seq_audio"][b] = [features[id(c)] for c in parts if isinstance(c, Clip)]
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
