"""Round 1 unit test of host/d1_litert.py: the render, tokenizer and read-out parts, against the provider's code.

    $REF host/test_d1_host.py

The host imports only tokenizers and numpy. The test also reads the provider's `prompt.py` and `api.py` (standard
library only) from hf_small/ to compare with them. No weights, no graph:
1. every row of fixtures/rows.json (the provider's render and ids, written by scripts/build_rows.py): the host's text,
   ids, read-out groups, option codes, answer slot and state length are equal, bit for bit;
2. the read-out: for every question, a random hidden state h and random float32 rows E for its read-out ids; the host's
   `readout(h, E, groups)` against the provider's `prompt.readout(tokenizer, q, logz)` with logz[i] = h . E[i] - c
   (c a random log-sum-exp: the provider's log-softmax constant); max |difference| is printed and must be <= 1e-12;
3. `answer` against the provider's `api.answer` on the same probabilities, and `input_tokens` against the provider's
   count (the row for one question, trunk + branches for several);
4. the graph plumbing with a stand-in graph (numpy): bucket choice, right padding with <|pad|>, `valid`, the answer slot;
   the CompiledModel graph class refuses a missing file (its run on real graphs: host/test_d1_host_tiny.py);
5. the request errors the provider raises: no instructions -> KeyError.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
K = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(K / "scripts"))
import d1_litert as H  # noqa: E402
from d1_common import FIXTURES, HF_SMALL, ROWS, provider  # noqa: E402


class ProviderTokenizer:
    """The `.encode(text, add_special_tokens=False) -> list` the provider's prompt.py calls, over the host's tokenizer
    (equal to the provider's AutoTokenizer on every fixture row: rows.json `raw_equal`, and part 1 below)."""

    def __init__(self, tok: H.D1Tokenizer):
        self.tok = tok

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return self.tok.encode(text)


def main() -> int:
    tok = H.D1Tokenizer(HF_SMALL / H.TOKENIZER_FILE)
    host = H.D1Host(tok)
    prompt, api = provider("prompt"), provider("api")
    ptok = ProviderTokenizer(tok)
    fixtures = json.loads(FIXTURES.read_text())
    rows_doc = json.loads(ROWS.read_text())
    by_key = {(r["id"], r["qid"]): r for r in rows_doc["rows"]}
    skipped = {(s["id"], s["qid"]) for s in rows_doc["skipped"]}

    # 1. rows
    n_rows = 0
    for rec in fixtures["records"]:
        req = rec["request"]
        if req.get("images"):
            continue   # pictures: round 5
        qs = {k: v for k, v in req["questions"].items() if (rec["id"], k) not in skipped}
        rows = host.rows({"state": req["state"], "questions": qs})
        for row in rows:
            want = by_key[(rec["id"], row.name)]
            assert row.text == want["text"], (rec["id"], row.name)
            assert row.ids == want["ids"], (rec["id"], row.name)
            assert row.groups == want["readout_ids"], (rec["id"], row.name, row.groups, want["readout_ids"])
            assert row.answer_slot == want["answer_slot"] and row.state_len == want["state_len"], (rec["id"], row.name)
            if row.question.type == "choice":
                assert [c for c, _ in H.aliases(tok, list(row.question.criteria))] == want["codes"]
            assert H.option_keys(row.question) == want["keys"], (rec["id"], row.name)
            n_rows += 1
    image_rows = sum(1 for r in rows_doc["rows"] if r.get("image_expansion_pending"))
    assert n_rows + image_rows == rows_doc["rows_count"], (n_rows, image_rows, rows_doc["rows_count"])
    print(f"1. rows: {n_rows} of {rows_doc['rows_count']} rows equal to the provider's (text, ids, groups, codes, slot, "
          f"state length); {image_rows} picture row left for round 5")

    # 2. read-out
    rng = np.random.default_rng(0)
    worst, n_q = 0.0, 0
    for rec in fixtures["records"]:
        for qid, qd in rec["request"]["questions"].items():
            if (rec["id"], qid) in skipped:
                continue
            q, pq = H.as_question(qd), prompt.as_question(qd)
            groups = H.readout_ids(tok, q)
            assert groups == prompt.readout_ids(ptok, pq)
            flat = sorted({i for g in groups for i in g})
            h = rng.standard_normal(H.HIDDEN).astype(np.float32)
            E = (rng.standard_normal((len(flat), H.HIDDEN)) * 0.05).astype(np.float32)
            table = H.ReadoutTable(flat, E)
            logits = dict(zip(flat, table.logits(h, flat).tolist()))
            c = float(rng.uniform(5, 15))
            p_provider = prompt.readout(ptok, pq, {i: v - c for i, v in logits.items()})
            p_host = H.readout(h, table, groups)
            worst = max(worst, max(abs(a - b) for a, b in zip(p_host, p_provider)))
            assert H.answer(q, p_host) == api.answer(pq, p_host)
            n_q += 1
    assert worst <= 1e-12, worst
    print(f"2. read-out: {n_q} questions, host vs provider (logz = logits - c) max |dp| = {worst:.3e}; "
          f"3. answer(): equal to api.answer on all {n_q}")

    # 3b. input tokens: the provider's runner counts the row (one question) or trunk + branches (several)
    for rec in fixtures["records"][:5] + [r for r in fixtures["records"] if r["id"] in ("own_fiveq_09", "card_text_001")]:
        req = {"state": rec["request"]["state"], "questions": rec["request"]["questions"]}
        rows = host.rows(req)
        if len(rows) == 1:
            want = len(by_key[(rec["id"], rows[0].name)]["ids"])
        else:
            pre = prompt.prefix_text(ptok, req["state"], H.BOS)
            want = len(ptok.encode(pre)) + sum(len(ptok.encode(prompt.suffix_text(ptok, prompt.as_question(v))))
                                                 for v in req["questions"].values())
        assert host.input_tokens(req, rows) == want, (rec["id"], host.input_tokens(req, rows), want)
    print("3b. input_tokens: equal to the provider's count on 7 requests (one-question and several-question)")

    # 4. graph plumbing with a stand-in graph
    calls = []

    def fake_graph(L):
        def run(ids, valid):
            assert ids.dtype == np.int32 and valid.dtype == np.float32 and ids.shape == valid.shape == (1, L)
            n = int(valid.sum())
            assert np.all(valid[0, :n] == 1) and np.all(valid[0, n:] == 0) and np.all(ids[0, n:] == H.PAD_ID)
            calls.append((L, n))
            out = np.zeros((1, L, H.HIDDEN), dtype=np.float32)
            out[0, n - 1] = 1.0 / np.sqrt(H.HIDDEN)   # a known vector at the answer slot only
            return out
        return run

    graphs = {L: fake_graph(L) for L in H.BUCKETS}
    rec = next(r for r in fixtures["records"] if r["id"] == "card_text_001")
    rows = host.rows(rec["request"])
    flat = sorted({i for r in rows for g in r.groups for i in g})
    table = H.ReadoutTable(flat, rng.standard_normal((len(flat), H.HIDDEN)).astype(np.float32))
    out = H.D1Host(tok, graphs, table).decide(rec["request"])
    assert [L for L, _ in calls] == [H.pick_L(len(r.ids)) for r in rows] and [n for _, n in calls] == [len(r.ids) for r in rows]
    assert set(out["answers"]) == {"refund", "team", "urgency"} and out["usage"]["output_tokens"] == 0
    for r in rows:   # the answer comes from the vector placed at the answer slot
        h = np.full(H.HIDDEN, 1.0 / np.sqrt(H.HIDDEN), dtype=np.float32)
        assert out["answers"][r.name] == H.answer(r.question, H.readout(h, table, r.groups))
    try:
        H.LiteRTRowGraph("none.tflite")
        raise AssertionError("a missing graph file was accepted")
    except FileNotFoundError:
        pass
    try:
        H.pick_L(4097)
        raise AssertionError("no RowTooLong")
    except H.RowTooLong:
        pass
    print(f"4. graph plumbing: {len(calls)} rows padded to buckets {[L for L, _ in calls]}, answers from the slot; "
          "the LiteRT graph class refuses a missing file (FileNotFoundError)")

    # 5. the provider's parser errors
    rec = next(r for r in fixtures["records"] if r["id"] == "own_email_03")
    for fn in (lambda: host.rows(rec["request"]), lambda: prompt.as_question(rec["request"]["questions"]["next_step"])):
        try:
            fn()
            raise AssertionError("no KeyError")
        except KeyError:
            pass
    print("5. a question without instructions: KeyError in the host and in the provider's parser")
    random.seed(0)
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
