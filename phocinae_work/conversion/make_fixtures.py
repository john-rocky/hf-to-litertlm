"""Step 1: fixtures from the public test split (LocalLLaMA/typed-decisions, the author's normalized typed_test rows).

    python make_fixtures.py

Rows (fixtures/rows_laya.json), every one built with the publisher's tokenizer replica (phocinae/tokenizer.py) and
`Engine._build_sequence` (head_max_len 192, max_len 512), options rendered the laya / SDK way (key: description,
level i: text, false: … / true: …), the dataset state as json.dumps(state, ensure_ascii=False):
  * test      — all 400 cases x 5 questions = 2,000 decisions, the rows behind the card's 0.906 (typed_test README);
  * flip      — the author's flip400 protocol on the 600 choice questions: reversed criteria order and three random
                orders (random.Random(0/1/2), one rng per seed, shuffled per question in file order = eval_flip.py);
  * control   — the first case with three numbers of its state changed (tool_errors 0 -> 7, constraint_violations
                0 -> 3, irreversible_actions 0 -> 2): a graph compared with the ORIGINAL reference must move by more
                than the bar on these rows (kev's red arm);
  * server    — fixtures/requests_server.json: the first 100 cases as phocinae-server requests (noul / choice with the
                criteria keys as options / score = the server's nine levels), run end to end by `Engine.run` in the
                oracle step.
Every head, option and state string is also encoded with Hugging Face `tokenizers` and compared with the replica
(count of differing strings in the summary; the SDK's Kotlin port was checked against laya's ids the same way).
"""
import collections
import copy
import json
import random
import statistics

from common import (QTYPES, SRC_DATASETS, W, Builder, author_tokenizer, dump_json, hf_tokenizer, render_options_laya,
                    serialize_state)

CONTROL_CASE = 0
CONTROL_EDITS = {"tool_errors": 7, "constraint_violations": 3, "irreversible_actions": 2}
SERVER_CASES = 100
FLIP_SEEDS = [0, 1, 2]


def main():
    rows_path = SRC_DATASETS / "typed_test" / "test_typed_400.jsonl"
    cases = [json.loads(l) for l in rows_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(cases) == 400 and all(len(c["questions"]) == 5 for c in cases)
    tok = author_tokenizer()
    hf = hf_tokenizer()
    b = Builder(tok)
    strings = {}

    def enc(s):
        if s not in strings:
            strings[s] = list(tok.encode(s))
        return strings[s]

    rows = []

    def add(row_id, group, case, qid, q, order_note=None):
        opts = render_options_laya(q)
        st = serialize_state(case["state"])
        state_ids = enc(st.replace("<mask>", " "))
        built = b.build(q["type"], q["instructions"], opts, state_ids)
        assert len(built["markers"]) == len(opts), (row_id, len(built["markers"]), len(opts))
        gold = case["gold"][qid]
        rows.append(dict(id=row_id, group=group, case_id=case["id"], workflow=case["workflow"], qid=qid, qtype=built["qtype"],
                         qtype_name=q["type"], K=len(opts), options=opts,
                         # the answer keys the gold labels use: criteria keys (choice / noul with criteria), false/true (noul without), level index (score)
                         option_keys=(["false", "true"] if q["type"] == "noul" else list(q["criteria"].keys()) if isinstance(q.get("criteria"), dict)
                                      else [str(i) for i in range(len(opts))]), gold_label=gold["label"], gold_probabilities=gold["probabilities"],
                         ids=built["ids"], markers=built["markers"], row_len=len(built["ids"]), head_len=built["head_len"],
                         state_tokens=built["state_tokens"], state_truncated=built["state_truncated"], squeezed=built["squeezed"],
                         order=order_note))

    # test rows
    for c in cases:
        for qid, q in c["questions"].items():
            add(f"{c['id']}/{qid}", "test", c, qid, q)
    # flip rows (the author's build_variants, seeds 0..2 over the cases in file order)
    rngs = {s: random.Random(s) for s in FLIP_SEEDS}
    for c in cases:
        for qid, q in c["questions"].items():
            if q["type"] != "choice" or not isinstance(q.get("criteria"), dict):
                continue
            q_rev = copy.deepcopy(q)
            q_rev["criteria"] = dict(reversed(list(q["criteria"].items())))
            add(f"{c['id']}/{qid}#rev", "flip", c, qid, q_rev, order_note=list(q_rev["criteria"].keys()))
        for s in FLIP_SEEDS:
            for qid, q in c["questions"].items():
                if q["type"] != "choice" or not isinstance(q.get("criteria"), dict):
                    continue
                items = list(q["criteria"].items())
                rngs[s].shuffle(items)
                q_r = copy.deepcopy(q)
                q_r["criteria"] = dict(items)
                add(f"{c['id']}/{qid}#s{s}", "flip", c, qid, q_r, order_note=list(q_r["criteria"].keys()))
    # control rows
    c0 = copy.deepcopy(cases[CONTROL_CASE])
    ts = c0["state"]["trace_summary"]
    for k, v in CONTROL_EDITS.items():
        assert k in ts, k
        ts[k] = v
    c0["id"] = "ctrl/" + c0["id"]
    for qid, q in c0["questions"].items():
        add(f"{c0['id']}/{qid}", "control", c0, qid, q)

    # tokenizer cross-check (author's replica vs HF tokenizers) on every string the rows used
    mism = []
    for s, ids in strings.items():
        got = hf.encode(s, add_special_tokens=False).ids
        if got != ids:
            mism.append(s[:80])
    # server requests (the official runtime's own protocol)
    reqs = []
    for c in cases[:SERVER_CASES]:
        qs = []
        for qid, q in c["questions"].items():
            if q["type"] == "noul":
                qs.append({"id": qid, "type": "noul"})
            elif q["type"] == "choice":
                qs.append({"id": qid, "type": "choice", "options": list(q["criteria"].keys())})
            else:
                qs.append({"id": qid, "type": "score"})
        reqs.append({"case_id": c["id"], "workflow": c["workflow"], "state": serialize_state(c["state"]), "questions": qs})

    lens = [r["row_len"] for r in rows if r["group"] == "test"]
    by_type = collections.Counter(r["qtype_name"] for r in rows if r["group"] == "test")
    by_K = collections.Counter((r["qtype_name"], r["K"]) for r in rows if r["group"] == "test")
    buckets = {L: sum(1 for n in lens if n <= L) for L in (128, 192, 256, 320, 384, 448, 512)}
    summary = dict(source=dict(file=str(rows_path.relative_to(W)), cases=len(cases), dataset="LocalLLaMA/typed-decisions test split, "
                   "normalized by the author (github.com/Phocinae/Phocinae-Largha-150M-v1 datasets/typed_test, sha256 of the jsonl "
                   "03cdab92… per its README)"),
                   rows=dict(total=len(rows), **collections.Counter(r["group"] for r in rows)), by_type=dict(by_type),
                   by_type_K={f"{t}:{k}": n for (t, k), n in sorted(by_K.items())},
                   row_len=dict(min=min(lens), median=statistics.median(lens), p90=sorted(lens)[int(0.9 * len(lens))], max=max(lens),
                                fits_at_most=buckets),
                   state_truncated=sum(r["state_truncated"] for r in rows), squeezed=sum(r["squeezed"] for r in rows),
                   tokenizer_crosscheck=dict(strings=len(strings), hf_tokenizers_differs=len(mism), examples=mism[:5]),
                   control=dict(case=cases[CONTROL_CASE]["id"], edits=CONTROL_EDITS), server_requests=len(reqs),
                   protocol="laya build_sequence via phocinae-server Engine._build_sequence (head_max_len 192, max_len 512); "
                            "options: key: description / level i: text / false: … true: …; state json.dumps(ensure_ascii=False)")
    dump_json(W / "fixtures" / "rows_laya.json", dict(summary=summary, rows=rows))
    dump_json(W / "fixtures" / "requests_server.json", dict(cases=reqs))
    dump_json(W / "fixtures" / "summary.json", summary)
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
