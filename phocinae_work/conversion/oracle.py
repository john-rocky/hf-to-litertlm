"""Step 2: the reference = the publisher's runtime (phocinae-server 0.1.5 `Engine`, pure PyTorch) on the CPU in fp32.

    python oracle.py [--threads 4] [--limit N]

For every fixture row (fixtures/rows_laya.json) the author's `DecisionModel` forward runs on that row alone
(batch 1, no padding): marker logits [K], act logits [2], and the head's CLS row (pooled_cls [384], read by running the
same modules as DecisionModel.forward; the full forward is asserted bit-equal on the logits). Probabilities follow
`Engine._probs` (logits / T[qtype], float64 softmax), act_probability = softmax(act)[0], near tie = top-2 gap <= 0.02.
Also:
  * batch shape: the 5 questions of each test case collated as the server does inside one request (right padding to
    the longest row and to K max) -> max |dlogit| and |dp| of the batched forward against the batch-1 rows (the card's
    "deterministic for a fixed batch shape" caveat, measured);
  * server protocol: `Engine.run(state, questions, with_scores=True)` on fixtures/requests_server.json (the official
    request shape: DEFAULT_INS, bare option labels, nine score levels), with the ids the engine built, so the graph can be
    replayed on the official runtime's own inputs.
Outputs: oracle/oracle.json (rows), oracle/batch_shape.json, oracle/server_runs.json, oracle/env.json.
"""
import argparse
import importlib.metadata
import json
import platform
import time

import numpy as np
import torch

from common import (HF, QTYPES, W, act_probability, dump_json, probs_from_logits, sha256_file, temperatures)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from phocinae.engine import Engine
    t0 = time.time()
    eng = Engine(str(HF), device="cpu", warm=False)
    load_s = time.time() - t0
    model = eng.model
    temps = temperatures()
    assert [round(x, 7) for x in temps] == [round(x, 7) for x in eng.temperature]
    fx = json.loads((W / "fixtures" / "rows_laya.json").read_text())
    rows = fx["rows"][: a.limit] if a.limit else fx["rows"]

    def forward_row(ids, markers, qtype):
        with torch.no_grad():
            input_ids = torch.tensor([ids], dtype=torch.long)
            att = torch.ones_like(input_ids)
            mpos = torch.tensor([markers], dtype=torch.long)
            mmask = torch.ones_like(mpos, dtype=torch.bool)
            qt = torch.tensor([qtype], dtype=torch.long)
            logits, act = model(input_ids, att, mpos, mmask, qt)
            # the same modules again, to read the head's CLS row (DecisionModel.forward keeps it internal)
            h = model.encoder(input_ids, att)
            h = h + model.type_emb(qt)[:, None, :]
            h = model.head(h, src_key_padding_mask=~att.bool())
            logits2, act2 = model._score(h, mpos, mmask)
            assert torch.equal(logits, logits2) and torch.equal(act, act2)
            return logits[0].numpy().astype(np.float64), act[0].numpy().astype(np.float64), h[0, 0].numpy().astype(np.float32)

    out_rows = []
    pooled = {}
    t1 = time.time()
    for i, r in enumerate(rows):
        lg, ac, cls = forward_row(r["ids"], r["markers"], r["qtype"])
        p = probs_from_logits(lg, r["qtype"], temps)
        srt = sorted(p, reverse=True)
        gap = srt[0] - (srt[1] if len(srt) > 1 else 0.0)
        out_rows.append(dict(id=r["id"], group=r["group"], qtype=r["qtype"], K=r["K"], row_len=r["row_len"], markers=r["markers"],
                             logits=[float(x) for x in lg], act_logits=[float(x) for x in ac], probs=p,
                             act_probability=act_probability(ac), argmax=int(np.argmax(p)), top2_gap=gap, near_tie=gap <= 0.02,
                             gold_label=r["gold_label"], option_keys=r["option_keys"]))
        pooled[r["id"]] = cls
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(rows)} {time.time() - t1:.0f}s", flush=True)
    ref_s = time.time() - t1
    np.savez_compressed(W / "oracle" / "pooled_cls.npz", **{k.replace("/", "__"): v for k, v in pooled.items()})

    # accuracy of the reference on the test rows (agreement with gold), as a sanity anchor against the card's 0.906
    test = [o for o in out_rows if o["group"] == "test"]
    correct = sum(1 for o in test if o["option_keys"][o["argmax"]] == o["gold_label"])
    by_type = {}
    for t, ti in QTYPES.items():
        sub = [o for o in test if o["qtype"] == ti]
        by_type[t] = dict(n=len(sub), correct=sum(1 for o in sub if o["option_keys"][o["argmax"]] == o["gold_label"]))
    flips = {}
    base = {o["id"]: o for o in out_rows if o["group"] == "test"}
    for suffix in ("rev", "s0", "s1", "s2"):
        n = f = 0
        for o in out_rows:
            if o["group"] != "flip" or not o["id"].endswith("#" + suffix):
                continue
            b0 = base[o["id"].split("#")[0]]
            row_fx = next(r for r in rows if r["id"] == o["id"])
            chosen = row_fx["option_keys"][o["argmax"]]          # option_keys of a flip row = the reordered keys
            n += 1
            f += chosen != b0["option_keys"][b0["argmax"]]
        flips[suffix] = dict(n=n, flips=f, rate=round(f / max(1, n), 4))

    # batch-shape probe: each test case's 5 rows collated like the server (right padding)
    bs = []
    cases = {}
    for r in rows:
        if r["group"] == "test":
            cases.setdefault(r["case_id"], []).append(r)
    t2 = time.time()
    for cid, rs in list(cases.items()):
        L = max(len(r["ids"]) for r in rs)
        K = max(len(r["markers"]) for r in rs)
        ids = torch.full((len(rs), L), eng.tok.pad_id, dtype=torch.long)
        att = torch.zeros((len(rs), L), dtype=torch.long)
        mpos = torch.zeros((len(rs), K), dtype=torch.long)
        mmask = torch.zeros((len(rs), K), dtype=torch.bool)
        for i, r in enumerate(rs):
            ids[i, :len(r["ids"])] = torch.tensor(r["ids"])
            att[i, :len(r["ids"])] = 1
            mpos[i, :len(r["markers"])] = torch.tensor(r["markers"])
            mmask[i, :len(r["markers"])] = True
        with torch.no_grad():
            lg, ac = model(ids, att, mpos, mmask, torch.tensor([r["qtype"] for r in rs]))
        for i, r in enumerate(rs):
            ref = next(o for o in out_rows if o["id"] == r["id"])
            l1 = lg[i, :r["K"]].numpy().astype(np.float64)
            p1 = probs_from_logits(l1, r["qtype"], temps)
            bs.append(dict(id=r["id"], L=L, K=K, max_abs_dlogit=float(np.abs(l1 - np.asarray(ref["logits"])).max()),
                           max_abs_dp=float(np.abs(np.asarray(p1) - np.asarray(ref["probs"])).max()),
                           act_dp=abs(act_probability(ac[i].numpy()) - ref["act_probability"]),
                           argmax_equal=int(np.argmax(p1)) == ref["argmax"]))
    batch_s = time.time() - t2
    batch_doc = dict(cases=len(cases), rows=len(bs), max_abs_dlogit=max(x["max_abs_dlogit"] for x in bs),
                     max_abs_dp=max(x["max_abs_dp"] for x in bs), max_act_dp=max(x["act_dp"] for x in bs),
                     argmax_flips=sum(not x["argmax_equal"] for x in bs), seconds=round(batch_s, 1),
                     note="the 5 rows of a case collated with right padding (the server's in-request batch) vs the same rows alone")

    # server protocol runs
    reqs = json.loads((W / "fixtures" / "requests_server.json").read_text())["cases"]
    runs = []
    t3 = time.time()
    for req in reqs:
        answers, confs, actions, usage, scores = eng.run(req["state"], req["questions"], perm=False, with_scores=True)
        state_ids, items = eng._encode_request(req["state"], req["questions"])
        built = [dict(id=it["q"]["id"], type=it["q"]["type"], ids=it["ids"], markers=it["markers"], qtype=it["qtype"], n_opts=it["n_opts"],
                      options=it["spec"][2]) for it in items]
        runs.append(dict(case_id=req["case_id"], answers=answers, confidence=confs, action=actions, usage=usage, scores=scores, rows=built))
    server_s = time.time() - t3

    env = dict(python=platform.python_version(), torch=torch.__version__, threads=a.threads, machine=platform.machine(),
               macos=platform.mac_ver()[0], engine="phocinae-server 0.1.5 (src/phocinae-server @ 705508c3), Engine(device='cpu')",
               weights_sha256=sha256_file(HF / "model.safetensors"), temperature=temps,
               params=int(sum(p.numel() for p in model.parameters())), load_s=round(load_s, 1), ref_rows_s=round(ref_s, 1),
               server_runs_s=round(server_s, 1), versions={p: importlib.metadata.version(p) for p in ("torch", "numpy")})
    dump_json(W / "oracle" / "oracle.json", dict(env=env, accuracy=dict(test_rows=len(test), correct=correct, acc=round(correct / len(test), 4),
                                                                          by_type=by_type, card_claims=0.906),
                                                 flip=flips, near_tie=sum(o["near_tie"] for o in test), rows=out_rows))
    dump_json(W / "oracle" / "batch_shape.json", dict(summary=batch_doc, rows=bs))
    dump_json(W / "oracle" / "server_runs.json", dict(env=env, runs=runs))
    dump_json(W / "oracle" / "env.json", env)
    print(json.dumps(dict(env=env, accuracy=dict(acc=round(correct / len(test), 4), by_type=by_type), flip=flips,
                          batch_shape=batch_doc, server_runs=len(runs)), indent=1))


if __name__ == "__main__":
    main()
