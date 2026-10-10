"""Step 6b: the official runtime's own inputs through the graph — phocinae-server's HTTP protocol replayed.

    python server_replay.py --tflite exports/phocinae_L512_ids_v2_fp16fc_i8emb_shift.tflite --L 512 [--accel cpu|gpu] [--table ...]

oracle/server_runs.json holds, for 100 test cases, `Engine.run(state, questions, with_scores=True)` of phocinae-server
(DEFAULT_INS, bare option labels, the server's nine score levels) together with the ids / markers / qtype the engine
built. Each row runs through the graph; the probabilities (the server's temperatures, float64 softmax) are compared
with the server's `scores` (rounded to 4 decimals by the server) and the decoded answers (choice argmax, score 2 +
argmax, noul p[true] >= 0.5) with `answers`; act_probability with `action`. Output: results/server_replay_{mode}_L{L}_{variant}.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from common import QTYPES, W, act_features, act_probability, dump_json, probs_from_logits, temperatures
from litert_parity import HIDDEN, PAD_ID, load_table, open_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", required=True)
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--accel", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--f32", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--table", default="")
    ap.add_argument("--act", default="exports/phocinae_act_head_fp32.tflite")
    a = ap.parse_args()
    path = Path(a.tflite) if Path(a.tflite).is_absolute() else W / a.tflite
    variant = path.stem.replace(f"phocinae_L{a.L}_", "")
    mode = "cpu" if a.accel == "cpu" else ("gpu_f32" if a.f32 else "gpu_f16")
    out = W / "results" / f"server_replay_{mode}_L{a.L}_{variant}.json"
    assert not out.exists(), f"never overwrite {out}"
    temps = temperatures()
    runs = json.loads((W / "oracle" / "server_runs.json").read_text())["runs"]
    table = load_table(W / a.table) if a.table else None
    m, key, ins, outs, info = open_model(path, a.accel, a.f32, a.threads)
    am, akey, ains, aouts, _ = open_model(W / a.act, "cpu", False, 1)
    L = a.L
    recs = []
    skipped = 0
    for run in runs:
        for row in run["rows"]:
            n = len(row["ids"])
            if n > L:
                skipped += 1
                continue
            ids = np.full((1, L), PAD_ID, dtype=np.int32)
            ids[0, :n] = np.asarray(row["ids"], dtype=np.int32)
            mask = np.zeros((1, L), dtype=np.float32)
            mask[0, :n] = 1.0
            onehot = np.zeros((1, 3), dtype=np.float32)
            onehot[0, row["qtype"]] = 1.0
            if table is not None:
                ins["inputs_embeds"].write(np.ascontiguousarray(table[ids[0]][None]))
            else:
                ins["input_ids"].write(ids)
            ins["attention_mask"].write(mask)
            ins["qtype_onehot"].write(onehot)
            m.run_by_name(key, ins, outs)
            tl = np.asarray(outs["token_logits"].read(L, np.float32), dtype=np.float32).reshape(L)
            pc = np.asarray(outs["pooled_cls"].read(HIDDEN, np.float32), dtype=np.float32).reshape(HIDDEN)
            raw = tl[row["markers"]].astype(np.float64)
            p = probs_from_logits(raw, row["qtype"], temps)
            ains["pooled_cls"].write(np.ascontiguousarray(pc[None]))
            ains["feats"].write(np.asarray([act_features(raw)], dtype=np.float32))
            am.run_by_name(akey, ains, aouts)
            al = np.asarray(aouts["act_logits"].read(2, np.float32), dtype=np.float64).reshape(2)
            qid, t = row["id"], row["type"]
            srv_scores = np.asarray(run["scores"][qid], dtype=np.float64)
            srv_ans = run["answers"][qid]
            if t == "choice":
                ans = int(np.argmax(p))
            elif t == "score":
                ans = 2 + int(np.argmax(p))
            else:
                ans = bool(p[1] >= 0.5)
            recs.append(dict(case=run["case_id"], qid=qid, type=t, row_len=n, K=row["n_opts"], finite=bool(np.isfinite(tl[:n]).all()),
                             max_abs_dp_vs_server_scores=float(np.abs(np.asarray(p) - srv_scores).max()), answer=ans, server_answer=srv_ans,
                             answer_equal=(ans == srv_ans), confidence=round(float(max(p)), 4), server_confidence=run["confidence"][qid],
                             act_probability=round(act_probability(al), 4), server_act_probability=run["action"][qid]["act_probability"]))
    fin = [r for r in recs if r["finite"]]
    by_type = {}
    for t in QTYPES:
        sub = [r for r in fin if r["type"] == t]
        if sub:
            by_type[t] = dict(rows=len(sub), answers_equal=sum(r["answer_equal"] for r in sub), max_abs_dp=max(r["max_abs_dp_vs_server_scores"] for r in sub))
    summary = dict(file=str(path.relative_to(W)), L=L, accel=a.accel, gpu_enforce_f32=a.f32 if a.accel == "gpu" else None, table=a.table or None,
                   rows=len(recs), skipped_longer_than_L=skipped, finite=len(fin), answers_equal=sum(r["answer_equal"] for r in fin),
                   max_abs_dp_vs_server_scores=max(r["max_abs_dp_vs_server_scores"] for r in fin),
                   confidence_rounded_equal=sum(abs(r["confidence"] - r["server_confidence"]) <= 1e-4 + 1e-9 for r in fin),
                   act_probability_max_abs_d=max(abs(r["act_probability"] - r["server_act_probability"]) for r in fin), by_type=by_type,
                   note="server scores / confidence / act_probability are rounded to 4 decimals by phocinae-server; |dp| includes that rounding",
                   model_info=info)
    dump_json(out, dict(summary=summary, rows=recs))
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
