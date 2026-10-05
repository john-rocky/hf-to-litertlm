"""The reference: the author's code at tag kev-1.0, CPU, float32, one torch thread.

    python scripts/oracle_kev.py                         # adapter checkpoint (LoRA merged at load, the author's default)
    python scripts/oracle_kev.py --ckpt merged/kev-0.8b-v1.0/checkpoint --tag merged   # the full-weight checkpoint
    python scripts/oracle_kev.py --model 4b              # jaredpalmer/kev-4b@591dcb5b, tag 4b
    python scripts/oracle_kev.py --model 4b --ckpt merged/kev-4b-v1.0/checkpoint --tag merged_4b

The checkpoint is resolved by the author's own resolver (kev.checkpoint.Checkpoint) into HF_HOME (downloaded on first
use). Per request: SystemOneRequest.model_validate -> kev.api.to_record -> DecisionModel.encode (serving context,
strict) -> kev.model.rows_of. Each question is one causal row (state + its branch, positions 0..L-1). Hidden states
come from the author's DecisionModel._rows_hidden (eval, no_grad); the readout is the checkpoint's PointerHead on
h[decide] and h[opts] (the </opt> tokens), once at T = 1 (z_pre) and once at the checkpoint's temperature (z_post);
probs = softmax(z_post). DecisionModel.forward(enc) (row form, the path of the published numbers) must give z_post
again (asserted), and DecisionModel.probs(enc) (the serving path: state once, rows on its cache) is compared.

Outputs (never overwritten): oracle/oracle_<tag>.json (no big arrays), oracle/hidden_<tag>.npz (h_sel [1+K, d] per
question under "<id>/<qid>", and every position of three rows under "<id>/<qid>/full"), results/oracle_summary[_<tag>].json."""
import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

K = Path(__file__).resolve().parents[1]
ADAPTER = "jaredpalmer/kev-0.8b@788ddbdd65715bb03a56788c822f6c632c9a551d"
ADAPTERS = {"0.8b": ADAPTER, "4b": "jaredpalmer/kev-4b@591dcb5bd6d05eb0b5131ea6608f93f10243335c"}   # --model -> default ckpt / tag
EXPECTED_SPECIAL = {"<|fim_prefix|>": 248060, "<|fim_middle|>": 248061, "<|box_start|>": 248049, "<|box_end|>": 248050, "<|fim_suffix|>": 248062}
EXPECTED_PAD = 248044
NEAR_TIE = 0.02
FULL_ROWS = ("tv4_000", None, "own_fiveq_09")   # None = the first semif record


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 24):
            h.update(block)
    return h.hexdigest()


def pct(values, q):
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(ADAPTERS), default="0.8b")
    ap.add_argument("--ckpt", default=None, help="default: the --model adapter checkpoint")
    ap.add_argument("--tag", default=None, help="default: the --model name")
    ap.add_argument("--limit", type=int, default=0, help="first N records only (smoke test; outputs get a _smoke suffix)")
    ap.add_argument("--threads", type=int, default=1,
                    help="torch intra-op threads. 1 by default: the GDN depthwise conv runs PyTorch's per-channel slow conv2d "
                         "path (110,592 calls per row), which is 20x slower at 12 threads (5.1 s vs 0.26 s for a 94-token row)")
    a = ap.parse_args()
    a.ckpt = a.ckpt or ADAPTERS[a.model]
    a.tag = a.tag or a.model
    torch.set_num_threads(a.threads)
    suffix = a.tag + ("_smoke" if a.limit else "")
    out_json, out_npz = K / f"oracle/oracle_{suffix}.json", K / f"oracle/hidden_{suffix}.npz"
    out_sum = K / ("results/oracle_summary.json" if suffix == "0.8b" else f"results/oracle_summary_{suffix}.json")
    for p in (out_json, out_npz, out_sum):
        assert not p.exists(), f"refusing to overwrite {p}"

    from kev.api import SystemOneRequest, to_answers, to_record
    from kev.checkpoint import Checkpoint, LoadOptions
    from kev.model import MAX_BRANCH, MAX_PACKED, MAX_STATE, SERVE_MAX_BRANCH, SERVE_MAX_STATE, SPECIAL, fits, rows_of

    torch.manual_seed(0)
    fixtures = json.loads((K / "fixtures/requests.json").read_text())
    records = fixtures["records"][: a.limit] if a.limit else fixtures["records"]
    first_semif = next(r["id"] for r in fixtures["records"] if r["source"] == "semif")
    full_rows = {rid or first_semif for rid in FULL_ROWS}

    t0 = time.time()
    ck = Checkpoint(a.ckpt)
    tok, model = ck.load("cpu", LoadOptions(dtype=torch.float32))
    load_seconds = round(time.time() - t0, 1)
    assert not model.training and model.backend == "torch" and model.hybrid
    special = {t: tok.convert_tokens_to_ids(t) for t in SPECIAL}
    assert special == EXPECTED_SPECIAL, special
    assert tok.pad_token_id == EXPECTED_PAD and model.pad_id == EXPECTED_PAD, (tok.pad_token_id, model.pad_id)
    decide_id, close_id = special["<|fim_suffix|>"], special["<|box_end|>"]
    T = float(ck.meta.temperature)
    assert model.head.temperature == T
    head_sd = {k: v for k, v in model.head.state_dict().items()}
    meta = {k: getattr(ck.meta, k) for k in ck.meta.KNOWN if k != "head"}
    meta["head"] = {k: {"shape": list(v.shape), "dtype": str(v.dtype), "sha256": hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest()}
                    for k, v in head_sd.items()}
    meta["extra_keys"] = sorted(ck.meta.extra)
    meta["checkpoint_full"] = ck.full
    meta["saved_dtype"] = ck.saved_dtype() if ck.full else None

    npz, rows_out, req_out = {}, [], []
    batch_vs_single = {}
    with torch.no_grad():
        for n, r in enumerate(records):
            req = SystemOneRequest.model_validate(r["request"])
            rec, qmeta = to_record(req)
            enc = model.encode(tok, rec, max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH, strict=True)
            S, Sp, rows = rows_of(enc)
            specs = []
            for k, br in enumerate(rows):
                ids, pos = S + br["ids"], Sp + br["pos"]
                assert pos == list(range(len(ids))), (r["id"], k)
                decide, opts = len(S) + br["decide"], [len(S) + o for o in br["opts"]]
                assert decide == len(ids) - 1 and ids[decide] == decide_id, (r["id"], k)
                assert all(ids[o] == close_id for o in opts) and len(opts) == len(qmeta[k]["keys"]), (r["id"], k)
                specs.append((ids, pos, decide, opts))
            started = time.time()
            hs = model._rows_hidden([(ids, pos) for ids, pos, _, _ in specs])
            seconds_hidden = time.time() - started
            z_pre, z_post, probs = [], [], []
            for (ids, pos, decide, opts), h in zip(specs, hs):
                h_dec, h_opts = h[decide], h[torch.tensor(opts)]
                model.head.temperature = 1.0
                z_pre.append(model.head(h_dec, h_opts))
                model.head.temperature = T
                z_post.append(model.head(h_dec, h_opts))
                probs.append(F.softmax(z_post[-1], -1))
            model.head.temperature = T
            started = time.time()
            logits = model.forward(enc)
            seconds_forward = time.time() - started
            fwd_equal = all(torch.equal(a_, b_) for a_, b_ in zip(logits, z_post))
            fwd_maxdiff = max(float((a_ - b_).abs().max()) for a_, b_ in zip(logits, z_post))
            assert all(torch.allclose(a_, b_, rtol=0, atol=1e-6) for a_, b_ in zip(logits, z_post)), (r["id"], fwd_maxdiff)
            started = time.time()
            p_serv = model.probs(enc)
            seconds_serving = time.time() - started
            answers = to_answers([p.tolist() for p in probs], qmeta)
            if r["id"] == "own_fiveq_09":   # one row per pass vs the author's batched pass (the exported graph runs one row per call)
                singles = [model._rows_hidden([(ids, pos)])[0] for ids, pos, _, _ in specs]
                batch_vs_single = {"record": r["id"], "rows": len(specs),
                                   "max_abs_hidden_diff": max(float((a_ - b_).abs().max()) for a_, b_ in zip(singles, hs)),
                                   "bit_equal": all(torch.equal(a_, b_) for a_, b_ in zip(singles, hs))}
            fits_train = fits(rec, tok, max_state=MAX_STATE, max_branch=MAX_BRANCH, max_packed=MAX_PACKED)
            for k, ((ids, pos, decide, opts), h) in enumerate(zip(specs, hs)):
                qid, keys = qmeta[k]["id"], qmeta[k]["keys"]
                p = probs[k]
                top = torch.topk(p, min(2, len(p)))
                gap = float(top.values[0] - top.values[1]) if len(p) > 1 else 1.0
                argmax_key = keys[int(torch.argmax(p))]
                gold = r["gold"].get(qid)
                npz[f"{r['id']}/{qid}"] = torch.cat([h[decide][None], h[torch.tensor(opts)]]).numpy().astype(np.float32)
                if r["id"] in full_rows and k == 0:
                    npz[f"{r['id']}/{qid}/full"] = h.numpy().astype(np.float32)
                rows_out.append({
                    "id": r["id"], "source": r["source"], "qid": qid, "type": qmeta[k]["type"], "keys": keys,
                    "row_len": len(ids), "row_ids": ids, "decide_idx": decide, "opt_idx": opts,
                    "z_pre": z_pre[k].tolist(), "z_post": z_post[k].tolist(), "probs": p.tolist(),
                    "argmax_key": argmax_key, "top2_gap": gap, "near_tie": gap <= NEAR_TIE,
                    "gold_key": gold, "gold_match": None if gold is None else gold == argmax_key,
                    "forward_equal": torch.equal(logits[k], z_post[k]),
                    "probs_serving_maxdiff": float((p_serv[k] - p).abs().max()),
                    "answer": answers[qid],
                })
            req_out.append({"id": r["id"], "source": r["source"], "questions": len(specs),
                            "usage": {"input_tokens": len(enc["ids"])}, "state_tokens": len(S),
                            "row_lens": [len(s[0]) for s in specs], "fits_training_context": fits_train,
                            "state_truncated": enc["state_truncated"],
                            "seconds": round(seconds_forward, 4), "seconds_hidden": round(seconds_hidden, 4),
                            "seconds_serving": round(seconds_serving, 4), "forward_bit_equal": fwd_equal,
                            "forward_max_abs_diff": fwd_maxdiff, "answers": answers})
            if n % 25 == 0 or r["source"] in ("own", "red_arm"):
                print(json.dumps({"n": n, "id": r["id"], "tokens": len(enc["ids"]), "fwd_s": round(seconds_forward, 2),
                                  "argmax": [x["argmax_key"] for x in rows_out[-len(specs):]]}), flush=True)

        # determinism: tv4_000 again, bit for bit
        rerun = None
        base = next((x for x in records if x["id"] == "tv4_000"), None)
        if base is not None:
            rec, qmeta = to_record(SystemOneRequest.model_validate(base["request"]))
            enc = model.encode(tok, rec, max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH, strict=True)
            again = model.forward(enc)
            first = [torch.tensor(x["z_post"], dtype=torch.float32) for x in rows_out if x["id"] == "tv4_000"]
            rerun = {"record": "tv4_000", "bit_equal": all(torch.equal(a_, b_) for a_, b_ in zip(again, first))}
            assert rerun["bit_equal"], "tv4_000 re-run is not bit-identical"

    np.savez(out_npz, **npz)
    doc = {
        "checkpoint": a.ckpt, "tag": a.tag, "path": "kev.checkpoint.Checkpoint.load('cpu', LoadOptions(dtype=torch.float32)); "
                                                    "readout = PointerHead on DecisionModel._rows_hidden rows; forward(enc) asserted equal",
        "kev": {"tag": "kev-1.0", "commit": "6b719c3c3f367295f6ef336f4f751cf5ff970abc"},
        "fixtures_sha256": sha256_file(K / "fixtures/requests.json"),
        "special_token_ids": special, "pad_token_id": tok.pad_token_id, "temperature": T,
        "context": {"serving": [SERVE_MAX_STATE, SERVE_MAX_BRANCH], "training": [MAX_STATE, MAX_BRANCH, MAX_PACKED]},
        "requests": req_out, "questions": rows_out,
    }
    out_json.write_text(json.dumps(doc, ensure_ascii=False) + "\n")

    # summary
    def by(src):
        return [x for x in rows_out if x["source"] == src]
    real = [x for x in rows_out if x["source"] != "red_arm"]
    gold = {}
    for src in ("tv4", "tv4x", "tv4s", "semif", "own"):
        qs = [x for x in by(src) if x["gold_key"] is not None]
        if qs:
            gold[src] = {"match": sum(x["gold_match"] for x in qs), "questions": len(qs)}
    red = {}
    red_rows, base_rows = by("red_arm"), [x for x in rows_out if x["id"] == "tv4_000"]
    if red_rows and base_rows:
        dp = max(abs(p - q) for p, q in zip(red_rows[0]["probs"], base_rows[0]["probs"]))
        red = {"vs": "tv4_000", "max_abs_dp": dp, "argmax_red": red_rows[0]["argmax_key"], "argmax_base": base_rows[0]["argmax_key"],
               "fails_bar_0.02": dp > 0.02}
    secs = [q["seconds"] for q in req_out]
    summary = {
        "checkpoint": a.ckpt, "tag": a.tag, "load_seconds": load_seconds, "platform": platform.platform(),
        "torch_threads": torch.get_num_threads(), "torch": torch.__version__,
        "meta": meta, "pad_token_id": tok.pad_token_id, "special_token_ids": special,
        "model": {"hybrid": model.hybrid, "dtype": model.dtype, "backend": model.backend, "lm_class": type(model.lm).__name__,
                  "attn_implementation": getattr(model.lm.config, "_attn_implementation", None)},
        "requests": len(req_out), "questions": len(rows_out), "questions_excluding_red_arm": len(real),
        "by_source": {src: {"requests": sum(1 for q in req_out if q["source"] == src), "questions": len(by(src)),
                            "types": {t: sum(1 for x in by(src) if x["type"] == t) for t in ("choice", "noul", "score")}}
                      for src in ("tv4", "tv4x", "tv4s", "semif", "own", "red_arm")},
        "options_max": max(len(x["keys"]) for x in rows_out),
        "near_tie": {"threshold": NEAR_TIE, "count": sum(x["near_tie"] for x in real),
                     "ids": [f"{x['id']}/{x['qid']}" for x in real if x["near_tie"]]},
        "gold_match": gold, "gold_note": "own = our subset (answers intended by the record's author), not a benchmark",
        "forward_equal_all": all(x["forward_equal"] for x in rows_out),
        "forward_max_abs_diff": max(q["forward_max_abs_diff"] for q in req_out),
        "serving_path_max_abs_dp": max(x["probs_serving_maxdiff"] for x in rows_out),
        "serving_path_argmax_note": "serving path = DecisionModel.probs (state once, rows on its cache)",
        "seconds_forward": {"p50": statistics.median(secs), "max": max(secs), "total": round(sum(secs), 1)},
        "seconds_total_wall": round(time.time() - t0, 1),
        "rerun_tv4_000": rerun, "batch_vs_single_row": batch_vs_single, "red_arm": red,
        "fits_training_context": {"yes": sum(q["fits_training_context"] for q in req_out), "no_ids": [q["id"] for q in req_out if not q["fits_training_context"]]},
        "files": {"json": str(out_json.relative_to(K)), "json_sha256": sha256_file(out_json),
                  "npz": str(out_npz.relative_to(K)), "npz_sha256": sha256_file(out_npz), "npz_keys": len(npz)},
    }
    out_sum.write_text(json.dumps(summary, indent=1, default=str) + "\n")
    print(json.dumps({k: summary[k] for k in ("requests", "questions", "gold_match", "near_tie", "forward_equal_all",
                                              "serving_path_max_abs_dp", "seconds_forward", "rerun_tv4_000", "red_arm",
                                              "batch_vs_single_row")}, indent=1, default=str))


if __name__ == "__main__":
    main()
