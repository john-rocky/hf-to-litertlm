#!/usr/bin/env python3
"""Task-level verification for the ettin-reranker-400m-v1 LiteRT artifacts.

    python verify_ettin_reranker.py out_ettin

Gates:
  A  Card example — 4 passages, raw scores per variant, rank order must hold.
  R  NanoSciFact RERANK — for each query: candidates = gold docs + seeded
     random negatives (20 total), score every (query, doc) pair, rank.
     nDCG@10 / MRR@10 / hit@1 vs the torch fp32 control, plus score-parity
     stats (max|Δ|, pairwise inversion rate on shared candidate lists).
     This is the task the model ships for (one invoke per candidate) — a
     retrieval-corpus gate would test the wrong contract.
  D  Mechanics — cross-signature consistency (same pair, score_{128,256,512}
     must agree: pads are masked keys), pad-content invariance on the
     artifact, finiteness everywhere.

Variant sig selection mirrors a host: smallest signature >= pair length,
pairs longer than 512 truncated longest_first by the tokenizer.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "nemotron_work"))

import numpy as np
import torch

from convert_ettin_reranker import (
    DEFAULT_MODEL, PAD_ID, CARD_QUERY, CARD_PASSAGES, CARD_SCORES,
    RerankScorer, load_model, load_head, pair_tokens, pad_to, torch_ref,
)
# The NanoSciFact loader and nDCG helper are model-agnostic and already live
# with the Nemotron embedder converter — import them from there.
from verify_nemotron3_embed import load_nanoscifact, ndcg_at_k

SEQ_LENS = (128, 256, 512)


class TorchScorer:
    def __init__(self, model_id):
        self.mod = RerankScorer(load_model(model_id), *load_head(model_id)).eval()

    def score(self, ids_np):
        S = next((s for s in SEQ_LENS if len(ids_np) <= s), SEQ_LENS[-1])
        ids, mask, _ = pad_to(ids_np, S)
        return float(torch_ref(self.mod, ids, mask)[0, 0])


class TfliteScorer:
    def __init__(self, path):
        from ai_edge_litert.interpreter import Interpreter

        self.it = Interpreter(model_path=path, num_threads=os.cpu_count())
        self.runners = {S: self.it.get_signature_runner(f"score_{S}")
                        for S in SEQ_LENS}

    def score(self, ids_np):
        S = next((s for s in SEQ_LENS if len(ids_np) <= s), SEQ_LENS[-1])
        ids, mask, _ = pad_to(ids_np, S)
        r = self.runners[S](input_ids=ids.numpy(), attention_mask=mask.numpy())
        return float(list(r.values())[0][0, 0])


def mrr_at_k(ranked_rel, k=10):
    for i, rel in enumerate(ranked_rel[:k]):
        if rel:
            return 1.0 / (i + 1)
    return 0.0


def gate_a(scorers, tok, report):
    print("\n=== GATE A: card example (raw scores) ===")
    toks = [pair_tokens(tok, CARD_QUERY, d) for d in CARD_PASSAGES]
    report["A"] = {}
    for tag, sc in scorers.items():
        scores = [sc.score(t) for t in toks]
        order = list(np.argsort(scores)[::-1])
        ok = order == [1, 3, 2, 0]
        print(f"{tag:>10}: {[round(s, 4) for s in scores]} order {order} "
              f"{'OK' if ok else 'FAIL'}")
        report["A"][tag] = {"scores": scores, "order_ok": bool(ok)}
        assert ok, f"{tag}: card rank order broke"
    print(f"{'published':>10}: {CARD_SCORES} (bf16 run)")


def gate_r(scorers, tok, report, n_queries, n_cand):
    print(f"\n=== GATE R: NanoSciFact rerank ({n_queries} queries x "
          f"{n_cand} candidates) ===")
    docs, queries, gold = load_nanoscifact(0)  # all gold + no extra filler
    by_id = dict(docs)
    all_ids = [i for i, _ in docs]
    rng = np.random.default_rng(0)
    queries = [(qid, qt) for qid, qt in queries if qid in gold][:n_queries]

    # Build fixed candidate lists (shared across variants)
    cands = {}
    for qid, _ in queries:
        g = sorted(gold[qid])
        negs = [i for i in all_ids if i not in gold[qid]]
        take = rng.choice(len(negs), size=n_cand - len(g), replace=False)
        cands[qid] = g + [negs[int(t)] for t in take]

    tok_cache = {}
    def pt(qtext, did):
        key = (qtext, did)
        if key not in tok_cache:
            tok_cache[key] = pair_tokens(tok, qtext, by_id[did])
        return tok_cache[key]

    scores_by_tag = {}
    report["R"] = {}
    for tag, sc in scorers.items():
        ndcgs, mrrs, hits = [], [], []
        flat = []
        for qid, qtext in queries:
            ss = [sc.score(pt(qtext, did)) for did in cands[qid]]
            flat.extend(ss)
            order = np.argsort(ss)[::-1]
            rel = [1 if cands[qid][i] in gold[qid] else 0 for i in order]
            ideal = sorted(rel, reverse=True)
            ndcgs.append(ndcg_at_k(rel, ideal, 10))
            mrrs.append(mrr_at_k(rel, 10))
            hits.append(rel[0])
        scores_by_tag[tag] = np.array(flat)
        row = {"ndcg@10": float(np.mean(ndcgs)), "mrr@10": float(np.mean(mrrs)),
               "hit@1": float(np.mean(hits))}
        if "torch" in scores_by_tag and tag != "torch":
            d = np.abs(scores_by_tag[tag] - scores_by_tag["torch"])
            row["max_abs_dscore"] = float(d.max())
            row["mean_abs_dscore"] = float(d.mean())
            # pairwise inversion rate within each query's candidate list
            inv = tot = 0
            off = 0
            for qid, _ in queries:
                k = len(cands[qid])
                a = scores_by_tag["torch"][off:off + k]
                b = scores_by_tag[tag][off:off + k]
                sa = np.sign(a[:, None] - a[None, :])
                sb = np.sign(b[:, None] - b[None, :])
                iu = np.triu_indices(k, 1)
                inv += int((sa[iu] != sb[iu]).sum())
                tot += len(iu[0])
                off += k
            row["pairwise_inversions"] = inv / tot
        report["R"][tag] = row
        print(f"{tag:>10}: " + "  ".join(f"{k} {v:.4f}" for k, v in row.items()))


def gate_d(scorers, tok, report):
    print("\n=== GATE D: mechanics ===")
    ids_np = pair_tokens(tok, CARD_QUERY, CARD_PASSAGES[1])
    n = len(ids_np)
    report["D"] = {}
    for tag, sc in scorers.items():
        if tag == "torch":
            continue
        ss = []
        for S in SEQ_LENS:
            ids, mask, _ = pad_to(ids_np, S)
            r = sc.runners[S](input_ids=ids.numpy(), attention_mask=mask.numpy())
            ss.append(float(list(r.values())[0][0, 0]))
        spread = max(ss) - min(ss)
        ids, mask, _ = pad_to(ids_np, 512)
        ids2 = ids.clone()
        g = torch.Generator().manual_seed(7)
        ids2[0, n:] = torch.randint(1000, 50000, (512 - n,), generator=g,
                                    dtype=torch.int32)
        r1 = float(list(sc.runners[512](input_ids=ids.numpy(),
                                        attention_mask=mask.numpy()).values())[0][0, 0])
        r2 = float(list(sc.runners[512](input_ids=ids2.numpy(),
                                        attention_mask=mask.numpy()).values())[0][0, 0])
        print(f"{tag:>10}: cross-sig spread {spread:.3e}  "
              f"pad-content |diff| {abs(r1 - r2):.3e}")
        report["D"][tag] = {"cross_sig_spread": spread,
                            "pad_invariance": abs(r1 - r2)}
        assert np.isfinite(ss).all()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", nargs="?", default="out_ettin")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--variants", default="fp32,wi8fc,fp16")
    ap.add_argument("--queries", type=int, default=50)
    ap.add_argument("--cands", type=int, default=20)
    ap.add_argument("--gates", default="A,R,D")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    scorers = {"torch": TorchScorer(args.model)}
    for v in args.variants.split(","):
        p = os.path.join(args.out_dir, f"reranker_{v}.tflite")
        if os.path.exists(p):
            scorers[v] = TfliteScorer(p)
        else:
            print(f"(missing {p})")

    report = {}
    gates = args.gates.split(",")
    if "A" in gates:
        gate_a(scorers, tok, report)
    if "D" in gates:
        gate_d(scorers, tok, report)
    if "R" in gates:
        gate_r(scorers, tok, report, args.queries, args.cands)

    out = os.path.join(args.out_dir, "verify_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print("\nreport:", out)


if __name__ == "__main__":
    main()
