#!/usr/bin/env python3
"""Quality gates for the converted mxbai-edge-colbert-v0-32m LiteRT artifacts.

    python verify_mxbai_colbert.py <out_dir> [--gates A,D,B,E]

Host contract (pylate wheel + repo onnx_config.json, reimplemented in
convert_mxbai_colbert.py and gated there against the card's published MaxSim
scores AND kept-vector shapes): lowercase text, pad with MASK (50284), insert
[Q]=50368 / [D]=50369 at position 1, NO query expansion, 32-punctuation
skiplist filters DOCUMENT vectors host-side (specials kept), queries keep all
valid positions, per-token unit vectors, MaxSim.

Gates
  A  card example  — README Red Planet example (1 query x Venus/Mars/Jupiter/
                     Saturn). Torch must reproduce the published
                     [11.2081, 11.5308, 11.4104, 11.4756]; every variant must
                     keep Mars on top. Margins are ~0.1 at 32M — this is the
                     tightest rank-order anchor of any encoder ship.
  B  NanoSciFact   — English MaxSim retrieval nDCG@10 / recall@5 / hit@1.
  D  mechanics     — cross-signature agreement, pad-content invariance,
                     fp32 all-position finiteness on the long signature
                     (sliding-window NaN wall), unit-norm deviation.
  E  cross-variant — doc banks encoded by torch, queries by each variant
                     (server-built index, on-device queries).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "nemotron_work"))

import numpy as np
import torch

from convert_mxbai_colbert import (CARD_DOCS, CARD_QUERY, CARD_SCORES,
                                   CARD_Q_VECS, CARD_D0_VECS, PAD_ID,
                                   doc_tokens, doc_keep, maxsim, query_tokens,
                                   skiplist_ids)
# The NanoSciFact loader and nDCG helper are model-agnostic and already live
# with the Nemotron embedder converter — import them from there.
from verify_nemotron3_embed import load_nanoscifact, ndcg_at_k

MODEL_DIR = os.environ.get("MXBAI_MODEL", "mixedbread-ai/mxbai-edge-colbert-v0-32m")
SEQ_LENS = (48, 128, 256, 512)


def pad_to_signature(ids, lens):
    n = len(ids)
    S = next((s for s in lens if n <= s), lens[-1])
    a = np.full((1, S), PAD_ID, np.int32)
    m = np.zeros((1, S), np.int32)
    a[0, :min(n, S)] = ids[:S]
    m[0, :min(n, S)] = 1
    return a, m, min(n, S)


class TorchEncoder:
    name = "torch_fp32"

    def __init__(self):
        from convert_mxbai_colbert import (ColbertEncoder, load_folded_head,
                                           load_model)

        W, _ = load_folded_head(MODEL_DIR)
        self.mod = ColbertEncoder(load_model(MODEL_DIR), W).eval()

    def run(self, ids):
        a, m, n = pad_to_signature(ids, SEQ_LENS)
        with torch.inference_mode():
            out = self.mod(torch.from_numpy(a), torch.from_numpy(m)).numpy()[0]
        return out[:n]


class TfliteEncoder:
    def __init__(self, path, name):
        from ai_edge_litert.interpreter import Interpreter

        self.name = name
        self.it = Interpreter(model_path=path, num_threads=os.cpu_count())
        self.runners = {}
        for S in SEQ_LENS:
            try:
                self.runners[S] = self.it.get_signature_runner(f"encode_{S}")
            except Exception:
                pass
        self.lens = sorted(self.runners)

    def run(self, ids):
        a, m, n = pad_to_signature(ids, self.lens)
        out = list(self.runners[a.shape[1]](input_ids=a, attention_mask=m)
                   .values())[0][0]
        return out[:n]


def encode_query(enc, tok, text):
    return enc.run(query_tokens(tok, text))


def encode_doc(enc, tok, skip, text):
    ids = doc_tokens(tok, text)
    v = enc.run(ids)
    return v[doc_keep(ids[:len(v)], skip)]


def build_bank(enc, tok, skip, docs, log_every=500):
    banks, bounds, off = [], [], 0
    t0 = time.time()
    for i, text in enumerate(docs):
        v = encode_doc(enc, tok, skip, text)
        banks.append(v)
        bounds.append((off, off + len(v)))
        off += len(v)
        if log_every and (i + 1) % log_every == 0:
            print(f"    [{enc.name}] {i+1}/{len(docs)} docs "
                  f"{time.time()-t0:.0f}s", flush=True)
    return np.concatenate(banks, 0), bounds


def score_queries(enc, tok, queries, bank, bounds):
    out = []
    for qtext in queries:
        q = encode_query(enc, tok, qtext)
        sims = q @ bank.T
        out.append(np.array([sims[:, a:b].max(axis=1).sum() for a, b in bounds]))
    return out


def retrieval_metrics(score_rows, golds):
    ndcgs, r5, r1 = [], [], []
    for scores, gold in zip(score_rows, golds):
        order = np.argsort(-scores)
        rel = [1.0 if j in gold else 0.0 for j in order[:10]]
        ndcgs.append(ndcg_at_k(rel, [1.0] * max(1, len(gold)), 10))
        r5.append(float(any(rel[:5])))
        r1.append(float(rel[0]))
    return {"ndcg@10": round(float(np.mean(ndcgs)), 4),
            "recall@5": round(float(np.mean(r5)), 4),
            "hit@1": round(float(np.mean(r1)), 4)}


def gate_a(encoders, tok, skip, report):
    print("\n=== GATE A: card Red Planet example, MaxSim oracle ===")
    for enc in encoders:
        q = encode_query(enc, tok, CARD_QUERY)
        dvecs = [encode_doc(enc, tok, skip, d) for d in CARD_DOCS]
        scores = [maxsim(q, dv) for dv in dvecs]
        ok = int(np.argmax(scores)) == 1  # Mars on top
        row = {"scores": [round(s, 4) for s in scores], "mars_top": ok,
               "q_vecs": len(q), "d0_vecs": len(dvecs[0])}
        if enc.name == "torch_fp32":
            row["max_abs_vs_published"] = round(
                max(abs(a - b) for a, b in zip(scores, CARD_SCORES)), 4)
            assert len(q) == CARD_Q_VECS and len(dvecs[0]) == CARD_D0_VECS
        report.setdefault("A_card", {})[enc.name] = row
        print(f"  {enc.name:11s} {row['scores']} mars_top {ok}"
              + (f"  vs published max|d| {row['max_abs_vs_published']}"
                 if "max_abs_vs_published" in row else ""))
        assert ok, f"{enc.name}: Mars is not the top passage"


def gate_retrieval(tag, encoders, tok, skip, docs, queries, golds, report,
                   cross=False):
    torch_enc = next((e for e in encoders if e.name == "torch_fp32"), None)
    shared = None
    if cross:
        assert torch_enc is not None
        print(f"  building torch doc bank ({len(docs)} docs) ...")
        shared = build_bank(torch_enc, tok, skip, docs)
    for enc in encoders:
        bank, bounds = shared if cross else build_bank(enc, tok, skip, docs)
        rows = score_queries(enc, tok, queries, bank, bounds)
        row = retrieval_metrics(rows, golds)
        row["vectors"] = int(bank.shape[0])
        report.setdefault(tag, {})[enc.name] = row
        print(f"  {enc.name:11s} nDCG@10 {row['ndcg@10']:.4f} "
              f"recall@5 {row['recall@5']:.4f} hit@1 {row['hit@1']:.4f}")


def gate_b(encoders, tok, skip, report, n_docs, cross=False):
    tag = "E_cross_scifact" if cross else "B_nanoscifact"
    print(f"\n=== GATE {'E' if cross else 'B'}: NanoSciFact MaxSim "
          f"({'torch docs x variant queries' if cross else 'same-variant'}, "
          f"corpus {n_docs}) ===")
    docs, queries, gold = load_nanoscifact(n_docs)
    ids = [i for i, _ in docs]
    texts = [t for _, t in docs]
    idx_gold, qtexts = [], []
    for qid, qtext in queries:
        g = gold.get(qid, set())
        if not g:
            continue
        qtexts.append(qtext)
        idx_gold.append({j for j, di in enumerate(ids) if di in g})
    print(f"  {len(qtexts)} queries, {len(texts)} docs")
    gate_retrieval(tag, encoders, tok, skip, texts, qtexts, idx_gold, report,
                   cross)


def gate_d(encoders, tok, report):
    print("\n=== GATE D: mechanics on the shipped artifacts ===")
    text = "Mount Fuji is the highest mountain in Japan, at 3,776 meters."
    ids = doc_tokens(tok, text)
    n = len(ids)
    print(f"  probe doc = {n} tokens")
    for enc in encoders:
        row = {}
        if isinstance(enc, TfliteEncoder):
            vecs = {}
            for S in enc.lens:
                if S < n:
                    continue
                a = np.full((1, S), PAD_ID, np.int32)
                m = np.zeros((1, S), np.int32)
                a[0, :n] = ids
                m[0, :n] = 1
                v = list(enc.runners[S](input_ids=a, attention_mask=m)
                         .values())[0][0]
                vecs[S] = v
            base = vecs[min(vecs)][:n]
            row["cross_sig_max_abs"] = {
                str(S): float(np.abs(v[:n] - base).max()) for S, v in vecs.items()}

            S = max(vecs)
            v1 = vecs[S]
            if enc.name == "fp32":
                assert np.isfinite(v1).all(), (
                    "fp32 NaN at some position in the long signature — the "
                    "sliding-window diagonal guard failed (int8 would hide this)")
                row["fp32_all_positions_finite"] = True
            a2 = np.full((1, S), PAD_ID, np.int32)
            m2 = np.zeros((1, S), np.int32)
            a2[0, :n] = ids
            m2[0, :n] = 1
            a2[0, n:] = np.random.default_rng(1).integers(1000, 50000, S - n)
            v2 = list(enc.runners[S](input_ids=a2, attention_mask=m2)
                      .values())[0][0]
            row["pad_invariance_max_abs"] = float(np.abs(v1[:n] - v2[:n]).max())
            row["unit_norm_max_dev"] = float(
                np.abs(np.linalg.norm(v1[:n], axis=1) - 1.0).max())
            print(f"  {enc.name:11s} cross-sig max|d| {row['cross_sig_max_abs']}")
            print(f"  {'':11s} pad-invariance {row['pad_invariance_max_abs']:.3e} "
                  f"(must be 0) | unit-norm dev {row['unit_norm_max_dev']:.2e}")
        report.setdefault("D_mechanics", {})[enc.name] = row


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--gates", default="A,D,B,E")
    ap.add_argument("--scifact-docs", type=int, default=600)
    ap.add_argument("--no-torch", action="store_true")
    ap.add_argument("--variants", default="fp32,wi8fc,fp16")
    ap.add_argument("--report", default="verify_report.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    skip = skiplist_ids(tok)

    encoders = []
    if not args.no_torch:
        print("loading torch reference ...")
        encoders.append(TorchEncoder())
    for v in args.variants.split(","):
        p = os.path.join(args.out_dir, f"colbert_{v}.tflite")
        if os.path.exists(p):
            print(f"loading {p} ({os.path.getsize(p)/1e6:.0f} MB) ...")
            encoders.append(TfliteEncoder(p, v))
        else:
            print(f"(missing {p})")

    report = {"variants": [e.name for e in encoders]}
    gates = set(args.gates.upper().split(","))
    if "A" in gates:
        gate_a(encoders, tok, skip, report)
    if "D" in gates:
        gate_d(encoders, tok, report)
    if "B" in gates:
        gate_b(encoders, tok, skip, report, args.scifact_docs)
    if "E" in gates:
        gate_b(encoders, tok, skip, report, args.scifact_docs, cross=True)

    out = os.path.join(args.out_dir, args.report)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
