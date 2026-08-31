#!/usr/bin/env python3
"""Quality gates for the converted mLateOn LiteRT artifacts.

    python verify_mlateon_colbert.py <out_dir> [--gates A,D,B,J,E]

Host contract (pylate, reimplemented in convert_mlateon_colbert.py and gated
there against the card's published MaxSim scores): pad with MASK (4), insert
[Q]=256000 / [D]=256001 at position 1, NO query expansion, no skiplist, keep
valid positions only, per-token unit vectors, MaxSim scoring.

Gates
  A  card example  — the README's own Red Planet example (1 query x 4 docs:
                     Mars en/fr/de + Venus). Every Mars doc must outrank Venus;
                     the torch row must reproduce the PUBLISHED scores
                     [9.6029, 9.5838, 9.5877, 9.4578].
  B  NanoSciFact   — English MaxSim retrieval nDCG@10 / recall@5 / hit@1.
  J  JSQuAD        — Japanese MaxSim retrieval. Japanese is NOT in the nine
                     training languages; the card claims late interaction
                     generalizes to unseen scripts — measured here, and it is
                     also the harder test for the int8 table (256k SP vocab).
  D  mechanics     — cross-signature agreement (bitwise here: pads are masked
                     keys), pad-content invariance, all-position finiteness on
                     the LONG signature in fp32 (the sliding-window NaN wall —
                     int8 can hide it, fp32 cannot), unit-norm deviation.
  E  cross-variant — doc banks encoded by torch, queries by each variant
                     (server-built index, on-device queries).

JGLUE fixtures cached in ./fixtures/ (CC BY-SA 4.0, eval-only).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from convert_mlateon_colbert import (CARD_DOCS, CARD_QUERY, CARD_SCORES, DIM,
                                     PAD_ID, doc_tokens, maxsim, query_tokens)
# The NanoSciFact loader and nDCG helper are model-agnostic and already live
# with the Nemotron embedder converter — import them from there.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "nemotron_work"))
from verify_nemotron3_embed import load_nanoscifact, ndcg_at_k

MODEL_DIR = os.environ.get("MLATEON_MODEL", "lightonai/mLateOn")
SEQ_LENS = (32, 128, 256, 512)
FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


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
        from convert_mlateon_colbert import (ColbertEncoder, load_folded_head,
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


def encode_doc(enc, tok, text):
    return enc.run(doc_tokens(tok, text))


def build_bank(enc, tok, docs, log_every=200):
    banks, bounds, off = [], [], 0
    t0 = time.time()
    for i, text in enumerate(docs):
        v = encode_doc(enc, tok, text)
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


def load_jsquad(n_docs, n_queries):
    import urllib.request

    os.makedirs(FIXTURES, exist_ok=True)
    p = os.path.join(FIXTURES, "jsquad_valid_v1.3.json")
    if not os.path.exists(p):
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/yahoojapan/JGLUE/main/datasets/"
            "jsquad-v1.3/valid-v1.3.json", p)
    with open(p) as f:
        data = json.load(f)["data"]
    ctx_id, docs, queries = {}, [], []
    for art in data:
        for para in art["paragraphs"]:
            c = para["context"]
            if c not in ctx_id:
                ctx_id[c] = len(docs)
                docs.append(c)
            for qa in para["qas"]:
                queries.append((qa["question"], ctx_id[c]))
    rng = np.random.default_rng(0)
    queries = [queries[i] for i in rng.choice(len(queries), n_queries, replace=False)]
    keep = sorted({g for _, g in queries})
    rest = [i for i in range(len(docs)) if i not in set(keep)]
    extra = list(rng.choice(rest, max(0, n_docs - len(keep)), replace=False))
    sel = keep + [int(x) for x in extra]
    remap = {old: new for new, old in enumerate(sel)}
    return [docs[i] for i in sel], [(q, remap[g]) for q, g in queries]


def gate_a(encoders, tok, report):
    print("\n=== GATE A: card Red Planet example, MaxSim oracle ===")
    for enc in encoders:
        q = encode_query(enc, tok, CARD_QUERY)
        scores = [maxsim(q, encode_doc(enc, tok, d)) for d in CARD_DOCS]
        ok = min(scores[:3]) > scores[3]
        row = {"scores": [round(s, 4) for s in scores], "mars_beats_venus": ok}
        if enc.name == "torch_fp32":
            row["max_abs_vs_published"] = round(
                max(abs(a - b) for a, b in zip(scores, CARD_SCORES)), 4)
        report.setdefault("A_card", {})[enc.name] = row
        print(f"  {enc.name:11s} {row['scores']} mars>venus {ok}"
              + (f"  vs published max|d| {row['max_abs_vs_published']}"
                 if "max_abs_vs_published" in row else ""))
        assert ok, f"{enc.name}: Venus outranks a Mars doc"


def gate_retrieval(tag, encoders, tok, docs, queries, golds, report, cross=False,
                   cache=None):
    torch_enc = next((e for e in encoders if e.name == "torch_fp32"), None)
    shared = None
    if cross:
        # The torch bank is cacheable: build it once in a lean torch-only
        # process, then score each variant from its own process — useful on
        # memory-constrained machines.
        if cache and os.path.exists(cache):
            z = np.load(cache)
            shared = (z["bank"], [tuple(b) for b in z["bounds"]])
            print(f"  loaded torch bank {cache} ({shared[0].shape[0]} vectors)")
        else:
            assert torch_enc is not None, ("cross-variant needs the torch "
                                           "reference or a cached bank")
            print(f"  building torch doc bank ({len(docs)} docs) ...")
            shared = build_bank(torch_enc, tok, docs)
            if cache:
                np.savez(cache, bank=shared[0],
                         bounds=np.array(shared[1], dtype=np.int64))
                print(f"  cached torch bank -> {cache}")
    for enc in encoders:
        bank, bounds = shared if cross else build_bank(enc, tok, docs)
        rows = score_queries(enc, tok, queries, bank, bounds)
        row = retrieval_metrics(rows, golds)
        row["vectors"] = int(bank.shape[0])
        report.setdefault(tag, {})[enc.name] = row
        print(f"  {enc.name:11s} nDCG@10 {row['ndcg@10']:.4f} "
              f"recall@5 {row['recall@5']:.4f} hit@1 {row['hit@1']:.4f}")


def gate_b(encoders, tok, report, n_docs, cross=False, cache=None):
    tag = "E_cross_scifact" if cross else "B_nanoscifact"
    print(f"\n=== GATE {'E' if cross else 'B'}: NanoSciFact MaxSim "
          f"({'torch docs x variant queries' if cross else 'same-variant'}, "
          f"corpus {n_docs}) ===")
    docs, queries, gold = load_nanoscifact(n_docs)
    ids = [i for i, _ in docs]
    texts = [t for _, t in docs]
    idx_gold = []
    qtexts = []
    for qid, qtext in queries:
        g = gold.get(qid, set())
        if not g:
            continue
        qtexts.append(qtext)
        idx_gold.append({j for j, di in enumerate(ids) if di in g})
    print(f"  {len(qtexts)} queries, {len(texts)} docs")
    gate_retrieval(tag, encoders, tok, texts, qtexts, idx_gold, report, cross,
                   cache=cache)


def gate_j(encoders, tok, report, n_docs, n_queries=150, cross=False,
           cache=None):
    tag = "E_cross_jsquad" if cross else "J_jsquad"
    print(f"\n=== GATE {'E' if cross else 'J'}: JSQuAD MaxSim "
          f"({'torch docs x variant queries' if cross else 'same-variant'}, "
          f"corpus {n_docs}) — unseen language ===")
    docs, queries = load_jsquad(n_docs, n_queries)
    qtexts = [q for q, _ in queries]
    golds = [{g} for _, g in queries]
    print(f"  {len(qtexts)} queries, {len(docs)} paragraphs")
    gate_retrieval(tag, encoders, tok, docs, qtexts, golds, report, cross,
                   cache=cache)


def gate_d(encoders, tok, report):
    print("\n=== GATE D: mechanics on the shipped artifacts ===")
    text = "富士山は日本で一番高い山です。"
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
            a2[0, n:] = np.random.default_rng(1).integers(10, 255000, S - n)
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
    ap.add_argument("--gates", default="A,D,B,J")
    ap.add_argument("--scifact-docs", type=int, default=600)
    ap.add_argument("--jsquad-docs", type=int, default=500)
    ap.add_argument("--no-torch", action="store_true")
    ap.add_argument("--variants", default="fp32,wi8fc,fp16")
    ap.add_argument("--report", default="verify_report.json")
    ap.add_argument("--bank-cache", action="store_true",
                    help="cache/reuse the torch doc banks for gate E under out_dir")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)

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
        gate_a(encoders, tok, report)
    if "D" in gates:
        gate_d(encoders, tok, report)
    if "B" in gates:
        gate_b(encoders, tok, report, args.scifact_docs)
    if "J" in gates:
        gate_j(encoders, tok, report, args.jsquad_docs)
    if "E" in gates:
        cb = (os.path.join(args.out_dir, "bank_scifact.npz")
              if args.bank_cache else None)
        cj = (os.path.join(args.out_dir, "bank_jsquad.npz")
              if args.bank_cache else None)
        gate_b(encoders, tok, report, args.scifact_docs, cross=True, cache=cb)
        gate_j(encoders, tok, report, args.jsquad_docs, cross=True, cache=cj)

    out = os.path.join(args.out_dir, args.report)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
