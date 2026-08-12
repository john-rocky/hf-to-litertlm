#!/usr/bin/env python3
"""Quality gates for the converted LFM2.5-ColBERT-350M LiteRT artifacts.

    .venv-092/bin/python scripts/verify_lfm25_colbert.py <out_dir> [--gates A,B,C]

Unlike the dense embedders in this repo, most of this file IS the deliverable:
a late-interaction model is useless without the host contract, so the pylate
1.6.0 behaviour is reimplemented here in ~40 lines and gated against the torch
reference. Everything below is read out of pylate's source and the repo's
config, never inferred:

  query   tokenize to query_length-1 = 31, padding="max_length", pad = EOS (7);
          insert [Q] (64400) at position 1; attention_mask keeps 0 on the
          expansion positions (attend_to_expansion_tokens: false); ALL 32
          vectors are scored (no skiplist on queries).
  doc     tokenize to document_length-1 = 511; insert [D] (64401) at position
          1; keep positions where skiplist_mask AND attention_mask.
  score   MaxSim — per query vector, the max dot product over kept doc vectors,
          summed over query vectors. Vectors are unit length, so dot == cosine.

  A  card example  — ranking oracle on the base card's own query/passage set
  B  NanoSciFact   — MaxSim retrieval nDCG@10 / recall@5, the task the model is for
  C  mechanics     — per-token parity, expansion-vector survival, pad-id
                     sensitivity, and doc padded-vs-unpadded drift on the
                     shipped artifacts
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

# The NanoSciFact loader and nDCG helper are model-agnostic and already live
# with the Nemotron embedder converter — import them from there.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "nemotron_work"))
from verify_nemotron3_embed import load_nanoscifact, ndcg_at_k

MODEL_DIR = os.environ.get("LFM25_COLBERT_MODEL",
                           "LiquidAI/LFM2.5-ColBERT-350M")
PAD_ID, BOS_ID, Q_ID, D_ID = 7, 1, 64400, 64401
QUERY_LEN, DOC_LEN, DIM = 32, 512, 128
SEQ_LENS = (32, 128, 256, 512)

QUERIES = [
    "What is the capital of France?",
    "Which city is Japan's capital?",
]
DOCUMENTS = [
    "Paris is the capital and largest city of France. Located on the Seine River in northern France, it serves as the country's political, economic, and cultural center.",
    "Tokyo, officially the Tokyo Metropolis, is the capital of Japan. It is the most populous metropolitan area in the world and serves as Japan's administrative, financial, and commercial hub.",
    "Berlin is the capital and largest city of Germany. Reunified in 1990 after the fall of the Berlin Wall, it now serves as a major cultural and political center in Europe.",
]


# ---------------------------------------------------------------- host contract
def skiplist_ids(tok):
    words = json.load(open(os.path.join(MODEL_DIR,
                                        "config_sentence_transformers.json")))["skiplist_words"]
    return set(tok.convert_tokens_to_ids(w) for w in words)


def query_tokens(tok, text):
    """-> ids[32], attention_mask[32]. All 32 positions are scored."""
    e = tok(text, padding="max_length", max_length=QUERY_LEN - 1, truncation=True)
    ids = [e["input_ids"][0], Q_ID] + e["input_ids"][1:]
    am = [e["attention_mask"][0], 1] + e["attention_mask"][1:]
    return np.array(ids[:QUERY_LEN], np.int32), np.array(am[:QUERY_LEN], np.int32)


def doc_tokens(tok, text):
    """-> ids[n], attention_mask[n] (unpadded); n <= 512."""
    e = tok(text, truncation=True, max_length=DOC_LEN - 1)
    ids = [e["input_ids"][0], D_ID] + e["input_ids"][1:]
    am = [e["attention_mask"][0], 1] + e["attention_mask"][1:]
    return np.array(ids, np.int32), np.array(am, np.int32)


def maxsim(q, d):
    """q [Nq,128], d [Nd,128] unit vectors -> ColBERT MaxSim score."""
    return float((q @ d.T).max(axis=1).sum())


# ---------------------------------------------------------------- encoders
def pad_to_signature(ids, am, lens=SEQ_LENS):
    """Right-pad to the smallest available signature length, pylate's pad id."""
    n = len(ids)
    S = next((s for s in lens if n <= s), lens[-1])
    a = np.full(S, PAD_ID, np.int32)
    m = np.zeros(S, np.int32)
    a[:min(n, S)] = ids[:S]
    m[:min(n, S)] = am[:S]
    return a, m, min(n, S)


class TorchEncoder:
    """The reference. `padded=True` mirrors what a fixed-shape signature sees, so
    a torch-vs-tflite number isolates the export instead of folding in the
    padding delta; `padded=False` is pylate's own default (documents unpadded)."""

    def __init__(self, padded=True, name=None):
        from convert_lfm25_colbert import ColbertEncoder, load_dense, load_model

        self.mod = ColbertEncoder(load_model(MODEL_DIR), load_dense(MODEL_DIR)).eval()
        self.padded = padded
        self.name = name or ("torch_padded" if padded else "torch_unpadded")

    def run(self, ids, am):
        n = len(ids)
        if self.padded:
            ids, am, n = pad_to_signature(ids, am)
        with torch.inference_mode():
            out = self.mod(torch.tensor(ids[None], dtype=torch.int32),
                           torch.tensor(am[None], dtype=torch.int32)).numpy()[0]
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

    def run(self, ids, am):
        a, m, n = pad_to_signature(ids, am, self.lens)
        out = list(self.runners[len(a)](input_ids=a[None], attention_mask=m[None])
                   .values())[0][0]
        return out[:n]


def encode_query(enc, tok, text):
    ids, am = query_tokens(tok, text)
    return enc.run(ids, am)                      # all 32 vectors kept


def encode_doc(enc, tok, text, skip):
    ids, am = doc_tokens(tok, text)
    v = enc.run(ids, am)
    keep = (am[:len(v)] == 1) & ~np.isin(ids[:len(v)], list(skip))
    return v[keep]


# ---------------------------------------------------------------- gates
def gate_a(encoders, tok, skip, report):
    print("\n=== GATE A: card example, MaxSim ranking oracle ===")
    for enc in encoders:
        Q = [encode_query(enc, tok, q) for q in QUERIES]
        D = [encode_doc(enc, tok, d, skip) for d in DOCUMENTS]
        s = np.array([[maxsim(q, d) for d in D] for q in Q])
        want = np.array([0, 1])
        margin = float(min(s[i, want[i]] - np.max(np.delete(s[i], want[i]))
                           for i in range(len(want))))
        row = {"argmax_correct": bool((s.argmax(1) == want).all()),
               "min_margin": round(margin, 4),
               "scores": [[round(float(x), 3) for x in r] for r in s]}
        report.setdefault("A_card", {})[enc.name] = row
        print(f"  {enc.name:11s} argmax {row['argmax_correct']}  margin {margin:+.3f}  "
              f"{row['scores']}")
        assert row["argmax_correct"], f"{enc.name} ranks the card example wrong"


def gate_b(encoders, tok, skip, report, n_docs):
    print(f"\n=== GATE B: NanoSciFact MaxSim retrieval (corpus {n_docs}) ===")
    docs, queries, gold = load_nanoscifact(n_docs)
    print(f"  {len(queries)} queries, {len(docs)} docs")
    for enc in encoders:
        t0 = time.time()
        # One flat [N,128] bank + per-doc segment bounds: MaxSim is then a single
        # matmul per query plus a segment max, instead of 600 small matmuls.
        banks, bounds, off = [], [], 0
        for i, (_, text) in enumerate(docs):
            v = encode_doc(enc, tok, text, skip)
            banks.append(v)
            bounds.append((off, off + len(v)))
            off += len(v)
            if (i + 1) % 200 == 0:
                print(f"    [{enc.name}] {i+1}/{len(docs)} docs "
                      f"{time.time()-t0:.0f}s", flush=True)
        bank = np.concatenate(banks, 0)
        doc_ids = [i for i, _ in docs]
        ndcgs, r5, r1 = [], [], []
        for qid, qtext in queries:
            g = gold.get(qid, set())
            if not g:
                continue
            q = encode_query(enc, tok, qtext)
            sims = q @ bank.T                                   # [32, N]
            scores = np.array([sims[:, a:b].max(axis=1).sum() for a, b in bounds])
            order = np.argsort(-scores)
            rel = [1.0 if doc_ids[j] in g else 0.0 for j in order[:10]]
            ndcgs.append(ndcg_at_k(rel, [1.0] * len(g), 10))
            r5.append(float(any(rel[:5])))
            r1.append(float(rel[0]))
        row = {"ndcg@10": round(float(np.mean(ndcgs)), 4),
               "recall@5": round(float(np.mean(r5)), 4),
               "hit@1": round(float(np.mean(r1)), 4),
               "n_queries": len(ndcgs), "vectors": int(bank.shape[0]),
               "vecs_per_doc": round(bank.shape[0] / len(docs), 1)}
        report.setdefault("B_retrieval", {})[enc.name] = row
        print(f"  {enc.name:11s} nDCG@10 {row['ndcg@10']:.4f} recall@5 "
              f"{row['recall@5']:.4f} hit@1 {row['hit@1']:.4f}  "
              f"({row['vectors']} vectors, {row['vecs_per_doc']}/doc)")


def gate_c(encoders, tok, skip, report):
    print("\n=== GATE C: mechanics on the shipped artifacts ===")
    ref = encoders[0]
    qids, qam = query_tokens(tok, QUERIES[0])
    n_real = int(qam.sum())
    dids, dam = doc_tokens(tok, DOCUMENTS[1])
    # Compare LIKE FOR LIKE. A tflite signature is a fixed shape, so it always
    # sees a padded document; running the torch reference unpadded would fold
    # the padding delta into what reads as conversion error. Pad the reference
    # to the same signature length and report the padding delta separately.
    n = len(dids)
    S_doc = next(s for s in SEQ_LENS if s >= n)
    base_q = ref.run(qids, qam)
    base_d = ref.run(dids, dam)
    unpadded = TorchEncoder(padded=False)
    pad_delta = float((base_d * unpadded.run(dids, dam)).sum(-1).min())
    print(f"  reference doc: {n} tokens -> encode_{S_doc}; padded-vs-unpadded "
          f"min per-token cos {pad_delta:.6f} (a property of the model, not the export)")
    report.setdefault("C_mechanics", {})["doc_padded_vs_unpadded_min_cos"] = round(pad_delta, 6)
    for enc in encoders:
        row = {}
        q = enc.run(qids, qam)
        d = enc.run(dids, dam)
        row["query_min_cos_vs_torch"] = round(float((q * base_q).sum(-1).min()), 6)
        row["doc_min_cos_vs_torch"] = round(float((d * base_d).sum(-1).min()), 6)
        row["unit_norm_max_dev"] = round(
            float(np.abs(np.linalg.norm(q, axis=1) - 1).max()), 6)
        # expansion vectors must stay distinct — collapse = padding states zeroed
        exp = q[n_real:]
        row["expansion_vectors"] = int(len(exp))
        row["expansion_spread"] = round(float(np.abs(exp - exp[0]).max()), 4)
        # pad id is part of the contract (pads are NOT zeroed in this model)
        alt = qids.copy()
        alt[n_real:] = 0
        row["pad_id_sensitivity"] = round(
            float(np.abs(enc.run(alt, qam) - q).max()), 4)
        report.setdefault("C_mechanics", {})[enc.name] = row
        print(f"  {enc.name:11s} per-token cos vs torch: query "
              f"{row['query_min_cos_vs_torch']:.6f} doc {row['doc_min_cos_vs_torch']:.6f}"
              f" | unit-norm dev {row['unit_norm_max_dev']:.1e}")
        print(f"  {'':11s} expansion {row['expansion_vectors']} vectors, spread "
              f"{row['expansion_spread']:.4f} (must be > 0) | pad-id sensitivity "
              f"{row['pad_id_sensitivity']:.4f} (must be > 0)")
        assert row["expansion_spread"] > 1e-3, "expansion vectors collapsed"
        assert row["unit_norm_max_dev"] < 1e-3, "vectors are not unit length"

    # Doc drift from routing the same text into a longer signature.
    for enc in encoders:
        if not isinstance(enc, TfliteEncoder):
            continue
        n = len(dids)
        outs = {}
        for S in enc.lens:
            if S < n:
                continue
            a = np.full((1, S), PAD_ID, np.int32)
            m = np.zeros((1, S), np.int32)
            a[0, :n], m[0, :n] = dids, dam
            outs[S] = list(enc.runners[S](input_ids=a,
                                          attention_mask=m).values())[0][0][:n]
        if len(outs) > 1:
            lo = min(outs)
            drift = {str(S): round(float((v * outs[lo]).sum(-1).min()), 6)
                     for S, v in outs.items()}
            report.setdefault("C_mechanics", {})[enc.name]["cross_sig_min_cos"] = drift
            print(f"  {enc.name:11s} same doc via encode_{sorted(outs)}: min per-token "
                  f"cos vs encode_{lo} {drift}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--gates", default="A,B,C")
    ap.add_argument("--docs", type=int, default=600)
    ap.add_argument("--no-torch", action="store_true")
    ap.add_argument("--unpadded-ref", action="store_true",
                    help="also score pylate's own unpadded document path, "
                         "which isolates the cost of a static shape")
    ap.add_argument("--variants", default="fp32,wi8fc,fp16")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    assert tok.pad_token_id == PAD_ID, (tok.pad_token_id, PAD_ID)
    skip = skiplist_ids(tok)
    print(f"skiplist: {len(skip)} token ids")

    encoders = []
    if not args.no_torch:
        print("loading torch reference (padded, matches a fixed-shape signature) ...")
        encoders.append(TorchEncoder(padded=True))
        if args.unpadded_ref:
            print("loading torch reference (unpadded = pylate's own default) ...")
            encoders.append(TorchEncoder(padded=False))
    for v in args.variants.split(","):
        p = os.path.join(args.out_dir, f"colbert_{v}.tflite")
        if os.path.exists(p):
            print(f"loading {p} ({os.path.getsize(p)/1e6:.0f} MB) ...")
            encoders.append(TfliteEncoder(p, v))
        else:
            print(f"(missing {p})")

    out = os.path.join(args.out_dir, "verify_report.json")
    report = json.load(open(out)) if os.path.exists(out) else {}
    report["variants"] = [e.name for e in encoders]
    report["model"] = MODEL_DIR
    gates = set(args.gates.upper().split(","))
    if "A" in gates:
        gate_a(encoders, tok, skip, report)
    if "C" in gates:
        gate_c(encoders, tok, skip, report)
    if "B" in gates:
        gate_b(encoders, tok, skip, report, args.docs)

    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
