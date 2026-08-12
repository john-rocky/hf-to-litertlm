#!/usr/bin/env python3
"""Quality gates for the converted LFM2.5-Embedding-350M LiteRT artifacts.

    .venv-092/bin/python scripts/verify_lfm25_embedding.py <out_dir> [--gates A,B,C,D]

Same gate set as the Nemotron / granite embedders (and it reuses those modules'
model-agnostic metric + dataset loaders), but the model contract differs again:
CLS pooling, 1024 dims, pad id 0, and asymmetric `query: ` / `document: `
prompts WITH the trailing space — read off `config_sentence_transformers.json`,
which agrees with the card prose. Gate C therefore runs the prefixed text, and
`--prefix-ab` measures the prefix on a set with enough resolution to settle it
(the card's own 2x3 example is far below the resolution needed).

  A  card example  — the card's 2-query/3-document ranking, cross-variant
  B  STS17         — multilingual + cross-lingual Spearman (graded, cannot
                     saturate to a flattering floor)
  C  NanoSciFact   — retrieval nDCG@10 / recall@5, the task the model is for
  D  mechanics     — cross-signature agreement, pad-content invariance, mask
                     liveness, short-text-in-longest-signature, on the shipped
                     artifacts

Every gate runs on each variant (torch fp32 reference, fp32/wi8fc/fp16 tflite)
so the numbers are comparable; a variant is judged against the reference, never
against an absolute threshold.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

# The STS17 / NanoSciFact loaders and the metric helpers are model-agnostic
# and already live with the Nemotron embedder converter — import them from
# there rather than keeping a second copy in sync.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "nemotron_work"))
from verify_nemotron3_embed import (
    load_nanoscifact, load_sts17, ndcg_at_k, spearman,
)

MODEL_DIR = os.environ.get("LFM25_EMBED_MODEL",
                           "LiquidAI/LFM2.5-Embedding-350M")
PAD_ID = 0
DIM = 1024
SEQ_LENS = (64, 128, 256, 512)
Q_PREFIX = "query: "
D_PREFIX = "document: "

# The card's own usage example (README "Encoding queries and documents"). It
# prints no numbers, so this is a RANKING oracle, not a value oracle: q0 must
# pick the Paris passage and q1 the Tokyo one.
QUERIES = [
    "What is the capital of France?",
    "Which city is Japan's capital?",
]
DOCUMENTS = [
    "Paris is the capital and largest city of France. Located on the Seine River in northern France, it serves as the country's political, economic, and cultural center.",
    "Tokyo, officially the Tokyo Metropolis, is the capital of Japan. It is the most populous metropolitan area in the world and serves as Japan's administrative, financial, and commercial hub.",
    "Berlin is the capital and largest city of Germany. Reunified in 1990 after the fall of the Berlin Wall, it now serves as a major cultural and political center in Europe.",
]


class TorchEncoder:
    name = "torch_fp32"

    def __init__(self, tok):
        from convert_lfm25_embedding import Embedder, load_model

        self.mod = Embedder(load_model(MODEL_DIR)).eval()
        self.tok = tok

    def encode_ids(self, ids):
        t = torch.tensor([ids], dtype=torch.int32)
        m = torch.ones(1, len(ids), dtype=torch.int32)
        with torch.inference_mode():
            return self.mod(t, m).numpy()[0]


class TfliteEncoder:
    def __init__(self, path, name):
        from ai_edge_litert.interpreter import Interpreter

        self.name = name
        self.it = Interpreter(model_path=path, num_threads=os.cpu_count())
        self.runners = {}
        for S in SEQ_LENS:
            try:
                self.runners[S] = self.it.get_signature_runner(f"embed_{S}")
            except Exception:
                pass
        self.lens = sorted(self.runners)

    def encode_ids(self, ids):
        S = next((s for s in self.lens if len(ids) <= s), self.lens[-1])
        ids = ids[:S]
        n = len(ids)
        a = np.full((1, S), PAD_ID, dtype=np.int32)
        m = np.zeros((1, S), dtype=np.int32)
        a[0, :n] = ids
        m[0, :n] = 1
        return list(self.runners[S](input_ids=a, attention_mask=m).values())[0][0]


def encode_texts(enc, tok, texts, prefix="", max_tokens=512, log_every=0):
    out = np.empty((len(texts), DIM), dtype=np.float32)
    t0 = time.time()
    for i, t in enumerate(texts):
        ids = tok(prefix + t, add_special_tokens=True)["input_ids"][:max_tokens]
        out[i] = enc.encode_ids(ids)
        if log_every and (i + 1) % log_every == 0:
            el = time.time() - t0
            print(f"    [{enc.name}] {i+1}/{len(texts)}  {el:.0f}s "
                  f"({el/(i+1)*1000:.0f} ms/text)", flush=True)
    return out


def gate_a(encoders, tok, report):
    print("\n=== GATE A: card usage example (ranking oracle) ===")
    ref = None
    for enc in encoders:
        q = encode_texts(enc, tok, QUERIES, Q_PREFIX)
        d = encode_texts(enc, tok, DOCUMENTS, D_PREFIX)
        s = q @ d.T
        # q0 -> Paris (0), q1 -> Tokyo (1); Berlin is the distractor.
        want = np.array([0, 1])
        margin = float(min(s[i, want[i]] - np.max(np.delete(s[i], want[i]))
                           for i in range(len(want))))
        row = {"argmax_correct": bool((s.argmax(1) == want).all()),
               "min_margin": round(margin, 4),
               "sims": [[round(float(x), 4) for x in r] for r in s]}
        if ref is None:
            ref = s
        else:
            row["max_abs_vs_torch"] = round(float(np.abs(s - ref).max()), 4)
        report.setdefault("A_card", {})[enc.name] = row
        print(f"  {enc.name:11s} argmax-correct {row['argmax_correct']}  "
              f"min-margin {row['min_margin']:+.4f}  sims {row['sims']}"
              + (f"  vs torch {row['max_abs_vs_torch']:.4f}"
                 if "max_abs_vs_torch" in row else ""))
        assert row["argmax_correct"], f"{enc.name} ranks the card example wrong"


def gate_b(encoders, tok, report, limit):
    print(f"\n=== GATE B: STS17 Spearman (<= {limit} pairs/lang, no prefix) ===")
    data = load_sts17(limit)
    if not data:
        print("  no data — skipped")
        return
    for enc in encoders:
        per_lang = {}
        for pair, rows in data.items():
            a = encode_texts(enc, tok, [r[0] for r in rows])
            b = encode_texts(enc, tok, [r[1] for r in rows])
            per_lang[pair] = round(spearman((a * b).sum(1),
                                            np.array([r[2] for r in rows])), 4)
        report.setdefault("B_sts17", {})[enc.name] = {
            "per_lang": per_lang,
            "mean": round(float(np.mean(list(per_lang.values()))), 4)}
        print(f"  {enc.name:11s} mean {np.mean(list(per_lang.values())):.4f}  "
              f"{per_lang}")


def _retrieval(enc, tok, docs, queries, gold, q_prefix, d_prefix, log_every=200):
    D = encode_texts(enc, tok, [t for _, t in docs], d_prefix, log_every=log_every)
    Q = encode_texts(enc, tok, [t for _, t in queries], q_prefix)
    sims = Q @ D.T
    doc_ids = [i for i, _ in docs]
    ndcgs, r5, r1 = [], [], []
    for qi, (qid, _) in enumerate(queries):
        g = gold.get(qid, set())
        if not g:
            continue
        order = np.argsort(-sims[qi])
        rel = [1.0 if doc_ids[j] in g else 0.0 for j in order[:10]]
        ndcgs.append(ndcg_at_k(rel, [1.0] * len(g), 10))
        r5.append(float(any(rel[:5])))
        r1.append(float(rel[0]))
    return {"ndcg@10": round(float(np.mean(ndcgs)), 4),
            "recall@5": round(float(np.mean(r5)), 4),
            "hit@1": round(float(np.mean(r1)), 4), "n_queries": len(ndcgs)}


def gate_c(encoders, tok, report, n_docs, prefix_ab):
    print(f"\n=== GATE C: NanoSciFact retrieval (corpus {n_docs}, "
          f"{Q_PREFIX!r}/{D_PREFIX!r}) ===")
    docs, queries, gold = load_nanoscifact(n_docs)
    print(f"  {len(queries)} queries, {len(docs)} docs")
    for enc in encoders:
        row = _retrieval(enc, tok, docs, queries, gold, Q_PREFIX, D_PREFIX)
        report.setdefault("C_retrieval", {})[enc.name] = row
        print(f"  {enc.name:11s} nDCG@10 {row['ndcg@10']:.4f} "
              f"recall@5 {row['recall@5']:.4f} hit@1 {row['hit@1']:.4f}")

    if prefix_ab:
        # The repo documents the prefixes and warns that omitting them degrades
        # retrieval. Confirm that on a set with enough resolution to see it,
        # rather than on the card's 6-pair example (which cannot resolve it).
        print("  -- prefix control (documented to matter) --")
        enc = encoders[-1]
        for tag, qp, dp in (("none", "", ""),
                            ("no-space", "query:", "document:"),
                            ("documented", Q_PREFIX, D_PREFIX)):
            r = _retrieval(enc, tok, docs, queries, gold, qp, dp, log_every=0)
            report.setdefault("C_prefix_ab", {})[tag] = r
            print(f"     {enc.name} prefix {tag:11s} -> nDCG@10 "
                  f"{r['ndcg@10']:.4f}  recall@5 {r['recall@5']:.4f}")


def gate_d(encoders, tok, report):
    print("\n=== GATE D: graph mechanics on the shipped artifacts ===")
    ids = tok(D_PREFIX + DOCUMENTS[1], add_special_tokens=True)["input_ids"]
    n = len(ids)
    print(f"  probe text = {n} tokens")
    for enc in encoders:
        if not isinstance(enc, TfliteEncoder):
            continue
        vecs = {}
        for S in enc.lens:
            if S < n:
                continue
            a = np.full((1, S), PAD_ID, np.int32)
            m = np.zeros((1, S), np.int32)
            a[0, :n] = ids
            m[0, :n] = 1
            vecs[S] = list(enc.runners[S](input_ids=a,
                                          attention_mask=m).values())[0][0]
        base = vecs[min(vecs)]
        row = {"cross_sig_max_abs": {str(S): float(np.abs(v - base).max())
                                     for S, v in vecs.items()}}
        S = min(vecs)
        a = np.full((1, S), PAD_ID, np.int32)
        m = np.zeros((1, S), np.int32)
        a[0, :n] = ids
        m[0, :n] = 1
        v1 = list(enc.runners[S](input_ids=a, attention_mask=m).values())[0][0]
        a2 = a.copy()
        a2[0, n:] = np.random.default_rng(1).integers(10, 65535, S - n)
        v2 = list(enc.runners[S](input_ids=a2, attention_mask=m).values())[0][0]
        row["pad_invariance_max_abs"] = float(np.abs(v1 - v2).max())
        m3 = m.copy()
        m3[0, n - 5:] = 0
        v3 = list(enc.runners[S](input_ids=a, attention_mask=m3).values())[0][0]
        row["mask_live_max_abs"] = float(np.abs(v1 - v3).max())

        # Short text in the LONGEST signature is the common real case (a 10-token
        # query padded to 512). It is where a mask bug or a fully-masked softmax
        # row shows up, so gate it explicitly against the short-signature answer.
        Smax, short_ids = max(enc.lens), ids[:12]
        sn = len(short_ids)
        outs = {}
        for S2 in (min(enc.lens), Smax):
            if S2 < sn:
                continue
            a4 = np.full((1, S2), PAD_ID, np.int32)
            m4 = np.zeros((1, S2), np.int32)
            a4[0, :sn] = short_ids
            m4[0, :sn] = 1
            outs[S2] = list(enc.runners[S2](input_ids=a4,
                                            attention_mask=m4).values())[0][0]
        row["short_in_long_finite"] = bool(np.isfinite(outs[Smax]).all())
        row["short_in_long_max_abs"] = float(
            np.abs(outs[Smax] - outs[min(outs)]).max())
        report.setdefault("D_mechanics", {})[enc.name] = row
        print(f"  {enc.name:11s} cross-sig {row['cross_sig_max_abs']}")
        print(f"  {'':11s} pad-invariance {row['pad_invariance_max_abs']:.3e} "
              f"(must be 0) | mask-live {row['mask_live_max_abs']:.3e} (must be >0)")
        print(f"  {'':11s} short({sn} tok) in embed_{Smax}: finite="
              f"{row['short_in_long_finite']} vs embed_{min(outs)} "
              f"max|d| {row['short_in_long_max_abs']:.3e}")
        assert row["mask_live_max_abs"] > 0, f"{enc.name} ignores attention_mask"
        assert row["short_in_long_finite"], f"{enc.name} non-finite at embed_{Smax}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--gates", default="A,B,C,D")
    ap.add_argument("--sts-limit", type=int, default=100)
    ap.add_argument("--docs", type=int, default=600)
    ap.add_argument("--no-torch", action="store_true")
    ap.add_argument("--variants", default="fp32,wi8fc,fp16")
    ap.add_argument("--prefix-ab", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)

    encoders = []
    if not args.no_torch:
        print("loading torch reference ...")
        encoders.append(TorchEncoder(tok))
    for v in args.variants.split(","):
        p = os.path.join(args.out_dir, f"embed_{v}.tflite")
        if os.path.exists(p):
            print(f"loading {p} ({os.path.getsize(p)/1e6:.0f} MB) ...")
            encoders.append(TfliteEncoder(p, v))
        else:
            print(f"(missing {p})")

    # MERGE into any existing report. A partial re-run (`--gates B,C`) used to
    # overwrite the file, silently deleting the gates it did not run — so the
    # card would quote A/D numbers with no primary source left on disk. Same
    # family of trap as the stale quantized artifact in the converter.
    out = os.path.join(args.out_dir, "verify_report.json")
    report = {}
    if os.path.exists(out):
        with open(out) as f:
            report = json.load(f)
        print(f"merging into existing {out} (gates {sorted(k for k in report if k[1:2] == '_')})")
    report["variants"] = [e.name for e in encoders]
    report["model"] = MODEL_DIR
    gates = set(args.gates.upper().split(","))
    if "A" in gates:
        gate_a(encoders, tok, report)
    if "D" in gates:
        gate_d(encoders, tok, report)
    if "B" in gates:
        gate_b(encoders, tok, report, args.sts_limit)
    if "C" in gates:
        gate_c(encoders, tok, report, args.docs, args.prefix_ab)

    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
