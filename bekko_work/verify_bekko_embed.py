#!/usr/bin/env python3
"""Quality gates for the converted bekko-embedding-v1-a25m LiteRT artifacts.

    python verify_bekko_embed.py <out_dir> [--gates A,B,C,D]

Gates
  A  card oracle   — reproduce the model card's documented quickstart scores
                     (external truth, not our own harness). NO prefixes: bekko
                     is trained without query:/passage: prefixes (README FAQ).
  B  STS          — JSTS (JGLUE v1.3 valid, the headline Japanese-RAG market)
                     Spearman, plus STS17 multilingual subset, plus Matryoshka
                     truncation (384/256/128/64) on JSTS. Graded metrics:
                     cannot saturate to a flattering floor.
  C  JSQuAD       — Japanese retrieval nDCG@10 / recall@5 / hit@1: question ->
                     Wikipedia paragraph, i.e. the RAG task the demand is for.
  D  mechanics     — cross-signature agreement, pad-content invariance, mask
                     liveness and finiteness on the shipped artifacts. The
                     cross-signature probe deliberately puts a SHORT text into
                     the 512 signature: pad queries past n_valid+64 have empty
                     sliding windows, the exact shape that NaNs if the
                     diagonal guard is missing.

Every gate runs on each variant (torch fp32 reference, fp32/wi8fc/embt8/fp16
tflite) so the numbers are comparable; a variant is judged against the
reference, never against an absolute threshold.

JGLUE fixtures are cached in bekko_work/fixtures/ (downloaded from
github.com/yahoojapan/JGLUE, datasets/{jsts,jsquad}-v1.3, CC BY-SA 4.0 —
evaluation only, not shipped).
"""
import argparse
import gzip
import json
import os
import time
import urllib.request

import numpy as np
import torch

MODEL_DIR = os.environ.get("BEKKO_EMBED_MODEL", "hotchpotch/bekko-embedding-v1-a25m")
PAD_ID = 0
DIM = 384
SEQ_LENS = (64, 128, 256, 512)
FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
JGLUE_RAW = "https://raw.githubusercontent.com/yahoojapan/JGLUE/main/datasets"

# README quickstart, read 2026-08-31 ("exact scores vary slightly by backend").
CARD_QUERY = "What are the characteristics of sushi?"
CARD_DOCS = [
    "A warm noodle soup served in broth with sliced toppings.",
    "天ぷらは魚や野菜に衣をつけて揚げた料理です。",
    "Une fine crepe garnie de sucre, de beurre ou de fruits.",
    "A Japanese dish made with vinegared rice, often shaped with seafood, vegetables, or egg.",
]
CARD_SCORES = np.array([0.2953, 0.2785, 0.3209, 0.4378])
CARD_CORPUS = [
    "Sushi is a Japanese dish of vinegared rice topped with seafood.",
    "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
    "Mount Fuji is the highest mountain in Japan, at 3,776 meters.",
    "Python is a programming language known for its readability.",
    "La Sagrada Família es una basílica de Barcelona diseñada por Antoni Gaudí.",
]
CARD_XLING = [
    ("日本で一番高い山は？", 2, 0.457),
    ("Who designed the famous basilica in Barcelona?", 4, 0.563),
]
STS17_PAIRS = ["en-en", "es-en", "ko-ko", "en-ar"]
MATRYOSHKA_DIMS = (384, 256, 128, 64)


# --------------------------------------------------------------------------
class TorchEncoder:
    name = "torch_fp32"

    def __init__(self, tok):
        from transformers import AutoModel
        from convert_bekko_embed import Embedder

        m = AutoModel.from_pretrained(
            MODEL_DIR, dtype=torch.float32, attn_implementation="sdpa"
        ).eval()
        self.mod = Embedder(m).eval()
        self.tok = tok

    def encode_ids(self, ids):
        S = len(ids)
        t = torch.tensor([ids], dtype=torch.int32)
        m = torch.ones(1, S, dtype=torch.int32)
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

    def pick(self, n):
        for S in self.lens:
            if n <= S:
                return S
        return self.lens[-1]

    def encode_ids(self, ids):
        """Right-pad to the smallest signature that fits; truncate past the max."""
        S = self.pick(len(ids))
        ids = ids[:S]
        n = len(ids)
        a = np.full((1, S), PAD_ID, dtype=np.int32)
        m = np.zeros((1, S), dtype=np.int32)
        a[0, :n] = ids
        m[0, :n] = 1
        r = self.runners[S](input_ids=a, attention_mask=m)
        return list(r.values())[0][0]


def encode_texts(enc, tok, texts, max_tokens=512, log_every=0):
    out = np.empty((len(texts), DIM), dtype=np.float32)
    t0 = time.time()
    for i, t in enumerate(texts):
        ids = tok(t, add_special_tokens=True)["input_ids"][:max_tokens]
        out[i] = enc.encode_ids(ids)
        if log_every and (i + 1) % log_every == 0:
            el = time.time() - t0
            print(f"    [{enc.name}] {i+1}/{len(texts)}  {el:.0f}s "
                  f"({el/(i+1)*1000:.0f} ms/text)", flush=True)
    return out


# --------------------------------------------------------------------------
def spearman(a, b):
    def rank(x):
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), dtype=np.float64)
        r[order] = np.arange(len(x), dtype=np.float64)
        x_sorted = x[order]
        i = 0
        while i < len(x):
            j = i
            while j + 1 < len(x) and x_sorted[j + 1] == x_sorted[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = np.mean(r[order[i:j + 1]])
            i = j + 1
        return r

    ra, rb = rank(np.asarray(a, float)), rank(np.asarray(b, float))
    ra -= ra.mean(); rb -= rb.mean()
    return float(ra @ rb / np.sqrt((ra @ ra) * (rb @ rb)))


def ndcg_at_k(ranked_rel, ideal_rel, k=10):
    def dcg(rels):
        return sum(r / np.log2(i + 2) for i, r in enumerate(rels[:k]))
    idcg = dcg(sorted(ideal_rel, reverse=True))
    return dcg(ranked_rel) / idcg if idcg > 0 else 0.0


def truncate_renorm(V, d):
    W = V[:, :d].astype(np.float64)
    n = np.linalg.norm(W, axis=1, keepdims=True)
    return (W / np.where(n == 0, 1, n)).astype(np.float32)


def fetch(name, rel):
    os.makedirs(FIXTURES, exist_ok=True)
    p = os.path.join(FIXTURES, name)
    if not os.path.exists(p):
        url = f"{JGLUE_RAW}/{rel}"
        print(f"  downloading {url}")
        urllib.request.urlretrieve(url, p)
    return p


# --------------------------------------------------------------------------
def gate_a(encoders, tok, report):
    print("\n=== GATE A: model-card oracle (external truth, no prefixes) ===")
    for enc in encoders:
        q = encode_texts(enc, tok, [CARD_QUERY])
        d = encode_texts(enc, tok, CARD_DOCS)
        s = (q @ d.T)[0]
        vs_card = float(np.abs(s - CARD_SCORES).max())
        argmax_ok = bool(s.argmax() == 3)

        C = encode_texts(enc, tok, CARD_CORPUS)
        xl = []
        xl_ok = True
        for text, gold_idx, gold_score in CARD_XLING:
            v = encode_texts(enc, tok, [text])[0]
            sims = C @ v
            xl.append(round(float(sims[gold_idx]), 4))
            xl_ok &= bool(sims.argmax() == gold_idx)
        row = {"scores": [round(float(x), 4) for x in s],
               "max_abs_vs_card": round(vs_card, 4), "argmax_ok": argmax_ok,
               "xling_gold_sims": xl, "xling_argmax_ok": xl_ok}
        report.setdefault("A_card", {})[enc.name] = row
        print(f"  {enc.name:11s} scores {row['scores']} max|d| vs card {vs_card:.4f} "
              f"argmax {argmax_ok} | xling {xl} argmax {xl_ok}")
    print(f"  card scores: {[float(x) for x in CARD_SCORES]}  card xling: "
          f"{[g for _, _, g in CARD_XLING]}")


def load_jsts(limit):
    rows = []
    with open(fetch("jsts_valid_v1.3.json", "jsts-v1.3/valid-v1.3.json")) as f:
        for line in f:
            r = json.loads(line)
            rows.append((r["sentence1"], r["sentence2"], float(r["label"])))
    return rows[:limit] if limit else rows


def load_sts17(limit):
    from huggingface_hub import hf_hub_download

    data = {}
    for pair in STS17_PAIRS:
        try:
            p = hf_hub_download("mteb/sts17-crosslingual-sts",
                                f"test/{pair}.jsonl.gz", repo_type="dataset")
        except Exception as e:
            print(f"  (skip {pair}: {type(e).__name__})")
            continue
        rows = []
        with gzip.open(p, "rt") as f:
            for line in f:
                r = json.loads(line)
                rows.append((r["sentence1"], r["sentence2"], float(r["score"])))
        data[pair] = rows[:limit] if limit else rows
    return data


def gate_b(encoders, tok, report, limit):
    print(f"\n=== GATE B: JSTS Spearman + Matryoshka + STS17 subset "
          f"(<= {limit} pairs/set) ===")
    jsts = load_jsts(limit)
    sts17 = load_sts17(min(limit, 100))
    print(f"  JSTS {len(jsts)} pairs; STS17 {sorted(sts17)}")
    for enc in encoders:
        s1 = encode_texts(enc, tok, [r[0] for r in jsts], log_every=200)
        s2 = encode_texts(enc, tok, [r[1] for r in jsts])
        gold = np.array([r[2] for r in jsts])
        by_dim = {}
        for d in MATRYOSHKA_DIMS:
            pred = (truncate_renorm(s1, d) * truncate_renorm(s2, d)).sum(1)
            by_dim[str(d)] = round(spearman(pred, gold), 4)
        per_lang = {}
        for pair, rows in sts17.items():
            a = encode_texts(enc, tok, [r[0] for r in rows])
            b = encode_texts(enc, tok, [r[1] for r in rows])
            per_lang[pair] = round(
                spearman((a * b).sum(1), np.array([r[2] for r in rows])), 4)
        report.setdefault("B_sts", {})[enc.name] = {
            "jsts_by_dim": by_dim, "sts17": per_lang}
        print(f"  {enc.name:11s} JSTS by dim {by_dim}  STS17 {per_lang}")


def load_jsquad(n_docs, n_queries):
    with open(fetch("jsquad_valid_v1.3.json", "jsquad-v1.3/valid-v1.3.json")) as f:
        data = json.load(f)["data"]
    ctx_id = {}
    docs = []
    queries = []
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
    return ([docs[i] for i in sel],
            [(q, remap[g]) for q, g in queries])


def gate_c(encoders, tok, report, n_docs, n_queries=150):
    print(f"\n=== GATE C: JSQuAD retrieval (corpus {n_docs} docs, "
          f"{n_queries} questions) ===")
    docs, queries = load_jsquad(n_docs, n_queries)
    print(f"  {len(queries)} queries, {len(docs)} unique paragraphs")
    for enc in encoders:
        D = encode_texts(enc, tok, docs, log_every=200)
        Q = encode_texts(enc, tok, [q for q, _ in queries])
        sims = Q @ D.T
        ndcgs, r5, r1 = [], [], []
        for qi, (_, gold) in enumerate(queries):
            order = np.argsort(-sims[qi])
            rel = [1.0 if j == gold else 0.0 for j in order[:10]]
            ndcgs.append(ndcg_at_k(rel, [1.0], 10))
            r5.append(float(any(rel[:5])))
            r1.append(float(rel[0]))
        row = {"ndcg@10": round(float(np.mean(ndcgs)), 4),
               "recall@5": round(float(np.mean(r5)), 4),
               "hit@1": round(float(np.mean(r1)), 4), "n_queries": len(queries)}
        report.setdefault("C_retrieval", {})[enc.name] = row
        print(f"  {enc.name:11s} nDCG@10 {row['ndcg@10']:.4f} "
              f"recall@5 {row['recall@5']:.4f} hit@1 {row['hit@1']:.4f}")


def gate_e(encoders, tok, report, n_docs, n_queries=150):
    """Mixed-variant retrieval: documents encoded by the TORCH reference,
    queries by each tflite variant. This is the actual RAG deployment shape
    (index built on desktop with the upstream model, queried on device) and it
    is the gate that separates per-vector fidelity from task fidelity: a
    variant can be task-lossless same-variant (gate C) and still sit in a
    slightly rotated space that degrades cross-variant lookup."""
    print(f"\n=== GATE E: cross-variant retrieval (docs=torch_fp32, "
          f"queries=variant; corpus {n_docs}) ===")
    torch_enc = next((e for e in encoders if e.name == "torch_fp32"), None)
    if torch_enc is None:
        print("  needs the torch reference — skipped")
        return
    docs, queries = load_jsquad(n_docs, n_queries)
    D = encode_texts(torch_enc, tok, docs, log_every=200)
    for enc in encoders:
        Q = encode_texts(enc, tok, [q for q, _ in queries])
        sims = Q @ D.T
        ndcgs, r5, r1 = [], [], []
        for qi, (_, gold) in enumerate(queries):
            order = np.argsort(-sims[qi])
            rel = [1.0 if j == gold else 0.0 for j in order[:10]]
            ndcgs.append(ndcg_at_k(rel, [1.0], 10))
            r5.append(float(any(rel[:5])))
            r1.append(float(rel[0]))
        row = {"ndcg@10": round(float(np.mean(ndcgs)), 4),
               "recall@5": round(float(np.mean(r5)), 4),
               "hit@1": round(float(np.mean(r1)), 4)}
        report.setdefault("E_cross_variant", {})[enc.name] = row
        print(f"  {enc.name:11s} nDCG@10 {row['ndcg@10']:.4f} "
              f"recall@5 {row['recall@5']:.4f} hit@1 {row['hit@1']:.4f}")


def gate_d(encoders, tok, report):
    print("\n=== GATE D: graph mechanics on the shipped artifacts ===")
    # SHORT text on purpose: in the 512 signature, pad queries past n+64 have
    # empty sliding windows — the NaN shape if the diagonal guard is missing.
    text = "富士山は日本で一番高い山です。"
    ids = tok(text, add_special_tokens=True)["input_ids"]
    n = len(ids)
    print(f"  probe text = {n} tokens (short by design)")
    for enc in encoders:
        row = {}
        if isinstance(enc, TfliteEncoder):
            vecs = {}
            for S in enc.lens:
                if S < n:
                    continue
                a = np.full((1, S), PAD_ID, dtype=np.int32)
                m = np.zeros((1, S), dtype=np.int32)
                a[0, :n] = ids; m[0, :n] = 1
                v = list(enc.runners[S](input_ids=a, attention_mask=m).values())[0][0]
                assert np.isfinite(v).all(), f"{enc.name} embed_{S}: NaN/inf output"
                vecs[S] = v
            base = vecs[min(vecs)]
            row["cross_sig_max_abs"] = {
                str(S): float(np.abs(v - base).max()) for S, v in vecs.items()}
            row["cross_sig_cos"] = {
                str(S): round(float(v @ base), 8) for S, v in vecs.items()}

            S = max(vecs)  # scribble the LONG signature's pad region
            a = np.full((1, S), PAD_ID, dtype=np.int32)
            m = np.zeros((1, S), dtype=np.int32)
            a[0, :n] = ids; m[0, :n] = 1
            v1 = list(enc.runners[S](input_ids=a, attention_mask=m).values())[0][0]
            a2 = a.copy()
            a2[0, n:] = np.random.default_rng(1).integers(10, 255000, S - n)
            v2 = list(enc.runners[S](input_ids=a2, attention_mask=m).values())[0][0]
            row["pad_invariance_max_abs"] = float(np.abs(v1 - v2).max())

            m3 = m.copy(); m3[0, n - 4:] = 0
            v3 = list(enc.runners[S](input_ids=a, attention_mask=m3).values())[0][0]
            row["mask_live_max_abs"] = float(np.abs(v1 - v3).max())

            print(f"  {enc.name:11s} cross-sig max|d| {row['cross_sig_max_abs']}")
            print(f"  {'':11s} pad-invariance {row['pad_invariance_max_abs']:.3e} "
                  f"(must be 0) | mask-live {row['mask_live_max_abs']:.3e} (must be >0)")
        report.setdefault("D_mechanics", {})[enc.name] = row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--gates", default="A,D,B,C")
    ap.add_argument("--sts-limit", type=int, default=300)
    ap.add_argument("--docs", type=int, default=800)
    ap.add_argument("--no-torch", action="store_true")
    ap.add_argument("--variants", default="fp32,wi8fc,embt8,fp16")
    ap.add_argument("--report", default="verify_report.json")
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

    report = {"variants": [e.name for e in encoders]}
    gates = set(args.gates.upper().split(","))
    if "A" in gates:
        gate_a(encoders, tok, report)
    if "D" in gates:
        gate_d(encoders, tok, report)
    if "B" in gates:
        gate_b(encoders, tok, report, args.sts_limit)
    if "C" in gates:
        gate_c(encoders, tok, report, args.docs)
    if "E" in gates:
        gate_e(encoders, tok, report, args.docs)

    out = os.path.join(args.out_dir, args.report)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
