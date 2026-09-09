#!/usr/bin/env python3
"""Quality gates for the converted Giga-Embeddings-instruct-480M-0826 LiteRT artifacts.

    python verify_giga_embed.py <out_dir> [--gates A,B,C,D] \
        [--texts giga_work/gate_texts.json] [--st-oracle <st_oracle.json>]

Gates
  A  parity      — per-text cosine of every artifact vs the PyTorch fp32 reference on the
                   check set (the vendor's own 2-D-mask forward + manual mean pool + L2, i.e.
                   the README's "Transformers" example, batch 1); optionally also vs a
                   sentence-transformers oracle JSON produced by giga_work/st_oracle.py.
                   Prints the README example similarity (Russian query vs Moscow/Paris).
  B  STS         — STS17 en-en + STS22 ru Spearman (the model's two languages), no prompt
                   (symmetric task; the README says a prompt is optional there).
  C  NanoSciFact — retrieval nDCG@10 / recall@5 with the config's query prompt, documents raw.
  D  mechanics   — cross-signature agreement, pad-content invariance and mask liveness on the
                   shipped artifacts.

Every gate runs on each variant (torch fp32 reference, fp32/wi8fc/fp16 tflite) so the numbers
are comparable; a variant is judged against the reference, never against an absolute threshold.
Partial --gates runs MERGE into verify_report.json (never overwrite: card numbers need their
primary source).
"""
import argparse
import gzip
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

MODEL_DIR = os.environ.get("GIGA_EMBED_MODEL", "ai-sage/Giga-Embeddings-instruct-480M-0826")
PAD_ID = 2
DIM = 1024
SEQ_LENS = (64, 128, 256, 512)
DEFAULT_TEXTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gate_texts.json")


def load_prompts():
    """Read the prompt strings out of the vendor config — never retype them from prose."""
    from huggingface_hub import hf_hub_download

    p = (os.path.join(MODEL_DIR, "config_sentence_transformers.json") if os.path.isdir(MODEL_DIR)
         else hf_hub_download(MODEL_DIR, "config_sentence_transformers.json"))
    pr = json.load(open(p))["prompts"]
    return pr["query"], pr.get("document", "")


# --------------------------------------------------------------------------
# encoders: one interface, four backends
# --------------------------------------------------------------------------
class TorchEncoder:
    """The vendor's own path: remote-code forward with a 2-D mask, manual mean pool + L2."""
    name = "torch_fp32"

    def __init__(self):
        from transformers import AutoModel

        m = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True, dtype=torch.float32,
                                      attn_implementation="sdpa").eval()
        assert m.config.is_causal is False
        self.model = m

    def encode_ids(self, ids):
        t = torch.tensor([ids], dtype=torch.int64)
        am = torch.ones_like(t)
        with torch.inference_mode():
            h = self.model(input_ids=t, attention_mask=am).last_hidden_state
            mask = am.unsqueeze(-1).to(h.dtype)
            e = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
            return F.normalize(e, dim=-1).numpy()[0]


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
        S = self.pick(len(ids))
        ids = ids[:S]
        n = len(ids)
        a = np.full((1, S), PAD_ID, dtype=np.int32)
        m = np.zeros((1, S), dtype=np.int32)
        a[0, :n] = ids
        m[0, :n] = 1
        r = self.runners[S](input_ids=a, attention_mask=m)
        return list(r.values())[0][0]


def encode_texts(enc, tok, texts, prefix="", max_tokens=512, log_every=0):
    out = np.empty((len(texts), DIM), dtype=np.float32)
    t0 = time.time()
    for i, t in enumerate(texts):
        ids = tok(prefix + t, add_special_tokens=True)["input_ids"][:max_tokens]
        out[i] = enc.encode_ids(ids)
        if log_every and (i + 1) % log_every == 0:
            el = time.time() - t0
            print(f"    [{enc.name}] {i+1}/{len(texts)}  {el:.0f}s ({el/(i+1)*1000:.0f} ms/text)", flush=True)
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


def cos_rows(a, b):
    an = a / np.linalg.norm(a, axis=1, keepdims=True)
    bn = b / np.linalg.norm(b, axis=1, keepdims=True)
    return (an * bn).sum(1)


# --------------------------------------------------------------------------
def gate_a(encoders, tok, report, texts, st_oracle):
    print(f"\n=== GATE A: parity on the check set ({len(texts)} texts) ===")
    n_tok = [len(tok(t, add_special_tokens=True)["input_ids"]) for t in texts]
    print(f"  tokens per text: {n_tok}")
    vecs = {}
    for enc in encoders:
        vecs[enc.name] = encode_texts(enc, tok, texts)
    ref_name = encoders[0].name
    ref = vecs[ref_name]
    rows = {"tokens": n_tok, "reference": ref_name}
    for enc in encoders[1:]:
        c = cos_rows(vecs[enc.name], ref)
        rows[enc.name] = {"cos_min": round(float(c.min()), 6), "cos_mean": round(float(c.mean()), 6),
                          "cos_per_text": [round(float(x), 6) for x in c]}
        print(f"  {enc.name:11s} vs {ref_name}: cos min {c.min():.6f} mean {c.mean():.6f}")
    if st_oracle:
        O = np.array(json.load(open(st_oracle))["embeddings"], dtype=np.float64)
        assert O.shape == ref.shape, (O.shape, ref.shape)
        rows["st_oracle"] = {}
        for name, v in vecs.items():
            c = cos_rows(v.astype(np.float64), O)
            rows["st_oracle"][name] = {"cos_min": round(float(c.min()), 6), "cos_mean": round(float(c.mean()), 6)}
            print(f"  {name:11s} vs sentence-transformers oracle: cos min {c.min():.6f} mean {c.mean():.6f}")
    # README example: text 0 (Russian query) vs texts 3, 4 (Moscow / Paris)
    if len(texts) >= 5:
        rows["readme_example"] = {}
        for name, v in vecs.items():
            s = (v[0] @ v[3:5].T).tolist()
            rows["readme_example"][name] = [round(float(x), 4) for x in s]
            print(f"  {name:11s} README example (query vs Moscow, Paris): {[round(float(x),4) for x in s]}")
    report["A_parity"] = rows
    # keep the vectors so the engine gate can be scored against torch directly
    for name, v in vecs.items():
        report.setdefault("A_vectors", {})[name] = v.astype(float).tolist()


def load_sts(limit):
    from huggingface_hub import hf_hub_download

    data = {}
    for tag, repo, fn in (("sts17 en-en", "mteb/sts17-crosslingual-sts", "test/en-en.jsonl.gz"),
                          ("sts22 ru", "mteb/sts22-crosslingual-sts", "test/ru.jsonl.gz")):
        try:
            p = hf_hub_download(repo, fn, repo_type="dataset")
        except Exception as e:
            print(f"  (skip {tag}: {type(e).__name__})")
            continue
        rows = []
        with gzip.open(p, "rt") as f:
            for line in f:
                r = json.loads(line)
                rows.append((r["sentence1"], r["sentence2"], float(r["score"])))
        data[tag] = rows[:limit] if limit else rows
    return data


def gate_b(encoders, tok, report, limit):
    print(f"\n=== GATE B: STS Spearman (<= {limit} pairs/set, no prompt) ===")
    data = load_sts(limit)
    if not data:
        print("  no data — skipped")
        return
    print(f"  sets: {[(k, len(v)) for k, v in data.items()]}")
    for enc in encoders:
        per = {}
        for tag, rows in data.items():
            s1 = encode_texts(enc, tok, [r[0] for r in rows])
            s2 = encode_texts(enc, tok, [r[1] for r in rows])
            pred = (s1 * s2).sum(1)
            gold = np.array([r[2] for r in rows])
            per[tag] = round(spearman(pred, gold), 4)
        report.setdefault("B_sts", {})[enc.name] = per
        print(f"  {enc.name:11s} {per}")


def load_nanoscifact(n_docs):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    R = "zeta-alpha-ai/NanoSciFact"
    corpus = pq.read_table(hf_hub_download(R, "corpus/train-00000-of-00001.parquet", repo_type="dataset")).to_pydict()
    queries = pq.read_table(hf_hub_download(R, "queries/train-00000-of-00001.parquet", repo_type="dataset")).to_pydict()
    qrels = pq.read_table(hf_hub_download(R, "qrels/train-00000-of-00001.parquet", repo_type="dataset")).to_pydict()
    gold = {}
    for q, c in zip(qrels["query-id"], qrels["corpus-id"]):
        gold.setdefault(str(q), set()).add(str(c))
    ids = [str(i) for i in corpus["_id"]]
    texts = corpus["text"]
    keep_ids = sorted({c for s in gold.values() for c in s})
    rng = np.random.default_rng(0)
    rest = [i for i in ids if i not in set(keep_ids)]
    extra = list(rng.choice(rest, size=max(0, n_docs - len(keep_ids)), replace=False)) if n_docs else rest
    sel = list(keep_ids) + [str(x) for x in extra]
    by_id = dict(zip(ids, texts))
    return ([(i, by_id[i]) for i in sel], list(zip([str(i) for i in queries["_id"]], queries["text"])), gold)


def gate_c(encoders, tok, report, n_docs, q_prompt, d_prompt):
    print(f"\n=== GATE C: NanoSciFact retrieval (corpus subsampled to {n_docs}) ===")
    docs, queries, gold = load_nanoscifact(n_docs)
    print(f"  {len(queries)} queries, {len(docs)} docs ({sum(len(v) for v in gold.values())} qrels); "
          f"query prompt {q_prompt!r}, doc prompt {d_prompt!r}")
    for enc in encoders:
        D = encode_texts(enc, tok, [t for _, t in docs], d_prompt, log_every=200)
        Q = encode_texts(enc, tok, [t for _, t in queries], q_prompt)
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
        row = {"ndcg@10": round(float(np.mean(ndcgs)), 4), "recall@5": round(float(np.mean(r5)), 4),
               "hit@1": round(float(np.mean(r1)), 4), "n_queries": len(ndcgs)}
        report.setdefault("C_retrieval", {})[enc.name] = row
        print(f"  {enc.name:11s} nDCG@10 {row['ndcg@10']:.4f} recall@5 {row['recall@5']:.4f} hit@1 {row['hit@1']:.4f}")


def gate_d(encoders, tok, report, texts):
    print("\n=== GATE D: graph mechanics on the shipped artifacts ===")
    text = texts[5] if len(texts) > 5 else texts[-1]
    ids = tok(text, add_special_tokens=True)["input_ids"]
    n = len(ids)
    print(f"  probe text = {n} tokens")
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
                vecs[S] = list(enc.runners[S](input_ids=a, attention_mask=m).values())[0][0]
            base = vecs[min(vecs)]
            row["cross_sig_max_abs"] = {str(S): float(np.abs(v - base).max()) for S, v in vecs.items()}
            row["cross_sig_cos"] = {str(S): round(float(v @ base), 8) for S, v in vecs.items()}

            S = min(vecs)
            a = np.full((1, S), PAD_ID, dtype=np.int32)
            m = np.zeros((1, S), dtype=np.int32)
            a[0, :n] = ids; m[0, :n] = 1
            v1 = list(enc.runners[S](input_ids=a, attention_mask=m).values())[0][0]
            a2 = a.copy()
            a2[0, n:] = np.random.default_rng(1).integers(10, 128000, S - n)
            v2 = list(enc.runners[S](input_ids=a2, attention_mask=m).values())[0][0]
            row["pad_invariance_max_abs"] = float(np.abs(v1 - v2).max())

            m3 = m.copy(); m3[0, n - 5:] = 0
            v3 = list(enc.runners[S](input_ids=a, attention_mask=m3).values())[0][0]
            row["mask_live_max_abs"] = float(np.abs(v1 - v3).max())
            row["finite"] = bool(np.isfinite(v1).all())

            print(f"  {enc.name:11s} cross-sig max|d| {row['cross_sig_max_abs']}")
            print(f"  {'':11s} pad-invariance {row['pad_invariance_max_abs']:.3e} (must be 0) | "
                  f"mask-live {row['mask_live_max_abs']:.3e} (must be >0) | finite {row['finite']}")
        report.setdefault("D_mechanics", {})[enc.name] = row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--gates", default="A,B,C,D")
    ap.add_argument("--texts", default=DEFAULT_TEXTS)
    ap.add_argument("--st-oracle", default=None)
    ap.add_argument("--sts-limit", type=int, default=150)
    ap.add_argument("--docs", type=int, default=600)
    ap.add_argument("--no-torch", action="store_true")
    ap.add_argument("--variants", default="fp32,wi8fc,fp16")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    probe = tok("hi", add_special_tokens=True)["input_ids"]
    assert probe[0] == 1 and probe[-1] == 2, f"tokenizer no longer adds <s>/</s>: {probe}"
    q_prompt, d_prompt = load_prompts()
    texts = json.load(open(args.texts))["texts"]

    encoders = []
    if not args.no_torch:
        print("loading torch reference ...")
        encoders.append(TorchEncoder())
    for v in args.variants.split(","):
        p = os.path.join(args.out_dir, f"embed_{v}.tflite")
        if os.path.exists(p):
            print(f"loading {p} ({os.path.getsize(p)/1e6:.0f} MB) ...")
            encoders.append(TfliteEncoder(p, v))
        else:
            print(f"(missing {p})")

    out = os.path.join(args.out_dir, "verify_report.json")
    report = json.load(open(out)) if os.path.exists(out) else {}
    report["variants"] = [e.name for e in encoders]
    report["model"] = MODEL_DIR
    report["prompts"] = {"query": q_prompt, "document": d_prompt}
    gates = set(args.gates.upper().split(","))
    if "A" in gates:
        gate_a(encoders, tok, report, texts, args.st_oracle)
    if "D" in gates:
        gate_d(encoders, tok, report, texts)
    if "B" in gates:
        gate_b(encoders, tok, report, args.sts_limit)
    if "C" in gates:
        gate_c(encoders, tok, report, args.docs, q_prompt, d_prompt)

    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
