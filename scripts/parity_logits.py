"""Output-level parity: per-token logit agreement between the ORIGINAL PyTorch
model and the converted LiteRT (.tflite from a .litertlm) model.

This is the strict notion of conversion parity (not benchmark-score parity): feed
the SAME token ids to both and compare the next-token distributions position by
position (teacher forcing — no sampling cascade).

  python scripts/parity_logits.py --tflite <dumped.tflite> --hf src_models/qwen3-1.7b [--n 48]

Reports top-1 / top-5 agreement, mean KL(pt||lt), and logit Pearson r.
"""
import argparse, json, numpy as np

def build_ids(hf, n):
    import transformers
    tok = transformers.AutoTokenizer.from_pretrained(hf)
    q = json.loads(open("evaldata/gsm8k_test.jsonl").readline())["question"]
    # same structured template the .litertlm applies at runtime
    text = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
    ids = tok(text, add_special_tokens=False)["input_ids"][:n]
    return tok, ids

def pt_logits(hf, ids):
    import torch, transformers
    m = transformers.AutoModelForCausalLM.from_pretrained(hf, dtype=torch.float32).eval()
    with torch.no_grad():
        out = m(torch.tensor([ids])).logits[0].float().numpy()  # [seq, vocab]
    return out

def lt_logits(tflite, ids, kv=4096):
    import ai_edge_litert.interpreter as I
    itp = I.Interpreter(model_path=tflite); itp.allocate_tensors()
    run = itp.get_signature_runner("decode")
    det = run.get_input_details()
    caches = {n: np.zeros([int(x) for x in d["shape"]], dtype=np.float32)
              for n, d in det.items() if n.startswith("kv_cache")}
    out = []
    for t, tk in enumerate(ids):
        mask = np.full((1, 1, 1, kv), np.float32(-1e30), dtype=np.float32)
        mask[0, 0, 0, : t + 1] = 0.0
        o = run(tokens=np.array([[tk]], np.int32),
                input_pos=np.array([t], np.int32), mask=mask, **caches)
        out.append(o["logits"][0, 0].copy())
        for n in caches:
            caches[n] = o[n]
    return np.array(out)  # [seq, vocab]

def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x); return e / e.sum(-1, keepdims=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", required=True)
    ap.add_argument("--hf", required=True)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    tok, ids = build_ids(a.hf, a.n)
    pt = pt_logits(a.hf, ids)
    lt = lt_logits(a.tflite, ids)
    m = min(len(pt), len(lt))
    pt, lt = pt[:m], lt[:m]
    pt1, lt1 = pt.argmax(-1), lt.argmax(-1)
    top1 = float((pt1 == lt1).mean())
    pt5 = np.argsort(-pt, -1)[:, :5]
    top5 = float(np.mean([lt1[i] in pt5[i] for i in range(m)]))
    P, Q = softmax(pt), softmax(lt)
    kl = float(np.mean(np.sum(P * (np.log(P + 1e-9) - np.log(Q + 1e-9)), -1)))
    r = float(np.mean([np.corrcoef(pt[i], lt[i])[0, 1] for i in range(m)]))
    print(f"== logit parity {a.tag} (n={m} positions) ==")
    print(f"  top-1 next-token agreement : {100*top1:.1f}%")
    print(f"  top-5 agreement            : {100*top5:.1f}%")
    print(f"  mean KL(pt||lt)            : {kl:.4f} nats")
    print(f"  mean per-pos logit Pearson : {r:.4f}")

if __name__ == "__main__":
    main()
