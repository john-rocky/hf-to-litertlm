"""Logit parity WITH controls: compare every backend's next-token logits to the
fp32 PyTorch reference on identical token ids (teacher forced).

Backends: torch-fp32 (reference), torch-bf16 (precision floor), MLX-4bit (the
known-good 4-bit control), LiteRT-int4 (our conversion). If LiteRT-int4 tracks
the MLX-4bit control, the on-device conversion is at parity at the logit level.

  python scripts/parity_logits_controls.py --hf src_models/qwen3-0.6b \
      --tflite <dump.tflite> --mlx out/mlx/qwen3-0.6b-4bit --n 48
"""
import argparse, json, numpy as np

def build_ids(hf, n):
    import transformers
    tok = transformers.AutoTokenizer.from_pretrained(hf)
    q = json.loads(open("evaldata/gsm8k_test.jsonl").readline())["question"]
    text = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
    return tok(text, add_special_tokens=False)["input_ids"][:n]

def torch_logits(hf, ids, dtype):
    import torch, transformers
    dt = {"fp32": torch.float32, "bf16": torch.bfloat16}[dtype]
    m = transformers.AutoModelForCausalLM.from_pretrained(hf, dtype=dt).eval()
    with torch.no_grad():
        return m(torch.tensor([ids])).logits[0].float().numpy()

def mlx_logits(path, ids):
    from mlx_lm import load
    import mlx.core as mx
    model, _ = load(path)
    out = model(mx.array([ids])).astype(mx.float32)
    return np.array(out)[0]

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
    return np.array(out)

def softmax(x):
    x = x - x.max(-1, keepdims=True); e = np.exp(x); return e / e.sum(-1, keepdims=True)

def cmp(ref, x):
    m = min(len(ref), len(x)); ref, x = ref[:m], x[:m]
    r1 = ref.argmax(-1); x1 = x.argmax(-1)
    top1 = float((r1 == x1).mean())
    r5 = np.argsort(-ref, -1)[:, :5]
    top5 = float(np.mean([x1[i] in r5[i] for i in range(m)]))
    P, Q = softmax(ref), softmax(x)
    kl = float(np.mean(np.sum(P * (np.log(P + 1e-9) - np.log(Q + 1e-9)), -1)))
    return top1, top5, kl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--tflite", required=True)
    ap.add_argument("--mlx", default=None)
    ap.add_argument("--n", type=int, default=48)
    a = ap.parse_args()
    ids = build_ids(a.hf, a.n)
    ref = torch_logits(a.hf, ids, "fp32")
    rows = []
    rows.append(("torch-bf16   (precision floor)", cmp(ref, torch_logits(a.hf, ids, "bf16"))))
    if a.mlx:
        rows.append(("MLX 4bit     (4-bit control)", cmp(ref, mlx_logits(a.mlx, ids))))
    rows.append(("LiteRT int4  (our conversion)", cmp(ref, lt_logits(a.tflite, ids))))
    print(f"\n== logit agreement vs torch-fp32 reference (n={len(ids)} positions) ==")
    print(f"  {'backend':32} {'top1':>7} {'top5':>7} {'KL':>8}")
    for name, (t1, t5, kl) in rows:
        print(f"  {name:32} {100*t1:6.1f}% {100*t5:6.1f}% {kl:8.4f}")

if __name__ == "__main__":
    main()
