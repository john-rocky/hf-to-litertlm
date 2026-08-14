"""Output-level parity for models whose .tflite exceeds the TFLite Python
Interpreter's flatbuffer size check (~4 GiB; a 4B fp32 export is 16 GB).

Same metric and mask convention as scripts/parity_logits.py, but split into
three stages so the PyTorch side (torch/transformers venv) and the LiteRT side
(ai-edge-litert >= 2.2 venv, CompiledModel loader — the only Python loader
that takes big appended-buffer models) can run in different venvs:

  # 1. torch venv: teacher-forced fp32 logits from the HF reference
  python scripts/parity_logits_bigmodel.py pt --hf Qwen/Qwen3.5-4B --n 48 --out pt.npz
  # 2. litert venv: decode-walk the same ids on the CompiledModel CPU path
  python scripts/parity_logits_bigmodel.py lt --tflite model.tflite --ids pt.npz --out lt.npz
  # 3. anywhere with numpy: the parity_logits.py report
  python scripts/parity_logits_bigmodel.py cmp --pt pt.npz --lt lt.npz
"""
import argparse
import json

import numpy as np

NEG = np.float32(-1e30)


def build_ids(hf, n):
    import transformers
    tok = transformers.AutoTokenizer.from_pretrained(hf)
    q = json.loads(open("evaldata/gsm8k_test.jsonl").readline())["question"]
    text = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
    return tok(text, add_special_tokens=False)["input_ids"][:n]


def cmd_pt(a):
    import torch
    import transformers
    ids = build_ids(a.hf, a.n)
    m = transformers.AutoModelForCausalLM.from_pretrained(
        a.hf, dtype=torch.float32).eval()
    with torch.no_grad():
        pt = m(torch.tensor([ids])).logits[0].float().numpy()
    np.savez_compressed(a.out, ids=np.array(ids, np.int32), pt=pt)
    print(f"pt logits {pt.shape} -> {a.out}")


def cmd_lt(a):
    from ai_edge_litert.compiled_model import CompiledModel
    from ai_edge_litert.hardware_accelerator import HardwareAccelerator
    ids = np.load(a.ids)["ids"]
    model = CompiledModel.from_file(a.tflite,
                                    hardware_accel=HardwareAccelerator.CPU)
    in_det = model.get_input_tensor_details("decode")
    out_det = model.get_output_tensor_details("decode")
    states = {n: np.zeros(in_det[n]["shape"], np.float32)
              for n in in_det if n.startswith("kv_")}
    logits_name = [n for n in out_det if n not in states][0]
    kv = int(in_det["mask"]["shape"][-1])
    out = []
    for t, tk in enumerate(ids):
        mask = np.full((1, 1, 1, kv), NEG, dtype=np.float32)
        mask[0, 0, 0, : t + 1] = 0.0
        inputs = {"tokens": np.array([[tk]], np.int32),
                  "input_pos": np.array([t], np.int32), "mask": mask, **states}
        in_bufs = {}
        for name, arr in inputs.items():
            b = model.create_input_buffer_by_name("decode", name)
            b.write(np.ascontiguousarray(arr))
            in_bufs[name] = b
        out_bufs = {n: model.create_output_buffer_by_name("decode", n)
                    for n in out_det}
        model.run_by_name("decode", in_bufs, out_bufs)
        for n in states:
            shape = out_det[n]["shape"]
            states[n] = np.array(out_bufs[n].read(int(np.prod(shape)),
                                                  np.float32),
                                 dtype=np.float32).reshape(shape)
        shape = out_det[logits_name]["shape"]
        out.append(np.array(out_bufs[logits_name].read(int(np.prod(shape)),
                                                       np.float32),
                            dtype=np.float32).reshape(-1))
        for b in list(in_bufs.values()) + list(out_bufs.values()):
            try:
                b.destroy()
            except Exception:
                pass
        print(f"step {t}: top1={int(np.argmax(out[-1]))}", flush=True)
    np.savez_compressed(a.out, lt=np.array(out))
    print(f"lt logits {np.array(out).shape} -> {a.out}")


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def cmd_cmp(a):
    pt = np.load(a.pt)["pt"]
    lt = np.load(a.lt)["lt"]
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


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pt")
    p.add_argument("--hf", required=True)
    p.add_argument("--n", type=int, default=48)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_pt)
    p = sub.add_parser("lt")
    p.add_argument("--tflite", required=True)
    p.add_argument("--ids", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_lt)
    p = sub.add_parser("cmp")
    p.add_argument("--pt", required=True)
    p.add_argument("--lt", required=True)
    p.add_argument("--tag", default="")
    p.set_defaults(fn=cmd_cmp)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
