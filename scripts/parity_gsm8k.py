"""Robust GSM8K parity: a LiteRT-LM quantized .litertlm vs its original bf16.

Same prompt (0-shot CoT that asks for a '#### <n>' final line) + same greedy decode +
same answer extraction for BOTH, so the only variable is quantization. Designed to put
the model at its real GSM8K level (not floored) so recipe deltas are visible, not noise.

  # bf16 baseline (slow, MPS) — run once
  python scripts/parity_gsm8k.py --which bf16 --n 150 --hf src_models/falcon3-3b
  # each quantized recipe (fast, via litert-mac-verify)
  python scripts/parity_gsm8k.py --which int4 --n 150 --litertlm out/<name>/model.litertlm --tag <name>
"""
import argparse, json, os, re, subprocess, sys, time, types

# scipy stub prelude — ONLY when scipy is actually broken. On a healthy scipy (e.g. the
# Accelerate-built scipy on Darwin 27, see darwin27-scipy-libomp-trap) stubbing is HARMFUL:
# the fake scipy.optimize lacks `milp`, which transformers' bf16 path pulls via
# sklearn → scipy.stats → `from scipy.optimize import milp`. So skip the stub when real
# scipy imports cleanly; only stub the broken-prebuilt case.
def _scipy_healthy():
    try:
        import scipy.optimize  # noqa: F401
        from scipy.optimize import milp  # noqa: F401
        import scipy.sparse.linalg._propack  # noqa: F401
        return True
    except Exception:
        return False

if not _scipy_healthy():
    _pp = types.ModuleType("scipy.sparse.linalg._propack")
    _pp.__file__ = "<stub:scipy._propack>"; _pp.__spec__ = None
    for _nm in ("_spropack", "_dpropack", "_cpropack", "_zpropack"):
        setattr(_pp, _nm, type("_S", (), {"__getattr__": lambda s, n: (lambda *a, **k: None), "__call__": lambda s, *a, **k: None})())
    sys.modules["scipy.sparse.linalg._propack"] = _pp
    _opt = types.ModuleType("scipy.optimize")
    _opt.__file__ = "<stub:scipy.optimize>"; _opt.__spec__ = None
    _opt.linear_sum_assignment = lambda *a, **k: None
    sys.modules["scipy.optimize"] = _opt

VERIFY = os.path.expanduser("~/code/litert-mac-verify/.build/release/litert-mac-verify")
DATA = "evaldata/gsm8k_test.jsonl"
COT = ("\n\nSolve this step by step. After your reasoning, write the final answer on its own "
       "line in the exact form:\n#### <number>")

def load_q(n):
    out = []
    for line in open(DATA):
        d = json.loads(line)
        out.append((d["question"], d["answer"].split("####")[-1].strip().replace(",", "")))
        if len(out) >= n:
            break
    return out

def extract(text):
    """GSM8K-standard: prefer '#### N', then 'answer is/: N', then the last number."""
    if not text:
        return None
    t = text.replace(",", "")
    m = re.findall(r"####\s*\$?(-?\d+(?:\.\d+)?)", t)
    if m: return m[-1].rstrip(".0") if "." in m[-1] else m[-1]
    m = re.findall(r"\\boxed\{\s*\$?(-?\d+(?:\.\d+)?)", t)  # OLMo-2 etc. mark the final answer with \boxed{}
    if m: return m[-1].rstrip(".0") if "." in m[-1] else m[-1]
    m = re.findall(r"(?:answer|total|result)\s*(?:is|:|=)\s*\$?(-?\d+(?:\.\d+)?)", t, re.I)
    if m: return m[-1].rstrip(".0") if "." in m[-1] else m[-1]
    m = re.findall(r"-?\d+(?:\.\d+)?", t)
    if not m: return None
    v = m[-1]
    return v.rstrip(".0") if "." in v else v

def norm(x):
    if x is None: return None
    x = x.replace(",", "").lstrip("$")
    try:
        f = float(x); return str(int(f)) if f == int(f) else str(f)
    except: return x

def run_bf16(qs, path, max_tokens, bf16_template="chat"):
    import torch, transformers
    _DEV = os.environ.get("PARITY_DEVICE", "mps")  # set PARITY_DEVICE=cpu for 7B (MPS hits a hard Metal abort on this machine)
    tok = transformers.AutoTokenizer.from_pretrained(path)
    model = transformers.AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(_DEV).eval()
    eot = tok.eos_token_id  # model-agnostic (Qwen3=<|im_end|>, Falcon=<|endoftext|>)
    pad = tok.pad_token_id if tok.pad_token_id is not None else eot
    c = 0
    for i, (q, gold) in enumerate(qs):
        if bf16_template == "chatml_simple":  # match the .litertlm's embedded simple template
            prompt = f"<|im_start|>user\n{q}{COT}<|im_end|>\n<|im_start|>assistant\n"
        elif bf16_template == "llama_simple":  # match the .litertlm's embedded Llama-3 simple template (BOS auto-added by tokenizer = runtime's start_token)
            prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{q}{COT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        elif bf16_template == "olmo_simple":  # match the .litertlm's embedded OLMo-2 simple template (BOS auto-added by tokenizer = runtime's start_token)
            prompt = f"<|user|>\n{q}{COT}\n<|assistant|>\n"
        elif bf16_template == "granite_simple":  # match the .litertlm's embedded Granite-3.x simple template (BOS auto-added by tokenizer = runtime's start_token)
            prompt = f"<|start_of_role|>user<|end_of_role|>{q}{COT}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"
        elif bf16_template == "deepseek_r1":  # match the .litertlm's embedded DeepSeek-R1-Distill template (BOS auto-added by tokenizer = runtime's start_token)
            prompt = f"<｜User｜>{q}{COT}<｜Assistant｜>"
        elif bf16_template == "phi_simple":  # match the .litertlm's embedded Phi simple template (<|user|>…<|end|><|assistant|>)
            prompt = f"<|user|>{q}{COT}<|end|><|assistant|>"
        else:
            prompt = tok.apply_chat_template([{"role": "user", "content": q + COT}],
                                             tokenize=False, add_generation_prompt=True)
        try:
            ids = tok(prompt, return_tensors="pt").to(_DEV)
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=max_tokens, do_sample=False, eos_token_id=eot, pad_token_id=pad)
            gen = [t for t in out[0][ids.input_ids.shape[1]:].tolist() if 0 <= t < len(tok)]  # guard out-of-range ids → tokenizer C++ OverflowError
            txt = tok.decode(gen, skip_special_tokens=True)
        except Exception as e:  # don't let one bad question kill a 100-q run
            print(f"  bf16 {i+1}/{len(qs)} ERR {type(e).__name__}: {str(e)[:60]}", flush=True)
            txt = ""
        ok = norm(extract(txt)) == norm(gold); c += ok
        print(f"  bf16 {i+1}/{len(qs)} {'OK' if ok else '..'} pred={norm(extract(txt))} gold={norm(gold)}", flush=True)
    return c

def run_mlx(qs, mlx_path, max_tokens):
    from mlx_lm import load, generate
    import mlx.core as mx
    model, tok = load(mlx_path)
    c = 0
    for i, (q, gold) in enumerate(qs):
        prompt = tok.apply_chat_template([{"role": "user", "content": q + COT}],
                                         tokenize=False, add_generation_prompt=True)
        try:
            txt = generate(model, tok, prompt=prompt, max_tokens=max_tokens, verbose=False)
        except Exception as e:
            print(f"  mlx {i+1}/{len(qs)} ERR {type(e).__name__}: {str(e)[:60]}", flush=True); txt = ""
        try:
            mx.clear_cache()  # free Metal buffers between questions so they don't accumulate → OOM
        except Exception:
            pass
        ok = norm(extract(txt)) == norm(gold); c += ok
        print(f"  mlx {i+1}/{len(qs)} {'OK' if ok else '..'} pred={norm(extract(txt))} gold={norm(gold)}", flush=True)
    return c

def run_litertlm(qs, litertlm, max_tokens, greedy=False):
    c = 0
    for i, (q, gold) in enumerate(qs):
        try:
            cmd = [VERIFY, litertlm, q + COT, "--max-tokens", str(max_tokens)]
            if greedy:
                cmd.append("--greedy")
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=400)
            m = re.search(r"^OUTPUT: \[(.*)\]$", p.stdout + "\n" + p.stderr, re.M)
            txt = m.group(1).replace("⏎", "\n") if m else ""
        except Exception:
            txt = ""
        ok = norm(extract(txt)) == norm(gold); c += ok
        print(f"  q {i+1}/{len(qs)} {'OK' if ok else '..'} pred={norm(extract(txt))} gold={norm(gold)}", flush=True)
    return c

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--hf", default="src_models/falcon3-3b")
    ap.add_argument("--litertlm", default="out/falcon3-3b-mixed4/model.litertlm")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--which", choices=["bf16", "int4", "mlx"], required=True)
    ap.add_argument("--mlx-path", default=None)
    ap.add_argument("--bf16-template", choices=["chat", "chatml_simple", "llama_simple", "olmo_simple", "granite_simple", "deepseek_r1", "phi_simple"], default="chat")
    ap.add_argument("--greedy", action="store_true", help="run the .litertlm at temperature 0 (match bf16)")
    ap.add_argument("--tag", default=None, help="label for the result json")
    args = ap.parse_args()
    qs = load_q(args.n)
    t = time.time()
    if args.which == "bf16":
        c = run_bf16(qs, args.hf, args.max_tokens, args.bf16_template); tag = args.tag or "bf16"
    elif args.which == "mlx":
        c = run_mlx(qs, args.mlx_path, args.max_tokens); tag = args.tag or "mlx"
    else:
        c = run_litertlm(qs, args.litertlm, args.max_tokens, args.greedy)
        tag = args.tag or os.path.basename(os.path.dirname(args.litertlm))
    acc = c / len(qs)
    print(f"== {tag}: {c}/{len(qs)} = {100*acc:.1f}%   ({time.time()-t:.0f}s)")
    os.makedirs("reports/parity", exist_ok=True)
    json.dump({"tag": tag, "n": len(qs), "correct": c, "acc": acc, "max_tokens": args.max_tokens},
              open(f"reports/parity/gsm8k_{tag}.json", "w"), indent=2)
    print(f"   wrote reports/parity/gsm8k_{tag}.json")

if __name__ == "__main__":
    main()
