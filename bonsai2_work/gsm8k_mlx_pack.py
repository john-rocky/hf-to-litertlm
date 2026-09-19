#!/usr/bin/env python3
"""GSM8K reference on PrismML's own runtime (the MLX pack + bundled loader), same prompt / extraction / scoring as
scripts/parity_gsm8k.py (0-shot CoT, '#### <n>' line, greedy), thinking OFF via the pack's chat template
(enable_thinking=False -> the empty <think> block the simple ChatML template in the .litertlm also emits).
Per-question rows are appended to <out>.jsonl as they finish.
Usage: gsm8k_mlx_pack.py <pack_dir> <out.jsonl> [--n 100] [--max-tokens 2048]"""
import argparse, importlib.util, json, os, sys, time
ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("out"); ap.add_argument("--n", type=int, default=100); ap.add_argument("--max-tokens", type=int, default=2048)
a = ap.parse_args()
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("pg", os.path.join(root, "scripts", "parity_gsm8k.py")); pg = importlib.util.module_from_spec(spec)
os.chdir(root); spec.loader.exec_module(pg)
sys.path.insert(0, os.path.join(a.pack, "runtime"))
import mlx.core as mx
from vision_artifact import load_vl_model
from transformers import AutoTokenizer
model, _, _ = load_vl_model(a.pack, load_processor=False); lm = model.language_model
tok = AutoTokenizer.from_pretrained(a.pack)
EOS = {248046, 248044}
def generate(ids, max_tokens):
    cache = lm.make_cache(); out = []
    o = lm(mx.array([ids]), cache=cache); lg = (o.logits if hasattr(o, "logits") else o)[0, -1]
    nxt = int(mx.argmax(lg).item())
    for _ in range(max_tokens):
        if nxt in EOS: break
        out.append(nxt)
        o = lm(mx.array([[nxt]]), cache=cache); lg = (o.logits if hasattr(o, "logits") else o)[0, -1]
        nxt = int(mx.argmax(lg).item())
    return out
qs = pg.load_q(a.n); c = 0; t0 = time.time()
done = set()
if os.path.exists(a.out):
    for line in open(a.out): done.add(json.loads(line)["i"])
for i, (q, gold) in enumerate(qs):
    if i in done: continue
    ids = tok.apply_chat_template([{"role": "user", "content": q + pg.COT}], add_generation_prompt=True, enable_thinking=False, tokenize=True, return_dict=False)
    ids = list(ids["input_ids"]) if hasattr(ids, "keys") else list(ids)
    t = time.time(); gen = generate(ids, a.max_tokens); txt = tok.decode(gen)
    pred = pg.norm(pg.extract(txt)); ok = pred == pg.norm(gold); c += ok
    row = {"i": i, "ok": bool(ok), "pred": pred, "gold": pg.norm(gold), "n_tokens": len(gen), "sec": round(time.time() - t, 1), "text": txt}
    open(a.out, "a").write(json.dumps(row) + "\n")
    print(f"  mlx {i+1}/{len(qs)} {'OK' if ok else '..'} pred={pred} gold={pg.norm(gold)} tokens={len(gen)} {row['sec']}s", flush=True)
    mx.clear_cache()
rows = [json.loads(l) for l in open(a.out)]
print(f"GSM8K mlx-pack greedy thinking-off n={len(rows)}: {sum(r['ok'] for r in rows)}/{len(rows)}  ({time.time()-t0:.0f}s)")
