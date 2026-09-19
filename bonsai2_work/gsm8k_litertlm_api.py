#!/usr/bin/env python3
"""GSM8K on a .litertlm through the litert_lm Python API with ONE engine load (a 27B GPU init is minutes, so the
per-question `litert-mac-verify` loader of scripts/parity_gsm8k.py is impractical here). Same prompt / extraction /
scoring as scripts/parity_gsm8k.py; greedy (top_k=1); a fresh conversation per question; rows appended to <out>.jsonl.
Usage: gsm8k_litertlm_api.py <bundle> <out.jsonl> [--backend gpu|cpu] [--n 100] [--max-tokens 2048]"""
import argparse, importlib.util, json, os, sys, time
ap = argparse.ArgumentParser(); ap.add_argument("bundle"); ap.add_argument("out"); ap.add_argument("--backend", default="gpu")
ap.add_argument("--n", type=int, default=100); ap.add_argument("--max-tokens", type=int, default=2048); a = ap.parse_args()
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(root)
spec = importlib.util.spec_from_file_location("pg", os.path.join(root, "scripts", "parity_gsm8k.py")); pg = importlib.util.module_from_spec(spec); spec.loader.exec_module(pg)
from litert_lm import engine as engine_lib, interfaces
t0 = time.time()
eng = engine_lib.Engine(a.bundle, backend=interfaces.Backend.GPU if a.backend == "gpu" else interfaces.Backend.CPU, max_num_tokens=a.max_tokens + 512)
print(f"engine ready in {time.time()-t0:.0f}s", flush=True)
done = {json.loads(l)["i"] for l in open(a.out)} if os.path.exists(a.out) else set()
for i, (q, gold) in enumerate(pg.load_q(a.n)):
    if i in done: continue
    conv = eng.create_conversation(sampler_config=interfaces.SamplerConfig(top_k=1))
    t = time.time()
    try:
        resp = conv.send_message(q + pg.COT, max_output_tokens=a.max_tokens)
        txt = "".join(c.get("text", "") for c in resp.get("content", []) if isinstance(c, dict))
    except Exception as e:
        txt = ""; print(f"  q {i+1} ERR {type(e).__name__}: {str(e)[:80]}", flush=True)
    conv.close()
    pred = pg.norm(pg.extract(txt)); ok = pred == pg.norm(gold)
    open(a.out, "a").write(json.dumps({"i": i, "ok": bool(ok), "pred": pred, "gold": pg.norm(gold), "chars": len(txt), "sec": round(time.time()-t, 1), "text": txt}) + "\n")
    print(f"  q {i+1}/{a.n} {'OK' if ok else '..'} pred={pred} gold={pg.norm(gold)} {round(time.time()-t,1)}s", flush=True)
rows = [json.loads(l) for l in open(a.out)]
print(f"GSM8K litertlm {a.backend} greedy n={len(rows)}: {sum(r['ok'] for r in rows)}/{len(rows)}  total {time.time()-t0:.0f}s")
