#!/usr/bin/env python3
"""GSM8K on a .litertlm with ONE FRESH PROCESS PER QUESTION (`litert-lm run`, greedy by default), because a second
conversation on the same engine corrupts this family's linear-attention state (cross-conversation cache dedup rewinds
the step, not the recurrent state: engine-crossconv-dedup-linear-state) — measured here: same-engine run gave 3/21
with prompt echoes and loops, fresh engines answer the same questions correctly. `--cache disk` keeps the GPU compile
cache next to the bundle so init is a fraction of the first run. Same prompt / extraction / scoring as
scripts/parity_gsm8k.py. Rows appended to <out>.jsonl (resumable).
Usage: gsm8k_litertlm_cli.py <bundle> <out.jsonl> [--backend gpu|cpu] [--n 100] [--max-num-tokens 2560]"""
import argparse, importlib.util, json, os, subprocess, sys, time
ap = argparse.ArgumentParser(); ap.add_argument("bundle"); ap.add_argument("out"); ap.add_argument("--backend", default="gpu")
ap.add_argument("--n", type=int, default=100); ap.add_argument("--max-num-tokens", type=int, default=2560); ap.add_argument("--litert-lm", default=os.path.expanduser("~/venvs/lt0171run/bin/litert-lm"))
ap.add_argument("--shard", default="0/1", help="k/K: this worker takes question i when i %% K == k; rows go to <out>.shardk.jsonl; workers run in parallel (each is its own process per question anyway)")
a = ap.parse_args()
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(root)
spec = importlib.util.spec_from_file_location("pg", os.path.join(root, "scripts", "parity_gsm8k.py")); pg = importlib.util.module_from_spec(spec); spec.loader.exec_module(pg)
import glob
k, K = (int(x) for x in a.shard.split("/"))
outk = a.out if K == 1 else a.out.replace(".jsonl", f".shard{k}.jsonl")
def done_set():
    d = set()
    for f in [a.out] + glob.glob(a.out.replace(".jsonl", ".shard*.jsonl")):
        if os.path.exists(f):
            for l in open(f): d.add(json.loads(l)["i"])
    return d
t0 = time.time()
for i, (q, gold) in enumerate(pg.load_q(a.n)):
    if i % K != k or i in done_set(): continue
    t = time.time()
    p = subprocess.run([a.litert_lm, "run", a.bundle, "--backend", a.backend, "--cache", "no", "--max-num-tokens", str(a.max_num_tokens), "--prompt", q + pg.COT],
                       capture_output=True, text=True, timeout=1800, stdin=subprocess.DEVNULL)
    txt = "\n".join(l for l in p.stdout.splitlines() if not l.startswith(("INFO:", "WARNING:", "I0", "W0")))
    pred = pg.norm(pg.extract(txt)); ok = pred == pg.norm(gold)
    open(outk, "a").write(json.dumps({"i": i, "ok": bool(ok), "pred": pred, "gold": pg.norm(gold), "chars": len(txt), "sec": round(time.time()-t, 1), "rc": p.returncode, "text": txt}) + "\n")
    print(f"  q {i+1}/{a.n} {'OK' if ok else '..'} pred={pred} gold={pg.norm(gold)} {round(time.time()-t,1)}s rc={p.returncode}", flush=True)
rows = {}
for f in [a.out] + glob.glob(a.out.replace(".jsonl", ".shard*.jsonl")):
    if os.path.exists(f):
        for l in open(f): r = json.loads(l); rows[r["i"]] = r
print(f"GSM8K litertlm-cli {a.backend} fresh-process greedy shard {a.shard} n_total={len(rows)}: {sum(r['ok'] for r in rows.values())}/{len(rows)}  total {time.time()-t0:.0f}s")
