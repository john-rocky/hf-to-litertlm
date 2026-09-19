"""Run specific GSM8K questions (0-based indices) on a bundle via the Python API, one engine, greedy; print the answers.
Usage: probe_q.py <bundle> <cpu|gpu> <i,i,i> [max_tokens]"""
import importlib.util, os, sys, time
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(root)
spec = importlib.util.spec_from_file_location("pg", os.path.join(root, "scripts", "parity_gsm8k.py")); pg = importlib.util.module_from_spec(spec); spec.loader.exec_module(pg)
from litert_lm import engine as engine_lib, interfaces
M, backend, idx = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(",")]; MT = int(sys.argv[4]) if len(sys.argv) > 4 else 600
qs = pg.load_q(max(idx) + 1); t0 = time.time()
eng = engine_lib.Engine(M, backend=interfaces.Backend.GPU if backend == "gpu" else interfaces.Backend.CPU, max_num_tokens=4096)
print(f"[{backend}] engine ready {time.time()-t0:.0f}s", flush=True)
for i in idx:
    q, gold = qs[i]; conv = eng.create_conversation(sampler_config=interfaces.SamplerConfig(top_k=1)); t = time.time()
    resp = conv.send_message(q + pg.COT, max_output_tokens=MT); txt = "".join(c.get("text", "") for c in resp.get("content", []) if isinstance(c, dict)); conv.close()
    print(f"[{backend}] q{i+1} gold={gold} pred={pg.norm(pg.extract(txt))} {time.time()-t:.0f}s chars={len(txt)}\n    head: {txt[:160]!r}\n    tail: {txt[-120:]!r}", flush=True)
