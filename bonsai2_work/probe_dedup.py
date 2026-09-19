"""Is the same-engine contamination avoidable? One GPU engine: q4 fresh -> q4 again -> q19 -> q4 with a unique system
message -> q4 after engine-level tricks. Prints preds (fresh-engine reference: q4=540, q19=7). Also lists API methods."""
import importlib.util, os, sys, time, random
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(root)
spec = importlib.util.spec_from_file_location("pg", os.path.join(root, "scripts", "parity_gsm8k.py")); pg = importlib.util.module_from_spec(spec); spec.loader.exec_module(pg)
from litert_lm import engine as engine_lib, interfaces
M = sys.argv[1]; qs = pg.load_q(20); t0 = time.time()
eng = engine_lib.Engine(M, backend=interfaces.Backend.GPU, max_num_tokens=4096)
print(f"engine ready {time.time()-t0:.0f}s", flush=True)
print("engine methods:", [m for m in dir(eng) if not m.startswith('_')])
def ask(i, **kw):
    q, gold = qs[i]; conv = eng.create_conversation(sampler_config=interfaces.SamplerConfig(top_k=1), **kw); t = time.time()
    resp = conv.send_message(q + pg.COT, max_output_tokens=600); txt = "".join(c.get("text", "") for c in resp.get("content", []) if isinstance(c, dict))
    if i == 3 and not hasattr(ask, "printed"): print("conversation methods:", [m for m in dir(conv) if not m.startswith('_')]); ask.printed = True
    conv.close(); print(f"q{i+1} gold={gold} pred={pg.norm(pg.extract(txt))} chars={len(txt)} {time.time()-t:.0f}s {kw.keys()} | {txt[:70]!r}", flush=True)
ask(3); ask(3); ask(18); ask(3)
ask(3, system_message=f"Session {random.randint(10**6, 10**7)}. You are a helpful assistant."); ask(18, system_message=f"Session {random.randint(10**6, 10**7)}. You are a helpful assistant.")
ask(3, system_message=f"Session {random.randint(10**6, 10**7)}. You are a helpful assistant.")
