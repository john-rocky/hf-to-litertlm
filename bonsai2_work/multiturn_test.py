"""Two-turn conversation through the litert_lm Python API (the CLI's engine): renders the template branches the
single-turn `run` never touches (assistant history, reasoning_content|trim ...). Usage: multiturn_test.py <model> [max_out]"""
import sys, time
from litert_lm import engine as engine_lib
from litert_lm import interfaces
M = sys.argv[1]; MO = int(sys.argv[2]) if len(sys.argv) > 2 else 200
eng = engine_lib.Engine(M, max_num_tokens=2048)
conv = eng.create_conversation(sampler_config=interfaces.SamplerConfig(top_k=1))
for i, q in enumerate(["What is 17 + 25? Answer with just the number.", "Now multiply that result by 2. Answer with just the number."]):
    t = time.time(); resp = conv.send_message(q, max_output_tokens=MO)
    text = "".join(c.get("text", "") for c in resp.get("content", []) if isinstance(c, dict))
    print(f"turn {i+1} ({time.time()-t:.1f}s): {text[-220:]!r}", flush=True)
conv.close(); print("MULTITURN_OK")
