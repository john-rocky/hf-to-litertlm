"""LiteRT-LM #3444 re-check on the PyPI runtime.

Does gemma-4-E2B, opened with max_num_tokens=4096, still die near token_count ~2000 with
FAILED_PRECONDITION? Five ~785-token user turns with 4-token replies, greedy.

    python probe_3444_ceiling.py gemma-4-E2B-it.litertlm        # CPU backend
    python probe_3444_ceiling.py gemma-4-E2B-it.litertlm gpu    # the wheel's GPU backend (WebGPU)

Runs unchanged on litert-lm 0.16.1 and 0.17.0 (pip install litert-lm==<version>)."""
import sys, time, litert_lm
bundle = sys.argv[1]
para = ("The quick brown fox jumps over the lazy dog while the river keeps flowing past the old mill. ") * 40  # ~700+ tokens
eng = litert_lm.Engine(bundle, backend=litert_lm.Backend.CPU() if len(sys.argv) < 3 or sys.argv[2] != "gpu" else litert_lm.Backend.GPU(), max_num_tokens=4096)
print("engine max_num_tokens accepted: 4096", flush=True)
sc = litert_lm.SamplerConfig(top_k=1, temperature=0.0, seed=0)
with eng as engine, engine.create_conversation(sampler_config=sc) as conv:
    for i in range(1, 8):
        t0 = time.time()
        try:
            r = conv.send_message(f"Turn {i}. Read this and reply with the single word OK.\n{para}", max_output_tokens=4)
            print(f"turn {i}: OK {time.time()-t0:.1f}s token_count={conv.token_count} -> {str(r)[:60]!r}", flush=True)
        except Exception as e:
            print(f"turn {i}: FAIL after {time.time()-t0:.1f}s token_count_before_turn=? {type(e).__name__}: {str(e)[:300]}", flush=True)
            break
