"""Smoke-test the batched_mm MoE export+runtime path with a TINY random-weight OLMoE
before committing to the real 7B (~14GB) download. Same arch code path (routing/attention)
as OLMoE-1B-7B, just tiny dims -> if this exports and runs on the runtime, the real one will.

    ~/clipconv/bin/python scripts/convert_moe_tiny.py <out_dir>
"""
import sys, os, json, traceback
import torch
from transformers import OlmoeConfig, OlmoeForCausalLM, AutoTokenizer

out_dir = sys.argv[1]
os.makedirs(out_dir, exist_ok=True)
src_dir = os.path.join(out_dir, "tiny_olmoe_src")
PROBE_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "probes")
os.makedirs(PROBE_DIR, exist_ok=True)

# Tiny config; keep real vocab so the OLMoE tokenizer stays compatible.
cfg = OlmoeConfig(
    hidden_size=128, intermediate_size=128, num_hidden_layers=2,
    num_attention_heads=8, num_key_value_heads=8,
    num_experts=8, num_experts_per_tok=2,
    vocab_size=50304, max_position_embeddings=1024,
)
torch.manual_seed(0)
model = OlmoeForCausalLM(cfg).eval()
model.save_pretrained(src_dir)
try:
    tok = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924-Instruct")
    tok.save_pretrained(src_dir)
except Exception as e:  # noqa: BLE001
    print("tokenizer fetch failed:", e)

res = {"name": "tiny-olmoe", "ok": False, "stage": None, "error_type": None,
       "error_head": None, "failing_op": None}
try:
    from litert_torch.generative.export_hf.export import export
    res["stage"] = "export-call"
    export(model=src_dir, output_dir=out_dir,
           prefill_lengths=[128], cache_length=1024,
           quantization_recipe="dynamic_wi8_afp32", trust_remote_code=True)
    res["ok"] = True
    res["stage"] = "done"
except BaseException as e:  # noqa: BLE001
    tb = traceback.format_exc()
    res["error_type"] = type(e).__name__
    msg = str(e).strip()
    res["error_head"] = (msg.splitlines()[0] if msg else tb.strip().splitlines()[-1])[:400]
    for ln in tb.splitlines():
        s = ln.strip()
        if ("While executing" in s) or ("aten." in s) or ("modeling_olmoe" in s):
            res["failing_op"] = s[:400]
    with open(os.path.join(PROBE_DIR, "tiny-olmoe.trace.txt"), "w") as f:
        f.write(tb)
with open(os.path.join(PROBE_DIR, "tiny-olmoe.json"), "w") as f:
    json.dump(res, f, indent=2)
print("TINY_RESULT", json.dumps(res))
