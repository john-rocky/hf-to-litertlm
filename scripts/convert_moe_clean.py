"""Convert an HF MoE checkpoint to LiteRT via the batched_mm (generic gather+bmm)
experts path. Relies on export_hf/core/export_lib.py auto-forcing `batched_mm`
(it detects set_experts_implementation + _can_set_experts_implementation). No scipy
stubs — the clipconv venv's scipy is healthy now, and the old stubs broke on
transformers' new `milp` import.

    ~/clipconv/bin/python scripts/convert_moe_clean.py <hf_model> <out_dir> [quant] [name]

env: PREFILL (default 128), CACHE (default 1024).
"""
import sys, os, json, traceback

model_id = sys.argv[1]
out_dir = sys.argv[2]
quant = sys.argv[3] if len(sys.argv) > 3 else "dynamic_wi8_afp32"
name = sys.argv[4] if len(sys.argv) > 4 else model_id.split("/")[-1]

PROBE_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "probes")
os.makedirs(PROBE_DIR, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)
res = {"model": model_id, "name": name, "quant": quant, "ok": False,
       "stage": None, "error_type": None, "error_head": None, "failing_op": None}
try:
    from litert_torch.generative.export_hf.export import export
    res["stage"] = "export-call"
    export(model=model_id, output_dir=out_dir,
           prefill_lengths=[int(os.environ.get("PREFILL", "128"))],
           cache_length=int(os.environ.get("CACHE", "1024")),
           quantization_recipe=quant, trust_remote_code=True)
    res["ok"] = True
    res["stage"] = "done"
except BaseException as e:  # noqa: BLE001
    tb = traceback.format_exc()
    res["error_type"] = type(e).__name__
    msg = str(e).strip()
    res["error_head"] = (msg.splitlines()[0] if msg else tb.strip().splitlines()[-1])[:400]
    for ln in tb.splitlines():
        s = ln.strip()
        if ("While executing" in s) or s.startswith("%") or ("aten." in s) or ("torch.ops" in s):
            res["failing_op"] = s[:400]
            break
    with open(os.path.join(PROBE_DIR, f"{name}.trace.txt"), "w") as f:
        f.write(tb)
with open(os.path.join(PROBE_DIR, f"{name}.json"), "w") as f:
    json.dump(res, f, indent=2)
print("CONVERT_RESULT", json.dumps(res))
