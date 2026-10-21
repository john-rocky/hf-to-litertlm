#!/usr/bin/env python3
"""Round 3 rehearsal of the tower's LiteRT path on a tiny random ClefTower (no checkpoint read): the same module code
(scripts/r3_tower.py), the same N = 256 patches of 1536 values, hidden 64, 4 heads, FFN 96, depth 2, out 80 ->
litert_torch.convert (the default path) -> the static scan -> the fp16-FC form (scripts/r3_fp16fc.py) -> CompiledModel
CPU (XNNPACK 8 threads), Metal fp32 and Metal default, each against the torch module on two random inputs.

    TMPDIR=<K>/tmp ~/venvs/lt094dev/bin/python scripts/r3_tiny_tfl.py      -> results/r3_tiny_tfl.json,
        out/r3/tiny_tower_fp32.tflite, out/r3/tiny_tower_v2_fp16fc.tflite (deleted at the end), logs/r3_tiny_*.runtime.log

What it can show before the real export: an op the scan forbids (GATHER_ND / SELECT_V2 / BROADCAST_TO / INT64 / CUSTOM /
rank > 4), a node Metal does not take (the delegation lines), a wrong signature name, a numeric gap between the file and
the module far above float32 noise."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

K = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K / "scripts"))
import r3_tower as RT  # noqa: E402
import r3_fp16fc as RQ  # noqa: E402
import r3_lrt as RL  # noqa: E402
from clef_scan import full_scan  # noqa: E402

FORBID = ("GATHER_ND", "SELECT_V2", "BROADCAST_TO")


def tower_form() -> dict:
    """The form scripts/r3_metal_bisect.py chose (cache/r3/tower_form.json), else the default."""
    p = K / "cache/r3/tower_form.json"
    return json.loads(p.read_text())["form"] if p.exists() else dict(RT.DEFAULT_FORM)


def stats(got, want) -> dict:
    g = np.asarray(got, np.float64).ravel()
    w = np.asarray(want, np.float64).ravel()
    return {"rel_rms": float(np.linalg.norm(g - w) / np.linalg.norm(w)), "max_abs": float(np.abs(g - w).max()),
            "cos": float(g @ w / (np.linalg.norm(g) * np.linalg.norm(w))), "finite": bool(np.isfinite(g).all())}


def main() -> int:
    out = K / "results/r3_tiny_tfl.json"
    assert not out.exists(), f"refusing to overwrite {out}"
    import litert_torch

    torch.set_grad_enabled(False)
    t_all = time.time()
    n, hidden, heads = 256, 64, 4
    form = tower_form()
    tower, xs = RT.tiny_tower(form, n=n, hidden=hidden, heads=heads)
    feeds = [tower.feed(x) for x in xs]
    ys = [tower(**f)["rows"].numpy() for f in feeds]
    f32 = K / "out/r3/tiny_tower_fp32.tflite"
    f16 = K / "out/r3/tiny_tower_v2_fp16fc.tflite"
    f32.parent.mkdir(parents=True, exist_ok=True)
    for p in (f32, f16):
        if p.exists():
            p.unlink()                         # the rehearsal's own regenerable files
    doc = {"what": "round 3 rehearsal: tiny random ClefTower -> litert-torch -> scan -> fp16fc -> CompiledModel",
           "tiny": {"n": n, "hidden": hidden, "heads": heads, "ffn": 96, "depth": 2, "out": 80}, "form": form}
    t0 = time.perf_counter()
    edge = litert_torch.convert(tower, sample_kwargs=feeds[0])
    edge.export(str(f32))
    doc["convert_seconds"] = round(time.perf_counter() - t0, 2)
    s = full_scan(f32)
    doc["scan"] = {k: s[k] for k in ("bytes", "operator_count", "op_histogram", "signatures", "forbidden_counts",
                                     "int64_tensor_count", "rank_gt4_tensor_count", "custom_op_count", "stablehlo_ops",
                                     "gather_count", "tensor_rank_histogram", "batch_matmul_count", "fully_connected_count")}
    doc["scan"]["all_constant_input_ops"] = s["lightweight"]["all_constant_input_ops"]
    doc["fp16fc"] = RQ.fp16fc(f32, f16)
    doc["fold_census_fp32"] = {k: v for k, v in RQ.fold_census(f32).items() if k != "rows"}
    res = {}
    for f in (f32, f16):
        for accel in ("cpu", "gpu_f32", "gpu_default"):
            key = f"{f.stem}_{accel}"
            item = {}
            log = K / f"logs/r3_tiny_{key}.runtime.log"
            rl = RL.RuntimeLog(log)
            try:
                with rl:
                    m = RL.Model(f, accel)
                    outs = [m.run({k: v.numpy() for k, v in f.items()})["rows"] for f in feeds]
                    again = m.run({k: v.numpy() for k, v in feeds[0].items()})["rows"]
                    item.update(m.info())
                    m.close()
                item.update({"vs_torch": [stats(o, y) for o, y in zip(outs, ys)],
                             "repeat_bit_equal": bool(np.array_equal(again, outs[0])), "status": "OK"})
            except BaseException as e:  # a refused compile is a result
                item.update(status="FAIL", error=f"{type(e).__name__}: {e}")
            item["delegation"] = rl.delegation
            item["runtime_log_key_lines"] = rl.key_lines[:20]
            item["runtime_log"] = str(log.relative_to(K))
            res[key] = item
            print(key, item.get("status"), item.get("error", ""), [round(v["rel_rms"], 9) for v in item.get("vs_torch", [])],
                  item.get("delegation"), flush=True)
    doc["runtime"] = res
    sig = s["signatures"]
    checks = {"signature_patches_to_rows": (len(sig) == 1 and sorted(i["name"] for i in sig[0]["inputs"]) == sorted(feeds[0])
                                            and [o["name"] for o in sig[0]["outputs"]] == ["rows"]),
              "forbidden_0": all(s["forbidden_counts"].get(k, 0) == 0 for k in FORBID),
              "int64_0": s["int64_tensor_count"] == 0, "rank_gt4_0": s["rank_gt4_tensor_count"] == 0,
              "custom_0": s["custom_op_count"] == 0, "gather_0": s["gather_count"] == 0,
              "fp16fc_checks": doc["fp16fc"]["checks_pass"],
              "cpu_ok": all(res[f"{f.stem}_cpu"]["status"] == "OK" for f in (f32, f16)),
              "metal_f32_ok": all(res[f"{f.stem}_gpu_f32"]["status"] == "OK" for f in (f32, f16)),
              "fp32_cpu_close": res[f"{f32.stem}_cpu"].get("status") == "OK"
              and max(v["rel_rms"] for v in res[f"{f32.stem}_cpu"]["vs_torch"]) < 1e-5,
              "fp32_metal_f32_close": res[f"{f32.stem}_gpu_f32"].get("status") == "OK"
              and max(v["rel_rms"] for v in res[f"{f32.stem}_gpu_f32"]["vs_torch"]) < 1e-4}
    doc.update(checks=checks, checks_pass=all(checks.values()), seconds=round(time.time() - t_all, 1))
    for p in (f32, f16):
        p.unlink()
    out.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(json.dumps({"checks": checks, "scan": {k: doc["scan"][k] for k in ("operator_count", "op_histogram", "forbidden_counts")}},
                     indent=1))
    return 0 if doc["checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
