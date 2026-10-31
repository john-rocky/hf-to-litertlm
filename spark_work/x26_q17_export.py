#!/usr/bin/env python3
"""x26_q17_export.py <args of spark_work/convert_spark.py, recipe name Q17> -- the Spark-X2.6-1.7B mixed int4 / int8
export (the `_int4-mlp` file: attention int8, MLP gate / up int4 blockwise-32 OCTAV, MLP down int8, lm_head int8), in one
process:

1. The 2^-14 floor on blockwise int4 scales, identical to spark_work/x26_export_clamped.py: ai_edge_quantizer's tensor_zp_scale_from_min_max wrapped so a blockwise 4-bit
   symmetric request returns max(scale, 2^-14); every floored tensor logged, `CLAMP_TOTAL` printed at exit.
2. A recipe registered as `ai_edge_quantizer.recipe.Q17` (export_lib resolves the name through recipe_lib.__dict__):
   rule 1 = dynamic_wi8_afp32's rule (regex .*, every op: int8 channelwise, min-max) = the int8 file's base; then one rule
   per int4 group of Q17_PLAN = the BOCTAV4 rule of scripts/export_simple_template.py (int4 BLOCKWISE_32, OCTAV) with
   operation FULLY_CONNECTED and that group's scope regex from spark_work/x26_q17_groups.py (checked on every FC of the
   int8 and the int4 b32 graphs). ai-edge-quantizer uses the last matching rule, so an FC is int4 exactly when its scope
   matches an int4 group's regex. The embedding lookup stays int8 (rule 1); EXTERNALIZE_EMBEDDER=1 puts it in its own
   section as in the b32 form.
3. spark_work/convert_spark.py runs in this process via runpy (NO_START_TOKEN / USE_JINJA / CACHE / PREFILL, then
   scripts/export_simple_template.py, whose own recipe registrations leave Q17 in place).

  Q17_PLAN=attn=8,mlp_gu=4,mlp_down=8,lm_head=8 EXTERNALIZE_EMBEDDER=1 python \
      spark_work/x26_q17_export.py src_models/Spark-X2.6-1.7B-export out/spark26-1.7b-q17 templates/spark26_think.jinja Q17
"""
import atexit
import copy
import importlib.util
import json
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "convert_spark.py")

import transformers  # noqa: E402,F401  (the export script's first import; keep the order)
from litert_torch.generative.export_hf.core import export_lib  # noqa: E402,F401
import numpy as np  # noqa: E402
from ai_edge_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor as _u  # noqa: E402
import ai_edge_quantizer.recipe as _aqr  # noqa: E402

_FLOOR = np.float32(2.0 ** -14)
_orig = _u.tensor_zp_scale_from_min_max
_stats = {"tensors": 0, "scales": 0}


def _clamped(min_value, max_value, num_bits, symmetric, granularity, clipping_values=None):
  zp, scale = _orig(min_value, max_value, num_bits, symmetric, granularity, clipping_values)
  if _u.is_blockwise(granularity) and num_bits == 4:
    if not symmetric:
      raise ValueError("q17_export: asymmetric blockwise int4 is not covered by this floor")
    low = scale < _FLOOR
    n = int(np.count_nonzero(low))
    if n:
      scale = np.where(low, _FLOOR, scale).astype(scale.dtype)
      _stats["tensors"] += 1
      _stats["scales"] += n
      print(f"CLAMP floor 2^-14: {n} of {scale.size} block scales", flush=True)
  return zp, scale


_u.tensor_zp_scale_from_min_max = _clamped
atexit.register(lambda: print(f"CLAMP_TOTAL tensors={_stats['tensors']} scales={_stats['scales']}", flush=True))
print("PATCHED uniform_quantize_tensor.tensor_zp_scale_from_min_max: blockwise int4 scales floored at 2^-14", flush=True)

_spec = importlib.util.spec_from_file_location("q17_groups", os.path.join(HERE, "x26_q17_groups.py"))
_g = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_g)
plan = dict(kv.split("=") for kv in os.environ["Q17_PLAN"].split(","))
if set(plan) != set(_g.GROUP_RX) or not set(plan.values()) <= {"4", "8"}:
  raise SystemExit(f"Q17_PLAN must give 4 or 8 for each of {sorted(_g.GROUP_RX)}: {plan}")
_I4 = _aqr.dynamic_wi4_afp32()[0]
_I8 = copy.deepcopy(_I4)
_I8["op_config"]["weight_tensor_config"]["num_bits"] = 8
_BO4 = copy.deepcopy(_I4)
_BO4["algorithm_key"] = _aqr.AlgorithmName.OCTAV
_BO4["op_config"]["weight_tensor_config"]["granularity"] = "BLOCKWISE_32"
_rules = [copy.deepcopy(_I8)]
for grp, bits in plan.items():
  if bits == "4":
    r = copy.deepcopy(_BO4)
    r["regex"] = _g.GROUP_RX[grp]
    r["operation"] = "FULLY_CONNECTED"
    _rules.append(r)
_aqr.Q17 = lambda: copy.deepcopy(_rules)
print("Q17_PLAN", plan, flush=True)
print("Q17_RECIPE " + json.dumps(_rules, default=str), flush=True)

sys.argv = [SCRIPT] + sys.argv[1:]
runpy.run_path(SCRIPT, run_name="__main__")
