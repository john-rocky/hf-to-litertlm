"""Export a model with temp files kept, then dump the tflite op histogram.

Used to substantiate "MLA decompress lowers to GENERIC ops (no Flex/Custom)".
    ~/clipconv/bin/python scripts/extract_ops_dsv3.py <hf_model_or_dir> <out_dir>
"""
import sys, types, os, glob, collections

class _D:
    def __getattr__(self, n): return lambda *a, **k: None
    def __call__(self, *a, **k): return None
_pp = types.ModuleType("scipy.sparse.linalg._propack"); _pp.__file__ = "<stub>"; _pp.__spec__ = None
for _nm in ("_spropack", "_dpropack", "_cpropack", "_zpropack"): setattr(_pp, _nm, _D())
sys.modules["scipy.sparse.linalg._propack"] = _pp
_opt = types.ModuleType("scipy.optimize"); _opt.__file__ = "<stub>"; _opt.__spec__ = None
_opt.linear_sum_assignment = lambda *a, **k: None
sys.modules["scipy.optimize"] = _opt
import inspect as _inspect
_orig = _inspect.getsourcefile
_inspect.getsourcefile = lambda o: (_orig(o) if _try(o) else None)
def _try(o):
    try: _orig(o); return True
    except Exception: return False

model = sys.argv[1]
out_dir = sys.argv[2]
os.makedirs(out_dir, exist_ok=True)

from litert_torch.generative.export_hf.export import export
export(
    model=model, output_dir=out_dir,
    prefill_lengths=[int(os.environ.get("PREFILL", "32"))],
    cache_length=int(os.environ.get("CACHE", "128")),
    quantization_recipe="dynamic_wi8_afp32",
    trust_remote_code=True, keep_temporary_files=True,
)

from ai_edge_litert.interpreter import Interpreter
for tfl in sorted(glob.glob(os.path.join(out_dir, "*.tflite"))):
    itp = Interpreter(model_path=tfl)
    itp.allocate_tensors()
    ops = collections.Counter(d["op_name"] for d in itp._get_ops_details())
    flex = [k for k in ops if "Flex" in k or "CUSTOM" in k.upper() or "DELEGATE" in k.upper()]
    print(f"\n=== {os.path.basename(tfl)}  ({sum(ops.values())} ops, {len(ops)} types) ===")
    for k, v in ops.most_common():
        print(f"  {v:5d}  {k}")
    print(f"  FLEX/CUSTOM: {flex if flex else 'NONE'}")
