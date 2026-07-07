"""Export an HF model keeping temp files (so model.tflite / model_quantized.tflite remain).

    ~/clipconv/bin/python scripts/export_keep.py <model_path_or_id> <out_dir> [prefill] [cache]
"""
import sys, types, os
class _D:
    def __getattr__(self, n): return lambda *a, **k: None
    def __call__(self, *a, **k): return None
_pp = types.ModuleType("scipy.sparse.linalg._propack"); _pp.__file__="<s>"; _pp.__spec__=None
for _nm in ("_spropack","_dpropack","_cpropack","_zpropack"): setattr(_pp,_nm,_D())
sys.modules["scipy.sparse.linalg._propack"]=_pp
_opt=types.ModuleType("scipy.optimize"); _opt.__file__="<s>"; _opt.__spec__=None
_opt.linear_sum_assignment=lambda *a,**k: None
sys.modules["scipy.optimize"]=_opt
import inspect as _inspect
_orig=_inspect.getsourcefile
def _try(o):
    try: _orig(o); return True
    except Exception: return False
_inspect.getsourcefile=lambda o:(_orig(o) if _try(o) else None)

model=sys.argv[1]; out=sys.argv[2]
prefill=int(sys.argv[3]) if len(sys.argv)>3 else 16
cache=int(sys.argv[4]) if len(sys.argv)>4 else 64
os.makedirs(out, exist_ok=True)
from litert_torch.generative.export_hf.export import export
export(model=model, output_dir=out, prefill_lengths=[prefill], cache_length=cache,
       quantization_recipe="dynamic_wi8_afp32", trust_remote_code=True, keep_temporary_files=True)
print("EXPORT_DONE", out)
