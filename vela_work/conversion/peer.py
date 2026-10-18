"""Load a sibling conversion folder's module by file path (no sys.path games: a `graph.py` / `quantize.py` of another folder must never
shadow this folder's own; the first lookup export imported phocinae_work's graph.py through sys.path.insert(0, ...))."""
import importlib.util
import sys
from pathlib import Path

W = Path(__file__).resolve().parents[1]
PHOCINAE = W.parent / "phocinae_work" / "conversion"


def load(name, path):
    path = Path(path)
    spec = importlib.util.spec_from_file_location(f"peer_{name}", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tflite_scan = load("tflite_scan", PHOCINAE / "tflite_scan.py")          # generic static scanner (scan, sha256_file)
sys.modules.setdefault("tflite_scan", tflite_scan)                        # phocinae's quantize.py does `from tflite_scan import scan`


def load_phocinae_quantize():
    """phocinae_work/scripts/quantize.py does `from common import W, dump_json, sha256_file` (its own common.py): register that
    module under the name for the duration of the import only, then drop it again so nothing of this folder sees a foreign `common`."""
    had = "common" in sys.modules
    prev = sys.modules.get("common")
    sys.modules["common"] = load("common", PHOCINAE / "common.py")
    try:
        return load("quantize", PHOCINAE / "quantize.py")
    finally:
        if had:
            sys.modules["common"] = prev
        else:
            del sys.modules["common"]
