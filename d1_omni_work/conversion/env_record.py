"""Record the reference venv (venv-ref) and the exporter venv versions -> results/env.json.

    venv-ref/bin/python scripts/env_record.py

The exporter venv is only imported here (litert_torch / ai_edge_litert / ai_edge_quantizer), nothing is run in it.
"""
import importlib.metadata as md
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

K = Path(__file__).resolve().parents[1]
EXPORTER_PY = Path.home() / "venvs/lt094dev/bin/python"

PROBE = r"""
import importlib.metadata as md, json, sys
out = {"python": sys.version.split()[0], "executable": sys.executable}
for dist, mod in [("litert-torch", "litert_torch"), ("ai-edge-litert", "ai_edge_litert"),
                  ("ai-edge-quantizer", "ai_edge_quantizer"), ("torch", "torch"), ("transformers", "transformers")]:
    rec = {}
    try:
        rec["dist_version"] = md.version(dist)
    except Exception as e:
        rec["dist_version"] = None
        rec["dist_error"] = repr(e)
    try:
        m = __import__(mod)
        rec["import_ok"] = True
        rec["module_file"] = getattr(m, "__file__", None)
        rec["module_version"] = getattr(m, "__version__", None)
    except Exception as e:
        rec["import_ok"] = False
        rec["import_error"] = repr(e)[:400]
    out[mod] = rec
print("ENVJSON " + json.dumps(out))
"""


def ref_versions():
    import PIL
    import soundfile
    import torch
    import torchvision
    import transformers
    import huggingface_hub
    from transformers import Siglip2VisionModel  # noqa: F401  (the vision tower class the provider code needs)

    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "torchvision": torchvision.__version__,
        "soundfile": soundfile.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "pillow": PIL.__version__,
        "numpy": md.version("numpy"),
        "safetensors": md.version("safetensors"),
        "tokenizers": md.version("tokenizers"),
        "siglip2_vision_model_import": True,
        "transformers_ge_5_15": tuple(int(x) for x in transformers.__version__.split(".")[:2]) >= (5, 15),
        "torch_threads": torch.get_num_threads(),
    }


def exporter_versions():
    if not EXPORTER_PY.exists():
        return {"python_path": str(EXPORTER_PY), "present": False}
    t0 = time.time()
    p = subprocess.run([str(EXPORTER_PY), "-c", PROBE], capture_output=True, text=True, timeout=600)
    line = [x for x in p.stdout.splitlines() if x.startswith("ENVJSON ")]
    rec = json.loads(line[-1][8:]) if line else {}
    rec.update({"python_path": str(EXPORTER_PY), "present": True, "returncode": p.returncode,
                "seconds": round(time.time() - t0, 1), "stderr_tail": p.stderr.strip().splitlines()[-3:]})
    return rec


def main():
    out = {
        "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "venv_ref": {"path": str(K / "venv-ref"), **ref_versions()},
        "venv_ref_requirements": "requirements-ref.txt",
        "install_command": 'uv pip install --python venv-ref/bin/python torch "transformers>=5.15" torchvision pillow '
                           'soundfile numpy safetensors "huggingface_hub>=0.30" (versions left to the resolver, no pin)',
        "exporter_venv_lt094dev": exporter_versions(),
        "host": {"uname": " ".join(os.uname())},
    }
    path = K / "results/env.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"venv_ref": {k: out["venv_ref"][k] for k in ("python", "torch", "transformers", "torchvision",
                                                                    "soundfile", "huggingface_hub")},
                      "exporter": {k: (v.get("dist_version"), v.get("import_ok")) if isinstance(v, dict) else v
                                   for k, v in out["exporter_venv_lt094dev"].items()
                                   if k in ("litert_torch", "ai_edge_litert", "ai_edge_quantizer", "torch",
                                            "transformers", "python")}}, indent=1))


if __name__ == "__main__":
    main()
