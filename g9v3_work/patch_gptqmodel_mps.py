#!/usr/bin/env python3
"""patch_gptqmodel_mps.py [GPTQMODEL_PACKAGE_DIR] - the five edits that let gptqmodel 7.3.5 quantize on Apple silicon (MPS).
Without them the worker pool keys `mps` and `mps:0` apart (the module never finds its device) and eight concurrent MPS workers
crash Metal. Default directory = the `gptqmodel` package of the running python (run it with the venv's python). Each edit asserts
its anchor text exists exactly once and prints a unified diff; a second run stops at the first edit (the anchor is gone)."""
import difflib
import os
import sys

if len(sys.argv) > 1:
    G = sys.argv[1]
else:
    import importlib.util
    G = os.path.dirname(importlib.util.find_spec("gptqmodel").origin)
EDITS = [
    ("utils/threadx.py",
     '    def _key(self, dev: torch.device) -> str:\n        idx = "" if dev.index is None else f":{dev.index}"\n',
     '    def _key(self, dev: torch.device) -> str:\n        if dev.type == "mps":  # MPS edit 1: one pool key for mps / mps:0\n            return "mps"\n        idx = "" if dev.index is None else f":{dev.index}"\n'),
    ("utils/threadx.py",
     '            if include_mps and _mps_available():\n                discovered.append(torch.device("mps"))\n',
     '            if include_mps and _mps_available():\n                discovered.append(torch.device("mps", 0))  # MPS edit 1: = tensor.device\n'),
    ("looper/gptq_processor.py",
     '            expected_device = torch.device(expected_device)\n',
     '            expected_device = torch.device(expected_device)\n            if expected_device.type == "mps":  # MPS edit 2\n                expected_device = torch.device("mps", 0)\n'),
    ("looper/stage_subset.py",
     '            actual_device = get_device(module_weight)\n',
     '            actual_device = get_device(module_weight)\n            if getattr(actual_device, "type", None) == "mps" and target_device.type == "mps":  # MPS edit 2b\n                target_device = actual_device\n'),
    ("__init__.py",
     '            "mps": 8,\n',
     '            "mps": 1,  # MPS edit 3: 8 concurrent MPS threads crash Metal\n'),
]
for rel, old, new in EDITS:
    p = os.path.join(G, rel)
    s = open(p).read()
    n = s.count(old)
    if n != 1:
        sys.exit(f"anchor found {n} times in {rel}: {old!r}")
    t = s.replace(old, new)
    open(p, "w").write(t)
    sys.stdout.writelines(difflib.unified_diff(s.splitlines(True), t.splitlines(True), "a/" + rel, "b/" + rel, n=0))
print("PATCHED 5 edits in", G)
