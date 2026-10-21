#!/usr/bin/env python3
"""Round 3: every graph form of scripts/r3_tower.py computes the same values in torch as the default form (the one the
torch check measured), on the tiny random tower (no checkpoint read) — so a form change for the GPU never changes the
function. Value equality (np.array_equal: -0.0 == 0.0), max |diff| recorded.

    venv-ref/bin/python scripts/r3_forms_check.py            -> results/r3_forms_check_<torch version>.json
    ~/venvs/lt094dev/bin/python scripts/r3_forms_check.py"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

K = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K / "scripts"))
import r3_tower as RT  # noqa: E402


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else ""
    out = K / f"results/r3_forms_check{tag}_torch{torch.__version__.split('+')[0]}.json"
    assert not out.exists(), f"refusing to overwrite {out}"
    torch.set_grad_enabled(False)
    rows = []
    merges = ("reshape", "strided", "split4") if tag else ("reshape", "strided")
    poss = ("const", "pos4", "posin") if tag == "_pos" else ("const",)
    for n in (256, 784):
        base, xs = RT.tiny_tower(None, n=n)
        y0 = [base(x)["rows"].numpy() for x in xs]
        for rope, rot, merge, pos in itertools.product(("bcast", "full", "bhnd", "fullbhnd"), ("slice", "mm"), merges, poss):
            form = {"rope": rope, "rot": rot, "merge": merge, "pos": pos}
            t, xs2 = RT.tiny_tower(form, n=n)
            assert all(torch.equal(a, b) for a, b in zip(xs, xs2))
            y = [t(**t.feed(x))["rows"].numpy() for x in xs]
            rel = max(float(np.linalg.norm(a.astype(np.float64) - b) / np.linalg.norm(b)) for a, b in zip(y, y0))
            rows.append({"n": n, "form": form, "equal": all(np.array_equal(a, b) for a, b in zip(y, y0)),
                         "max_abs": max(float(np.abs(a - b).max()) for a, b in zip(y, y0)), "rel_rms": rel,
                         "reassociates": merge == "split4"})
    exact = [r for r in rows if not r["reassociates"]]
    reas = [r for r in rows if r["reassociates"]]
    doc = {"what": "graph forms vs the default form in torch (tiny random tower)", "torch": torch.__version__,
           "default": RT.DEFAULT_FORM, "rows": rows, "all_equal": all(r["equal"] for r in exact),
           "reassociating_max_rel_rms": max((r["rel_rms"] for r in reas), default=None)}
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"torch": torch.__version__, "all_equal": doc["all_equal"],
                      "not_equal": [r for r in exact if not r["equal"]],
                      "reassociating_max_rel_rms": doc["reassociating_max_rel_rms"]}, indent=1))
    return 0 if doc["all_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
