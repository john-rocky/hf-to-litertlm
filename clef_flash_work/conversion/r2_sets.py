"""Round 2: the pre-registered gate row sets (in the round record) and the reference rows they are scored against.

    import r2_sets as S
    runs = S.gate_runs(L, "gpu_f32")       # [{"id", "arm", "source", "tokens", "ids", "kind", ...}], in a fixed order
    row = S.ref_row(run); npz = S.ref_npz(run)

kind = fixture (ref/records_ref.json, arm text), heldout (Core AI's held-out reference, rev 17f0b0ad = the same
weights, arm text; K/ref/heldout), red (ref/records_ref_red.json).
"Smallest bucket L" = tokens in (previous L, L]. "First N" = fixture text runs in the order of ref/records_ref.json
whose tokens <= L, the first N of them (held-out never). A set is the union, each run once, in the order: fixture
(file order), held-out (file order), red arms (file order).

    L     gpu_f32 / gpu_default / cpu where marked "all"           cpu (and gpu_f32 at 2048 / 4096: head 40)
    256   all that fit (cpu too)                                   -
    512   all that fit                                             cpu: smallest bucket + first 20 + red arms
    1024  all that fit (cpu too)                                   -
    2048  smallest bucket + first 40 + red arms                    cpu: smallest bucket + first 20 + red arms
    4096  smallest bucket + first 40 + red arms                    cpu: smallest bucket + first 20 + red arms"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

K = Path(__file__).resolve().parents[1]
HELDOUT = K / "ref/heldout"   # the held-out reference (records file + npz), see conversion/README.md
BUCKETS = (256, 512, 1024, 2048, 4096)
ACCS = ("cpu", "gpu_f32", "gpu_default")


@lru_cache(maxsize=None)
def _load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def fixture_rows() -> list:
    return [r for r in _load(str(K / "ref/records_ref.json"))["rows"] if r["arm"] == "text"]


def heldout_rows() -> list:
    return [r for r in _load(str(HELDOUT / "records_ref.json"))["rows"] if r["arm"] == "text"]


def red_rows() -> list:
    return list(_load(str(K / "ref/records_ref_red.json"))["rows"])


def smallest_bucket(tokens: int) -> int:
    return next(b for b in BUCKETS if tokens <= b)


def _run(r: dict, kind: str) -> dict:
    return {"id": r["id"], "arm": r["arm"], "source": r["source"], "tokens": r["tokens"], "kind": kind}


def gate_runs(L: int, acc: str) -> list:
    assert L in BUCKETS and acc in ACCS, (L, acc)
    fx, ho, red = fixture_rows(), heldout_rows(), red_rows()
    every = (L in (256, 1024)) or (L == 512 and acc != "cpu")
    if every:
        pick_fx = [r for r in fx if r["tokens"] <= L]
        pick_ho = [r for r in ho if r["tokens"] <= L]
    else:
        head_n = 20 if acc == "cpu" else 40
        head = [r for r in fx if r["tokens"] <= L][:head_n]
        small = [r for r in fx if smallest_bucket(r["tokens"]) == L]
        keys = {(r["id"], r["arm"]) for r in head + small}
        pick_fx = [r for r in fx if (r["id"], r["arm"]) in keys]
        pick_ho = [r for r in ho if smallest_bucket(r["tokens"]) == L]
    pick_red = [r for r in red if r["tokens"] <= L]
    return [_run(r, "fixture") for r in pick_fx] + [_run(r, "heldout") for r in pick_ho] + [_run(r, "red") for r in pick_red]


def ref_row(run: dict) -> dict:
    src = {"fixture": fixture_rows, "heldout": heldout_rows, "red": red_rows}[run["kind"]]()
    return next(r for r in src if r["id"] == run["id"] and r["arm"] == run["arm"])


def ref_npz(run: dict) -> Path:
    base = HELDOUT / "npz" if run["kind"] == "heldout" else K / "ref/npz"
    return base / f"{run['id']}__{run['arm']}.npz"


def key(run: dict) -> str:
    return f"{run['id']}__{run['arm']}"


def summary() -> dict:
    out = {}
    for L in BUCKETS:
        for acc in ("cpu", "gpu_f32"):
            runs = gate_runs(L, acc)
            out[f"L{L}_{acc}"] = {"calls": len(runs), "fixture": sum(r["kind"] == "fixture" for r in runs),
                                 "heldout": sum(r["kind"] == "heldout" for r in runs),
                                 "red": sum(r["kind"] == "red" for r in runs),
                                 "max_tokens": max(r["tokens"] for r in runs)}
    return out


if __name__ == "__main__":
    print(json.dumps(summary(), indent=1))
