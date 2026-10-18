"""The EmbeddingEngine bundle on the Mac (litert-lm) against the reference: the parity bar + the special-token control.

    python3   # with litert-lm 0.18.0 and numpy scripts/engine_gate.py --bundle exports/engine/<bundle>.litertlm [--backend cpu|gpu]
        [--groups pair,stsb,long128,long256,long512,edge512] [--limit N] [--no-special-tokens] [--tag t]

Every fixture row of the groups (--limit per group) goes through EmbeddingEngine.compute_embedding(text) with
insert_special_tokens on (the default; the bundle's metadata declares <bos> 2 and <eos> 1 = the tokenizer's post-processor)
and normalize on. The vectors are scored with scripts/embed_gate.py vs oracle/embeddings.npz; --no-special-tokens is the
control that must FAIL. Output results/engine_{backend}_parity_{bundle stem}{tag}.json. Never overwrites.
Needs: numpy and litert-lm 0.18.0; metrics.py / embed_gate.py are numpy-only.
"""
import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / "scripts"))
from embed_gate import BAR, score  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--backend", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--groups", default="pair,stsb,long128,long256,long512,edge512")
    ap.add_argument("--limit", type=int, default=0, help="rows per group (0 = all)")
    ap.add_argument("--no-special-tokens", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    import importlib.metadata
    import litert_lm
    from litert_lm.embedding_engine import EmbeddingEngine, EmbeddingOptions
    bundle = Path(a.bundle) if Path(a.bundle).is_absolute() else W / a.bundle
    tag = a.tag + ("_nospecial" if a.no_special_tokens else "") + (f"_lim{a.limit}" if a.limit else "")
    out = W / "results" / f"engine_{a.backend}_parity_{bundle.stem}{tag}.json"
    assert not out.exists(), f"never overwrite {out}"
    fx = json.loads((W / "fixtures" / "texts.json").read_text())
    oracle = json.loads((W / "oracle" / "oracle.json").read_text())
    ref = np.load(W / "oracle" / "embeddings.npz")
    groups = a.groups.split(",")
    rows = []
    for g in groups:
        rs = [r for r in fx["rows"] if r["group"] == g and r["key"] in ref.files]
        rows += rs[: a.limit] if a.limit else rs
    backend = litert_lm.Backend.CPU() if a.backend == "cpu" else litert_lm.Backend.GPU()
    t0 = time.time()
    eng = EmbeddingEngine(str(bundle), backend=backend)
    load_s = time.time() - t0
    opts = EmbeddingOptions(normalize=True, insert_special_tokens=not a.no_special_tokens)
    vec, lat, errors = {}, [], {}
    t1 = time.time()
    for i, r in enumerate(rows):
        t = time.perf_counter()
        try:
            res = eng.compute_embedding(r["text"], opts)
            vec[r["key"]] = np.asarray(list(res.embedding), dtype=np.float32)
        except Exception as e:  # noqa: BLE001
            errors[r["key"]] = repr(e)[:200]
        lat.append((time.perf_counter() - t) * 1e3)
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(rows)} {time.time() - t1:.0f}s", flush=True)
    sc = score(vec, fx, oracle, ref) if vec else None
    per_row = sc.pop("per_row") if sc else None
    by_len = {}
    for r in rows:
        if r["key"] in vec:
            b = 64 if r["row_len"] <= 64 else 128 if r["row_len"] <= 128 else 256 if r["row_len"] <= 256 else 512
            by_len.setdefault(b, []).append(lat[rows.index(r)])
    summary = dict(bundle=str(bundle.relative_to(W)), bytes=bundle.stat().st_size, backend=a.backend, insert_special_tokens=not a.no_special_tokens, normalize=True,
                   load_s=round(load_s, 2), texts=len(rows), errors=len(errors), error_samples=dict(list(errors.items())[:3]),
                   **({k: sc[k] for k in ("rows", "cos", "max_abs", "norm", "pairs", "top1", "metrics", "bar", "bar_core")} if sc else {}),
                   bar_pass=bool(sc and sc["bar_core"] and not errors),
                   latency_ms=dict(median=round(statistics.median(lat), 2), by_window={k: round(statistics.median(v), 2) for k, v in sorted(by_len.items())},
                                   note="contended Mac, informational; the card's Mac ms come from the CompiledModel timing rows"),
                   seconds=round(time.time() - t1, 1),
                   env=dict(python=platform.python_version(), macos=platform.mac_ver()[0], litert_lm=importlib.metadata.version("litert-lm")))
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(dict(summary=summary, rows=per_row), indent=1, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k in ("bundle", "backend", "insert_special_tokens", "load_s", "rows", "errors", "cos", "top1", "pairs", "metrics", "bar_pass", "latency_ms", "seconds")}, indent=1, default=str))


if __name__ == "__main__":
    main()
