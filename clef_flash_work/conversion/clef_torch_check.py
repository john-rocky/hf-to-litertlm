#!/usr/bin/env python3
"""Round 1 step 7: the row graph in PyTorch fp32 (lt094dev: torch 2.13.0, transformers 5.14.1) on reference text runs,
before any export. The hidden states go to the author's head in venv-ref (scripts/clef_score.py) for the parity line.

    ~/venvs/lt094dev/bin/python scripts/clef_torch_check.py [--L 1024] [--threads 8]
        -> cache/r1/torch/<id>__text.npy (hidden at the real positions, float32 [T, 4096]),
           cache/r1/stock/own_t01__text.npy (the environment row), results/torch_check_L1024.json (refuses to overwrite)

Runs: every own_text record (13, the Core AI round 1 maximum 967 tokens among them; the longest is
recorded), the first 3 own_json records, the first 3 SemIf records and the 3 red arms = 22 text runs, each through
ClefRowPrefill at L (the export's form: right pads, mask, positions 0..L-1) with the
ids the reference used (ref/records_ref.json, ref/records_ref_red.json); embeds / cos / sin from the NumPy host
(scripts/clef_host.py). Before the form is applied, the same weights in the stock classes of transformers 5.14.1 (sdpa,
no patch, no padding, its own positions) run own_t01 once: the environment row (torch 2.13 + transformers 5.14.1
against the reference's torch 2.9.0 + transformers 5.17.0). Per run: the hidden max |diff| against the reference's
last_hidden (ref/npz/<id>__text.npz) at the real positions, non-finite counts, head-interleave calls (2 per
GatedDeltaNet = 48), seconds."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

K = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K / "scripts"))
import clef_host as H  # noqa: E402

N_JSON, N_SEMIF = 3, 3


def select_runs(ref_rows, red_rows):
    own_text = [r for r in ref_rows if r["source"] == "own_text" and r["arm"] == "text"]
    own_json = [r for r in ref_rows if r["source"] == "own_json" and r["arm"] == "text"][:N_JSON]
    semif = [r for r in ref_rows if r["source"] == "semif_authored144" and r["arm"] == "text"][:N_SEMIF]
    return own_text + own_json + semif + list(red_rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    L = a.L
    out = K / f"results/torch_check_L{L}.json"
    cache = K / "cache/r1/torch"
    assert not out.exists(), f"refusing to overwrite {out}"
    cache.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    t_wall = time.time()
    ref = json.loads((K / "ref/records_ref.json").read_text())
    red = json.loads((K / "ref/records_ref_red.json").read_text())
    assert ref["complete"] and red["complete"], "the reference is not complete"
    runs = select_runs(ref["rows"], red["rows"])
    assert all(r["tokens"] <= L for r in runs), [(r["id"], r["tokens"]) for r in runs if r["tokens"] > L]
    import clef_graph as G
    table = H.EmbedTable(G.SNAP)
    model, info = G.load_text_model()
    doc = {"step": f"round 1 step 7: ClefRowPrefill in torch fp32 at L={L}", "L": L, "form": G.FORM, "load_info": info,
           "threads": torch.get_num_threads(), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "versions": {"torch": torch.__version__, "transformers": __import__("transformers").__version__,
                        "numpy": np.__version__},
           "reference": {"records": "ref/records_ref.json", "red": "ref/records_ref_red.json",
                         "versions": ref.get("versions"), "torch_num_threads": ref.get("torch_num_threads")},
           "host": {"embed_table": table.info, "inv_freq_first": float(H.inv_freq()[0]), "rope": "clef_host.mrope_tables"}}

    def ref_hidden(r):
        z = np.load(K / f"ref/npz/{r['id']}__{r['arm']}.npz")
        assert list(z["input_ids"]) == r["ids"]
        return z["last_hidden"]

    # environment row: the stock classes on the same weights, before the form changes any class
    r0 = next(r for r in runs if r["id"] == "own_t01")
    stock = G.stock_twin(model)
    emb = torch.from_numpy(table.rows(r0["ids"]))[None]
    t0 = time.perf_counter()
    hs = stock(inputs_embeds=emb, use_cache=False).last_hidden_state[0].numpy()
    (K / "cache/r1/stock").mkdir(parents=True, exist_ok=True)
    np.save(K / "cache/r1/stock/own_t01__text.npy", hs.astype(np.float32))
    rh = ref_hidden(r0)
    doc["stock_row"] = {"id": "own_t01", "arm": "text", "tokens": r0["tokens"], "seconds": round(time.perf_counter() - t0, 1),
                        "hidden_file": "cache/r1/stock/own_t01__text.npy",
                        "hidden_max_abs_vs_reference": float(np.abs(hs - rh).max()),
                        "hidden_bit_equal_to_reference": bool(np.array_equal(hs, rh)),
                        "attn_implementation": stock.config._attn_implementation,
                        "what": "stock transformers 5.14.1 classes (no patch), sdpa, unpadded, positions from the model"}
    print(json.dumps(doc["stock_row"]), flush=True)
    del stock
    doc["form_apply"] = G.apply_form(model)
    counter = G.count_interleave()
    g = G.graph(model, L)
    rows = []
    for i, r in enumerate(runs):
        T = r["tokens"]
        embeds, valid, cos, sin = H.row_inputs(r["ids"], L, table)
        c0 = counter["calls"]
        t0 = time.perf_counter()
        h = g(torch.from_numpy(embeds), torch.from_numpy(valid), torch.from_numpy(cos), torch.from_numpy(sin))["hidden"]
        sec = time.perf_counter() - t0
        h = h[0].numpy()
        hr = h[:T].astype(np.float32)
        name = f"{r['id']}__{r['arm']}.npy"
        np.save(cache / name, hr)
        rh = ref_hidden(r)
        row = {"id": r["id"], "arm": r["arm"], "source": r["source"], "tokens": T, "seconds": round(sec, 2),
               "hidden_file": f"cache/r1/torch/{name}", "interleave_calls": counter["calls"] - c0,
               "nonfinite_real": int((~np.isfinite(hr)).sum()), "nonfinite_all": int((~np.isfinite(h)).sum()),
               "hidden_max_abs_vs_reference": float(np.abs(hr - rh).max()),
               "hidden_max_rel_vs_reference": float(np.abs(hr - rh).max() / np.abs(rh).max()),
               "last_position_max_abs": float(np.abs(hr[-1] - rh[-1]).max())}
        rows.append(row)
        print(f"{i + 1}/{len(runs)} {r['id']} T={T} {sec:.1f}s max|dh| {row['hidden_max_abs_vs_reference']:.3e} "
              f"interleave {row['interleave_calls']}", flush=True)
    doc.update(runs=len(rows), rows=rows, interleave_ratios=counter["ratios"], interleave_shapes=counter["shapes"],
               interleave_calls_per_forward=sorted({r["interleave_calls"] for r in rows}),
               all_finite=all(r["nonfinite_real"] == 0 for r in rows),
               hidden_max_abs_vs_reference=max(r["hidden_max_abs_vs_reference"] for r in rows),
               longest=max(rows, key=lambda r: r["tokens"])["id"], seconds_wall=round(time.time() - t_wall, 1),
               contract=G.contract(L))
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: doc[k] for k in ("runs", "all_finite", "hidden_max_abs_vs_reference", "interleave_calls_per_forward",
                                          "longest", "seconds_wall")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
