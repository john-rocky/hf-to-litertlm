"""Where does the int8 (channelwise = one scale per vocabulary row) table lose the bar? Simulate the symmetric int8 rounding of
every row of the checkpoint's table, rank the rows by relative error, and list the tokens of the fixture rows a LiteRT gate
file reports below the bar (--gate-rows <results/litert_*_rows_*.json>). Writes results/table_int8_rowerr.json (never overwrites).
    P=python3   # a Python 3.12 with litert-torch 0.10.0, litert-converter, ai-edge-litert 2.2.0, ai-edge-quantizer; $P scripts/table_int8_analysis.py --gate-rows results/litert_cpu_rows_L64_ids_v2_fp16fc_i8emb_shift.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open
from tokenizers import Tokenizer

W = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-rows", required=True)
    ap.add_argument("--bar", type=float, default=0.999)
    ap.add_argument("--out", default="results/table_int8_rowerr.json")
    a = ap.parse_args()
    out = W / a.out
    assert not out.exists(), f"never overwrite {out}"
    with safe_open(str(W / "hf" / "model.safetensors"), "np") as f:
        t = f.get_tensor("embeddings.tok_embeddings.weight").astype(np.float32)
    absmax = np.abs(t).max(1)
    scale = absmax / 127.0
    q = np.clip(np.rint(t / scale[:, None]), -127, 127) * scale[:, None]
    err = np.linalg.norm(q - t, axis=1) / np.maximum(np.linalg.norm(t, axis=1), 1e-12)
    # fp16 rounding of the same rows, for the comparison column
    e16 = np.linalg.norm(t.astype(np.float16).astype(np.float32) - t, axis=1) / np.maximum(np.linalg.norm(t, axis=1), 1e-12)
    peak = absmax / np.maximum(np.sqrt((t * t).mean(1)), 1e-12)   # outlier ratio: max |x| over the row's rms
    tok = Tokenizer.from_file(str(W / "hf" / "tokenizer.json"))
    order = np.argsort(-err)
    worst = [dict(id=int(i), token=tok.id_to_token(int(i)), rel_err_int8=float(err[i]), rel_err_fp16=float(e16[i]), peak_over_rms=float(peak[i]),
                  absmax=float(absmax[i]), norm=float(np.linalg.norm(t[i]))) for i in order[:30]]
    fx = json.loads((W / "fixtures" / "texts.json").read_text())
    by_key = {r["key"]: r for r in fx["rows"]}
    gate = json.loads((W / a.gate_rows).read_text())
    rows = gate["rows"] if isinstance(gate, dict) else gate
    low = sorted([r for r in rows if r["cos"] < a.bar], key=lambda r: r["cos"])
    detail = []
    for r in low:
        ids = by_key[r["key"]]["ids"]
        toks = [dict(id=int(i), token=tok.id_to_token(int(i)), rel_err_int8=float(err[i]), peak_over_rms=float(peak[i])) for i in ids]
        detail.append(dict(key=r["key"], cos=r["cos"], row_len=len(ids), text=by_key[r["key"]]["text"][:200],
                           worst_token=max(toks, key=lambda d: d["rel_err_int8"]), tokens=toks))
    summary = dict(vocab=int(t.shape[0]), dim=int(t.shape[1]), rel_err_int8=dict(mean=float(err.mean()), p50=float(np.median(err)), p99=float(np.percentile(err, 99)),
                   p999=float(np.percentile(err, 99.9)), max=float(err.max())), rel_err_fp16=dict(mean=float(e16.mean()), max=float(e16.max())),
                   rows_over_5pct=int((err > 0.05).sum()), rows_over_10pct=int((err > 0.10).sum()), rows_over_20pct=int((err > 0.20).sum()),
                   gate_rows_below_bar=len(low), bar=a.bar, gate_file=a.gate_rows)
    doc = dict(summary=summary, worst_table_rows=worst, failing_fixture_rows=detail)
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1))
    for d in detail[:8]:
        print(f"{d['key']} cos {d['cos']:.5f} worst token {d['worst_token']}  text: {d['text'][:90]}")
    print("worst table rows:", [(w["id"], w["token"], round(w["rel_err_int8"], 3)) for w in worst[:10]])


if __name__ == "__main__":
    main()
