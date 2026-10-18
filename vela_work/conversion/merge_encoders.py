"""The multi-signature encoder of the EmbeddingEngine bundle: encoder_64 / 128 / 256 / 512 in ONE .tflite (the runtime picks
the smallest signature >= the token count), traced in one litert_torch conversion so the weights are shared.

    P=python3   # a Python 3.12 with litert-torch 0.10.0, litert-converter, ai-edge-litert 2.2.0, ai-edge-quantizer
    $P scripts/merge_encoders.py --Ls 64,128,256,512 --ln-shift "$(cat results/ln_shift_set.txt)" --out exports/engine/vela_encoder_fp32_shift.tflite
Then scripts/quantize.py --variant wfp16 on it (the FC weights fp16; the table lives in the embedder file).
Guard: results/torch_parity_L{L}_shift.json for every L. The traced rows are the longest fitting fixture rows. Never overwrites.
"""
import argparse
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / "scripts"))
from peer import tflite_scan  # noqa: E402
scan, sha256_file = tflite_scan.scan, tflite_scan.sha256_file

from graph import EngineEncoderGraph, VelaGraph, embed_rows, load_body, load_rope, parse_shift, row_inputs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ls", default="64,128,256,512")
    ap.add_argument("--ln-shift", default="")
    ap.add_argument("--out", default="exports/engine/vela_encoder_fp32_shift.tflite")
    ap.add_argument("--pool", choices=["where", "arith"], default="where", help="graph.py pooling form (arith: the GPU delegate's answer; the guard files carry _arith)")
    a = ap.parse_args()
    torch.set_num_threads(4)
    Ls = [int(x) for x in a.Ls.split(",")]
    out = W / a.out
    assert not out.exists(), f"never overwrite {out}"
    out.parent.mkdir(parents=True, exist_ok=True)
    shift = parse_shift(a.ln_shift)
    tag = ("_shift" if shift else "") + ("_arith" if a.pool == "arith" else "")
    for L in Ls:
        tp = json.loads((W / "results" / f"torch_parity_L{L}{tag}.json").read_text())["summary"]
        assert tp["finite"] == tp["rows"] and tp["cos_min"] >= 0.9999 and tp["embeds_bit_equal"], (L, tp)
    body = load_body(W / "hf")
    rope = load_rope(W / "oracle" / "rope_L512.npz")
    table = body.embeddings.tok_embeddings.weight.detach().float()
    fx = json.loads((W / "fixtures" / "texts.json").read_text())["rows"]
    import litert_torch
    t0 = time.time()
    conv, refs, mods = None, {}, {}
    for L in sorted(Ls, reverse=True):
        g = VelaGraph(body, L, rope, ln_shift=shift, pool_mode=a.pool).eval()
        mod = EngineEncoderGraph(g).eval()
        probe = max((r for r in fx if r["row_len"] <= L), key=lambda r: r["row_len"])
        ids, mask = row_inputs(probe["ids"], L)
        kw = {"embeddings": embed_rows(table, ids).contiguous(), "input_mask": mask}
        with torch.no_grad():
            refs[L] = (kw, list(mod(**kw).values())[0].numpy().reshape(768))
        mods[L] = mod
        conv = litert_torch.signature(f"encoder_{L}", mod, sample_kwargs=kw) if conv is None else conv.signature(f"encoder_{L}", mod, sample_kwargs=kw)
    conv.convert().export(str(out))
    rec = dict(file=str(out.relative_to(W)), Ls=Ls, ln_shift=shift, convert_seconds=round(time.time() - t0, 1), bytes=out.stat().st_size, sha256=sha256_file(out),
               versions={p: importlib.metadata.version(p) for p in ("torch", "transformers", "litert-torch", "litert-converter", "ai-edge-litert")})
    sc = scan(str(out), with_sha=False)
    rec["scan"] = sc
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=str(out), num_threads=4)
    smoke = {}
    for L in Ls:
        kw, ref = refs[L]
        r = it.get_signature_runner(f"encoder_{L}")(embeddings=kw["embeddings"].numpy(), input_mask=kw["input_mask"].numpy())
        got = list(r.values())[0].reshape(768)
        smoke[f"encoder_{L}"] = dict(max_abs_d=float(np.abs(got - ref).max()), cos=float(got.astype(np.float64) @ ref.astype(np.float64)), finite=bool(np.isfinite(got).all()))
    rec["smoke"] = smoke
    rec["status"] = "PASS" if all(s["max_abs_d"] <= 1e-4 and s["finite"] for s in smoke.values()) and sc["int64_tensor_count"] == 0 and sc["custom_op_count"] == 0 else "FAIL"
    (W / "results" / f"export_encoder_multi{tag}.json").write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps({k: v for k, v in rec.items() if k != "scan"}, indent=1, default=str))
    print("ops:", json.dumps(sc["op_histogram"], sort_keys=True))


if __name__ == "__main__":
    main()
