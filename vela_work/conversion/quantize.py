"""Step 5: storage variants of an fp32 graph with ai-edge-quantizer (weights only; activations stay fp32).

    P=python3   # a Python 3.12 with litert-torch 0.10.0, litert-converter, ai-edge-litert 2.2.0, ai-edge-quantizer
    $P scripts/quantize.py --src exports/vela_L512_ids_fp32_shift.tflite --variant v2_fp16fc_i8emb

variants (recipes = phocinae_work/scripts/quantize.py build_recipe, imported by path):
  v2_fp16fc_i8emb   FLOAT_CASTING 16-bit FULLY_CONNECTED + dynamic int8 EMBEDDING_LOOKUP (channelwise): the one-file GPU / CPU form
  v3_fp16fc_fp16emb FLOAT_CASTING 16-bit on FULLY_CONNECTED and EMBEDDING_LOOKUP (CPU only: the GPU delegate refuses an fp16 table)
  v1_wi8fc_i8emb    dynamic int8 FULLY_CONNECTED + dynamic int8 EMBEDDING_LOOKUP (granite-embedding's ship recipe)
  v5_i8emb          dynamic int8 EMBEDDING_LOOKUP only (fp32 FC: isolates the table's share of the residual)
  wfp16             FLOAT_CASTING 16-bit FULLY_CONNECTED only (the embeds / engine forms; the table ships separately)
  i8emb_lookup      dynamic int8 EMBEDDING_LOOKUP (channelwise) on the one-op embedder graph (= v5 on vela_lookup_fp32)
Output: exports/<src stem with fp32 -> variant>.tflite + results/quant_<stem>.json. Never overwrites.
"""
import argparse
import importlib.metadata
import json
import sys
import time
from pathlib import Path

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / "scripts"))
import peer  # noqa: E402
pq = peer.load_phocinae_quantize()   # phocinae's build_recipe / weight_dtypes: recipe code
scan, sha256_file = peer.tflite_scan.scan, peer.tflite_scan.sha256_file


def build_recipe(variant):
    if variant == "i8emb_lookup":
        return pq.build_recipe("v5_i8emb")
    if variant == "wfp16_lookup":          # the embedder with an fp16 table (the int8 table fails the bar: REPRODUCE.md)
        return pq.build_recipe("v3_fp16fc_fp16emb")
    return pq.build_recipe(variant)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--variant", choices=pq.VARIANTS + ["i8emb_lookup", "wfp16_lookup"], required=True)
    a = ap.parse_args()
    src = Path(a.src) if Path(a.src).is_absolute() else W / a.src
    assert src.exists() and "fp32" in src.name, src
    dst = src.with_name(src.name.replace("fp32", a.variant))
    assert not dst.exists(), f"never overwrite {dst}"
    out = W / "results" / f"quant_{dst.stem}.json"
    assert not out.exists(), out
    t0 = time.time()
    recipe, need_cal = build_recipe(a.variant)
    from ai_edge_quantizer import quantizer
    qt = quantizer.Quantizer(str(src), recipe)
    if qt.need_calibration:
        raise SystemExit(f"STOP: recipe {a.variant} needs calibration")
    qt.quantize().export_model(str(dst))
    rec = dict(src=str(src.relative_to(W)), dst=str(dst.relative_to(W)), variant=a.variant, recipe=recipe, need_calibration=bool(need_cal),
               seconds=round(time.time() - t0, 1), src_bytes=src.stat().st_size, bytes=dst.stat().st_size, sha256=sha256_file(dst),
               weight_dtypes=pq.weight_dtypes(dst), scan=scan(str(dst), with_sha=False),
               versions={p: importlib.metadata.version(p) for p in ("ai-edge-litert", "litert-torch")},
               quantizer=str(sorted(Path(quantizer.__file__).parent.parent.glob("ai_edge_quantizer*dist-info"))[-1].name))
    out.write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps({k: rec[k] for k in ("dst", "bytes", "sha256", "seconds", "weight_dtypes")}, indent=1, default=str))
    print("ops:", json.dumps(rec["scan"]["op_histogram"], sort_keys=True))


if __name__ == "__main__":
    main()
