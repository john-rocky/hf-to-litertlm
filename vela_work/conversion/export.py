"""Step 4: export the fp32 graphs with litert_torch (one file per window L and form).

    P=python3   # a Python 3.12 with litert-torch 0.10.0, litert-converter, ai-edge-litert 2.2.0, ai-edge-quantizer
    $P scripts/export.py --L 512 --form ids|embeds|engine [--ln-shift ...] [--tag t]
    $P scripts/export.py --lookup                                   # the EmbeddingEngine embedder (token -> row), once

ids    : exports/vela_L{L}_ids_fp32{tag}.tflite      input_ids int32 [1,L], attention_mask f32 [1,L] -> embedding f32 [1,768]
embeds : exports/vela_L{L}_embeds_fp32{tag}.tflite   inputs_embeds f32 [1,L,768], attention_mask -> embedding (host lookup)
engine : exports/vela_L{L}_engine_fp32{tag}.tflite   embeddings f32 [1,L,768], input_mask f32 [1,L] -> encodings [1,768],
                                                     signature encoder_{L} (LiteRT-LM EmbeddingEngine TF_LITE_TEXT_ENCODER)
lookup : exports/vela_lookup_fp32.tflite             token int32 [1] -> embedding f32 [1,1,768], signature embedder
Guard: results/torch_parity_L{L}{tag}.json must exist with cos_min >= 0.9999 (the graph before conversion). After the export
the file is scanned statically (phocinae_work/scripts/tflite_scan.py, imported by path) and the traced row is run through
CompiledModel on the CPU against the PyTorch graph (max |d| <= 1e-4 or the export is marked FAIL). Never overwrites.
"""
import argparse
import importlib.metadata
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / "scripts"))
from peer import tflite_scan  # noqa: E402  (phocinae_work/scripts/tflite_scan.py by file path; see peer.py)
scan, sha256_file = tflite_scan.scan, tflite_scan.sha256_file

from graph import EmbedsGraph, EngineEncoderGraph, IdsGraph, LookupGraph, VelaGraph, embed_rows, load_body, load_rope, parse_shift, row_inputs  # noqa: E402


def dump_json(path, obj):
    path = Path(path)
    assert not path.exists(), f"never overwrite {path}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=512)
    ap.add_argument("--form", choices=["ids", "embeds", "engine"], default="ids")
    ap.add_argument("--lookup", action="store_true")
    ap.add_argument("--ln-shift", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-guard", action="store_true")
    ap.add_argument("--pool", choices=["where", "arith"], default="where", help="arith = MUL / SUM / MAXIMUM / RSQRT pooling (no SELECT_V2 / GREATER / POW; the GPU delegate's answer), tag _arith")
    a = ap.parse_args()
    torch.set_num_threads(4)
    (W / "exports").mkdir(exist_ok=True)
    body = load_body(W / "hf")
    shift = parse_shift(a.ln_shift)
    tag = a.tag or (("_shift" if shift else "") + ("_arith" if a.pool == "arith" else ""))
    t0 = time.time()
    import litert_torch
    versions = {p: importlib.metadata.version(p) for p in ("torch", "transformers", "litert-torch", "litert-converter", "ai-edge-litert")}
    if a.lookup:
        path = W / "exports" / "vela_lookup_fp32.tflite"
        assert not path.exists(), f"never overwrite {path}"
        mod = LookupGraph(body).eval()
        kw = {"token": torch.tensor([2], dtype=torch.int32)}
        litert_torch.signature("embedder", mod, sample_kwargs=kw).convert().export(str(path))
        rec = dict(file=str(path.relative_to(W)), bytes=path.stat().st_size, sha256=sha256_file(path), seconds=round(time.time() - t0, 1),
                   versions=versions, scan=scan(str(path), with_sha=False))
        dump_json(W / "results" / "export_lookup.json", rec)
        print(json.dumps({k: rec[k] for k in ("file", "bytes", "sha256", "seconds")}, indent=1))
        return
    L = a.L
    path = W / "exports" / f"vela_L{L}_{a.form}_fp32{tag}.tflite"
    assert not path.exists(), f"never overwrite {path}"
    guard = None
    if not a.no_guard:
        tp = json.loads((W / "results" / f"torch_parity_L{L}{tag}.json").read_text())["summary"]
        assert tp["finite"] == tp["rows"] and tp["cos_min"] >= 0.9999 and tp["embeds_bit_equal"], f"torch parity at L={L}{tag} did not pass: {tp}"
        guard = dict(file=f"results/torch_parity_L{L}{tag}.json", summary=tp)
    g = VelaGraph(body, L, load_rope(W / "oracle" / "rope_L512.npz"), ln_shift=shift, pool_mode=a.pool).eval()
    fx = json.loads((W / "fixtures" / "texts.json").read_text())["rows"]
    probe = max((r for r in fx if r["row_len"] <= L), key=lambda r: r["row_len"])        # the longest fitting row is traced
    ids, mask = row_inputs(probe["ids"], L)
    table = body.embeddings.tok_embeddings.weight.detach().float()
    if a.form == "ids":
        mod = IdsGraph(g)
        kw = {"input_ids": ids, "attention_mask": mask}
        sig = "serving_default"
    elif a.form == "embeds":
        mod = EmbedsGraph(g)
        kw = {"inputs_embeds": embed_rows(table, ids).contiguous(), "attention_mask": mask}
        sig = "serving_default"
    else:
        mod = EngineEncoderGraph(g)
        kw = {"embeddings": embed_rows(table, ids).contiguous(), "input_mask": mask}
        sig = f"encoder_{L}"
    with torch.no_grad():
        ref = list(mod(**kw).values())[0]
    record = dict(L=L, form=a.form, ln_shift=shift, pool=a.pool, status="RUNNING", file=str(path.relative_to(W)), guard=guard, probe_row=probe["key"], probe_len=probe["row_len"],
                  sample_kwargs={k: dict(shape=list(v.shape), dtype=str(v.dtype).replace("torch.", "")) for k, v in kw.items()}, signature=sig, versions=versions)
    out_json = W / "results" / f"export_L{L}_{a.form}{tag}.json"
    try:
        t1 = time.time()
        if sig == "serving_default":
            litert_torch.convert(mod, sample_kwargs=kw).export(str(path))
        else:
            litert_torch.signature(sig, mod, sample_kwargs=kw).convert().export(str(path))
        record["convert_seconds"] = round(time.time() - t1, 1)
        record["bytes"] = path.stat().st_size
        record["sha256"] = sha256_file(path)
        record["peak_rss_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30, 2)
        sc = scan(str(path), with_sha=False)
        record["scan"] = sc
        from ai_edge_litert.compiled_model import CompiledModel, CpuOptions, HardwareAccelerator, Options
        m = CompiledModel.from_file(str(path), options=Options(HardwareAccelerator.CPU, cpu_options=CpuOptions(num_threads=4)))
        sigs = m.get_signature_list()
        assert list(sigs) == [sig], sigs
        in_det, out_det = m.get_input_tensor_details(sig), m.get_output_tensor_details(sig)
        assert sorted(in_det) == sorted(kw), (in_det, kw.keys())
        ins = {n: m.create_input_buffer_by_name(sig, n) for n in in_det}
        outs = {n: m.create_output_buffer_by_name(sig, n) for n in out_det}
        for n, v in kw.items():
            ins[n].write(np.ascontiguousarray(v.numpy()))
        m.run_by_name(sig, ins, outs)
        oname = next(iter(out_det))
        got = np.asarray(outs[oname].read(768, np.float32), dtype=np.float32).reshape(768)
        d = float(np.abs(got - ref.numpy().reshape(768)).max())
        record["smoke"] = dict(signature=sig, inputs=list(in_det), outputs=list(out_det), max_abs_d=d, cos=float(got.astype(np.float64) @ ref.numpy().reshape(768).astype(np.float64)),
                               pass_=d <= 1e-4 and bool(np.isfinite(got).all()))
        record["status"] = ("PASS" if record["smoke"]["pass_"] and sc["custom_op_count"] == 0 and sc["rank_gt4_tensor_count"] == 0 and sc["int64_tensor_count"] == 0 else "FAIL")
    except Exception as e:  # noqa: BLE001
        import traceback
        record["status"] = "ERROR"
        record["error"] = repr(e)
        (W / "logs").mkdir(exist_ok=True)
        (W / "logs" / f"export_L{L}_{a.form}{tag}.traceback.txt").write_text(traceback.format_exc())
    record["seconds"] = round(time.time() - t0, 1)
    dump_json(out_json, record)
    print(json.dumps({k: v for k, v in record.items() if k not in ("scan",)}, indent=1, default=str))
    print("ops:", json.dumps(record.get("scan", {}).get("op_histogram", {}), sort_keys=True))


if __name__ == "__main__":
    main()
