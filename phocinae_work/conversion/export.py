"""Step 4: export the fp32 graphs with litert_torch (one file per window L, signature serving_default).

    python export.py --L 512 [--form ids|embeds] [--ln-shift ...] [--tag t]      # main graph
    python export.py --act                                                      # the act head (once)

ids    : exports/phocinae_L{L}_ids_fp32{tag}.tflite     input_ids int32 [1,L], attention_mask f32 [1,L], qtype_onehot f32 [1,3]
embeds : exports/phocinae_L{L}_embeds_fp32{tag}.tflite  inputs_embeds f32 [1,L,384], attention_mask, qtype_onehot (host lookup)
act    : exports/phocinae_act_head_fp32.tflite           pooled_cls f32 [1,384], feats f32 [1,4] -> act_logits f32 [1,2]
Guard: results/torch_parity_L{L}{tag}.json must exist and pass the bar (step 3). After the export the file is scanned
statically (tflite_scan.py) and the row that was traced is run through CompiledModel on the CPU and compared with the
PyTorch graph (max |d token_logits| <= 1e-4 or the export is marked FAIL). Existing files are never overwritten.
"""
import argparse
import importlib.metadata
import json
import resource
import time

import numpy as np
import torch

from common import HF, W, dump_json, sha256_file
from graph import ActHeadGraph, EmbedsGraph, IdsGraph, PhocinaeGraph, embed_rows, load_decision_model, row_inputs
from tflite_scan import scan
from torch_parity import parse_shift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=512)
    ap.add_argument("--form", choices=["ids", "embeds"], default="ids")
    ap.add_argument("--act", action="store_true")
    ap.add_argument("--ln-shift", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-guard", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(4)
    W.joinpath("exports").mkdir(exist_ok=True)
    dm, eng = load_decision_model(HF)
    shift = parse_shift(a.ln_shift)
    tag = a.tag or ("_shift" if shift else "")
    t0 = time.time()
    import litert_torch
    versions = {p: importlib.metadata.version(p) for p in ("torch", "litert-torch", "litert-converter", "ai-edge-litert")}
    if a.act:
        path = W / "exports" / "phocinae_act_head_fp32.tflite"
        assert not path.exists(), f"never overwrite {path}"
        g = ActHeadGraph(dm).eval()
        kw = {"pooled_cls": torch.zeros(1, 384, dtype=torch.float32), "feats": torch.zeros(1, 4, dtype=torch.float32)}
        litert_torch.convert(g, sample_kwargs=kw).export(str(path))
        rec = dict(file=str(path.relative_to(W)), bytes=path.stat().st_size, sha256=sha256_file(path), seconds=round(time.time() - t0, 1),
                   versions=versions, scan=scan(str(path), with_sha=False))
        dump_json(W / "results" / "export_act_head.json", rec)
        print(json.dumps({k: rec[k] for k in ("file", "bytes", "sha256", "seconds")}, indent=1))
        return
    L = a.L
    path = W / "exports" / f"phocinae_L{L}_{a.form}_fp32{tag}.tflite"
    assert not path.exists(), f"never overwrite {path}"
    guard = None
    if not a.no_guard:
        tp = json.loads((W / "results" / f"torch_parity_L{L}{tag}.json").read_text())
        s = tp["summary"]
        ok = (s["finite"] == s["rows"] and s["max_abs_dp"] <= 0.02 and s["mean_abs_dp"] <= 0.002
              and s["argmax_equal_non_near_tie"].split("/")[0] == s["argmax_equal_non_near_tie"].split("/")[1])
        assert ok, f"torch parity at L={L} did not pass: {s}"
        guard = dict(file=f"results/torch_parity_L{L}{tag}.json", summary=s)
    g = PhocinaeGraph(dm, L, host_lookup=(a.form == "embeds"), ln_shift=shift).eval()
    fx = json.loads((W / "fixtures" / "rows_laya.json").read_text())["rows"]
    probe = next(r for r in fx if r["row_len"] <= L)
    ids, mask, onehot = row_inputs(probe["ids"], L, qtype=probe["qtype"])
    if a.form == "ids":
        mod = IdsGraph(g)
        kw = {"input_ids": ids, "attention_mask": mask, "qtype_onehot": onehot}
    else:
        mod = EmbedsGraph(g)
        table = dm.encoder.embeddings.tok_embeddings.weight.detach().float()
        kw = {"inputs_embeds": embed_rows(table, ids).contiguous(), "attention_mask": mask, "qtype_onehot": onehot}
    with torch.no_grad():
        ref = mod(**kw)
    record = dict(L=L, form=a.form, ln_shift=shift, status="RUNNING", file=str(path.relative_to(W)), guard=guard, probe_row=probe["id"],
                  sample_kwargs={k: dict(shape=list(v.shape), dtype=str(v.dtype).replace("torch.", "")) for k, v in kw.items()},
                  convert_call="litert_torch.convert(module, sample_kwargs=kw).export(path)", versions=versions)
    out_json = W / "results" / f"export_L{L}_{a.form}{tag}.json"
    try:
        t1 = time.time()
        litert_torch.convert(mod, sample_kwargs=kw).export(str(path))
        record["convert_seconds"] = round(time.time() - t1, 1)
        record["bytes"] = path.stat().st_size
        record["sha256"] = sha256_file(path)
        record["peak_rss_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30, 2)
        sc = scan(str(path), with_sha=False)
        record["scan"] = sc
        # CPU smoke on the probe row through CompiledModel
        from ai_edge_litert.compiled_model import CompiledModel, CpuOptions, HardwareAccelerator, Options
        m = CompiledModel.from_file(str(path), options=Options(HardwareAccelerator.CPU, cpu_options=CpuOptions(num_threads=4)))
        sigs = m.get_signature_list()
        assert len(sigs) == 1, sigs
        key = next(iter(sigs))
        in_det, out_det = m.get_input_tensor_details(key), m.get_output_tensor_details(key)
        assert sorted(in_det) == sorted(kw) and sorted(out_det) == ["pooled_cls", "token_logits"], (in_det, out_det)
        ins = {n: m.create_input_buffer_by_name(key, n) for n in in_det}
        outs = {n: m.create_output_buffer_by_name(key, n) for n in out_det}
        for n, v in kw.items():
            ins[n].write(np.ascontiguousarray(v.numpy()))
        m.run_by_name(key, ins, outs)
        tl = np.asarray(outs["token_logits"].read(L, np.float32), dtype=np.float32)
        pc = np.asarray(outs["pooled_cls"].read(384, np.float32), dtype=np.float32)
        d_tl = float(np.abs(tl.reshape(-1) - ref["token_logits"].numpy().reshape(-1)).max())
        d_pc = float(np.abs(pc.reshape(-1) - ref["pooled_cls"].numpy().reshape(-1)).max())
        record["smoke"] = dict(signature=key, inputs=list(in_det), outputs=list(out_det), max_abs_d_token_logits=d_tl,
                               max_abs_d_pooled_cls=d_pc, pass_=d_tl <= 1e-4 and d_pc <= 1e-3)
        record["status"] = ("PASS" if record["smoke"]["pass_"] and sc["custom_op_count"] == 0 and sc["rank_gt4_tensor_count"] == 0
                            and sc["int64_tensor_count"] == 0 else "FAIL")
    except Exception as e:  # noqa: BLE001
        import traceback
        record["status"] = "ERROR"
        record["error"] = repr(e)
        (W / "logs" / f"export_L{L}_{a.form}{tag}.traceback.txt").write_text(traceback.format_exc())
    record["seconds"] = round(time.time() - t0, 1)
    dump_json(out_json, record)
    print(json.dumps({k: v for k, v in record.items() if k not in ("scan",)}, indent=1, default=str))
    print("ops:", json.dumps(record.get("scan", {}).get("op_histogram", {}), sort_keys=True))


if __name__ == "__main__":
    main()
