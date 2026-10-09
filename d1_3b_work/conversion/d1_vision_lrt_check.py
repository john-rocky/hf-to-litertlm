"""Round 3 acceptance 4-5 / round 6d acceptance 3 (LiteRT side): an exported vision tower or projector file through
the CompiledModel API on every tile of the pictures, against the torch export form and transformers.

    $EXPORT scripts/d1_vision_lrt_check.py --tflite exports/tiny_vision_tower_fp32.tflite --accel cpu
    $EXPORT scripts/d1_vision_lrt_check.py \
        --tflite exports/tiny_vision_tower_fp32.tflite --accel gpu --f32        (Metal float32; without --f32 = default)
    round 6d: ... --tflite exports/real_vision_tower_fp32.tflite --rtag realv --log-prefix r6d_ \
        --table cache/real/tables/vision_position_table.safetensors \
        --extra coco_cats:cache/realv/coco_cats.jpg:cache/realv/proc_coco_cats.npz

The graph kind comes from the signature (pixels / pos / mask -> tower, soft -> projector); --tag (default: the file
name's prefix) names the torch rows to compare with ({cache}/{tag}_torch_rows.npz of d1_vision_torch_check.py, same
--rtag / --root / --extra), which also hold transformers' rows (HF uncut features, HF mm): transformers is not run
here. --table: the position table file (default: the weights' table, read with torch from --source).
Tower: per tile, the host's inputs (picture -> cap_pixels -> preprocess -> tower_inputs) -> `features`; the real rows
vs our torch VisionTower and vs transformers' Siglip2VisionModel on the processor's tensors (uncut): max |diff| and
relative = max |diff| / max |reference| of the tile; then the tile's pad rows of pixels / pos are replaced by noise and
the real rows must not move (bit-equal).
Projector: per tile, the torch rows' unshuffled cells (zero rows after them) -> `mm`; the first k rows vs our torch
Projector and vs transformers' get_image_features (whose input is transformers' own features).
Delegation evidence as in d1_check.py (runtime VERBOSE lines -> the runtime log, `Replacing N out of M` parsed); a
refused compile is recorded with its error text verbatim (status FAIL). Outputs (Out names, never overwritten):
results/{rname}_{accel}_check.json, {cache}/litert_{rname}_{accel}.npz (the real rows of every tile),
logs/{log_prefix}{rname}_{accel}.runtime.log; rname = the file stem with {tag} replaced by {rtag}.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

K = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K / "scripts"))
sys.path.insert(0, str(K / "host"))
import d1_vision as V  # noqa: E402
from d1_check import LINE_KEYS, PARTITIONED, REPLACING, runtime_log_verbose  # noqa: E402
from d1_vision_host_check import bits_equal  # noqa: E402

SIGNATURE = "serving_default"


def load_table(table: str | None, source: str) -> tuple[np.ndarray, dict]:
    if table:
        from safetensors.numpy import load_file

        p = K / table
        return load_file(str(p))["table"], {"from": table, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    import d1_vision_graph as VG

    tower, _, _, _ = VG.load_vision(source)
    side = tower.embeddings.position_embedding_size
    t = tower.embeddings.position_embedding.weight.detach().numpy().reshape(side, side, -1).astype(np.float32)
    return t, {"from": f"weights ({source})"}


def rel(d: float, ref: np.ndarray) -> float:
    m = float(np.abs(ref).max())
    return d / m if m else float("inf")


def main() -> int:
    import d1_vision_graph as VG
    from d1_vision_torch_check import pictures

    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", required=True)
    ap.add_argument("--accel", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--f32", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--source", default="tiny")
    VG.Out.add_args(ap, tag_default=None)
    ap.add_argument("--table", default=None, help="the position table file (K-relative safetensors, key `table`)")
    ap.add_argument("--extra", action="append", default=[], help="name:image:processor_npz, as the torch check had")
    a = ap.parse_args()
    path = Path(a.tflite) if Path(a.tflite).is_absolute() else K / a.tflite
    stem = path.stem
    o = VG.Out.from_args(a, tag=a.tag or stem.split("_")[0])
    rn = o.rname(stem)
    acc = "cpu" if a.accel == "cpu" else ("gpu_f32" if a.f32 else "gpu_default")
    out = o.result(f"{rn}_{acc}_check.json")
    npz_out = o.cache(f"litert_{rn}_{acc}.npz")
    log = o.log(f"{rn}_{acc}.runtime.log")
    for p in (out, npz_out, log):
        assert not p.exists(), f"refusing to overwrite {p}"
    table, table_info = load_table(a.table, a.source)
    rows_path = o.cache(f"{o.tag}_torch_rows.npz")
    rows_npz = np.load(rows_path)
    pics = pictures(a.extra)
    log_f = open(log, "w")
    saved_fd2 = os.dup(2)
    os.dup2(log_f.fileno(), 2)
    doc = {"what": "exported vision graph through CompiledModel vs torch / transformers (round 3; round 6d)",
           "tflite": o.rel(path), "tflite_bytes": path.stat().st_size, "accel": acc, "out": o.as_dict(),
           "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "ai_edge_litert": importlib.metadata.version("ai-edge-litert"),
           "runtime_log": o.rel(log), "torch_rows": o.rel(rows_path), "position_table": table_info,
           "pictures": [p[0] for p in pics]}
    try:
        doc["logger"] = runtime_log_verbose()
        from ai_edge_litert.compiled_model import CompiledModel, CpuOptions, GpuOptions, HardwareAccelerator, Options

        if a.accel == "gpu":
            gopt = GpuOptions(enforce_f32=a.f32)
            opts = Options(hardware_accelerators=HardwareAccelerator.GPU, gpu_options=gopt)
            doc["options"] = {"accelerator": "GPU (Metal)", "gpu_options": gopt._as_flat_kwargs()}
        else:
            opts = Options(hardware_accelerators=HardwareAccelerator.CPU, cpu_options=CpuOptions(num_threads=a.threads))
            doc["options"] = {"accelerator": "CPU", "threads": a.threads}
        t0 = time.perf_counter()
        model = CompiledModel.from_file(str(path), options=opts)
        doc["compile_seconds"] = round(time.perf_counter() - t0, 3)
        try:
            doc["is_fully_accelerated"] = bool(model.is_fully_accelerated())
        except Exception as e:
            doc["is_fully_accelerated"] = f"unavailable: {type(e).__name__}: {e}"
        in_det, out_det = model.get_input_tensor_details(SIGNATURE), model.get_output_tensor_details(SIGNATURE)
        kind = "tower" if "pixels" in in_det else "projector"
        out_name = "features" if kind == "tower" else "mm"
        oshape = [int(x) for x in out_det[out_name]["shape"]]
        doc.update(kind=kind, inputs={n: str(d["shape"]) for n, d in in_det.items()}, output=oshape)
        ins = {n: model.create_input_buffer_by_name(SIGNATURE, n) for n in in_det}
        outs = {out_name: model.create_output_buffer_by_name(SIGNATURE, out_name)}
        size = int(np.prod(oshape))

        def run(feed):
            for n, v in feed.items():
                ins[n].write(np.ascontiguousarray(v, dtype=np.float32))
            model.run_by_name(SIGNATURE, ins, outs)
            return np.asarray(outs[out_name].read(size, np.float32), dtype=np.float32).reshape(oshape)[0]

        rng = np.random.default_rng(4)
        res, store, run_ms = [], {}, []
        from PIL import Image

        t_loop = time.perf_counter()
        for name, img_path, _ in pics:
            pic = V.preprocess(V.cap_pixels(Image.open(img_path)))
            for ti, tile in enumerate(pic.tiles):
                key = f"{name}__{ti}"
                hh, ww = tile.grid
                n = hh * ww
                k = (hh // 2) * (ww // 2)
                if kind == "tower":
                    feed = V.tower_inputs(tile, table)
                    t = time.perf_counter()
                    y = run(feed)
                    run_ms.append((time.perf_counter() - t) * 1000)
                    real = y[:n]
                    mine, hf = rows_npz[f"features__{key}"], rows_npz[f"hf__{key}"]
                    d_t = float(np.abs(real.astype(np.float64) - mine).max())
                    d_h = float(np.abs(real.astype(np.float64) - hf).max())
                    row = {"case": name, "tile": ti, "grid": [hh, ww], "vs_torch": d_t, "vs_torch_rel": rel(d_t, mine),
                           "vs_hf": d_h, "vs_hf_rel": rel(d_h, hf), "max_abs": float(np.abs(real).max()),
                           "nonfinite": int((~np.isfinite(y)).sum()), "nonfinite_real": int((~np.isfinite(real)).sum())}
                    if n < V.SETTINGS.max_num_patches:
                        noisy = {kk: v.copy() for kk, v in feed.items()}
                        noisy["pixels"][0, n:] = (rng.standard_normal(noisy["pixels"][0, n:].shape) * 3).astype(np.float32)
                        noisy["pos"][0, n:] = (rng.standard_normal(noisy["pos"][0, n:].shape) * 3).astype(np.float32)
                        y2 = run(noisy)
                        row["pad_noise_bits_equal"] = bits_equal(y2[:n].copy(), real.copy())
                        row["pad_noise_max_abs"] = float(np.abs(y2[:n].astype(np.float64) - real).max())
                    store[f"features__{key}"] = np.ascontiguousarray(real)
                else:
                    t = time.perf_counter()
                    y = run({"soft": V.projector_input(rows_npz[f"soft__{key}"])})
                    run_ms.append((time.perf_counter() - t) * 1000)
                    mine, hf = rows_npz[f"mm__{key}"], rows_npz[f"hfmm__{key}"]
                    d_t = float(np.abs(y[:k].astype(np.float64) - mine).max())
                    d_h = float(np.abs(y[:k].astype(np.float64) - hf).max())
                    row = {"case": name, "tile": ti, "tokens": k, "vs_torch": d_t, "vs_torch_rel": rel(d_t, mine),
                           "vs_hf": d_h, "vs_hf_rel": rel(d_h, hf), "max_abs": float(np.abs(y[:k]).max()),
                           "nonfinite": int((~np.isfinite(y)).sum()), "nonfinite_real": int((~np.isfinite(y[:k])).sum())}
                    store[f"mm__{key}"] = np.ascontiguousarray(y[:k])
                res.append(row)
        doc["loop_seconds"] = round(time.perf_counter() - t_loop, 1)
        for b in list(ins.values()) + list(outs.values()):
            try:
                b.destroy()
            except Exception:
                pass
        model.close()
        np.savez(npz_out, **store)
        pads = [r["pad_noise_bits_equal"] for r in res if "pad_noise_bits_equal" in r]
        worst = max(res, key=lambda r: r["vs_hf_rel"])
        doc.update(rows=res, tiles=len(res), run_ms_median=round(float(np.median(run_ms)), 3),
                   max_abs_vs_torch=max(r["vs_torch"] for r in res), max_abs_vs_hf=max(r["vs_hf"] for r in res),
                   max_rel_vs_torch=max(r["vs_torch_rel"] for r in res), max_rel_vs_hf=max(r["vs_hf_rel"] for r in res),
                   worst_tile_vs_hf={kk: worst[kk] for kk in ("case", "tile", "vs_hf", "vs_hf_rel", "max_abs")},
                   max_abs_output=max(r["max_abs"] for r in res),
                   nonfinite_total=sum(r["nonfinite"] for r in res),
                   nonfinite_real_total=sum(r["nonfinite_real"] for r in res), pad_tiles=len(pads),
                   pad_bits_equal=sum(pads), pad_noise_max_abs=max([r.get("pad_noise_max_abs", 0.0) for r in res]),
                   litert_npz=o.rel(npz_out), timing_note="contended Mac; informational only")
        doc["status"] = "OK"
    except BaseException as e:
        doc["status"] = "FAIL"
        doc["error"] = f"{type(e).__name__}: {e}"
        import traceback

        traceback.print_exc()
    finally:
        sys.stderr.flush()
        os.dup2(saved_fd2, 2)
        log_f.close()
    lines = log.read_text(errors="replace").splitlines()
    doc["runtime_log_lines"] = len(lines)
    doc["runtime_log_key_lines"] = [ln for ln in lines if LINE_KEYS.search(ln)][:80]
    doc["delegation"] = {
        "replacing": [{"delegated": int(m[0]), "total": int(m[1]), "delegate": m[2], "partitions": int(m[3])}
                      for m in REPLACING.findall("\n".join(lines))],
        "partitioned": [{"subgraph": int(m[0]), "selected": int(m[1]), "total": int(m[2]), "partitions": int(m[3])}
                        for m in PARTITIONED.findall("\n".join(lines))]}
    out.write_text(json.dumps(doc, indent=1) + "\n")
    brief = {k: doc.get(k) for k in ("status", "error", "kind", "accel", "tiles", "compile_seconds", "is_fully_accelerated",
                                     "max_abs_vs_torch", "max_rel_vs_torch", "max_abs_vs_hf", "max_rel_vs_hf",
                                     "worst_tile_vs_hf", "max_abs_output", "nonfinite_total", "pad_tiles", "pad_bits_equal",
                                     "pad_noise_max_abs", "run_ms_median", "loop_seconds", "delegation")}
    print(json.dumps(brief, indent=1))
    print("\n".join(doc["runtime_log_key_lines"][:12]))
    return 0 if doc["status"] == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
