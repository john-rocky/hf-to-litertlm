"""Writes the eight DiT graphs of iris-3b Turbo (float32 weights) and the DiT's host tables.

    python export_dit.py --out OUT [--iris DIR] [--upstream DIR]

The upstream IrisDiT (github.com/speridlabs/iris-3b) is re-authored in dit_modules.py as chunk modules whose every op
the LiteRT GPU delegate takes: the text adapter, three dual-stream chunks (3 + 3 + 2 blocks), three single-stream
chunks (6 + 6 + 4 blocks) and the pixel stage, for 768 x 768 images (2,304 patch tokens, 300 text tokens). Each graph
is traced with zeros of its input shapes (TRACE_SHAPES): the export keeps the shapes and dtypes, not the values.

Host tables, OUT/host/<name>.bin (raw little-endian float32, C order; OUT/host/dit_manifest.json lists shape, bytes
and sha256):
  y_pos                  the adapter's text position embedding [1, 300, 2560]
  cos_j, sin_j           rotary tables of the joint [text; image] sequence [1, 2604, 1, 64]
  cos_i, sin_i           rotary tables of the image tokens, for the pixel stage [1, 2304, 1, 64]
  pix_pos                the pixel stage's sincos position table, patch-major [1, 2304, 256, 16]
  sigmas                 the four Turbo sigmas of distill.yaml
  t_emb_s<i>, mods_s<i>  per sampler step i: the timestep embedding [1, 1, 2560] and every block's adaLN rows
                         [1, 192, 2560] (shared core + block bias, 1 + scale on rows 1 and 4; dual blocks image, then text)
  noise_pix              the starting noise of seed 0 (torch.randn(1, 3, 768, 768) from torch.Generator().manual_seed(0)),
                         patch-major [1, 2304, 256, 3]
Everything that depends on t only is computed here because the Turbo sigmas are fixed. Torch runs on 12 threads, the
setting the published tables were made with.

Writes OUT/fp32/dit/dit_<name>.tflite and OUT/fp32/dit/graphs.json; a graph whose file is already there is skipped.
--iris defaults to OUT/weights/iris-3b/turbo (config.yaml, distill.yaml, model.safetensors), --upstream to OUT/upstream.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

SIZE = 768
DUAL = (3, 3, 2)
SINGLE = (6, 6, 4)
SEED = 0
THREADS = 12
NAMES = ["dit_adapter", "dit_dual0", "dit_dual1", "dit_dual2", "dit_single0", "dit_single1", "dit_single2",
         "dit_pixel"]
TRACE_SHAPES = {
    "dit_adapter": [(1, 1, 3600, 2560), (1, 20, 300, 300), (1, 300, 2560)],
    "dit_dual0": [(1, 1, 2304, 768), (1, 300, 2560), (1, 2604, 1, 64), (1, 2604, 1, 64), (1, 36, 2560)],
    "dit_dual1": [(1, 2304, 2560), (1, 300, 2560), (1, 2604, 1, 64), (1, 2604, 1, 64), (1, 36, 2560)],
    "dit_dual2": [(1, 2304, 2560), (1, 300, 2560), (1, 2604, 1, 64), (1, 2604, 1, 64), (1, 24, 2560)],
    "dit_single0": [(1, 2604, 2560), (1, 2604, 1, 64), (1, 2604, 1, 64), (1, 36, 2560)],
    "dit_single1": [(1, 2604, 2560), (1, 2604, 1, 64), (1, 2604, 1, 64), (1, 36, 2560)],
    "dit_single2": [(1, 2604, 2560), (1, 2604, 1, 64), (1, 2604, 1, 64), (1, 24, 2560)],
    "dit_pixel": [(1, 2304, 2560), (1, 1, 2560), (1, 2304, 256, 3), (1, 2304, 256, 16), (1, 2304, 1, 64),
                  (1, 2304, 1, 64)],
}


def log(t0, msg):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - t0:6.0f}s] {msg}", flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def load_upstream(ckpt):
    """(IrisDiT in float32, config, sigmas) from an exported checkpoint directory."""
    import torch
    from omegaconf import OmegaConf

    from iris3b.config import inference_config
    from iris3b.models.dit import IrisDiT
    from iris3b.sampling import load_for_inference

    raw, weights = load_for_inference(ckpt)
    cfg = inference_config(raw)
    sigmas = list(OmegaConf.load(Path(ckpt) / "distill.yaml").dmd.sigmas)
    with torch.device("meta"):
        model = IrisDiT(cfg.model)
    model.load_state_dict(weights, strict=True, assign=True)
    return model.eval().to(dtype=torch.float32), cfg, sigmas


def write_bin(folder, name, array, files):
    a = np.ascontiguousarray(np.asarray(array).astype("<f4"))
    path = folder / f"{name}.bin"
    tmp = folder / f"{name}.bin.part"
    a.tofile(tmp)
    os.replace(tmp, path)
    files[name] = {"file": path.name, "shape": list(a.shape), "dtype": "float32", "bytes": int(a.nbytes),
                   "sha256": sha256(path)}


def host_tables(model, cfg, sigmas, folder):
    import torch

    import dit_modules as dm

    p = cfg.model.patch_size
    files = {}
    write_bin(folder, "y_pos", model.y_pos_embedding.detach().float().numpy(), files)
    h = dm.host_tables(model, SIZE, SIZE)
    for k in ("cos_j", "sin_j", "cos_i", "sin_i", "pix_pos"):
        write_bin(folder, k, h[k].numpy(), files)
    write_bin(folder, "sigmas", np.asarray(sigmas, np.float32), files)
    for i, s in enumerate(sigmas):
        sigma = torch.full((1, 1, 1, 1), s, dtype=torch.float32)
        t_emb, mods = dm.step_tensors(model, sigma.flatten() * 1000)
        write_bin(folder, f"t_emb_s{i}", t_emb.numpy(), files)
        write_bin(folder, f"mods_s{i}", mods.numpy(), files)
    g = torch.Generator(device="cpu").manual_seed(SEED)
    noise = torch.randn(1, cfg.model.in_channels, SIZE, SIZE, generator=g, device="cpu")
    write_bin(folder, "noise_pix", dm.patchify(noise.float(), p).numpy(), files)
    # the trace shapes below must be the shapes these tables feed
    rows = dm.mod_rows(DUAL, SINGLE)
    assert files["cos_j"]["shape"] == list(TRACE_SHAPES["dit_dual0"][2]), files["cos_j"]
    assert files["cos_i"]["shape"] == list(TRACE_SHAPES["dit_pixel"][4]), files["cos_i"]
    assert files["pix_pos"]["shape"] == list(TRACE_SHAPES["dit_pixel"][3]), files["pix_pos"]
    assert files["y_pos"]["shape"] == list(TRACE_SHAPES["dit_adapter"][2]), files["y_pos"]
    for name, (lo, hi) in rows.items():
        assert TRACE_SHAPES[name][-1][1] == hi - lo, (name, lo, hi)
    manifest = {"format": "raw little-endian float32, C order", "image_hw": [SIZE, SIZE], "patch": p,
                "threads": THREADS, "seed": SEED, "mod_rows": {k: list(v) for k, v in rows.items()},
                "files": files}
    (folder / "dit_manifest.json").write_text(json.dumps(manifest, indent=1))
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out", help="output root (default: out)")
    ap.add_argument("--iris", default=None, help="iris-3b Turbo checkpoint directory (default: OUT/weights/iris-3b/turbo)")
    ap.add_argument("--upstream", default=None, help="speridlabs/iris-3b checkout (default: OUT/upstream)")
    args = ap.parse_args()
    root = Path(args.out)
    ckpt = Path(args.iris) if args.iris else root / "weights" / "iris-3b" / "turbo"
    upstream = Path(args.upstream) if args.upstream else root / "upstream"
    sys.path.insert(0, str(upstream / "src"))
    t0 = time.time()

    import torch

    torch.set_num_threads(THREADS)
    import litert_torch

    import dit_modules as dm

    model, cfg, sigmas = load_upstream(ckpt)
    log(t0, f"upstream DiT loaded ({sum(p_.numel() for p_ in model.parameters()) / 1e9:.3f} B parameters), "
            f"sigmas {sigmas}")
    host = root / "host"
    host.mkdir(parents=True, exist_ok=True)
    files = host_tables(model, cfg, sigmas, host)
    log(t0, f"host tables: {len(files)} files in {host}")

    out_dir = root / "fp32" / "dit"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "graphs.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    graphs = dm.build_graphs(model, DUAL, SINGLE)
    assert list(graphs) == NAMES, list(graphs)
    for name in NAMES:
        path = out_dir / f"{name}.tflite"
        if path.exists() and path.stat().st_size > 0:
            log(t0, f"SKIP {path.name}")
            continue
        params = sum(p_.numel() for p_ in graphs[name].parameters())
        sample = tuple(torch.zeros(s, dtype=torch.float32) for s in TRACE_SHAPES[name])
        tc = time.time()
        with torch.no_grad():
            edge = litert_torch.convert(graphs[name], sample)
            tmp = out_dir / f"{name}.part.tflite"
            edge.export(str(tmp))
        os.replace(tmp, path)
        meta[name] = {"bytes": path.stat().st_size, "sha256": sha256(path), "params": params,
                      "seconds": round(time.time() - tc, 1), "litert_torch": litert_torch.__version__,
                      "torch": torch.__version__}
        meta_path.write_text(json.dumps(meta, indent=1))
        print("WROTE", json.dumps({"file": path.name, **meta[name]}), flush=True)
        del edge
    log(t0, "DIT_GRAPHS_DONE")


if __name__ == "__main__":
    main()
