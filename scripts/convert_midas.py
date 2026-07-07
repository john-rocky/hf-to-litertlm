"""MiDaS_small (monocular depth, EfficientNet-Lite3 CNN backbone) -> LiteRT.

Why this model: litert-samples has NO depth task, and MiDaS_small is the CNN
MiDaS (NOT the DPT/ViT variants) -> it avoids the C12/C15/C19 ViT walls that
shelved DA3/MoGe. Goal: a clean, no-patch, GPU-friendly official-sample candidate.

Pipeline (same as probe_convert_module.py): hub-load -> eager fwd -> convert ->
op histogram (flag GPU-hostile ops) -> CPU parity (tflite vs eager).

    ~/clipconv/bin/python scripts/convert_midas.py [out_dir] [size]
"""
import _stub  # noqa: F401  (scipy/_propack + getsourcefile guards; MUST be first)
import sys, os, json, traceback

import torch
from probe_convert_module import read_op_histogram, run_parity

# ops the ML Drift GPU delegate / converter thesis flags as hostile (desktop-visible subset)
BANNED = {"GATHER_ND", "SELECT_V2", "BROADCAST_TO", "WHERE"}


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "out/midas-small"
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    os.makedirs(out_dir, exist_ok=True)
    res = {"model": "MiDaS_small", "size": size, "ok": False, "stage": None,
           "error_type": None, "error_head": None, "failing_op": None}
    try:
        res["stage"] = "hub-load"
        # MiDaS_small nests a 2nd hub repo (the efficientnet-lite3 backbone) whose
        # trust prompt can't be answered headless -> pre-trust both repos.
        hub_dir = torch.hub.get_dir()
        os.makedirs(hub_dir, exist_ok=True)
        tl = os.path.join(hub_dir, "trusted_list")
        trusted = {l.strip() for l in open(tl)} if os.path.exists(tl) else set()
        for r in ("intel-isl_MiDaS", "rwightman_gen-efficientnet-pytorch"):
            if r not in trusted:
                with open(tl, "a") as f:
                    f.write(r + "\n")
        import litert_torch
        # The efficientnet-lite3 backbone tries to download ImageNet-pretrained
        # weights, but MiDaS overwrites the full state_dict right after -> no-op that
        # download (unnecessary, and makes convert offline-safe once MiDaS .pt cached).
        try:
            geff = os.path.join(hub_dir, "rwightman_gen-efficientnet-pytorch_master")
            if geff not in sys.path:
                sys.path.insert(0, geff)
            import geffnet.helpers as _gh
            _gh.load_pretrained = lambda *a, **k: None
        except Exception:
            pass
        # Cached repos + offline-resilient: skip the github ref validation (it does a
        # DNS/HTTP call that can fail), and fall back to the local cache on any URLError.
        try:
            model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small",
                                   trust_repo=True, skip_validation=True).eval()
        except Exception:
            local = os.path.join(hub_dir, "intel-isl_MiDaS_master")
            model = torch.hub.load(local, "MiDaS_small", source="local",
                                   trust_repo=True).eval()

        res["stage"] = "eager-forward"
        nchw = torch.randn(1, 3, size, size)
        with torch.no_grad():
            out = model(nchw)                       # native NCHW signature -> reference
        res["eager_out_shape"] = list(out.shape)
        print("EAGER OK", res["eager_out_shape"])

        res["stage"] = "convert"
        # Channel-last I/O: exported model takes NHWC (1,H,W,3) -> GPU-friendlier
        # and matches litert-samples' interleaved-RGB input convention.
        clio = litert_torch.to_channel_last_io(model, args=[0])
        nhwc = nchw.permute(0, 2, 3, 1).contiguous()
        sample = nhwc
        edge = litert_torch.convert(clio, (nhwc,))
        tfl = os.path.join(out_dir, f"midas_small_{size}.tflite")
        edge.export(tfl)
        res["tflite"] = tfl
        res["tflite_mb"] = round(os.path.getsize(tfl) / 1e6, 2)
        print("CONVERT OK", tfl, res["tflite_mb"], "MB")

        res["stage"] = "read-ops"
        ops = read_op_histogram(tfl)
        res["ops"] = ops
        banned_hit = sorted(k for k in ops.get("hist", {}) if k.upper() in BANNED)
        res["banned_ops"] = banned_hit
        res["flex"] = ops.get("flex")
        res["custom"] = ops.get("custom")

        res["stage"] = "parity"
        res["parity"] = run_parity(tfl, (sample,), out)

        res["ok"] = True
        res["stage"] = "done"
    except BaseException as e:  # noqa: BLE001
        tb = traceback.format_exc()
        res["error_type"] = type(e).__name__
        msg = str(e).strip()
        res["error_head"] = (msg.splitlines()[0] if msg else tb.strip().splitlines()[-1])[:500]
        for ln in tb.splitlines():
            s = ln.strip()
            if ("aten." in s) or ("Lowering not found" in s) or ("While executing" in s):
                res["failing_op"] = s[:500]
                break
        os.makedirs("reports/probes", exist_ok=True)
        with open("reports/probes/midas-small.trace.txt", "w") as f:
            f.write(tb)

    os.makedirs("reports/probes", exist_ok=True)
    with open("reports/probes/midas-small.json", "w") as f:
        json.dump(res, f, indent=2)
    print("MIDAS_RESULT " + json.dumps(res))


if __name__ == "__main__":
    main()
