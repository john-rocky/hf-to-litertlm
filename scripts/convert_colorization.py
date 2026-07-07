"""Colorful Image Colorization (Zhang et al. ECCV16, richzhang/colorization, BSD-2) -> LiteRT.

A VGG-style dilated-conv net: L (grayscale) -> ab (color). The annealed-mean decode
(softmax over 313 ab bins + 1x1 conv to ab cluster centers) is IN the graph, then a bilinear
upsample. All conv / softmax / resize / mul -> GPU-clean, no SE / shuffle / gather / PReLU.
Input L [1,1,256,256] (range 0..100), output ab [1,2,256,256] (range ~-110..110).

    ~/clipconv/bin/python scripts/convert_colorization.py [out_dir] [size]
"""
import _stub  # noqa: F401
import os, sys, json, traceback

import torch
import types as _types
_ip = _types.ModuleType("IPython")  # eccv16.py has a debug `from IPython import embed`, unused at runtime
_ip.embed = lambda *a, **k: None
sys.modules["IPython"] = _ip
# colorizers/util.py imports skimage for LAB helpers we don't use (the model is L->ab only).
_sk = _types.ModuleType("skimage")
_sk.__path__ = []
sys.modules["skimage"] = _sk
sys.modules["skimage.color"] = _types.ModuleType("skimage.color")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "colorization_work", "repo"))
from colorizers import eccv16
from probe_convert_module import read_op_histogram, run_parity

BANNED = {"GATHER_ND", "SELECT_V2", "BROADCAST_TO", "WHERE", "POW"}


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "out/colorization"
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    os.makedirs(out_dir, exist_ok=True)
    res = {"model": "eccv16-colorization", "size": size, "ok": False, "stage": None,
           "error_type": None, "error_head": None, "failing_op": None}
    try:
        res["stage"] = "load-weights"
        model = eccv16(pretrained=True).eval()
        print("WEIGHTS OK")

        res["stage"] = "eager-forward"
        nchw = torch.rand(1, 1, size, size) * 100.0  # L in 0..100
        with torch.no_grad():
            out = model(nchw)
        res["eager_out_shape"] = list(out.shape)
        print("EAGER OK", res["eager_out_shape"])

        res["stage"] = "convert"
        import litert_torch
        clio = litert_torch.to_channel_last_io(model, args=[0], outputs=[0])
        nhwc = nchw.permute(0, 2, 3, 1).contiguous()  # [1,H,W,1]
        edge = litert_torch.convert(clio, (nhwc,))
        tfl = os.path.join(out_dir, f"colorization_{size}.tflite")
        edge.export(tfl)
        res["tflite"] = tfl
        res["tflite_mb"] = round(os.path.getsize(tfl) / 1e6, 2)
        print("CONVERT OK", tfl, res["tflite_mb"], "MB")

        res["stage"] = "read-ops"
        ops = read_op_histogram(tfl)
        res["ops"] = ops
        res["banned_ops"] = sorted(k for k in ops.get("hist", {}) if k.upper() in BANNED)
        res["flex"] = ops.get("flex")
        res["custom"] = ops.get("custom")

        res["stage"] = "parity"
        out_nhwc = out.permute(0, 2, 3, 1).contiguous()  # match exported layout
        res["parity"] = run_parity(tfl, (nhwc,), out_nhwc)
        res["ok"] = True
        res["stage"] = "done"
    except BaseException as e:  # noqa: BLE001
        tb = traceback.format_exc()
        res["error_type"] = type(e).__name__
        msg = str(e).strip()
        res["error_head"] = (msg.splitlines()[0] if msg else tb.strip().splitlines()[-1])[:400]
        for ln in tb.splitlines():
            s = ln.strip()
            if ("aten." in s) or ("Lowering not found" in s) or ("While executing" in s):
                res["failing_op"] = s[:400]
                break

    print("COLOR_RESULT " + json.dumps(res))


if __name__ == "__main__":
    main()
