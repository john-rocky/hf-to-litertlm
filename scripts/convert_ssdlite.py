"""torchvision SSDlite320-MobileNetV3-Large (detection) -> LiteRT.

Export the RAW detection head (per-anchor class logits + box regression); the anchor
decode + NMS is done in Kotlin, keeping the graph pure-conv = GPU-clean. MobileNetV3 has
no channel shuffle (unlike NanoDet's ShuffleNetV2 -> 5D transpose / C11).

    ~/clipconv/bin/python scripts/convert_ssdlite.py [out_dir] [size]
"""
import _stub  # noqa: F401
import os, sys, json, traceback

import torch
import torch.nn as nn
from probe_convert_module import read_op_histogram

BANNED = {"GATHER_ND", "SELECT_V2", "BROADCAST_TO", "WHERE", "POW"}


class SSDRaw(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        feats = list(self.m.backbone(x).values())
        out = self.m.head(feats)
        return out["cls_logits"], out["bbox_regression"]


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "out/ssdlite"
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 320
    os.makedirs(out_dir, exist_ok=True)
    res = {"model": "ssdlite320_mobilenet_v3", "size": size, "ok": False, "stage": None,
           "error_type": None, "error_head": None, "failing_op": None}
    try:
        res["stage"] = "build"
        from torchvision.models.detection import (
            ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights)
        m = ssdlite320_mobilenet_v3_large(
            weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT).eval()
        model = SSDRaw(m).eval()
        print("BUILD OK")

        res["stage"] = "eager-forward"
        nchw = torch.rand(1, 3, size, size)
        with torch.no_grad():
            cls, box = model(nchw)
        res["out_shapes"] = [list(cls.shape), list(box.shape)]
        print("EAGER OK", res["out_shapes"])

        res["stage"] = "convert"
        import litert_torch
        clio = litert_torch.to_channel_last_io(model, args=[0])
        nhwc = nchw.permute(0, 2, 3, 1).contiguous()
        edge = litert_torch.convert(clio, (nhwc,))
        tfl = os.path.join(out_dir, f"ssdlite_{size}.tflite")
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

    print("SSDLITE_RESULT " + json.dumps(res))


if __name__ == "__main__":
    main()
