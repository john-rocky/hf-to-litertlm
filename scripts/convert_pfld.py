"""PFLD (Practical Facial Landmark Detector, 98 points) -> LiteRT.

A MobileNetV2-style CNN that regresses 98 facial landmarks directly via a Linear layer
(no SE / channel-shuffle / adaptive-pool / argmax / NMS) -> pure conv + ReLU + fixed
AvgPool2d + FC, all GPU-clean. Output [1,196] = 98 (x,y) normalized to the input.

    ~/clipconv/bin/python scripts/convert_pfld.py [out_dir] [size]
"""
import _stub  # noqa: F401
import os, sys, json, traceback

import torch
import torch.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pfld_work"))
from pfld import PFLDInference
from probe_convert_module import read_op_histogram, run_parity

BANNED = {"GATHER_ND", "SELECT_V2", "BROADCAST_TO", "WHERE", "POW"}
CKPT = os.path.join(os.path.dirname(__file__), "..", "pfld_work", "checkpoint.pth.tar")


class Landmarks(nn.Module):
    """Return only the landmarks (drop the auxiliary feature output)."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)[1]  # (features, landmarks) -> landmarks [1,196]


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "out/pfld"
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 112
    os.makedirs(out_dir, exist_ok=True)
    res = {"model": "PFLD", "size": size, "ok": False, "stage": None,
           "error_type": None, "error_head": None, "failing_op": None}
    try:
        res["stage"] = "load-weights"
        net = PFLDInference().eval()
        ck = torch.load(CKPT, map_location="cpu", weights_only=False)
        sd = ck["pfld_backbone"] if "pfld_backbone" in ck else ck
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        net.load_state_dict(sd, strict=True)
        model = Landmarks(net).eval()
        print("WEIGHTS OK")

        res["stage"] = "eager-forward"
        nchw = torch.rand(1, 3, size, size)
        with torch.no_grad():
            out = model(nchw)
        res["eager_out_shape"] = list(out.shape)
        print("EAGER OK", res["eager_out_shape"])

        res["stage"] = "convert"
        import litert_torch
        clio = litert_torch.to_channel_last_io(model, args=[0])
        nhwc = nchw.permute(0, 2, 3, 1).contiguous()
        sample = nhwc
        edge = litert_torch.convert(clio, (nhwc,))
        tfl = os.path.join(out_dir, f"pfld_{size}.tflite")
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
        res["parity"] = run_parity(tfl, (sample,), out)
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

    print("PFLD_RESULT " + json.dumps(res))


if __name__ == "__main__":
    main()
