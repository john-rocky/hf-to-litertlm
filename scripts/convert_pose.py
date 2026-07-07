"""Lightweight-OpenPose (MobileNet, heatmap pose) -> LiteRT.

Why this model: litert-samples has NO pose task, and MoveNet's official tflite bakes the
keypoint decode (GATHER_ND) into the graph -> only 106/297 nodes on the GPU delegate.
A heatmap model keeps the GRAPH pure-conv (GPU-clean) and the argmax/decode is done in
Kotlin -> full GPU residency. This wraps Daniil-Osokin/lightweight-human-pose to return
the final-stage heatmaps only ([1,19,H/8,W/8]); the app argmaxes each keypoint channel.

    ~/clipconv/bin/python scripts/convert_pose.py [out_dir] [size]
"""
import _stub  # noqa: F401
import sys, os, json, traceback

import torch
import torch.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pose"))
from models.with_mobilenet import PoseEstimationWithMobileNet
from probe_convert_module import read_op_histogram, run_parity

BANNED = {"GATHER_ND", "SELECT_V2", "BROADCAST_TO", "WHERE", "POW"}
CKPT = os.path.join(os.path.dirname(__file__), "..", "pose", "cp.pth")


class PoseHeatmaps(nn.Module):
    """Return only the final-stage heatmaps (drop PAFs and intermediate stages)."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)[-2]  # [heatmaps, pafs] per stage -> final heatmaps


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "out/pose"
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    os.makedirs(out_dir, exist_ok=True)
    res = {"model": "lightweight-openpose", "size": size, "ok": False, "stage": None,
           "error_type": None, "error_head": None, "failing_op": None}
    try:
        res["stage"] = "load-weights"
        net = PoseEstimationWithMobileNet(num_refinement_stages=1).eval()
        ck = torch.load(CKPT, map_location="cpu", weights_only=False)
        sd = ck["state_dict"] if "state_dict" in ck else ck
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        net.load_state_dict(sd, strict=False)  # checkpoint has extra refinement stages; use stage-1 (fast config)
        model = PoseHeatmaps(net).eval()
        print("WEIGHTS OK")

        res["stage"] = "eager-forward"
        nchw = torch.rand(1, 3, size, size)
        with torch.no_grad():
            out = model(nchw)
        res["eager_out_shape"] = list(out.shape)
        print("EAGER OK", res["eager_out_shape"])

        res["stage"] = "convert"
        import litert_torch
        clio = litert_torch.to_channel_last_io(model, args=[0], outputs=[0])
        nhwc = nchw.permute(0, 2, 3, 1).contiguous()
        sample = nhwc
        edge = litert_torch.convert(clio, (nhwc,))
        tfl = os.path.join(out_dir, f"pose_{size}.tflite")
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
        res["error_head"] = (msg.splitlines()[0] if msg else tb.strip().splitlines()[-1])[:500]
        for ln in tb.splitlines():
            s = ln.strip()
            if ("aten." in s) or ("Lowering not found" in s) or ("While executing" in s):
                res["failing_op"] = s[:500]
                break

    print("POSE_RESULT " + json.dumps(res))


if __name__ == "__main__":
    main()
