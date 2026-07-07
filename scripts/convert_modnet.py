"""MODNet (trimap-free portrait matting, pure-CNN) -> LiteRT.

Why this model: litert-samples has NO portrait-matting task. MODNet predicts a soft
alpha matte (hair-level detail) from a single RGB image with no trimap. The whole net
is GPU-clean: MobileNetV2 backbone (depthwise-separable InvertedResidual), Conv+IBNorm
+ReLU, an SE block (global avg-pool -> FC -> sigmoid -> scale), bilinear F.interpolate
upsampling, sigmoid head. No PixelShuffle, no attention/softmax, no grid_sample.

The graph bakes `inference=True` (the semantic/detail aux heads are dropped) and the
[0,1]->[-1,1] input normalization, so the app feeds a plain [0,1] RGB image and gets a
single-channel [0,1] alpha matte. Channel-last NHWC I/O (in AND out) matches the
litert-samples convention.

    ~/clipconv/bin/python scripts/convert_modnet.py [out_dir] [size]
"""
import _stub  # noqa: F401  (scipy/_propack + getsourcefile guards; MUST be first)
import sys, os, json, traceback

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "third_party", "MODNet"))
from src.models.modnet import MODNet, SEBlock  # noqa: E402
from probe_convert_module import read_op_histogram, run_parity  # noqa: E402


def _se_forward(self, x):
    """SE channel scale, converter-friendly. Two equivalent rewrites (same math,
    same weights, no litert_torch fork):
      1. `AdaptiveAvgPool2d(1).view(b,c)` -> `x.mean(dim=(2,3))`. The adaptive pool
         emits a 5D intermediate squeezed by GATHER_ND (a banned GPU op) under the
         NHWC relayout; a plain spatial-mean reduce lowers cleanly (like the
         InstanceNorm reductions already in the graph).
      2. `x * w.expand_as(x)` -> broadcast-mul of the [b,c,1,1] weight. The explicit
         aten.expand can't be relaid out to NHWC; the broadcast-mul is the canonical
         channelwise-scale pattern and lowers to a clean MUL.
    """
    b, c, _, _ = x.size()
    w = x.mean(dim=(2, 3))           # global avg pool -> [b, c]
    w = self.fc(w)                   # [b, c]
    w = w[:, :, None, None]          # [b, c, 1, 1]
    return x * w


SEBlock.forward = _se_forward

BANNED = {"GATHER_ND", "SELECT_V2", "BROADCAST_TO", "WHERE", "POW"}

from huggingface_hub import hf_hub_download  # noqa: E402
WEIGHTS = hf_hub_download(
    "DavG25/modnet-pretrained-models",
    "models/modnet_photographic_portrait_matting.ckpt",
)


class MODNetMatte(nn.Module):
    """Wrap MODNet: [0,1] RGB in -> single-channel [0,1] alpha matte out."""

    def __init__(self, modnet):
        super().__init__()
        self.modnet = modnet

    def forward(self, x):
        x = x * 2.0 - 1.0  # [0,1] -> [-1,1] (the official Normalize(0.5,0.5))
        _, _, matte = self.modnet(x, True)  # inference=True -> matte only
        return matte


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "out/modnet"
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    os.makedirs(out_dir, exist_ok=True)
    res = {"model": "MODNet", "size": size, "ok": False, "stage": None,
           "error_type": None, "error_head": None, "failing_op": None}
    try:
        res["stage"] = "load-weights"
        core = MODNet(backbone_pretrained=False)
        sd = torch.load(WEIGHTS, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        core.load_state_dict(sd, strict=True)
        model = MODNetMatte(core).eval()
        print("WEIGHTS OK")

        res["stage"] = "eager-forward"
        nchw = torch.rand(1, 3, size, size)  # [0,1] like the app input
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
        tfl = os.path.join(out_dir, f"modnet_{size}.tflite")
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
        res["error_head"] = (msg.splitlines()[0] if msg else tb.strip().splitlines()[-1])[:500]
        for ln in tb.splitlines():
            s = ln.strip()
            if ("aten." in s) or ("Lowering not found" in s) or ("While executing" in s):
                res["failing_op"] = s[:500]
                break
        os.makedirs("reports/probes", exist_ok=True)
        with open("reports/probes/modnet.trace.txt", "w") as f:
            f.write(tb)

    os.makedirs("reports/probes", exist_ok=True)
    with open("reports/probes/modnet.json", "w") as f:
        json.dump(res, f, indent=2)
    print("MODNET_RESULT " + json.dumps(res))


if __name__ == "__main__":
    main()
