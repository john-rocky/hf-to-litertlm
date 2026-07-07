"""VDSR (Very Deep Super-Resolution, CVPR'16) -> LiteRT.

Why this model: litert-samples has NO super-resolution task. VDSR takes a bicubic-
upscaled LR (so input == output size, NO in-network upsampling) and refines it -> the
graph is just conv + ReLU + a global residual add. No PixelShuffle (C7), no PReLU (C6),
no attention -> structurally GPU-clean (unlike Real-ESRGAN, which needs C6/C7 patches).

Weights: twtygqyy/pytorch-vdsr model_epoch_50.pth (whole-model pickle; we register the
arch under a fake `vdsr` module so it unpickles). Y-channel (1ch) model.

    ~/clipconv/bin/python scripts/convert_vdsr.py [out_dir] [size]
"""
import _stub  # noqa: F401
import sys, os, json, types, traceback
from math import sqrt

import torch
import torch.nn as nn
from probe_convert_module import read_op_histogram, run_parity

BANNED = {"GATHER_ND", "SELECT_V2", "BROADCAST_TO", "WHERE", "POW", "PRELU_MISSING"}
WEIGHTS = os.path.join(os.path.dirname(__file__), "..", "vdsr", "v.pth")


class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.residual_layer = nn.Sequential(*[Conv_ReLU_Block() for _ in range(18)])
        self.input = nn.Conv2d(1, 64, 3, 1, 1, bias=False)
        self.output = nn.Conv2d(64, 1, 3, 1, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        return torch.add(out, residual)


# Register the classes under a fake `vdsr` module so the whole-model pickle unpickles.
_m = types.ModuleType("vdsr")
_m.Conv_ReLU_Block = Conv_ReLU_Block
_m.Net = Net
sys.modules["vdsr"] = _m


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "out/vdsr"
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    os.makedirs(out_dir, exist_ok=True)
    res = {"model": "VDSR", "size": size, "ok": False, "stage": None,
           "error_type": None, "error_head": None, "failing_op": None}
    try:
        res["stage"] = "load-weights"
        ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
        model = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model = getattr(model, "module", model).eval()  # unwrap DataParallel
        # rebuild a clean Net and copy weights (drops any DataParallel/training cruft)
        clean = Net().eval()
        clean.load_state_dict(model.state_dict())
        print("WEIGHTS OK")

        res["stage"] = "eager-forward"
        nchw = torch.rand(1, 1, size, size)  # 1-channel (Y), bicubic-upscaled LR
        with torch.no_grad():
            out = clean(nchw)
        res["eager_out_shape"] = list(out.shape)
        print("EAGER OK", res["eager_out_shape"])

        res["stage"] = "convert"
        import litert_torch
        clio = litert_torch.to_channel_last_io(clean, args=[0], outputs=[0])
        nhwc = nchw.permute(0, 2, 3, 1).contiguous()
        sample = nhwc
        edge = litert_torch.convert(clio, (nhwc,))
        tfl = os.path.join(out_dir, f"vdsr_{size}.tflite")
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

    print("VDSR_RESULT " + json.dumps(res))


if __name__ == "__main__":
    main()
