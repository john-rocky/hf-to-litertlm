"""Zero-DCE (low-light enhancement, tiny pure-CNN curve estimator) -> LiteRT.

Why this model: litert-samples has NO low-light / enhancement task, and Zero-DCE's
DCE-Net is a 7-conv CNN (conv+ReLU+tanh+concat + elementwise curve application) ->
no attention, no GPU-hostile ops. Goal: a clean, no-patch, GPU-friendly sample.

The graph outputs the FINAL enhanced image (the 8 curve iterations are baked in as
elementwise ops, GPU-clean). `x*x` is used instead of `torch.pow(x,2)` so it lowers
to MUL, not POW. Channel-last NHWC I/O matches the litert-samples RGB convention.

    ~/clipconv/bin/python scripts/convert_zerodce.py [out_dir] [size]
"""
import _stub  # noqa: F401  (scipy/_propack + getsourcefile guards; MUST be first)
import sys, os, json, traceback

import torch
import torch.nn as nn
from probe_convert_module import read_op_histogram, run_parity

BANNED = {"GATHER_ND", "SELECT_V2", "BROADCAST_TO", "WHERE", "POW"}

WEIGHTS = os.path.join(os.path.dirname(__file__), "..", "zerodce", "Epoch99.pth")


class ZeroDCE(nn.Module):
    """DCE-Net (enhance_net_nopool), returning only the final enhanced image."""

    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        f = 32
        self.e_conv1 = nn.Conv2d(3, f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(f, f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(f, f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(f, f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(f * 2, f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(f * 2, f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(f * 2, 24, 3, 1, 1, bias=True)

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        for r in torch.split(x_r, 3, dim=1):  # 8 curve iterations (x^2 == x*x -> MUL)
            x = x + r * (x * x - x)
        return x


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "out/zerodce"
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    os.makedirs(out_dir, exist_ok=True)
    res = {"model": "Zero-DCE", "size": size, "ok": False, "stage": None,
           "error_type": None, "error_head": None, "failing_op": None}
    try:
        res["stage"] = "load-weights"
        model = ZeroDCE().eval()
        sd = torch.load(WEIGHTS, map_location="cpu")
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
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
        tfl = os.path.join(out_dir, f"zerodce_{size}.tflite")
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
        with open("reports/probes/zerodce.trace.txt", "w") as f:
            f.write(tb)

    os.makedirs("reports/probes", exist_ok=True)
    with open("reports/probes/zerodce.json", "w") as f:
        json.dump(res, f, indent=2)
    print("ZERODCE_RESULT " + json.dumps(res))


if __name__ == "__main__":
    main()
