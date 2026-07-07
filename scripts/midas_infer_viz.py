"""Run the converted MiDaS_small .tflite on a real image and save a side-by-side
input|depth visualization. Confirms the converted graph produces real depth
(beyond the random-input parity), and yields a demo artifact for the sample PR.

    ~/clipconv/bin/python scripts/midas_infer_viz.py <tflite> [image_path_or_url] [out.png]
"""
import sys, os, urllib.request
import numpy as np
from PIL import Image
import matplotlib.cm as cm
from ai_edge_litert.interpreter import Interpreter

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_image(src, size=256):
    if src.startswith("http"):
        tmp = "/tmp/midas_input.jpg"
        urllib.request.urlretrieve(src, tmp)
        src = tmp
    img = Image.open(src).convert("RGB")
    disp = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(disp, dtype=np.float32) / 255.0
    norm_hwc = (arr - IMAGENET_MEAN) / IMAGENET_STD      # ImageNet norm (MiDaS), HWC
    return np.asarray(disp), norm_hwc.astype(np.float32)


def to_input(norm_hwc, input_detail):
    """Build the model input array in the converted model's layout (NHWC or NCHW)."""
    shape = list(input_detail["shape"])
    x = norm_hwc[None] if shape[-1] == 3 else np.transpose(norm_hwc, (2, 0, 1))[None]
    return x.astype(input_detail["dtype"])


def main():
    tfl = sys.argv[1] if len(sys.argv) > 1 else "out/midas-small/midas_small_256.tflite"
    src = sys.argv[2] if len(sys.argv) > 2 else "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
    out = sys.argv[3] if len(sys.argv) > 3 else "out/midas-small/midas_depth_demo.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    disp, norm_hwc = load_image(src)

    interp = Interpreter(model_path=tfl)
    interp.allocate_tensors()
    di = interp.get_input_details()[0]
    do = interp.get_output_details()[0]
    interp.set_tensor(di["index"], to_input(norm_hwc, di))
    interp.invoke()
    depth = interp.get_tensor(do["index"]).squeeze()      # (256,256) inverse depth

    d = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_rgb = (cm.get_cmap("inferno")(d)[:, :, :3] * 255).astype(np.uint8)

    pad = np.full((disp.shape[0], 8, 3), 255, dtype=np.uint8)
    combo = np.concatenate([disp.astype(np.uint8), pad, depth_rgb], axis=1)
    Image.fromarray(combo).save(out)
    print(f"VIZ OK  depth range [{depth.min():.3f},{depth.max():.3f}]  -> {out}")


if __name__ == "__main__":
    main()
