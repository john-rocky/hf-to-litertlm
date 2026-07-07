"""Post-training FLOAT16 weight quantization of the MiDaS_small .tflite via
AI Edge Quantizer (FLOAT_CASTING). Halves size (~66->~33 MB), GPU-native fp16,
negligible quality loss — the right recipe for a GPU depth sample (int8
dynamic-range favors CPU/XNNPACK, not the ML Drift GPU path).

    ~/clipconv/bin/python scripts/quantize_midas_fp16.py <fp32.tflite> [fp16_out.tflite]
"""
import sys, os
import numpy as np
from ai_edge_quantizer import quantizer, recipe_manager
from ai_edge_quantizer.recipe import AlgorithmName, qtyping
from ai_edge_litert.interpreter import Interpreter

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def fp16_recipe():
    rm = recipe_manager.RecipeManager()
    op_config = qtyping.OpQuantizationConfig(
        weight_tensor_config=qtyping.TensorQuantizationConfig(
            num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
        compute_precision=qtyping.ComputePrecision.FLOAT,
    )
    rm.add_quantization_config(
        regex=".*",
        operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        op_config=op_config,
        algorithm_key=AlgorithmName.FLOAT_CASTING,
    )
    return rm.get_quantization_recipe()


def run(tfl, norm_hwc):
    interp = Interpreter(model_path=tfl)
    interp.allocate_tensors()
    di, do = interp.get_input_details()[0], interp.get_output_details()[0]
    shape = list(di["shape"])
    x = norm_hwc[None] if shape[-1] == 3 else np.transpose(norm_hwc, (2, 0, 1))[None]
    interp.set_tensor(di["index"], x.astype(di["dtype"]))
    interp.invoke()
    return interp.get_tensor(do["index"]).astype("float64").reshape(-1)


def main():
    fp32 = sys.argv[1] if len(sys.argv) > 1 else "out/midas-small/midas_small_256.tflite"
    out = sys.argv[2] if len(sys.argv) > 2 else "out/midas-small/midas_small_256_fp16.tflite"

    if os.path.exists(out):
        os.remove(out)
    qt = quantizer.Quantizer(float_model=fp32)
    qt.load_quantization_recipe(fp16_recipe())
    result = qt.quantize()
    result.export_model(out)

    s32, s16 = os.path.getsize(fp32) / 1e6, os.path.getsize(out) / 1e6
    print(f"SIZE  fp32 {s32:.1f} MB -> fp16 {s16:.1f} MB  ({s16/s32*100:.0f}%)")

    # op histogram of the fp16 model
    interp = Interpreter(model_path=out)
    interp.allocate_tensors()
    hist = {}
    for d in interp._get_ops_details():
        hist[d["op_name"]] = hist.get(d["op_name"], 0) + 1
    print("FP16 ops:", dict(sorted(hist.items(), key=lambda x: -x[1])))

    # fidelity: fp16 vs fp32 on a real image (dog)
    import urllib.request
    from PIL import Image
    tmp = "/tmp/midas_input.jpg"
    if not os.path.exists(tmp):
        urllib.request.urlretrieve("https://github.com/pytorch/hub/raw/master/images/dog.jpg", tmp)
    img = Image.open(tmp).convert("RGB").resize((256, 256), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    norm_hwc = ((arr - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float32)
    a, b = run(fp32, norm_hwc), run(out, norm_hwc)
    corr = float(np.corrcoef(a, b)[0, 1])
    maxdiff = float(np.max(np.abs(a - b)))
    rng = a.max() - a.min()
    print(f"FIDELITY fp16 vs fp32 (real image): corr {corr:.8f}  max|diff| {maxdiff:.4f}  "
          f"(rel {maxdiff/rng*100:.3f}% of depth range)")


if __name__ == "__main__":
    main()
