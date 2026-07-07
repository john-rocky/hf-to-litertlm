"""FLOAT16 weight quantization of the Zero-DCE .tflite (AI Edge Quantizer,
FLOAT_CASTING) — GPU-native fp16, matches the MiDaS/U2Net sample convention.

    ~/clipconv/bin/python scripts/quantize_zerodce_fp16.py <fp32.tflite> [fp16_out.tflite]
"""
import sys, os
import numpy as np
from ai_edge_quantizer import quantizer, recipe_manager
from ai_edge_quantizer.recipe import AlgorithmName, qtyping
from ai_edge_litert.interpreter import Interpreter


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


def run(tfl, hwc):
    interp = Interpreter(model_path=tfl)
    interp.allocate_tensors()
    di, do = interp.get_input_details()[0], interp.get_output_details()[0]
    interp.set_tensor(di["index"], hwc[None].astype(di["dtype"]))
    interp.invoke()
    return interp.get_tensor(do["index"]).astype("float64").reshape(-1)


def main():
    fp32 = sys.argv[1] if len(sys.argv) > 1 else "out/zerodce/zerodce_512.tflite"
    out = sys.argv[2] if len(sys.argv) > 2 else "out/zerodce/zerodce_512_fp16.tflite"

    if os.path.exists(out):
        os.remove(out)
    qt = quantizer.Quantizer(float_model=fp32)
    qt.load_quantization_recipe(fp16_recipe())
    qt.quantize().export_model(out)

    s32, s16 = os.path.getsize(fp32) / 1e6, os.path.getsize(out) / 1e6
    print(f"SIZE  fp32 {s32:.2f} MB -> fp16 {s16:.2f} MB  ({s16/s32*100:.0f}%)")

    interp = Interpreter(model_path=out)
    interp.allocate_tensors()
    size = int(interp.get_input_details()[0]["shape"][1])
    hist = {}
    for d in interp._get_ops_details():
        hist[d["op_name"]] = hist.get(d["op_name"], 0) + 1
    print("FP16 ops:", dict(sorted(hist.items(), key=lambda x: -x[1])))

    # fidelity: fp16 vs fp32 on a dark random [0,1] image (low-light proxy)
    rng = np.random.default_rng(0)
    hwc = (rng.random((size, size, 3), dtype=np.float32) * 0.3)  # dim input
    a, b = run(fp32, hwc), run(out, hwc)
    corr = float(np.corrcoef(a, b)[0, 1])
    maxdiff = float(np.max(np.abs(a - b)))
    print(f"FIDELITY fp16 vs fp32: corr {corr:.8f}  max|diff| {maxdiff:.5f}")


if __name__ == "__main__":
    main()
