"""Post-hoc quantization of the exported fp32 graphs with ai-edge-quantizer.

  python quantize.py --mode fp16|drq8|drq8fc [--only text,sound,decoder] [--src out/fp32] [--out out/<mode>]
  fp16   = FLOAT_CASTING of every weight (fp16 storage, fp32 compute on CPU; GPU delegates read fp16 weights natively)
  drq8   = dynamic-range int8 (channelwise weights, fp32 activations) on FULLY_CONNECTED + CONV_2D + DEPTHWISE_CONV_2D
  drq8fc = the same on FULLY_CONNECTED only (leaves the convolutions in fp32)
The output keeps the graph's name with the mode as suffix (ema_text_L64_fp16.tflite), so verify_tflite.py --suffix reads it.
"""
import argparse, glob, os, time
from pathlib import Path
import ai_edge_quantizer.recipe as aqr
from ai_edge_quantizer import quantizer

HERE = Path(__file__).resolve().parent
FP16 = [{"regex": ".*", "operation": "*", "algorithm_key": aqr.AlgorithmName.FLOAT_CASTING,
         "op_config": {"weight_tensor_config": {"num_bits": 16, "symmetric": True, "granularity": "TENSORWISE", "dtype": "FLOAT"},
                       "compute_precision": "FLOAT", "explicit_dequantize": True, "skip_checks": False, "min_weight_elements": 0}}]


def recipe(mode):
    if mode == "fp16":
        return FP16
    ops = {"drq8": (aqr.TFLOperationName.FULLY_CONNECTED, aqr.TFLOperationName.CONV_2D, aqr.TFLOperationName.DEPTHWISE_CONV_2D),
           "drq8fc": (aqr.TFLOperationName.FULLY_CONNECTED,)}[mode]
    rules = []
    for op in ops:
        rules.extend(aqr.dynamic_wi8c_afp32(operation_name=op))
    return rules


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--only", default="text,sound,decoder")
    ap.add_argument("--src", default="out/fp32")
    ap.add_argument("--out", default="")
    ap.add_argument("--match", default="", help="only file names containing this (e.g. multi)")
    a = ap.parse_args()
    out = HERE / (a.out or f"out/{a.mode}")
    out.mkdir(parents=True, exist_ok=True)
    kinds = a.only.split(",")
    for src in sorted(glob.glob(str(HERE / a.src / "*_fp32.tflite"))):
        name = Path(src).name
        if not any(f"ema_{k}_" in name for k in kinds) or (a.match and a.match not in name):
            continue
        dst = out / name.replace("_fp32.tflite", f"_{a.mode}.tflite")
        t0 = time.time()
        qt = quantizer.Quantizer(src)
        qt.load_quantization_recipe(recipe(a.mode))
        qt.quantize().export_model(str(dst), overwrite=True)
        print(f"{a.mode:7s} {name:36s} {os.path.getsize(src)/1e6:6.2f} MB -> {os.path.getsize(dst)/1e6:6.2f} MB  {time.time()-t0:4.1f} s", flush=True)


if __name__ == "__main__":
    main()
