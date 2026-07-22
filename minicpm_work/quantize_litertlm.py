#!/usr/bin/env python3
"""Quantize an exported (unquantized) MiniCPM5 .litertlm and repackage it.

Reproduces the packaging of litert-community/MiniCPM5-1B: export with
--quantization_recipe="" first, then quantize the tf_lite_prefill_decode tflite
with an ai-edge-quantizer recipe, then rebuild the .litertlm (metadata +
tokenizer + quantized tflite) with litert-lm-builder.

Modes:
  inspect <file.tflite|file.litertlm>   dump per-tensor quantization layout
  apply   <in.litertlm> <out.litertlm> [--recipe wi4b32_wi8|wi8]
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))


def extract_sections(litertlm_path, out_dir):
    """Dump litertlm sections; returns dict with pbtext/tokenizer/tflite paths."""
    from litert_lm_builder import peek_litertlm_file

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "peek.txt"), "w") as f:
        peek_litertlm_file(litertlm_path, out_dir, f)
    found = {}
    for name in os.listdir(out_dir):
        p = os.path.join(out_dir, name)
        if name.endswith(".pbtext"):
            found["metadata"] = p
        elif name.endswith(".zlib"):
            found["tokenizer_zlib"] = p
        elif name.endswith(".spiece"):
            found["tokenizer_sp"] = p
        elif name.endswith(".tflite"):
            found["tflite"] = p
        elif name == "model.toml":
            found["toml"] = p
    return found


def inspect_tflite(tflite_path):
    """Print per-tensor quant layout of FULLY_CONNECTED / EMBEDDING_LOOKUP weights."""
    from ai_edge_litert.interpreter import Interpreter

    ip = Interpreter(model_path=tflite_path, experimental_preserve_all_tensors=False)
    seen = {}
    for d in ip.get_tensor_details():
        name, dt = d["name"], str(d["dtype"])
        q = d.get("quantization_parameters") or {}
        scales = q.get("scales")
        nscales = 0 if scales is None else len(scales)
        if "int4" in dt.lower() or dt == "<class 'numpy.int8'>" or nscales:
            shape = list(d["shape"])
            qd = q.get("quantized_dimension")
            key = (dt, nscales and "per-axis(%d sc, dim %s)" % (nscales, qd) or "none")
            seen.setdefault(key, []).append((name, shape))
    for (dt, gran), tensors in sorted(seen.items(), key=lambda kv: -len(kv[1])):
        print(f"== dtype {dt}  {gran}  x{len(tensors)}")
        for name, shape in tensors[:6]:
            print("   ", name, shape)
        if len(tensors) > 6:
            print(f"    ... {len(tensors) - 6} more")


# Output-tensor name of the lm_head FULLY_CONNECTED in the decode subgraph
# (verified in our fp32 export; the vocab matrix is shared with the embedder,
# so this + the EMBEDDING_LOOKUP rule land the single deduped tensor at int8,
# matching the official file: 163 FCs int4-b32 + one vocab tensor int8-cw).
LM_HEAD_REGEX = "decode_logits_output"


def build_recipe(kind, algo="minmax"):
    """Recipe list matching the official minicpm_wi4b32_wi8_afp32 layout.

    algo: minmax | octav — applied to the int4 blockwise linears. OCTAV clips
    outliers (smaller scales) and empirically recovers most of the min-max
    int4 GSM8K gap; int8 tensors stay min-max.
    """
    from ai_edge_quantizer import recipe_manager
    from ai_edge_quantizer import qtyping
    from ai_edge_quantizer.algorithm_manager import AlgorithmName

    G = qtyping.QuantGranularity
    OP = qtyping.TFLOperationName
    alg4 = {"minmax": AlgorithmName.MIN_MAX_UNIFORM_QUANT,
            "octav": AlgorithmName.OCTAV}[algo]
    rm = recipe_manager.RecipeManager()
    if kind == "wi8":
        rm.add_dynamic_config(regex=".*", operation_name=OP.ALL_SUPPORTED, num_bits=8)
    elif kind == "wi8fc":
        # FC + embedding ONLY. For hybrid-conv models (LFM2/2.5 ShortConv):
        # post-hoc ALL_SUPPORTED int8 breaks the conv layers (no output);
        # leave convs float (or quantize them at export time instead).
        rm.add_dynamic_config(regex=".*", operation_name=OP.FULLY_CONNECTED, num_bits=8)
        rm.add_dynamic_config(
            regex=".*", operation_name=OP.EMBEDDING_LOOKUP, num_bits=8,
            granularity=G.CHANNELWISE,
        )
    elif kind == "wi4b32_wi8":
        # Official card: "mixed INT4-block32(linear)/INT8(embed and lmhead)".
        # int4 b32 for all FULLY_CONNECTED, then int8 overrides for embedder +
        # the lm_head/logits FC (regex refined from official-tflite inspection).
        rm.add_dynamic_config(
            regex=".*", operation_name=OP.FULLY_CONNECTED, num_bits=4,
            granularity=G.BLOCKWISE_32, algorithm_key=alg4,
        )
        rm.add_dynamic_config(
            regex=LM_HEAD_REGEX, operation_name=OP.FULLY_CONNECTED, num_bits=8,
            granularity=G.CHANNELWISE,
        )
        rm.add_dynamic_config(
            regex=".*", operation_name=OP.EMBEDDING_LOOKUP, num_bits=8,
            granularity=G.CHANNELWISE,
        )
    else:
        raise ValueError(kind)
    return rm.get_quantization_recipe()


def apply_quant(in_litertlm, out_litertlm, kind, algo="minmax"):
    from ai_edge_quantizer import quantizer

    tmp = tempfile.mkdtemp(prefix="mc5q_")
    parts = extract_sections(in_litertlm, tmp)
    print("sections:", {k: os.path.basename(v) for k, v in parts.items()})

    qt = quantizer.Quantizer(parts["tflite"], build_recipe(kind, algo))
    if qt.need_calibration:
        raise SystemExit("recipe unexpectedly needs calibration")
    res = qt.quantize()
    qpath = os.path.join(tmp, "model_quant.tflite")
    res.export_model(qpath)
    print("quantized tflite:", os.path.getsize(qpath) / 1e6, "MB")

    shutil.move(qpath, parts["tflite"])  # overwrite fp tflite in place
    if "tokenizer_zlib" in parts:
        # builder wants a plain tokenizer.json (it zlib-compresses itself)
        import zlib

        tokjson = os.path.join(tmp, "tokenizer.json")
        with open(parts["tokenizer_zlib"], "rb") as f:
            raw = f.read()
        # section payload = uint64 uncompressed-size prefix, then zlib stream
        data = zlib.decompress(raw[8:] if raw[:2] != b"\x78\x9c" else raw)
        with open(tokjson, "wb") as f:
            f.write(data)
        tok_args = ["hf_tokenizer", "--path", tokjson]
    else:
        tok_args = ["sp_tokenizer", "--path", parts["tokenizer_sp"]]

    out_abs = os.path.abspath(out_litertlm)
    cmd = [
        "litert-lm-builder",
        "llm_metadata", "--path", parts["metadata"],
        *tok_args,
        "tflite_model", "--path", parts["tflite"], "--model_type", "prefill_decode",
        "output", "--path", out_abs,
    ]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("wrote", out_abs, os.path.getsize(out_abs) / 1e6, "MB")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    pi = sub.add_parser("inspect")
    pi.add_argument("path")
    pa = sub.add_parser("apply")
    pa.add_argument("infile")
    pa.add_argument("outfile")
    pa.add_argument("--recipe", default="wi4b32_wi8", choices=["wi4b32_wi8", "wi8", "wi8fc"])
    pa.add_argument("--algo", default="minmax", choices=["minmax", "octav"])
    args = ap.parse_args()

    if args.mode == "inspect":
        path = args.path
        if path.endswith(".litertlm"):
            tmp = tempfile.mkdtemp(prefix="mc5i_")
            path = extract_sections(path, tmp)["tflite"]
        inspect_tflite(path)
    else:
        apply_quant(args.infile, args.outfile, args.recipe, args.algo)


if __name__ == "__main__":
    main()
