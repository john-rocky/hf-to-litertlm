#!/usr/bin/env python3
"""aot_compile_sm8850.py MODEL.tflite OUTDIR - ahead-of-time compiles one .tflite for the Hexagon NPU of the Qualcomm SM8850
(Galaxy S26) with ai-edge-litert's Qualcomm compiler plugin and prints the compilation report.
Runs inside the image of the Dockerfile next to this file. The output is
OUTDIR/<model name>_Qualcomm_SM8850_apply_plugin.tflite."""
import sys

from ai_edge_litert.aot.aot_compile import aot_compile
from ai_edge_litert.aot.vendors.qualcomm import target as qc

model, outdir = sys.argv[1], sys.argv[2]
res = aot_compile(model, output_dir=outdir, target=[qc.Target(soc_model=qc.SocModel.SM8850)], keep_going=True)
print(res.compilation_report())
