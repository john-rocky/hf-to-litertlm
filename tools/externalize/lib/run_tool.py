#!/usr/bin/env python3
"""Run a LiteRT tool module from a tools directory under `python -I` (which keeps the current and the script's
directory off sys.path): the tools directory is put first on sys.path by hand.

  python -I run_tool.py <tools dir> <module> [args...]   e.g. ... litert.tools.externalize_tflite_flatbuffer --input_model ...
"""
import runpy
import sys

if len(sys.argv) < 3:
    sys.exit(__doc__)
sys.path.insert(0, sys.argv.pop(1))
module = sys.argv.pop(1)
sys.argv[0] = module
runpy.run_module(module, run_name="__main__", alter_sys=True)
