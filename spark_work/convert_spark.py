#!/usr/bin/env python3
"""Spark-X2.5 (XHToken, model_type spark2_5, remote code, THINKING) -> .litertlm.

Wraps scripts/export_simple_template.py with the facts this checkpoint needs:

1. The export reads the PATCHED export dir made by spark_work/patch_modeling.py
   (attention-interface dispatch so litert-torch's lrt_transposed_attention and
   KV cache are used). Pass that dir, not the vendor dir.
2. JINJA path (USE_JINJA=1): the vendor prompt format always opens with a default
   system block, which the runtime's structured prefix/suffix templates cannot
   express, so the bundle carries templates/spark25_think.jinja verbatim. The
   literal `<think>` in it makes litert-torch auto-declare the `thought` channel; the generation prompt ends with the
   vendor's own `<think>` opener.
3. NO_START_TOKEN=1: tokenizer_config says add_bos_token=false and the template
   emits its own <｜start▁of▁sentence｜> per message; the builder would still
   write start_token from tokenizer.bos_token and the runtime would prepend a
   second one.
4. HF tokenizer.json path (byte-level BPE with three Split regexes + Digits +
   ByteLevel); no SP conversion.
5. int4 recipes MUST run with EXTERNALIZE_EMBEDDER=1: the vocab is tied, and int4 on
   the lm_head FULLY_CONNECTED + int8 on EMBEDDING_LOOKUP make the quantizer copy the
   131072-row table once per signature (measured on the tiny checkpoint: 4.58 vs 1.68
   bytes/param, scripts/check_bundle_sanity.py). int8 needs no split.

Usage (same positional contract as the driver):
    ~/venvs/ltconv040dev/bin/python spark_work/convert_spark.py \
        src_models/Spark-X2.5-1.7B-export out/spark-1.7b-int8 templates/spark25_think.jinja dynamic_wi8_afp32
    EXTERNALIZE_EMBEDDER=1 ~/venvs/ltconv040dev/bin/python spark_work/convert_spark.py \
        src_models/Spark-X2.5-1.7B-export out/spark-1.7b-int4 templates/spark25_think.jinja BOCTAV4
    env: CACHE (default 4096), PREFILL (default ladder 1024..1), EXTERNALIZE_EMBEDDER (int4: required)
"""
import os
import runpy
import sys

os.environ["NO_START_TOKEN"] = "1"
os.environ["USE_JINJA"] = "1"
os.environ.setdefault("CACHE", "4096")
os.environ.setdefault("PREFILL", "1024,512,256,128,64,32,16,8,4,2,1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
driver = os.path.join(ROOT, "scripts", "export_simple_template.py")
sys.argv = [driver] + sys.argv[1:]
runpy.run_path(driver, run_name="__main__")
