#!/usr/bin/env python3
"""Take the LM export's graphs and executor metadata out of litert-torch's work directory.

litert-torch 0.10.0 ends its export by packing its own .litertlm through `litert_lm_builder.TfLiteModelType`, which
litert-lm-builder 0.19.x (the version this bundle is packed with, build_bundle.py) no longer has. That last step stops
with `AttributeError: module 'litert_lm_builder' has no attribute 'TfLiteModelType'`; by then the embedder and the
prefill / decode graphs are exported and quantized, and executor_metadata.pb (the KV cache state buffers) is written,
all in the work directory (EXPORT_DIR/tmp*) that the export leaves behind when it stops. This script accepts that one
stop and nothing else: the log must show every export task done, the packaging failed, and that AttributeError as the
error. It moves

  tmp*/embedder_quantized.tflite -> OUT/embedder_int8.tflite
  tmp*/model_quantized.tflite    -> OUT/prefill_decode_int8.tflite
  tmp*/executor_metadata.pb      -> OUT/executor_metadata.pb

and deletes EXPORT_DIR (the float graphs left in it are about 6.9 GB).

  python collect_lm_export.py EXPORT_DIR EXPORT_LOG OUT
"""
import re
import shutil
import sys
from pathlib import Path

STOP = "AttributeError: module 'litert_lm_builder' has no attribute 'TfLiteModelType'"
DONE = ["[ DONE] LiteRT GenAI Export > Load source model",
        "[ DONE] LiteRT GenAI Export > Export embedder model",
        "[ DONE] LiteRT GenAI Export > Export text prefill-decode model",
        "[ DONE] LiteRT GenAI Export > Export tokenizer",
        "[ FAIL] LiteRT GenAI Export > Package model"]
MOVE = {"embedder_quantized.tflite": "embedder_int8.tflite",
        "model_quantized.tflite": "prefill_decode_int8.tflite",
        "executor_metadata.pb": "executor_metadata.pb"}


def main():
    if len(sys.argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2
    export_dir, log, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    text = log.read_text(errors="replace")
    # the progress lines can wrap at any width: compare without whitespace
    squeeze = lambda s: re.sub(r"\s+", "", s)
    flat = squeeze(text)
    missing = [d for d in DONE if squeeze(d) not in flat]
    errors = [ln for ln in text.splitlines() if re.match(r"^\w+(Error|Exception)\b", ln)]
    if missing or errors != [STOP]:
        print(f"the export did not stop the way this recipe expects:\n  missing progress lines: {missing}\n"
              f"  error lines: {errors}\n  expected exactly: {STOP}", file=sys.stderr)
        return 1
    work = [p for p in export_dir.iterdir() if p.is_dir() and p.name.startswith("tmp")]
    if len(work) != 1:
        print(f"expected one tmp* work dir in {export_dir}, found {work}", file=sys.stderr)
        return 1
    absent = [n for n in MOVE if not (work[0] / n).is_file()]
    if absent:
        print(f"{work[0]} lacks {absent}", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    for a, b in MOVE.items():
        shutil.move(str(work[0] / a), str(out / b))
        print(f"{a} -> {out / b} ({(out / b).stat().st_size:,} B)")
    shutil.rmtree(export_dir)
    print("COLLECT_LM_EXPORT_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
