#!/usr/bin/env python3
"""Mac timing of the audio encoder alone (house protocol for a .tflite on Mac: ai_edge_litert Interpreter, 8 threads,
median of 20 warm invokes, the first invoke reported separately). The file timed is the fp16 encoder tflite, and it is
first checked to be byte-identical to the bundle's tf_lite_audio_encoder_hw section (offsets read from the bundle
header with litert_lm_builder, not typed in). Input = one real 30.24 s window (en_clip00, zero-padded, runtime framing).

  ~/venvs/lt094dev/bin/python bench_encoder.py      # -> bench/mac_encoder.json, bench/mac_encoder.log (tee)
"""
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.join(HERE, "out", "bundle", "Fun-ASR-Nano-2512.litertlm")
ENC = os.path.join(HERE, "out", "audio_encoder", "audio_encoder_504f_fp16.tflite")
CLIP = os.path.join(HERE, "out", "fixtures", "en_clip00.wav")
THREADS, WARM = 8, 20


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def audio_section(path):
    from litert_lm_builder import litertlm_core
    from litert_lm_builder import litertlm_header_schema_py_generated as schema
    with open(path, "rb") as f:
        head = f.read(65536)
    end = int.from_bytes(head[litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET:litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET + 8], "little")
    meta = schema.LiteRTLMMetaData.GetRootAs(bytearray(head[litertlm_core.HEADER_BEGIN_BYTE_OFFSET:end]), 0)
    for i in range(meta.SectionMetadata().ObjectsLength()):
        so = meta.SectionMetadata().Objects(i)
        for j in range(so.ItemsLength()):
            it = so.Items(j)
            if it.Key() and it.Key().decode() == "model_type" and it.ValueType() == schema.VData.StringValue:
                sv = schema.StringValue()
                sv.Init(it.Value().Bytes, it.Value().Pos)
                if sv.Value().decode() == "tf_lite_audio_encoder_hw":
                    return so.BeginOffset(), so.EndOffset()
    raise SystemExit("no tf_lite_audio_encoder_hw section")


def window(path):
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    buf = np.zeros(504 * 960, np.float32)
    buf[:len(x)] = x.astype(np.float32) / 32768.0
    return buf.reshape(1, 504, 960), len(x) / 16000.0


def main():
    from importlib.metadata import version
    from ai_edge_litert.interpreter import Interpreter
    load = subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()
    b0, b1 = audio_section(BUNDLE)
    with open(BUNDLE, "rb") as f:
        f.seek(b0)
        sec = f.read(b1 - b0)
    enc = open(ENC, "rb").read()
    same = sec == enc
    print(f"bundle audio section [{b0}, {b1}) {len(sec):,} B sha256 {sha256_bytes(sec)} | fp16 tflite {len(enc):,} B "
          f"sha256 {sha256_bytes(enc)} | byte-identical {same}", flush=True)
    assert same, "the fp16 tflite is not the bundle's encoder section"
    x, secs = window(CLIP)
    it = Interpreter(model_path=ENC, num_threads=THREADS)
    run = it.get_signature_runner("encode")
    t0 = time.perf_counter()
    out = run(audio=x)
    first = (time.perf_counter() - t0) * 1e3
    valid = int(np.nonzero(out["mask"][0])[0].max()) + 1
    warm = []
    for _ in range(WARM):
        t0 = time.perf_counter()
        run(audio=x)
        warm.append((time.perf_counter() - t0) * 1e3)
    doc = {"file": os.path.relpath(ENC, HERE), "bytes": len(enc), "sha256": sha256_bytes(enc),
           "bundle_section": [b0, b1], "bundle_section_identical": same, "clip": "en_clip00", "clip_s": round(secs, 3),
           "window_s": 30.24, "valid_tokens": valid, "threads": THREADS, "ai_edge_litert": version("ai-edge-litert"),
           "first_invoke_ms": round(first, 1), "warm_runs": WARM, "warm_ms": [round(v, 1) for v in warm],
           "warm_median_ms": round(statistics.median(warm), 1), "warm_min_ms": round(min(warm), 1),
           "warm_max_ms": round(max(warm), 1), "uptime_before": load,
           "uptime_after": subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip(),
           "python": sys.version.split()[0]}
    json.dump(doc, open(os.path.join(HERE, "bench", "mac_encoder.json"), "w"), indent=1)
    print(json.dumps({k: doc[k] for k in ("first_invoke_ms", "warm_median_ms", "warm_min_ms", "warm_max_ms",
                                          "valid_tokens", "threads", "ai_edge_litert")}), flush=True)
    print("ENCODER_BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
