"""Assemble the litert-community upload directory from the gated build outputs (copy + rename + sha256 + manifest).
  ~/venvs/lt094dev/bin/python3 audio8_tts_work/assemble_ship.py [dest]   (default out/ship)
"""
import os, sys, json, hashlib, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

dest = sys.argv[1] if len(sys.argv) > 1 else os.path.join(C.OUT, "ship")
os.makedirs(os.path.join(dest, "voices", "en_librispeech_1272"), exist_ok=True)
os.makedirs(os.path.join(dest, "voices", "ja_funasr_example"), exist_ok=True)
FILES = {
    "slow_ar_int8.tflite": "slow/slow_drq8_c2048_p256.tflite",
    "slow_ar_int4.tflite": "slow/slow_bo4_c2048_p256.tflite",
    "fast_ar_int8.tflite": "fast/fast_drq8_v3.tflite",
    "codec_decoder_fp16_T128.tflite": "codec/codec_decoder_fp16_T128_g2.tflite",
    "codec_decoder_fp16_T192.tflite": "codec/codec_decoder_fp16_T192_g2.tflite",
    "codec_decoder_int8_T128.tflite": "codec/codec_decoder_i8nativefix_T128_g2.tflite",
    "codec_encoder_fp16_10s.tflite": "codec/codec_encoder_fp16_10s_g2.tflite",
}
manifest = {"source": {"repo": C.HF_REPO, "revision": C.HF_REV}, "files": {}}
for name, rel in FILES.items():
    src = os.path.join(C.OUT, rel)
    if not os.path.exists(src):
        print("MISSING", src); continue
    dst = os.path.join(dest, name); shutil.copyfile(src, dst)
    h = hashlib.sha256(open(dst, "rb").read()).hexdigest()
    manifest["files"][name] = {"bytes": os.path.getsize(dst), "sha256": h, "built_from": rel}
    print(f"{name:34s} {os.path.getsize(dst)/1e6:8.1f} MB {h[:12]}")
shutil.copyfile(os.path.join(C.SNAP, "tokenizer.json"), os.path.join(dest, "tokenizer.json"))
shutil.copyfile(os.path.join(C.WORK, "audio8_tts_litert.py"), os.path.join(dest, "audio8_tts_litert.py"))
import numpy as np
for key, vdir in (("en", "en_librispeech_1272"), ("ja", "ja_funasr_example")):
    codes = np.load(os.path.join(C.FIX, f"ref_{key}_codes.npy"))
    np.save(os.path.join(dest, "voices", vdir, "codes.npy"), codes.astype(np.uint16))
    json.dump({"reference_text": C.REFS[key]["text"], "shape": list(codes.shape), "sample_rate": C.SR,
               "source": ("LibriSpeech dev-clean 1272-128104-0013 (CC BY 4.0), 16 kHz -> 44.1 kHz resample_poly" if key == "en"
                          else "FunAudioLLM/Fun-ASR-Nano-2512 example/ja.mp3 (Apache-2.0), 16 kHz -> 44.1 kHz resample_poly"),
               "encoder": "PyTorch fp32 codec encoder of the source checkpoint (codes identical to codec_encoder_fp16_10s.tflite on 99.4-99.8% of entries)"},
              open(os.path.join(dest, "voices", vdir, "meta.json"), "w"), ensure_ascii=False, indent=2)
json.dump(manifest, open(os.path.join(dest, "build_manifest.json"), "w"), indent=1)
print("->", dest)
