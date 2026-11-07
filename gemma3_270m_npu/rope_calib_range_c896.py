#!/usr/bin/env python3
"""rope_calib_range_c896.py BOUNDS_DIR OUT_DIR - widen the calibrated range of gemma-3-270m's RoPE position inputs to the
whole 896-token KV cache before stage 3 of litert-torch's npu_export (the gemma-3-270m form of
mnn_pocket_work/npu/rope_calib_range.py).

Stage 2 records each tensor's min / max over the calibration prompts, and stage 3 gives every activation one symmetric
int16 scale from that range. The aux section's rope graphs (prefill_rope_128 and decode_rope) quantize the token
position, the angle (position x inv_freq) and their concatenation to int16 too. The calibration of the first published
file (10 prompts x 32 decode steps, cache 896) reached position 383 in prefill_rope_128 and 359 in decode_rope, so every
later token got the rotation of that position. This step copies BOUNDS_DIR to OUT_DIR and, in the aux section's file
only, sets the max of every rope entry whose calibrated max is 383 or 359 to 1023.96875 = 32767 / 32: stage 3's scale
for them is then exactly 1/32, every integer position 0..1023 is exact, and the cache's last position 895 is inside the
range. The mins, every other entry (the mask and cache-update entries keep their max; those graphs hold positions in
int32 / bool) and every other file stay as they are; the tool checks that, and stops unless it finds exactly 15 such
entries (8 prefill, 7 decode). Prints one JSON line: the changed entries and the aux file's sha256 before / after."""
import hashlib
import json
import os
import shutil
import sys

AUX = "Section5_TFLiteModel_tf_lite_aux.tflite.json"
NEW_MAX = 32767.0 / 32.0
OLD = {383.0: "prefill", 359.0: "decode"}


def sha(path):
  with open(path, "rb") as f:
    return hashlib.sha256(f.read()).hexdigest()


def is_rope(key):
  return "RoPEEmbedder" in key or key.startswith(("prefill_rope_128_", "decode_rope_"))


def main(src, dst):
  if os.path.exists(dst):
    shutil.rmtree(dst)
  os.makedirs(dst)
  for name in sorted(os.listdir(src)):
    if name != AUX:
      shutil.copy2(os.path.join(src, name), os.path.join(dst, name))
  with open(os.path.join(src, AUX)) as f:
    before = json.load(f)
  with open(os.path.join(src, AUX)) as f:
    out = json.load(f)
  changed = []
  for key, v in out["model_qsvs"].items():
    mx = v.get("max")
    if is_rope(key) and isinstance(mx, list) and len(mx) == 1 and mx[0] in OLD:
      changed.append({"key": key, "phase": OLD[mx[0]], "old_max": mx[0], "new_max": NEW_MAX})
      v["max"] = [NEW_MAX]
  n_pre = sum(c["phase"] == "prefill" for c in changed)
  n_dec = len(changed) - n_pre
  if (n_pre, n_dec) != (8, 7):
    sys.exit(f"expected 8 prefill + 7 decode rope entries at max 383 / 359, found {n_pre} + {n_dec}")
  with open(os.path.join(dst, AUX), "w") as f:
    f.write(json.dumps(out))
  with open(os.path.join(dst, AUX)) as f:
    back = json.load(f)
  diff = [k for k in before["model_qsvs"] if before["model_qsvs"][k] != back["model_qsvs"][k]]
  assert sorted(diff) == sorted(c["key"] for c in changed), "the aux file differs beyond the rope entries"
  assert set(before["model_qsvs"]) == set(back["model_qsvs"]) and set(before) == set(back)
  assert before.get("metadata") == back.get("metadata")
  for c in changed:
    assert back["model_qsvs"][c["key"]].get("min") == before["model_qsvs"][c["key"]].get("min")
  for name in sorted(os.listdir(src)):
    if name != AUX:
      assert sha(os.path.join(src, name)) == sha(os.path.join(dst, name)), name
  print(json.dumps({"new_max": NEW_MAX, "int16_scale": NEW_MAX / 32767.0, "changed": changed,
                    "aux_sha256_before": sha(os.path.join(src, AUX)), "aux_sha256_after": sha(os.path.join(dst, AUX))}))


if __name__ == "__main__":
  if len(sys.argv) != 3:
    sys.exit(__doc__)
  main(sys.argv[1], sys.argv[2])
