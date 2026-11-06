#!/usr/bin/env python3
"""rope_calib_range.py BOUNDS_DIR OUT_DIR - widen the calibrated range of the RoPE position inputs to the whole KV cache
before stage 3.

Stage 2 records each tensor's min / max over the calibration prompts, and stage 3 gives every activation one symmetric
int16 scale from that range. The aux section's rope graphs (prefill_rope_128 and decode_rope) quantize the token
position, and the angles computed from it, to int16 too. With the calibration's own range, a position past the largest
one the calibration reached (511 in prefill_rope_128, 527 in decode_rope for this set) saturates at it, so every later
token gets the rotation of that position. This step copies BOUNDS_DIR to OUT_DIR and, in the aux section's file only,
sets the max of the 11 rope entries whose calibrated max is 511 or 527 to 2047.9375 = 32767 / 16. Stage 3's scale for
them is then exactly 1/16, and every integer position 0..2047 of the 2,048-token cache is exact. The mins and every other
entry and file stay as they are; the tool checks that, and stops if it does not find exactly 11 such entries."""
import hashlib
import json
import os
import shutil
import sys

AUX = "Section5_TFLiteModel_tf_lite_aux.tflite.json"
NEW_MAX = 32767.0 / 16.0


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
    if is_rope(key) and isinstance(mx, list) and len(mx) == 1 and mx[0] in (511.0, 527.0):
      changed.append((key, mx[0]))
      v["max"] = [NEW_MAX]
  if len(changed) != 11:
    sys.exit(f"expected 11 rope entries with max 511 or 527, found {len(changed)}")
  with open(os.path.join(dst, AUX), "w") as f:
    f.write(json.dumps(out))
  with open(os.path.join(dst, AUX)) as f:
    back = json.load(f)
  diff = [k for k in before["model_qsvs"] if before["model_qsvs"][k] != back["model_qsvs"][k]]
  assert sorted(diff) == sorted(k for k, _ in changed), "the aux file differs beyond the rope entries"
  assert before["metadata"] == back["metadata"] and set(before) == set(back)
  for name in sorted(os.listdir(src)):
    if name != AUX:
      assert sha(os.path.join(src, name)) == sha(os.path.join(dst, name)), name
  print(f"{len(changed)} rope entries -> max {NEW_MAX} (int16 scale {NEW_MAX / 32767.0}); {AUX} sha256 {sha(os.path.join(dst, AUX))}")
  for key, old in changed:
    print(f"  {old} -> {NEW_MAX}  {key[-90:]}")


if __name__ == "__main__":
  main(sys.argv[1], sys.argv[2])
