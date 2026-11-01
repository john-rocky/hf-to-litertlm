#!/usr/bin/env python3
"""The detector of the pipeline arms: an open-vocabulary detector finds the objects each request names.

  HF_HUB_DISABLE_XET=1 python tools/demo_scene_check/detect.py fetch --cache DIR/hf_cache
  HF_HUB_OFFLINE=1 python tools/demo_scene_check/detect.py run --fixtures DIR/fixtures.json --cache DIR/hf_cache \
      --out DIR/c/detections.json [--device cpu] [--dry]

Detector = IDEA-Research/grounding-dino-base at a pinned revision (Apache-2.0), fp32. fetch downloads its eight files
into --cache (a cache of its own, not the shared ~/.cache/huggingface) and checks every file's sha256 and size against
the values below (exit 1 on a difference). run reads them with the same eight names: snapshot_download with
local_files_only and no allow_patterns asks for every file of the revision and raises IncompleteSnapshotError on a
cache that holds only these.
Input = the source picture (long side 1024) through the detector's own processor. One forward pass per phrase (the
text "<phrase>." lowercased, the model card's form), post_process_grounded_object_detection with threshold 0.35 and
text_threshold 0.25 (the card's example uses 0.4 / 0.3; the lower values keep one more weak box per phrase, in the
arm's favour); every box kept with its score, best first, as [left, top, right, bottom] in 0-1000 of the photo. The
phrases are the scene file's hand-written det_query of every reference and listed object (the arm gets the object
names for free: no parser between the request and the detector, in the arm's favour).
"""
import argparse
import hashlib
import json
import os
import sys
import time

REPO = "IDEA-Research/grounding-dino-base"
REV = "12bdfa3120f3e7ec7b434d90674b3396eccf88eb"
THRESH, TEXT_THRESH = 0.35, 0.25
FILES = {  # sha256, bytes (read from the Hub files of REV)
    "README.md": ("a0d03193076262a585dcb1edfe4b3b72fac678055008b688feb188cceb7f977d", 2580),
    "config.json": ("eda416dae6f49419ff831b1c190ec430a060b19aae688dbaf2425a075b650608", 1737),
    "model.safetensors": ("5548f844c928c4b6f411fa8cbcc2bfa8dbbba437cb1d513975519f93c2a9ed21", 933400872),
    "preprocessor_config.json": ("8454179ba95e2ad22947835aad7b45862a601fc0055ab88bf1ee70892d3aea60", 457),
    "special_tokens_map.json": ("b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3", 125),
    "tokenizer.json": ("d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66", 711396),
    "tokenizer_config.json": ("d40ab645b68211910b9170d22433d43186a6ec8ee6fd10ba170524b25bf4fb56", 1237),
    "vocab.txt": ("07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3", 231508),
}


def sha256(p):
  h = hashlib.sha256()
  with open(p, "rb") as f:
    for b in iter(lambda: f.read(1 << 24), b""):
      h.update(b)
  return h.hexdigest()


def fetch(a):
  from huggingface_hub import snapshot_download
  cache = os.path.abspath(a.cache)
  if os.path.join(".cache", "huggingface") in cache:
    raise SystemExit("--cache: use a directory of its own, not the shared Hugging Face cache")
  p = snapshot_download(REPO, revision=REV, cache_dir=cache, allow_patterns=list(FILES))
  rep = {"repo": REPO, "revision": REV, "path": p, "files": {}}
  bad = []
  for f, (want, n) in FILES.items():
    fp = os.path.join(p, f)
    got = sha256(fp) if os.path.exists(fp) else None
    size = os.path.getsize(fp) if got else None
    rep["files"][f] = {"sha256": got, "bytes": size, "same": got == want and size == n}
    if not rep["files"][f]["same"]:
      bad.append(f)
  rep["check"] = "OK" if not bad else f"MISMATCH {bad}"
  json.dump(rep, open(os.path.join(cache, "detector_manifest.json"), "w"), indent=1)
  print(json.dumps(rep, indent=1))
  print("DETECTOR_FETCH", rep["check"], flush=True)
  return 0 if not bad else 1


def phrases_by_photo(fx):
  out = {}
  for it in fx["items"]:
    s = out.setdefault(it["photo"], [])
    for r in it["refs"] + it.get("llm_refs", []):
      if r["det_query"] not in s:
        s.append(r["det_query"])
  return out


def run(a):
  fx = json.load(open(a.fixtures))
  fx_dir = os.path.dirname(os.path.abspath(a.fixtures))
  want = phrases_by_photo(fx)
  print(want, flush=True)
  if a.dry:
    return 0
  import torch
  import transformers
  from huggingface_hub import snapshot_download
  from PIL import Image
  from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
  path = snapshot_download(REPO, revision=REV, cache_dir=os.path.abspath(a.cache), local_files_only=True,
                           allow_patterns=list(FILES))
  t0 = time.time()
  proc = AutoProcessor.from_pretrained(path)
  model = AutoModelForZeroShotObjectDetection.from_pretrained(path, dtype=torch.float32).eval().to(a.device)
  res = {"detector": REPO, "revision": REV, "threshold": THRESH, "text_threshold": TEXT_THRESH,
         "transformers": transformers.__version__, "torch": torch.__version__, "device": a.device,
         "load_s": round(time.time() - t0, 1), "photos": {}}
  for pid, phrases in want.items():
    ph = fx["photos"][pid]
    img = Image.open(os.path.join(fx_dir, ph["src"])).convert("RGB")
    W, H = img.size
    res["photos"][pid] = {"src": ph["src"], "src_sha256": ph["src_sha256"], "size_wh": [W, H], "phrases": {}}
    for q in phrases:
      text = q.lower().strip().rstrip(".") + "."
      t1 = time.time()
      inputs = proc(images=img, text=text, return_tensors="pt").to(a.device)
      with torch.inference_mode():
        outputs = model(**inputs)
      r = proc.post_process_grounded_object_detection(outputs, inputs.input_ids, threshold=THRESH,
                                                      text_threshold=TEXT_THRESH, target_sizes=[(H, W)])[0]
      boxes = []
      for b, s, lab in zip(r["boxes"].tolist(), r["scores"].tolist(), r.get("text_labels", r.get("labels"))):
        l, t, rr, bb = b
        boxes.append({"box": [round(max(0, l) / W * 1000, 1), round(max(0, t) / H * 1000, 1),
                              round(min(W, rr) / W * 1000, 1), round(min(H, bb) / H * 1000, 1)],
                      "score": round(float(s), 4), "label": lab if isinstance(lab, str) else str(lab)})
      boxes.sort(key=lambda x: -x["score"])
      res["photos"][pid]["phrases"][q] = {"text": text, "boxes": boxes, "seconds": round(time.time() - t1, 2)}
      print(f"[{pid}] {text!r}: {len(boxes)} boxes {[(x['box'], x['score']) for x in boxes[:3]]}", flush=True)
  os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
  json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
  print("DETECT_DONE", a.out, flush=True)
  return 0


def main():
  ap = argparse.ArgumentParser()
  sub = ap.add_subparsers(dest="cmd", required=True)
  f = sub.add_parser("fetch")
  f.add_argument("--cache", required=True)
  r = sub.add_parser("run")
  r.add_argument("--fixtures", required=True)
  r.add_argument("--cache", required=True)
  r.add_argument("--out", required=True)
  r.add_argument("--device", default="cpu")
  r.add_argument("--dry", action="store_true")
  a = ap.parse_args()
  return fetch(a) if a.cmd == "fetch" else run(a)


if __name__ == "__main__":
  sys.exit(main())
