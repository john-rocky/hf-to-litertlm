#!/usr/bin/env python3
"""check_pinned.py MODEL_DIR MANIFEST.json - check a local copy of ibm-granite/granite-3.0-1b-a400m-instruct against the
Hub's record for the pinned commit, file by file, before anything is exported from it. Nothing is downloaded.

  python granitemoe_work/check_pinned.py /path/to/granite-3.0-1b-a400m-instruct out/checkpoint_manifest.json

Every file the commit lists (HF API ?blobs=true) must be in MODEL_DIR with the Hub's size and hash: LFS files by
sha256, the others by git blob sha1 (sha1(b"blob <len>\\0" + data)). Files in MODEL_DIR that the commit does not list
are ignored. Writes MANIFEST.json (repo, revision, per-file hashes, verdict).
Exit 0 = every file matches, 1 = a mismatch or a missing file, 2 = usage.
"""
import hashlib
import json
import os
import sys
import urllib.request

REPO = "ibm-granite/granite-3.0-1b-a400m-instruct"
PINNED = "ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445"


def sha256_file(p):
  h = hashlib.sha256()
  with open(p, "rb") as f:
    for c in iter(lambda: f.read(4 << 20), b""):
      h.update(c)
  return h.hexdigest()


def git_blob_sha1(p):
  data = open(p, "rb").read()
  return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def main():
  if len(sys.argv) != 3:
    print(__doc__, file=sys.stderr)
    return 2
  model_dir, manifest = os.path.abspath(sys.argv[1]), sys.argv[2]
  if not os.path.isdir(model_dir):
    print(f"no directory {model_dir}", file=sys.stderr)
    return 2
  url = f"https://huggingface.co/api/models/{REPO}/revision/{PINNED}?blobs=true"
  api = json.load(urllib.request.urlopen(url, timeout=60))
  if api.get("sha") != PINNED:
    print(f"the Hub answered revision {api.get('sha')}, not {PINNED}", file=sys.stderr)
    return 1
  rows, ok = [], True
  for s in api["siblings"]:
    p = os.path.join(model_dir, s["rfilename"])
    row = {"file": s["rfilename"], "hub_size": s.get("size"), "hub_blob_sha1": s.get("blobId"),
           "hub_lfs_sha256": (s.get("lfs") or {}).get("sha256")}
    if not os.path.isfile(p):
      row["PASS"] = False
      row["error"] = "missing"
    else:
      row["size"] = os.path.getsize(p)
      if row["hub_lfs_sha256"]:
        row["sha256"] = sha256_file(p)
        row["PASS"] = row["sha256"] == row["hub_lfs_sha256"] and row["size"] == row["hub_size"]
      else:
        row["git_blob_sha1"] = git_blob_sha1(p)
        row["PASS"] = row["git_blob_sha1"] == row["hub_blob_sha1"] and row["size"] == row["hub_size"]
    ok = ok and row["PASS"]
    rows.append(row)
    print(f"{'OK ' if row['PASS'] else 'BAD'} {row['file']} {row.get('size')} B {row.get('error', '')}".rstrip())
  doc = {"repo": REPO, "revision": PINNED, "model_dir": model_dir, "files": rows, "verdict": "PASS" if ok else "FAIL"}
  with open(manifest, "w") as f:
    json.dump(doc, f, indent=1)
  print(doc["verdict"], model_dir)
  return 0 if ok else 1


if __name__ == "__main__":
  raise SystemExit(main())
